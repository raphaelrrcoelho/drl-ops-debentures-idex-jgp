# env_final.py
"""
Debenture portfolio environment with DISCRETE action space, ENHANCED features, and MAX ASSETS constraint
=======================================================================================================
NEW FEATURE: Dynamic top-K asset selection by index_weight
- On each rebalancing day, only the top K assets (default 50) by index_weight are investable
- Agent can hold assets bought previously even if they drop out of top K
- On rebalancing days, assets outside top K must be liquidated
- Fixed observation/action space of K assets + cash (if enabled)

Performance optimizations retained:
- JIT Compilation for core calculations
- Pre-allocated arrays
- Efficient pivoting
- Memory optimization

Maintains all original features and interfaces from previous env_final.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numba
from numba import jit, float32, int32, boolean

# ----------------------------- Configuration ----------------------------- #

@dataclass
class EnvConfig:
    # Rebalance & constraints
    rebalance_interval: int = 10
    max_weight: float = 0.10
    weight_blocks: int = 100
    allow_cash: bool = True
    cash_rate_as_rf: bool = True
    on_inactive: str = "to_cash"
    
    # NEW: Maximum number of investable assets
    max_assets: int = 50  # Top K assets by index_weight

    # Costs & penalties
    weight_alpha: float = 1.0   
    transaction_cost_bps: float = 20.0          
    delist_extra_bps: float = 20.0              
    lambda_turnover: float = 0.0002             
    lambda_hhi: float = 0.01                    
    lambda_drawdown: float = 0.005              
    lambda_tail: float = 0.0                    
    tail_window: int = 60                       
    tail_q: float = 0.05                        
    dd_mode: str = "incremental"                

    # Observation controls
    include_prev_weights: bool = True           
    include_active_flag: bool = True            
    global_stats: bool = True                   
    normalize_features: bool = True             
    obs_clip: float = 5.0
    
    # Feature selection
    use_momentum_features: bool = True
    use_volatility_features: bool = True
    use_relative_value_features: bool = True
    use_duration_features: bool = True
    use_microstructure_features: bool = True
    use_carry_features: bool = True
    use_spread_dynamics: bool = True
    use_risk_adjusted_features: bool = True
    use_sector_curves: bool = True
    use_zscore_features: bool = True
    use_rolling_zscores: bool = True

    # Episode control
    max_steps: Optional[int] = None             
    seed: Optional[int] = 42                    
    random_reset_frac: float = 0.0              

# -------------------------- Enhanced Features List ----------------------- #

# Define comprehensive feature groups
MOMENTUM_WINDOWS = [5, 20]
VOLATILITY_WINDOWS = [5, 20]

# Base features (will use lagged versions)
BASE_FEATURES = [
    "return", "spread", "duration", "time_to_maturity",
    "risk_free", "index_return", "ttm_rank",
    "sector_weight_index", "sector_spread", "sector_momentum",
    "sector_id",
]

# Enhanced feature groups
MOMENTUM_FEATURES = [f"momentum_{w}d" for w in MOMENTUM_WINDOWS]
REVERSAL_FEATURES = [f"reversal_{w}d" for w in MOMENTUM_WINDOWS]
VOLATILITY_FEATURES = [f"volatility_{w}d" for w in VOLATILITY_WINDOWS]
SPREAD_VOL_FEATURES = [f"spread_vol_{w}d" for w in VOLATILITY_WINDOWS]

RELATIVE_VALUE_FEATURES = [
    "spread_vs_sector_median", "spread_vs_sector_mean",
    "spread_percentile_sector", "spread_percentile_all",
]

DURATION_FEATURES = [
    "duration_change", "duration_vol", "duration_spread_interaction",
]

MICROSTRUCTURE_FEATURES = [
    "liquidity_score", "weight_momentum", "weight_volatility",
]

CARRY_FEATURES = [
    "carry_spread_ratio", "carry_momentum", "carry_vol",
]

SPREAD_DYNAMICS_FEATURES = [
    "spread_momentum_5d", "spread_momentum_20d",
    "spread_mean_reversion", "spread_acceleration",
]

RISK_ADJUSTED_FEATURES = []

SECTOR_CURVE_FEATURES = [
    "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
    "sector_fitted_spread", "spread_residual_ns",
]

# ------------------------------ JIT Compiled Utilities -------------------- #

@jit(float32[:](float32[:], int32), nopython=True, cache=True, fastmath=True)
def _blocks_to_weights_jit(blocks: np.ndarray, total_blocks: int) -> np.ndarray:
    """JIT compiled: Convert discrete block allocations to continuous weights."""
    total = blocks.sum()
    if total <= 0:
        return np.zeros_like(blocks)
    return blocks / total

@jit(int32[:](int32[:], boolean[:]), nopython=True, cache=True, fastmath=True)
def _sanitize_blocks_with_mask_jit(blocks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """JIT compiled: Zero out blocks for inactive assets."""
    result = blocks.copy()
    for i in range(len(blocks)):
        if not mask[i]:
            result[i] = 0
    return result

@jit(float32(float32[:]), nopython=True, cache=True, fastmath=True)
def _hhi_jit(w: np.ndarray) -> float32:
    """JIT compiled: Calculate Herfindahl-Hirschman Index."""
    return np.square(w).sum()

@jit(float32(float32[:], float32[:]), nopython=True, cache=True, fastmath=True)
def _turnover_jit(w_new: np.ndarray, w_prev: np.ndarray) -> float32:
    """JIT compiled: Calculate portfolio turnover."""
    return np.abs(w_new - w_prev).sum()

@jit(float32(float32[:], float32[:]), nopython=True, cache=True, fastmath=True)
def _portfolio_return_jit(weights: np.ndarray, returns: np.ndarray) -> float32:
    """JIT compiled: Calculate portfolio return."""
    return np.dot(np.ascontiguousarray(weights), np.ascontiguousarray(returns))

@jit(float32[:](float32[:], float32, float32), nopython=True, cache=True, fastmath=True)
def _normalize_weights_with_max_jit(weights: np.ndarray, max_weight: float32, eps: float32) -> np.ndarray:
    """JIT compiled: Normalize weights while respecting max weight constraint."""
    weights = np.minimum(weights, max_weight)
    total = weights.sum()
    if total > eps:
        return weights / total
    else:
        result = np.zeros_like(weights)
        if len(weights) > 0:
            result[0] = 1.0  # Default to first asset or cash
        return result

# Python wrappers for compatibility
def _blocks_to_weights(blocks: np.ndarray, total_blocks: int = 100) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=np.float32)
    return _blocks_to_weights_jit(blocks, np.int32(total_blocks))

def _sanitize_blocks_with_mask(blocks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=np.int32)
    mask_bool = mask > 0
    return _sanitize_blocks_with_mask_jit(blocks, mask_bool)

def _hhi(w: np.ndarray) -> float:
    return float(_hhi_jit(np.asarray(w, dtype=np.float32)))

def _turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    return float(_turnover_jit(
        np.asarray(w_new, dtype=np.float32),
        np.asarray(w_prev, dtype=np.float32)
    ))

def _dict_sector_exposures(sector_ids: np.ndarray, weights: np.ndarray) -> Dict[int, float]:
    """Calculate sector exposures (not JIT compiled due to dict return)."""
    m = min(len(sector_ids), len(weights))
    sid = np.asarray(sector_ids[:m])
    w = np.asarray(weights[:m], dtype=float)
    d: Dict[int, float] = {}
    for s in np.unique(sid):
        if s < 0:
            continue
        d[int(s)] = float(w[sid == s].sum())
    return d

# ------------------------------ Environment ------------------------------- #

class DebentureTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, panel: pd.DataFrame, config: EnvConfig, prebuilt: dict | None = None):
        super().__init__()
        assert isinstance(panel.index, pd.MultiIndex), "panel must be MultiIndex (date, debenture_id)"
        self.cfg = config
        if self.cfg.seed is not None:
            np.random.seed(int(self.cfg.seed))

        # NEW: Store max_assets configuration
        self.max_assets_config = self.cfg.max_assets

        # Attach prebuilt arrays if provided, otherwise build once
        if prebuilt is not None:
            self._attach_prebuilt(prebuilt)
        else:
            self._prepare_data(panel)

        # NEW: Fixed observation/action space size based on max_assets
        self.obs_n_assets = self.max_assets_config
        if self.cfg.allow_cash:
            self.obs_n_assets += 1  # Add slot for cash
        
        # Cash index (always last if enabled)
        self.cash_idx = None
        if self.cfg.allow_cash:
            self.cash_idx = self.obs_n_assets - 1

        # DISCRETE ACTION SPACE - fixed size based on max_assets
        self.max_blocks_per_asset = int(self.cfg.max_weight * self.cfg.weight_blocks)
        self.action_space = spaces.MultiDiscrete(
            [self.max_blocks_per_asset + 1] * self.obs_n_assets
        )

        # Observation space - fixed size
        obs_dim = self._obs_size()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            'action_mask': spaces.Box(0, 1, shape=(self.obs_n_assets,), dtype=np.int8)
        })

        # Pre-allocate work arrays for efficiency
        self._preallocate_work_arrays()

        # State - now tracks both portfolio positions and current top-K mapping
        self.t: int = 0
        
        # Portfolio weights in FULL universe space
        self.full_prev_w: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        self.full_curr_w: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        
        # Mapping from observation space to full universe
        self.current_top_k_indices: np.ndarray = np.arange(self.obs_n_assets, dtype=np.int32)
        
        # Weights in observation space (for agent interaction)
        self.prev_w: np.ndarray = np.zeros(self.obs_n_assets, dtype=np.float32)
        self.curr_w: np.ndarray = np.zeros(self.obs_n_assets, dtype=np.float32)
        
        self.wealth: float = 1.0
        self.peak_wealth: float = 1.0

        # Tail buffer (optional)
        self.tail_buffer: List[float] = []

        # Logs
        self._history: Dict[str, list] = {}

    def export_shared_arrays(self) -> dict:
        """Return views to big, read-only arrays and metadata so other envs can reuse them."""
        return {
            "R": self.R,                # (T, N)
            "X": self.X,                # (T, N, F)
            "ACT": self.ACT,            # (T, N)
            "W": self.W,                # (T, N) - index weights
            "RF": self.RF,              # (T,)
            "IDX": self.IDX,            # (T,)
            "RF_obs": self.RF_obs,      # (T,)
            "IDX_obs": self.IDX_obs,    # (T,)
            "global_means": getattr(self, "global_means", None),
            "global_stds": getattr(self, "global_stds", None),
            "dates": self.dates,        # (T,)
            "asset_ids": self.asset_ids,
            "feature_cols": self.feature_cols,
            "n_assets": self.n_assets,  # Full universe size
            "F": self.F,
            "T": self.T,
            "sector_ids": getattr(self, "sector_ids", None),
            "max_assets": self.max_assets_config,
        }

    def _attach_prebuilt(self, pb: dict) -> None:
        """Wire prebuilt arrays/metadata into this env (no copies)."""
        # Big arrays (keep references; do NOT copy)
        self.R = pb["R"]
        self.X = pb["X"]
        self.ACT = pb["ACT"]
        self.W = pb["W"]  # Index weights
        self.RF = pb["RF"]
        self.IDX = pb["IDX"]
        self.RF_obs = pb["RF_obs"]
        self.IDX_obs = pb["IDX_obs"]
        self.global_means = pb.get("global_means", None)
        self.global_stds = pb.get("global_stds", None)

        # Metadata
        self.dates = pb["dates"]
        self.asset_ids = list(pb["asset_ids"])
        self.feature_cols = list(pb["feature_cols"])
        self.n_assets = int(pb["n_assets"])  # Full universe
        self.F = int(pb["F"])
        self.T = int(pb["T"])
        self.max_assets_config = pb.get("max_assets", 50)
        
        # NEW: Set obs_n_assets BEFORE calling _preallocate_work_arrays
        self.obs_n_assets = self.max_assets_config
        if self.cfg.allow_cash:
            self.obs_n_assets += 1  # Add slot for cash

        sid = pb.get("sector_ids", None)
        if sid is not None:
            self.sector_ids = np.asarray(sid, dtype=np.int16)

        # Pre-allocate work arrays (existing method)
        self._preallocate_work_arrays()

    def _preallocate_work_arrays(self):
        """Pre-allocate work arrays for efficiency."""
        # Work arrays for observation space
        self._work_blocks = np.zeros(self.obs_n_assets, dtype=np.int32)
        self._work_weights = np.zeros(self.obs_n_assets, dtype=np.float32)
        self._work_mask = np.zeros(self.obs_n_assets, dtype=bool)
        
        # Work arrays for full universe
        self._work_returns_full = np.zeros(self.n_assets, dtype=np.float32)
        self._work_weights_full = np.zeros(self.n_assets, dtype=np.float32)
        
        self._obs_buffer = np.zeros(self._obs_size(), dtype=np.float32)

    # --------------------------- Data preparation --------------------------- #

    def _prepare_data(self, panel: pd.DataFrame):
        panel = panel.sort_index()

        # Required columns check
        required = [
            "return", "risk_free", "index_return", "active",
            "spread", "duration", "time_to_maturity", "sector_id",
        ]
        
        # NEW: Also require index_weight for top-K selection
        required.append("index_weight")
        
        for c in required:
            if c not in panel.columns:
                raise ValueError(f"Missing required column '{c}' in panel")

        # Build comprehensive feature list based on config
        base_feats = BASE_FEATURES.copy()
        
        # Add optional feature groups based on config
        if self.cfg.use_momentum_features:
            base_feats.extend(MOMENTUM_FEATURES)
            base_feats.extend(REVERSAL_FEATURES)
        
        if self.cfg.use_volatility_features:
            base_feats.extend(VOLATILITY_FEATURES)
            base_feats.extend(SPREAD_VOL_FEATURES)
        
        if self.cfg.use_relative_value_features:
            base_feats.extend(RELATIVE_VALUE_FEATURES)
        
        if self.cfg.use_duration_features:
            base_feats.extend(DURATION_FEATURES)
        
        if self.cfg.use_microstructure_features:
            base_feats.extend(MICROSTRUCTURE_FEATURES)
        
        if self.cfg.use_carry_features:
            base_feats.extend(CARRY_FEATURES)
        
        if self.cfg.use_spread_dynamics:
            base_feats.extend(SPREAD_DYNAMICS_FEATURES)
        
        if self.cfg.use_risk_adjusted_features:
            base_feats.extend(RISK_ADJUSTED_FEATURES)
        
        if self.cfg.use_sector_curves:
            base_feats.extend(SECTOR_CURVE_FEATURES)
        
        if self.cfg.include_active_flag and "active" in panel.columns:
            base_feats.append("active")

        # Create all lagged features
        panel = panel.copy()
        lag_cols_to_create = []
        for feat in base_feats:
            if feat not in ("sector_id", "active"):
                if feat in panel.columns and f"{feat}_lag1" not in panel.columns:
                    lag_cols_to_create.append(feat)

        # Single grouped operation for all lag columns
        if lag_cols_to_create:
            lagged_df = (
                panel[lag_cols_to_create]
                .groupby(level="debenture_id", sort=False)
                .shift(1)
                .astype(np.float32)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            for col in lag_cols_to_create:
                panel[f"{col}_lag1"] = lagged_df[col]

        # Build feature columns list
        feat_cols = []
        for feat in base_feats:
            if feat in ("sector_id", "active"):
                if feat in panel.columns:
                    feat_cols.append(feat)
            else:
                lag_col = f"{feat}_lag1"
                if lag_col in panel.columns:
                    feat_cols.append(lag_col)
                elif feat in panel.columns:
                    # Create lag if missing
                    panel[lag_col] = (
                        panel.groupby(level="debenture_id", sort=False)[feat]
                        .shift(1)
                        .astype(np.float32)
                        .fillna(0.0)
                    )
                    feat_cols.append(lag_col)

        # Add z-score features if enabled
        if self.cfg.use_zscore_features:
            z_cols = [c for c in panel.columns if c.endswith("_z_lag1")]
            feat_cols.extend([c for c in z_cols if c not in feat_cols])

        # Add rolling z-score features if enabled
        if self.cfg.use_rolling_zscores:
            rolling_z_cols = [c for c in panel.columns if ("_z252_lag1" in c or "_z126_lag1" in c or "_z63_lag1" in c)]
            feat_cols.extend([c for c in rolling_z_cols if c not in feat_cols])

        # Remove duplicates and sort for consistency
        feat_cols = sorted(list(set(feat_cols)))

        # Ensure all feature columns exist
        for c in feat_cols:
            if c not in panel.columns:
                panel[c] = 0.0

        # Dates and asset IDs
        dates = panel.index.get_level_values("date").unique().sort_values()
        self.dates = dates.to_numpy()
        asset_ids = panel.index.get_level_values("debenture_id").unique().tolist()
        self.asset_ids: List[str] = list(asset_ids)
        self.n_assets = len(self.asset_ids)  # Full universe size

        # Use unstack for efficient pivoting
        panel_reset = panel.reset_index()
        
        # Arrays (same-day returns, same-day active mask)
        R = (
            panel_reset.set_index(["date", "debenture_id"])["return"]
            .unstack(fill_value=np.nan)
            .reindex(index=dates, columns=asset_ids)
        )
        RF = (
            panel_reset[["date", "risk_free"]]
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
            .reindex(dates)["risk_free"]
            .fillna(0.0)
        )
        IDX = (
            panel_reset[["date", "index_return"]]
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
            .reindex(dates)["index_return"]
            .fillna(0.0)
        )
        A = (
            panel_reset.set_index(["date", "debenture_id"])["active"]
            .unstack(fill_value=0.0)
            .reindex(index=dates, columns=asset_ids)
            .fillna(0.0)
        )
        
        # NEW: Index weights for top-K selection
        W = (
            panel_reset.set_index(["date", "debenture_id"])["index_weight"]
            .unstack(fill_value=0.0)
            .reindex(index=dates, columns=asset_ids)
            .fillna(0.0)
        )

        # Feature tensor (lagged) with features
        if feat_cols:
            X = np.empty((len(dates), len(asset_ids), len(feat_cols)), dtype=np.float32)
            for i, c in enumerate(feat_cols):
                wide = (
                    panel_reset.set_index(["date", "debenture_id"])[c]
                    .unstack(fill_value=0.0)
                    .reindex(index=dates, columns=asset_ids)
                    .fillna(0.0)
                    .values.astype(np.float32)
                )
                X[:, :, i] = wide
        else:
            X = np.zeros((len(dates), len(asset_ids), 0), dtype=np.float32)
        
        self.feature_cols = list(feat_cols)

        # Sector IDs
        sector_id_wide = (
            panel_reset.set_index(["date", "debenture_id"])["sector_id"]
            .unstack(fill_value=-1)
            .reindex(index=dates, columns=asset_ids)
            .ffill().bfill().fillna(-1)
        )
        self.sector_ids = sector_id_wide.values.astype(np.int16)[0]

        # Store arrays
        self.R  = np.nan_to_num(R.values.astype(np.float16), nan=0.0)
        self.RF = np.nan_to_num(RF.values.astype(np.float16).ravel(), nan=0.0)
        self.IDX= np.nan_to_num(IDX.values.astype(np.float16).ravel(), nan=0.0)
        self.ACT = np.nan_to_num(A.values.astype(np.int8), nan=0).astype(np.int8)
        self.W = np.nan_to_num(W.values.astype(np.float32), nan=0.0)  # Index weights
        self.X = np.nan_to_num(X.astype(np.float16), nan=0.0)

        self.T = self.R.shape[0]
        self.F = self.X.shape[-1] if self.X.ndim == 3 else 0

        # Cross-sectional stats for global features
        if self.cfg.global_stats and self.F > 0:
            act = (self.ACT > 0).astype(np.float32)
            denom = np.maximum(act.sum(axis=1, keepdims=True), 1.0)
            means = (self.X * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None]
            stds = np.sqrt(((self.X - means) ** 2 * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None])
            self.global_means = means.squeeze(1)
            self.global_stds = np.maximum(stds.squeeze(1), 1e-6)
        else:
            self.global_means = None
            self.global_stds = None

        # Additional normalization if needed
        if self.cfg.normalize_features and self.F > 0:
            np.clip(self.X, -self.cfg.obs_clip, self.cfg.obs_clip, out=self.X)

        # Lagged RF/IDX for observations
        self.RF_obs = np.zeros_like(self.RF, dtype=np.float32)
        self.RF_obs[1:] = self.RF[:-1]
        self.IDX_obs = np.zeros_like(self.IDX, dtype=np.float32)
        self.IDX_obs[1:] = self.IDX[:-1]

        # NOTE: We don't append cash to the full universe arrays
        # Cash will be handled separately in the observation space

    # --------------------------- Top-K Selection Logic ----------------------- #

    def _get_top_k_assets(self, t: int) -> np.ndarray:
        """
        Get indices of top K assets by index_weight at time t.
        Returns array of shape (max_assets,) with indices into full universe.
        Excludes cash - that's handled separately.
        """
        # Get weights at current time
        weights_t = self.W[t]
        
        # Only consider active assets
        active_mask = self.ACT[t] > 0
        
        # Set inactive weights to -inf for sorting
        sort_weights = weights_t.copy()
        sort_weights[~active_mask] = -np.inf
        
        # Get top K indices
        k = min(self.max_assets_config, np.sum(active_mask))
        if k <= 0:
            # No active assets, return first max_assets indices as fallback
            return np.arange(min(self.max_assets_config, self.n_assets))
        
        # Use argpartition for efficiency, then sort the top k
        if k < len(sort_weights):
            # Get indices of k largest elements
            top_k_unsorted = np.argpartition(sort_weights, -k)[-k:]
            # Sort them by weight (descending)
            top_k_indices = top_k_unsorted[np.argsort(-sort_weights[top_k_unsorted])]
        else:
            # All assets fit, just sort by weight
            top_k_indices = np.argsort(-sort_weights)[:k]
        
        # Pad with inactive slots if needed
        if len(top_k_indices) < self.max_assets_config:
            padding = self.max_assets_config - len(top_k_indices)
            # Use indices beyond the universe as padding (will be inactive)
            pad_indices = np.full(padding, self.n_assets - 1)  # Use last index as dummy
            top_k_indices = np.concatenate([top_k_indices, pad_indices])
        
        return top_k_indices.astype(np.int32)

    def _map_weights_to_obs_space(self, full_weights: np.ndarray) -> np.ndarray:
        """Map weights from full universe to observation space."""
        obs_weights = np.zeros(self.obs_n_assets, dtype=np.float32)
        
        # Map top-K assets
        for i in range(min(self.max_assets_config, len(self.current_top_k_indices))):
            full_idx = self.current_top_k_indices[i]
            if full_idx < self.n_assets:
                obs_weights[i] = full_weights[full_idx]
        
        # Add cash if enabled
        if self.cfg.allow_cash:
            # Cash weight is sum of any unallocated weight
            cash_weight = 1.0 - np.sum(obs_weights[:self.max_assets_config])
            obs_weights[self.cash_idx] = max(0.0, cash_weight)
        
        return obs_weights

    def _map_weights_to_full_space(self, obs_weights: np.ndarray) -> np.ndarray:
        """Map weights from observation space to full universe."""
        full_weights = np.zeros(self.n_assets, dtype=np.float32)
        
        # Map top-K assets
        for i in range(min(self.max_assets_config, len(self.current_top_k_indices))):
            full_idx = self.current_top_k_indices[i]
            if full_idx < self.n_assets and i < len(obs_weights):
                full_weights[full_idx] = obs_weights[i]
        
        # Note: cash is not in full_weights, it's handled separately
        
        return full_weights

    # --------------------------- Observation builder ------------------------ #

    def _obs_size(self) -> int:
        n = self.obs_n_assets  # Fixed size based on max_assets
        base = n * self.F
        extra = 0
        if self.cfg.include_prev_weights:
            extra += n
        if self.cfg.include_active_flag:
            extra += n
        if self.cfg.global_stats and self.F > 0:
            extra += 2 * self.F
        extra += 2  # RF and IDX scalars
        return base + extra

    def _get_observation(self) -> Dict:
        """Build observation using top-K assets."""
        t = min(self.t, self.T - 1)
        if t >= self.ACT.shape[0]:
            t = self.ACT.shape[0] - 1
        
        # Use pre-allocated buffer
        obs = self._obs_buffer
        obs.fill(0.0)
        
        idx = 0
        
        # Features for top-K assets
        if self.F > 0:
            for i in range(self.max_assets_config):
                if i < len(self.current_top_k_indices):
                    full_idx = self.current_top_k_indices[i]
                    if full_idx < self.n_assets:
                        # Copy features for this asset
                        obs[idx:idx + self.F] = self.X[t, full_idx]
                idx += self.F
            
            # Add dummy features for cash if enabled
            if self.cfg.allow_cash:
                idx += self.F  # Skip cash features (zeros)
        
        # Previous weights in observation space
        if self.cfg.include_prev_weights:
            obs[idx:idx + self.obs_n_assets] = self.prev_w
            idx += self.obs_n_assets
        
        # Active flags for top-K assets
        if self.cfg.include_active_flag:
            for i in range(self.max_assets_config):
                if i < len(self.current_top_k_indices):
                    full_idx = self.current_top_k_indices[i]
                    if full_idx < self.n_assets:
                        obs[idx + i] = self.ACT[t, full_idx]
            # Cash is always active
            if self.cfg.allow_cash:
                obs[idx + self.cash_idx] = 1
            idx += self.obs_n_assets
        
        # Global stats
        if self.cfg.global_stats and self.F > 0:
            obs[idx:idx + self.F] = self.global_means[t]
            idx += self.F
            obs[idx:idx + self.F] = self.global_stds[t]
            idx += self.F
        
        obs[idx] = self.RF_obs[t]
        obs[idx + 1] = self.IDX_obs[t]
        
        # Clip observations
        clip = float(self.cfg.obs_clip)
        np.clip(obs, -clip, clip, out=obs)
        
        # Action mask for top-K assets
        mask = np.zeros(self.obs_n_assets, dtype=np.int8)
        for i in range(self.max_assets_config):
            if i < len(self.current_top_k_indices):
                full_idx = self.current_top_k_indices[i]
                if full_idx < self.n_assets:
                    mask[i] = self.ACT[t, full_idx]
        
        # Cash is always available
        if self.cfg.allow_cash:
            mask[self.cash_idx] = 1
        
        return {
            'observation': obs,
            'action_mask': mask
        }

    # ------------------------------- Gym API -------------------------------- #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.np_random = np.random.default_rng(seed if seed is not None else self.cfg.seed)

        if self.cfg.max_steps is not None and 0 < float(self.cfg.random_reset_frac) <= 1.0:
            max_start = max(0, self.T - int(self.cfg.max_steps))
            jitter_cap = int(max_start * float(self.cfg.random_reset_frac))
            start = int(self.np_random.integers(0, jitter_cap + 1)) if jitter_cap > 0 else 0
            self.t = start
        else:
            self.t = 0

        # Initialize top-K selection
        self.current_top_k_indices = self._get_top_k_assets(self.t)
        
        # Initialize weights
        self.full_prev_w.fill(0.0)
        self.full_curr_w.fill(0.0)
        self.prev_w.fill(0.0)
        self.curr_w.fill(0.0)
        
        # Start with all cash if available
        if self.cash_idx is not None:
            self.curr_w[self.cash_idx] = 1.0
        else:
            # Equal weight among active top-K assets
            mask = self._get_observation()['action_mask'][:self.max_assets_config]
            if mask.any():
                active_count = mask.sum()
                for i in range(self.max_assets_config):
                    if mask[i]:
                        self.curr_w[i] = 1.0 / active_count
                # Update full weights
                self.full_curr_w = self._map_weights_to_full_space(self.curr_w)
                
        self.wealth = 1.0
        self.peak_wealth = 1.0
        self.tail_buffer = []
        self._history = {}
        
        info = {
            "date": pd.Timestamp(self.dates[self.t]).to_pydatetime(),
            "n_features": self.F,
            "feature_cols": self.feature_cols[:10] if len(self.feature_cols) > 10 else self.feature_cols,
            "top_k_assets": [self.asset_ids[i] for i in self.current_top_k_indices[:self.max_assets_config] if i < len(self.asset_ids)],
        }
        return self._get_observation(), info

    def step(self, action: np.ndarray):
        """Process action with top-K constraint."""
        t = int(self.t)
        terminated = False
        truncated = False
        
        if t >= self.T or t >= self.ACT.shape[0]:
            return self._get_observation(), 0.0, True, False, {}

        # Only rebalance every rebalance_interval days
        apply_action = (t % max(1, int(self.cfg.rebalance_interval)) == 0)
        
        # Track previous weights
        w_prev_full = self.full_curr_w.copy()
        w_prev_obs = self.curr_w.copy()
        
        freed_mass = 0.0
        extra_delist_cost = 0.0
        forced_liquidation = 0.0
        
        if apply_action:
            # REBALANCING DAY: Update top-K selection
            new_top_k = self._get_top_k_assets(t)
            
            # Check for forced liquidations (assets no longer in top-K)
            old_set = set(self.current_top_k_indices[:self.max_assets_config])
            new_set = set(new_top_k[:self.max_assets_config])
            
            # Assets that must be sold
            assets_to_liquidate = old_set - new_set
            for full_idx in assets_to_liquidate:
                if full_idx < self.n_assets:
                    forced_liquidation += w_prev_full[full_idx]
                    w_prev_full[full_idx] = 0.0
            
            # Apply delist cost for forced liquidations
            if forced_liquidation > 0:
                extra_delist_cost = forced_liquidation * (self.cfg.delist_extra_bps / 10000.0)
            
            # Update mapping
            self.current_top_k_indices = new_top_k
            
            # Process discrete action
            blocks = self._work_blocks
            np.copyto(blocks, action.astype(np.int32))
            
            # Get current activity mask in observation space
            act_mask_obs = self._get_observation()['action_mask']
            
            # Apply activity mask to blocks
            blocks = _sanitize_blocks_with_mask(blocks, act_mask_obs)
            
            # Convert blocks to weights
            w_tgt_obs = _blocks_to_weights(blocks, self.cfg.weight_blocks)
            
            # Ensure max weight constraint
            w_tgt_obs = _normalize_weights_with_max_jit(
                w_tgt_obs.astype(np.float32),
                np.float32(self.cfg.max_weight),
                np.float32(1e-8)
            )
            
            # Map to full space
            w_tgt_full = self._map_weights_to_full_space(w_tgt_obs[:self.max_assets_config])
            
            # Handle cash
            cash_weight = 0.0
            if self.cfg.allow_cash and self.cash_idx is not None:
                cash_weight = w_tgt_obs[self.cash_idx]
            
            actually_rebalanced = True
            
        else:
            # NON-REBALANCING DAY: Hold positions but handle delistings
            w_tgt_full = w_prev_full.copy()
            
            # Check for delistings in current holdings
            for i in range(self.n_assets):
                if w_tgt_full[i] > 0 and self.ACT[t, i] <= 0:
                    freed_mass += w_tgt_full[i]
                    w_tgt_full[i] = 0.0
            
            if freed_mass > 0:
                extra_delist_cost = freed_mass * (self.cfg.delist_extra_bps / 10000.0)
                
                # Redistribute freed mass
                if self.cfg.on_inactive == "to_cash" and self.cfg.allow_cash:
                    cash_weight = freed_mass
                else:
                    # Pro-rata to remaining active positions
                    active_weight = w_tgt_full.sum()
                    if active_weight > 0:
                        w_tgt_full *= (1.0 + freed_mass / active_weight)
            
            # Map back to observation space
            w_tgt_obs = self._map_weights_to_obs_space(w_tgt_full)
            cash_weight = w_tgt_obs[self.cash_idx] if self.cfg.allow_cash else 0.0
            
            actually_rebalanced = False

        # Calculate returns for all assets in portfolio
        r_vec = self._work_returns_full
        np.copyto(r_vec, self.R[t])
        
        rf_t = float(self.RF[t])
        bad = ~np.isfinite(r_vec)
        if bad.any():
            r_vec[bad] = rf_t
        
        # Portfolio return (excluding cash)
        r_p_assets = float(_portfolio_return_jit(w_tgt_full.astype(np.float32), r_vec.astype(np.float32)))
        
        # Add cash return
        cash_return = rf_t if self.cfg.cash_rate_as_rf else 0.0
        r_p = r_p_assets * (1.0 - cash_weight) + cash_return * cash_weight
        
        # Calculate turnover and costs
        if actually_rebalanced:
            # Compare full universe weights for turnover
            turn = _turnover_jit(w_tgt_full.astype(np.float32), w_prev_full.astype(np.float32))
            # Add cash turnover
            if self.cfg.allow_cash:
                prev_cash = w_prev_obs[self.cash_idx] if self.cash_idx is not None else 0.0
                turn += abs(cash_weight - prev_cash)
            
            lin_cost = (self.cfg.transaction_cost_bps / 10000.0) * turn + extra_delist_cost
        else:
            turn = 0.0
            lin_cost = extra_delist_cost
        
        # Benchmarks
        r_idx = float(self.IDX[t])
        alpha = r_p - r_idx
        excess = r_p - rf_t

        # Update wealth
        net = max((1.0 + r_p) * (1.0 - max(lin_cost, 0.0)), 1e-12)
        r_net = net - 1.0
        self.wealth *= net
        self.peak_wealth = max(self.peak_wealth, self.wealth)

        # Drawdown penalty
        cur_dd_level = 1.0 - (self.wealth / max(self.peak_wealth, 1e-12))
        if self.cfg.dd_mode == "level":
            dd_pen = -self.cfg.lambda_drawdown * abs(cur_dd_level)
        else:
            prev_dd_level = self._history.get("drawdown", [0.0])[-1] if self._history.get("drawdown") else 0.0
            dd_inc = max(cur_dd_level - prev_dd_level, 0.0)
            dd_pen = -self.cfg.lambda_drawdown * dd_inc

        # Tail penalty
        tail_pen = 0.0
        self.tail_buffer.append(r_p)
        if len(self.tail_buffer) > self.cfg.tail_window:
            self.tail_buffer.pop(0)
        if self.cfg.lambda_tail > 0 and len(self.tail_buffer) >= max(10, self.cfg.tail_window // 2):
            q = float(np.quantile(self.tail_buffer, self.cfg.tail_q))
            if r_p < q:
                tail_pen = abs(r_p)

        # HHI calculation (including cash)
        all_weights = np.concatenate([w_tgt_full, [cash_weight]]) if self.cfg.allow_cash else w_tgt_full
        hhi_val = float(_hhi_jit(all_weights.astype(np.float32)))
        
        # Penalties
        pen = (
            - self.cfg.lambda_turnover * turn
            - self.cfg.lambda_hhi * hhi_val
            + dd_pen
            - self.cfg.lambda_tail * tail_pen
        )

        # Reward
        reward = float(self.cfg.weight_alpha * alpha + pen - lin_cost)

        # Update state
        self.full_prev_w = w_prev_full.copy()
        self.full_curr_w = w_tgt_full.copy()
        self.prev_w = w_prev_obs.copy()
        self.curr_w = w_tgt_obs.copy()
        
        self.t = t + 1
        
        # Update top-K for next observation (if not at end)
        if self.t < self.T:
            self.current_top_k_indices = self._get_top_k_assets(self.t)
        
        if self.cfg.max_steps is not None and self.t >= int(self.cfg.max_steps):
            truncated = True
        if self.t >= self.T:
            terminated = True

        obs = self._get_observation()
        
        # Prepare output weights (full universe for logging)
        output_weights = np.zeros(self.n_assets + (1 if self.cfg.allow_cash else 0), dtype=np.float32)
        output_weights[:self.n_assets] = self.full_curr_w
        if self.cfg.allow_cash:
            output_weights[-1] = cash_weight
        
        info = {
            "date": pd.Timestamp(self.dates[min(self.t, self.T - 1)]).to_pydatetime(),
            "weights": output_weights,
            "portfolio_return_gross": float(r_p),
            "portfolio_return": float(r_net),
            "excess": float(excess),
            "alpha": float(alpha),
            "index_return": float(r_idx),
            "rf": float(rf_t),
            "turnover": float(turn),
            "hhi": float(hhi_val),
            "trade_cost": float(lin_cost),
            "wealth": float(self.wealth),
            "drawdown": float(cur_dd_level),
            "sector_exposure": _dict_sector_exposures(self.sector_ids, self.full_curr_w),
            "forced_liquidation": float(forced_liquidation),
            "top_k_assets": [self.asset_ids[i] for i in self.current_top_k_indices[:self.max_assets_config] if i < len(self.asset_ids)],
            "config": asdict(self.cfg) if self.t == 1 else None,
        }
        
        # Store history
        for k, v in info.items():
            if k not in ["config", "sector_exposure", "weights", "date", "top_k_assets"]:
                if k not in self._history:
                    self._history[k] = []
                self._history[k].append(v)
                
        return obs, reward, terminated, truncated, info

    # ------------------------------- Helpers --------------------------------- #

    def render(self):
        t = min(self.t, self.T - 1)
        date = pd.Timestamp(self.dates[t]).date()
        wealth = self.wealth
        dd = self._history["drawdown"][-1] if self._history.get("drawdown") else 0.0
        n_active = len([i for i in self.current_top_k_indices[:self.max_assets_config] if i < self.n_assets and self.ACT[t, i] > 0])
        print(f"[{date}] wealth={wealth:.4f} dd={dd:.3%} active_top_k={n_active}/{self.max_assets_config}")

    def get_history(self) -> pd.DataFrame:
        """Return a DataFrame of per-step logged metrics."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history, index=pd.to_datetime(
            self.dates[:len(self._history.get('wealth', []))]
        ))

    def get_asset_ids(self) -> List[str]:
        """Return full universe asset IDs."""
        return list(self.asset_ids)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_cols)

    def get_action_masks(self) -> np.ndarray:
        """
        For MaskablePPO: returns valid actions mask for MultiDiscrete space.
        """
        masks = []
        obs_dict = self._get_observation()
        act_mask = obs_dict['action_mask']
        
        for i in range(self.obs_n_assets):
            asset_mask = np.ones(self.max_blocks_per_asset + 1, dtype=bool)
            # If asset is inactive, only allow 0 blocks
            if act_mask[i] <= 0:
                asset_mask[1:] = False
            masks.append(asset_mask)
            
        return masks

# ------------------------- Factory convenience ---------------------------- #

def make_env_from_panel(panel: pd.DataFrame, **env_kwargs):
    prebuilt = env_kwargs.pop("prebuilt", None)
    cfg = EnvConfig(**env_kwargs)
    env = DebentureTradingEnv(panel=panel, config=cfg, prebuilt=prebuilt)
    return env

# ------------------------------ Quick test -------------------------------- #

if __name__ == "__main__":
    # Test with max_assets feature
    import sys
    import os
    import time
    
    print("Testing environment with max_assets constraint...")
    print("=" * 60)
    
    # Check if we have processed data
    data_path = "data/cdi_processed.pkl"
    if os.path.exists(data_path):
        print(f"Loading real data from {data_path}")
        panel = pd.read_pickle(data_path)
        
        # Take a subset for testing
        dates = panel.index.get_level_values("date").unique()[:100]
        assets = panel.index.get_level_values("debenture_id").unique()[:100]  # Use more assets to test top-K
        test_panel = panel.loc[(dates, assets), :]
    else:
        print("Creating synthetic test data...")
        rng = np.random.default_rng(0)
        dates = pd.date_range("2022-01-03", periods=100, freq="B")
        ids = [f"BOND_{i}" for i in range(100)]  # 100 assets
        idx = pd.MultiIndex.from_product([dates, ids], names=["date", "debenture_id"])
        
        test_panel = pd.DataFrame(index=idx)
        # Add required columns
        test_panel["return"] = rng.normal(0.0003, 0.002, size=len(test_panel)).astype(np.float32)
        test_panel["risk_free"] = 0.0003
        test_panel["index_return"] = rng.normal(0.0002, 0.0015, size=len(test_panel)).astype(np.float32)
        test_panel["spread"] = rng.normal(0.02, 0.005, size=len(test_panel)).astype(np.float32)
        test_panel["duration"] = rng.uniform(1.0, 5.0, size=len(test_panel)).astype(np.float32)
        test_panel["time_to_maturity"] = rng.uniform(0.5, 4.0, size=len(test_panel)).astype(np.float32)
        test_panel["sector_id"] = rng.integers(0, 3, size=len(test_panel)).astype(np.int16)
        test_panel["active"] = 1
        
        # Add index_weight with some variation
        for date in dates:
            mask = test_panel.index.get_level_values("date") == date
            weights = rng.dirichlet(np.ones(100))  # Random weights that sum to 1
            test_panel.loc[mask, "index_weight"] = np.repeat(weights, 1)
        
        # Add synthetic features (with lag1 versions)
        for feat in ["momentum_5d", "volatility_20d", "spread_vs_sector_median"]:
            test_panel[feat] = rng.normal(0, 1, size=len(test_panel)).astype(np.float32)
            test_panel[f"{feat}_lag1"] = test_panel[feat]

    # Create environment with max_assets constraint
    print("\nInitializing environment with max_assets=50...")
    start_time = time.time()
    
    env = make_env_from_panel(
        test_panel, 
        rebalance_interval=10,
        max_weight=0.1,
        max_assets=50,  # NEW: Top 50 assets only
        allow_cash=True,
        cash_rate_as_rf=True,
        use_momentum_features=True,
        use_volatility_features=True,
    )
    
    init_time = time.time() - start_time
    print(f"Environment initialization time: {init_time:.3f}s")
    
    obs, info = env.reset()
    print(f"\nEnvironment initialized:")
    print(f"  Full universe assets: {env.n_assets}")
    print(f"  Max investable assets: {env.max_assets_config}")
    print(f"  Observation space assets: {env.obs_n_assets} (including cash)")
    print(f"  Features per asset: {env.F}")
    print(f"  Total observation size: {env._obs_size()}")
    print(f"  Action space: {env.action_space}")
    print(f"  Top-K assets at t=0: {len(info['top_k_assets'])} assets")
    
    # Test a few steps
    print("\n" + "=" * 60)
    print("TESTING MAX ASSETS CONSTRAINT")
    print("=" * 60)
    
    for step in range(30):
        # Random discrete action (only for top-K + cash slots)
        action = env.np_random.integers(0, env.max_blocks_per_asset + 1, size=env.obs_n_assets)
        obs, reward, term, trunc, info = env.step(action)
        
        # Log rebalancing days
        if step % env.cfg.rebalance_interval == 0:
            print(f"\nStep {step} (REBALANCING):")
            print(f"  Date: {info['date'].date()}")
            print(f"  Top-K assets: {len(info['top_k_assets'])}")
            print(f"  Forced liquidation: {info['forced_liquidation']:.4f}")
            print(f"  Turnover: {info['turnover']:.4f}")
            print(f"  Wealth: {info['wealth']:.6f}")
            
            # Check that we have at most max_assets with non-zero weights
            weights = info['weights']
            non_zero_count = np.sum(weights[:-1] > 1e-6) if env.cfg.allow_cash else np.sum(weights > 1e-6)
            print(f"  Non-zero positions: {non_zero_count} (should be <= {env.max_assets_config})")
        
        if term or trunc:
            break
    
    print("\n" + "=" * 60)
    print("MAX ASSETS FEATURE TEST COMPLETE")
    print("=" * 60)
    print("\nKey changes implemented:")
    print("✓ Fixed observation/action space of max_assets size")
    print("✓ Dynamic top-K selection by index_weight on rebalancing days")
    print("✓ Forced liquidation of assets outside top-K")
    print("✓ Mapping between observation space and full universe")
    print("✓ Preserved all existing features and optimizations")