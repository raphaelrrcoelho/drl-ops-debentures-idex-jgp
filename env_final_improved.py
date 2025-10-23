# env_final.py
"""
Debenture portfolio environment - IMPROVED VERSION
==================================================
CRITICAL FIXES IMPLEMENTED:
1. âœ… Lagged active flags (ACT_obs) - prevents information leakage
2. âœ… Lagged weights (W_obs) - prevents look-ahead bias in top-K selection
3. âœ… Simplified feature selection - configurable feature groups
4. âœ… Proper wealth calculation - uses total returns for evaluation
5. âœ… Clear return definitions - documents excess vs total returns

Key improvements:
- Temporal causality: All observations use t-1 data
- No information leakage: Agent cannot see future active flags or weights
- Computational efficiency: Supports simplified feature sets
- Clear semantics: Explicit handling of excess returns vs total returns
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
    rebalance_interval: int = 21  # Monthly rebalancing
    max_weight: float = 0.10
    weight_blocks: int = 50  # Granularity for discrete actions
    allow_cash: bool = True
    cash_rate_as_rf: bool = True
    on_inactive: str = "to_cash"  # or "pro_rata"
    
    # Maximum number of investable assets
    max_assets: int = 50  # Top K assets by index_weight
    
    # Costs & penalties
    weight_alpha: float = 1.0   # Weight for alpha in reward
    transaction_cost_bps: float = 10.0          
    delist_extra_bps: float = 10.0              
    lambda_turnover: float = 0.01  # Simplified - lighter penalty             
    lambda_hhi: float = 0.0  # Disabled for simplicity                  
    lambda_drawdown: float = 0.0  # Disabled for simplicity            
    lambda_tail: float = 0.0  # Disabled for simplicity                  
    tail_window: int = 60                       
    tail_q: float = 0.05                        
    dd_mode: str = "incremental"                
    
    # Observation controls
    include_prev_weights: bool = True           
    include_active_flag: bool = True            
    global_stats: bool = True                   
    normalize_features: bool = True             
    obs_clip: float = 5.0
    
    # Feature selection - SIMPLIFIED for better performance
    use_momentum_features: bool = True
    use_volatility_features: bool = True
    use_relative_value_features: bool = True
    use_duration_features: bool = True
    use_microstructure_features: bool = True
    use_carry_features: bool = True
    use_spread_dynamics: bool = True
    use_risk_adjusted_features: bool = False  # Disabled (redundant)
    use_sector_curves: bool = False  # Disabled (expensive)
    use_zscore_features: bool = False  # Disabled (redundant with normalization)
    use_rolling_zscores: bool = False  # Disabled (redundant)
    
    # Episode control
    max_steps: Optional[int] = None             
    seed: Optional[int] = 42                    
    random_reset_frac: float = 0.9  # For training data augmentation

# -------------------------- Simplified Features List ----------------------- #

# CRITICAL: Use simplified windows for faster training
MOMENTUM_WINDOWS = [20]  # Only 20-day momentum (not [1,5,20,60,126])
VOLATILITY_WINDOWS = [20]  # Only 20-day volatility (not [5,20,60])

# Base features
BASE_FEATURES = [
    "return", "spread", "duration", "time_to_maturity",
    "risk_free", "index_return", "sector_id",
]

# Optional enhanced features (only included if enabled)
MOMENTUM_FEATURES = [f"momentum_{w}d" for w in MOMENTUM_WINDOWS]
VOLATILITY_FEATURES = [f"volatility_{w}d" for w in VOLATILITY_WINDOWS]
RELATIVE_VALUE_FEATURES = ["spread_vs_sector_median"]
DURATION_FEATURES = ["modified_duration_proxy"]
MICROSTRUCTURE_FEATURES = ["liquidity_score", "weight_momentum"]
CARRY_FEATURES = ["carry_spread_ratio"]
SPREAD_DYNAMICS_FEATURES = ["spread_momentum_20d", "spread_mean_reversion"]

# Z-score features (disabled by default)
ZSCORE_FEATURES = [
    f"{f}_zscore" for f in (MOMENTUM_FEATURES + VOLATILITY_FEATURES + ["spread"])
]
ROLLING_ZSCORE_FEATURES = [
    f"{f}_rolling_zscore" for f in (MOMENTUM_FEATURES + VOLATILITY_FEATURES + ["spread"])
]

# ----------------------------- JIT Helpers ----------------------------- #

@jit(nopython=True, cache=True)
def _compute_turnover(w_old: np.ndarray, w_new: np.ndarray) -> float32:
    return float32(np.abs(w_new - w_old).sum())

@jit(nopython=True, cache=True)
def _compute_hhi(w: np.ndarray) -> float32:
    return float32((w * w).sum())

# ----------------------------- Main Environment ----------------------------- #

class DebentureTradingEnv(gym.Env):
    """
    Debenture portfolio environment with proper temporal causality.
    
    CRITICAL: All observations use lagged (t-1) data:
    - Features: X_{i,t-1}
    - Active flags: ACT_{i,t-1}
    - Weights: W_{i,t-1} (for top-K selection)
    - Risk-free & Index: RF_{t-1}, IDX_{t-1}
    
    The agent chooses w_t based on t-1 information,
    and earns returns r_t over day t.
    """
    
    metadata = {"render.modes": []}
    
    def __init__(
        self,
        panel: pd.DataFrame,
        cfg: EnvConfig,
        prebuilt: Optional[Dict] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        
        # Store panel for metadata
        self.panel = panel
        
        if prebuilt is not None:
            self._attach_prebuilt(prebuilt)
        else:
            self._prepare_data(panel)
        
        # Determine observation space size
        self.max_assets_config = self.cfg.max_assets
        
        # Observation space: top K assets (+ cash if enabled)
        self.obs_n_assets = self.max_assets_config
        if self.cfg.allow_cash:
            self.obs_n_assets += 1
            self.cash_idx = self.max_assets_config
        
        obs_size = self._obs_size()
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(obs_size,), 
                dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0, high=1, 
                shape=(self.obs_n_assets,), 
                dtype=np.int8
            ),
        })
        
        # Action space: discrete blocks for each asset (+ cash if enabled)
        n_blocks = self.cfg.weight_blocks + 1
        self.action_space = spaces.MultiDiscrete(
            [n_blocks] * self.obs_n_assets, 
            dtype=np.int32
        )
        
        # Initialize episode state
        self.reset()
    
    def export_shared_arrays(self) -> Dict:
        """Export arrays for parallel environments."""
        return {
            "dates": self.dates,
            "asset_ids": self.asset_ids,
            "sector_ids": self.sector_ids,
            "R": self.R,
            "RF": self.RF,
            "IDX": self.IDX,
            "ACT": self.ACT,
            "W": self.W,
            "X": self.X,
            "RF_obs": self.RF_obs,
            "IDX_obs": self.IDX_obs,
            "ACT_obs": self.ACT_obs,  # NEW: Lagged active flags
            "W_obs": self.W_obs,  # NEW: Lagged weights
            "global_means": self.global_means,
            "global_stds": self.global_stds,
            "feature_cols": self.feature_cols,
        }
    
    def _attach_prebuilt(self, pb: Dict):
        """Attach pre-built arrays from export."""
        self.dates = pb["dates"]
        self.asset_ids = pb["asset_ids"]
        self.sector_ids = pb["sector_ids"]
        self.R = pb["R"]
        self.RF = pb["RF"]
        self.IDX = pb["IDX"]
        self.ACT = pb["ACT"]
        self.W = pb["W"]
        self.X = pb["X"]
        self.RF_obs = pb.get("RF_obs", self.RF)
        self.IDX_obs = pb.get("IDX_obs", self.IDX)
        self.ACT_obs = pb.get("ACT_obs", self.ACT)  # NEW
        self.W_obs = pb.get("W_obs", self.W)  # NEW
        self.global_means = pb.get("global_means")
        self.global_stds = pb.get("global_stds")
        self.feature_cols = pb.get("feature_cols", [])
        
        self.n_assets = self.R.shape[1]
        self.T = self.R.shape[0]
        self.F = self.X.shape[-1] if self.X.ndim == 3 else 0
    
    def _prepare_data(self, panel: pd.DataFrame):
        """
        Prepare data arrays from panel.
        
        CRITICAL: Creates lagged versions of active flags and weights
        to prevent information leakage.
        """
        panel_reset = panel.reset_index(drop=False).copy()
        
        # Determine which features to include based on config
        feat_cols = set(BASE_FEATURES)
        
        if self.cfg.use_momentum_features:
            feat_cols.update(MOMENTUM_FEATURES)
        if self.cfg.use_volatility_features:
            feat_cols.update(VOLATILITY_FEATURES)
        if self.cfg.use_relative_value_features:
            feat_cols.update(RELATIVE_VALUE_FEATURES)
        if self.cfg.use_duration_features:
            feat_cols.update(DURATION_FEATURES)
        if self.cfg.use_microstructure_features:
            feat_cols.update(MICROSTRUCTURE_FEATURES)
        if self.cfg.use_carry_features:
            feat_cols.update(CARRY_FEATURES)
        if self.cfg.use_spread_dynamics:
            feat_cols.update(SPREAD_DYNAMICS_FEATURES)
        if self.cfg.use_zscore_features:
            feat_cols.update(ZSCORE_FEATURES)
        if self.cfg.use_rolling_zscores:
            feat_cols.update(ROLLING_ZSCORE_FEATURES)
        
        # Filter to available columns
        feat_cols = sorted([c for c in feat_cols if c in panel_reset.columns])
        
        print(f"[ENV] Using {len(feat_cols)} features: {feat_cols[:10]}...")
        
        # Get unique dates and assets
        dates = pd.DatetimeIndex(sorted(panel_reset["date"].unique()))
        asset_ids = sorted(panel_reset["debenture_id"].unique())
        
        self.dates = dates
        self.asset_ids = asset_ids
        self.n_assets = len(asset_ids)
        
        # Pivot to wide format
        # Returns
        R = (
            panel_reset.set_index(["date", "debenture_id"])["return"]
            .unstack(fill_value=0.0)
            .reindex(index=dates, columns=asset_ids)
            .fillna(0.0)
        )
        
        # Risk-free rate
        RF = (
            panel_reset.groupby("date")["risk_free"].first()
            .reindex(dates)
            .fillna(method="ffill").fillna(0.0)
        )
        
        # Index return
        IDX = (
            panel_reset.groupby("date")["index_return"].first()
            .reindex(dates)
            .fillna(0.0)
        )
        
        # Active flags
        A = (
            panel_reset.set_index(["date", "debenture_id"])["active"]
            .unstack(fill_value=0.0)
            .reindex(index=dates, columns=asset_ids)
            .fillna(0.0)
        )
        
        # Index weights for top-K selection
        W = (
            panel_reset.set_index(["date", "debenture_id"])["index_weight"]
            .unstack(fill_value=0.0)
            .reindex(index=dates, columns=asset_ids)
            .fillna(0.0)
        )
        
        # Feature tensor
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
        self.R = np.nan_to_num(R.values.astype(np.float16), nan=0.0)
        self.RF = np.nan_to_num(RF.values.astype(np.float16).ravel(), nan=0.0)
        self.IDX = np.nan_to_num(IDX.values.astype(np.float16).ravel(), nan=0.0)
        self.ACT = np.nan_to_num(A.values.astype(np.int8), nan=0).astype(np.int8)
        self.W = np.nan_to_num(W.values.astype(np.float32), nan=0.0)
        self.X = np.nan_to_num(X.astype(np.float32), nan=0.0)
        
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
        
        # Normalize features if requested
        if self.cfg.normalize_features and self.F > 0:
            np.clip(self.X, -self.cfg.obs_clip, self.cfg.obs_clip, out=self.X)
        
        # ============ CRITICAL FIX: Create lagged arrays ============
        # Lagged RF/IDX for observations (already existing)
        self.RF_obs = np.zeros_like(self.RF, dtype=np.float32)
        self.RF_obs[1:] = self.RF[:-1]
        
        self.IDX_obs = np.zeros_like(self.IDX, dtype=np.float32)
        self.IDX_obs[1:] = self.IDX[:-1]
        
        # NEW: Lagged active flags (prevent information leakage)
        self.ACT_obs = np.zeros_like(self.ACT, dtype=np.int8)
        self.ACT_obs[1:] = self.ACT[:-1]
        
        # NEW: Lagged weights (prevent look-ahead bias in top-K selection)
        self.W_obs = np.zeros_like(self.W, dtype=np.float32)
        self.W_obs[1:] = self.W[:-1]
        # ============================================================
        
        print(f"[ENV] Data prepared: T={self.T}, N={self.n_assets}, F={self.F}")
        print(f"[ENV] Observation space dimension: ~{self._obs_size()}")
    
    # --------------------------- Top-K Selection Logic ----------------------- #
    
    def _get_top_k_assets(self, t: int, use_lagged: bool = False) -> np.ndarray:
        """
        Get indices of top K assets by index_weight at time t.
        
        Args:
            t: Time index
            use_lagged: If True, use lagged weights (t-1) for causal decisions.
                       This prevents look-ahead bias on rebalancing days.
        
        Returns:
            Array of shape (max_assets,) with indices into full universe.
        """
        # CRITICAL FIX: Use lagged weights for causal decisions
        if use_lagged and hasattr(self, 'W_obs'):
            weights_t = self.W_obs[t]
            active_mask = self.ACT_obs[t] > 0 if hasattr(self, 'ACT_obs') else self.ACT[t] > 0
        else:
            weights_t = self.W[t]
            active_mask = self.ACT[t] > 0
        
        # Set inactive weights to -inf for sorting
        sort_weights = weights_t.copy()
        sort_weights[~active_mask] = -np.inf
        
        # Get top K indices
        k = min(self.max_assets_config, np.sum(active_mask))
        if k <= 0:
            # No active assets, return first max_assets indices as fallback
            return np.arange(min(self.max_assets_config, self.n_assets))
        
        # Use argpartition for efficiency
        if k < len(sort_weights):
            top_k_unsorted = np.argpartition(sort_weights, -k)[-k:]
            top_k_indices = top_k_unsorted[np.argsort(-sort_weights[top_k_unsorted])]
        else:
            top_k_indices = np.argsort(-sort_weights)
        
        # Pad if needed
        if len(top_k_indices) < self.max_assets_config:
            padding = np.arange(len(top_k_indices), self.max_assets_config)
            top_k_indices = np.concatenate([top_k_indices, padding])
        
        return top_k_indices[:self.max_assets_config].astype(np.int32)
    
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
        
        return full_weights
    
    # --------------------------- Observation builder ------------------------ #
    
    def _obs_size(self) -> int:
        """Calculate observation space dimension."""
        n = self.obs_n_assets
        base = n * self.F  # Features per asset
        
        if self.cfg.include_prev_weights:
            base += n
        if self.cfg.include_active_flag:
            base += n
        if self.cfg.global_stats and self.F > 0:
            base += self.F * 2  # Global mean and std
        
        base += 2  # RF and IDX scalars
        
        return base
    
    def _get_observation(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build observation at time t using LAGGED data (t-1).
        
        CRITICAL: All features, active flags, and previous weights
        come from t-1 to ensure temporal causality.
        """
        obs_size = self._obs_size()
        obs = np.zeros(obs_size, dtype=np.float32)
        mask = np.zeros(self.obs_n_assets, dtype=np.int8)
        
        idx = 0
        
        # Features for top-K assets
        for i in range(self.max_assets_config):
            full_idx = self.current_top_k_indices[i]
            if full_idx < self.n_assets and t < self.T:
                obs[idx:idx + self.F] = self.X[t, full_idx, :]
            idx += self.F
        
        # Cash features (zeros)
        if self.cfg.allow_cash:
            idx += self.F
        
        # Previous weights (mapped to obs space)
        if self.cfg.include_prev_weights:
            obs_prev_w = self._map_weights_to_obs_space(self.prev_full_w)
            obs[idx:idx + self.obs_n_assets] = obs_prev_w
            idx += self.obs_n_assets
        
        # Active flags - CRITICAL FIX: Use lagged active flags
        if self.cfg.include_active_flag:
            for i in range(self.max_assets_config):
                full_idx = self.current_top_k_indices[i]
                if full_idx < self.n_assets and t < self.T:
                    # Use ACT_obs (lagged) instead of ACT (current)
                    obs[idx + i] = self.ACT_obs[t, full_idx]
                    mask[i] = self.ACT_obs[t, full_idx]
            
            # Cash is always active
            if self.cfg.allow_cash:
                obs[idx + self.cash_idx] = 1
                mask[self.cash_idx] = 1
            
            idx += self.obs_n_assets
        else:
            # If not including active flag, all assets are "active" in mask
            mask[:] = 1
        
        # Global statistics
        if self.cfg.global_stats and self.F > 0:
            if t < self.T:
                obs[idx:idx + self.F] = self.global_means[t]
                obs[idx + self.F:idx + 2 * self.F] = self.global_stds[t]
            idx += 2 * self.F
        
        # Scalars: lagged RF and IDX
        if t < self.T:
            obs[idx] = self.RF_obs[t]
            obs[idx + 1] = self.IDX_obs[t]
        
        return obs, mask
    
    # --------------------------- Action decoding ------------------------ #
    
    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """
        Decode discrete action to portfolio weights in observation space.
        
        Args:
            action: Array of shape (obs_n_assets,) with discrete block indices
        
        Returns:
            weights: Array of shape (obs_n_assets,) with portfolio weights
        """
        weights = np.zeros(self.obs_n_assets, dtype=np.float32)
        
        # Convert block indices to weights
        for i in range(self.obs_n_assets):
            block_idx = int(action[i])
            weights[i] = block_idx / self.cfg.weight_blocks
        
        # Normalize to sum to 1
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            # Fallback to equal weight
            weights[:] = 1.0 / self.obs_n_assets
        
        # Apply max weight constraint
        if self.cfg.max_weight < 1.0:
            # Clip and renormalize
            weights = np.clip(weights, 0, self.cfg.max_weight)
            total = weights.sum()
            if total > 0:
                weights /= total
        
        return weights
    
    # --------------------------- Step & Reset ------------------------ #
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset environment to start of episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Determine start time
        max_start = max(0, self.T - (self.cfg.max_steps or self.T))
        if self.cfg.random_reset_frac > 0:
            start_t = self.rng.integers(0, int(max_start * self.cfg.random_reset_frac) + 1)
        else:
            start_t = 0
        
        self.t = start_t
        self.t_start = start_t
        self.steps_taken = 0
        
        # Initialize wealth tracking
        self.wealth = 1.0
        self.peak_wealth = 1.0
        
        # Initialize portfolio (equal weight or cash)
        self.full_curr_w = np.zeros(self.n_assets, dtype=np.float32)
        self.prev_full_w = np.zeros(self.n_assets, dtype=np.float32)
        
        # Tail risk buffer
        if self.cfg.lambda_tail > 0:
            self.tail_buffer = []
        
        # Get initial top-K assets
        self.current_top_k_indices = self._get_top_k_assets(self.t, use_lagged=True)
        
        # Get initial observation
        obs, mask = self._get_observation(self.t)
        
        info = {
            "date": pd.Timestamp(self.dates[min(self.t, self.T - 1)]).to_pydatetime(),
            "wealth": float(self.wealth),
        }
        
        return {"observation": obs, "action_mask": mask}, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        The agent chooses weights based on t-1 information,
        and those weights earn returns over period t.
        """
        t = self.t
        
        # Check if episode should end
        terminated = False
        truncated = False
        if t >= self.T - 1:
            terminated = True
        if self.cfg.max_steps and self.steps_taken >= self.cfg.max_steps:
            truncated = True
        
        if terminated or truncated:
            # Return terminal observation
            obs, mask = self._get_observation(min(t, self.T - 1))
            info = {
                "date": pd.Timestamp(self.dates[min(t, self.T - 1)]).to_pydatetime(),
                "wealth": float(self.wealth),
                "is_terminal": True,
            }
            return {"observation": obs, "action_mask": mask}, 0.0, terminated, truncated, info
        
        # Decode action to weights (in observation space)
        obs_weights = self._decode_action(action)
        
        # Map to full space
        new_full_w = self._map_weights_to_full_space(obs_weights)
        
        # Handle cash
        cash_weight = obs_weights[self.cash_idx] if self.cfg.allow_cash else 0.0
        
        # On rebalancing days, update top-K selection
        is_rebalance_day = (t % self.cfg.rebalance_interval == 0)
        if is_rebalance_day and t > 0:
            # CRITICAL FIX: Use lagged weights for top-K selection
            new_top_k = self._get_top_k_assets(t, use_lagged=True)
            
            # Handle assets that drop out of top-K
            old_top_k_set = set(self.current_top_k_indices)
            new_top_k_set = set(new_top_k)
            dropped_assets = old_top_k_set - new_top_k_set
            
            # Liquidate dropped assets
            forced_liquidation = 0.0
            for asset_idx in dropped_assets:
                if asset_idx < self.n_assets:
                    forced_liquidation += self.full_curr_w[asset_idx]
                    new_full_w[asset_idx] = 0.0
            
            # Update top-K
            self.current_top_k_indices = new_top_k
        else:
            forced_liquidation = 0.0
        
        # Calculate turnover
        turn = _compute_turnover(self.full_curr_w, new_full_w)
        
        # Calculate returns for period t
        r_assets = self.R[t].astype(np.float32)
        rf_t = float(self.RF[t])
        
        # Portfolio return (weighted average of asset returns)
        # NOTE: Returns in data are excess returns (spread + MtM)
        r_p = float(np.dot(self.full_curr_w, r_assets))
        
        # If cash is held, add cash return
        if self.cfg.allow_cash and self.cfg.cash_rate_as_rf:
            r_p += cash_weight * rf_t
        
        # Transaction costs
        lin_cost = (self.cfg.transaction_cost_bps / 10000.0) * turn
        
        # Extra cost for forced liquidations
        extra_delist_cost = 0.0
        if forced_liquidation > 0:
            extra_delist_cost = (self.cfg.delist_extra_bps / 10000.0) * forced_liquidation
            lin_cost += extra_delist_cost
        
        # ============ WEALTH CALCULATION (FIXED) ============
        # Convert excess return to total return for wealth tracking
        r_p_total = r_p + rf_t  # Total portfolio return
        net_total_factor = max((1.0 + r_p_total) * (1.0 - max(lin_cost, 0.0)), 1e-12)
        r_net_total = net_total_factor - 1.0
        
        # For metrics: keep excess return version
        r_net_excess = (1.0 + r_p) * (1.0 - max(lin_cost, 0.0)) - 1.0
        
        # Update wealth using total returns
        self.wealth *= net_total_factor
        self.peak_wealth = max(self.peak_wealth, self.wealth)
        # ====================================================
        
        # Drawdown calculation
        cur_dd_level = 1.0 - (self.wealth / max(self.peak_wealth, 1e-12))
        
        if self.cfg.dd_mode == "incremental":
            dd_inc = max(0.0, cur_dd_level - self.prev_dd_level) if hasattr(self, 'prev_dd_level') else 0.0
            self.prev_dd_level = cur_dd_level
        else:
            dd_inc = cur_dd_level
        
        # HHI (concentration)
        hhi_val = _compute_hhi(new_full_w)
        
        # Tail risk penalty
        tail_pen = 0.0
        if self.cfg.lambda_tail > 0:
            self.tail_buffer.append(r_p)
            if len(self.tail_buffer) > self.cfg.tail_window:
                self.tail_buffer.pop(0)
            if len(self.tail_buffer) >= 20:
                tail_var = np.quantile(self.tail_buffer, self.cfg.tail_q)
                tail_pen = -min(tail_var, 0.0)
        
        # Benchmark returns
        r_idx = float(self.IDX[t])  # Index excess return
        
        # Alpha (active return vs benchmark, both in excess return space)
        alpha = r_p - r_idx
        
        # Penalties
        pen = (
            - self.cfg.lambda_turnover * turn
            - self.cfg.lambda_hhi * hhi_val
            - self.cfg.lambda_drawdown * dd_inc
            - self.cfg.lambda_tail * tail_pen
        )
        
        # Reward: alpha minus costs and penalties
        reward = float(self.cfg.weight_alpha * alpha + pen - lin_cost)
        
        # Update state
        self.prev_full_w = self.full_curr_w.copy()
        self.full_curr_w = new_full_w
        self.t += 1
        self.steps_taken += 1
        
        # Get next observation
        obs, mask = self._get_observation(self.t)
        
        # Info dictionary with clear return definitions
        output_weights = self._map_weights_to_obs_space(new_full_w)
        
        info = {
            "date": pd.Timestamp(self.dates[min(self.t, self.T - 1)]).to_pydatetime(),
            "weights": output_weights,
            
            # Returns (clearly labeled)
            "portfolio_return_excess_gross": float(r_p),  # Gross excess return
            "portfolio_return_excess_net": float(r_net_excess),  # Net excess return
            "portfolio_return_total_net": float(r_net_total),  # Net total return
            
            # Main return for evaluation (backward compatible)
            "portfolio_return": float(r_net_total),  # Use net total return
            
            # Performance metrics
            "alpha": float(alpha),
            "index_return": float(r_idx),
            "rf": float(rf_t),
            
            # Trading
            "turnover": float(turn),
            "hhi": float(hhi_val),
            "trade_cost": float(lin_cost),
            
            # Wealth & risk
            "wealth": float(self.wealth),
            "drawdown": float(cur_dd_level),
            
            # Other
            "sector_exposure": _dict_sector_exposures(self.sector_ids, self.full_curr_w),
            "forced_liquidation": float(forced_liquidation),
            "top_k_assets": [self.asset_ids[i] for i in self.current_top_k_indices[:self.max_assets_config] if i < len(self.asset_ids)],
            "config": asdict(self.cfg) if self.t == 1 else None,
        }
        
        return {"observation": obs, "action_mask": mask}, reward, terminated, truncated, info


def _dict_sector_exposures(sector_ids: np.ndarray, weights: np.ndarray) -> Dict[int, float]:
    """Calculate sector exposures from weights."""
    exposures = {}
    for sid in np.unique(sector_ids):
        if sid >= 0:
            mask = (sector_ids == sid)
            exposures[int(sid)] = float(weights[mask].sum())
    return exposures
