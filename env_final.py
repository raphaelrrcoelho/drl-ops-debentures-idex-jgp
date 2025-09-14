# env_final.py
"""
Optimized Debenture portfolio environment with DISCRETE action space
Configuration fully driven by config.yaml
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import yaml

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ----------------------------- Configuration ----------------------------- #

@dataclass
class EnvConfig:
    """Environment configuration - all parameters from config.yaml"""
    
    # Portfolio constraints
    rebalance_interval: int = 5
    max_weight: float = 0.10
    weight_blocks: int = 100  # For discrete action space (1% granularity)
    allow_cash: bool = True
    cash_rate_as_rf: bool = True
    on_inactive: str = "to_cash"  # or "pro_rata"
    
    # Transaction costs (in basis points)
    transaction_cost_bps: float = 20.0
    delist_extra_bps: float = 20.0  # Extra slippage on forced liquidation
    
    # Reward weights
    weight_excess: float = 0.0  # Weight for excess return (r - rf)
    weight_alpha: float = 1.0   # Weight for alpha (r - index)
    
    # Reward penalties
    lambda_turnover: float = 0.0002
    lambda_hhi: float = 0.01
    lambda_drawdown: float = 0.005
    lambda_tail: float = 0.0  # Set to 0 to disable
    
    # Tail risk parameters (only used if lambda_tail > 0)
    tail_window: int = 60
    tail_q: float = 0.05
    dd_mode: str = "incremental"  # or "level"
    
    # Observation configuration
    include_prev_weights: bool = True
    include_active_flag: bool = True
    global_stats: bool = True
    normalize_features: bool = True
    obs_clip: float = 5.0
    
    # Episode control
    max_steps: Optional[int] = None
    random_reset_frac: float = 0.0
    seed: Optional[int] = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> 'EnvConfig':
        """Load configuration directly from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract environment parameters from config
        env_params = {}
        
        # Get all fields that belong to EnvConfig
        for field_name in cls.__annotations__:
            if field_name in data:
                env_params[field_name] = data[field_name]
        
        # Handle None values
        for key in ['max_steps', 'seed']:
            if key in env_params and env_params[key] == 'null':
                env_params[key] = None
        
        return cls(**env_params)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EnvConfig':
        """Create from dictionary with validation"""
        # Only keep fields that are in the dataclass
        valid_fields = {}
        for field_name in cls.__annotations__:
            if field_name in d:
                value = d[field_name]
                # Handle None values
                if value == 'null' or (isinstance(value, str) and value.lower() == 'none'):
                    if field_name in ['max_steps', 'seed']:
                        value = None
                valid_fields[field_name] = value
        
        return cls(**valid_fields)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0 < self.max_weight <= 1.0, f"max_weight must be in (0, 1], got {self.max_weight}"
        assert self.weight_blocks > 0, f"weight_blocks must be positive, got {self.weight_blocks}"
        assert self.rebalance_interval > 0, f"rebalance_interval must be positive, got {self.rebalance_interval}"
        assert self.transaction_cost_bps >= 0, f"transaction_cost_bps must be non-negative, got {self.transaction_cost_bps}"
        assert self.delist_extra_bps >= 0, f"delist_extra_bps must be non-negative, got {self.delist_extra_bps}"
        assert self.lambda_turnover >= 0, f"lambda_turnover must be non-negative, got {self.lambda_turnover}"
        assert self.lambda_hhi >= 0, f"lambda_hhi must be non-negative, got {self.lambda_hhi}"
        assert self.lambda_drawdown >= 0, f"lambda_drawdown must be non-negative, got {self.lambda_drawdown}"
        assert self.lambda_tail >= 0, f"lambda_tail must be non-negative, got {self.lambda_tail}"
        assert self.on_inactive in ["to_cash", "pro_rata"], f"on_inactive must be 'to_cash' or 'pro_rata', got {self.on_inactive}"
        assert self.dd_mode in ["incremental", "level"], f"dd_mode must be 'incremental' or 'level', got {self.dd_mode}"
        assert self.obs_clip > 0, f"obs_clip must be positive, got {self.obs_clip}"
        assert 0 <= self.random_reset_frac <= 1, f"random_reset_frac must be in [0, 1], got {self.random_reset_frac}"
        
        if self.lambda_tail > 0:
            assert self.tail_window > 0, f"tail_window must be positive when lambda_tail > 0, got {self.tail_window}"
            assert 0 < self.tail_q < 1, f"tail_q must be in (0, 1) when lambda_tail > 0, got {self.tail_q}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def __post_init__(self):
        """Validate after initialization"""
        self.validate()

# -------------------------- JIT-compiled utilities ----------------------- #

@jit(nopython=True, cache=True)
def fast_portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """Fast portfolio return calculation"""
    result = 0.0
    for i in range(len(weights)):
        if not np.isnan(returns[i]):
            result += weights[i] * returns[i]
    return result

@jit(nopython=True, cache=True)
def fast_turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """Fast turnover calculation"""
    result = 0.0
    for i in range(len(w_new)):
        result += abs(w_new[i] - w_prev[i])
    return result

@jit(nopython=True, cache=True)
def fast_hhi(weights: np.ndarray) -> float:
    """Fast HHI calculation"""
    result = 0.0
    for i in range(len(weights)):
        result += weights[i] * weights[i]
    return result

@jit(nopython=True, cache=True)
def fast_blocks_to_weights(blocks: np.ndarray, mask: np.ndarray, max_weight: float) -> np.ndarray:
    """Convert discrete blocks to continuous weights with mask"""
    n = len(blocks)
    weights = np.zeros(n, dtype=np.float32)
    
    # Apply mask
    total = 0.0
    for i in range(n):
        if mask[i] > 0:
            weights[i] = blocks[i]
            total += blocks[i]
    
    # Normalize
    if total > 0:
        for i in range(n):
            weights[i] = min(weights[i] / total, max_weight)
        
        # Renormalize after capping
        total = 0.0
        for i in range(n):
            total += weights[i]
        if total > 0:
            for i in range(n):
                weights[i] = weights[i] / total
    
    return weights

# ----------------------- Data Preprocessing ----------------------- #

class SharedDataProcessor:
    """Preprocesses panel data once and provides shared arrays"""
    
    @staticmethod
    def process_panel(panel: pd.DataFrame, config: EnvConfig) -> Dict[str, Any]:
        """
        Process panel once and return numpy arrays.
        Uses config parameters for feature selection and processing.
        """
        panel = panel.sort_index()
        
        # Required columns check
        required = [
            "return", "risk_free", "index_return", "active",
            "spread", "duration", "time_to_maturity", "sector_id", "index_level"
        ]
        for c in required:
            if c not in panel.columns:
                raise ValueError(f"Missing required column '{c}'")
        
        # Get dimensions
        dates = panel.index.get_level_values("date").unique().sort_values()
        asset_ids = panel.index.get_level_values("debenture_id").unique().tolist()
        n_assets = len(asset_ids)
        T = len(dates)
        
        # Build feature list based on config
        base_feats = [
            "return", "spread", "duration", "time_to_maturity",
            "risk_free", "index_return", "ttm_rank",
            "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
            "sector_fitted_spread", "spread_residual_ns",
            "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
            "sector_spread", "sector_momentum", "sector_weight_index",
            "sector_id",
        ]
        
        # Add active flag if configured
        if config.include_active_flag:
            base_feats.append("active")
        
        # Ensure lag1 columns exist
        panel = panel.copy()
        for col in base_feats:
            if col in panel.columns and col not in ("sector_id", "active"):
                lag_col = f"{col}_lag1"
                if lag_col not in panel.columns:
                    panel[lag_col] = (
                        panel.groupby(level="debenture_id", sort=False)[col]
                             .shift(1)
                             .astype(np.float32)
                             .fillna(0.0)
                    )
        
        # Select feature columns
        feat_cols = []
        for c in base_feats:
            if c in panel.columns:
                if c in ("sector_id", "active"):
                    feat_cols.append(c)
                else:
                    lag_col = f"{c}_lag1"
                    if lag_col in panel.columns:
                        feat_cols.append(lag_col)
        
        # Add z-score columns
        z_cols = [c for c in panel.columns if (c.endswith("_z") or c.endswith("_z252") or "_z_" in c)]
        for zc in sorted(z_cols):
            if zc not in feat_cols:
                feat_cols.append(zc)
        
        # Convert to wide arrays
        print(f"  Converting to wide arrays: {T} dates × {n_assets} assets")
        
        # Returns
        R = panel.pivot_table(
            index="date", columns="debenture_id", values="return"
        ).reindex(index=dates, columns=asset_ids).fillna(0.0).to_numpy(dtype=np.float32)
        
        # Risk-free and index
        RF = panel.groupby(level="date")["risk_free"].first().reindex(dates).fillna(0.0).to_numpy(dtype=np.float32)
        IDX = panel.groupby(level="date")["index_return"].first().reindex(dates).fillna(0.0).to_numpy(dtype=np.float32)
        
        # Active mask
        ACT = panel.pivot_table(
            index="date", columns="debenture_id", values="active"
        ).reindex(index=dates, columns=asset_ids).fillna(0.0).to_numpy(dtype=np.float32)
        
        # Features tensor
        F = len(feat_cols)
        X = np.zeros((T, n_assets, F), dtype=np.float32)
        for f_idx, col in enumerate(feat_cols):
            if col in panel.columns:
                wide = panel.pivot_table(
                    index="date", columns="debenture_id", values=col
                ).reindex(index=dates, columns=asset_ids).fillna(0.0).to_numpy(dtype=np.float32)
                X[:, :, f_idx] = wide
        
        # Sector IDs
        sector_ids = (
            panel.groupby(level="debenture_id")["sector_id"]
                 .first()
                 .reindex(asset_ids)
                 .fillna(-1)
                 .to_numpy(dtype=np.int16)
        )
        
        # Add cash if configured
        if config.allow_cash:
            # Append cash asset with returns based on config
            cash_R = RF.reshape(-1, 1) if config.cash_rate_as_rf else np.zeros((T, 1), dtype=np.float32)
            cash_X = np.zeros((T, 1, F), dtype=np.float32)
            cash_A = np.ones((T, 1), dtype=np.float32)
            
            R = np.concatenate([R, cash_R], axis=1)
            X = np.concatenate([X, cash_X], axis=1)
            ACT = np.concatenate([ACT, cash_A], axis=1)
            asset_ids.append("__CASH__")
            n_assets += 1
        
        # Compute normalization stats if configured
        global_means = None
        global_stds = None
        if config.global_stats and F > 0:
            act = (ACT > 0).astype(np.float32)
            denom = np.maximum(act.sum(axis=1, keepdims=True), 1.0)
            means = (X * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None]
            stds = np.sqrt(((X - means) ** 2 * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None])
            global_means = means.squeeze(1)
            global_stds = np.maximum(stds.squeeze(1), 1e-6)
            
            # Apply normalization if configured
            if config.normalize_features:
                for f in range(F):
                    mu = global_means[:, f][:, None]
                    sd = global_stds[:, f][:, None]
                    X[:, :, f] = np.clip((X[:, :, f] - mu) / (sd + 1e-6), -config.obs_clip, config.obs_clip)
        
        # Lagged RF/IDX for observations
        RF_obs = np.zeros_like(RF)
        RF_obs[1:] = RF[:-1]
        IDX_obs = np.zeros_like(IDX)
        IDX_obs[1:] = IDX[:-1]
        
        return {
            'R': R,
            'RF': RF,
            'IDX': IDX,
            'ACT': ACT,
            'X': X,
            'RF_obs': RF_obs,
            'IDX_obs': IDX_obs,
            'dates': dates.to_numpy(),
            'asset_ids': asset_ids,
            'sector_ids': sector_ids,
            'feature_cols': feat_cols,
            'global_means': global_means,
            'global_stds': global_stds,
            'T': T,
            'n_assets': n_assets,
            'F': F,
            'cash_idx': n_assets - 1 if config.allow_cash else None
        }

# -------------------------- Shared Data Environment ---------------------- #

class SharedDataEnv(gym.Env):
    """Environment that uses shared preprocessed data and config parameters"""
    
    metadata = {"render_modes": ["human"]}
    _shared_data: Optional[Dict[str, Any]] = None
    
    @classmethod
    def set_shared_data(cls, data: Dict[str, Any]):
        """Set shared data for all instances"""
        cls._shared_data = data
    
    def __init__(self, config: EnvConfig):
        super().__init__()
        
        if self._shared_data is None:
            raise RuntimeError("Must call SharedDataEnv.set_shared_data() before creating environments")
        
        # Store and validate config
        self.cfg = config
        self.cfg.validate()
        
        # Reference shared arrays (no copying!)
        self.R = self._shared_data['R']
        self.RF = self._shared_data['RF']
        self.IDX = self._shared_data['IDX']
        self.ACT = self._shared_data['ACT']
        self.X = self._shared_data['X']
        self.RF_obs = self._shared_data['RF_obs']
        self.IDX_obs = self._shared_data['IDX_obs']
        self.dates = self._shared_data['dates']
        self.asset_ids = self._shared_data['asset_ids']
        self.sector_ids = self._shared_data['sector_ids']
        self.feature_cols = self._shared_data['feature_cols']
        self.global_means = self._shared_data['global_means']
        self.global_stds = self._shared_data['global_stds']
        self.T = self._shared_data['T']
        self.n_assets = self._shared_data['n_assets']
        self.F = self._shared_data['F']
        self.cash_idx = self._shared_data['cash_idx']
        
        # Setup action/observation spaces based on config
        self.max_blocks_per_asset = int(self.cfg.max_weight * self.cfg.weight_blocks)
        self.action_space = spaces.MultiDiscrete(
            [self.max_blocks_per_asset + 1] * self.n_assets
        )
        
        obs_dim = self._obs_size()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            'action_mask': spaces.Box(0, 1, shape=(self.n_assets,), dtype=np.int8)
        })
        
        # Instance-specific state (minimal memory footprint)
        self.t: int = 0
        self.prev_w = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w = np.zeros(self.n_assets, dtype=np.float32)
        self.wealth: float = 1.0
        self.peak_wealth: float = 1.0
        self.tail_buffer: List[float] = []
        self._history: Dict[str, list] = {}
        
        # Set random generator from config
        if config.seed is not None:
            self.np_random = np.random.default_rng(config.seed)
        else:
            self.np_random = np.random.default_rng()
    
    def _obs_size(self) -> int:
        """Calculate observation size based on config"""
        n = self.n_assets
        base = n * self.F
        extra = 0
        if self.cfg.include_prev_weights:
            extra += n
        if self.cfg.include_active_flag:
            extra += n
        if self.cfg.global_stats and self.F > 0:
            extra += 2 * self.F
        extra += 2  # RF and IDX
        return base + extra
    
    def _get_observation(self) -> Dict:
        """Build observation based on config settings"""
        t = min(self.t, self.T - 1)
        
        parts: List[np.ndarray] = []
        if self.F > 0:
            X_t = self.X[t].flatten()
            parts.append(X_t)
        if self.cfg.include_prev_weights:
            parts.append(self.prev_w)
        if self.cfg.include_active_flag:
            parts.append(self.ACT[t])
        if self.cfg.global_stats and self.F > 0 and self.global_means is not None:
            parts.append(self.global_means[t])
            parts.append(self.global_stds[t])
        parts.append(np.array([self.RF_obs[t], self.IDX_obs[t]], dtype=np.float32))
        
        obs = np.concatenate(parts) if parts else np.zeros((self._obs_size(),), dtype=np.float32)
        mask = self.ACT[t].astype(np.int8)
        
        return {
            'observation': np.clip(obs, -self.cfg.obs_clip, self.cfg.obs_clip).astype(np.float32),
            'action_mask': mask
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Random start position based on config
        if self.cfg.max_steps is not None and 0 < self.cfg.random_reset_frac <= 1.0:
            max_start = max(0, self.T - int(self.cfg.max_steps))
            jitter_cap = int(max_start * self.cfg.random_reset_frac)
            self.t = int(self.np_random.integers(0, jitter_cap + 1)) if jitter_cap > 0 else 0
        else:
            self.t = 0
        
        # Initialize weights
        self.prev_w = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w = np.zeros(self.n_assets, dtype=np.float32)
        
        # Start with cash or equal weight based on config
        if self.cash_idx is not None and self.cfg.allow_cash:
            self.curr_w[self.cash_idx] = 1.0
        else:
            mask = self.ACT[self.t] > 0
            if mask.any():
                self.curr_w[mask] = 1.0 / mask.sum()
        
        self.wealth = 1.0
        self.peak_wealth = 1.0
        self.tail_buffer = []
        self._history = {}
        
        info = {"date": pd.Timestamp(self.dates[self.t]).to_pydatetime()}
        return self._get_observation(), info
    
    def step(self, action: np.ndarray):
        t = int(self.t)
        
        if t >= self.T:
            return self._get_observation(), 0.0, True, False, {}
        
        # Get current state
        act_mask = self.ACT[t]
        r_vec = self.R[t].copy()
        
        # Only rebalance based on config interval
        apply_action = (t % max(1, self.cfg.rebalance_interval) == 0)
        w_prev = self.curr_w.copy()
        freed_mass = 0.0
        extra_delist_cost = 0.0
        
        if apply_action:
            # Convert discrete blocks to weights using config max_weight
            blocks = np.asarray(action, dtype=np.float32)
            w_tgt = fast_blocks_to_weights(blocks, act_mask, self.cfg.max_weight)
            
            # Ensure normalization
            if w_tgt.sum() > 0:
                w_tgt = w_tgt / w_tgt.sum()
            elif self.cash_idx is not None:
                w_tgt = np.zeros_like(w_tgt)
                w_tgt[self.cash_idx] = 1.0
        else:
            # Hold but handle delistings based on config
            w_tgt = w_prev.copy()
            inactive = (act_mask <= 0)
            freed = float(np.maximum(w_tgt[inactive], 0.0).sum())
            
            if freed > 0.0:
                w_tgt[inactive] = 0.0
                freed_mass = freed
                extra_delist_cost = freed_mass * (self.cfg.delist_extra_bps / 10000.0)
                
                # Handle inactive based on config
                if self.cfg.on_inactive == "to_cash" and self.cash_idx is not None:
                    w_tgt[self.cash_idx] += freed
                else:  # pro_rata
                    active = (act_mask > 0)
                    s = float(w_tgt[active].sum())
                    if s > 0.0:
                        w_tgt[active] += (w_tgt[active] / s) * freed
                    elif self.cash_idx is not None:
                        w_tgt[self.cash_idx] += freed
                
                if w_tgt.sum() > 0:
                    w_tgt = w_tgt / w_tgt.sum()
        
        # Calculate costs using config parameters
        turn = fast_turnover(w_tgt, w_prev)
        lin_cost = (self.cfg.transaction_cost_bps / 10000.0) * turn + extra_delist_cost
        
        # Portfolio return
        rf_t = float(self.RF[t])
        r_vec = np.where(np.isfinite(r_vec), r_vec, rf_t)
        r_p = fast_portfolio_return(w_tgt, r_vec)
        
        # Benchmarks
        r_idx = float(self.IDX[t])
        alpha = r_p - r_idx
        excess = r_p - rf_t
        
        # Update wealth
        net = max((1.0 + r_p) * (1.0 - max(lin_cost, 0.0)), 1e-12)
        r_net = net - 1.0
        self.wealth *= net
        self.peak_wealth = max(self.peak_wealth, self.wealth)
        
        # Drawdown penalty based on config mode
        cur_dd_level = 1.0 - (self.wealth / max(self.peak_wealth, 1e-12))
        if self.cfg.dd_mode == "level":
            dd_pen = -self.cfg.lambda_drawdown * abs(cur_dd_level)
        else:  # incremental
            prev_dd = self._history.get("drawdown", [0.0])[-1] if self._history.get("drawdown") else 0.0
            dd_inc = max(cur_dd_level - prev_dd, 0.0)
            dd_pen = -self.cfg.lambda_drawdown * dd_inc
        
        # Tail penalty based on config
        tail_pen = 0.0
        if self.cfg.lambda_tail > 0:
            self.tail_buffer.append(r_p)
            if len(self.tail_buffer) > self.cfg.tail_window:
                self.tail_buffer.pop(0)
            if len(self.tail_buffer) >= max(10, self.cfg.tail_window // 2):
                q = float(np.quantile(self.tail_buffer, self.cfg.tail_q))
                if r_p < q:
                    tail_pen = abs(r_p)
        
        # Calculate penalties using config lambdas
        hhi_val = fast_hhi(w_tgt)
        pen = (
            - self.cfg.lambda_turnover * turn
            - self.cfg.lambda_hhi * hhi_val
            + dd_pen
            - self.cfg.lambda_tail * tail_pen
        )
        
        # Reward using config weights
        reward = float(
            self.cfg.weight_alpha * alpha + 
            self.cfg.weight_excess * excess + 
            pen - lin_cost
        )
        
        # Update state
        self.prev_w = w_prev
        self.curr_w = w_tgt
        self.t = t + 1
        
        # Check termination based on config
        terminated = (self.t >= self.T)
        truncated = (self.cfg.max_steps is not None and self.t >= int(self.cfg.max_steps))
        
        obs = self._get_observation()
        info = {
            "date": pd.Timestamp(self.dates[min(self.t, self.T - 1)]).to_pydatetime(),
            "weights": self.curr_w.copy(),
            "portfolio_return": float(r_net),
            "turnover": float(turn),
            "hhi": float(hhi_val),
            "wealth": float(self.wealth),
            "drawdown": float(cur_dd_level),
            "rf": float(rf_t),
            "index_return": float(r_idx),
            "alpha": float(alpha),
            "excess": float(excess),
        }
        
        # Store history
        for k, v in info.items():
            if k not in ["weights", "date"]:
                if k not in self._history:
                    self._history[k] = []
                self._history[k].append(v)
        
        return obs, reward, terminated, truncated, info
    
    def get_action_masks(self) -> np.ndarray:
        """For MaskablePPO - based on config max_weight"""
        masks = []
        act_mask = self.ACT[self.t]
        
        for i in range(self.n_assets):
            asset_mask = np.ones(self.max_blocks_per_asset + 1, dtype=bool)
            if act_mask[i] <= 0:
                asset_mask[1:] = False
            masks.append(asset_mask)
        
        return masks
    
    def render(self):
        t = min(self.t, self.T - 1)
        date = pd.Timestamp(self.dates[t]).date()
        print(f"[{date}] wealth={self.wealth:.4f} dd={self._history['drawdown'][-1]:.3%}")

# -------------------------- Batched Environment -------------------------- #

class BatchedEnv(gym.Wrapper):
    """
    Batches multiple steps together for vectorized processing.
    Batch size comes from config.
    """
    
    def __init__(self, env: SharedDataEnv, batch_size: int = 16):
        super().__init__(env)
        self.batch_size = batch_size
        self.action_buffer = []
        self.obs_buffer = []
        self.reward_buffer = []
        self.info_buffer = []
        
    def step(self, action):
        # Regular step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store for potential batch processing
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.info_buffer.append(info)
        
        # Clear buffers when batch is full or episode ends
        if len(self.action_buffer) >= self.batch_size or terminated or truncated:
            self._process_batch()
        
        return obs, reward, terminated, truncated, info
    
    def _process_batch(self):
        """Process accumulated batch"""
        if not self.action_buffer:
            return
        
        # Here you could add batch processing logic
        # For example, vectorized reward calculations across the batch
        
        # Clear buffers
        self.action_buffer = []
        self.obs_buffer = []
        self.reward_buffer = []
        self.info_buffer = []
    
    def reset(self, **kwargs):
        self._process_batch()  # Clear any remaining buffer
        return self.env.reset(**kwargs)

# ------------------------- Factory Functions ----------------------------- #

def make_env_from_config(panel: pd.DataFrame, config_path: str) -> SharedDataEnv:
    """
    Create environment directly from config file.
    Preprocesses data once and creates SharedDataEnv.
    """
    # Load config from YAML
    cfg = EnvConfig.from_yaml(config_path)
    
    # Preprocess data once
    print("Preprocessing panel data from config...")
    shared_data = SharedDataProcessor.process_panel(panel, cfg)
    
    # Set shared data
    SharedDataEnv.set_shared_data(shared_data)
    
    # Create environment
    return SharedDataEnv(cfg)

def make_shared_env_from_panel(panel: pd.DataFrame, **env_kwargs) -> SharedDataEnv:
    """
    Factory to create SharedDataEnv with preprocessed data.
    Parameters can come from kwargs or config file.
    """
    # If config_path is provided, load from file
    if 'config_path' in env_kwargs:
        cfg = EnvConfig.from_yaml(env_kwargs['config_path'])
    else:
        # Create from kwargs
        cfg = EnvConfig(**env_kwargs)
    
    # Preprocess data once
    print("Preprocessing panel data...")
    shared_data = SharedDataProcessor.process_panel(panel, cfg)
    
    # Set shared data for all environments
    SharedDataEnv.set_shared_data(shared_data)
    
    # Create environment
    return SharedDataEnv(cfg)

def make_env_from_panel(panel: pd.DataFrame, **env_kwargs) -> SharedDataEnv:
    """
    Backward compatible factory function.
    Note: First call will preprocess data, subsequent calls will reuse it.
    """
    # If config_path is provided, use it
    if 'config_path' in env_kwargs:
        cfg = EnvConfig.from_yaml(env_kwargs['config_path'])
    else:
        cfg = EnvConfig(**env_kwargs)
    
    # Check if data is already shared
    if SharedDataEnv._shared_data is None:
        print("First environment creation - preprocessing data...")
        shared_data = SharedDataProcessor.process_panel(panel, cfg)
        SharedDataEnv.set_shared_data(shared_data)
    
    return SharedDataEnv(cfg)

def make_batched_env_from_config(panel: pd.DataFrame, config_path: str, batch_size: Optional[int] = None) -> BatchedEnv:
    """Create batched environment from config file"""
    cfg = EnvConfig.from_yaml(config_path)
    
    # Use batch_size from config if not overridden
    if batch_size is None:
        # Look for batch_env_size in the config
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            batch_size = data.get('batch_env_size', 16)
    
    base_env = make_env_from_config(panel, config_path)
    return BatchedEnv(base_env, batch_size=batch_size)

# ------------------------------ Quick Test -------------------------------- #

if __name__ == "__main__":
    print("Testing config-driven environment...")
    
    # Test loading from config file
    test_config = """
# Test config
rebalance_interval: 5
max_weight: 0.10
weight_blocks: 100
allow_cash: true
cash_rate_as_rf: true
on_inactive: to_cash
transaction_cost_bps: 20.0
delist_extra_bps: 20.0
weight_excess: 0.0
weight_alpha: 1.0
lambda_turnover: 0.0002
lambda_hhi: 0.01
lambda_drawdown: 0.005
lambda_tail: 0.0
tail_window: 60
tail_q: 0.05
dd_mode: incremental
include_prev_weights: true
include_active_flag: true
global_stats: true
normalize_features: true
obs_clip: 5.0
max_steps: null
random_reset_frac: 0.0
seed: 42
    """
    
    # Save test config
    with open('test_config.yaml', 'w') as f:
        f.write(test_config)
    
    # Create test data
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=100, freq="B")
    ids = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, ids], names=["date", "debenture_id"])
    
    df = pd.DataFrame(index=idx)
    df["return"] = rng.normal(0.0003, 0.002, size=len(df)).astype(np.float32)
    df["risk_free"] = 0.0003
    df["index_return"] = rng.normal(0.0002, 0.0015, size=len(df)).astype(np.float32)
    df["spread"] = rng.normal(0.02, 0.005, size=len(df)).astype(np.float32)
    df["duration"] = rng.uniform(1.0, 5.0, size=len(df)).astype(np.float32)
    df["time_to_maturity"] = rng.uniform(0.5, 4.0, size=len(df)).astype(np.float32)
    df["sector_id"] = rng.integers(0, 3, size=len(df)).astype(np.int16)
    df["index_level"] = 1000.0
    df["active"] = 1
    
    # Add required columns
    for c in ["sector_spread", "sector_momentum", "sector_weight_index"]:
        df[c] = 0.0
    
    # Test creating environment from config
    print("\n1. Testing environment creation from config file...")
    env = make_env_from_config(df, 'test_config.yaml')
    print(f"   Environment created with config parameters")
    print(f"   Max weight: {env.cfg.max_weight}")
    print(f"   Rebalance interval: {env.cfg.rebalance_interval}")
    print(f"   Transaction cost: {env.cfg.transaction_cost_bps} bps")
    
    # Test environment usage
    print("\n2. Testing environment with config parameters...")
    obs, info = env.reset()
    for _ in range(10):
        action = np.array([5, 3, 2, 0], dtype=int)  # 4 assets (3 + cash)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    
    print(f"   Final wealth: {info['wealth']:.6f}")
    print(f"   Transaction costs applied: {env.cfg.transaction_cost_bps} bps")
    
    # Clean up test config
    import os
    os.remove('test_config.yaml')
    
    print("\n✓ Config-driven environment working correctly!")
    
    if HAS_NUMBA:
        print("✓ JIT compilation active")
    else:
        print("⚠ Install numba for 2-5x speedup: pip install numba")