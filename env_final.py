# env_final_optimized.py
"""
OPTIMIZED Debenture portfolio environment
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from numba import jit, prange

# ----------------------------- JIT-compiled utilities ----------------------------- #

@jit(nopython=True, fastmath=True, cache=True)
def blocks_to_weights_jit(blocks: np.ndarray) -> np.ndarray:
    """Convert discrete blocks to weights - JIT compiled"""
    total = blocks.sum()
    if total <= 0:
        return np.zeros(blocks.shape[0], dtype=np.float32)
    return (blocks / total).astype(np.float32)

@jit(nopython=True, fastmath=True, cache=True)
def calculate_portfolio_return(weights: np.ndarray, returns: np.ndarray, 
                               rf: float) -> float:
    """Calculate portfolio return - JIT compiled"""
    for i in range(len(returns)):
        if not np.isfinite(returns[i]):
            returns[i] = rf
    return np.dot(weights, returns)

@jit(nopython=True, fastmath=True, cache=True)
def calculate_turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """Calculate turnover - JIT compiled"""
    return np.abs(w_new - w_prev).sum()

@jit(nopython=True, fastmath=True, cache=True)
def calculate_hhi(weights: np.ndarray) -> float:
    """Calculate HHI - JIT compiled"""
    return np.square(weights).sum()

# ----------------------------- Configuration ----------------------------- #

@dataclass
class EnvConfig:
    # Core parameters
    rebalance_interval: int = 10
    max_weight: float = 0.10
    weight_blocks: int = 100
    allow_cash: bool = True
    cash_rate_as_rf: bool = True
    on_inactive: str = "to_cash"
    
    # Costs & penalties  
    transaction_cost_bps: float = 20.0
    delist_extra_bps: float = 20.0
    lambda_turnover: float = 0.0002
    lambda_hhi: float = 0.01
    lambda_drawdown: float = 0.005
    lambda_tail: float = 0.001
    weight_alpha: float = 2.0
    
    # Optimization flags
    use_minimal_features: bool = True  # CRITICAL: Use only essential features
    max_assets: int = 30  # CRITICAL: Limit number of assets
    feature_subset_size: int = 5  # CRITICAL: Limit features per asset
    
    # Episode control
    max_steps: Optional[int] = 256
    seed: Optional[int] = 42
    random_reset_frac: float = 0.9
    
    # Observation controls (keeping for compatibility)
    normalize_features: bool = True
    obs_clip: float = 7.5
    include_prev_weights: bool = False
    include_active_flag: bool = False
    global_stats: bool = True
    
    # Other params for compatibility
    tail_window: int = 60
    tail_q: float = 0.05
    dd_mode: str = "incremental"

# ----------------------------- Optimized Environment ----------------------------- #

class DebentureTradingEnvOptimized(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    # Class-level cache for preprocessed data
    _data_cache = {}

    def __init__(self, panel: pd.DataFrame, config: EnvConfig):
        super().__init__()
        self.cfg = config
        
        # Use cached preprocessing if available
        panel_id = id(panel)
        cache_key = f"{panel_id}_{self.cfg.max_assets}_{self.cfg.feature_subset_size}"
        
        if cache_key in self._data_cache:
            self._load_cached_data(cache_key)
        else:
            self._prepare_data_optimized(panel)
            self._data_cache[cache_key] = self._get_cacheable_data()
        
        # CRITICAL: Use simple Discrete action space instead of MultiDiscrete
        # Map single integer to portfolio allocation
        self.n_actions = min(1000, self.n_assets * 10)  # Reduced action space
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Simplified observation space
        obs_dim = self._get_minimal_obs_dim()
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Pre-allocate state arrays
        self.prev_w = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w = np.zeros(self.n_assets, dtype=np.float32)
        
        # Pre-allocate observation buffer
        self._obs_buffer = np.zeros(obs_dim, dtype=np.float32)
        
        # Pre-allocate work arrays for step()
        self._work_returns = np.zeros(self.n_assets, dtype=np.float32)
        self._work_mask = np.zeros(self.n_assets, dtype=np.float32)

    def _prepare_data_optimized(self, panel: pd.DataFrame):
        """Ultra-optimized data preparation"""
        
        # Get config values
        max_assets = self.cfg.max_assets
        feature_subset = self.cfg.feature_subset_size
        
        # Limit assets to most liquid
        asset_ids = panel.index.get_level_values("debenture_id").unique()
        if len(asset_ids) > max_assets:
            # Keep assets with most observations
            asset_counts = panel.groupby(level="debenture_id").size()
            top_assets = asset_counts.nlargest(max_assets).index
            panel = panel[panel.index.get_level_values("debenture_id").isin(top_assets)]
            asset_ids = top_assets
        
        self.asset_ids = list(asset_ids[:max_assets])
        self.n_assets = len(self.asset_ids)
        
        # Get dates
        dates = panel.index.get_level_values("date").unique().sort_values()
        self.dates = dates.to_numpy()
        self.T = len(dates)
        
        # CRITICAL: Only extract essential columns
        # Returns and active flags are essential
        self.R = self._fast_pivot(panel, "return", dates, self.asset_ids, fillna=0.0)
        self.ACT = self._fast_pivot(panel, "active", dates, self.asset_ids, fillna=0.0)
        
        # Date-level data (no pivoting needed)
        self.RF = panel.groupby(level="date")["risk_free"].first().reindex(dates).fillna(0.0).to_numpy(dtype=np.float32)
        self.IDX = panel.groupby(level="date")["index_return"].first().reindex(dates).fillna(0.0).to_numpy(dtype=np.float32)
        
        # CRITICAL: Minimal feature extraction
        if self.cfg.use_minimal_features:
            # Only use 2-3 most important features
            essential_features = []
            for col in ["spread_lag1", "duration_lag1", "time_to_maturity_lag1"]:
                if col in panel.columns:
                    essential_features.append(col)
                if len(essential_features) >= 2:
                    break
        else:
            # Limited feature set
            all_lag_cols = [c for c in panel.columns if c.endswith("_lag1")]
            essential_features = all_lag_cols[:feature_subset]
        
        if essential_features:
            # Stack features efficiently
            feat_arrays = []
            for col in essential_features:
                arr = self._fast_pivot(panel, col, dates, self.asset_ids, fillna=0.0)
                feat_arrays.append(arr)
            self.X = np.stack(feat_arrays, axis=-1).astype(np.float32)
        else:
            # Ultra-minimal: just returns
            self.X = self.R.reshape(self.T, self.n_assets, 1).astype(np.float32)
        
        self.F = self.X.shape[-1] if self.X.ndim == 3 else 1

    def _fast_pivot(self, panel: pd.DataFrame, column: str, dates, asset_ids, fillna=0.0):
        """Fast pivoting using unstack"""
        if column not in panel.columns:
            return np.full((len(dates), len(asset_ids)), fillna, dtype=np.float32)
        
        # Use unstack for speed
        pivoted = panel[column].unstack(level="debenture_id")
        return pivoted.reindex(index=dates, columns=asset_ids).fillna(fillna).to_numpy(dtype=np.float32)

    def _get_cacheable_data(self):
        """Get data for caching"""
        return {
            'R': self.R, 'RF': self.RF, 'IDX': self.IDX, 'ACT': self.ACT,
            'X': self.X, 'F': self.F, 'T': self.T,
            'asset_ids': self.asset_ids, 'n_assets': self.n_assets,
            'dates': self.dates
        }
    
    def _load_cached_data(self, cache_key):
        """Load cached data"""
        cached = self._data_cache[cache_key]
        for key, value in cached.items():
            setattr(self, key, value)

    def _get_minimal_obs_dim(self) -> int:
        """Minimal observation dimension"""
        # Features + active flags + 2 scalars
        return self.n_assets * (self.F + 1) + 2

    def _get_observation_fast(self, t: int) -> np.ndarray:
        """Fast observation building using pre-allocated buffer"""
        t = min(t, self.T - 1)
        idx = 0
        
        # Features (already minimal)
        if self.F > 0:
            feat_flat = self.X[t].ravel()
            self._obs_buffer[idx:idx+len(feat_flat)] = feat_flat
            idx += len(feat_flat)
        
        # Active flags
        active_flat = self.ACT[t]
        self._obs_buffer[idx:idx+self.n_assets] = active_flat
        idx += self.n_assets
        
        # Scalars (lagged)
        self._obs_buffer[idx] = self.RF[max(0, t-1)]
        self._obs_buffer[idx+1] = self.IDX[max(0, t-1)]
        
        return self._obs_buffer

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        # Random start point for data augmentation
        if self.cfg.random_reset_frac > 0 and self.cfg.max_steps is not None:
            max_start = max(0, self.T - self.cfg.max_steps)
            jitter = int(max_start * self.cfg.random_reset_frac)
            self.t = np.random.randint(0, jitter + 1) if jitter > 0 else 0
        else:
            self.t = 0
        
        self.wealth = 1.0
        self.peak_wealth = 1.0
        
        # Reset weights
        self.prev_w.fill(0.0)
        self.curr_w.fill(0.0)
        
        # Initialize with equal weight or cash
        if self.cfg.allow_cash and self.n_assets > 0:
            # Start with cash
            self.curr_w[-1] = 1.0 if "_CASH_" in self.asset_ids[-1] else 0.0
        else:
            # Equal weight active assets
            mask = self.ACT[self.t] > 0
            if mask.any():
                self.curr_w[mask] = 1.0 / mask.sum()
        
        return self._get_observation_fast(self.t), {}

    def step(self, action: int):
        """Optimized step using JIT-compiled operations"""
        t = self.t
        
        if t >= self.T - 1:
            return self._get_observation_fast(t), 0.0, True, False, {}
        
        # Simple action mapping: higher action = more concentrated portfolio
        action_pct = action / max(self.n_actions - 1, 1)
        concentration = 1 - action_pct * 0.7  # 30% to 100% of assets
        n_hold = max(1, int(self.n_assets * concentration))
        
        # Only rebalance periodically
        if t % self.cfg.rebalance_interval == 0:
            # Get active mask
            mask = self.ACT[t]
            active_indices = np.where(mask > 0)[0]
            
            # Simple allocation strategy
            w_new = np.zeros(self.n_assets, dtype=np.float32)
            if len(active_indices) > 0:
                # Random selection for diversity (but deterministic within episode)
                selected = active_indices[:min(n_hold, len(active_indices))]
                w_new[selected] = 1.0 / len(selected)
        else:
            w_new = self.curr_w.copy()
            # Handle delistings
            w_new *= (self.ACT[t] > 0).astype(np.float32)
            if w_new.sum() > 0:
                w_new /= w_new.sum()
        
        # Use JIT functions for calculations
        self._work_returns[:] = self.R[t]
        turn = calculate_turnover(w_new, self.curr_w)
        r_p = calculate_portfolio_return(w_new, self._work_returns, self.RF[t])
        hhi = calculate_hhi(w_new)
        
        # Simple costs
        cost = (self.cfg.transaction_cost_bps / 10000.0) * turn
        
        # Simple reward
        alpha = r_p - self.IDX[t]
        reward = self.cfg.weight_alpha * alpha - cost - self.cfg.lambda_turnover * turn - self.cfg.lambda_hhi * hhi
        
        # Update state
        self.prev_w[:] = self.curr_w
        self.curr_w[:] = w_new
        self.wealth *= (1 + r_p - cost)
        self.peak_wealth = max(self.peak_wealth, self.wealth)
        self.t += 1
        
        terminated = (self.t >= self.T - 1)
        truncated = (self.cfg.max_steps is not None and self.t >= self.cfg.max_steps)
        
        info = {
            "portfolio_return": float(r_p - cost),
            "turnover": float(turn),
            "wealth": float(self.wealth),
            "alpha": float(alpha),
            "index_return": float(self.IDX[t]),
            "rf": float(self.RF[t]),
            "drawdown": 1.0 - (self.wealth / max(self.peak_wealth, 1e-12))
        }
        
        return self._get_observation_fast(self.t), reward, terminated, truncated, info

    def render(self):
        pass

    def get_asset_ids(self) -> List[str]:
        return list(self.asset_ids)

# Factory function
def make_env_from_panel(panel: pd.DataFrame, **env_kwargs) -> DebentureTradingEnvOptimized:
    """Factory: build optimized environment"""
    # Extract config values from kwargs or config.yaml
    cfg_dict = {}
    for key in EnvConfig.__dataclass_fields__.keys():
        if key in env_kwargs:
            cfg_dict[key] = env_kwargs[key]
    
    cfg = EnvConfig(**cfg_dict)
    return DebentureTradingEnvOptimized(panel=panel, config=cfg)