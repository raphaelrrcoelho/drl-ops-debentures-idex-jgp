# env_final_memmap.py
"""
Memory-efficient environment using memory-mapped arrays for true shared memory
across processes. Works with SubprocVecEnv for parallel training.
"""
from __future__ import annotations

import os
import tempfile
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, ClassVar
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import pickle

# ----------------------------- Configuration ----------------------------- #

@dataclass
class EnvConfig:
    # Rebalance & constraints
    rebalance_interval: int = 5                 
    max_weight: float = 0.10                    
    weight_blocks: int = 100                    
    allow_cash: bool = True                     
    cash_rate_as_rf: bool = True                
    on_inactive: str = "to_cash"                

    # Costs & penalties
    weight_excess: float = 0.0   
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

    # Episode control
    max_steps: Optional[int] = None             
    seed: Optional[int] = 42                    
    random_reset_frac: float = 0.0              

# ------------------------------ Utilities -------------------------------- #

def _blocks_to_weights(blocks: np.ndarray, total_blocks: int = 100) -> np.ndarray:
    blocks = np.asarray(blocks, dtype=float)
    total = blocks.sum()
    if total <= 0:
        return np.zeros_like(blocks, dtype=np.float32)
    return (blocks / total).astype(np.float32)

def _sanitize_blocks_with_mask(blocks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return blocks * (mask > 0).astype(int)

def _hhi(w: np.ndarray) -> float:
    return float(np.square(w).sum())

def _turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    return float(np.abs(w_new - w_prev).sum())

def _dict_sector_exposures(sector_ids: np.ndarray, weights: np.ndarray) -> Dict[int, float]:
    m = min(len(sector_ids), len(weights))
    sid = np.asarray(sector_ids[:m])
    w = np.asarray(weights[:m], dtype=float)
    d: Dict[int, float] = {}
    for s in np.unique(sid):
        if s < 0:
            continue
        d[int(s)] = float(w[sid == s].sum())
    return d

# ----------------------- Memory-Mapped Storage ---------------------------- #

class MemmapDataRegistry:
    """
    Registry for memory-mapped arrays that can be shared across processes.
    """
    _storage: ClassVar[Dict[str, Dict]] = {}
    _temp_dir: ClassVar[Optional[str]] = None
    
    @classmethod
    def get_temp_dir(cls, panel_hash: str) -> str:
        """Get deterministic temp directory based on panel hash."""
        # Use a fixed base directory with the panel hash
        temp_base = os.path.join(tempfile.gettempdir(), "deb_env_memmaps")
        os.makedirs(temp_base, exist_ok=True)
    
        # Directory name is deterministic based on panel hash
        temp_dir = os.path.join(temp_base, f"panel_{panel_hash}")
        os.makedirs(temp_dir, exist_ok=True)
        
        return temp_dir
    
    @classmethod
    def get_or_create(cls, panel_hash: str, panel: pd.DataFrame, config: EnvConfig) -> Dict:
        """Get existing memmap data or create new one."""
        if panel_hash not in cls._storage:
            # Use deterministic temp directory
            temp_dir = cls.get_temp_dir(panel_hash)  # <-- Pass panel_hash
            meta_path = os.path.join(temp_dir, f"{panel_hash}_meta.pkl")
            
            if os.path.exists(meta_path):
                # Load existing memmaps
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                cls._storage[panel_hash] = cls._load_memmaps(panel_hash, meta)
                print(f"[INFO] Loaded existing memory maps from {temp_dir}")
            else:
                # Create new memmaps
                print("First environment creation - preprocessing data...")
                data = cls._prepare_data(panel, config)
                cls._storage[panel_hash] = cls._save_memmaps(panel_hash, data, temp_dir)  # Pass temp_dir
                print(f"[INFO] Created memory maps in {temp_dir}")
        else:
            print("[INFO] Using cached memory maps from registry")
        
        return cls._storage[panel_hash]
    
    @classmethod
    def _save_memmaps(cls, panel_hash: str, data: Dict, temp_dir: str) -> Dict:
        """Save arrays as memory-mapped files."""
        memmap_data = {}
        meta = {}
        
        # Arrays to save as memmaps
        array_keys = ['R', 'RF', 'IDX', 'ACT', 'X', 'RF_obs', 'IDX_obs', 
                     'sector_ids', 'global_means', 'global_stds']
        
        for key in array_keys:
            if key in data:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    # Save as memmap
                    path = os.path.join(temp_dir, f"{panel_hash}_{key}.npy")
                    shape = arr.shape
                    dtype = arr.dtype
                    
                    # Create memmap and copy data
                    mmap = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
                    mmap[:] = arr[:]
                    mmap.flush()
                    
                    # Store memmap reference
                    memmap_data[key] = np.memmap(path, dtype=dtype, mode='r', shape=shape)
                    meta[key] = {'shape': shape, 'dtype': str(dtype), 'path': path}
                else:
                    # Handle None values
                    memmap_data[key] = None
                    meta[key] = None
        
        # Non-array data (keep in memory)
        for key in data:
            if key not in array_keys:
                memmap_data[key] = data[key]
                meta[key] = data[key]
        
        # Save metadata
        meta_path = os.path.join(temp_dir, f"{panel_hash}_meta.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        return memmap_data
    
    @classmethod
    def _load_memmaps(cls, panel_hash: str, meta: Dict) -> Dict:
        """Load existing memory-mapped arrays."""
        memmap_data = {}
        
        for key, value in meta.items():
            if isinstance(value, dict) and 'path' in value:
                # Load memmap
                memmap_data[key] = np.memmap(
                    value['path'], 
                    dtype=np.dtype(value['dtype']), 
                    mode='r', 
                    shape=value['shape']
                )
            else:
                # Regular data or None
                memmap_data[key] = value
        
        return memmap_data
    
    @classmethod
    def _prepare_data(cls, panel: pd.DataFrame, cfg: EnvConfig) -> Dict:
        """Prepare all data arrays (same logic as original)."""
        np.random.seed(42)
        panel = panel.sort_index()
        panel = panel.round(6)

        # Required columns
        required = [
            "return", "risk_free", "index_return", "active",
            "spread", "duration", "time_to_maturity", "sector_id", "index_level"
        ]
        for c in required:
            if c not in panel.columns:
                raise ValueError(f"Missing required column '{c}' in panel")

        # Base features (lagged versions)
        base_feats = [
            "return", "spread", "duration", "time_to_maturity",
            "risk_free", "index_return", "ttm_rank",
            "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
            "sector_fitted_spread", "spread_residual_ns",
            "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
            "sector_spread", "sector_momentum", "sector_weight_index",
            "sector_id",
        ]
        
        if cfg.include_active_flag and "active" in panel.columns:
            base_feats.append("active")

        panel = panel.copy()

        def ensure_lag1(col: str) -> str:
            if col in ("sector_id", "active"):
                return col
            lag_col = f"{col}_lag1"
            if lag_col not in panel.columns:
                panel[lag_col] = (
                    panel.groupby(level="debenture_id", sort=False)[col]
                         .shift(1)
                         .astype(np.float32)
                         .replace([np.inf, -np.inf], np.nan)
                         .fillna(0.0)
                )
            return lag_col

        feat_cols = [ensure_lag1(c) for c in base_feats if c in panel.columns]
        feat_cols = [c for c in feat_cols if "index_weight" not in c]

        z_like = [c for c in panel.columns if (c.endswith("_z") or c.endswith("_z252") or "_z_" in c)]
        for zc in sorted(z_like):
            lagc = ensure_lag1(zc)
            if lagc not in feat_cols:
                feat_cols.append(lagc)

        for c in feat_cols:
            if c not in panel.columns:
                panel[c] = 0.0

        # Dates and asset IDs
        dates = panel.index.get_level_values("date").unique().sort_values()
        dates_np = dates.to_numpy()
        asset_ids = panel.index.get_level_values("debenture_id").unique().tolist()
        n_assets = len(asset_ids)

        # Arrays
        R = (
            panel["return"].reset_index()
                 .pivot(index="date", columns="debenture_id", values="return")
                 .reindex(index=dates, columns=asset_ids)
        )
        RF = (
            panel.reset_index()[["date", "risk_free"]]
                 .drop_duplicates(subset=["date"], keep="last")
                 .set_index("date")
                 .reindex(dates)["risk_free"]
                 .fillna(0.0)
        )
        IDX = (
            panel.reset_index()[["date", "index_return"]]
                 .drop_duplicates(subset=["date"], keep="last")
                 .set_index("date")
                 .reindex(dates)["index_return"]
                 .fillna(0.0)
        )
        A = (
            panel["active"].reset_index()
                 .pivot(index="date", columns="debenture_id", values="active")
                 .reindex(index=dates, columns=asset_ids)
                 .fillna(0.0)
        )

        # Feature tensor
        feat_mats: List[np.ndarray] = []
        for c in feat_cols:
            wide = (
                panel[c].reset_index()
                     .pivot(index="date", columns="debenture_id", values=c)
                     .reindex(index=dates, columns=asset_ids)
                     .fillna(0.0)
                     .to_numpy(dtype=np.float32)
            )
            feat_mats.append(wide)
        X = np.stack(feat_mats, axis=-1) if feat_mats else np.zeros((len(dates), len(asset_ids), 0), dtype=np.float32)

        # Sector IDs
        sector_id_wide = (
            panel["sector_id"].reset_index()
                 .pivot(index="date", columns="debenture_id", values="sector_id")
                 .reindex(index=dates, columns=asset_ids)
                 .ffill().bfill().fillna(-1)
        )
        sector_ids = sector_id_wide.to_numpy(dtype=np.int16)[0]

        # Store arrays
        R_arr = np.nan_to_num(R.to_numpy(dtype=np.float32), nan=0.0)
        RF_arr = np.nan_to_num(RF.to_numpy(dtype=np.float32).ravel(), nan=0.0)
        IDX_arr = np.nan_to_num(IDX.to_numpy(dtype=np.float32).ravel(), nan=0.0)
        ACT_arr = np.nan_to_num(A.to_numpy(dtype=np.float32), nan=0.0)
        X_arr = np.nan_to_num(X, nan=0.0)
        T = R_arr.shape[0]
        F = X_arr.shape[-1] if X_arr.ndim == 3 else 0

        # Cross-sectional stats
        if cfg.global_stats and F > 0:
            act = (ACT_arr > 0).astype(np.float32)
            denom = np.maximum(act.sum(axis=1, keepdims=True), 1.0)
            means = (X_arr * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None]
            stds = np.sqrt(((X_arr - means) ** 2 * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None])
            global_means = means.squeeze(1)
            global_stds = np.maximum(stds.squeeze(1), 1e-6)
        else:
            global_means = None
            global_stds = None

        # Z-normalization
        if cfg.normalize_features and F > 0:
            Xn = X_arr.copy()
            eps = 1e-6
            for f in range(F):
                mu = global_means[:, f][:, None] if global_means is not None else 0.0
                sd = global_stds[:, f][:, None] if global_stds is not None else 1.0
                Xn[:, :, f] = (Xn[:, :, f] - mu) / (sd + eps)
            X_arr = np.clip(Xn, -cfg.obs_clip, cfg.obs_clip)

        # Lagged RF/IDX
        RF_obs = np.zeros_like(RF_arr, dtype=np.float32)
        RF_obs[1:] = RF_arr[:-1]
        IDX_obs = np.zeros_like(IDX_arr, dtype=np.float32)
        IDX_obs[1:] = IDX_arr[:-1]

        # Append cash if enabled
        if cfg.allow_cash:
            if cfg.cash_rate_as_rf:
                cash_R = RF_arr.reshape(-1, 1)
            else:
                cash_R = np.zeros((T, 1), dtype=np.float32)
            cash_X = np.zeros((T, 1, F), dtype=np.float32)
            cash_A = np.ones((T, 1), dtype=np.float32)
            R_arr = np.concatenate([R_arr, cash_R], axis=1)
            X_arr = np.concatenate([X_arr, cash_X], axis=1) if F > 0 else X_arr
            ACT_arr = np.concatenate([ACT_arr, cash_A], axis=1)
            asset_ids = list(asset_ids) + ["__CASH__"]
            n_assets += 1

        return {
            'dates': dates_np,
            'asset_ids': asset_ids,
            'n_assets': n_assets,
            'R': R_arr,
            'RF': RF_arr,
            'IDX': IDX_arr,
            'ACT': ACT_arr,
            'X': X_arr,
            'T': T,
            'F': F,
            'feature_cols': feat_cols,
            'sector_ids': sector_ids,
            'global_means': global_means,
            'global_stds': global_stds,
            'RF_obs': RF_obs,
            'IDX_obs': IDX_obs,
        }

# ------------------------------ Environment ------------------------------- #

class DebentureTradingEnv(gym.Env):
    def __init__(self, panel: pd.DataFrame, config: EnvConfig, panel_hash: Optional[str] = None):
        super().__init__()
        assert isinstance(panel.index, pd.MultiIndex), "panel must be MultiIndex (date, debenture_id)"
        self.cfg = config
        if self.cfg.seed is not None:
            np.random.seed(int(self.cfg.seed))
        
        # Use provided hash or generate deterministic one
        if panel_hash is not None:
            self.panel_hash = panel_hash
        else:
            date_range = panel.index.get_level_values("date")
            hash_str = f"{panel.shape}_{date_range.min()}_{date_range.max()}_{len(panel.index.get_level_values('debenture_id').unique())}"
            self.panel_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:16]
        
        # Get or create memory-mapped data
        self.shared = MemmapDataRegistry.get_or_create(self.panel_hash, panel, config)
        
        # Reference shared arrays (read-only memmaps)
        self.dates = self.shared['dates']
        self.asset_ids = self.shared['asset_ids']
        self.n_assets = self.shared['n_assets']
        self.R = self.shared['R']
        self.RF = self.shared['RF']
        self.IDX = self.shared['IDX']
        self.ACT = self.shared['ACT']
        self.X = self.shared['X']
        self.T = self.shared['T']
        self.F = self.shared['F']
        self.feature_cols = self.shared['feature_cols']
        self.sector_ids = self.shared['sector_ids']
        self.global_means = self.shared['global_means']
        self.global_stds = self.shared['global_stds']
        self.RF_obs = self.shared['RF_obs']
        self.IDX_obs = self.shared['IDX_obs']

        # Cash index
        self.cash_idx = None
        if self.cfg.allow_cash:
            self.cash_idx = self.n_assets - 1  

        # Action space
        self.max_blocks_per_asset = int(self.cfg.max_weight * self.cfg.weight_blocks)
        self.action_space = spaces.MultiDiscrete(
            [self.max_blocks_per_asset + 1] * self.n_assets
        )

        # Observation space
        obs_dim = self._obs_size()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            'action_mask': spaces.Box(0, 1, shape=(self.n_assets,), dtype=np.int8)
        })

        # Instance-specific state
        self.t: int = 0
        self.prev_w: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        self.wealth: float = 1.0
        self.peak_wealth: float = 1.0
        self.tail_buffer: List[float] = []
        self._history: Dict[str, list] = {}

    def _obs_size(self) -> int:
        n = self.n_assets
        base = n * self.F
        extra = 0
        if self.cfg.include_prev_weights:
            extra += n
        if self.cfg.include_active_flag:
            extra += n
        if self.cfg.global_stats and self.F > 0:
            extra += 2 * self.F
        extra += 2
        return base + extra

    def _get_observation(self) -> Dict:
        t = min(self.t, self.T - 1)
        if t >= self.ACT.shape[0]:
            t = self.ACT.shape[0] - 1
            
        parts: List[np.ndarray] = []
        if self.F > 0:
            # X_t = np.array(self.X[t])  # Convert memmap slice to array
            parts.append(self.X[t].ravel())
        if self.cfg.include_prev_weights:
            parts.append(self.prev_w.astype(np.float32).ravel())
        if self.cfg.include_active_flag:
            parts.append(np.array(self.ACT[t]).astype(np.float32).ravel())
        if self.cfg.global_stats and self.F > 0 and self.global_means is not None:
            parts.append(np.array(self.global_means[t]).astype(np.float32).ravel())
            parts.append(np.array(self.global_stds[t]).astype(np.float32).ravel())
            parts.append(np.array([self.RF_obs[t], self.IDX_obs[t]], dtype=np.float32).ravel())
        
        obs = np.concatenate(parts) if parts else np.zeros((self._obs_size(),), dtype=np.float32)
        clip = float(self.cfg.obs_clip)
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        mask = np.array(self.ACT[t]).astype(np.int8)
        
        return {
            'observation': np.clip(obs, -clip, clip).astype(np.float32),
            'action_mask': mask
        }

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

        self.prev_w = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w = np.zeros(self.n_assets, dtype=np.float32)
        if self.cash_idx is not None:
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
        terminated = False
        truncated = False
        
        if t >= self.T or t >= self.ACT.shape[0]:
            return self._get_observation(), 0.0, True, False, {}

        act_mask = np.array(self.ACT[t])  # Convert memmap slice
        r_vec = np.array(self.R[t])  # Convert memmap slice
        
        apply_action = (t % max(1, int(self.cfg.rebalance_interval)) == 0)
        w_prev = self.curr_w.copy()
        freed_mass = 0.0
        extra_delist_cost = 0.0

        if apply_action:
            blocks = np.asarray(action, dtype=int)
            blocks = _sanitize_blocks_with_mask(blocks, act_mask)
            w_tgt = _blocks_to_weights(blocks, self.cfg.weight_blocks)
            w_tgt = np.minimum(w_tgt, self.cfg.max_weight)
            
            if w_tgt.sum() > 0:
                w_tgt = w_tgt / w_tgt.sum()
            elif self.cash_idx is not None:
                w_tgt = np.zeros_like(w_tgt)
                w_tgt[self.cash_idx] = 1.0
        else:
            w_tgt = w_prev.copy()
            inactive = (act_mask <= 0)
            freed = float(np.maximum(w_tgt[inactive], 0.0).sum())
            if freed > 0.0:
                w_tgt[inactive] = 0.0
                freed_mass = freed
                extra_delist_cost = freed_mass * (self.cfg.delist_extra_bps / 10000.0)
                
                if self.cfg.on_inactive == "to_cash" and self.cash_idx is not None:
                    w_tgt[self.cash_idx] += freed
                else:
                    active = (act_mask > 0)
                    s = float(w_tgt[active].sum())
                    if s > 0.0:
                        w_tgt[active] += (w_tgt[active] / s) * freed
                    elif self.cash_idx is not None:
                        w_tgt[self.cash_idx] += freed
                        
                if w_tgt.sum() > 0:
                    w_tgt = w_tgt / w_tgt.sum()

        turn = _turnover(w_tgt, w_prev)
        lin_cost = (self.cfg.transaction_cost_bps / 10000.0) * turn + extra_delist_cost

        rf_t = float(self.RF[t])
        bad = ~np.isfinite(r_vec)
        if bad.any():
            r_vec[bad] = rf_t
        r_p = float(np.dot(w_tgt, r_vec))
        
        r_idx = float(self.IDX[t])
        alpha = r_p - r_idx
        excess = r_p - rf_t

        net = max((1.0 + r_p) * (1.0 - max(lin_cost, 0.0)), 1e-12)
        r_net = net - 1.0
        self.wealth *= net
        self.peak_wealth = max(self.peak_wealth, self.wealth)

        cur_dd_level = 1.0 - (self.wealth / max(self.peak_wealth, 1e-12))
        if self.cfg.dd_mode == "level":
            dd_pen = -self.cfg.lambda_drawdown * abs(cur_dd_level)
        else:
            prev_dd_level = self._history.get("drawdown", [0.0])[-1] if self._history.get("drawdown") else 0.0
            dd_inc = max(cur_dd_level - prev_dd_level, 0.0)
            dd_pen = -self.cfg.lambda_drawdown * dd_inc

        tail_pen = 0.0
        self.tail_buffer.append(r_p)
        if len(self.tail_buffer) > self.cfg.tail_window:
            self.tail_buffer.pop(0)
        if self.cfg.lambda_tail > 0 and len(self.tail_buffer) >= max(10, self.cfg.tail_window // 2):
            q = float(np.quantile(self.tail_buffer, self.cfg.tail_q))
            if r_p < q:
                tail_pen = abs(r_p)

        hhi_val = _hhi(w_tgt)
        pen = (
            - self.cfg.lambda_turnover * turn
            - self.cfg.lambda_hhi * hhi_val
            + dd_pen
            - self.cfg.lambda_tail * tail_pen
        )

        reward = float(self.cfg.weight_alpha * alpha + self.cfg.weight_excess * excess + pen - lin_cost)

        self.prev_w = w_prev
        self.curr_w = w_tgt
        self.t = t + 1
        
        if self.cfg.max_steps is not None and self.t >= int(self.cfg.max_steps):
            truncated = True
        if self.t >= self.T:
            terminated = True

        obs = self._get_observation()
        info = {
            "date": pd.Timestamp(self.dates[min(self.t, self.T - 1)]).to_pydatetime(),
            "weights": self.curr_w.copy(),
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
            "sector_exposure": _dict_sector_exposures(self.sector_ids, self.curr_w),
            "config": asdict(self.cfg) if self.t == 1 else None,
        }
        
        for k, v in info.items():
            if k not in ["config", "sector_exposure", "weights", "date"]:
                if k not in self._history:
                    self._history[k] = []
                self._history[k].append(v)
                
        return obs, reward, terminated, truncated, info

    def render(self):
        t = min(self.t, self.T - 1)
        date = pd.Timestamp(self.dates[t]).date()
        wealth = self.wealth
        dd = self._history["drawdown"][-1] if self._history.get("drawdown") else 0.0
        print(f"[{date}] wealth={wealth:.4f} dd={dd:.3%}")

    def get_history(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history, index=pd.to_datetime(
            self.dates[:len(self._history.get('wealth', []))]
        ))

    def get_asset_ids(self) -> List[str]:
        return list(self.asset_ids)

    def get_action_masks(self) -> np.ndarray:
        masks = []
        act_mask = self.ACT[self.t]
        
        for i in range(self.n_assets):
            asset_mask = np.ones(self.max_blocks_per_asset + 1, dtype=bool)
            if act_mask[i] <= 0:
                asset_mask[1:] = False
            masks.append(asset_mask)
            
        return masks

def make_env_from_panel(panel: pd.DataFrame, panel_hash: Optional[str] = None, **env_kwargs) -> DebentureTradingEnv:
    """Factory: build DebentureTradingEnv from a panel and EnvConfig kwargs."""
    cfg = EnvConfig(**env_kwargs)
    return DebentureTradingEnv(panel=panel, config=cfg, panel_hash=panel_hash)