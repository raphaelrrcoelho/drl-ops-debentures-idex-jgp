# env_final_discrete.py
"""
Debenture portfolio environment with DISCRETE action space for PPO
-------------------------------------------------------------------
Modified to use MultiDiscrete actions with 1% allocation blocks.
Each asset can receive 0-10 blocks (0-10% of portfolio).
Maintains all original features: dynamic universe, transaction costs,
penalties, sector features, and cash option.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# ----------------------------- Configuration ----------------------------- #

@dataclass
class EnvConfig:
    # Rebalance & constraints
    rebalance_interval: int = 5                 # days between applying new action
    max_weight: float = 0.10                    # per-asset cap
    weight_blocks: int = 100                    # Total blocks (100 = 1% granularity)
    allow_cash: bool = True                     # if True, append cash asset
    cash_rate_as_rf: bool = True                # if cash exists, accrues at rf
    on_inactive: str = "to_cash"                # {"to_cash","pro_rata"}

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
    """Convert discrete block allocations to continuous weights."""
    blocks = np.asarray(blocks, dtype=float)
    total = blocks.sum()
    if total <= 0:
        return np.zeros_like(blocks, dtype=np.float32)
    return (blocks / total).astype(np.float32)

def _sanitize_blocks_with_mask(blocks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out blocks for inactive assets."""
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

# ------------------------------ Environment ------------------------------- #

class DebentureTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, panel: pd.DataFrame, config: EnvConfig):
        super().__init__()
        assert isinstance(panel.index, pd.MultiIndex), "panel must be MultiIndex (date, debenture_id)"
        self.cfg = config
        if self.cfg.seed is not None:
            np.random.seed(int(self.cfg.seed))

        # Prepare arrays
        self._prepare_data(panel)

        # Cash index (if any)
        self.cash_idx = None
        if self.cfg.allow_cash:
            self.cash_idx = self.n_assets - 1  

        # DISCRETE ACTION SPACE
        # Each asset can get 0 to max_blocks_per_asset blocks
        self.max_blocks_per_asset = int(self.cfg.max_weight * self.cfg.weight_blocks)
        self.action_space = spaces.MultiDiscrete(
            [self.max_blocks_per_asset + 1] * self.n_assets
        )

        # Observation space (unchanged)
        obs_dim = self._obs_size()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            'action_mask': spaces.Box(0, 1, shape=(self.n_assets,), dtype=np.int8)
        })

        # State
        self.t: int = 0
        self.prev_w: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w: np.ndarray = np.zeros(self.n_assets, dtype=np.float32)
        self.wealth: float = 1.0
        self.peak_wealth: float = 1.0

        # Tail buffer (optional)
        self.tail_buffer: List[float] = []

        # Logs
        self._history: Dict[str, list] = {}

    # --------------------------- Data preparation --------------------------- #

    def _prepare_data(self, panel: pd.DataFrame):
        panel = panel.sort_index()

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
        
        if self.cfg.include_active_flag and "active" in panel.columns:
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
        self.dates = dates.to_numpy()
        asset_ids = panel.index.get_level_values("debenture_id").unique().tolist()
        self.asset_ids: List[str] = list(asset_ids)
        self.n_assets = len(self.asset_ids)

        # Arrays (same-day returns, same-day active mask)
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

        # Feature tensor (lagged)
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
        self.feature_cols = list(feat_cols)

        # Sector IDs
        sector_id_wide = (
            panel["sector_id"].reset_index()
                 .pivot(index="date", columns="debenture_id", values="sector_id")
                 .reindex(index=dates, columns=asset_ids)
                 .ffill().bfill().fillna(-1)
        )
        self.sector_ids = sector_id_wide.to_numpy(dtype=np.int16)[0]

        # Store arrays
        self.R = np.nan_to_num(R.to_numpy(dtype=np.float32), nan=0.0)
        self.RF = np.nan_to_num(RF.to_numpy(dtype=np.float32).ravel(), nan=0.0)
        self.IDX = np.nan_to_num(IDX.to_numpy(dtype=np.float32).ravel(), nan=0.0)
        self.ACT = np.nan_to_num(A.to_numpy(dtype=np.float32), nan=0.0)
        self.X = np.nan_to_num(X, nan=0.0)
        self.T = self.R.shape[0]
        self.F = self.X.shape[-1] if self.X.ndim == 3 else 0

        # Cross-sectional stats
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

        # Z-normalization
        if self.cfg.normalize_features and self.F > 0:
            Xn = self.X.copy()
            eps = 1e-6
            for f in range(self.F):
                mu = self.global_means[:, f][:, None] if self.global_means is not None else 0.0
                sd = self.global_stds[:, f][:, None] if self.global_stds is not None else 1.0
                Xn[:, :, f] = (Xn[:, :, f] - mu) / (sd + eps)
            self.X = np.clip(Xn, -self.cfg.obs_clip, self.cfg.obs_clip)

        # Lagged RF/IDX for observations
        self.RF_obs = np.zeros_like(self.RF, dtype=np.float32)
        self.RF_obs[1:] = self.RF[:-1]
        self.IDX_obs = np.zeros_like(self.IDX, dtype=np.float32)
        self.IDX_obs[1:] = self.IDX[:-1]

        # Append cash if enabled
        if self.cfg.allow_cash:
            if self.cfg.cash_rate_as_rf:
                cash_R = self.RF.reshape(-1, 1)
            else:
                cash_R = np.zeros((self.T, 1), dtype=np.float32)
            cash_X = np.zeros((self.T, 1, self.F), dtype=np.float32)
            cash_A = np.ones((self.T, 1), dtype=np.float32)
            self.R = np.concatenate([self.R, cash_R], axis=1)
            self.X = np.concatenate([self.X, cash_X], axis=1) if self.F > 0 else self.X
            self.ACT = np.concatenate([self.ACT, cash_A], axis=1)
            self.asset_ids.append("__CASH__")
            self.n_assets += 1

    # --------------------------- Observation builder ------------------------ #

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
        extra += 2  # RF and IDX scalars
        return base + extra

    def _get_observation(self) -> Dict:
        t = min(self.t, self.T - 1)
        if t >= self.ACT.shape[0]:
            t = self.ACT.shape[0] - 1
            
        parts: List[np.ndarray] = []
        if self.F > 0:
            X_t = self.X[t].copy()
            parts.append(X_t.reshape(-1))
        if self.cfg.include_prev_weights:
            parts.append(self.prev_w.astype(np.float32).ravel())
        if self.cfg.include_active_flag:
            parts.append(self.ACT[t].astype(np.float32).ravel())
        if self.cfg.global_stats and self.F > 0:
            parts.append(self.global_means[t].astype(np.float32).ravel())
            parts.append(self.global_stds[t].astype(np.float32).ravel())
        parts.append(np.array([self.RF_obs[t], self.IDX_obs[t]], dtype=np.float32).ravel())
        
        obs = np.concatenate(parts) if parts else np.zeros((self._obs_size(),), dtype=np.float32)
        clip = float(self.cfg.obs_clip)
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        mask = self.ACT[t].astype(np.int8)
        
        return {
            'observation': np.clip(obs, -clip, clip).astype(np.float32),
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

        # Initialize weights
        self.prev_w = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w = np.zeros(self.n_assets, dtype=np.float32)
        # Start with all cash if available
        if self.cash_idx is not None:
            self.curr_w[self.cash_idx] = 1.0
        else:
            # Equal weight active assets
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

        # Get current state
        act_mask = self.ACT[t]
        r_vec = self.R[t].copy()
        
        # Only rebalance every rebalance_interval days
        apply_action = (t % max(1, int(self.cfg.rebalance_interval)) == 0)
        w_prev = self.curr_w.copy()
        freed_mass = 0.0
        extra_delist_cost = 0.0

        if apply_action:
            # DISCRETE ACTION HANDLING
            blocks = np.asarray(action, dtype=int)
            
            # Apply activity mask to blocks
            blocks = _sanitize_blocks_with_mask(blocks, act_mask)
            
            # Convert blocks to weights
            w_tgt = _blocks_to_weights(blocks, self.cfg.weight_blocks)
            
            # Ensure max weight constraint is respected
            # (blocks should already respect this, but double-check)
            w_tgt = np.minimum(w_tgt, self.cfg.max_weight)
            
            # Renormalize if needed
            if w_tgt.sum() > 0:
                w_tgt = w_tgt / w_tgt.sum()
            elif self.cash_idx is not None:
                w_tgt = np.zeros_like(w_tgt)
                w_tgt[self.cash_idx] = 1.0
        else:
            # Hold position but handle delistings
            w_tgt = w_prev.copy()
            inactive = (act_mask <= 0)
            freed = float(np.maximum(w_tgt[inactive], 0.0).sum())
            if freed > 0.0:
                w_tgt[inactive] = 0.0
                freed_mass = freed
                extra_delist_cost = freed_mass * (self.cfg.delist_extra_bps / 10000.0)
                
                # Send freed mass to cash or redistribute
                if self.cfg.on_inactive == "to_cash" and self.cash_idx is not None:
                    w_tgt[self.cash_idx] += freed
                else:
                    active = (act_mask > 0)
                    s = float(w_tgt[active].sum())
                    if s > 0.0:
                        w_tgt[active] += (w_tgt[active] / s) * freed
                    elif self.cash_idx is not None:
                        w_tgt[self.cash_idx] += freed
                        
                # Renormalize
                if w_tgt.sum() > 0:
                    w_tgt = w_tgt / w_tgt.sum()

        # Calculate turnover and costs
        turn = _turnover(w_tgt, w_prev)
        lin_cost = (self.cfg.transaction_cost_bps / 10000.0) * turn + extra_delist_cost

        # Portfolio return
        rf_t = float(self.RF[t])
        bad = ~np.isfinite(r_vec)
        if bad.any():
            r_vec[bad] = rf_t
        r_p = float(np.dot(w_tgt, r_vec))
        
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

        # Other penalties
        hhi_val = _hhi(w_tgt)
        pen = (
            - self.cfg.lambda_turnover * turn
            - self.cfg.lambda_hhi * hhi_val
            + dd_pen
            - self.cfg.lambda_tail * tail_pen
        )

        # Reward
        reward = float(self.cfg.weight_alpha * alpha + self.cfg.weight_excess * excess + pen - lin_cost)

        # Update state
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
        
        # Store history
        for k, v in info.items():
            if k not in ["config", "sector_exposure", "weights", "date"]:
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
        print(f"[{date}] wealth={wealth:.4f} dd={dd:.3%}")

    def get_history(self) -> pd.DataFrame:
        """Return a DataFrame of per-step logged metrics."""
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history, index=pd.to_datetime(
            self.dates[:len(self._history.get('wealth', []))]
        ))

    def get_asset_ids(self) -> List[str]:
        return list(self.asset_ids)

    def get_action_masks(self) -> np.ndarray:
        """
        For MaskablePPO: returns valid actions mask for MultiDiscrete space.
        Since we're using independent blocks per asset, all actions are valid
        (the environment handles inactive assets internally).
        """
        # For MultiDiscrete, return mask per asset's action dimension
        # Each asset can take 0 to max_blocks_per_asset actions
        masks = []
        act_mask = self.ACT[self.t]
        
        for i in range(self.n_assets):
            asset_mask = np.ones(self.max_blocks_per_asset + 1, dtype=bool)
            # If asset is inactive, only allow 0 blocks
            if act_mask[i] <= 0:
                asset_mask[1:] = False
            masks.append(asset_mask)
            
        return masks

# ------------------------- Factory convenience ---------------------------- #

def make_env_from_panel(panel: pd.DataFrame, **env_kwargs) -> DebentureTradingEnv:
    """Factory: build DebentureTradingEnv from a panel and EnvConfig kwargs."""
    cfg = EnvConfig(**env_kwargs)
    return DebentureTradingEnv(panel=panel, config=cfg)

# ------------------------------ Quick test -------------------------------- #

if __name__ == "__main__":
    # Test with discrete actions
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=12, freq="B")
    ids = ["A", "B"]
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
    df["sector_spread"] = 0.03
    df["sector_momentum"] = 0.0
    df["sector_weight_index"] = 0.5
    
    # Create lag columns
    for c in ["return","spread","duration","time_to_maturity","risk_free","index_return",
              "sector_spread","sector_momentum","sector_weight_index"]:
        df[f"{c}_lag1"] = df[c]

    env = make_env_from_panel(df, rebalance_interval=5, max_weight=0.1, allow_cash=True, cash_rate_as_rf=True)
    obs, info = env.reset()
    
    print(f"Action space: {env.action_space}")
    print(f"Max blocks per asset: {env.max_blocks_per_asset}")
    
    for step in range(12):
        # Discrete action: allocate blocks [5, 3, 2] for 3 assets
        action = np.array([5, 3, 2], dtype=int) if env.n_assets == 3 else np.ones(env.n_assets, dtype=int)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            break
    
    print("OK:", info["date"].date(), "wealth:", round(info["wealth"], 6))