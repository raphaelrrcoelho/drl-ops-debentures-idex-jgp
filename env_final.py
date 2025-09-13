# env_final_merged.py
"""
Debenture portfolio environment for PPO (Online Portfolio Selection)
-------------------------------------------------------------------
This merged version starts from your original env_final.py and **integrates**
the requested changes from env_new.py with minimal, surgical edits:

1) **Reward = alpha vs. index** (r_p - r_index), minus penalties & costs.
   - We still log `excess` for diagnostics, but it does not enter the reward.
2) **Index weights are NOT passed to the agent** (excluded from observations).
3) **Sector features supported** (lagged at t-1, no look-ahead):
   - 'sector_spread', 'sector_momentum', 'sector_weight_index'
4) Observations include only **lagged** features; we also append lagged
   global scalars (rf_{t-1}, idx_{t-1}) for stability.
5) Keep all original mechanics: dynamic investable set, per-asset caps,
   transaction costs, delist slippage, concentration and drawdown penalties,
   optional tail penalty (off by default), and optional synthetic CASH asset.

Intended usage (same as before):
    env = make_env_from_panel(panel, rebalance_interval=5, max_weight=0.10, cash_rate_as_rf=True)
    # Use with stable-baselines3 PPO(MlpPolicy, ...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from typing import Optional

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
    allow_cash: bool = True                     # if True, append cash asset
    cash_rate_as_rf: bool = True                # if cash exists, accrues at rf
    on_inactive: str = "to_cash"                # {"to_cash","pro_rata"}

    # Costs & penalties
    weight_excess: float = 0.0   # keep default = 0.0
    weight_alpha:  float = 1.0   # keep default = 1.0 (pure alpha, matches your current behavior)
    transaction_cost_bps: float = 20.0           # linear cost per unit turnover
    delist_extra_bps: float = 20.0              # extra slippage on forced sells
    lambda_turnover: float = 0.0002             # turnover penalty weight
    lambda_hhi: float = 0.01                    # concentration penalty weight
    lambda_drawdown: float = 0.005              # drawdown penalty weight
    lambda_tail: float = 0.0                    # tail risk penalty weight (off by default)
    tail_window: int = 60                       # rolling window for tail quantile
    tail_q: float = 0.05                        # tail threshold
    dd_mode: str = "incremental"                # 'level' or 'incremental'

    # Observation controls
    include_prev_weights: bool = True           # append previous weights in obs
    include_active_flag: bool = True            # keep 'active' as feature
    global_stats: bool = True                   # add cross-section stats (mean/std)
    normalize_features: bool = True             # z-normalize features per-day
    obs_clip: float = 5.0                       # clip features after zscore

    # Episode control
    max_steps: Optional[int] = None             # optional truncate
    seed: Optional[int] = 42                    # RNG seed
    random_reset_frac: float = 0.0              # in [0,1]; if >0, reset() samples a random start
    # (e.g., 0.9 means we can start anywhere in the first 90% of the window)
    # so that max_steps steps still fit within T


# ------------------------------ Utilities -------------------------------- #

def _project_to_simplex(a: np.ndarray, temp: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """Softmax projection with numerical stability (sum=1, non-neg)."""
    x = np.asarray(a, dtype=np.float64) / max(temp, 1e-8)
    x -= np.max(x)
    e = np.exp(x)
    s = float(e.sum())
    if s <= eps:
        return np.ones_like(x, dtype=np.float32) / x.size
    return (e / s).astype(np.float32)


def _cap_and_renormalize(w: np.ndarray, cap: float, mask: Optional[np.ndarray]) -> np.ndarray:
    """Apply inactivity mask + per-asset cap, then renormalize to sum=1 over active."""
    ww = np.maximum(w, 0.0)
    if mask is not None:
        ww = ww * (mask > 0).astype(np.float32)
    s = float(ww.sum())
    if s <= 1e-12:
        if mask is not None and mask.sum() > 0:
            ww = (mask > 0).astype(np.float32)
            return ww / float(ww.sum())
        return np.ones_like(ww) / ww.size
    ww /= s

    # iterative clipping
    for _ in range(20):
        over = ww > cap
        if not np.any(over):
            break
        excess = float(np.maximum(ww[over] - cap, 0.0).sum())
        ww[over] = cap
        under = ~over
        if mask is not None:
            under = under & (mask > 0)
        pool = float(ww[under].sum())
        if pool > 1e-12:
            ww[under] += excess * (ww[under] / pool)
        else:
            # no room: uniform over actives
            if mask is not None and mask.sum() > 0:
                ww = (mask > 0).astype(np.float32)
                return ww / float(ww.sum())
            return np.ones_like(ww) / ww.size
    s = float(ww.sum())
    return ww / (s if s > 1e-12 else 1.0)


def _sanitize_hold(w_prev: np.ndarray, cap: float, mask: np.ndarray,
                   on_inactive: str = "to_cash",
                   cash_idx: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Force weights off inactives; send freed mass to cash or pro-rata over actives."""
    w = np.copy(w_prev)
    inactive = (mask <= 0)
    freed = float(np.maximum(w[inactive], 0.0).sum())
    if freed > 0.0:
        w[inactive] = 0.0
        if on_inactive == "to_cash" and cash_idx is not None:
            w[cash_idx] = float(w[cash_idx] + freed)
        else:
            active = (mask > 0)
            s = float(w[active].sum())
            if s > 0.0:
                w[active] += (w[active] / s) * freed
            elif cash_idx is not None:
                w[cash_idx] = float(w[cash_idx] + freed)
    w = _cap_and_renormalize(w, cap, mask)
    return w, freed


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
            self.cash_idx = self.n_assets - 1  # last column reserved for __CASH__

        # Spaces
        n_assets = self.n_assets
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32)
        obs_dim = self._obs_size()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            'action_mask': spaces.Box(0, 1, shape=(self.n_assets,), dtype=np.int8)
        })

        # State
        self.t: int = 0
        self.prev_w: np.ndarray = np.zeros(n_assets, dtype=np.float32)
        self.curr_w: np.ndarray = np.zeros(n_assets, dtype=np.float32)
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

        # Base features (we will consume their lagged versions when applicable)
        base_feats = [
            "return", "spread", "duration", "time_to_maturity",
            "risk_free", "index_return", "ttm_rank",
            # ANBIMA-style curve diagnostics if present
            "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
            "sector_fitted_spread", "spread_residual_ns",
            "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
            # NEW sector signals (must be provided by data_final.py)
            "sector_spread", "sector_momentum", "sector_weight_index",
            # contemporaneous descriptor
            "sector_id",
        ]
        # Include ACTIVE flag as a feature known at day start
        if self.cfg.include_active_flag and "active" in panel.columns:
            base_feats.append("active")

        panel = panel.copy()

        def ensure_lag1(col: str) -> str:
            # Never lag 'sector_id' and (optionally) 'active'
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

        # Drop any index_weight* columns from features (explicitly *not* passing them)
        feat_cols = [c for c in feat_cols if "index_weight" not in c]

        # Include any precomputed z-scores (only their lagged versions)
        z_like = [c for c in panel.columns if (c.endswith("_z") or c.endswith("_z252") or "_z_" in c)]
        for zc in sorted(z_like):
            lagc = ensure_lag1(zc)
            if lagc not in feat_cols:
                feat_cols.append(lagc)

        # Guarantee existence
        for c in feat_cols:
            if c not in panel.columns:
                panel[c] = 0.0

        # Dates and union of asset IDs
        dates = panel.index.get_level_values("date").unique().sort_values()
        self.dates = dates.to_numpy()
        asset_ids = panel.index.get_level_values("debenture_id").unique().tolist()
        self.asset_ids: List[str] = list(asset_ids)
        self.n_assets = len(self.asset_ids)

        # Reward arrays (same-day)
        R = (
            panel["return"].reset_index()
                 .pivot(index="date", columns="debenture_id", values="return")
                 .reindex(index=dates, columns=asset_ids)
                 #.fillna(0.0)
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

        # Investable mask (same-day)
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

        # Sector IDs (descriptor)
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

        # Per-day cross-sectional stats (computed on lagged features)
        if self.cfg.global_stats and self.F > 0:
            act = (self.ACT > 0).astype(np.float32)
            denom = np.maximum(act.sum(axis=1, keepdims=True), 1.0)
            means = (self.X * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None]
            stds = np.sqrt(((self.X - means) ** 2 * act[..., None]).sum(axis=1, keepdims=True) / denom[..., None])
            self.global_means = means.squeeze(1)  # [T,F]
            self.global_stds = np.maximum(stds.squeeze(1), 1e-6)  # avoid div-by-zero
        else:
            self.global_means = None
            self.global_stds = None

        # Optional daily z-normalization (on already-lagged features)
        if self.cfg.normalize_features and self.F > 0:
            Xn = self.X.copy()
            eps = 1e-6
            for f in range(self.F):
                mu = self.global_means[:, f][:, None] if self.global_means is not None else 0.0
                sd = self.global_stds[:, f][:, None] if self.global_stds is not None else 1.0
                Xn[:, :, f] = (Xn[:, :, f] - mu) / (sd + eps)
            self.X = np.clip(Xn, -self.cfg.obs_clip, self.cfg.obs_clip)

        # Build lagged RF/IDX for OBSERVATION global scalars
        self.RF_obs = np.zeros_like(self.RF, dtype=np.float32)
        self.RF_obs[1:] = self.RF[:-1]
        self.IDX_obs = np.zeros_like(self.IDX, dtype=np.float32)
        self.IDX_obs[1:] = self.IDX[:-1]

        # Append cash if enabled
        if self.cfg.allow_cash:
            # Extend arrays by 1 column for cash
            if self.cfg.cash_rate_as_rf:
                cash_R = self.RF.reshape(-1, 1)  # reward accrues at rf (same-day)
            else:
                cash_R = np.zeros((self.T, 1), dtype=np.float32)
            cash_X = np.zeros((self.T, 1, self.F), dtype=np.float32)  # features for cash = 0
            cash_A = np.ones((self.T, 1), dtype=np.float32)           # cash always active
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
            extra += 2 * self.F  # mean & std
        # Two global scalars (lagged): rf and idx
        extra += 2
        return base + extra

    def _get_observation(self) -> np.ndarray:
        t = min(self.t, self.T - 1)
        if t >= self.ACT.shape[0]:
            print(f"[WARN] t={t} exceeds ACT.shape[0]={self.ACT.shape[0]}, clamping to last available")
            t = self.ACT.shape[0] - 1
        parts: List[np.ndarray] = []
        if self.F > 0:
            X_t = self.X[t].copy()  # [N,F]
            if self.cfg.normalize_features and self.global_means is not None:
                # Already normalized in _prepare_data; nothing else required
                pass
            parts.append(X_t.reshape(-1))
        if self.cfg.include_prev_weights:
            parts.append(self.prev_w.astype(np.float32).ravel())
        if self.cfg.include_active_flag:
            parts.append(self.ACT[t].astype(np.float32).ravel())
        if self.cfg.global_stats and self.F > 0:
            parts.append(self.global_means[t].astype(np.float32).ravel())
            parts.append(self.global_stds[t].astype(np.float32).ravel())
        # Always append lagged rf and idx as 2D global scalars
        parts.append(np.array([self.RF_obs[t], self.IDX_obs[t]], dtype=np.float32).ravel())
        obs = np.concatenate(parts) if parts else np.zeros((self._obs_size(),), dtype=np.float32)
        clip = float(self.cfg.obs_clip) if self.cfg.obs_clip is not None else 10.0
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        mask = self.ACT[self.t].astype(np.int8)
        
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
            # choose a feasible start so that the episode fits (max_steps)
            max_start = max(0, +self.T - int(self.cfg.max_steps))
            jitter_cap = int(max_start * float(self.cfg.random_reset_frac))
            start = int(self.np_random.integers(0, jitter_cap + 1)) if jitter_cap > 0 else 0
            self.t = start
        else:
            self.t = 0

        # IMPORTANT: initialize using ACTIVE at the chosen start, not always ACT[0]
        mask0 = (self.ACT[self.t] > 0)
        self.prev_w = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w = np.zeros(self.n_assets, dtype=np.float32)
        self.curr_w[mask0] = 0.0  # start from cash; per your rules cash accrues rf if enabled
        self.wealth = 1.0
        # ... remainder unchanged (build obs with self.t, etc.) ...
        info = {"date": pd.Timestamp(self.dates[self.t]).to_pydatetime()}
        return self._get_observation(), info

    def step(self, action: np.ndarray):
        # Time index guard
        t = int(self.t)
        terminated = False
        truncated = False
        if t >= self.T or t >= self.ACT.shape[0]:
            # Not enough data, terminate episode
            print(f"[WARN] Episode terminated early: t={t}, T={self.T}, ACT.shape={self.ACT.shape}")
            return self._get_observation(), 0.0, True, False, {}

        # Day's arrays
        act_mask = self.ACT[t]
        r_vec = self.R[t].copy()

        # Rebalance decision only every 'rebalance_interval'
        apply_action = (t % max(1, int(self.cfg.rebalance_interval)) == 0)
        w_prev = self.curr_w.copy()
        freed_mass = 0.0
        extra_delist_cost = 0.0

        if apply_action:
            # Project raw logits to simplex, apply masks and caps
            w_tgt = _project_to_simplex(np.asarray(action, dtype=np.float32))
            w_tgt = _cap_and_renormalize(w_tgt, float(self.cfg.max_weight), act_mask)
        else:
            # Hold, but sanitize inactives
            w_tgt, freed_mass = _sanitize_hold(w_prev, float(self.cfg.max_weight), act_mask,
                                               on_inactive=self.cfg.on_inactive, cash_idx=self.cash_idx)
            extra_delist_cost = freed_mass * (self.cfg.delist_extra_bps / 10000.0)

        # Turnover & linear cost
        turn = _turnover(w_tgt, w_prev)
        lin_cost = (self.cfg.transaction_cost_bps / 10000.0) * turn + extra_delist_cost

        # Realized portfolio return (before cost)
        r_p = float(np.dot(w_tgt, r_vec))
        # Index return and alpha
        r_idx = float(self.IDX[t])
        alpha = r_p - r_idx
        rf_t = float(self.RF[t])
        # Missing quote? Treat as cash accrual for that day
        bad = ~np.isfinite(r_vec)
        if bad.any():
            r_vec[bad] = rf_t
        r_p = float(np.dot(w_tgt, r_vec))
        excess = r_p - rf_t  # diagnostic only

        # Wealth update
        net = max((1.0 + r_p) * (1.0 - max(lin_cost, 0.0)), 1e-12)
        r_net = net - 1.0
        self.wealth *= net
        self.peak_wealth = max(self.peak_wealth, self.wealth)

        # Drawdown penalty (incremental or level)
        cur_dd_level = 1.0 - (self.wealth / max(self.peak_wealth, 1e-12))
        if self.cfg.dd_mode == "level":
            dd_pen = -self.cfg.lambda_drawdown * abs(cur_dd_level)
        else:
            prev_dd_level = self._history["drawdown"][-1] if self._history.get("drawdown") else 0.0
            dd_inc = max(cur_dd_level - prev_dd_level, 0.0)
            dd_pen = -self.cfg.lambda_drawdown * dd_inc

        # Tail penalty (optional, off by default)
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

        # Reward: **alpha vs index** minus penalties and trading costs proxy
        reward = float(alpha + pen - lin_cost)
        reward = float(self.cfg.weight_alpha * alpha + self.cfg.weight_excess * excess + pen - lin_cost)


        # Advance time and commit weights
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
            "idx": float(r_idx),
            "rf": float(rf_t),
            "turnover": float(turn),
            "hhi": float(hhi_val),
            "trade_cost": float(lin_cost),
            "wealth": float(self.wealth),
            "drawdown": float(cur_dd_level),
            "sector_exposure": _dict_sector_exposures(self.sector_ids, self.curr_w),
            "config": asdict(self.cfg) if self.t == 1 else None,  # emitted once
        }
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
        return pd.DataFrame(self._history, index=pd.to_datetime(self.dates[:len(self._history.get('wealth', []))]))

    def get_asset_ids(self) -> List[str]:
        return list(self.asset_ids)


# ------------------------- Factory convenience ---------------------------- #

def make_env_from_panel(panel: pd.DataFrame, **env_kwargs) -> DebentureTradingEnv:
    """Factory: build DebentureTradingEnv from a panel and EnvConfig kwargs."""
    cfg = EnvConfig(**env_kwargs)
    return DebentureTradingEnv(panel=panel, config=cfg)


# ------------------------------ Quick test -------------------------------- #

if __name__ == "__main__":
    # Minimal smoke test using synthetic data if run directly
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
    # NEW sector signals
    df["sector_spread"] = 0.03
    df["sector_momentum"] = 0.0
    df["sector_weight_index"] = 0.5
    # mimic lag columns for test
    for c in ["return","spread","duration","time_to_maturity","risk_free","index_return",
              "sector_ns_beta0","ns_beta1_common","ns_lambda_common","sector_fitted_spread",
              "spread_residual_ns","sector_ns_level_1y","sector_ns_level_3y","sector_ns_level_5y",
              "sector_spread","sector_momentum","sector_weight_index"]:
        if c not in df.columns:
            df[c] = 0.0
        df[f"{c}_lag1"] = df[c]

    env = make_env_from_panel(df, rebalance_interval=5, max_weight=0.1, allow_cash=True, cash_rate_as_rf=True)
    obs, info = env.reset()
    for step in range(12):
        a = np.array([0.6, 0.4, 0.0], dtype=np.float32) if env.n_assets == 3 else np.ones(env.n_assets, dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    print("OK:", info["date"].date(), "wealth:", round(info["wealth"], 6))
