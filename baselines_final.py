# baselines_final.py
"""
Dynamic-universe baselines for debenture portfolio OPS - CORRECTED VERSION
---------------------------------------------------------------------------

CRITICAL FIX: Returns in the panel are now TOTAL returns (spread + CDI)
- Individual bond returns already include risk-free
- Index returns also include risk-free
- No need to add RF separately in portfolio calculations

For each walk-forward fold:
  • Load TRAIN→TEST union of IDs used at training (from training_union_ids.json);
    if missing, recompute union(train_start → test_end).
  • Build TEST panel reindexed to (test_dates × union_ids) with ACTIVE gating.
  • Simulate baseline strategies with the same rebalancing / costs / caps
    and dynamic (day-by-day) investability as the PPO env.
  • Append results to results/<universe>:
      - all_returns.csv            (long: date, strategy, return)
      - all_diagnostics.csv        (long: date, strategy, metric, value)
      - fold_metrics.csv           (per fold row)
      - all_metrics.csv            (per fold row)

Baselines implemented (dynamic universe aware), using **lagged inputs**:
  - EW                  : Equal-weight over ACTIVE names (capped)
  - INDEX               : Uses prior-day 'index_weight' if available; else EW
  - RP_VOL              : Risk parity by inverse rolling vol (uses returns up to t-1)
  - RP_DURATION         : Inverse prior-day 'duration' (1 / duration_{t-1})
  - CARRY_TILT          : Proportional to positive prior-day 'spread' (carry proxy)
  - MINVAR              : Min-variance proxy via Σ^{-1}·1 with ridge & non-negativity (Σ from returns up to t-1)

Conventions
-----------
• **Decisions at start of t** use only info up to t−1 (lagged signals).
• **Rewards at t** use realized TOTAL returns of t (minus costs), with cash at rf_t.
• Between rebalances, newly inactive positions are liquidated to cash (configurable).

Usage
-----
python baselines_final.py --universe cdi --strategies EW,INDEX,RP_VOL,RP_DURATION,CARRY_TILT,MINVAR \
    --rebalance_interval 10 --max_weight 0.10 --transaction_cost_bps 20 \
    --roll_window 60 --allow_cash --on_inactive to_cash
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0

# ------------------------------ I/O helpers ------------------------------ #

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def append_row_csv(path: str, row: dict, cols_order: Optional[List[str]] = None):
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    if cols_order:
        for c in cols_order:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols_order]
    df.to_csv(path, index=False)

def append_long_csv(path: str, df_new: pd.DataFrame, sort_by: Optional[List[str]] = None):
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        for c in df_new.columns:
            if c not in df.columns:
                df[c] = np.nan
        df = pd.concat([df, df_new], ignore_index=True, sort=False)
    else:
        df = df_new
    if sort_by:
        df = df.sort_values(sort_by)
    df.to_csv(path, index=False)

# --------------------------- Panel preparation --------------------------- #

REQUIRED_COLS = [
    "return", "spread", "duration", "sector_id", "active",
    "risk_free", "index_return", "time_to_maturity", "index_level", "index_weight"
]
SAFE_FILL_0 = ["return", "spread", "duration", "time_to_maturity", "ttm_rank"]
DATE_LEVEL = ["risk_free", "index_return", "index_level"]

def _date_level_map(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(dtype=float)
    return panel[[col]].groupby(level="date").first()[col].sort_index()

def union_ids_from_training(results_dir: str, fold: int) -> Optional[List[str]]:
    p = os.path.join(results_dir, "training_union_ids.json")
    if not os.path.exists(p):
        return None
    data = read_json(p)
    for rec in data:
        if int(rec.get("fold", -1)) == int(fold):
            return [str(x) for x in rec.get("ids", [])]
    return None

def compute_union_ids_from_panel(panel: pd.DataFrame, fold_cfg: Dict[str, str],
                                 max_assets: int = 50, rebalance_interval: int = 10) -> List[str]:
    """
    Compute union IDs using top-K logic matching training.
    """
    train_start = pd.to_datetime(fold_cfg["train_start"])
    test_end = pd.to_datetime(fold_cfg["test_end"])
    
    # Get all dates in train+test period
    dates = panel.index.get_level_values("date").unique()
    period_dates = dates[(dates >= train_start) & (dates <= test_end)].sort_values()
    
    # Identify rebalance dates
    rb_indices = np.arange(0, len(period_dates), max(1, rebalance_interval))
    rb_dates = period_dates[rb_indices]
    
    # Collect assets that are ever top-K
    ever_topk = set()
    
    for d in rb_dates:
        try:
            rows = panel.xs(d, level="date", drop_level=False)
        except KeyError:
            continue
        
        # Only active assets
        rows = rows[rows["active"] > 0]
        if rows.empty:
            continue
        
        # Get top-K by index_weight
        if "index_weight" in rows.columns:
            weights = rows["index_weight"].values
            if len(weights) > max_assets:
                top_idx = np.argpartition(-weights, max_assets-1)[:max_assets]
                sel_ids = rows.iloc[top_idx].index.get_level_values("debenture_id")
            else:
                sel_ids = rows.index.get_level_values("debenture_id")
            ever_topk.update(sel_ids.tolist())
    
    if not ever_topk:
        # Fallback: use all assets seen in the period
        sl = panel.loc[
            (panel.index.get_level_values("date") >= train_start) &
            (panel.index.get_level_values("date") <= test_end)
        ]
        ever_topk = set(sl.index.get_level_values("debenture_id").unique().tolist())
    
    return sorted(ever_topk)

def build_eval_panel_with_union(panel: pd.DataFrame, fold_cfg: Dict[str, str], union_ids: List[str]) -> pd.DataFrame:
    test_start = pd.to_datetime(fold_cfg["test_start"])
    test_end   = pd.to_datetime(fold_cfg["test_end"])
    dates = panel.index.get_level_values("date").unique().sort_values()
    test_dates = dates[(dates >= test_start) & (dates <= test_end)]

    idx = pd.MultiIndex.from_product([test_dates, union_ids], names=["date", "debenture_id"])
    test_aug = panel.reindex(idx).sort_index()

    for c in REQUIRED_COLS:
        if c not in test_aug.columns:
            test_aug[c] = np.nan

    # Broadcast date-level fields
    for name in DATE_LEVEL:
        m = _date_level_map(panel, name)
        if not m.empty:
            df = test_aug.reset_index()
            df[name] = df[name].astype(float).fillna(df["date"].map(m).astype(float))
            test_aug = df.set_index(["date", "debenture_id"]).sort_index()

    # Safe fills
    test_aug["active"] = test_aug["active"].fillna(0).astype(np.int8)
    for c in SAFE_FILL_0:
        if c in test_aug.columns:
            r = test_aug["return"]
            a = test_aug["active"]
            # CRITICAL: Returns are already TOTAL (spread + CDI), keep them as-is
            test_aug["return"] = np.where(a.astype(bool), r, 0.0)
    
    if "sector_id" in test_aug.columns:
        test_aug["sector_id"] = test_aug["sector_id"].fillna(-1).astype(np.int16)
    
    if "index_weight" in test_aug.columns:
        test_aug["index_weight"] = test_aug["index_weight"].fillna(0.0).astype(np.float32)
    
    for opt in ["rv_residual","sector_curve_slope","sector_curve_curv","ttm_rank"]:
        if opt in test_aug.columns:
            test_aug[opt] = test_aug[opt].fillna(0.0)

    if "index_level" in test_aug.columns:
        test_aug["index_level"] = test_aug["index_level"].astype(float).ffill()
        test_aug["index_level"] = test_aug["index_level"].fillna(0.0)

    for name in ["risk_free", "index_return"]:
        test_aug[name] = test_aug[name].astype(np.float32).fillna(0.0)
    
    # Add return components if available for diagnostics
    for col in ["mtm_return", "carry_return", "spread_return", "total_return"]:
        if col in panel.columns:
            m = panel[[col]].reindex(test_aug.index)
            test_aug[col] = m[col].astype(np.float32).fillna(0.0)
    
    return test_aug

# ------------------------------ Metrics ---------------------------------- #

def max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())

def annualize_mean(mu_daily: float, periods: int = int(TRADING_DAYS)) -> float:
    return (1.0 + mu_daily)**periods - 1.0

def annualize_vol(vol_daily: float, periods: int = int(TRADING_DAYS)) -> float:
    return vol_daily * np.sqrt(periods)

def compute_metrics(returns: pd.Series, rf: pd.Series, bench: pd.Series,
                    periods: int = int(TRADING_DAYS)) -> Dict[str, float]:
    """
    Calculate performance metrics.
    NOTE: returns are already TOTAL returns (including RF)
    """
    r = returns.fillna(0.0)
    r_excess = (r - rf).fillna(0.0)
    r_rel = (r - bench).fillna(0.0)

    mu = r.mean()
    sd = r.std(ddof=1)
    mu_ex = r_excess.mean()
    sd_ex = r_excess.std(ddof=1)
    downside = r_excess[r_excess < 0].std(ddof=1)

    sharpe_d = (mu_ex / sd_ex) if sd_ex > 0 else np.nan
    sortino_d = (mu_ex / downside) if downside and downside > 0 else np.nan

    wealth = (1.0 + r).cumprod()
    mdd = max_drawdown(wealth)
    n = max(1, len(r))
    cagr = float(wealth.iloc[-1] ** (periods / n) - 1.0)
    vol_ann = annualize_vol(sd, periods=periods)
    sharpe_ann = sharpe_d * np.sqrt(periods) if sharpe_d == sharpe_d else np.nan
    sortino_ann = sortino_d * np.sqrt(periods) if sortino_d == sortino_d else np.nan
    calmar = (cagr / abs(mdd)) if mdd < 0 else np.nan

    cov = np.cov(r.dropna(), bench.reindex_like(r).dropna())[0, 1] if len(r.dropna()) and len(bench.dropna()) else np.nan
    varb = bench.var(ddof=1)
    beta = (cov / varb) if varb and varb > 0 else np.nan
    alpha = (mu - bench.mean() * (beta if beta == beta else 0.0))

    ir_den = r_rel.std(ddof=1)
    information_ratio = (r_rel.mean() / ir_den) if ir_den and ir_den > 0 else np.nan

    pos = bench > 0
    neg = bench < 0
    up_capture = (r[pos].mean() / bench[pos].mean()) if pos.any() and bench[pos].mean() != 0 else np.nan
    down_capture = (r[neg].mean() / bench[neg].mean()) if neg.any() and bench[neg].mean() != 0 else np.nan

    hit_rate = (r > 0).mean()
    skew = r.skew()
    kurt = r.kurt()

    return {
        "CAGR": float(cagr),
        "Vol_ann": float(vol_ann),
        "Sharpe": float(sharpe_ann),
        "Sortino": float(sortino_ann),
        "MaxDD": float(mdd),
        "Calmar": float(calmar),
        "Alpha_daily": float(alpha),
        "Beta": float(beta) if beta == beta else np.nan,
        "Information_Ratio": float(information_ratio),
        "Up_capture": float(up_capture),
        "Down_capture": float(down_capture),
        "Hit_rate": float(hit_rate),
        "Skew": float(skew),
        "Kurtosis": float(kurt),
    }

# ---------------------------- Portfolio math ----------------------------- #

def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v, 0.0).astype(np.float64, copy=False)
    if v.sum() <= 0:
        return np.full_like(v, 1.0 / v.size)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, v.size + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.full_like(v, 1.0 / v.size)
    rho = rho[-1]
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return (w / s) if s > 0 else np.full_like(v, 1.0 / v.size)

def _cap_and_mask(w: np.ndarray, cap: float, mask: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0.0)
    w = w * mask
    if w.sum() <= 0:
        if mask.sum() <= 0:
            return w
        return (mask / mask.sum()).astype(np.float64)
    w = w / w.sum()
    for _ in range(10):
        over = w > cap
        if not np.any(over):
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over & (mask > 0)
        free = w[under].sum()
        if free <= 1e-12:
            break
        w[under] += w[under] / max(free, 1e-12) * excess
    s = w.sum()
    return w / s if s > 0 else w

def _sanitize_hold_to_cash(w_prev: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Zero out newly inactive; DO NOT renormalize (freed mass goes to cash)."""
    w = w_prev.copy()
    inactive = mask <= 0
    freed = float(w[inactive].sum())
    w[inactive] = 0.0
    return w, freed

def _turnover_total_with_cash(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    cash_prev = 1.0 - float(w_prev.sum())
    cash_new = 1.0 - float(w_new.sum())
    v_prev = np.concatenate([w_prev, [cash_prev]])
    v_new  = np.concatenate([w_new,  [cash_new]])
    return float(np.abs(v_new - v_prev).sum())

def _hhi(w: np.ndarray) -> float:
    return float((w * w).sum())

# ------------------------- Strategy weight rules ------------------------- #

def weights_EW(mask: np.ndarray, cap: float) -> np.ndarray:
    w = mask.astype(np.float64)
    if w.sum() > 0:
        w = w / w.sum()
    return _cap_and_mask(w, cap, mask)

def weights_INDEX(mask: np.ndarray, idx_w_row: Optional[np.ndarray], cap: float) -> np.ndarray:
    if idx_w_row is None or np.all(idx_w_row == 0):
        return weights_EW(mask, cap)
    w = np.maximum(idx_w_row, 0.0) * mask
    if w.sum() > 0:
        w = w / w.sum()
    return _cap_and_mask(w, cap, mask)

def weights_RP_VOL(mask: np.ndarray, ret_hist: np.ndarray, cap: float) -> np.ndarray:
    # ret_hist: [window, N] with NaNs for missing; already up to t-1
    # NOTE: Using TOTAL returns for volatility calculation
    if ret_hist.size == 0 or ret_hist.shape[0] <= 1:
        return weights_EW(mask, cap)
    vol = np.nanstd(ret_hist, axis=0, ddof=1)
    vol = np.where(vol <= 1e-8, np.nan, vol)
    inv = 1.0 / vol
    inv = np.nan_to_num(inv, nan=0.0)
    w = inv * mask
    if w.sum() <= 0:
        return weights_EW(mask, cap)
    w = w / w.sum()
    return _cap_and_mask(w, cap, mask)

def weights_RP_DURATION(mask: np.ndarray, duration_row_lag1: np.ndarray, cap: float) -> np.ndarray:
    dur = np.where(duration_row_lag1 <= 0, np.nan, duration_row_lag1)
    inv = 1.0 / dur
    inv = np.nan_to_num(inv, nan=0.0)
    w = inv * mask
    if w.sum() <= 0:
        return weights_EW(mask, cap)
    w = w / w.sum()
    return _cap_and_mask(w, cap, mask)

def weights_CARRY_TILT(mask: np.ndarray, spread_row_lag1: np.ndarray, cap: float) -> np.ndarray:
    s = np.maximum(spread_row_lag1, 0.0)
    w = s * mask
    if w.sum() <= 0:
        return weights_EW(mask, cap)
    w = w / w.sum()
    return _cap_and_mask(w, cap, mask)

def weights_MINVAR(mask: np.ndarray, cov: np.ndarray, ridge: float, cap: float) -> np.ndarray:
    N = cov.shape[0]
    cov_r = cov + np.eye(N) * ridge
    try:
        inv = np.linalg.pinv(cov_r)
    except Exception:
        inv = np.linalg.pinv(cov_r + np.eye(N) * (10 * ridge))
    ones = np.ones(N)
    raw = inv.dot(ones)
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0:
        return weights_EW(mask, cap)
    w = raw * mask
    if w.sum() <= 0:
        return weights_EW(mask, cap)
    w = w / w.sum()
    return _cap_and_mask(w, cap, mask)

# -------------------------- Baseline simulation -------------------------- #

class BaselineConfig:
    def __init__(self,
                 rebalance_interval: int = 10,
                 max_weight: float = 0.10,
                 transaction_cost_bps: float = 20.0,
                 allow_cash: bool = True,
                 on_inactive: str = "to_cash",
                 roll_window: int = 60,
                 ridge: float = 1e-4,
                 delist_extra_bps: float = 20.0,
                 max_assets: int = 50):
        self.rebalance_interval = int(rebalance_interval)
        self.max_weight = float(max_weight)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.allow_cash = bool(allow_cash)
        self.on_inactive = str(on_inactive)
        self.roll_window = int(roll_window)
        self.ridge = float(ridge)
        self.delist_extra_bps = float(delist_extra_bps)
        self.max_assets = int(max_assets)

def simulate_baseline(panel: pd.DataFrame, strategy: str, cfg: BaselineConfig) -> Dict[str, pd.DataFrame]:
    """
    panel: TEST×UNION panel (ACTIVE already 0 for missing asset-days).
    Returns dict with 'returns','diagnostics','weights' DataFrames.

    CRITICAL: Returns in panel are TOTAL returns (spread + CDI).
    Inputs/signals are lagged (up to t-1). Rewards are contemporaneous (t).
    """
    # Wide arrays
    dates = panel.index.get_level_values("date").unique().sort_values()
    asset_ids = panel.index.get_level_values("debenture_id").unique().astype(str).tolist()
    N = len(asset_ids)
    T = len(dates)

    def wide(col, fill=0.0):
        return (panel[col].reset_index()
                .pivot(index="date", columns="debenture_id", values=col)
                .reindex(index=dates, columns=asset_ids)
                .fillna(fill)
                .to_numpy(dtype=np.float64))

    # CRITICAL: R contains TOTAL returns (spread + CDI)
    R   = wide("return", np.nan)           # [T,N] realized day t (used for rewards)
    ACT = wide("active", 0.0)              # [T,N] investability at t
    DUR = wide("duration", 0.0)            # [T,N] EOD duration (we will lag)
    SPR = wide("spread", 0.0)              # [T,N] EOD spreads (we will lag)
    idxW = wide("index_weight", 0.0) if "index_weight" in panel.columns else None

    # Lag signals to enforce causality (use info up to t-1 for decisions at t)
    def lag1(arr: np.ndarray) -> np.ndarray:
        return np.vstack([np.zeros((1, arr.shape[1])), arr[:-1]])

    DUR_L = lag1(DUR)
    SPR_L = lag1(SPR)
    idxW_L = lag1(idxW) if idxW is not None else None

    # Date-level
    # NOTE: RF and IDX are already included in individual returns
    RF  = panel[["risk_free"]].groupby(level="date").first().reindex(dates)["risk_free"].fillna(0.0).to_numpy(dtype=np.float64)
    IDX = panel[["index_return"]].groupby(level="date").first().reindex(dates)["index_return"].fillna(0.0).to_numpy(dtype=np.float64)

    # State
    w = np.zeros(N, dtype=np.float64)    # asset weights (cash implicit)
    wealth = 1.0
    peak = 1.0

    ret_rows, diag_rows, w_rows = [], [], []

    # Helper to compute target weights at a rebalance time (uses lagged inputs)
    def target_weights(t: int) -> np.ndarray:
        mask = (ACT[t] > 0).astype(np.float64)
        
        # Apply top-K constraint if applicable
        if cfg.max_assets < N:
            # Get index weights to determine top-K
            idx_weights_t = idxW_L[t] if idxW_L is not None else np.ones(N) / N
            active_idx = np.where(mask > 0)[0]
            if len(active_idx) > cfg.max_assets:
                # Select top K by index weight
                active_weights = idx_weights_t[active_idx]
                top_k_in_active = np.argpartition(-active_weights, cfg.max_assets-1)[:cfg.max_assets]
                selected = active_idx[top_k_in_active]
                # Zero out non-selected
                new_mask = np.zeros_like(mask)
                new_mask[selected] = 1.0
                mask = new_mask
        
        if strategy == "EW":
            wt = weights_EW(mask, cfg.max_weight)
        elif strategy == "INDEX":
            wt = weights_INDEX(mask, idxW_L[t] if idxW_L is not None else None, cfg.max_weight)
        elif strategy == "RP_VOL":
            t0 = max(0, t - cfg.roll_window)
            ret_hist = R[t0:t] if t0 < t else np.full((1, N), np.nan)  # up to t-1
            wt = weights_RP_VOL(mask, ret_hist, cfg.max_weight)
        elif strategy == "RP_DURATION":
            wt = weights_RP_DURATION(mask, DUR_L[t], cfg.max_weight)
        elif strategy == "CARRY_TILT":
            wt = weights_CARRY_TILT(mask, SPR_L[t], cfg.max_weight)
        elif strategy == "MINVAR":
            t0 = max(0, t - cfg.roll_window)
            rh = R[t0:t]  # up to t-1, TOTAL returns
            if rh.shape[0] < 2:
                wt = weights_EW(mask, cfg.max_weight)
            else:
                rh_clean = rh.copy()
                rh_clean[:, mask <= 0] = 0.0
                cov = np.cov(rh_clean, rowvar=False) if rh_clean.shape[0] > 1 else np.eye(N) * 1e-4
                wt = weights_MINVAR(mask, cov, cfg.ridge, cfg.max_weight)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")
        return wt

    for t in range(T):
        mask = (ACT[t] > 0).astype(np.float64)
        rf_t, idx_t = RF[t], IDX[t]

        if (t % cfg.rebalance_interval) == 0:
            w_new = target_weights(t)
            freed_mass = 0.0
        else:
            # sanitize holds between rebalances
            if cfg.on_inactive == "to_cash":
                w_new, freed_mass = _sanitize_hold_to_cash(w, mask)
            else:  # "pro_rata"
                w_tmp, freed_mass = _sanitize_hold_to_cash(w, mask)
                active_sum = w_tmp.sum()
                if active_sum > 0 and freed_mass > 0:
                    w_new = w_tmp + (w_tmp / active_sum) * freed_mass
                else:
                    w_new = w_tmp

        # Costs on total turnover (including cash leg)
        turn_total = _turnover_total_with_cash(w_new, w)
        costs = (cfg.transaction_cost_bps / 10_000.0) * turn_total if cfg.transaction_cost_bps > 0 else 0.0

        # Extra slippage on forced sells (mirror env)
        if ('freed_mass' in locals()) and freed_mass > 0 and cfg.delist_extra_bps > 0:
            costs += (cfg.delist_extra_bps / 10_000.0) * float(freed_mass)
        
        # Realized return (same-day)
        # CRITICAL: R[t] already contains TOTAL returns (spread + CDI)
        r_vec = R[t].copy()
        bad = ~np.isfinite(r_vec)
        if bad.any():
            # For inactive/missing, use risk-free rate as total return
            r_vec[bad] = rf_t
        
        rp_assets = float(np.dot(w_new, r_vec))
        
        # Cash position earns risk-free
        cash_w = max(0.0, 1.0 - float(w_new.sum())) if cfg.allow_cash else 0.0
        rp_cash = cash_w * rf_t
        
        # Total portfolio return
        r_p = rp_assets + rp_cash - costs

        wealth *= (1.0 + r_p)
        peak = max(peak, wealth)
        dd = 1.0 - (wealth / max(peak, 1e-12))

        # Diagnostics
        hhi = _hhi(w_new)
        turn_assets = float(np.abs(w_new - w).sum())

        # Logs
        ret_rows.append({
            "date": pd.Timestamp(dates[t]),
            "portfolio_return": r_p,
            "excess": r_p - rf_t,
            "alpha": r_p - idx_t,
            "rf": rf_t,
            "idx": idx_t,
            "wealth": wealth,
            "drawdown": dd,
        })
        diag_rows.append({
            "date": pd.Timestamp(dates[t]),
            "hhi": hhi,
            "turnover": turn_total,
            "trade_cost": costs,
            "n_positions": float((w_new > 0).sum()),
            "cash_weight": cash_w,
        })
        w_rows.append(pd.Series(w_new, index=asset_ids, name=pd.Timestamp(dates[t])))

        # advance
        w = w_new

    returns = pd.DataFrame(ret_rows).set_index("date").sort_index()
    diagnostics = pd.DataFrame(diag_rows).set_index("date").sort_index()
    weights = pd.DataFrame(w_rows).sort_index()
    weights.index.name = "date"
    return {"returns": returns, "diagnostics": diagnostics, "weights": weights}

# ----------------------------------- Main --------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Dynamic-universe baselines for debenture OPS - CORRECTED")
    ap.add_argument("--universe", type=str, choices=["cdi","infra"], required=True)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_base", type=str, default=".")
    ap.add_argument("--strategies", type=str, default="EW,INDEX,RP_VOL,RP_DURATION,CARRY_TILT,MINVAR")
    ap.add_argument("--rebalance_interval", type=int, default=10)
    ap.add_argument("--max_weight", type=float, default=0.10)
    ap.add_argument("--transaction_cost_bps", type=float, default=20.0)
    ap.add_argument("--allow_cash", action="store_true", default=True)
    ap.add_argument("--on_inactive", type=str, default="to_cash", choices=["to_cash","pro_rata"])
    ap.add_argument("--roll_window", type=int, default=60)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--delist_extra_bps", type=float, default=20.0)
    ap.add_argument("--max_assets", type=int, default=50)
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    
    # Load config file if available
    config = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with config values if not specified via CLI
    rebalance_interval = config.get('rebalance_interval', args.rebalance_interval)
    max_weight = config.get('max_weight', args.max_weight)
    transaction_cost_bps = config.get('transaction_cost_bps', args.transaction_cost_bps)
    allow_cash = config.get('allow_cash', args.allow_cash)
    max_assets = config.get('max_assets', args.max_assets)
    delist_extra_bps = config.get('delist_extra_bps', args.delist_extra_bps)
    roll_window = config.get('roll_window', args.roll_window)
    ridge = config.get('ridge', args.ridge)

    universe = args.universe.lower()
    proc_path = os.path.join(args.data_dir, f"{universe}_processed.pkl")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"Processed panel not found: {proc_path}")

    results_dir = os.path.join(args.out_base, "results", universe)
    ensure_dir(results_dir)

    # Load master panel
    panel: pd.DataFrame = pd.read_pickle(proc_path)
    
    print(f"\n[DATA VERIFICATION]")
    print(f"  Panel loaded: {len(panel):,} observations")
    
    # Verify return structure
    if "total_return" in panel.columns and "spread_return" in panel.columns:
        total_mean = panel["total_return"].mean()
        spread_mean = panel["spread_return"].mean()
        rf_mean = panel["risk_free"].mean()
        print(f"  Mean total return: {total_mean*100:.4f}%")
        print(f"  Mean spread return: {spread_mean*100:.4f}%")
        print(f"  Mean risk-free: {rf_mean*100:.4f}%")
        print(f"  Verification: Total ≈ Spread + RF? {abs(total_mean - (spread_mean + rf_mean)) < 1e-5}")
    
    # Load folds & (optionally) training union ids
    folds_path = os.path.join(results_dir, "training_folds.json")
    if not os.path.exists(folds_path):
        raise FileNotFoundError("training_folds.json not found; run train.py first.")
    fold_specs: List[Dict[str, str]] = read_json(folds_path)

    have_union = os.path.exists(os.path.join(results_dir, "training_union_ids.json"))

    # Output sinks
    fold_metrics_csv = os.path.join(results_dir, "fold_metrics.csv")
    all_returns_csv = os.path.join(results_dir, "all_returns.csv")
    all_diags_csv = os.path.join(results_dir, "all_diagnostics.csv")
    all_metrics_csv = os.path.join(results_dir, "all_metrics.csv")

    strat_list = [s.strip().upper() for s in args.strategies.split(",") if s.strip()]

    cfg = BaselineConfig(
        rebalance_interval=rebalance_interval,
        max_weight=max_weight,
        transaction_cost_bps=transaction_cost_bps,
        allow_cash=allow_cash,
        on_inactive=args.on_inactive,
        roll_window=roll_window,
        ridge=ridge,
        delist_extra_bps=delist_extra_bps,
        max_assets=max_assets
    )
    
    print(f"\n[CONFIGURATION]")
    print(f"  Strategies: {', '.join(strat_list)}")
    print(f"  Rebalance interval: {cfg.rebalance_interval}")
    print(f"  Max weight: {cfg.max_weight:.1%}")
    print(f"  Transaction cost: {cfg.transaction_cost_bps} bps")
    print(f"  Max assets: {cfg.max_assets}")
    print(f"  Allow cash: {cfg.allow_cash}")

    for fold in fold_specs:
        fold_i = int(fold["fold"])
        print(f"\n[FOLD {fold_i}]")
        print(f"  Test period: {fold['test_start']} to {fold['test_end']}")
        
        union_ids = union_ids_from_training(results_dir, fold_i) if have_union else None
        if not union_ids:
            union_ids = compute_union_ids_from_panel(panel, fold, 
                                                     max_assets=cfg.max_assets,
                                                     rebalance_interval=cfg.rebalance_interval)
        
        print(f"  Union assets: {len(union_ids)}")
        test_aug = build_eval_panel_with_union(panel, fold, union_ids)
        
        # Check test panel statistics
        test_dates = test_aug.index.get_level_values("date").unique()
        active_per_day = test_aug.groupby(level="date")["active"].sum()
        print(f"  Test dates: {len(test_dates)}")
        print(f"  Active assets per day: mean={active_per_day.mean():.1f}, max={active_per_day.max()}")

        for strat in strat_list:
            print(f"  Simulating {strat}...")
            sim = simulate_baseline(test_aug, strategy=strat, cfg=cfg)

            # Persist fold result
            out_pkl = os.path.join(results_dir, f"fold_{fold_i}_{strat}_baseline_results.pkl")
            pd.to_pickle(sim, out_pkl)

            # Metrics per fold
            ret = sim["returns"]["portfolio_return"]
            rf = sim["returns"]["rf"]
            idx = sim["returns"]["idx"]
            metrics = compute_metrics(ret, rf=rf, bench=idx)

            row = {
                "fold": fold_i,
                "seed": -1,
                "strategy": f"{strat}",
                **metrics,
            }
            cols = ["fold","seed","strategy","CAGR","Vol_ann","Sharpe","Sortino","MaxDD","Calmar",
                    "Alpha_daily","Beta","Information_Ratio","Up_capture","Down_capture",
                    "Hit_rate","Skew","Kurtosis"]
            append_row_csv(fold_metrics_csv, row, cols_order=cols)
            append_row_csv(all_metrics_csv, row, cols_order=cols)

            # Append returns (long)
            ret_long = sim["returns"][["portfolio_return"]].reset_index().rename(columns={"portfolio_return":"return"})
            ret_long["strategy"] = f"{strat}_f{fold_i}"
            ret_long = ret_long[["date","strategy","return"]]
            append_long_csv(all_returns_csv, ret_long, sort_by=["date","strategy"])

            # Append diagnostics (long)
            di = sim["diagnostics"].reset_index()
            di["strategy"] = f"{strat}_f{fold_i}"
            di = di.melt(id_vars=["date","strategy"], var_name="metric", value_name="value")
            append_long_csv(all_diags_csv, di, sort_by=["date","strategy","metric"])

            print(f"    {strat}: Sharpe={metrics['Sharpe']:.3f}, CAGR={metrics['CAGR']:.3%}, MaxDD={metrics['MaxDD']:.3%}")

    print("\n[DONE] Baselines complete.")
    print(f"Updated files in {results_dir}:")
    print(f"  - fold_metrics.csv")
    print(f"  - all_metrics.csv")
    print(f"  - all_returns.csv")
    print(f"  - all_diagnostics.csv")

if __name__ == "__main__":
    main()