from __future__ import annotations
import os, json, argparse, math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ------------------------------ Name handling ------------------------------ #

def canonical_group(name: str) -> str:
    """
    Map run labels to canonical groups:
      - 'PPO_f3_s1' -> 'PPO'
      - 'CARRY_TILT' -> 'CARRY'
      - 'INDEX_f2' -> 'INDEX', 'EW_f1' -> 'EW', etc.
    """
    s = str(name).upper()
    if s.startswith("PPO"):
        return "PPO"
    if s.startswith("CARRY_TILT"):
        return "CARRY"
    # drop suffixes like _f3, _s2, etc.
    base = s.split("_")[0]
    return base


# --------------------------- Bootstrap machinery --------------------------- #

def moving_block_bootstrap(x: np.ndarray, block: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """
    Circular moving-block bootstrap on a (T,) series. Returns (B, T).
    """
    T = len(x)
    out = np.empty((B, T), dtype=np.float64)
    n_blocks = int(np.ceil(T / block))
    for b in range(B):
        idx = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, T))
            blk = np.arange(start, start + block)
            blk = np.where(blk >= T, blk - T, blk)  # wrap
            idx.extend(blk.tolist())
        out[b, :] = x[np.array(idx[:T])]
    return out


# ------------------------------- Core metrics ------------------------------ #

def cagr(r: np.ndarray) -> float:
    g = float(np.prod(1.0 + r)) if r.size else 1.0
    n = max(1, r.size)
    return g**(TRADING_DAYS / n) - 1.0

def ann_vol(r: np.ndarray) -> float:
    if r.size <= 1:
        return 0.0
    return float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS))

def sharpe(r: np.ndarray, rf: np.ndarray) -> float:
    ex = r - rf
    sd = float(np.std(ex, ddof=1))
    return (float(np.mean(ex)) / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else 0.0

def sortino(r: np.ndarray, rf: np.ndarray) -> float:
    ex = r - rf
    down = ex[ex < 0.0]
    dsd = float(np.std(down, ddof=1)) if down.size > 1 else 0.0
    mu = float(np.mean(ex))
    return (mu / dsd) * math.sqrt(TRADING_DAYS) if dsd > 0 else 0.0

def maxdd(r: np.ndarray) -> float:
    w = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(w)
    dd = w / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min()) if dd.size else 0.0  # negative

def calmar(r: np.ndarray) -> float:
    mdd = abs(min(0.0, maxdd(r)))
    return cagr(r) / mdd if mdd > 0 else 0.0

def var_cvar(r: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """
    1-day VaR/CVaR at `level` on *returns* (left tail).
    Returns (VaR_level, CVaR_level), both typically ≤ 0 for left tail.
    """
    if r.size == 0:
        return (0.0, 0.0)
    q = np.quantile(r, 1.0 - level)
    tail = r[r <= q]
    cvar = float(tail.mean()) if tail.size else float(q)
    return float(q), cvar

def drawdown_durations(r: np.ndarray) -> np.ndarray:
    """
    Return array of drawdown episode durations (in days).
    """
    if r.size == 0:
        return np.array([0], dtype=int)
    w = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(w)
    dd = w / np.maximum(peak, 1e-12) - 1.0
    durations = []
    cur = 0
    for x in dd:
        if x < 0:
            cur += 1
        elif cur > 0:
            durations.append(cur)
            cur = 0
    if cur > 0:
        durations.append(cur)
    if not durations:
        durations = [0]
    return np.array(durations, dtype=int)


# --------------------------- HAC / DM test utils --------------------------- #

def _bartlett_weight(k: int, q: int) -> float:
    return 1.0 - k / (q + 1.0)

def _auto_nw_lag(T: int) -> int:
    # Newey–West automatic lag (e.g., Andrews (1991) rule-of-thumb variant)
    return max(1, int(4 * (T / 100.0) ** (2.0 / 9.0)))

def _hac_spectral_density_zero(u: np.ndarray, lag: Optional[int] = None) -> float:
    """
    Bartlett-kernel HAC estimator of S(0) for the process u_t.
    Returns S_hat (not divided by T).
    """
    T = u.size
    if T <= 1:
        return 0.0
    q = _auto_nw_lag(T) if lag is None else int(lag)
    u = u - u.mean()
    gamma0 = float(np.dot(u, u)) / T
    s = gamma0
    for k in range(1, min(q, T - 1) + 1):
        cov = float(np.dot(u[k:], u[:-k])) / T
        s += 2.0 * _bartlett_weight(k, q) * cov
    return s

def nw_t_test_mean(x: np.ndarray, lag: Optional[int] = None) -> Tuple[float, float, int]:
    """
    Test H0: E[x]=0 with Newey–West HAC variance.
    Returns (t_stat, p_value, used_lag).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = x.size
    if T <= 1:
        return 0.0, 1.0, 0
    q = _auto_nw_lag(T) if lag is None else int(lag)
    S = _hac_spectral_density_zero(x, q)  # spectral density at 0 (not /T)
    var_mean = S / T
    se = math.sqrt(max(var_mean, 1e-16))
    t = float(x.mean()) / se
    # normal tail
    p = 2.0 * 0.5 * math.erfc(abs(t) / math.sqrt(2.0))
    return t, p, q

def diebold_mariano_test(r_a: np.ndarray, r_b: np.ndarray, lag: Optional[int] = None) -> Tuple[float, float, int]:
    """
    DM test on daily 'losses' ℓ = −r (so higher returns => lower loss).
    H0: E[d_t] = 0 where d_t = ℓ_a - ℓ_b. Returns (DM_stat, p_value, used_lag).
    """
    # losses: lower is better
    l_a = -np.asarray(r_a, dtype=float)
    l_b = -np.asarray(r_b, dtype=float)
    # align
    n = min(l_a.size, l_b.size)
    d = l_a[:n] - l_b[:n]
    T = d.size
    if T <= 1:
        return 0.0, 1.0, 0
    q = _auto_nw_lag(T) if lag is None else int(lag)
    S = _hac_spectral_density_zero(d, q)
    var_mean = S / T
    se = math.sqrt(max(var_mean, 1e-16))
    dm = float(d.mean()) / se
    p = 2.0 * 0.5 * math.erfc(abs(dm) / math.sqrt(2.0))
    return dm, p, q


# ------------------------------ I/O helpers -------------------------------- #

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_returns_long(path: str) -> pd.DataFrame:
    """
    Expects long format: date,strategy,return  (from evaluate_final.py / baselines_final.py)
    """
    df = pd.read_csv(path, parse_dates=["date"])
    # Defensive column normalization
    lc = {c.lower(): c for c in df.columns}
    date_col = lc.get("date", "date")
    strat_col = lc.get("strategy", "strategy")
    ret_col = lc.get("return", "return")
    out = df[[date_col, strat_col, ret_col]].rename(columns={date_col: "date", strat_col: "strategy", ret_col: "return"})
    # Coerce return to numeric (already net of costs; do not touch)
    out["return"] = pd.to_numeric(out["return"], errors="coerce")
    out = out.dropna(subset=["date", "strategy", "return"]).sort_values(["date", "strategy"])
    return out

def returns_wide_by_group(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to canonical group per date. If multiple runs per group on a day, take the mean.
    """
    d = df_long.copy()
    d["group"] = d["strategy"].map(canonical_group)
    d = (
        d.groupby(["date", "group"], as_index=False)["return"].mean()
        .pivot(index="date", columns="group", values="return")
        .sort_index()
    )
    # Drop all-NaN columns
    d = d.loc[:, d.notna().any(axis=0)]
    return d

def load_risk_free_series(res_dir: str, rf_csv: Optional[str]) -> pd.Series:
    """
    Try to load daily risk-free from results/<universe>/risk_free_used.csv (created by data_final.py).
    Falls back to zeros if not found. Accepts columns: 'risk_free','rf','cdi_daily','selic_daily','value'.
    """
    candidates = []
    if rf_csv:
        candidates.append(rf_csv)
    candidates.append(os.path.join(res_dir, "risk_free_used.csv"))
    for path in candidates:
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # rename common variants
                lc = {c.lower(): c for c in df.columns}
                date_col = lc.get("date", "date")
                s = pd.to_datetime(df[date_col], errors="coerce")
                if "risk_free" in lc:
                    vals = pd.to_numeric(df[lc["risk_free"]], errors="coerce")
                elif "rf" in lc:
                    vals = pd.to_numeric(df[lc["rf"]], errors="coerce")
                elif "cdi_daily" in lc:
                    vals = pd.to_numeric(df[lc["cdi_daily"]], errors="coerce")
                elif "selic_daily" in lc:
                    vals = pd.to_numeric(df[lc["selic_daily"]], errors="coerce")
                elif "value" in lc:
                    vals = pd.to_numeric(df[lc["value"]], errors="coerce")
                    # assume already in decimal daily
                else:
                    raise ValueError("No recognizable risk-free column.")
                ser = pd.Series(vals.values, index=s).dropna()
                ser.name = "risk_free"
                return ser
            except Exception:
                continue
    # fallback: zero
    return pd.Series(dtype=float, name="risk_free")


def save_json_update(path: str, payload: dict) -> None:
    old = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                old = json.load(f)
        except Exception:
            old = {}
    old.update(payload)
    with open(path, "w") as f:
        json.dump(old, f, indent=2, sort_keys=True)


# ------------------------------- Main routine ------------------------------ #

def compute_metric_cis(r: np.ndarray, rf: np.ndarray, B: int, block: int, rng: np.random.Generator,
                       var_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Compute 95% CIs (via moving-block bootstrap) for standard + new metrics.
    """
    # point series (rf aligned)
    T = min(r.size, rf.size) if rf.size else r.size
    r0 = r[:T]
    rf0 = rf[:T] if rf.size else np.zeros(T, dtype=float)

    # bootstrap samples
    boots = moving_block_bootstrap(r0, block=block, B=B, rng=rng)

    def ci_of(fn) -> Dict[str, float]:
        vals = np.apply_along_axis(lambda a: fn(a), 1, boots)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi)}

    # functions that need rf
    def sharpe_fn(a): return sharpe(a, rf0[:a.size])
    def sortino_fn(a): return sortino(a, rf0[:a.size])
    def var_fn(a): return var_cvar(a, level=var_level)[0]
    def cvar_fn(a): return var_cvar(a, level=var_level)[1]
    def dd_dur_max_fn(a): return float(drawdown_durations(a).max()) if a.size else 0.0
    def dd_dur_p95_fn(a):
        durs = drawdown_durations(a)
        return float(np.quantile(durs, 0.95)) if durs.size else 0.0

    out = {
        "CAGR": ci_of(cagr),
        "Vol_ann": ci_of(ann_vol),
        "Sharpe": ci_of(sharpe_fn),
        "Sortino": ci_of(sortino_fn),
        "MaxDD": ci_of(maxdd),
        "Calmar": ci_of(calmar),
        "VaR_95": ci_of(var_fn),
        "CVaR_95": ci_of(cvar_fn),
        "MaxDD_Duration": ci_of(dd_dur_max_fn),
        "DD_Duration_95p": ci_of(dd_dur_p95_fn),
    }
    return out

def pairwise_tests(wide: pd.DataFrame, target: str, lag: Optional[int] = None) -> pd.DataFrame:
    """
    PPO vs all other groups (that overlap in time), or target vs all.
    Returns DataFrame with mean-diff (NW) and DM test results on daily returns.
    """
    rows = []
    cols = [c for c in wide.columns if c != target]
    for b in cols:
        df = wide[[target, b]].dropna()
        if df.empty or df.shape[0] < 30:
            continue
        a = df[target].values
        c = df[b].values
        diff = a - c
        tnw, pnw, qnw = nw_t_test_mean(diff, lag=lag)
        dm, pdm, qdm = diebold_mariano_test(a, c, lag=lag)
        rows.append({
            "group_a": target,
            "group_b": b,
            "T": int(df.shape[0]),
            "mean_diff": float(diff.mean()),
            "t_nw": float(tnw),
            "p_nw": float(pnw),
            "nw_lag": int(qnw),
            "dm_stat": float(dm),
            "p_dm": float(pdm),
            "dm_lag": int(qdm),
        })
    return pd.DataFrame(rows)

def replication_regression(wide: pd.DataFrame, y_name: str, candidates: List[str],
                           lag: Optional[int] = None) -> Tuple[pd.DataFrame, float, int]:
    """
    OLS: y_t = alpha + X_t beta + eps_t, NW-robust SE.
    Returns (coef table, R2, used_lag).
    """
    # prepare
    cols = [c for c in candidates if c in wide.columns and c != y_name]
    if y_name not in wide.columns or len(cols) == 0:
        return pd.DataFrame(), float("nan"), 0
    df = wide[[y_name] + cols].dropna()
    if df.shape[0] < 30:
        return pd.DataFrame(), float("nan"), 0
    y = df[y_name].values.reshape(-1, 1)  # (T,1)
    X = df[cols].values                   # (T,K)
    T, K = X.shape
    X1 = np.hstack([np.ones((T, 1)), X])  # add intercept
    # OLS
    XtX = X1.T @ X1
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X1.T @ y)          # (K+1,1)
    e = (y - X1 @ beta).ravel()
    # NW covariance
    q = _auto_nw_lag(T) if lag is None else int(lag)
    Z = X1 * e.reshape(-1, 1)             # T × (K+1)
    S = Z.T @ Z  # k=0 term
    for k in range(1, min(q, T - 1) + 1):
        w = _bartlett_weight(k, q)
        S += w * (Z[k:].T @ Z[:-k] + Z[:-k].T @ Z[k:])
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 1e-16))
    # t, p
    tvals = (beta.ravel() / se)
    pvals = 2.0 * 0.5 * np.array([math.erfc(abs(t) / math.sqrt(2.0)) for t in tvals])
    # R^2
    ss_res = float((e**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # table
    names = ["Intercept"] + cols
    tab = pd.DataFrame({
        "var": names,
        "coef": beta.ravel(),
        "se_nw": se,
        "t": tvals,
        "p": pvals,
    })
    return tab, float(r2), q


def main():
    ap = argparse.ArgumentParser(description="Block-bootstrap CIs + pairwise tests + risk diagnostics")
    ap.add_argument("--universe", required=True, choices=["cdi", "infra"])
    ap.add_argument("--out_base", default=".")
    ap.add_argument("--returns_csv", default=None, help="override path to all_returns.csv")
    ap.add_argument("--rf_csv", default=None, help="override path to risk_free_used.csv")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=21, help="moving-block length (business days)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--var_level", type=float, default=0.95)
    ap.add_argument("--nw_lag", type=int, default=None, help="manual Newey–West lag (else automatic)")
    ap.add_argument("--target", default="PPO", help="target group for pairwise tests")
    ap.add_argument("--do_replication", action="store_true", help="fit PPO vs baselines OLS with NW SEs")
    ap.add_argument("--diagnostics_csv", default=None, help="override path to all_diagnostics.csv for turnover/HHI CIs")
    args = ap.parse_args()

    res_dir = os.path.join(args.out_base, "results", args.universe)
    ensure_dir(res_dir)
    rng = np.random.default_rng(args.seed)

    # --- Load returns (already NET of costs; do not adjust) ---
    returns_path = args.returns_csv or os.path.join(res_dir, "all_returns.csv")
    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"Missing returns file: {returns_path}")
    df_long = load_returns_long(returns_path)   # long: date,strategy,return
    wide = returns_wide_by_group(df_long)       # wide by canonical group

    # risk-free (optional)
    rf_ser = load_risk_free_series(res_dir, args.rf_csv)

    # --- Bootstrap CIs per group ---
    ci_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    for g in wide.columns:
        r = wide[g].dropna().values.astype(float)
        # align rf
        rf = rf_ser.reindex(wide.index).fillna(0.0).values.astype(float)
        cis = compute_metric_cis(r, rf, B=args.n_boot, block=args.block, rng=rng, var_level=args.var_level)
        ci_payload[g] = cis

    # --- Optional diagnostics CIs (turnover, HHI) ---
    diag_path = args.diagnostics_csv or os.path.join(res_dir, "all_diagnostics.csv")
    if os.path.exists(diag_path):
        try:
            diag = pd.read_csv(diag_path, parse_dates=["date"])
            # Expected columns: date,strategy,metric,value (as emitted by evaluate/baselines)
            # Bootstrap CIs of *average* turnover/HHI per group
            diag["group"] = diag["strategy"].map(canonical_group)
            for metric in ["turnover", "hhi"]:
                sub = diag[diag["metric"].str.lower() == metric]
                if sub.empty:
                    continue
                # build wide per group daily values
                wdg = (sub.groupby(["date", "group"], as_index=False)["value"].mean()
                            .pivot(index="date", columns="group", values="value")
                            .sort_index())
                for g in wdg.columns:
                    x = wdg[g].dropna().values.astype(float)
                    if x.size < 5:
                        continue
                    boots = moving_block_bootstrap(x, block=args.block, B=args.n_boot, rng=rng)
                    vals = boots.mean(axis=1)
                    lo, hi = np.percentile(vals, [2.5, 97.5])
                    key = "AvgTurnover" if metric == "turnover" else "AvgHHI"
                    ci_payload.setdefault(g, {})[key] = {"lo": float(lo), "hi": float(hi)}
        except Exception:
            pass

    # --- Save confidence_intervals.json (merge/update) ---
    ci_path = os.path.join(res_dir, "confidence_intervals.json")
    save_json_update(ci_path, ci_payload)

    # --- Pairwise tests: target vs others ---
    pt = pairwise_tests(wide, target=args.target, lag=args.nw_lag)
    if not pt.empty:
        pt.to_csv(os.path.join(res_dir, "pairwise_tests.csv"), index=False)

    # --- Replication/overlap analysis (optional) ---
    if args.do_replication and "PPO" in wide.columns:
        base_cands = [c for c in ["EW", "INDEX", "RP_VOL", "RP_DURATION", "CARRY", "MINVAR"] if c in wide.columns]
        tab, r2, q = replication_regression(wide, "PPO", base_cands, lag=args.nw_lag)
        if not tab.empty:
            tab["R2"] = r2
            tab["nw_lag"] = q
            tab.to_csv(os.path.join(res_dir, "replication_regression.csv"), index=False)
        # simple correlation matrix for transparency
        corr = wide[["PPO"] + base_cands].corr()
        corr.to_csv(os.path.join(res_dir, "pairwise_correlations.csv"))

    print(f"[OK] Wrote CIs to {ci_path}")
    if not pt.empty:
        print(f"[OK] Pairwise tests -> {os.path.join(res_dir, 'pairwise_tests.csv')}")
    if args.do_replication:
        print("[OK] Replication regression outputs (if applicable) written.")


if __name__ == "__main__":
    main()
