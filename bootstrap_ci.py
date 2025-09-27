# bootstrap_ci.py
"""
Enhanced Bootstrap Confidence Interval Generator
------------------------------------------------
Generates confidence intervals and statistical tests required for comprehensive analysis.
Ensures compatibility with generate_summary_report.py and cost_attribution_analyzer.py

Key improvements:
1. More robust file handling for universe-specific directories
2. Enhanced error handling and logging
3. Additional metrics for cost analysis
4. Better integration with the report pipeline
"""

from __future__ import annotations
import os
import json
import argparse
import math
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

TRADING_DAYS = 252


# ========================= Name Handling ========================= #

def canonical_group(name: str) -> str:
    """
    Map run labels to canonical groups:
      - 'PPO_f3_s1' -> 'PPO'
      - 'CARRY_TILT' -> 'CARRY_TILT'
      - 'INDEX_f2' -> 'INDEX', 'EW_f1' -> 'EW', etc.
    """
    s = str(name).upper()
    
    # PPO variants
    if s.startswith("PPO"):
        return "PPO"
    
    # Handle specific strategies
    if "CARRY_TILT" in s:
        return "CARRY_TILT"
    if "RP_VOL" in s:
        return "RP_VOL"
    if "RP_DURATION" in s:
        return "RP_DURATION"
    if "MINVAR" in s:
        return "MINVAR"
    
    # For others, use the base name before _f or _s
    for sep in ["_F", "_S"]:
        if sep in s:
            return s.split(sep)[0]
    
    # Default: first part before underscore
    return s.split("_")[0] if "_" in s else s


# ==================== Bootstrap Machinery ======================= #

def moving_block_bootstrap(x: np.ndarray, block: int, B: int, 
                          rng: np.random.Generator) -> np.ndarray:
    """
    Circular moving-block bootstrap on a (T,) series. Returns (B, T).
    """
    T = len(x)
    if T == 0:
        return np.empty((B, 0), dtype=np.float64)
    
    out = np.empty((B, T), dtype=np.float64)
    n_blocks = int(np.ceil(T / block))
    
    for b in range(B):
        idx = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, T))
            blk = np.arange(start, start + block)
            blk = np.where(blk >= T, blk - T, blk)  # wrap around
            idx.extend(blk.tolist())
        out[b, :] = x[np.array(idx[:T])]
    
    return out


# ======================= Core Metrics ========================= #

def cagr(r: np.ndarray) -> float:
    """Compound Annual Growth Rate."""
    if r.size == 0:
        return 0.0
    wealth_final = float(np.prod(1.0 + r))
    n_years = r.size / TRADING_DAYS
    return wealth_final**(1.0 / max(n_years, 1e-10)) - 1.0

def ann_vol(r: np.ndarray) -> float:
    """Annualized volatility."""
    if r.size <= 1:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))

def sharpe(r: np.ndarray, rf: np.ndarray) -> float:
    """Sharpe ratio (annualized)."""
    if r.size <= 1:
        return 0.0
    ex = r - rf[:r.size] if rf.size > 0 else r
    sd = float(np.std(ex, ddof=1))
    if sd <= 0:
        return 0.0
    return (float(np.mean(ex)) / sd) * np.sqrt(TRADING_DAYS)

def sortino(r: np.ndarray, rf: np.ndarray) -> float:
    """Sortino ratio (annualized)."""
    if r.size <= 1:
        return 0.0
    ex = r - rf[:r.size] if rf.size > 0 else r
    down = ex[ex < 0.0]
    if down.size <= 1:
        return 0.0
    dsd = float(np.std(down, ddof=1))
    if dsd <= 0:
        return 0.0
    mu = float(np.mean(ex))
    return (mu / dsd) * np.sqrt(TRADING_DAYS)

def maxdd(r: np.ndarray) -> float:
    """Maximum drawdown (negative value)."""
    if r.size == 0:
        return 0.0
    w = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(w)
    dd = w / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())  # negative value

def calmar(r: np.ndarray) -> float:
    """Calmar ratio: CAGR / |MaxDD|"""
    mdd = abs(maxdd(r))
    if mdd <= 0:
        return 0.0
    return cagr(r) / mdd

def information_ratio(r: np.ndarray, bench: np.ndarray) -> float:
    """Information ratio (annualized)."""
    if r.size <= 1 or bench.size <= 1:
        return 0.0
    min_len = min(r.size, bench.size)
    active = r[:min_len] - bench[:min_len]
    te = float(np.std(active, ddof=1))
    if te <= 0:
        return 0.0
    return (float(np.mean(active)) / te) * np.sqrt(TRADING_DAYS)

def var_cvar(r: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    """
    Value at Risk and Conditional Value at Risk at given confidence level.
    Returns (VaR, CVaR), both typically ≤ 0 for left tail.
    """
    if r.size == 0:
        return (0.0, 0.0)
    q = np.percentile(r, (1.0 - level) * 100)
    tail = r[r <= q]
    cvar = float(tail.mean()) if tail.size > 0 else float(q)
    return float(q), cvar

def drawdown_durations(r: np.ndarray) -> np.ndarray:
    """Return array of drawdown episode durations (in days)."""
    if r.size == 0:
        return np.array([0], dtype=int)
    
    w = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(w)
    dd = w / np.maximum(peak, 1e-12) - 1.0
    
    durations = []
    cur_duration = 0
    
    for x in dd:
        if x < -1e-6:  # In drawdown
            cur_duration += 1
        elif cur_duration > 0:
            durations.append(cur_duration)
            cur_duration = 0
    
    if cur_duration > 0:
        durations.append(cur_duration)
    
    return np.array(durations, dtype=int) if durations else np.array([0], dtype=int)


# ================= Statistical Test Utilities =================== #

def _bartlett_weight(k: int, q: int) -> float:
    """Bartlett kernel weight."""
    return max(0.0, 1.0 - k / (q + 1.0))

def _auto_nw_lag(T: int) -> int:
    """Newey-West automatic lag selection."""
    return max(1, min(int(4 * (T / 100.0) ** (2.0 / 9.0)), T // 3))

def _hac_variance(u: np.ndarray, lag: Optional[int] = None) -> float:
    """HAC variance estimator with Bartlett kernel."""
    T = u.size
    if T <= 1:
        return 0.0
    
    q = _auto_nw_lag(T) if lag is None else min(int(lag), T - 1)
    u_centered = u - u.mean()
    
    # Variance at lag 0
    gamma0 = float(np.dot(u_centered, u_centered)) / T
    s = gamma0
    
    # Add autocovariances
    for k in range(1, q + 1):
        if k >= T:
            break
        cov = float(np.dot(u_centered[k:], u_centered[:-k])) / T
        s += 2.0 * _bartlett_weight(k, q) * cov
    
    return max(s / T, 1e-16)  # Ensure positive

def diebold_mariano_test(r_a: np.ndarray, r_b: np.ndarray, 
                         lag: Optional[int] = None) -> Tuple[float, float, int]:
    """
    Diebold-Mariano test for comparing predictive accuracy.
    H0: E[d_t] = 0 where d_t = loss_a - loss_b
    Using squared loss (lower returns = higher loss).
    Returns (DM_stat, p_value, used_lag).
    """
    # Align series
    min_len = min(r_a.size, r_b.size)
    if min_len < 10:
        return (0.0, 1.0, 0)
    
    r_a = r_a[:min_len]
    r_b = r_b[:min_len]
    
    # Loss differential (using negative returns as loss)
    d = (-r_a) - (-r_b)  # Positive d means b is better
    
    # HAC variance
    q = _auto_nw_lag(min_len) if lag is None else int(lag)
    var_d = _hac_variance(d, q)
    se_d = np.sqrt(var_d)
    
    # Test statistic
    dm_stat = float(d.mean()) / se_d if se_d > 0 else 0.0
    
    # Two-sided p-value (normal approximation)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value, q


# ======================= I/O Helpers ========================= #

def ensure_dir(p: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)

def verify_files_exist(res_dir: str) -> bool:
    """Verify that required input files exist."""
    required_files = [
        "all_returns.csv",
        "all_diagnostics.csv"
    ]
    
    all_exist = True
    for file in required_files:
        path = os.path.join(res_dir, file)
        if not os.path.exists(path):
            print(f"[WARN] Missing required file: {path}")
            all_exist = False
        else:
            print(f"[OK] Found: {path}")
    
    return all_exist

def load_returns_long(path: str) -> pd.DataFrame:
    """Load returns in long format with proper error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Returns file not found: {path}")
    
    print(f"Loading returns from: {path}")
    df = pd.read_csv(path)
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Check required columns
    required_cols = ["date", "strategy", "return"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        # Try alternative column names
        if "portfolio_return" in df.columns and "return" not in df.columns:
            df["return"] = df["portfolio_return"]
        
        # Re-check
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Process data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    
    # Clean
    df = df.dropna(subset=["date", "strategy", "return"])
    df = df.sort_values(["date", "strategy"])
    
    print(f"  Loaded {len(df)} return observations for {df['strategy'].nunique()} strategies")
    
    return df[["date", "strategy", "return"]]

def load_diagnostics_long(path: str) -> Optional[pd.DataFrame]:
    """Load diagnostics in long format."""
    if not os.path.exists(path):
        print(f"[INFO] Diagnostics file not found: {path}")
        return None
    
    print(f"Loading diagnostics from: {path}")
    df = pd.read_csv(path)
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Check format
    if "date" in df.columns and "strategy" in df.columns and "metric" in df.columns and "value" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "strategy", "metric", "value"])
        print(f"  Loaded {len(df)} diagnostic observations")
        return df[["date", "strategy", "metric", "value"]]
    
    print("[WARN] Diagnostics file has unexpected format")
    return None

def returns_wide_by_group(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long format to wide format by canonical group."""
    d = df_long.copy()
    d["group"] = d["strategy"].map(canonical_group)
    
    # Average multiple runs for same group on same date
    d_agg = d.groupby(["date", "group"], as_index=False)["return"].mean()
    
    # Pivot to wide
    wide = d_agg.pivot(index="date", columns="group", values="return")
    wide = wide.sort_index()
    
    # Drop columns with all NaN
    wide = wide.loc[:, wide.notna().any(axis=0)]
    
    print(f"  Created wide format: {len(wide)} dates × {len(wide.columns)} groups")
    print(f"  Groups: {list(wide.columns)}")
    
    return wide

def load_risk_free_series(res_dir: str, rf_csv: Optional[str] = None) -> pd.Series:
    """Load risk-free rate series with fallback options."""
    candidates = []
    
    if rf_csv:
        candidates.append(rf_csv)
    
    candidates.extend([
        os.path.join(res_dir, "risk_free_used.csv"),
        os.path.join(res_dir, "rf.csv")
    ])
    
    for path in candidates:
        if os.path.exists(path):
            try:
                print(f"Loading risk-free rate from: {path}")
                df = pd.read_csv(path)
                df.columns = [c.lower() for c in df.columns]
                
                # Find date column
                date_col = "date" if "date" in df.columns else df.columns[0]
                dates = pd.to_datetime(df[date_col], errors="coerce")
                
                # Find value column
                value_col = None
                for col in ["risk_free", "rf", "cdi_daily", "selic_daily", "value"]:
                    if col in df.columns:
                        value_col = col
                        break
                
                if value_col:
                    vals = pd.to_numeric(df[value_col], errors="coerce")
                    ser = pd.Series(vals.values, index=dates)
                    ser = ser.dropna()
                    ser.name = "risk_free"
                    print(f"  Loaded {len(ser)} risk-free observations")
                    return ser
            except Exception as e:
                print(f"  Failed to load {path}: {e}")
                continue
    
    print("[WARN] No risk-free rate data found, using zeros")
    return pd.Series(dtype=float, name="risk_free")

def load_index_returns(res_dir: str) -> Optional[pd.Series]:
    """Load index returns for information ratio calculation."""
    # Try to load from all_returns.csv
    returns_path = os.path.join(res_dir, "all_returns.csv")
    
    if os.path.exists(returns_path):
        try:
            df = pd.read_csv(returns_path)
            df.columns = [c.lower() for c in df.columns]
            
            # Look for index return column
            if "idx" in df.columns or "index_return" in df.columns:
                idx_col = "idx" if "idx" in df.columns else "index_return"
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                idx_ser = df.groupby("date")[idx_col].mean()
                return idx_ser
        except:
            pass
    
    return None


# =================== Bootstrap CI Computation ==================== #

def compute_metric_cis(r: np.ndarray, rf: np.ndarray, bench: Optional[np.ndarray],
                       B: int, block: int, rng: np.random.Generator,
                       var_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    """
    Compute confidence intervals for all metrics using moving block bootstrap.
    """
    # Ensure alignment
    T = min(r.size, rf.size) if rf.size > 0 else r.size
    r0 = r[:T]
    rf0 = rf[:T] if rf.size > 0 else np.zeros(T, dtype=float)
    bench0 = bench[:T] if bench is not None and bench.size > 0 else None
    
    # Generate bootstrap samples
    boots = moving_block_bootstrap(r0, block=block, B=B, rng=rng)
    
    # Helper function to compute CI
    def ci_of(fn) -> Dict[str, float]:
        vals = np.array([fn(boots[i]) for i in range(B)])
        vals = vals[np.isfinite(vals)]  # Remove NaN/Inf
        if len(vals) == 0:
            return {"lo": 0.0, "hi": 0.0, "mean": 0.0, "std": 0.0}
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {
            "lo": float(lo),
            "hi": float(hi),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }
    
    # Define metric functions
    def sharpe_fn(a): 
        return sharpe(a, rf0[:a.size])
    
    def sortino_fn(a): 
        return sortino(a, rf0[:a.size])
    
    def var_fn(a): 
        return var_cvar(a, level=var_level)[0]
    
    def cvar_fn(a): 
        return var_cvar(a, level=var_level)[1]
    
    def dd_dur_mean_fn(a):
        durs = drawdown_durations(a)
        return float(durs.mean()) if durs.size > 0 else 0.0
    
    def dd_dur_max_fn(a):
        durs = drawdown_durations(a)
        return float(durs.max()) if durs.size > 0 else 0.0
    
    def ir_fn(a):
        if bench0 is not None:
            return information_ratio(a, bench0[:a.size])
        return 0.0
    
    # Compute CIs for all metrics
    result = {
        "CAGR": ci_of(cagr),
        "Vol_ann": ci_of(ann_vol),
        "Sharpe": ci_of(sharpe_fn),
        "Sortino": ci_of(sortino_fn),
        "MaxDD": ci_of(maxdd),
        "Calmar": ci_of(calmar),
        "VaR_95": ci_of(var_fn),
        "CVaR_95": ci_of(cvar_fn),
        "DD_Duration_Mean": ci_of(dd_dur_mean_fn),
        "DD_Duration_Max": ci_of(dd_dur_max_fn)
    }
    
    # Add information ratio if benchmark available
    if bench0 is not None:
        result["Information_Ratio"] = ci_of(ir_fn)
    
    return result

def compute_diagnostic_cis(diag_df: pd.DataFrame, metric: str, block: int, 
                          B: int, rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    """Compute CIs for diagnostic metrics (turnover, HHI, etc.)."""
    results = {}
    
    # Get canonical groups
    diag_df = diag_df.copy()
    diag_df["group"] = diag_df["strategy"].map(canonical_group)
    
    # Filter for specific metric
    metric_df = diag_df[diag_df["metric"].str.lower() == metric.lower()]
    
    if metric_df.empty:
        return results
    
    # Process each group
    for group in metric_df["group"].unique():
        group_data = metric_df[metric_df["group"] == group]
        
        # Daily average values
        daily_avg = group_data.groupby("date")["value"].mean()
        vals = daily_avg.values
        
        if vals.size < 10:
            continue
        
        # Bootstrap
        boots = moving_block_bootstrap(vals, block=block, B=B, rng=rng)
        
        # Statistics
        boot_means = boots.mean(axis=1)
        boot_stds = boots.std(axis=1)
        
        results[group] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "ci_mean_lo": float(np.percentile(boot_means, 2.5)),
            "ci_mean_hi": float(np.percentile(boot_means, 97.5)),
            "ci_std_lo": float(np.percentile(boot_stds, 2.5)),
            "ci_std_hi": float(np.percentile(boot_stds, 97.5))
        }
    
    return results


# ==================== Statistical Tests ======================== #

def pairwise_tests(wide: pd.DataFrame, target: str, 
                  lag: Optional[int] = None) -> pd.DataFrame:
    """
    Perform pairwise statistical tests between target and other strategies.
    """
    if target not in wide.columns:
        print(f"[WARN] Target '{target}' not found in data")
        return pd.DataFrame()
    
    results = []
    other_cols = [c for c in wide.columns if c != target]
    
    for other in other_cols:
        # Get overlapping data
        pair_df = wide[[target, other]].dropna()
        
        if pair_df.shape[0] < 30:
            continue
        
        target_returns = pair_df[target].values
        other_returns = pair_df[other].values
        
        # Compute statistics
        diff = target_returns - other_returns
        
        # T-test (parametric)
        t_stat, t_pval = stats.ttest_rel(target_returns, other_returns)
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pval = stats.wilcoxon(target_returns, other_returns)
        except:
            w_stat, w_pval = np.nan, np.nan
        
        # Diebold-Mariano test
        dm_stat, dm_pval, dm_lag = diebold_mariano_test(target_returns, other_returns, lag)
        
        # Summary statistics
        results.append({
            "Strategy_A": target,
            "Strategy_B": other,
            "N_obs": len(pair_df),
            "Mean_A": float(target_returns.mean()),
            "Mean_B": float(other_returns.mean()),
            "Mean_Diff": float(diff.mean()),
            "Std_Diff": float(diff.std()),
            "T_stat": float(t_stat),
            "T_pval": float(t_pval),
            "Wilcoxon_stat": float(w_stat),
            "Wilcoxon_pval": float(w_pval),
            "DM_stat": float(dm_stat),
            "DM_pval": float(dm_pval),
            "DM_lag": int(dm_lag),
            "Correlation": float(np.corrcoef(target_returns, other_returns)[0, 1])
        })
    
    return pd.DataFrame(results)


# ======================== Main Routine =========================== #

def main():
    ap = argparse.ArgumentParser(
        description="Generate bootstrap confidence intervals and statistical tests"
    )
    ap.add_argument("--universe", required=True, choices=["cdi", "infra"],
                   help="Universe to analyze")
    ap.add_argument("--out_base", default=".", 
                   help="Base output directory")
    ap.add_argument("--n_boot", type=int, default=2000,
                   help="Number of bootstrap samples")
    ap.add_argument("--block", type=int, default=21,
                   help="Block size for moving block bootstrap (trading days)")
    ap.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    ap.add_argument("--var_level", type=float, default=0.95,
                   help="Confidence level for VaR/CVaR")
    ap.add_argument("--target", default="PPO",
                   help="Target strategy for pairwise comparisons")
    ap.add_argument("--verbose", action="store_true",
                   help="Enable verbose output")
    
    args = ap.parse_args()
    
    # Setup directories
    res_dir = os.path.join(args.out_base, "results", args.universe)
    
    print("\n" + "="*80)
    print(f"BOOTSTRAP CONFIDENCE INTERVAL GENERATION")
    print(f"Universe: {args.universe.upper()}")
    print("="*80)
    
    # Verify directory structure
    if not os.path.exists(res_dir):
        raise FileNotFoundError(f"Results directory not found: {res_dir}")
    
    ensure_dir(res_dir)
    
    # Verify required files
    if not verify_files_exist(res_dir):
        print("\n[ERROR] Missing required files. Please run training and evaluation first.")
        return
    
    # Initialize random generator
    rng = np.random.default_rng(args.seed)
    
    # ================ Load Data ================== #
    
    print("\n" + "-"*40)
    print("Loading data...")
    print("-"*40)
    
    # Load returns (already net of costs)
    returns_path = os.path.join(res_dir, "all_returns.csv")
    df_returns = load_returns_long(returns_path)
    
    # Convert to wide format by group
    wide = returns_wide_by_group(df_returns)
    
    # Load risk-free rate
    rf_series = load_risk_free_series(res_dir)
    
    # Load index returns (for information ratio)
    idx_series = load_index_returns(res_dir)
    
    # Load diagnostics
    diag_path = os.path.join(res_dir, "all_diagnostics.csv")
    df_diag = load_diagnostics_long(diag_path)
    
    # ============== Compute Bootstrap CIs =============== #
    
    print("\n" + "-"*40)
    print(f"Computing bootstrap CIs (B={args.n_boot}, block={args.block})...")
    print("-"*40)
    
    ci_results = {}
    
    for group in wide.columns:
        print(f"\nProcessing {group}...")
        
        # Get returns for this group
        r = wide[group].dropna().values
        
        # Align risk-free rate
        group_dates = wide[group].dropna().index
        rf = rf_series.reindex(group_dates).fillna(0.0).values if not rf_series.empty else np.zeros(len(r))
        
        # Align benchmark (index)
        bench = idx_series.reindex(group_dates).values if idx_series is not None else None
        
        # Compute CIs
        cis = compute_metric_cis(
            r, rf, bench,
            B=args.n_boot, 
            block=args.block, 
            rng=rng,
            var_level=args.var_level
        )
        
        ci_results[group] = cis
        
        if args.verbose:
            print(f"  CAGR: {cis['CAGR']['mean']:.3%} [{cis['CAGR']['lo']:.3%}, {cis['CAGR']['hi']:.3%}]")
            print(f"  Sharpe: {cis['Sharpe']['mean']:.3f} [{cis['Sharpe']['lo']:.3f}, {cis['Sharpe']['hi']:.3f}]")
    
    # ============= Diagnostic Metric CIs ================ #
    
    if df_diag is not None:
        print("\n" + "-"*40)
        print("Computing diagnostic metric CIs...")
        print("-"*40)
        
        for metric in ["turnover", "hhi", "trade_cost"]:
            print(f"\nProcessing {metric}...")
            
            metric_cis = compute_diagnostic_cis(
                df_diag, metric, 
                block=args.block, 
                B=args.n_boot, 
                rng=rng
            )
            
            # Add to results
            for group, stats in metric_cis.items():
                if group not in ci_results:
                    ci_results[group] = {}
                
                # Store as capitalized metric name
                metric_name = f"Avg_{metric.capitalize()}"
                ci_results[group][metric_name] = {
                    "lo": stats["ci_mean_lo"],
                    "hi": stats["ci_mean_hi"],
                    "mean": stats["mean"],
                    "std": stats["std"]
                }
    
    # =============== Save Results ================== #
    
    print("\n" + "-"*40)
    print("Saving results...")
    print("-"*40)
    
    # Save confidence intervals
    ci_path = os.path.join(res_dir, "confidence_intervals.json")
    with open(ci_path, "w") as f:
        json.dump(ci_results, f, indent=2, sort_keys=True)
    print(f"  Saved CIs to: {ci_path}")
    
    # ============== Statistical Tests ================ #
    
    print("\n" + "-"*40)
    print(f"Performing pairwise tests ({args.target} vs others)...")
    print("-"*40)
    
    pairwise_df = pairwise_tests(wide, args.target)
    
    if not pairwise_df.empty:
        pairwise_path = os.path.join(res_dir, "pairwise_tests.csv")
        pairwise_df.to_csv(pairwise_path, index=False)
        print(f"  Saved pairwise tests to: {pairwise_path}")
        
        # Print summary
        print(f"\n  Pairwise test summary:")
        for _, row in pairwise_df.iterrows():
            sig = "***" if row["DM_pval"] < 0.01 else "**" if row["DM_pval"] < 0.05 else "*" if row["DM_pval"] < 0.1 else ""
            print(f"    {row['Strategy_A']} vs {row['Strategy_B']}: "
                  f"Δ={row['Mean_Diff']:.3%}, DM p={row['DM_pval']:.3f} {sig}")
    
    # ============== Correlation Matrix ================ #
    
    print("\n" + "-"*40)
    print("Computing correlation matrix...")
    print("-"*40)
    
    corr_matrix = wide.corr()
    corr_path = os.path.join(res_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_path)
    print(f"  Saved correlations to: {corr_path}")
    
    # ================ Summary Report ================== #
    
    print("\n" + "="*80)
    print("BOOTSTRAP CI GENERATION COMPLETE")
    print("="*80)
    
    print("\nGenerated files:")
    print(f"  1. {ci_path}")
    if not pairwise_df.empty:
        print(f"  2. {pairwise_path}")
    print(f"  3. {corr_path}")
    
    print("\nSummary statistics:")
    print(f"  Strategies analyzed: {len(ci_results)}")
    print(f"  Bootstrap samples: {args.n_boot}")
    print(f"  Block size: {args.block} days")
    
    print("\nTop performers (by Sharpe ratio):")
    sharpe_scores = [(g, ci["Sharpe"]["mean"]) for g, ci in ci_results.items()]
    sharpe_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (group, sharpe_val) in enumerate(sharpe_scores[:5], 1):
        ci = ci_results[group]["Sharpe"]
        print(f"  {i}. {group}: {sharpe_val:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]")
    
    print("\nYou can now run:")
    print(f"  python generate_summary_report.py --universe {args.universe} --analyze_costs --make_wrapper")


if __name__ == "__main__":
    main()