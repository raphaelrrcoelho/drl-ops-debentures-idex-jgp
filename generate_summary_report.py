# generate_summary_report.py
"""
Enhanced summary report generator with support for all new analyses
-------------------------------------------------------------------

This script now includes:
1. Moving block bootstrap confidence intervals for all metrics
2. Sensitivity analysis results for transaction costs and reward parameters
3. Diebold-Mariano tests for statistical significance
4. Risk exposure analysis (VaR, CVaR, drawdown durations)
5. Correlation matrices and heatmaps
6. Comprehensive LaTeX table generation

All analyses are integrated into a cohesive dissertation-ready report.
"""

from __future__ import annotations

import os
import json
import glob
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

TRADING_DAYS = 252.0


# ------------------------------- I/O utils -------------------------------- #

def first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_csv_idx(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    return pd.read_csv(path, index_col=0)

def load_csv_raw(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    return pd.read_csv(path)

def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def save_table_tex_and_csv(df: pd.DataFrame, tex_path: str, csv_path: str, 
                          float_fmt="%.4f", caption: str = "", label: str = ""):
    df.to_csv(csv_path, float_format=float_fmt, index=True)
    
    # Create LaTeX table with formatting
    latex_str = df.to_latex(escape=False, float_format=lambda x: float_fmt % x)
    
    # Add table environment
    if caption:
        latex_str = latex_str.replace("\\begin{tabular}", "\\begin{table}[ht]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\begin{tabular}")
        latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}")
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)


# ------------------------------ Format helpers ---------------------------- #

def returns_to_wide(df_returns: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_returns is None or df_returns.empty:
        return None

    d = df_returns.copy()
    lower = {c.lower(): c for c in d.columns}
    if "strategy" in lower and "return" in lower:
        date_col = lower.get("date", "date")
        strat_col = lower["strategy"]
        ret_col = lower["return"]
        if date_col in d.columns:
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d[ret_col] = pd.to_numeric(d[ret_col], errors="coerce")
        wide = d.pivot_table(index=date_col, columns=strat_col, values=ret_col, aggfunc="first").sort_index()
    else:
        if not isinstance(d.index, pd.DatetimeIndex):
            try:
                d.index = pd.to_datetime(d.index, errors="coerce")
            except Exception:
                pass
        wide = d

    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide = wide.loc[:, wide.notna().any(axis=0)]
    wide = wide.sort_index()
    return wide

def metrics_to_long(df_metrics: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_metrics is None:
        return None
    df = df_metrics.copy()
    if "strategy" not in df.columns:
        df = df.reset_index().rename(columns={"index": "strategy"})
    return df

def pick_strategy_order(columns: List[str], top_k: int) -> List[str]:
    priority = [c for c in columns if c.upper().startswith("PPO")]
    baseline_pref = [n for n in ["INDEX","EW","RP_VOL","RP_DURATION","CARRY_TILT","MINVAR"] if n in columns]
    seen, ordered = set(), []
    for c in priority + baseline_pref + columns:
        if c not in seen and c in columns:
            seen.add(c); ordered.append(c)
    return ordered[:max(3, top_k)]

def wealth_from_returns(wide_returns: pd.DataFrame) -> pd.DataFrame:
    wr = wide_returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (1.0 + wr).cumprod()

def last_wealth_rank(wide_returns: pd.DataFrame, k: int) -> List[str]:
    w = wealth_from_returns(wide_returns)
    if w.empty: return []
    return list(w.iloc[-1].sort_values(ascending=False).index[:k])

def canonical_group(name: str) -> str:
    s = str(name).upper()
    if s.startswith("PPO"):
        return "PPO"
    return s.split("_")[0]

def group_returns(wide_returns: pd.DataFrame) -> pd.DataFrame:
    groups = {}
    for col in wide_returns.columns:
        g = canonical_group(col)
        if g not in groups:
            groups[g] = []
        groups[g].append(col)
    out = {}
    for g, cols in groups.items():
        out[g] = wide_returns[cols].mean(axis=1)
    df = pd.DataFrame(out).sort_index()
    return df

def load_bootstrap_cis(ci_path: str) -> Optional[Dict]:
    if not os.path.exists(ci_path):
        return None
    try:
        with open(ci_path, 'r') as f:
            return json.load(f)
    except:
        return None

def format_with_ci(point: float, ci_low: float, ci_high: float, fmt: str = "%.3f") -> str:
    return f"{fmt % point} ({fmt % ci_low}, {fmt % ci_high})"

# ------------------------------ Plot helpers ------------------------------ #

def plot_cum_wealth(returns_df: pd.DataFrame, cis: Optional[Dict], fig_path: str, 
                   title: str, logy: bool = False, include_ci: bool = True):
    plt.figure(figsize=(10, 6))
    wealth = wealth_from_returns(returns_df)
    
    # Define a color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(wealth.columns)))
    
    for i, col in enumerate(wealth.columns):
        plt.plot(wealth.index, wealth[col], label=col, color=colors[i], linewidth=2)
        
        # Add confidence intervals if available
        if include_ci and cis and col in cis and "CAGR" in cis[col]:
            # Convert CAGR CI to wealth CI
            ci_data = cis[col]["CAGR"]
            point = wealth[col].iloc[-1]
            ci_low = point * (1 + ci_data.get("lo", 0)) / (1 + ci_data.get("hi", 0))
            ci_high = point * (1 + ci_data.get("hi", 0)) / (1 + ci_data.get("lo", 0))
            
            # Add shaded area for the final value uncertainty
            plt.fill_betweenx([ci_low, ci_high], wealth.index[-1] - pd.Timedelta(days=5), 
                             wealth.index[-1] + pd.Timedelta(days=5), 
                             color=colors[i], alpha=0.2)
    
    plt.title(title, pad=20)
    plt.xlabel("Date")
    plt.ylabel("Wealth (× initial)")
    if logy:
        plt.yscale("log")
    plt.legend(ncol=2, fontsize=9, loc='upper left' if logy else 'best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_diag_timeseries(diag_long: pd.DataFrame, metrics: List[str], fig_path: str, 
                        title: str, cis: Optional[Dict] = None):
    if diag_long is None or diag_long.empty:
        return
        
    plt.figure(figsize=(11, 4.5))
    diag_long = diag_long.copy()
    diag_long["date"] = pd.to_datetime(diag_long["date"], errors="coerce")
    sel = diag_long[diag_long["metric"].str.lower().isin([m.lower() for m in metrics])]
    if sel.empty:
        return
        
    colors = plt.cm.Set2(np.linspace(0, 1, len(sel["strategy"].unique())))
    
    for i, ((metric, strategy), g) in enumerate(sel.groupby(["metric","strategy"])):
        g = g.sort_values("date")
        plt.plot(g["date"], g["value"], label=f"{strategy} · {metric}", color=colors[i % len(colors)])
    
    plt.title(title, pad=20)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend(ncol=2, fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(corr: pd.DataFrame, fig_path: str, title: str):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect='auto')
    plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
    plt.title(title, pad=20)
    
    # Add value annotations
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha="center", va="center", 
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=8)
    
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sensitivity_analysis(sensitivity_data: pd.DataFrame, param_name: str, 
                             metric_name: str, fig_path: str, title: str):
    if sensitivity_data is None or sensitivity_data.empty:
        return
        
    plt.figure(figsize=(8, 5))
    
    for strategy in sensitivity_data["strategy"].unique():
        strat_data = sensitivity_data[sensitivity_data["strategy"] == strategy]
        plt.plot(strat_data[param_name], strat_data[metric_name], 
                marker='o', label=strategy, linewidth=2)
    
    plt.xlabel(param_name.replace("_", " ").title())
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title, pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------- Statistical add-ons ---------------------------- #

def _phi_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))

def dm_test_returns(r1: pd.Series, r2: pd.Series, h: int = 5) -> tuple[float, float]:
    x = pd.concat([r1, r2], axis=1).dropna().to_numpy()
    if x.shape[0] < max(20, h + 5):
        return (float("nan"), float("nan"))
    d = x[:,1] - x[:,0]
    T = d.shape[0]
    dbar = d.mean()
    K = max(1, min(h, T - 1))
    gamma0 = np.var(d, ddof=0)
    var = gamma0
    for k in range(1, K + 1):
        cov = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        w = 1.0 - k / (K + 1.0)
        var += 2.0 * w * cov
    var /= T
    if var <= 0:
        return (float("nan"), float("nan"))
    stat = dbar / math.sqrt(var)
    p = 2.0 * (1.0 - _phi_cdf(abs(stat)))
    return (float(stat), float(p))

def run_dm_matrix(wide_group_returns: pd.DataFrame, pairs: list[str], h: int = 5) -> pd.DataFrame:
    rows = []
    for p in pairs:
        if ":" not in p:
            continue
        a, b = p.split(":")
        a, b = a.strip(), b.strip()
        if a in wide_group_returns.columns and b in wide_group_returns.columns:
            stat, pval = dm_test_returns(wide_group_returns[a], wide_group_returns[b], h=h)
            rows.append({"A": a, "B": b, "DM_stat": stat, "p_value": pval})
    return pd.DataFrame(rows)

def exposures_ols(wide_group_returns: pd.DataFrame, y_name: str, x_names: list[str]) -> pd.DataFrame:
    d = wide_group_returns[[c for c in x_names + [y_name] if c in wide_group_returns.columns]].dropna()
    if d.empty or len(x_names) == 0:
        return pd.DataFrame()
    X = d[x_names].to_numpy()
    y = d[y_name].to_numpy()
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ beta
    resid = y - yhat
    sst = ((y - y.mean()) ** 2).sum()
    ssr = ((yhat - y.mean()) ** 2).sum()
    r2 = ssr / sst if sst > 1e-12 else np.nan
    out = {"alpha": beta[0], "R2": r2}
    for i, name in enumerate(x_names):
        out[f"beta_{name}"] = beta[i + 1]
    return pd.DataFrame([out])

def var_cvar(series: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    x = series.dropna().to_numpy()
    if x.size < 10:
        return (float("nan"), float("nan"))
    q = np.quantile(x, 1.0 - alpha)
    tail = x[x <= q]
    cvar = tail.mean() if tail.size > 0 else np.nan
    return (-q, -cvar)

def drawdown_durations(wealth: pd.Series) -> tuple[float, int]:
    w = wealth.dropna().to_numpy()
    if w.size == 0:
        return (float("nan"), 0)
    peak = -np.inf
    cur = 0
    durations = []
    for val in w:
        if val >= peak:
            if cur > 0:
                durations.append(cur)
            peak = val
            cur = 0
        else:
            cur += 1
    if cur > 0:
        durations.append(cur)
    return (np.mean(durations) if durations else 0.0, int(max(durations) if durations else 0))

def risk_extras_table(wide_group_returns: pd.DataFrame, alpha: float = 0.95) -> pd.DataFrame:
    rows = []
    W = wealth_from_returns(wide_group_returns)
    for col in wide_group_returns.columns:
        v, cv = var_cvar(wide_group_returns[col], alpha=alpha)
        mean_d, max_d = drawdown_durations(W[col])
        rows.append({"strategy": col, "VaR_95": v, "CVaR_95": cv,
                     "DD_Duration_mean_days": mean_d, "DD_Duration_max_days": max_d})
    return pd.DataFrame(rows).set_index("strategy")

def correlation_matrix_table(wide_group_returns: pd.DataFrame) -> pd.DataFrame:
    C = wide_group_returns.corr()
    C.index.name = "strategy"
    return C

# ------------------------- Sensitivity analysis --------------------------- #

def load_sensitivity_results(results_dir: str, pattern: str = "sensitivity_*.csv") -> pd.DataFrame:
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return pd.DataFrame()
    
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Extract strategy and parameters from filename
            filename = os.path.basename(file)
            parts = filename.replace("sensitivity_", "").replace(".csv", "").split("_")
            if "f" in parts and "s" in parts:
                # PPO file: sensitivity_PPO_f0_s0.csv
                strategy = parts[0]
                fold = parts[1].replace("f", "")
                seed = parts[2].replace("s", "")
                df["strategy"] = strategy
                df["fold"] = fold
                df["seed"] = seed
            else:
                # Baseline file: sensitivity_EW_f0.csv
                strategy = parts[0]
                fold = parts[1].replace("f", "")
                df["strategy"] = strategy
                df["fold"] = fold
                df["seed"] = -1
                
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def analyze_sensitivity_data(sensitivity_data: pd.DataFrame, param_col: str = "param_transaction_cost_bps") -> pd.DataFrame:
    if sensitivity_data.empty:
        return pd.DataFrame()
    
    # Group by strategy and parameter value
    grouped = sensitivity_data.groupby(["strategy", param_col])
    
    results = []
    for (strategy, param_val), group in grouped:
        if group.empty:
            continue
            
        # Calculate mean and std for each metric
        metrics = ["Sharpe", "CAGR", "Vol_ann", "MaxDD"]
        row = {"strategy": strategy, param_col: param_val}
        for metric in metrics:
            if metric in group.columns:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std()
        
        results.append(row)
    
    return pd.DataFrame(results)

# Add these imports to the top of generate_summary_report.py
import matplotlib.pyplot as plt
import seaborn as sns

def generate_descriptive_analysis_report(ppo_btr, panel, results_dir, universe_name):
    """
    Generates a descriptive analysis report for the PPO agent's behavior.
    
    Args:
        ppo_btr: BacktestResult object for the PPO strategy.
        panel: The full data panel with asset info and market data.
        results_dir: Directory to save the plots.
        universe_name: Name of the universe for titles.
    """
    print("\n--- Generating Descriptive Analysis Report for PPO Agent ---")
    
    weights = ppo_btr.get_weights().rename(columns={'weight': 'ppo_weight'})
    
    # Merge weights with panel data to get asset characteristics
    analysis_df = weights.join(panel, on=['date', 'asset_id'], how='left')
    analysis_df = analysis_df[analysis_df['ppo_weight'] > 1e-6] # Focus on actual holdings

    # --- 1. Sector Allocation Over Time ---
    sector_alloc = analysis_df.groupby(['date', 'sector'])['ppo_weight'].sum().unstack().fillna(0)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sector_alloc.plot.area(ax=ax, linewidth=0.5, colormap='viridis')
    ax.set_title(f'PPO Agent: Sector Allocation Over Time ({universe_name})', fontsize=16)
    ax.set_ylabel('Portfolio Weight')
    ax.set_xlabel('Date')
    ax.legend(title='Sector', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(results_dir / f'ppo_sector_allocation_{universe_name.lower()}.png', dpi=300)
    plt.close()
    print(f"Saved sector allocation plot to {results_dir}")

    # --- 2. Top 10 High-Conviction Holdings ---
    # Identify assets most frequently in the top 5 positions
    top_5_holdings = analysis_df.groupby('date').apply(lambda x: x.nlargest(5, 'ppo_weight')['asset_id']).reset_index(drop=True)
    top_conviction_assets = top_5_holdings.value_counts().nlargest(10)
    
    top_holdings_df = pd.DataFrame({
        'Asset ID': top_conviction_assets.index,
        'Days in Top 5': top_conviction_assets.values
    })
    
    # Get average weight when held
    avg_weights = analysis_df[analysis_df['asset_id'].isin(top_holdings_df)]
    avg_weights = avg_weights.groupby('asset_id')['ppo_weight'].mean()
    
    top_holdings_df = top_holdings_df.map(avg_weights)
    top_holdings_df = top_holdings_df.merge(panel[['asset_id', 'issuer']].drop_duplicates(), on='asset_id', how='left')
    
    print("\nTop 10 High-Conviction Holdings (Most frequently in Top 5 positions):")
    print(top_holdings_df.to_string(index=False))

    # --- 3. Macro Context and Allocation to Inflation-Linked Bonds ---
    market_data = panel.index.get_level_values('date').unique()
    cdi_rate = panel.loc[market_data]['cdi_rate'].groupby('date').first() * 252 * 100 # Annualized %
    ipca_rate = panel.loc[market_data]['ipca_yoy'].groupby('date').first() * 100 # YoY %
    
    # Identify inflation-linked bonds (assuming 'IPCA' is in the asset_id or another column)
    analysis_df['is_inflation_linked'] = analysis_df['asset_id'].str.contains('IPCA', case=False)
    inflation_alloc = analysis_df.groupby(['date', 'is_inflation_linked'])['ppo_weight'].sum().unstack().fillna(0)
    inflation_alloc.columns = [str(c) for c in inflation_alloc.columns]
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Panel A: PPO Cumulative Wealth
    ppo_btr.get_cumulative_returns().plot(ax=axes[0], title=f'PPO Performance and Macro Context ({universe_name})')
    axes[0].set_ylabel('Cumulative Wealth')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Panel B: Allocation to IPCA vs CDI bonds
    inflation_alloc.plot.area(ax=axes[1], stacked=True, colormap='coolwarm')
    axes[1].set_ylabel('Allocation Type')
    axes[1].set_ylim(0, 1)
    axes[1].legend(title='Bond Type')
    
    # Panel C: Macro Indicators
    cdi_rate.plot(ax=axes[2], label='CDI Rate (Ann. %)', color='darkblue')
    ipca_rate.plot(ax=axes[2], label='IPCA Inflation (YoY %)', color='darkred', linestyle='--')
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'ppo_macro_context_{universe_name.lower()}.png'), dpi=300)
    plt.close()
    print(f"Saved macro context plot to {results_dir}")
    print("--- Descriptive Analysis Report Complete ---")


# ------------------------------- Main routine ------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Comprehensive summary report generator")
    ap.add_argument("--universe", required=True, choices=["cdi","infra"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_base", default=".")
    ap.add_argument("--top_strategies", type=int, default=12)
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--exposure_targets", type=str, default="EW,INDEX,MINVAR,RP_VOL,CARRY_TILT")
    ap.add_argument("--dm_pairs", type=str, default="PPO:EW,PPO:INDEX,PPO:MINVAR,PPO:RP_VOL")
    ap.add_argument("--var_alpha", type=float, default=0.95)
    ap.add_argument("--make_wrapper", action="store_true")
    ap.add_argument("--save_corr_heatmap", action="store_true")
    ap.add_argument("--cost_delta_bps", type=str, default="")
    ap.add_argument("--include_sensitivity", action="store_true", help="Include sensitivity analysis results")
    args = ap.parse_args()

    res_dir = os.path.join(args.out_base, "results", args.universe)
    fig_dir = os.path.join(res_dir, "figures")
    ensure_dir(fig_dir)

    # Load artifacts
    df_ret_long = load_csv_raw(os.path.join(res_dir, "all_returns.csv"))
    df_metrics_long = load_csv_raw(os.path.join(res_dir, "all_metrics.csv"))
    df_diag_long = load_csv_raw(os.path.join(res_dir, "all_diagnostics.csv"))
    conf_json = load_json(os.path.join(res_dir, "confidence_intervals.json"))
    folds_json = load_json(os.path.join(res_dir, "training_folds.json"))
    cfg_json = load_json(os.path.join(res_dir, "training_config.json"))
    rf_df = load_csv_raw(os.path.join(res_dir, "risk_free_used.csv"))

    # Load sensitivity data if requested
    sensitivity_data = None
    if args.include_sensitivity:
        sensitivity_data = load_sensitivity_results(res_dir)
    
    panel_path = os.path.join(args.data_dir, f"{args.universe}_panel.parquet")
    panel = pd.read_parquet(panel_path) if os.path.exists(panel_path) else None
    if panel is None:
        print(f" Main data panel not found at {panel_path}. Descriptive analysis will be skipped.")

    # Returns wide (by run), and group-averaged returns
    wide = returns_to_wide(df_ret_long)
    if wide is None or wide.empty:
        print("[ERROR] No returns found.")
        return
    wide_groups = group_returns(wide)

    # Load bootstrap CIs
    cis = load_bootstrap_cis(os.path.join(res_dir, "confidence_intervals.json"))

    # Top lines for cum wealth plot (group level)
    top_cols = pick_strategy_order(list(wide_groups.columns), args.top_strategies)
    plot_cum_wealth(wide_groups[top_cols], cis, os.path.join(fig_dir, "drl_vs_baselines_cumulative.png"),
                    "Cumulative Wealth — PPO vs Baselines (Group-Averaged)", logy=args.logy)

    # Diagnostics (turnover, HHI) if present
    if df_diag_long is not None and not df_diag_long.empty:
        plot_diag_timeseries(df_diag_long, ["turnover","hhi"], 
                            os.path.join(fig_dir, "diagnostics_turnover_hhi.png"),
                            "Diagnostics: Turnover & HHI (by Run/Strategy)")

    # ---------------------- Summary metrics table (groups) ----------------------
    if df_metrics_long is not None and not df_metrics_long.empty:
        mlong = metrics_to_long(df_metrics_long)
        mlong["group"] = mlong["strategy"].map(canonical_group)
        agg_cols = [c for c in ["CAGR","Vol_ann","Sharpe","Sortino","MaxDD","Calmar"] if c in mlong.columns]
        summary = mlong.groupby("group")[agg_cols].median().sort_index()
    else:
        # Fallback: compute from group returns with rf if available
        rf = None
        if rf_df is not None and "date" in rf_df.columns and "risk_free" in rf_df.columns:
            rf = rf_df.set_index(pd.to_datetime(rf_df["date"]))["risk_free"].astype(float)
        
        def _compute_metrics(r: pd.Series, rf_series: Optional[pd.Series]):
            r = r.dropna()
            rf_al = rf_series.reindex(r.index).fillna(0.0) if rf_series is not None else pd.Series(0.0, index=r.index)
            ex = r - rf_al
            mu, sd = r.mean(), r.std(ddof=1)
            mu_ex, sd_ex = ex.mean(), ex.std(ddof=1)
            dsd = ex[ex < 0].std(ddof=1)
            wealth = (1.0 + r).cumprod()
            peak = wealth.cummax()
            dd = wealth / peak - 1.0
            mdd = float(dd.min())
            cagr = (wealth.iloc[-1]) ** (TRADING_DAYS / max(1, len(r))) - 1.0
            sharpe = (mu_ex / sd_ex) * np.sqrt(TRADING_DAYS) if sd_ex > 0 else np.nan
            sortino = (mu_ex / dsd) * np.sqrt(TRADING_DAYS) if dsd and dsd > 0 else np.nan
            vol = sd * np.sqrt(TRADING_DAYS)
            calmar = cagr / abs(mdd) if mdd < 0 else np.nan
            return cagr, vol, sharpe, sortino, mdd, calmar
        
        summary_rows = {}
        for col in wide_groups.columns:
            summary_rows[col] = _compute_metrics(wide_groups[col], rf)
        
        summary = pd.DataFrame.from_dict(
            {k: {"CAGR": v[0], "Vol_ann": v[1], "Sharpe": v[2], "Sortino": v[3], "MaxDD": v[4], "Calmar": v[5]}
             for k,v in summary_rows.items()}, orient="index"
        ).sort_index()

    # Add confidence intervals to summary table if available
    if cis:
        summary_with_ci = summary.copy()
        for metric in ["CAGR", "Vol_ann", "Sharpe", "Sortino", "MaxDD", "Calmar"]:
            if metric in summary.columns:
                ci_cols = []
                for group in summary.index:
                    if group in cis and metric in cis[group]:
                        ci_data = cis[group][metric]
                        ci_cols.append(format_with_ci(
                            summary.loc[group, metric], 
                            ci_data.get("lo", np.nan), 
                            ci_data.get("hi", np.nan)
                        ))
                    else:
                        ci_cols.append(f"{summary.loc[group, metric]:.3f}")
                summary_with_ci[metric] = ci_cols

        save_table_tex_and_csv(summary_with_ci, 
                              os.path.join(res_dir, "summary_metrics.tex"),
                              os.path.join(res_dir, "summary_metrics.csv"),
                              caption="Performance Metrics with 95% Confidence Intervals",
                              label="tab:performance_metrics")
    else:
        save_table_tex_and_csv(summary, 
                              os.path.join(res_dir, "summary_metrics.tex"),
                              os.path.join(res_dir, "summary_metrics.csv"),
                              caption="Performance Metrics",
                              label="tab:performance_metrics")

    # --------------------------- Exposures / Mimicry ---------------------------
    xnames = [s.strip() for s in args.exposure_targets.split(",") if s.strip()]
    if "PPO" in wide_groups.columns:
        X_present = [n for n in xnames if n in wide_groups.columns]
        expos = exposures_ols(wide_groups, "PPO", X_present)
        if expos is not None and not expos.empty:
            save_table_tex_and_csv(expos, 
                                  os.path.join(res_dir, "exposure_regression.tex"),
                                  os.path.join(res_dir, "exposure_regression.csv"),
                                  caption="PPO Strategy Exposure Analysis",
                                  label="tab:exposure_regression")

    # ----------------------------- DM comparisons -----------------------------
    pairs = [p.strip() for p in args.dm_pairs.split(",") if ":" in p]
    if pairs:
        dm = run_dm_matrix(wide_groups, pairs, h=5)
        if dm is not None and not dm.empty:
            save_table_tex_and_csv(dm.set_index(["A","B"]), 
                                  os.path.join(res_dir, "dm_tests.tex"),
                                  os.path.join(res_dir, "dm_tests.csv"),
                                  caption="Diebold-Mariano Test Results",
                                  label="tab:dm_tests")

    # ------------------------- Risk extras (VaR/ES, DD) -----------------------
    riskx = risk_extras_table(wide_groups, alpha=args.var_alpha)
    save_table_tex_and_csv(riskx, 
                          os.path.join(res_dir, "risk_extras.tex"),
                          os.path.join(res_dir, "risk_extras.csv"),
                          caption="Risk Measures (VaR, CVaR, Drawdown Durations)",
                          label="tab:risk_measures")

    # --------------------------- Return correlations ---------------------------
    corr = correlation_matrix_table(wide_groups)
    save_table_tex_and_csv(corr, 
                          os.path.join(res_dir, "return_correlations.tex"),
                          os.path.join(res_dir, "return_correlations.csv"),
                          caption="Return Correlation Matrix",
                          label="tab:correlation_matrix")

    if args.save_corr_heatmap:
        plot_correlation_heatmap(corr, os.path.join(fig_dir, "return_correlations.png"),
                                "Return Correlations (Group-Level)")
        
    # ------------------ DESCRIPTIVE ANALYSIS OF PPO AGENT -------------------
    # This block extracts the necessary data for the PPO agent and calls the
    # new descriptive analysis function to generate plots on sector allocation,
    # high-conviction holdings, and macro context.
    if "PPO" in wide_groups.columns and panel is not None and df_diag_long is not None:
        # We need to reconstruct a lightweight BacktestResult object or pass the raw data.
        # Here, we'll prepare the necessary DataFrames for the function.
        
        # The function expects a BacktestResult object, so we'll create a simple
        # mock object that has the methods it needs.
        class SimpleBacktestResult:
            def __init__(self, returns_series, diagnostics_df):
                self._returns = returns_series
                self._diagnostics = diagnostics_df
            def get_cumulative_returns(self):
                return (1 + self._returns.fillna(0)).cumprod()
            def get_weights(self):
                weights_df = self._diagnostics[
                    (self._diagnostics['strategy'].str.upper().str.startswith('PPO')) &
                    (self._diagnostics['metric'] == 'weight')
                ].copy()
                # The metric is 'weight', the value is the weight, and the 'variable' column holds the asset_id
                weights_df = weights_df.rename(columns={'value': 'weight', 'variable': 'asset_id'})
                weights_df['date'] = pd.to_datetime(weights_df['date'])
                return weights_df.set_index(['date', 'asset_id'])[['weight']]

        ppo_results_obj = SimpleBacktestResult(wide_groups['PPO'], df_diag_long)
        
        # Now call the function
        generate_descriptive_analysis_report(
            ppo_btr=ppo_results_obj,
            panel=panel,
            results_dir=fig_dir, # Save plots in the figures directory
            universe_name=args.universe.upper()
        )
    else:
        print("[INFO] Skipping PPO descriptive analysis because PPO results, panel, or diagnostics are missing.")

    # ------------------------- Sensitivity analysis ----------------------------
    if args.include_sensitivity and sensitivity_data is not None and not sensitivity_data.empty:
        # Analyze transaction cost sensitivity
        cost_sensitivity = analyze_sensitivity_data(sensitivity_data, "param_transaction_cost_bps")
        if not cost_sensitivity.empty:
            save_table_tex_and_csv(cost_sensitivity, 
                                  os.path.join(res_dir, "cost_sensitivity.tex"),
                                  os.path.join(res_dir, "cost_sensitivity.csv"),
                                  caption="Transaction Cost Sensitivity Analysis",
                                  label="tab:cost_sensitivity")
            
            # Plot sensitivity
            plot_sensitivity_analysis(cost_sensitivity, "param_transaction_cost_bps", "Sharpe_mean",
                                    os.path.join(fig_dir, "cost_sensitivity_sharpe.png"),
                                    "Sharpe Ratio vs Transaction Cost")
            
            plot_sensitivity_analysis(cost_sensitivity, "param_transaction_cost_bps", "CAGR_mean",
                                    os.path.join(fig_dir, "cost_sensitivity_cagr.png"),
                                    "CAGR vs Transaction Cost")

    # --------------------------- Master TeX wrapper ---------------------------
    if args.make_wrapper:
        tex = r"""
\section{Results and Analysis}

\subsection{Performance Comparison}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures/drl_vs_baselines_cumulative.png}
\caption{Cumulative wealth of PPO compared to baseline strategies. Shaded areas represent 95\% confidence intervals based on moving block bootstrap.}
\label{fig:cumulative_wealth}
\end{figure}

\input{summary_metrics.tex}

\subsection{Risk Analysis}

\input{risk_extras.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures/diagnostics_turnover_hhi.png}
\caption{Portfolio diagnostics: turnover and Herfindahl-Hirschman Index (HHI) over time.}
\label{fig:diagnostics}
\end{figure}

\subsection{Statistical Significance}

\input{dm_tests.tex}

\subsection{Correlation Analysis}

\input{return_correlations.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{figures/return_correlations.png}
\caption{Correlation matrix of strategy returns. Values close to 1 indicate high positive correlation, values close to -1 indicate high negative correlation.}
\label{fig:correlation_heatmap}
\end{figure}

\subsection{Exposure Analysis}

\input{exposure_regression.tex}

\subsection{Sensitivity Analysis}

\input{cost_sensitivity.tex}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures/cost_sensitivity_sharpe.png}
\caption{Sensitivity of Sharpe ratio to transaction costs. Higher values indicate better cost tolerance.}
\label{fig:cost_sensitivity_sharpe}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{figures/cost_sensitivity_cagr.png}
\caption{Sensitivity of CAGR to transaction costs. Steeper declines indicate higher sensitivity to costs.}
\label{fig:cost_sensitivity_cagr}
\end{figure}
"""
        with open(os.path.join(res_dir, "summary_report.tex"), "w", encoding="utf-8") as f:
            f.write(tex)

    print("[DONE] Comprehensive summary report artifacts written.")

if __name__ == "__main__":
    main()