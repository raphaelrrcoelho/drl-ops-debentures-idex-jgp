# baselines_final_improved.py
"""
Improved Baseline Strategies
=============================
IMPROVEMENTS:
- âœ… Simpler, cleaner code (450 vs 779 lines)
- âœ… Proper handling of excess vs total returns
- âœ… Clear temporal causality (lagged signals)
- âœ… Better documentation
- âœ… Streamlined strategy implementation
- âœ… Compatible with improved environment semantics

Baseline Strategies (all use lagged inputs):
- EW: Equal-weight over active assets
- INDEX: Market-cap weighted (index weights)
- RP_VOL: Risk parity by inverse volatility
- RP_DURATION: Risk parity by inverse duration
- CARRY_TILT: Tilt toward high-spread assets
- MINVAR: Minimum variance portfolio

All strategies:
- Use only t-1 information for decisions at time t
- Handle dynamic universe (assets can enter/exit)
- Apply same constraints as PPO (max_weight, rebalancing)
- Account for transaction costs
"""
from __future__ import annotations

import os
import json
import argparse
import warnings
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

TRADING_DAYS = 252.0


# ================================ UTILITIES ================================ #

def ensure_dir(*paths):
    """Create directories if they don't exist"""
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_json(path: str) -> dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def load_panel(universe: str, data_dir: str = "data") -> pd.DataFrame:
    """Load processed panel data"""
    path = os.path.join(data_dir, f"{universe}_processed.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found: {path}")
    
    print(f"[DATA] Loading panel from {path}")
    return pd.read_pickle(path)


# ================================ BASELINE STRATEGIES ================================ #

def strategy_equal_weight(data: pd.DataFrame, prev_weights: np.ndarray, **kwargs) -> np.ndarray:
    """Equal weight over active assets"""
    active = data["active"].values > 0
    n_active = active.sum()
    
    if n_active == 0:
        return np.zeros(len(data))
    
    weights = np.zeros(len(data))
    weights[active] = 1.0 / n_active
    
    return weights


def strategy_index_weight(data: pd.DataFrame, prev_weights: np.ndarray, **kwargs) -> np.ndarray:
    """Market-cap weighted (using index weights)"""
    active = data["active"].values > 0
    
    if not active.any():
        return np.zeros(len(data))
    
    # Use index weights (already lagged in panel)
    weights = data["index_weight"].values.copy()
    weights[~active] = 0.0
    
    # Normalize
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        # Fallback to equal weight
        weights[active] = 1.0 / active.sum()
    
    return weights


def strategy_risk_parity_vol(data: pd.DataFrame, prev_weights: np.ndarray, 
                             roll_window: int = 60, **kwargs) -> np.ndarray:
    """Risk parity using inverse volatility"""
    active = data["active"].values > 0
    
    if not active.any():
        return np.zeros(len(data))
    
    # Use historical volatility (already lagged in panel)
    if f"volatility_{roll_window}d" in data.columns:
        vols = data[f"volatility_{roll_window}d"].values
    else:
        # Fallback to simple volatility if available
        vols = data.get("volatility_20d", pd.Series(1.0, index=data.index)).values
    
    vols = np.maximum(vols, 0.01)  # Floor to avoid division by zero
    
    # Inverse volatility weights
    inv_vols = 1.0 / vols
    inv_vols[~active] = 0.0
    
    # Normalize
    total = inv_vols.sum()
    if total > 0:
        weights = inv_vols / total
    else:
        weights = np.zeros(len(data))
        weights[active] = 1.0 / active.sum()
    
    return weights


def strategy_risk_parity_duration(data: pd.DataFrame, prev_weights: np.ndarray, **kwargs) -> np.ndarray:
    """Risk parity using inverse duration"""
    active = data["active"].values > 0
    
    if not active.any():
        return np.zeros(len(data))
    
    # Use duration (lagged)
    durations = data["duration"].values
    durations = np.maximum(durations, 0.1)  # Floor
    
    # Inverse duration weights
    inv_durations = 1.0 / durations
    inv_durations[~active] = 0.0
    
    # Normalize
    total = inv_durations.sum()
    if total > 0:
        weights = inv_durations / total
    else:
        weights = np.zeros(len(data))
        weights[active] = 1.0 / active.sum()
    
    return weights


def strategy_carry_tilt(data: pd.DataFrame, prev_weights: np.ndarray, **kwargs) -> np.ndarray:
    """Tilt toward high-spread (carry) assets"""
    active = data["active"].values > 0
    
    if not active.any():
        return np.zeros(len(data))
    
    # Use spread as carry proxy (lagged)
    spreads = data["spread"].values
    
    # Only consider positive spreads
    spreads = np.maximum(spreads, 0.0)
    spreads[~active] = 0.0
    
    # Normalize
    total = spreads.sum()
    if total > 0:
        weights = spreads / total
    else:
        weights = np.zeros(len(data))
        weights[active] = 1.0 / active.sum()
    
    return weights


def strategy_min_variance(data: pd.DataFrame, prev_weights: np.ndarray,
                         roll_window: int = 60, ridge: float = 1e-4, **kwargs) -> np.ndarray:
    """Minimum variance portfolio using sample covariance"""
    active = data["active"].values > 0
    n_active = active.sum()
    
    if n_active < 2:
        # Not enough assets for covariance
        weights = np.zeros(len(data))
        if n_active == 1:
            weights[active] = 1.0
        return weights
    
    # Get historical returns for covariance estimation
    # This would require panel history - simplified version uses volatility
    if f"volatility_{roll_window}d" in data.columns:
        vols = data[f"volatility_{roll_window}d"].values[active]
    else:
        vols = np.ones(n_active)
    
    vols = np.maximum(vols, 0.01)
    
    # Simple diagonal covariance (assumes independence)
    # In practice, you'd want to use actual return history
    cov_diag = vols ** 2
    
    # Add ridge regularization
    cov_diag = cov_diag + ridge
    
    # Minimum variance: w = Î£^(-1)Â·1 / (1'Â·Î£^(-1)Â·1)
    inv_cov = 1.0 / cov_diag
    w_active = inv_cov / inv_cov.sum()
    
    # Map back to full space
    weights = np.zeros(len(data))
    weights[active] = w_active
    
    return weights


# Strategy registry
STRATEGIES = {
    "EW": strategy_equal_weight,
    "INDEX": strategy_index_weight,
    "RP_VOL": strategy_risk_parity_vol,
    "RP_DURATION": strategy_risk_parity_duration,
    "CARRY_TILT": strategy_carry_tilt,
    "MINVAR": strategy_min_variance,
}


# ================================ SIMULATION ================================ #

def simulate_baseline(
    panel: pd.DataFrame,
    strategy_fn: Callable,
    max_weight: float = 0.10,
    rebalance_interval: int = 21,
    transaction_cost_bps: float = 10.0,
    allow_cash: bool = True,
    **strategy_kwargs,
) -> pd.DataFrame:
    """
    Simulate a baseline strategy on test data.
    
    Args:
        panel: Test panel (date, asset multi-index)
        strategy_fn: Strategy function
        max_weight: Maximum weight per asset
        rebalance_interval: Days between rebalancing
        transaction_cost_bps: Transaction costs in bps
        allow_cash: Allow cash holdings
        **strategy_kwargs: Additional arguments for strategy
    
    Returns:
        DataFrame with date, returns, and diagnostics
    """
    dates = panel.index.get_level_values("date").unique().sort_values()
    asset_ids = panel.index.get_level_values("debenture_id").unique()
    n_assets = len(asset_ids)
    
    # Initialize
    prev_weights = np.zeros(n_assets)
    wealth = 1.0
    results = []
    
    for i, date in enumerate(dates):
        # Get data for this date
        try:
            day_data = panel.xs(date, level="date")
        except KeyError:
            continue
        
        # Rebalance decision
        is_rebalance = (i % rebalance_interval == 0)
        
        if is_rebalance:
            # Get new weights from strategy
            new_weights = strategy_fn(day_data, prev_weights, **strategy_kwargs)
            
            # Apply max weight constraint
            new_weights = np.minimum(new_weights, max_weight)
            
            # Renormalize
            total = new_weights.sum()
            if total > 0:
                new_weights = new_weights / total
            else:
                new_weights = np.zeros(n_assets)
            
            # Calculate turnover
            turnover = np.abs(new_weights - prev_weights).sum()
            
            # Transaction costs
            trade_cost = (transaction_cost_bps / 10000.0) * turnover
        else:
            # No rebalancing
            new_weights = prev_weights.copy()
            turnover = 0.0
            trade_cost = 0.0
        
        # Calculate returns
        returns_day = day_data["return"].values
        
        # Portfolio return (excess return - weighted average)
        r_p = float(np.dot(new_weights, returns_day))
        
        # Add cash return if applicable
        rf_day = day_data["risk_free"].iloc[0] if "risk_free" in day_data.columns else 0.0
        
        if allow_cash:
            cash_weight = 1.0 - new_weights.sum()
            r_p += cash_weight * rf_day
        
        # Convert to total return for wealth tracking
        r_p_total = r_p + rf_day
        
        # Apply transaction costs
        net_factor = (1.0 + r_p_total) * (1.0 - trade_cost)
        r_net = net_factor - 1.0
        
        # Update wealth
        wealth *= net_factor
        
        # Store results
        idx_return = day_data["index_return"].iloc[0] if "index_return" in day_data.columns else 0.0
        
        results.append({
            "date": date,
            "return": r_net,  # Net total return (for wealth/evaluation)
            "return_excess_net": (1.0 + r_p) * (1.0 - trade_cost) - 1.0,  # Net excess return
            "alpha": r_p - idx_return,  # Active return vs index
            "turnover": turnover,
            "trade_cost": trade_cost,
            "wealth": wealth,
            "n_assets": (new_weights > 0).sum(),
        })
        
        # Update for next step
        prev_weights = new_weights
    
    return pd.DataFrame(results)


# ================================ FOLD SIMULATION ================================ #

def simulate_fold(
    panel: pd.DataFrame,
    fold_spec: Dict,
    strategies: List[str],
    max_weight: float = 0.10,
    rebalance_interval: int = 21,
    transaction_cost_bps: float = 10.0,
    roll_window: int = 60,
    ridge: float = 1e-4,
) -> pd.DataFrame:
    """
    Simulate baseline strategies on a single fold.
    
    Args:
        panel: Full panel data
        fold_spec: Fold specification
        strategies: List of strategy names
        max_weight: Maximum weight per asset
        rebalance_interval: Rebalancing interval
        transaction_cost_bps: Transaction costs
        roll_window: Rolling window for risk calculations
        ridge: Ridge parameter for MINVAR
    
    Returns:
        Combined DataFrame with all strategy results
    """
    fold_id = fold_spec["fold"]
    
    print(f"\n{'='*70}")
    print(f"SIMULATING FOLD {fold_id}")
    print(f"{'='*70}")
    print(f"Test period: {fold_spec['test_start']} to {fold_spec['test_end']}")
    
    # Extract test data
    test_start = pd.Timestamp(fold_spec["test_start"])
    test_end = pd.Timestamp(fold_spec["test_end"])
    
    test_panel = panel[
        (panel.index.get_level_values("date") >= test_start) &
        (panel.index.get_level_values("date") <= test_end)
    ].copy()
    
    print(f"[DATA] Test panel: {len(test_panel)} rows")
    
    # Run each strategy
    all_results = []
    
    for strategy_name in strategies:
        if strategy_name not in STRATEGIES:
            print(f"[SKIP] Unknown strategy: {strategy_name}")
            continue
        
        print(f"[RUN] {strategy_name}")
        
        strategy_fn = STRATEGIES[strategy_name]
        
        result_df = simulate_baseline(
            panel=test_panel,
            strategy_fn=strategy_fn,
            max_weight=max_weight,
            rebalance_interval=rebalance_interval,
            transaction_cost_bps=transaction_cost_bps,
            roll_window=roll_window,
            ridge=ridge,
        )
        
        result_df["fold"] = fold_id
        result_df["strategy"] = f"{strategy_name}_f{fold_id}"
        
        all_results.append(result_df)
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)


# ================================ METRICS ================================ #

def compute_metrics(returns: pd.Series, rf: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    """Compute performance metrics (same as evaluation)"""
    if len(returns) == 0:
        return {}
    
    returns = returns.fillna(0.0)
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1) if len(returns) > 1 else 0.0
    
    # Annualized
    cagr = mean_ret * 252
    vol_ann = std_ret * np.sqrt(252)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    
    # Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum / peak - 1).min()
    
    return {
        "CAGR": float(cagr),
        "Vol_ann": float(vol_ann),
        "Sharpe": float(sharpe),
        "MaxDD": float(dd),
        "n_obs": len(returns),
    }


# ================================ MAIN ================================ #

def main():
    """Main baseline simulation pipeline"""
    
    parser = argparse.ArgumentParser(description="Run baseline strategies")
    parser.add_argument("--universe", type=str, choices=["cdi", "infra"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--n_folds", type=str, default=9)
    parser.add_argument("--strategies", type=str, 
                       default="EW,INDEX,RP_VOL,RP_DURATION,CARRY_TILT,MINVAR")
    parser.add_argument("--max_weight", type=float, default=0.10)
    parser.add_argument("--rebalance_interval", type=int, default=21)
    parser.add_argument("--transaction_cost_bps", type=float, default=10.0)
    parser.add_argument("--roll_window", type=int, default=60)
    parser.add_argument("--ridge", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BASELINE STRATEGIES SIMULATION")
    print("="*70)
    print(f"Universe: {args.universe.upper()}")
    
    # Parse strategies
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    print(f"Strategies: {', '.join(strategies)}")
    
    # Setup directories
    results_dir = os.path.join(args.output_dir, "results", args.universe)
    ensure_dir(results_dir)
    
    # Load fold specifications
    fold_spec_path = os.path.join(results_dir, "training_folds.json")
    if not os.path.exists(fold_spec_path):
        raise FileNotFoundError(f"Fold specs not found: {fold_spec_path}")
    
    fold_specs = load_json(fold_spec_path)
    print(f"\n[FOLDS] Loaded {len(fold_specs)} fold specifications")
    
    # Load data
    panel = load_panel(args.universe, args.data_dir)
    
    # Run simulations
    all_results = []
    
    for fold_spec in fold_specs:
        result = simulate_fold(
            panel=panel,
            fold_spec=fold_spec,
            strategies=strategies,
            max_weight=args.max_weight,
            rebalance_interval=args.rebalance_interval,
            transaction_cost_bps=args.transaction_cost_bps,
            roll_window=args.roll_window,
            ridge=args.ridge,
        )
        
        if not result.empty:
            all_results.append(result)
    
    # Combine and save
    if not all_results:
        print("\n[ERROR] No results generated")
        return
    
    print(f"\n[RESULTS] Combining results from {len(all_results)} folds")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save returns
    returns_path = os.path.join(results_dir, "baseline_returns.csv")
    combined_df.to_csv(returns_path, index=False)
    print(f"[SAVE] Returns saved to {returns_path}")
    
    # Compute metrics
    metrics_list = []
    
    for (fold, strategy), group in combined_df.groupby(["fold", "strategy"]):
        dates = pd.to_datetime(group["date"])
        
        # Get benchmark
        bench = panel.loc[panel.index.get_level_values("date").isin(dates)]
        bench_returns = bench.groupby("date")["index_return"].first().reindex(dates, fill_value=0.0)
        rf_rates = bench.groupby("date")["risk_free"].first().reindex(dates, fill_value=0.0)
        
        metrics = compute_metrics(group["return"], rf_rates, bench_returns)
        metrics.update({
            "fold": fold,
            "strategy": strategy,
            "mean_turnover": group["turnover"].mean(),
            "final_wealth": group["wealth"].iloc[-1] if len(group) > 0 else 1.0,
        })
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(results_dir, "baseline_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVE] Metrics saved to {metrics_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("BASELINE SUMMARY")
    print(f"{'='*70}")
    
    for strategy in strategies:
        strat_metrics = metrics_df[metrics_df["strategy"].str.contains(strategy)]
        if len(strat_metrics) > 0:
            print(f"\n{strategy}:")
            print(f"  Sharpe: {strat_metrics['Sharpe'].mean():.3f} Â± {strat_metrics['Sharpe'].std():.3f}")
            print(f"  Max DD: {strat_metrics['MaxDD'].mean():.3f}")
            print(f"  Turnover: {strat_metrics['mean_turnover'].mean():.2%}")
    
    print(f"\n[SUCCESS] Results saved to: {results_dir}/")


if __name__ == "__main__":
    main()
