# evaluate_final_improved.py
"""
Improved PPO Evaluation Script
===============================
IMPROVEMENTS:
- âœ… Simpler, cleaner code (350 vs 796 lines)
- âœ… Proper handling of returns from improved environment
- âœ… Clear distinction between excess and total returns
- âœ… Better logging and progress tracking
- âœ… Proper compatibility with improved environment
- âœ… Streamlined fold processing

Compatible with:
- env_final_improved.py
- train_final_improved.py
- All model checkpoints
"""
from __future__ import annotations

import os
import json
import argparse
import warnings
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Stable Baselines 3
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    raise RuntimeError("Install: pip install sb3-contrib")

# Environment
try:
    from env_final import DebentureTradingEnv, EnvConfig
except ImportError:
    raise RuntimeError("env_final.py (improved version) required")


# ================================ UTILITIES ================================ #

def ensure_dir(*paths):
    """Create directories if they don't exist"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


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


# ================================ ENVIRONMENT CREATION ================================ #

def make_eval_env(panel: pd.DataFrame, cfg: EnvConfig, seed: int = 0):
    """Create evaluation environment with action masking"""
    env = DebentureTradingEnv(panel, cfg)
    env.reset(seed=seed)
    
    # Wrap with action masker
    def mask_fn(env):
        return env.unwrapped._get_observation(env.unwrapped.t)[1]
    
    return ActionMasker(env, mask_fn)


# ================================ EVALUATION ================================ #

def evaluate_model(
    model_path: str,
    panel: pd.DataFrame,
    env_cfg: EnvConfig,
    seed: int = 0,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path: Path to saved model
        panel: Test data panel
        env_cfg: Environment configuration
        seed: Random seed
        deterministic: Use deterministic policy
    
    Returns:
        DataFrame with date, returns, and diagnostics
    """
    print(f"[EVAL] Loading model from {model_path}")
    
    # Create environment
    env = make_eval_env(panel, env_cfg, seed=seed)
    
    # Load model
    try:
        model = MaskablePPO.load(model_path, env=env)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        env.close()
        raise
    
    # Run evaluation
    print(f"[EVAL] Running evaluation (deterministic={deterministic})")
    
    obs, info = env.reset(seed=seed)
    done = False
    
    results = []
    step = 0
    
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store results
        if "date" in info:
            results.append({
                "date": info["date"],
                "return": info.get("portfolio_return", 0.0),  # Net total return
                "return_excess_net": info.get("portfolio_return_excess_net", 0.0),
                "alpha": info.get("alpha", 0.0),
                "turnover": info.get("turnover", 0.0),
                "hhi": info.get("hhi", 0.0),
                "wealth": info.get("wealth", 1.0),
                "drawdown": info.get("drawdown", 0.0),
                "trade_cost": info.get("trade_cost", 0.0),
            })
        
        step += 1
        if step % 100 == 0:
            print(f"  Step {step}: wealth={info.get('wealth', 1.0):.3f}")
    
    env.close()
    
    print(f"[EVAL] Completed {len(results)} steps")
    
    return pd.DataFrame(results)


# ================================ FOLD EVALUATION ================================ #

def evaluate_fold(
    universe: str,
    panel: pd.DataFrame,
    fold_spec: Dict,
    seed: int,
    env_cfg: EnvConfig,
    model_dir: str,
    deterministic: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Evaluate PPO model on a single fold.
    
    Args:
        universe: Universe name
        panel: Full panel data
        fold_spec: Fold specification
        seed: Random seed
        env_cfg: Environment configuration
        model_dir: Directory with trained models
        deterministic: Use deterministic policy
    
    Returns:
        DataFrame with evaluation results or None if model not found
    """
    fold_id = fold_spec["fold"]
    
    print(f"\n{'='*70}")
    print(f"EVALUATING FOLD {fold_id} | SEED {seed}")
    print(f"{'='*70}")
    print(f"Test period: {fold_spec['test_start']} to {fold_spec['test_end']}")
    print(f"Test days: {fold_spec['test_days']}")
    
    # Check if model exists
    model_path = os.path.join(model_dir, f"model_fold_{fold_id}_seed_{seed}.zip")
    if not os.path.exists(model_path):
        print(f"[SKIP] Model not found: {model_path}")
        return None
    
    # Extract test data
    test_start = pd.Timestamp(fold_spec["test_start"])
    test_end = pd.Timestamp(fold_spec["test_end"])
    
    test_panel = panel[
        (panel.index.get_level_values("date") >= test_start) &
        (panel.index.get_level_values("date") <= test_end)
    ].copy()
    
    print(f"[DATA] Test panel: {len(test_panel)} rows")
    
    # Evaluate
    try:
        results_df = evaluate_model(
            model_path=model_path,
            panel=test_panel,
            env_cfg=env_cfg,
            seed=seed,
            deterministic=deterministic,
        )
        
        # Add fold and seed info
        results_df["fold"] = fold_id
        results_df["seed"] = seed
        results_df["strategy"] = f"PPO_f{fold_id}_s{seed}"
        
        # Calculate cumulative wealth
        results_df["cum_wealth"] = (1 + results_df["return"]).cumprod()
        
        return results_df
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ================================ METRICS COMPUTATION ================================ #

def compute_metrics(returns: pd.Series, rf: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    """
    Compute performance metrics.
    
    Args:
        returns: Strategy returns (net total returns for PPO)
        rf: Risk-free rate
        benchmark: Benchmark returns (index)
    
    Returns:
        Dictionary of metrics
    """
    if len(returns) == 0:
        return {}
    
    # Clean data
    returns = returns.fillna(0.0)
    rf = rf.fillna(0.0) if len(rf) > 0 else pd.Series([0.0] * len(returns))
    benchmark = benchmark.fillna(0.0) if len(benchmark) > 0 else returns
    
    # Align series
    returns, rf, benchmark = returns.align(rf, benchmark, join='inner')
    
    # Basic statistics
    n = len(returns)
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1) if n > 1 else 0.0
    
    # Annualized metrics
    cagr = mean_ret * 252
    vol_ann = std_ret * np.sqrt(252)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    
    # Downside deviation
    downside = returns[returns < 0].std(ddof=1) if (returns < 0).sum() > 1 else std_ret
    sortino = (mean_ret / downside * np.sqrt(252)) if downside > 0 else 0.0
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns / running_max - 1)
    max_dd = drawdown.min()
    
    # Calmar ratio
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0
    
    # Alpha and beta vs benchmark
    bench_mean = benchmark.mean()
    if std_ret > 0 and benchmark.std() > 0:
        cov = np.cov(returns, benchmark)[0, 1]
        beta = cov / benchmark.var()
        alpha = mean_ret - beta * bench_mean
    else:
        beta = 0.0
        alpha = mean_ret - bench_mean
    
    # Information ratio
    active_ret = returns - benchmark
    tracking_error = active_ret.std(ddof=1) if n > 1 else 0.0
    ir = (active_ret.mean() / tracking_error * np.sqrt(252)) if tracking_error > 0 else 0.0
    
    # Capture ratios
    pos_bench = benchmark > 0
    neg_bench = benchmark < 0
    up_capture = (returns[pos_bench].mean() / benchmark[pos_bench].mean()) if pos_bench.sum() > 0 else 1.0
    down_capture = (returns[neg_bench].mean() / benchmark[neg_bench].mean()) if neg_bench.sum() > 0 else 1.0
    
    # Additional stats
    hit_rate = (returns > 0).mean()
    skew = returns.skew() if n > 2 else 0.0
    kurt = returns.kurtosis() if n > 3 else 0.0
    
    return {
        "CAGR": float(cagr),
        "Vol_ann": float(vol_ann),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDD": float(max_dd),
        "Calmar": float(calmar),
        "Alpha_daily": float(alpha * 252),  # Annualized
        "Beta": float(beta),
        "Information_Ratio": float(ir),
        "Up_capture": float(up_capture),
        "Down_capture": float(down_capture),
        "Hit_rate": float(hit_rate),
        "Skew": float(skew),
        "Kurtosis": float(kurt),
    }


# ================================ MAIN ================================ #

def main():
    """Main evaluation pipeline"""
    
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO models"
    )
    parser.add_argument("--universe", type=str, choices=["cdi", "infra"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--model_dir", type=str, default=None,
                       help="Model directory (default: output_dir/models/universe/ppo)")
    parser.add_argument("--n_folds", type=int, default=9)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy")
    parser.add_argument("--strategies", type=str, default="PPO",
                       help="Strategies to evaluate (comma-separated)")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("PPO MODEL EVALUATION")
    print("="*70)
    print(f"Universe: {args.universe.upper()}")
    print(f"Deterministic: {args.deterministic}")
    
    # Setup directories
    if args.model_dir is None:
        model_dir = os.path.join(args.output_dir, "models", args.universe, "ppo")
    else:
        model_dir = args.model_dir
    
    results_dir = os.path.join(args.output_dir, "results", args.universe)
    ensure_dir(results_dir)
    
    print(f"Model directory: {model_dir}")
    print(f"Results directory: {results_dir}")
    
    # Load fold specifications
    fold_spec_path = os.path.join(results_dir, "training_folds.json")
    if not os.path.exists(fold_spec_path):
        raise FileNotFoundError(f"Fold specs not found: {fold_spec_path}")
    
    fold_specs = load_json(fold_spec_path)
    print(f"\n[FOLDS] Loaded {len(fold_specs)} fold specifications")
    
    # Load configuration
    config_path = os.path.join(results_dir, "training_config.json")
    if os.path.exists(config_path):
        config = load_json(config_path)
        env_cfg_dict = config.get("env_config", {})
    else:
        print("[WARN] No training config found, using defaults")
        env_cfg_dict = {}
    
    # Create environment config
    env_cfg = EnvConfig(**{k: v for k, v in env_cfg_dict.items() if k in EnvConfig.__dataclass_fields__})
    
    print(f"\n[CONFIG] Environment Configuration:")
    print(f"  Max assets: {env_cfg.max_assets}")
    print(f"  Rebalance interval: {env_cfg.rebalance_interval}")
    print(f"  Transaction costs: {env_cfg.transaction_cost_bps} bps")
    
    # Load data
    print(f"\n[DATA] Loading {args.universe} universe")
    panel = load_panel(args.universe, args.data_dir)
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    # Evaluation schedule
    total_evals = len(fold_specs) * len(seeds)
    print(f"\n[SCHEDULE] Evaluation Schedule:")
    print(f"  Folds: {len(fold_specs)}")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    print(f"  Total evaluations: {total_evals}")
    
    # Run evaluations
    all_results = []
    completed = 0
    skipped = 0
    
    for fold_spec in fold_specs:
        for seed in seeds:
            result = evaluate_fold(
                universe=args.universe,
                panel=panel,
                fold_spec=fold_spec,
                seed=seed,
                env_cfg=env_cfg,
                model_dir=model_dir,
                deterministic=args.deterministic,
            )
            
            if result is not None:
                all_results.append(result)
                completed += 1
            else:
                skipped += 1
    
    # Combine results
    if not all_results:
        print("\n[ERROR] No results to save!")
        return
    
    print(f"\n[RESULTS] Combining results from {completed} evaluations")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save returns
    returns_path = os.path.join(results_dir, "ppo_returns.csv")
    combined_df.to_csv(returns_path, index=False)
    print(f"[SAVE] Returns saved to {returns_path}")
    
    # Compute and save metrics per fold
    metrics_list = []
    
    for (fold, seed), group in combined_df.groupby(["fold", "seed"]):
        # Get benchmark
        dates = pd.to_datetime(group["date"])
        bench = panel.loc[panel.index.get_level_values("date").isin(dates)]
        bench_returns = bench.groupby("date")["index_return"].first().reindex(dates, fill_value=0.0)
        rf_rates = bench.groupby("date")["risk_free"].first().reindex(dates, fill_value=0.0)
        
        # Compute metrics
        metrics = compute_metrics(
            group["return"],
            rf_rates,
            bench_returns,
        )
        
        metrics.update({
            "fold": fold,
            "seed": seed,
            "strategy": f"PPO_f{fold}_s{seed}",
            "n_days": len(group),
            "final_wealth": group["cum_wealth"].iloc[-1] if len(group) > 0 else 1.0,
            "mean_turnover": group["turnover"].mean(),
            "mean_hhi": group["hhi"].mean(),
        })
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(results_dir, "ppo_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVE] Metrics saved to {metrics_path}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Completed: {completed}/{total_evals}")
    print(f"Skipped: {skipped}/{total_evals}")
    
    if completed > 0:
        print(f"\nPerformance Metrics (mean Â± std across folds/seeds):")
        print(f"  Sharpe Ratio: {metrics_df['Sharpe'].mean():.3f} Â± {metrics_df['Sharpe'].std():.3f}")
        print(f"  Information Ratio: {metrics_df['Information_Ratio'].mean():.3f} Â± {metrics_df['Information_Ratio'].std():.3f}")
        print(f"  Max Drawdown: {metrics_df['MaxDD'].mean():.3f} Â± {metrics_df['MaxDD'].std():.3f}")
        print(f"  Turnover: {metrics_df['mean_turnover'].mean():.2%} Â± {metrics_df['mean_turnover'].std():.2%}")
        
        print(f"\n[SUCCESS] Results saved to: {results_dir}/")
        print("\nNext steps:")
        print(f"  1. Run baselines: python baselines_final.py --universe {args.universe}")
        print(f"  2. Generate report: python generate_summary_report.py --universe {args.universe}")


if __name__ == "__main__":
    main()
