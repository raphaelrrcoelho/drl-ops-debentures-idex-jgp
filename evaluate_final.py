# evaluate_final.py
"""
Evaluate MaskablePPO with discrete actions on fold-specific asset universes
Optimized for compatibility with top-K training approach
Includes sensitivity analysis for transaction costs and reward parameters
"""
from __future__ import annotations

import os
import json
import argparse
import itertools
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# SB3
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception as e:
    raise RuntimeError("sb3-contrib and stable-baselines3 are required.") from e

import warnings
warnings.filterwarnings("ignore", message=".*get_schedule_fn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*constant_fn.*deprecated.*")

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("PyTorch is required for evaluating. Please install torch.") from e

# Our env
from env_final import make_env_from_panel, EnvConfig

TRADING_DAYS = 252.0

# ------------------------------ IO helpers -------------------------------- #

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

# --------------------------- Panel / calendar ----------------------------- #

REQUIRED_COLS = [
    "return", "spread", "duration", "sector_id", "active",
    "risk_free", "index_return", "time_to_maturity", 
    "index_weight",  # Now required for top-K selection
]

# Enhanced features that should be preserved
ENHANCED_FEATURE_GROUPS = {
    "momentum": ["momentum_5d", "momentum_20d"],
    "reversal": ["reversal_5d", "reversal_20d"],
    "volatility": ["volatility_5d", "volatility_20d"],
    "spread_vol": ["spread_vol_5d", "spread_vol_20d"],
    "relative_value": [
        "spread_vs_sector_median", "spread_vs_sector_mean",
        "spread_percentile_sector", "spread_percentile_all"
    ],
    "duration_risk": [
        "duration_change", "duration_vol", "duration_spread_interaction"
    ],
    "microstructure": ["liquidity_score", "weight_momentum", "weight_volatility"],
    "carry": ["carry_spread_ratio", "carry_momentum", "carry_vol"],
    "spread_dynamics": [
        "spread_momentum_5d", "spread_momentum_20d",
        "spread_mean_reversion", "spread_acceleration"
    ],
    "risk_adjusted": [],
    "sector_curves": [
        "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
        "sector_fitted_spread", "spread_residual_ns"
    ]
}

SAFE_FILL_0 = ["return", "spread", "duration", "time_to_maturity", "ttm_rank"]
DATE_LEVEL = ["risk_free", "index_return"]

def _date_level_map(panel: pd.DataFrame, col: str) -> pd.Series:
    """First value per date; used to broadcast date-level series."""
    if col not in panel.columns:
        return pd.Series(dtype=float)
    return panel[[col]].groupby(level="date").first()[col].sort_index()

def union_ids_from_training(results_dir: str, fold: int) -> Optional[List[str]]:
    """Load the union IDs that were used during training for this fold"""
    path = os.path.join(results_dir, "training_union_ids.json")
    if not os.path.exists(path):
        return None
    data = read_json(path)
    for rec in data:
        if int(rec.get("fold", -1)) == int(fold):
            return [str(x) for x in rec.get("ids", [])]
    return None

def compute_union_ids_from_panel(panel: pd.DataFrame, fold_cfg: Dict[str, str],
                                 env_cfg: EnvConfig) -> List[str]:
    """
    Compute union IDs using the same top-K logic as training.
    This ensures consistency between training and evaluation.
    """
    train_start = pd.to_datetime(fold_cfg["train_start"])
    test_end = pd.to_datetime(fold_cfg["test_end"])
    
    # Get all dates in train+test period
    dates = panel.index.get_level_values("date").unique()
    period_dates = dates[(dates >= train_start) & (dates <= test_end)].sort_values()
    
    # Identify rebalance dates
    rebalance_interval = env_cfg.rebalance_interval
    rb_indices = np.arange(0, len(period_dates), max(1, rebalance_interval))
    rb_dates = period_dates[rb_indices]
    
    # Collect assets that are ever top-K
    ever_topk = set()
    max_assets = env_cfg.max_assets
    
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

def build_eval_panel_with_union(panel: pd.DataFrame, fold_cfg: Dict[str, str], 
                                union_ids: List[str], feature_config: Dict) -> pd.DataFrame:
    """
    Build evaluation panel for TEST dates × UNION IDS.
    Compatible with both full universe and top-K training approaches.
    """
    test_start = pd.to_datetime(fold_cfg["test_start"])
    test_end = pd.to_datetime(fold_cfg["test_end"])

    dates = panel.index.get_level_values("date").unique().sort_values()
    test_dates = dates[(dates >= test_start) & (dates <= test_end)]
    
    print(f"[EVAL] Building panel for {len(test_dates)} test dates × {len(union_ids)} assets")
    
    idx = pd.MultiIndex.from_product([test_dates, union_ids], names=["date", "debenture_id"])
    test_aug = panel.reindex(idx).sort_index()

    # Ensure required columns exist
    for c in REQUIRED_COLS:
        if c not in test_aug.columns:
            test_aug[c] = np.nan

    # Add enhanced features based on config
    all_enhanced_features = []
    feature_flags = {
        "use_momentum_features": feature_config.get('use_momentum_features', True),
        "use_volatility_features": feature_config.get('use_volatility_features', False),
        "use_relative_value_features": feature_config.get('use_relative_value_features', True),
        "use_duration_features": feature_config.get('use_duration_features', True),
        "use_microstructure_features": feature_config.get('use_microstructure_features', False),
        "use_carry_features": feature_config.get('use_carry_features', True),
        "use_spread_dynamics": feature_config.get('use_spread_dynamics', True),
        "use_risk_adjusted_features": feature_config.get('use_risk_adjusted_features', False),
        "use_sector_curves": feature_config.get('use_sector_curves', True),
        "use_zscore_features": feature_config.get('use_zscore_features', False),
        "use_rolling_zscores": feature_config.get('use_rolling_zscores', False),
    }

    # Collect features based on config flags
    for group_name, features in ENHANCED_FEATURE_GROUPS.items():
        flag_name = f"use_{group_name.replace('_risk', '').replace('_dynamics', '')}_features"
        if feature_flags.get(flag_name, True):
            all_enhanced_features.extend(features)

    # Add lagged versions
    lagged_features = []
    for feat in all_enhanced_features:
        lag_feat = f"{feat}_lag1"
        if lag_feat in panel.columns:
            lagged_features.append(lag_feat)
        elif feat in panel.columns:
            lagged_features.append(feat)
    
    # Copy features from original panel
    for feat in lagged_features:
        if feat not in test_aug.columns and feat in panel.columns:
            feat_series = panel[feat].reindex(test_aug.index)
            test_aug[feat] = feat_series

    # Broadcast date-level fields
    for name in DATE_LEVEL:
        m = _date_level_map(panel, name)
        if not m.empty:
            df = test_aug.reset_index()
            df[name] = df[name].astype(float).fillna(df["date"].map(m).astype(float))
            test_aug = df.set_index(["date", "debenture_id"]).sort_index()

    # Safe fills & types
    test_aug["active"] = test_aug["active"].fillna(0).astype(np.int8)
    for c in SAFE_FILL_0:
        if c in test_aug.columns:
            r = test_aug["return"]
            a = test_aug["active"]
            test_aug["return"] = np.where(a.astype(bool), r, 0.0)
    
    if "sector_id" in test_aug.columns:
        test_aug["sector_id"] = test_aug["sector_id"].fillna(-1).astype(np.int16)
    
    if "index_weight" in test_aug.columns:
        test_aug["index_weight"] = test_aug["index_weight"].fillna(0.0).astype(np.float32)

    for name in ["risk_free", "index_return"]:
        test_aug[name] = test_aug[name].astype(np.float32).fillna(0.0)

    # Fill enhanced features with 0 if missing
    for feat in lagged_features:
        if feat in test_aug.columns:
            test_aug[feat] = test_aug[feat].fillna(0.0).astype(np.float32)

    print(f"[EVAL] Panel features: {len([c for c in test_aug.columns if c.endswith('_lag1')])} lagged")
    print(f"[EVAL] Feature flags enabled: {sum(feature_flags.values())}/{len(feature_flags)}")
    
    # Report sparsity
    total_cells = len(test_aug) * len(test_aug.columns)
    missing_cells = test_aug.isnull().sum().sum()
    print(f"[EVAL] Panel sparsity: {missing_cells/total_cells:.1%}")
    
    return test_aug

# ------------------------------ Metrics ---------------------------------- #

def wealth_from_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()

def max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())

def compute_metrics(returns: pd.Series, rf: pd.Series, bench: pd.Series,
                    periods: int = int(TRADING_DAYS)) -> Dict[str, float]:
    """
    Calculate performance metrics consistent with baselines.
    NOTE: returns are already TOTAL returns (including RF)
    """
    r = returns.fillna(0.0).astype(float)
    rf_a = rf.reindex(r.index).fillna(0.0).astype(float)
    bench_a = bench.reindex(r.index).fillna(0.0).astype(float)

    r_excess = (r - rf_a).fillna(0.0)
    r_rel = (r - bench_a).fillna(0.0)

    mu = r.mean()
    sd = r.std(ddof=1)
    mu_ex = r_excess.mean()
    sd_ex = r_excess.std(ddof=1)
    downside = r_excess[r_excess < 0].std(ddof=1)

    sharpe_d = (mu_ex / sd_ex) if sd_ex > 0 else np.nan
    sortino_d = (mu_ex / downside) if downside and downside > 0 else np.nan

    wealth = wealth_from_returns(r)
    mdd = max_drawdown(wealth)
    n = max(1, len(r))
    cagr = float(wealth.iloc[-1] ** (periods / n) - 1.0)
    vol_ann = float(sd * np.sqrt(periods))
    sharpe_ann = sharpe_d * np.sqrt(periods) if sharpe_d == sharpe_d else np.nan
    sortino_ann = sortino_d * np.sqrt(periods) if sortino_d == sortino_d else np.nan
    calmar = (cagr / abs(mdd)) if mdd < 0 else np.nan

    # Additional metrics from baselines
    cov = np.cov(r.dropna(), bench_a.reindex(r.index).dropna())[0, 1] if len(r.dropna()) and len(bench_a.dropna()) else np.nan
    varb = bench_a.var(ddof=1)
    beta = (cov / varb) if varb and varb > 0 else np.nan
    alpha = (mu - bench_a.mean() * (beta if beta == beta else 0.0))

    ir_den = r_rel.std(ddof=1)
    information_ratio = (r_rel.mean() / ir_den) if ir_den and ir_den > 0 else np.nan

    pos = bench_a > 0
    neg = bench_a < 0
    up_capture = (r[pos].mean() / bench_a[pos].mean()) if pos.any() and bench_a[pos].mean() != 0 else np.nan
    down_capture = (r[neg].mean() / bench_a[neg].mean()) if neg.any() and bench_a[neg].mean() != 0 else np.nan

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

# ------------------------- Weights serialization ------------------------- #

def append_weights_long(results_dir: str,
                        weights_df: pd.DataFrame,
                        strategy: str,
                        fold: int,
                        seed: int):
    """Save portfolio weights in long format"""
    out = weights_df.copy()
    out.index = pd.to_datetime(out.index)
    df_long = out.reset_index().melt(id_vars=["index"], var_name="debenture_id", value_name="weight")
    df_long = df_long.rename(columns={"index": "date"})
    df_long["strategy"] = strategy
    df_long["fold"] = int(fold)
    df_long["seed"] = int(seed)
    path = os.path.join(results_dir, "weights_long.csv")
    append_long_csv(path, df_long, sort_by=["date", "strategy"])

def write_topk_holdings_snapshot(results_dir: str,
                                 weights_df: pd.DataFrame,
                                 strategy: str,
                                 fold: int,
                                 seed: int,
                                 k: int = 5):
    """Save daily top-k holdings for visualization"""
    w = weights_df.copy()
    if w.empty:
        return
    w.index = pd.to_datetime(w.index)
    snaps = []
    for dt, row in w.iterrows():
        top = row.sort_values(ascending=False).head(k)
        for rank, (deb_id, wt) in enumerate(top.items(), start=1):
            snaps.append({
                "date": dt, "strategy": strategy, "fold": int(fold), "seed": int(seed),
                "debenture_id": str(deb_id), "weight": float(wt), "rank": int(rank)
            })
    if snaps:
        df_top = pd.DataFrame(snaps)
        path = os.path.join(results_dir, "topk_holdings.csv")
        append_long_csv(path, df_top, sort_by=["date", "strategy", "rank"])

# ------------------------- Mask function for discrete -------------------- #

def mask_fn(env):
    """Action mask function for MaskablePPO"""
    if hasattr(env, 'get_action_masks'):
        masks = env.get_action_masks()
        if isinstance(masks, list):
            return np.concatenate([m.flatten() for m in masks])
        return masks
    else:
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'nvec'):
            total_actions = sum(env.action_space.nvec)
            return np.ones(total_actions, dtype=bool)
        return np.ones(100, dtype=bool)

# ------------------------- Sensitivity Analysis --------------------------- #

def run_sensitivity_analysis(model_path: str, test_panel: pd.DataFrame, 
                            base_env_cfg: EnvConfig, param_variations: Dict[str, List[float]],
                            results_dir: str, strategy_label: str, fold: int, seed: int):
    """
    Run sensitivity analysis by varying parameters and evaluating performance.
    Compatible with top-K training approach.
    """
    print(f"[SENSITIVITY] Starting analysis for {strategy_label}")
    
    import dataclasses
    
    # Generate all combinations
    param_names = list(param_variations.keys())
    param_values = list(param_variations.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    
    for i, combo in enumerate(combinations):
        # Create modified config
        modified_cfg = asdict(base_env_cfg)
        for j, param_name in enumerate(param_names):
            modified_cfg[param_name] = combo[j]
        
        # Create environment
        def make_env():
            env = make_env_from_panel(test_panel, **modified_cfg)
            env = ActionMasker(env, mask_fn)
            return env

        venv = DummyVecEnv([make_env])
        
        # Load VecNormalize if available
        vecnorm_path = os.path.join(results_dir, "..", "models", strategy_label.split("_")[0].lower(), 
                                    "ppo", f"vecnorm_fold_{fold}_seed_{seed}.pkl")
        if os.path.exists(vecnorm_path):
            import pickle
            with open(vecnorm_path, 'rb') as f:
                vecnorm_stats = pickle.load(f)
            
            venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
            venv.obs_rms = vecnorm_stats['obs_rms']
            venv.ret_rms = vecnorm_stats['ret_rms']
            venv.gamma = vecnorm_stats['gamma']
            venv.training = False

        device = "cpu"
        model = MaskablePPO.load(model_path, device=device)
        
        # Run evaluation
        obs = venv.reset()
        test_dates = test_panel.index.get_level_values("date").unique()
        returns = []
        
        for t in range(len(test_dates)):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            returns.append(info.get("portfolio_return", 0.0))
            if np.any(dones):
                break
        
        # Calculate metrics
        returns_series = pd.Series(returns)
        rf_series = pd.Series(0.0, index=range(len(returns)))
        idx_series = pd.Series(0.0, index=range(len(returns)))
        metrics = compute_metrics(returns_series, rf_series, idx_series)
        
        # Store results
        result_row = {
            "strategy": strategy_label,
            "fold": fold,
            "seed": seed,
            "combo_id": i,
            **{f"param_{param_names[j]}": combo[j] for j in range(len(param_names))},
            **metrics
        }
        results.append(result_row)
        
        if (i + 1) % max(1, len(combinations) // 5) == 1:
            print(f"  Combo {i+1}/{len(combinations)}: {dict(zip(param_names, combo))}")
        
        venv.close()
    
    # Save results
    sensitivity_df = pd.DataFrame(results)
    sensitivity_path = os.path.join(results_dir, f"sensitivity_{strategy_label}_f{fold}_s{seed}.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    
    print(f"[SENSITIVITY] Saved results to {sensitivity_path}")
    
    return sensitivity_df

# ------------------------------- Main eval -------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Evaluate PPO with top-K constraint and sensitivity analysis")
    ap.add_argument("--universe", required=True, choices=["cdi", "infra"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_base", default=".")
    ap.add_argument("--seeds", default="", help="Comma-separated seeds; default from config")
    ap.add_argument("--folds", default="", help="Comma-separated fold indices; default all")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--save_weights", action="store_true", default=True)
    ap.add_argument("--save_topk", action="store_true", default=True)
    ap.add_argument("--topk_k", type=int, default=5)
    ap.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    ap.add_argument("--sensitivity_params", type=str, 
                   default="lambda_turnover,lambda_hhi,lambda_drawdown,transaction_cost_bps",
                   help="Parameters to vary in sensitivity analysis")
    ap.add_argument("--sensitivity_values", type=str, default="0.5,1.0,2.0",
                   help="Multiplicative factors for sensitivity")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    # Load config file
    config = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[CONFIG] Loaded configuration from {args.config}")
    
    # Directories
    res_dir = os.path.join(args.out_base, "results", args.universe)
    models_dir = os.path.join(args.out_base, "models", args.universe, "ppo")
    ensure_dir(res_dir)

    # Load processed panel
    panel_path = os.path.join(args.data_dir, f"{args.universe}_processed.pkl")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Missing {panel_path} (run data_final.py first).")
    panel: pd.DataFrame = pd.read_pickle(panel_path)
    
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Panel must be MultiIndex (date, debenture_id).")
    
    print(f"[DATA] Loaded panel with {len(panel):,} observations")
    print(f"[DATA] Assets: {panel.index.get_level_values('debenture_id').nunique()}")
    print(f"[DATA] Dates: {panel.index.get_level_values('date').nunique()}")

    # Load training configuration
    folds_path = os.path.join(res_dir, "training_folds.json")
    cfg_path = os.path.join(res_dir, "training_config.json")
    
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"Missing {folds_path}.")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path}.")

    folds = read_json(folds_path)
    train_cfg = read_json(cfg_path)
    
    # Environment config - merge training config with config file
    env_cfg_dict = train_cfg.get("env_config", {})
    
    if config:
        # Override with config file values
        for k in ['rebalance_interval', 'max_weight', 'weight_blocks', 'allow_cash', 
                  'cash_rate_as_rf', 'on_inactive', 'transaction_cost_bps', 
                  'delist_extra_bps', 'normalize_features', 'obs_clip', 
                  'include_prev_weights', 'include_active_flag', 'global_stats',
                  'weight_alpha', 'lambda_turnover', 'lambda_hhi', 'lambda_drawdown',
                  'lambda_tail', 'tail_window', 'tail_q', 'dd_mode', 'max_steps',
                  'random_reset_frac', 'max_assets']:
            if k in config:
                env_cfg_dict[k] = config[k]
        
        # Feature flags
        for k in ['use_momentum_features', 'use_volatility_features', 
                  'use_relative_value_features', 'use_duration_features',
                  'use_microstructure_features', 'use_carry_features',
                  'use_spread_dynamics', 'use_risk_adjusted_features',
                  'use_sector_curves', 'use_zscore_features', 'use_rolling_zscores']:
            if k in config:
                env_cfg_dict[k] = config[k]
    
    env_cfg = EnvConfig(**env_cfg_dict)
    
    print(f"[CONFIG] Max assets: {env_cfg.max_assets}")
    print(f"[CONFIG] Rebalance interval: {env_cfg.rebalance_interval}")
    print(f"[CONFIG] Transaction cost: {env_cfg.transaction_cost_bps} bps")

    # Seeds
    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        config_seeds = config.get('seeds', '')
        if config_seeds:
            seeds = [int(s) for s in str(config_seeds).split(",") if s.strip()]
        else:
            seeds = [int(s) for s in str(train_cfg.get("seeds", "0")).split(",") if s.strip()]

    # Fold subset
    if args.folds.strip():
        sel_folds = set(int(x) for x in args.folds.split(",") if x.strip())
        folds = [f for f in folds if int(f.get("fold", -1)) in sel_folds]

    # Risk-free series
    rf_path = os.path.join(res_dir, "risk_free_used.csv")
    if not os.path.exists(rf_path):
        print(f"[WARN] Missing {rf_path}, using zero risk-free rate")
        rf_series = pd.Series(0.0)
    else:
        rf_df = pd.read_csv(rf_path, parse_dates=["date"]).set_index("date")
        rf_series = rf_df["risk_free"].astype(float)

    # Sensitivity setup
    if args.sensitivity:
        sensitivity_params = [p.strip() for p in args.sensitivity_params.split(",")]
        sensitivity_factors = [float(v.strip()) for v in args.sensitivity_values.split(",")]
        
        # Override with config if available
        if config:
            if 'sensitivity_params' in config:
                sensitivity_params = [p.strip() for p in config['sensitivity_params'].split(",")]
            if 'sensitivity_values' in config:
                sensitivity_factors = [float(v.strip()) for v in config['sensitivity_values'].split(",")]
        
        param_variations = {}
        for param in sensitivity_params:
            if hasattr(env_cfg, param):
                base_value = getattr(env_cfg, param)
                param_variations[param] = [base_value * factor for factor in sensitivity_factors]
                print(f"[SENSITIVITY] {param}: {param_variations[param]}")

    # Evaluation settings
    eval_config = config.get('evaluation', {}) if config else {}
    deterministic = eval_config.get('deterministic', True)
    save_weights = eval_config.get('save_weights', True)
    save_topk = eval_config.get('save_topk', True)
    topk_k = eval_config.get('topk_k', 5)

    print(f"\n[EVALUATION START]")
    print(f"  Folds: {len(folds)}")
    print(f"  Seeds: {seeds}")
    print(f"  Deterministic: {deterministic}")
    
    # Iterate folds
    for fcfg in folds:
        fold = int(fcfg["fold"])
        print(f"\n[FOLD {fold}]")
        print(f"  Test period: {fcfg['test_start']} to {fcfg['test_end']}")

        # Get union IDs from training or compute with top-K logic
        union = union_ids_from_training(res_dir, fold)
        if union is None:
            print(f"  Computing union IDs with top-{env_cfg.max_assets} logic...")
            union = compute_union_ids_from_panel(panel, fcfg, env_cfg)
        else:
            print(f"  Using {len(union)} union IDs from training")

        # Build test panel
        test_aug = build_eval_panel_with_union(panel, fcfg, union, env_cfg_dict)

        # Date-level series for test
        test_dates = test_aug.index.get_level_values("date").unique().sort_values()
        idx_map = _date_level_map(test_aug, "index_return")
        rf_test = rf_series.reindex(test_dates).fillna(0.0)
        bench_test = idx_map.reindex(test_dates).fillna(0.0)

        # Iterate seeds
        for seed in seeds:
            label = f"PPO_f{fold}_s{seed}"
            model_path = os.path.join(models_dir, f"model_fold_{fold}_seed_{seed}.zip")
            
            if not os.path.exists(model_path):
                print(f"  [WARN] Missing model {model_path}; skipping.")
                continue
            
            print(f"  Evaluating seed {seed}...")

            # Sensitivity analysis if requested
            if args.sensitivity and param_variations:
                sensitivity_results = run_sensitivity_analysis(
                    model_path, test_aug, env_cfg, param_variations,
                    res_dir, label, fold, seed
                )

            # Standard evaluation
            device = "cpu"
            model = MaskablePPO.load(model_path, device=device)

            # Create environment
            def make_env():
                env = make_env_from_panel(test_aug, **asdict(env_cfg))
                env = ActionMasker(env, mask_fn)
                return env

            venv = DummyVecEnv([make_env])

            # Load VecNormalize if available
            vecnorm_path = os.path.join(models_dir, f"vecnorm_fold_{fold}_seed_{seed}.pkl")
            if os.path.exists(vecnorm_path):
                import pickle
                with open(vecnorm_path, 'rb') as f:
                    vecnorm_stats = pickle.load(f)
                
                # Create VecNormalize with the loaded statistics
                venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
                venv.obs_rms = vecnorm_stats['obs_rms']
                venv.ret_rms = vecnorm_stats['ret_rms']
                venv.gamma = vecnorm_stats['gamma']
                venv.training = False
                venv.norm_reward = False

            # Run episode
            obs = venv.reset()
            done = False

            # Collectors
            rows_ret = []
            rows_diag = []
            weights_by_date = []
            asset_ids = list(union)

            t = 0
            while not done and t < len(test_dates):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = venv.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                done = dones[0] if isinstance(dones, (list, tuple, np.ndarray)) else dones

                # Date
                dt = test_dates[min(len(test_dates)-1, t)]
                t += 1

                # Portfolio return (net of costs)
                r_p = float(info.get("portfolio_return", 0.0))
                rows_ret.append({"date": dt, "strategy": label, "return": r_p})

                # Diagnostics
                for m, v in [
                    ("turnover", info.get("turnover", np.nan)),
                    ("hhi", info.get("hhi", np.nan)),
                    ("drawdown", info.get("drawdown", np.nan)),
                    ("alpha", info.get("alpha", np.nan)),
                    ("excess", info.get("excess", np.nan)),
                    ("rf", info.get("rf", np.nan)),
                    ("idx", info.get("index_return", np.nan)),
                    ("wealth", info.get("wealth", np.nan)),
                ]:
                    rows_diag.append({
                        "date": dt, "strategy": label, "metric": m, 
                        "value": float(v) if v is not None else np.nan
                    })

                # Weights snapshot
                w = info.get("weights", None)
                if w is not None and save_weights:
                    w = np.asarray(w, dtype=float).ravel()
                    # Handle cash position if present
                    if env_cfg.allow_cash and len(w) == len(asset_ids) + 1:
                        # Remove cash weight (last position)
                        w = w[:-1]
                    if w.size == len(asset_ids):
                        weights_by_date.append((dt, w))

            venv.close()

            # Save returns & diagnostics
            df_ret = pd.DataFrame(rows_ret)
            df_diag = pd.DataFrame(rows_diag)
            append_long_csv(os.path.join(res_dir, "all_returns.csv"), df_ret, sort_by=["date", "strategy"])
            append_long_csv(os.path.join(res_dir, "all_diagnostics.csv"), df_diag, sort_by=["date", "strategy", "metric"])

            # Save weights if collected
            if weights_by_date and save_weights:
                dates_w = [d for d, _ in weights_by_date]
                mat = np.vstack([w for _, w in weights_by_date])
                df_wide = pd.DataFrame(mat, index=pd.to_datetime(dates_w), columns=[str(x) for x in asset_ids]).sort_index()
            else:
                df_wide = pd.DataFrame()

            # Save pickle with full results
            out_pkl = {
                "returns": df_ret.set_index("date")["return"].astype(float),
                "diagnostics": df_diag,
                "weights": df_wide,
                "asset_ids": [str(x) for x in asset_ids],
                "feature_config": env_cfg_dict,
                "union_size": len(union),
                "max_assets": env_cfg.max_assets,
            }
            pkl_path = os.path.join(res_dir, f"fold_{fold}_seed_{seed}_results.pkl")
            pd.to_pickle(out_pkl, pkl_path)

            # Save weights in long format
            if not df_wide.empty and save_weights:
                append_weights_long(res_dir, df_wide, label, fold, seed)
                if save_topk:
                    write_topk_holdings_snapshot(res_dir, df_wide, label, fold, seed, k=topk_k)

            # Calculate and save metrics
            r_series = df_ret.set_index("date")["return"].astype(float)
            m = compute_metrics(r_series, rf_test, bench_test)
            row = {"strategy": label, "fold": fold, "seed": seed}
            row.update(m)

            fold_cols = ["strategy", "fold", "seed", "CAGR", "Vol_ann", "Sharpe", "Sortino", 
             "MaxDD", "Calmar", "Alpha_daily", "Beta", "Information_Ratio", 
             "Up_capture", "Down_capture", "Hit_rate", "Skew", "Kurtosis"]
            append_row_csv(os.path.join(res_dir, "fold_metrics.csv"), row, cols_order=fold_cols)
            append_row_csv(os.path.join(res_dir, "all_metrics.csv"), row, cols_order=fold_cols)

            print(f"    Sharpe={m['Sharpe']:.3f}, CAGR={m['CAGR']:.3%}, MaxDD={m['MaxDD']:.3%}")

    print("\n[EVALUATION COMPLETE]")
    print(f"Results saved to: {res_dir}")

if __name__ == "__main__":
    main()