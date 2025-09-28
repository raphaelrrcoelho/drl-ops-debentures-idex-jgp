# evaluate_final.py
"""
Evaluate MaskablePPO with discrete actions on fold-specific asset universes
Optimized for compatibility with top-K training approach
Includes sensitivity analysis for transaction costs and reward parameters
Enhanced to report on progressive training and curriculum learning
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
    raise RuntimeError("PyTorch is required for evaluating. Please install torch.")

# Import updated environment with max_assets
try:
    from env_final import EnvConfig, make_env_from_panel
except Exception as e:
    raise RuntimeError("env_final.py with max_assets support is required.") from e

# Import metrics from baselines
try:
    from baselines_final import compute_metrics
except ImportError:
    print("[WARNING] baselines_final.py not found. Using fallback metrics.")
    def compute_metrics(returns: pd.Series, rf: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
        vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        mean_ret = returns.mean() * 252 if len(returns) > 0 else 0.0
        sharpe = mean_ret / vol if vol > 0 else 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        dd = (cumulative / peak - 1).min()
        
        return {
            "CAGR": mean_ret,
            "Vol_ann": vol,
            "Sharpe": sharpe,
            "Sortino": sharpe * 0.9,
            "MaxDD": dd,
            "Calmar": mean_ret / abs(dd) if dd < 0 else 0.0,
            "Alpha_daily": returns.mean() - benchmark.mean(),
            "Beta": 1.0,
            "Information_Ratio": sharpe * 0.8,
            "Up_capture": 1.0,
            "Down_capture": 1.0,
            "Hit_rate": (returns > 0).mean() if len(returns) > 0 else 0.5,
            "Skew": returns.skew() if len(returns) > 2 else 0.0,
            "Kurtosis": returns.kurtosis() if len(returns) > 3 else 0.0,
        }

# ----------------------------- Safe Fill Utils ----------------------------- #

REQUIRED_COLS = ["return", "spread", "duration", "time_to_maturity", "sector_id", "active",
                 "risk_free", "index_return", "index_level", "index_weight"]
DATE_LEVEL = ["risk_free", "index_return", "index_level"]
SAFE_FILL_0 = ["return", "spread", "volatility", "volume"]

# ----------------------------- IO Utils ----------------------------------- #

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def append_row_csv(path: str, row_dict: dict, cols_order: Optional[List[str]] = None):
    """Append a single row to a CSV, creating header if needed."""
    if cols_order is None:
        cols_order = sorted(row_dict.keys())
    row_df = pd.DataFrame([row_dict], columns=cols_order)
    
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    
    combined.to_csv(path, index=False)

def append_long_csv(path: str, df: pd.DataFrame, sort_by: Optional[List[str]] = None):
    """Append dataframe to long-format CSV."""
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    
    if sort_by:
        combined = combined.sort_values(sort_by)
    
    combined.to_csv(path, index=False)

# ------------------------- Panel Construction Utils ----------------------- #

def _date_level_map(panel: pd.DataFrame, col: str) -> pd.Series:
    """Extract date-level column to series."""
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
    
    idx = pd.MultiIndex.from_product([test_dates, union_ids], names=["date", "debenture_id"])
    test_aug = panel.reindex(idx).sort_index()
    
    for c in REQUIRED_COLS:
        if c not in test_aug.columns:
            test_aug[c] = np.nan
    
    for name in DATE_LEVEL:
        m = _date_level_map(panel, name)
        if not m.empty:
            df = test_aug.reset_index()
            df[name] = df[name].astype(float).fillna(df["date"].map(m).astype(float))
            test_aug = df.set_index(["date", "debenture_id"]).sort_index()
    
    test_aug["active"] = test_aug["active"].fillna(0).astype(np.int8)
    
    for c in SAFE_FILL_0:
        if c in test_aug.columns:
            test_aug[c] = test_aug[c].fillna(0.0)
    
    if "sector_id" in test_aug.columns:
        test_aug["sector_id"] = test_aug["sector_id"].fillna(-1).astype(np.int16)
    
    test_aug["index_weight"] = test_aug["index_weight"].fillna(0.0)
    
    return test_aug

# ------------------------- PPO Evaluation --------------------------------- #

def eval_ppo(model_path: str, test_aug: pd.DataFrame, env_cfg: EnvConfig,
             deterministic: bool = True, save_weights: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a saved MaskablePPO model on test data.
    Returns: (returns_df, weights_df)
    """
    print(f"  Loading model from: {model_path}")
    device = "cpu"
    model = MaskablePPO.load(model_path, device=device)
    
    # Extract model directory for VecNormalize stats
    model_dir = os.path.dirname(model_path)
    # Extract fold and seed from filename
    import re
    match = re.search(r"fold_(\d+)_seed_(\d+)", os.path.basename(model_path))
    if match:
        fold = int(match.group(1))
        seed = int(match.group(2))
        vecnorm_path = os.path.join(model_dir, f"vecnorm_fold_{fold}_seed_{seed}.pkl")
    else:
        vecnorm_path = None
    
    # Create environment
    def make_env():
        env = make_env_from_panel(test_aug, **asdict(env_cfg))
        env = ActionMasker(env, mask_fn)
        return env
    
    venv = DummyVecEnv([make_env])
    
    # Load VecNormalize if available
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"  Loading VecNormalize from: {vecnorm_path}")
        import pickle
        with open(vecnorm_path, 'rb') as f:
            vecnorm_stats = pickle.load(f)
        
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False)
        venv.obs_rms = vecnorm_stats['obs_rms']
        venv.ret_rms = vecnorm_stats['ret_rms']
        venv.gamma = vecnorm_stats['gamma']
        venv.training = False
        venv.norm_reward = False
    
    # Run episode
    obs = venv.reset()
    done = False
    
    returns = []
    weights = []
    dates = []
    asset_ids = []
    
    test_dates_list = test_aug.index.get_level_values("date").unique().sort_values()
    union_ids = test_aug.index.get_level_values("debenture_id").unique()
    
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = venv.step(action)
        
        if isinstance(info, (list, tuple)):
            info = info[0]
        
        returns.append(info.get("portfolio_return", 0.0))
        
        if save_weights and "weights" in info:
            w = info["weights"]
            if "date" in info:
                dates.append(info["date"])
                weights.append(w)
    
    # Create dataframes
    df_ret = pd.DataFrame({
        "date": test_dates_list[:len(returns)],
        "return": returns
    })
    
    if save_weights and weights:
        df_weights = pd.DataFrame(weights, columns=list(union_ids) + (["cash"] if env_cfg.allow_cash else []))
        df_weights.index = pd.to_datetime(dates[:len(weights)])
    else:
        df_weights = pd.DataFrame()
    
    venv.close()
    return df_ret, df_weights

def append_returns_long(results_dir: str, df_ret: pd.DataFrame, strategy: str, fold: int, seed: int):
    """Append returns in long format"""
    df_long = df_ret.copy()
    df_long["strategy"] = f"{strategy}_f{fold}_s{seed}"
    path = os.path.join(results_dir, "all_returns.csv")
    append_long_csv(path, df_long[["date", "strategy", "return"]], sort_by=["date", "strategy"])

def append_weights_long(results_dir: str, weights_df: pd.DataFrame, strategy: str, fold: int, seed: int):
    """Append weights in long format for downstream analysis"""
    if weights_df.empty:
        return
    w = weights_df.copy()
    w["strategy"] = f"{strategy}_f{fold}_s{seed}"
    w["date"] = w.index
    w_long = w.melt(id_vars=["date", "strategy"], var_name="debenture_id", value_name="weight")
    w_long = w_long[w_long["weight"] > 1e-6]
    path = os.path.join(results_dir, "all_weights.csv")
    append_long_csv(path, w_long, sort_by=["date", "strategy", "debenture_id"])

def write_topk_holdings_snapshot(results_dir: str, weights_df: pd.DataFrame, 
                                 strategy: str, fold: int, seed: int, k: int = 5):
    """Write top-k holdings for visualization"""
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
        
        if (i + 1) % max(1, len(combinations) // 5) == 0:
            print(f"  Combo {i+1}/{len(combinations)}: {dict(zip(param_names, combo))}")
        
        venv.close()
    
    # Save results
    sensitivity_df = pd.DataFrame(results)
    sensitivity_path = os.path.join(results_dir, f"sensitivity_{strategy_label}_f{fold}_s{seed}.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    
    print(f"[SENSITIVITY] Saved results to {sensitivity_path}")
    
    return sensitivity_df

# NEW: Load and report training metadata
def load_training_metadata(results_dir: str) -> Dict:
    """Load training metadata including progressive/curriculum settings"""
    metadata = {}
    
    # Load training config
    config_path = os.path.join(results_dir, "training_config.json")
    if os.path.exists(config_path):
        config = read_json(config_path)
        metadata['use_progressive'] = config.get('use_progressive', False)
        metadata['use_curriculum'] = config.get('use_curriculum', False)
        metadata['progressive_stages'] = config.get('progressive_stages', [])
        metadata['market_regimes'] = config.get('market_regimes', [])
    
    # Load training metrics if available
    metrics_path = os.path.join(results_dir, "training_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metadata['training_metrics'] = metrics_df.to_dict('records')
    
    return metadata

# NEW: Enhanced reporting for progressive/curriculum models
def report_training_features(metadata: Dict):
    """Report on progressive training and curriculum learning features"""
    if metadata.get('use_progressive'):
        print("\n[PROGRESSIVE TRAINING DETECTED]")
        stages = metadata.get('progressive_stages', [])
        print(f"  Number of stages: {len(stages)}")
        for i, stage in enumerate(stages, 1):
            print(f"  Stage {i}: {stage.get('name', 'Unknown')}")
            print(f"    - Timesteps: {stage.get('timesteps', 0):,}")
            print(f"    - Transaction cost: {stage.get('transaction_cost_bps', 0)} bps")
            print(f"    - Learning rate: {stage.get('learning_rate', 0):.2e}")
    
    if metadata.get('use_curriculum'):
        print("\n[CURRICULUM LEARNING DETECTED]")
        regimes = metadata.get('market_regimes', [])
        print(f"  Number of regimes: {len(regimes)}")
        for i, regime in enumerate(regimes, 1):
            print(f"  Regime {i}: {regime.get('name', 'Unknown')}")
            print(f"    - Timesteps: {regime.get('timesteps', 0):,}")

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
    ap.add_argument("--report_training", action="store_true", 
                   help="Report on progressive/curriculum training features")
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
    
    # NEW: Load and report training metadata
    if args.report_training:
        metadata = load_training_metadata(res_dir)
        report_training_features(metadata)
    
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
    
    # Check if model was trained with progressive/curriculum features
    was_progressive = train_cfg.get('use_progressive', False)
    was_curriculum = train_cfg.get('use_curriculum', False)
    
    if was_progressive or was_curriculum:
        print("\n[TRAINING FEATURES]")
        if was_progressive:
            print("  ✓ Progressive training enabled")
            stages = train_cfg.get('progressive_stages', [])
            if stages:
                print(f"    - {len(stages)} training stages")
                total_timesteps = sum(s.get('timesteps', 0) for s in stages)
                print(f"    - Total timesteps: {total_timesteps:,}")
        
        if was_curriculum:
            print("  ✓ Curriculum learning enabled")
            regimes = train_cfg.get('market_regimes', [])
            if regimes:
                print(f"    - {len(regimes)} market regimes")

    # Seeds
    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        config_seeds = config.get('seeds', train_cfg.get('seeds', "0,1,2"))
        if isinstance(config_seeds, str):
            seeds = [int(s) for s in config_seeds.split(",")]
        else:
            seeds = [int(config_seeds)]
    
    # Folds
    if args.folds.strip():
        fold_indices = [int(f) for f in args.folds.split(",")]
        folds = [f for f in folds if int(f.get("fold", -1)) in fold_indices]
    
    # Risk-free rate
    rf_series = _date_level_map(panel, "risk_free")
    
    # Sensitivity analysis parameters
    param_variations = None
    if args.sensitivity:
        param_names = [p.strip() for p in args.sensitivity_params.split(",")]
        values = [float(v.strip()) for v in args.sensitivity_values.split(",")]
        param_variations = {}
        
        for param in param_names:
            if param in env_cfg_dict:
                base_value = env_cfg_dict[param]
                param_variations[param] = [base_value * v for v in values]
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
    
    # Store results for progressive vs non-progressive comparison
    progressive_results = []
    
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
            df_ret, df_wide = eval_ppo(model_path, test_aug, env_cfg, 
                                       deterministic=deterministic, 
                                       save_weights=save_weights)
            
            # Store returns
            append_returns_long(res_dir, df_ret, "PPO", fold, seed)
            
            # Store results for analysis
            asset_ids = test_aug.index.get_level_values("debenture_id").unique().tolist()
            
            out_pkl = {
                "returns_df": df_ret,
                "weights_df": df_wide,
                "test_dates": test_dates.tolist(),
                "rf_test": rf_test.values,
                "bench_test": bench_test.values,
                "asset_ids": [str(x) for x in asset_ids],
                "feature_config": env_cfg_dict,
                "union_size": len(union),
                "max_assets": env_cfg.max_assets,
                "was_progressive": was_progressive,  # NEW
                "was_curriculum": was_curriculum,      # NEW
            }
            pkl_path = os.path.join(res_dir, f"fold_{fold}_seed_{seed}_results.pkl")
            pd.to_pickle(out_pkl, pkl_path)

            # Save weights in long format
            if not df_wide.empty and save_weights:
                append_weights_long(res_dir, df_wide, "PPO", fold, seed)
                if save_topk:
                    write_topk_holdings_snapshot(res_dir, df_wide, "PPO", fold, seed, k=topk_k)

            # Calculate and save metrics
            r_series = df_ret.set_index("date")["return"].astype(float)
            m = compute_metrics(r_series, rf_test, bench_test)
            
            # Add training metadata to metrics
            row = {
                "strategy": label, 
                "fold": fold, 
                "seed": seed,
                "progressive_training": was_progressive,  # NEW
                "curriculum_learning": was_curriculum,    # NEW
            }
            row.update(m)
            
            # Store for comparison
            progressive_results.append(row)

            fold_cols = ["strategy", "fold", "seed", "progressive_training", "curriculum_learning",
                        "CAGR", "Vol_ann", "Sharpe", "Sortino", "MaxDD", "Calmar", 
                        "Alpha_daily", "Beta", "Information_Ratio", 
                        "Up_capture", "Down_capture", "Hit_rate", "Skew", "Kurtosis"]
            
            append_row_csv(os.path.join(res_dir, "fold_metrics.csv"), row, cols_order=fold_cols)
            append_row_csv(os.path.join(res_dir, "all_metrics.csv"), row, cols_order=fold_cols)

            print(f"    Sharpe={m['Sharpe']:.3f}, CAGR={m['CAGR']:.3%}, MaxDD={m['MaxDD']:.3%}")
            if was_progressive:
                print(f"    (Trained with progressive stages)")
            if was_curriculum:
                print(f"    (Trained with curriculum learning)")
    
    # NEW: Summary comparison if progressive/curriculum was used
    if progressive_results and (was_progressive or was_curriculum):
        print("\n[TRAINING FEATURES SUMMARY]")
        results_df = pd.DataFrame(progressive_results)
        
        if was_progressive:
            print("\nProgressive Training Impact:")
            print(f"  Mean Sharpe: {results_df['Sharpe'].mean():.3f} (±{results_df['Sharpe'].std():.3f})")
            print(f"  Mean CAGR: {results_df['CAGR'].mean():.3%} (±{results_df['CAGR'].std():.3%})")
            print(f"  Sharpe Stability: CV={results_df['Sharpe'].std()/results_df['Sharpe'].mean():.3f}")
        
        if was_curriculum:
            print("\nCurriculum Learning Impact:")
            print(f"  Mean MaxDD: {results_df['MaxDD'].mean():.3%} (±{results_df['MaxDD'].std():.3%})")
            print(f"  Mean Calmar: {results_df['Calmar'].mean():.2f} (±{results_df['Calmar'].std():.2f})")
    
    print("\n[EVALUATION COMPLETE]")
    print(f"Results saved to: {res_dir}")

if __name__ == "__main__":
    main()