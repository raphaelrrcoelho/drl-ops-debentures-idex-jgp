# evaluate_final.py
"""
Evaluate MaskablePPO with discrete actions on fold-specific asset universes
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
    "risk_free", "index_return", "time_to_maturity", "index_level",
]

SAFE_FILL_0 = ["return", "spread", "duration", "time_to_maturity", "ttm_rank"]
DATE_LEVEL = ["risk_free", "index_return", "index_level"]

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

def compute_union_ids_from_panel(panel: pd.DataFrame, fold_cfg: Dict[str, str]) -> List[str]:
    """Fallback: compute union IDs from panel if file is missing"""
    train_start = pd.to_datetime(fold_cfg["train_start"])
    test_end = pd.to_datetime(fold_cfg["test_end"])
    sl = panel.loc[
        (panel.index.get_level_values("date") >= train_start) &
        (panel.index.get_level_values("date") <= test_end)
    ]
    return sl.index.get_level_values("debenture_id").unique().astype(str).tolist()

def build_eval_panel_with_union(panel: pd.DataFrame, fold_cfg: Dict[str, str], 
                                union_ids: List[str]) -> pd.DataFrame:
    """
    TEST dates × UNION IDS; missing pairs => ACTIVE=0 and safe feature fills.
    Mirrors the baselines' dynamic universe preparation.
    """
    test_start = pd.to_datetime(fold_cfg["test_start"])
    test_end = pd.to_datetime(fold_cfg["test_end"])

    dates = panel.index.get_level_values("date").unique().sort_values()
    test_dates = dates[(dates >= test_start) & (dates <= test_end)]
    idx = pd.MultiIndex.from_product([test_dates, union_ids], names=["date", "debenture_id"])
    test_aug = panel.reindex(idx).sort_index()

    # Ensure required columns exist
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

    # Safe fills & types
    test_aug["active"] = test_aug["active"].fillna(0).astype(np.int8)
    for c in SAFE_FILL_0:
        if c in test_aug.columns:
            r = test_aug["return"]
            a = test_aug["active"]
            test_aug["return"] = np.where(a.astype(bool), r, 0.0)
    if "sector_id" in test_aug.columns:
        test_aug["sector_id"] = test_aug["sector_id"].fillna(-1).astype(np.int16)

    for name in ["risk_free", "index_return"]:
        test_aug[name] = test_aug[name].astype(np.float32).fillna(0.0)

    if "index_level" in test_aug.columns:
        test_aug["index_level"] = test_aug["index_level"].astype(float).ffill().fillna(0.0)

    return test_aug

# ------------------------------ Metrics ---------------------------------- #

def wealth_from_returns(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()

def max_drawdown(wealth: pd.Series) -> float:
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())

def compute_metrics(returns: pd.Series, rf: pd.Series, bench: pd.Series) -> Dict[str, float]:
    """
    Matches baselines: Sharpe/Sortino on excess vs rf, CAGR/Vol on raw, Calmar vs MaxDD.
    """
    r = returns.fillna(0.0).astype(float)
    rf_a = rf.reindex(r.index).fillna(0.0).astype(float)
    bench_a = bench.reindex(r.index).fillna(0.0).astype(float)

    r_ex = r - rf_a
    mu = r.mean()
    sd = r.std(ddof=1)
    mu_ex = r_ex.mean()
    sd_ex = r_ex.std(ddof=1)
    down = r_ex[r_ex < 0.0].std(ddof=1)

    sharpe_d = (mu_ex / sd_ex) if sd_ex > 0 else np.nan
    sortino_d = (mu_ex / down) if down and down > 0 else np.nan

    wealth = wealth_from_returns(r)
    mdd = max_drawdown(wealth)
    n = max(1, len(r))
    cagr = float(wealth.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
    vol_ann = float(sd * np.sqrt(TRADING_DAYS))
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan

    return {
        "CAGR": cagr,
        "Vol_ann": vol_ann,
        "Sharpe": sharpe_d * np.sqrt(TRADING_DAYS) if not np.isnan(sharpe_d) else np.nan,
        "Sortino": sortino_d * np.sqrt(TRADING_DAYS) if not np.isnan(sortino_d) else np.nan,
        "MaxDD": mdd,
        "Calmar": calmar,
    }

# ------------------------- Weights serialization ------------------------- #

def append_weights_long(results_dir: str,
                        weights_df: pd.DataFrame,
                        strategy: str,
                        fold: int,
                        seed: int):
    """
    weights_df: index = dates, columns = debenture_ids; values = weights
    """
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
    """
    Daily top-k holdings snapshot for quick case-study plots.
    """
    w = weights_df.copy()
    if w.empty:
        return
    w.index = pd.to_datetime(w.index)
    snaps = []
    for dt, row in w.iterrows():
        top = row.sort_values(ascending=False).head(k)
        for rank, (deb_id, wt) in enumerate(top.items(), start=1):
            snaps.append({"date": dt, "strategy": strategy, "fold": int(fold), "seed": int(seed),
                          "debenture_id": str(deb_id), "weight": float(wt), "rank": int(rank)})
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
    Run sensitivity analysis by varying parameters and evaluating performance
    """
    print(f"[SENSITIVITY] Starting analysis for {strategy_label}")
    
    import dataclasses
    
    # Generate all combinations of parameter variations
    param_names = list(param_variations.keys())
    param_values = list(param_variations.values())
    combinations = list(itertools.product(*param_values))
    
    results = []
    
    for i, combo in enumerate(combinations):
        # Create modified config
        modified_cfg = asdict(base_env_cfg)
        for j, param_name in enumerate(param_names):
            modified_cfg[param_name] = combo[j]
        
        # Create environment with modified config
        def make_env():
            env = make_env_from_panel(test_panel, **modified_cfg)
            env = ActionMasker(env, mask_fn)
            return env

        venv = DummyVecEnv([make_env])
        
        # Load VecNormalize if available
        vecnorm_path = os.path.join(results_dir, "..", "models", strategy_label.split("_")[0].lower(), 
                                    "ppo", f"vecnorm_fold_{fold}_seed_{seed}.pkl")
        if os.path.exists(vecnorm_path):
            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = False
            venv.norm_reward = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model
        model = MaskablePPO.load(model_path, device=device)
        
        # Run evaluation
        obs = venv.reset()
        test_dates = test_panel.index.get_level_values("date").unique()
        returns = []
        diagnostics = []
        
        for t in range(len(test_dates)):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            
            # Store results
            returns.append(info.get("portfolio_return", 0.0))
            diagnostics.append({
                "turnover": info.get("turnover", 0.0),
                "hhi": info.get("hhi", 0.0),
                "wealth": info.get("wealth", 1.0)
            })
            
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
        
        print(f"  Combo {i+1}/{len(combinations)}: {dict(zip(param_names, combo))}")
        venv.close()
    
    # Save sensitivity results
    sensitivity_df = pd.DataFrame(results)
    sensitivity_path = os.path.join(results_dir, f"sensitivity_{strategy_label}_f{fold}_s{seed}.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    
    return sensitivity_df

# ------------------------------- Main eval -------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Evaluate PPO on dynamic reconstitution with sensitivity analysis")
    ap.add_argument("--universe", required=True, choices=["cdi", "infra"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_base", default=".")
    ap.add_argument("--seeds", default="", help="Comma-separated seeds; default=from training_config.json")
    ap.add_argument("--folds", default="", help="Comma-separated fold indices to run; default=all from training_folds.json")
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy for SB3.PPO.predict")
    ap.add_argument("--save_topk", action="store_true", help="Also write daily top-k holdings snapshots")
    ap.add_argument("--topk_k", type=int, default=5)
    ap.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    ap.add_argument("--sensitivity_params", type=str, default="lambda_turnover,lambda_hhi,lambda_drawdown,transaction_cost_bps",
                   help="Comma-separated list of parameters to vary in sensitivity analysis")
    ap.add_argument("--sensitivity_values", type=str, default="0.5,1.0,2.0",
                   help="Comma-separated list of multiplicative factors to apply to parameters")
    args = ap.parse_args()

    res_dir = os.path.join(args.out_base, "results", args.universe)
    models_dir = os.path.join(args.out_base, "models", args.universe, "ppo")
    ensure_dir(res_dir)

    # Load processed panel
    panel_path = os.path.join(args.data_dir, f"{args.universe}_processed.pkl")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Missing {panel_path} (run data_final.py first).")
    panel: pd.DataFrame = pd.read_pickle(panel_path)
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "debenture_id"]:
        raise ValueError("Panel must be MultiIndex (date, debenture_id).")

    # Folds & config
    folds_path = os.path.join(res_dir, "training_folds.json")
    cfg_path = os.path.join(res_dir, "training_config.json")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"Missing {folds_path}.")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path}.")

    folds = read_json(folds_path)
    train_cfg = read_json(cfg_path)
    env_cfg = EnvConfig(**train_cfg.get("env_config", {}))

    # Strategy label
    base_label = "PPO"

    # Seeds
    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        # default to training seeds if available; otherwise [0]
        sconf = str(train_cfg.get("seeds", "0"))
        seeds = [int(s) for s in sconf.split(",") if s.strip()]

    # Fold subset
    if args.folds.strip():
        sel_folds = set(int(x) for x in args.folds.split(",") if x.strip())
        folds = [f for f in folds if int(f.get("fold", -1)) in sel_folds]

    # Risk-free (daily)
    rf_path = os.path.join(res_dir, "risk_free_used.csv")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Missing {rf_path} (written by data_final.py).")
    rf_df = pd.read_csv(rf_path, parse_dates=["date"]).set_index("date")
    rf_series = rf_df["risk_free"].astype(float)

    # Sensitivity analysis setup
    if args.sensitivity:
        sensitivity_params = [p.strip() for p in args.sensitivity_params.split(",") if p.strip()]
        sensitivity_factors = [float(v.strip()) for v in args.sensitivity_values.split(",") if v.strip()]
        
        # Create parameter variations
        param_variations = {}
        for param in sensitivity_params:
            if hasattr(env_cfg, param):
                base_value = getattr(env_cfg, param)
                param_variations[param] = [base_value * factor for factor in sensitivity_factors]
            else:
                print(f"[WARN] Parameter {param} not found in env config, skipping")

    # Iterate folds
    for fcfg in folds:
        fold = int(fcfg["fold"])

        # Union IDs exactly as in training (fallback to recompute)
        union = union_ids_from_training(res_dir, fold)
        if union is None:
            print(f"[INFO] Computing union IDs for fold {fold}")
            union = compute_union_ids_from_panel(panel, fcfg)

        # Build TEST panel (dates × union_ids)
        test_aug = build_eval_panel_with_union(panel, fcfg, union)

        # Build env with SAME config used during training
        env = make_env_from_panel(test_aug, **asdict(env_cfg))
        env = ActionMasker(env, mask_fn)

        def make_env():
            env = make_env_from_panel(test_aug, **asdict(env_cfg))
            env = ActionMasker(env, mask_fn)
            return env

        venv = DummyVecEnv([make_env])

        # Date-level series for test
        test_dates = test_aug.index.get_level_values("date").unique().sort_values()
        idx_map = _date_level_map(test_aug, "index_return")
        rf_test = rf_series.reindex(test_dates).fillna(0.0)
        bench_test = idx_map.reindex(test_dates).fillna(0.0)

        # Iterate seeds
        for seed in seeds:
            label = f"{base_label}_f{fold}_s{seed}"
            model_path = os.path.join(models_dir, f"model_fold_{fold}_seed_{seed}.zip")
            if not os.path.exists(model_path):
                print(f"[WARN] Missing model {model_path}; skipping.")
                continue

            # Run sensitivity analysis if requested
            if args.sensitivity and param_variations:
                sensitivity_results = run_sensitivity_analysis(
                    model_path, test_aug, env_cfg, param_variations,
                    res_dir, label, fold, seed
                )
                print(f"[SENSITIVITY] Completed analysis for {label}")

            device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
            # Standard evaluation
            model = MaskablePPO.load(model_path, device=device)

            # Check for VecNormalize stats
            vecnorm_path = os.path.join(models_dir, f"vecnorm_fold_{fold}_seed_{seed}.pkl")
            if os.path.exists(vecnorm_path):
                venv = VecNormalize.load(vecnorm_path, venv)
                venv.training = False
                venv.norm_reward = False
                env = venv

            # Reset and run episode
            if hasattr(env, 'num_envs'):  # VecEnv
                obs = env.reset()
            else:
                obs, _ = env.reset(seed=seed)

            done = False

            # Collectors
            rows_ret = []
            rows_diag = []
            weights_by_date = []
            asset_ids = list(union)

            t = 0
            while not done and t < len(test_dates):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                
                if hasattr(env, 'num_envs'):  # VecEnv
                    obs, rewards, dones, infos = env.step(action)
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    done = dones[0] if isinstance(dones, (list, tuple, np.ndarray)) else dones
                else:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                # Step date
                dt = test_dates[min(len(test_dates)-1, t)]
                t += 1

                # returns: use NET portfolio return from env (costs already applied internally!)
                r_p = float(info.get("portfolio_return", 0.0))

                rows_ret.append({"date": dt, "strategy": label, "return": r_p})

                # diagnostics: turnover, hhi, drawdown, alpha, excess, rf, idx, wealth
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
                    rows_diag.append({"date": dt, "strategy": label, "metric": m, "value": float(v) if v is not None else np.nan})

                # weights snapshot (vector aligned to asset_ids)
                w = info.get("weights", None)
                if w is not None:
                    w = np.asarray(w, dtype=float).ravel()
                    if w.size == len(asset_ids):
                        weights_by_date.append((dt, w))

            # --- Save long returns & diagnostics ---
            df_ret = pd.DataFrame(rows_ret)
            df_diag = pd.DataFrame(rows_diag)
            append_long_csv(os.path.join(res_dir, "all_returns.csv"), df_ret, sort_by=["date", "strategy"])
            append_long_csv(os.path.join(res_dir, "all_diagnostics.csv"), df_diag, sort_by=["date", "strategy", "metric"])

            # --- Save pickle with timeseries + wide weights ---
            if weights_by_date:
                dates_w = [d for d, _ in weights_by_date]
                mat = np.vstack([w for _, w in weights_by_date])
                df_wide = pd.DataFrame(mat, index=pd.to_datetime(dates_w), columns=[str(x) for x in asset_ids]).sort_index()
            else:
                df_wide = pd.DataFrame()

            out_pkl = {
                "returns": df_ret.set_index("date")["return"].astype(float),
                "diagnostics": df_diag,
                "weights": df_wide,
                "asset_ids": [str(x) for x in asset_ids],
            }
            pkl_path = os.path.join(res_dir, f"fold_{fold}_seed_{seed}_results.pkl")
            pd.to_pickle(out_pkl, pkl_path)

            # --- Append weights_long.csv and optional top-k snapshots ---
            if not df_wide.empty:
                append_weights_long(res_dir, df_wide, label, fold, seed)
                if args.save_topk:
                    write_topk_holdings_snapshot(res_dir, df_wide, label, fold, seed, k=args.topk_k)

            # --- Per-run metrics (consistent with baselines) ---
            r_series = df_ret.set_index("date")["return"].astype(float)
            m = compute_metrics(r_series, rf_test, bench_test)
            row = {"strategy": label, "fold": fold, "seed": seed}
            row.update(m)

            fold_cols = ["strategy", "fold", "seed", "CAGR", "Vol_ann", "Sharpe", "Sortino", "MaxDD", "Calmar"]
            append_row_csv(os.path.join(res_dir, "fold_metrics.csv"), row, cols_order=fold_cols)
            append_row_csv(os.path.join(res_dir, "all_metrics.csv"), row, cols_order=fold_cols)

            print(f"  {label}: Sharpe={m['Sharpe']:.3f}, CAGR={m['CAGR']:.3%}")

    print("\n[DONE] Evaluation complete (no ex-post cost applied; returns are net-of-costs from env).")

if __name__ == "__main__":
    main()