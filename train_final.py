# train_final.py
"""
Train PPO with dynamic reconstitution and in-sample validation for reward parameters
------------------------------------------------------------------------

Key changes:
1. Added in-sample validation protocol for reward parameters (lambda_turnover, lambda_hhi, lambda_drawdown)
2. Simplified reward function (removed alpha term)
3. Added configuration file support for hyperparameters
"""

from __future__ import annotations

import os, json, time, argparse, random, yaml, re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure reproducibility across runs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# ---- put this near the top of the file (before importing torch/SB3) ----
import multiprocessing as mp
#try:
#    mp.set_start_method("spawn", force=True)
#except RuntimeError:
#    # already set in this interpreter
#       pass


try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("PyTorch is required for training. Please install torch.") from e

# SB3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.utils import get_action_masks
except Exception as e:
    raise RuntimeError("stable-baselines3 >= 1.7 required. Please install sb3.") from e

# Optional: parallel execution across folds/seeds
try:
    from joblib import Parallel, delayed
    from joblib import parallel_backend
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


# ------------------------------- Configs ----------------------------------- #

@dataclass
class PPOConfig:
    policy: str = "MultiInputPolicy"
    total_timesteps: int = 10_000
    learning_rate: float = 5e-6
    n_steps: int = 12288
    batch_size: int = 4096
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.075
    clip_range_vf: float = 0.075
    ent_coef: float = 0.0003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.3
    target_kl: Optional[float] = 0.125
    net_arch: tuple = (256, 256)
    ortho_init: bool = True
    activation: str = "tanh"


# --------------------------- Environment Config ---------------------------- #
try:
    import gymnasium as gym
except Exception as e:
    raise RuntimeError("gymnasium is required.") from e

try:
    from env_final import EnvConfig, make_env_from_panel
except Exception as e:
    raise RuntimeError("env_final.py with EnvConfig/make_env_from_panel is required.") from e


# -------------------------- Required Panel Columns ------------------------- #

REQUIRED_COLS = [
    "return",
    "spread",
    "duration",
    "time_to_maturity",
    "sector_id",
    "active",
    # date-level (broadcast per date)
    "risk_free",
    "index_return",
    "index_level",
]


# ------------------------------ Helper Utils ------------------------------- #

def _policy_kwargs_from_cfg(ppo_cfg: PPOConfig) -> dict:
    act_fn = nn.Tanh if str(ppo_cfg.activation).lower() == "tanh" else nn.ReLU
    return dict(net_arch=ppo_cfg.net_arch, activation_fn=act_fn, ortho_init=ppo_cfg.ortho_init)

def set_global_seed(seed: int = 0, torch_threads: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def log_memory_usage_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


# ---------------------------- Fold construction ---------------------------- #

def folds_from_dates(dates: pd.DatetimeIndex, n_folds: int = 9, embargo_days: int = 3) -> List[Dict[str, str]]:
    """
    Build expanding window folds where:
    - Training always starts from the first date
    - Training end expands with each fold
    - Test follows training with an embargo gap
    """
    dates = pd.to_datetime(pd.Index(dates).unique().sort_values())
    T = len(dates)
    
    if n_folds <= 1 or T < 10:
        return [{
            "fold": 0,
            "train_start": str(dates[0].date()),
            "train_end": str(dates[int(T * 0.7)].date()),
            "test_start": str(dates[min(int(T * 0.7) + embargo_days, T - 1)].date()),
            "test_end": str(dates[-1].date()),
        }]
    
    # Create cuts for the END of each fold's test period
    # This ensures each fold gets progressively more data
    cuts = np.linspace(0, T - 1, n_folds + 1, dtype=int)
    
    folds = []
    for i in range(n_folds):
        # Training always starts from the beginning (expanding window)
        train_start = dates[0]
        
        # For fold i, we need enough training data
        # Use a portion of the available data up to this fold's cut point
        if i == 0:
            # First fold should have at least some reasonable amount of training data
            # Use first 1/(n_folds+1) of data for training, rest for test
            train_end_idx = max(int(T / (n_folds + 1)), 10)  # At least 10 days
        else:
            # For subsequent folds, train up to the previous fold's test end
            train_end_idx = cuts[i]
        
        train_end = dates[min(train_end_idx, T - 1)]
        
        # Test starts after embargo
        test_start_idx = min(train_end_idx + embargo_days + 1, T - 1)
        test_end_idx = cuts[i + 1] if i + 1 < len(cuts) else T - 1
        
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        
        # Skip if insufficient data
        if test_start_idx >= test_end_idx or train_end_idx < 5:
            continue
            
        fold = {
            "fold": int(i),
            "train_start": str(pd.Timestamp(train_start).date()),
            "train_end": str(pd.Timestamp(train_end).date()),
            "test_start": str(pd.Timestamp(test_start).date()),
            "test_end": str(pd.Timestamp(test_end).date()),
        }
        folds.append(fold)
    
    return folds

def _periodical_alpha_from_daily(dates, r_port, r_bench, period="W-FRI"):
    """
    Build periodical (close) active returns using log-relative compounding:
      alpha_w = exp( sum_t log( (1+r_p)/(1+r_b) ) ) - 1
    This is stable when daily r are tiny and avoids overlap.
    """
    idx = pd.to_datetime(pd.Series(dates), utc=False)
    rp = pd.Series(np.asarray(r_port, dtype=float), index=idx).astype(float)
    rb = pd.Series(np.asarray(r_bench, dtype=float), index=idx).astype(float)

    # log-relative daily
    lr = np.log1p(rp) - np.log1p(rb)
    # non-overlapping periodical aggregation (close)
    lr_p = lr.resample(period).sum(min_count=1)
    alpha_p = np.expm1(lr_p).dropna()
    return alpha_p


# ------------------------------ Panel Builders ----------------------------- #

def _date_level_map(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(dtype=float)
    return panel[[col]].groupby(level="date").first()[col].sort_index()

def build_train_panel_with_union(panel: pd.DataFrame, fold_cfg: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reindex TRAIN window to (dates × union-of-IDs computed on [train_start, test_end]).
    Fill ACTIVE=0 for missing pairs and broadcast date-level values safely.
    """
    train_start = pd.to_datetime(fold_cfg["train_start"])
    train_end   = pd.to_datetime(fold_cfg["train_end"])
    test_end    = pd.to_datetime(fold_cfg["test_end"])

    union_slice = panel.loc[
        (panel.index.get_level_values("date") >= train_start) &
        (panel.index.get_level_values("date") <= test_end)
    ]
    union_ids = union_slice.index.get_level_values("debenture_id").unique().tolist()

    train_dates = panel.index.get_level_values("date").unique().sort_values()
    train_dates = train_dates[(train_dates >= train_start) & (train_dates <= train_end)]

    idx = pd.MultiIndex.from_product([train_dates, union_ids], names=["date", "debenture_id"])
    train_aug = panel.reindex(idx)

    dates = train_aug.index.get_level_values("date").unique()
    if len(dates) < 10:  # Minimum threshold
        print(f"[WARN] Fold {fold_cfg['fold']} has only {len(dates)} dates in training data")
        print(f"  Train period: {fold_cfg['train_start']} to {fold_cfg['train_end']}")

    # Safe fills mirroring evaluate path
    train_aug["active"] = train_aug["active"].fillna(0).astype(np.int8)
    for c in ["return", "spread", "duration", "time_to_maturity"]:
        if c in train_aug.columns:
            train_aug[c] = train_aug[c].astype(float).fillna(0.0)
    if "sector_id" in train_aug.columns:
        train_aug["sector_id"] = train_aug["sector_id"].fillna(-1).astype(np.int16)

    # Broadcast date-level columns
    for name in ["risk_free", "index_return", "index_level"]:
        m = _date_level_map(panel, name)
        if not m.empty:
            df = train_aug.reset_index()
            df[name] = df[name].astype(float).fillna(df["date"].map(m).astype(float))
            train_aug = df.set_index(["date", "debenture_id"]).sort_index()

    for name in ["risk_free", "index_return"]:
        train_aug[name] = train_aug[name].astype(np.float32).fillna(0.0)

    return train_aug.sort_index(), union_ids


# ---------------------- Safe Wrappers & Callbacks -------------------------- #

class SafeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, clip: float = 7.5):
        super().__init__(env)
        self._clip = float(clip)

    def observation(self, obs):
        # Handle dict observations for MaskablePPO
        if isinstance(obs, dict):
            processed = {}
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    val = np.nan_to_num(val, nan=0.0, posinf=self._clip, neginf=-self._clip)
                    processed[key] = np.clip(val, -self._clip, self._clip).astype(np.float32)
                else:
                    processed[key] = val
            return processed
        else:
            # Regular array observation
            obs = np.nan_to_num(obs, nan=0.0, posinf=self._clip, neginf=-self._clip)
            return np.clip(obs, -self._clip, self._clip).astype(np.float32)

class SafeRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._warned = False

    def reward(self, rew):
        if not np.isfinite(rew):
            if not self._warned:
                print("[WARN] Non-finite reward encountered; replacing with 0.0 (once per episode).")
                self._warned = True
            return 0.0
        return float(rew)

class StopOnNaNCallback(BaseCallback):
    def _on_step(self) -> bool:
        try:
            for p in self.model.policy.parameters():
                if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    print("[ERROR] NaN/Inf detected in policy parameters — stopping training.")
                    return False
        except Exception:
            pass
        return True

class MetricsLoggerCallback(BaseCallback):
    """
    Lightweight training logger. Collects a few final stats at training end and
    writes them to <results>/<universe>/training_metrics.json.

    NOTE: BaseCallback is abstract and requires _on_step() -> bool, even if we
    don't need per-step logic.
    """
    def __init__(self, out_json_path: str, fold: int, seed: int, verbose: int = 0):
        super().__init__(verbose)
        self.out_json_path = out_json_path
        self.fold = int(fold)
        self.seed = int(seed)
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        # quick sanity check on observations
        try:
            obs = self.model.policy.obs_to_tensor(self.model.rollout_buffer.observations)[0]
            if not torch.isfinite(obs).all():
                print("[ERROR] Non-finite observation tensor at training start.")
                # Returning False here would abort training; we just warn.
        except Exception:
            pass

    def _on_step(self) -> bool:
        # Required by BaseCallback; return True to continue training.
        return True

    def _on_rollout_end(self) -> None:
        # Optional: sanity check buffer content
        try:
            ob = self.model.rollout_buffer.observations
            if isinstance(ob, np.ndarray) and not np.isfinite(ob).all():
                bad = np.where(~np.isfinite(ob))
                print(f"[WARN] Non-finite obs in rollout buffer at positions {bad[0][:3]}...")
        except Exception:
            pass

    def _on_training_end(self) -> None:
        elapsed = (time.time() - self.start_time) if self.start_time is not None else float("nan")
        stats = {}
        try:
            maybe = getattr(self.model, "logger", None)
            if maybe is not None and hasattr(maybe, "name_to_value"):
                for k in ["train/entropy_loss", "train/policy_gradient_loss", "train/value_loss", "train/approx_kl"]:
                    if k in maybe.name_to_value:
                        stats[k.split("/")[-1]] = float(maybe.name_to_value[k])
        except Exception:
            pass

        rec = {
            "fold": self.fold,
            "seed": self.seed,
            "elapsed_sec": float(elapsed),
            "rss_mb": float(log_memory_usage_mb()),
            **stats,
        }

        # Append atomically to JSON list
        try:
            old = []
            if os.path.exists(self.out_json_path):
                with open(self.out_json_path, "r") as f:
                    old = json.load(f)
                    if not isinstance(old, list):
                        old = []
            old.append(rec)
            tmp = self.out_json_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(old, f, indent=2)
            os.replace(tmp, self.out_json_path)
        except Exception as e:
            print(f"[WARN] Failed writing training_metrics.json: {e}")


# ------------------------ In-Sample Validation ----------------------------- #

def validate_reward_params(
    train_aug: pd.DataFrame,
    fold_cfg: Dict[str, str],
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    lambda_grid: Dict[str, List[float]],
    selection_metric: str = "ir",
    seed: int = 0,
    inner_total_timesteps: int = 10_000,
) -> Dict[str, float]:
    """
    Grid-search λ penalties on a terminal validation slice:
      • Inner-train on [train_start, train_end_80%]
      • Deterministic eval on (train_end_80%, train_end]
      • Pick λ that maximizes `selection_metric` (ir|sharpe|sortino|calmar)

    Assumes `train_aug` is the TRAIN window reindexed to (dates × union-of-IDs) with
    safe fills and date-level columns broadcast (same format your `train_one` builds).
    """
    # ---- helpers -------------------------------------------------------------
    import dataclasses
    from dataclasses import asdict
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan

    # Optional wrappers (no-ops if not available)
    try:
        from env_final import SafeRewardWrapper, SafeObsWrapper
    except Exception:
        class SafeRewardWrapper(gym.RewardWrapper):
            def reward(self, r): return 0.0 if not np.isfinite(r) else float(r)
        class SafeObsWrapper(gym.ObservationWrapper):
            def __init__(self, env, clip=7.5): super().__init__(env); self.clip=float(clip)
            def observation(self, obs):
                obs = np.nan_to_num(obs, nan=0.0, posinf=self.clip, neginf=-self.clip)
                return np.clip(obs, -self.clip, self.clip).astype(np.float32)

    TRADING_DAYS = 252
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_vecenv(panel_slice: pd.DataFrame, cfg: EnvConfig, gamma: float) -> VecNormalize:
        def _make():
            e = make_env_from_panel(panel_slice, **asdict(cfg))
            e = SafeRewardWrapper(e)
            e = SafeObsWrapper(e, clip=getattr(cfg, "obs_clip", 7.5) or 7.5)
            return e
        venv = DummyVecEnv([_make])                 # single env for inner runs
        venv = VecCheckNan(venv)
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=gamma)
        return venv

    def _wealth_cagr(r: np.ndarray) -> float:
        n = max(1, r.size)
        return float((np.prod(1.0 + r)) ** (TRADING_DAYS / n) - 1.0)

    def _maxdd(r: np.ndarray) -> float:
        if r.size == 0: return 0.0
        w = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(w)
        dd = w / np.maximum(peak, 1e-12) - 1.0
        return float(dd.min())  # negative

    def _metric_score(returns: np.ndarray, idx: np.ndarray, rf: np.ndarray, how: str) -> float:
        how = str(how).lower()
        if returns.size == 0: return -np.inf
        if how == "sharpe":
            ex = returns - rf
            sd = ex.std(ddof=1)
            return float((ex.mean() / sd) * np.sqrt(TRADING_DAYS)) if sd > 0 else -np.inf
        if how == "sortino":
            ex = returns - rf
            down = ex[ex < 0]
            dsd = down.std(ddof=1) if down.size > 1 else np.nan
            return float((ex.mean() / dsd) * np.sqrt(TRADING_DAYS)) if (isinstance(dsd, float) and dsd > 0) else -np.inf
        if how == "calmar":
            cagr = _wealth_cagr(returns)
            mdd  = _maxdd(returns)
            return float(cagr / abs(mdd)) if mdd < 0 else -np.inf
        # default: IR vs index
        ar = returns - idx
        sd = ar.std(ddof=1)
        return float((ar.mean() / sd) * np.sqrt(TRADING_DAYS)) if sd > 0 else -np.inf

    # ---- split TRAIN into inner-train / validation ---------------------------
    dates_all = train_aug.index.get_level_values(0).unique().sort_values()
    train_start = pd.to_datetime(fold_cfg["train_start"])
    train_end   = pd.to_datetime(fold_cfg["train_end"])

    # constrain to TRAIN window in case train_aug is larger
    dates = dates_all[(dates_all >= train_start) & (dates_all <= train_end)]
    if dates.size < 10:
        # too short to split: fall back to current λs
        return {
            "lambda_turnover": float(env_cfg.lambda_turnover),
            "lambda_hhi": float(env_cfg.lambda_hhi),
            "lambda_drawdown": float(env_cfg.lambda_drawdown),
        }

    cut = int(np.floor(dates.size * 0.8))
    cut = min(max(1, cut), dates.size - 1)
    tr_dates = dates[:cut]
    va_dates = dates[cut:]

    def _slice(df: pd.DataFrame, dfrom: pd.Timestamp, dto: pd.Timestamp) -> pd.DataFrame:
        mask = (df.index.get_level_values(0) >= dfrom) & (df.index.get_level_values(0) <= dto)
        return df[mask].copy()

    train_slice = _slice(train_aug, tr_dates.min(), tr_dates.max())
    val_slice   = _slice(train_aug, va_dates.min(),   va_dates.max())

    if train_slice.empty or val_slice.empty:
        return {
            "lambda_turnover": float(env_cfg.lambda_turnover),
            "lambda_hhi": float(env_cfg.lambda_hhi),
            "lambda_drawdown": float(env_cfg.lambda_drawdown),
        }

    # ---- grid over λ ---------------------------------------------------------
    grid_to  = lambda_grid.get("lambda_turnover", [env_cfg.lambda_turnover])
    grid_hhi = lambda_grid.get("lambda_hhi",      [env_cfg.lambda_hhi])
    grid_dd  = lambda_grid.get("lambda_drawdown", [env_cfg.lambda_drawdown])

    best_score = -np.inf
    best = (float(env_cfg.lambda_turnover), float(env_cfg.lambda_hhi), float(env_cfg.lambda_drawdown))

    for lam_to in grid_to:
        for lam_hhi in grid_hhi:
            for lam_dd in grid_dd:
                cfg_i = dataclasses.replace(
                    env_cfg,
                    lambda_turnover=float(lam_to),
                    lambda_hhi=float(lam_hhi),
                    lambda_drawdown=float(lam_dd),
                    seed=int(seed),
                )

                # === inner TRAIN ===
                venv_tr = _build_vecenv(train_slice, cfg_i, gamma=ppo_cfg.gamma)
                policy_kwargs = dict(
                    net_arch=ppo_cfg.net_arch,
                    activation_fn=(nn.Tanh if str(ppo_cfg.activation).lower() == "tanh" else nn.ReLU),
                    ortho_init=ppo_cfg.ortho_init,
                )
                model = MaskablePPO(
                    ppo_cfg.policy,
                    venv_tr,
                    learning_rate=ppo_cfg.learning_rate,
                    n_steps=ppo_cfg.n_steps,  # Keep at 2048 or even increase to 4096
                    batch_size=ppo_cfg.batch_size,  # Keep at 512 or increase to 1024
                    n_epochs=ppo_cfg.n_epochs,
                    gamma=ppo_cfg.gamma,
                    gae_lambda=ppo_cfg.gae_lambda,
                    clip_range=ppo_cfg.clip_range,
                    clip_range_vf=ppo_cfg.clip_range_vf,
                    ent_coef=ppo_cfg.ent_coef,
                    vf_coef=ppo_cfg.vf_coef,
                    max_grad_norm=ppo_cfg.max_grad_norm,
                    target_kl=ppo_cfg.target_kl,
                    tensorboard_log=None,
                    policy_kwargs=policy_kwargs,
                    verbose=0,
                    device=device,
                )
                model.learn(total_timesteps=int(inner_total_timesteps), progress_bar=False)

                # === inner VAL (deterministic rollout) ===
                venv_va = _build_vecenv(val_slice, cfg_i, gamma=ppo_cfg.gamma)
                # copy normalization stats from train -> val
                try:
                    venv_va.obs_rms = venv_tr.obs_rms
                    venv_va.ret_rms = venv_tr.ret_rms
                except Exception:
                    pass
                venv_va.training = False
                venv_va.norm_reward = False

                obs = venv_va.reset()  # VecEnv.reset(): obs only
                ret_list, idx_list, rf_list = [], [], []
                T = val_slice.index.get_level_values(0).unique().size

                for _ in range(T):
                    action, _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(env))
                    # VecEnv.step() => (obs, rewards, dones, infos)
                    obs, rewards, dones, infos = venv_va.step(action)
                    info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

                    # robust fields from info
                    rp = float(info0.get("portfolio_return", info0.get("portfolio_return_net", 0.0)))
                    if "portfolio_return_net" not in info0 and "trade_cost" in info0 and info0.get("is_net", None) is False:
                        # gross -> net fallback
                        rp = (1.0 + rp) * (1.0 - max(float(info0["trade_cost"]), 0.0)) - 1.0
                    idxr = float(info0.get("index_return", info0.get("benchmark_return", 0.0)))
                    rfr  = float(info0.get("risk_free", info0.get("rf", 0.0)))

                    ret_list.append(rp); idx_list.append(idxr); rf_list.append(rfr)

                    if np.any(dones):
                        break

                r  = np.asarray(ret_list, dtype=float)
                ib = np.asarray(idx_list, dtype=float)
                rf = np.asarray(rf_list, dtype=float)

                score = _metric_score(r, ib, rf, selection_metric)
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best = (float(lam_to), float(lam_hhi), float(lam_dd))

                # tidy
                try: venv_tr.close()
                except Exception: pass
                try: venv_va.close()
                except Exception: pass

    return {"lambda_turnover": best[0], "lambda_hhi": best[1], "lambda_drawdown": best[2]}


def evaluate_on_validation(env, model) -> dict:
    """
    Deterministic one-episode roll; returns sharpe, sortino, ir, calmar.

    Works with either:
      • a vectorized env (e.g., VecNormalize(DummyVecEnv([...])))
      • a plain Gymnasium env (with Safe* wrappers)

    IR is computed vs the env's 'idx' (index_return). Reward stays unchanged.
    """
    import numpy as np

    def _maxdd(r: np.ndarray) -> float:
        if r.size == 0:
            return 0.0
        w = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(w)
        dd = w / np.maximum(peak, 1e-12) - 1.0
        return float(dd.min())  # <-- no premature float(...) on an array

    # Detect vectorized env (DummyVecEnv / VecNormalize) vs plain
    is_vec = hasattr(env, "num_envs")

    # Reset
    if is_vec:
        obs = env.reset()
        # SB3 returns np.ndarray for vec envs; ensure shape (n_envs, ...)
        if isinstance(obs, tuple):  # (obs, info) for some gym APIs
            obs = obs[0]
    else:
        obs, _ = env.reset()

    rets, excess, alpha = [], [], []
    idxs, dates = [], []

    done = False
    steps = 0
    while not done:
        if is_vec:
            action, _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(env))
            obs, r, terminated, truncated, info = env.step(action)
            i = info[0] if isinstance(info, (list, tuple)) else info
            done = bool(terminated or truncated)
        else:
            action, _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(env))(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            i = info

        rp = float(i.get("portfolio_return", 0.0))
        rf = float(i.get("risk_free", 0.0))
        idx = float(i.get("index_return", 0.0))
        dt = pd.Timestamp(i.get("date"))
        rets.append(rp)
        excess.append(rp - rf)
        alpha.append(rp - idx)
        idxs.append(idx)
        dates.append(dt)

        steps += 1
        if steps > 1_000_000:  # safety
            break

    r = np.asarray(rets, dtype=float)
    ex = np.asarray(excess, dtype=float)
    al = np.asarray(alpha, dtype=float)

    out = {}
    if r.size > 1:
        n = max(1, r.size)
        cagr = float((np.prod(1.0 + r)) ** (252.0 / n) - 1.0)
        vol = float(r.std(ddof=1) * np.sqrt(252.0))
        out.update({"cagr": cagr, "vol": vol})

    if ex.size > 1:
        sd = float(ex.std(ddof=1))
        out["sharpe"] = float((ex.mean() / sd) * np.sqrt(252.0)) if sd > 0 else float("nan")

    if al.size > 1:
        sd_a = float(al.std(ddof=1))
        out["ir"] = float((al.mean() / sd_a) * np.sqrt(252.0)) if sd_a > 0 else float("nan")

    out["maxdd"] = float(_maxdd(r)) if r.size else float("nan")
    return out


# ------------------------------ Train One ---------------------------------- #

def _latest_checkpoint(ckpt_dir: str) -> Tuple[Optional[str], Optional[int]]:
    if not os.path.isdir(ckpt_dir):
        return None, None
    best = (None, -1)
    for fn in os.listdir(ckpt_dir):
        if fn.endswith(".zip"):
            m = re.search(r"ppo_checkpoint_(\d+)_steps\.zip", fn)
            if not m:
                continue
            steps = int(m.group(1))
            if steps > best[1]:
                best = (os.path.join(ckpt_dir, fn), steps)
    return best


def train_one(
    universe: str,
    panel: pd.DataFrame,
    fold_cfg: Dict[str, str],
    seed: int,
    ppo_cfg: PPOConfig,
    env_cfg: EnvConfig,
    out_base: str,
    lambda_grid: Optional[Dict[str, List[float]]] = None,
    save_freq: int = 50_000,
    selection_metric: str = "ir",
    resume: bool = False,
    n_envs: int = 1,
    vec_kind: str = "subproc",
    episode_len: Optional[int] = None,
    reset_jitter_frac: float = 0.9,
) -> Dict[str, str]:
    """
    Train a PPO policy for a single (fold, seed).

    Side-effects:
      - Ensures results/, models/, tensorboard/ dirs
      - Appends/updates training_union_ids.json with this fold's union IDs
      - Saves final model to models/<universe>/ppo/model_fold_<f>_seed_<s>.zip
      - Saves rolling checkpoints to models/<universe>/ppo/fold_<f>_seed_<s>/ppo_checkpoint_XXXX_steps.zip
    """
    # --------------------- housekeeping & dirs --------------------- #
    fold_i = int(fold_cfg["fold"])
    seed = int(seed)

    results_dir = os.path.join(out_base, "results", universe)
    model_dir   = os.path.join(out_base, "models", universe, "ppo")
    tb_dir      = os.path.join(out_base, "tb", universe)
    ensure_dirs(results_dir, model_dir, tb_dir)

    # TensorBoard subdir per run
    tb_dir = os.path.join(tb_dir, f"fold_{fold_i}_seed_{seed}")

    # Checkpoint directory
    ckpt_dir = os.path.join(model_dir, f"fold_{fold_i}_seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --------------------- build training panel -------------------- #
    # Uses union (train..test) to avoid universe drift between train/test
    train_aug, union_ids = build_train_panel_with_union(panel, fold_cfg)

    # Persist/merge union IDs for evaluation reproducibility
    union_path = os.path.join(results_dir, "training_union_ids.json")
    try:
        existing = []
        if os.path.exists(union_path):
            with open(union_path, "r") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        existing = [r for r in existing if int(r.get("fold", -1)) != fold_i]
        existing.append({"fold": int(fold_i), "ids": list(map(str, union_ids))})
        with open(union_path, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not persist union ids: {e}")

    # ------------------- pick reward penalties (λ) ----------------- #
    if lambda_grid is not None and isinstance(lambda_grid, dict):
        best = validate_reward_params(
            train_aug, fold_cfg, env_cfg, ppo_cfg, lambda_grid=lambda_grid, selection_metric=selection_metric
        )
        # Update env_cfg with selected λ
        env_cfg = EnvConfig(**{**asdict(env_cfg), **best})
        print(f"[INFO] Selected λ on validation: {best}")

    # ------------------------- build env ---------------------------- #
    # Make vectorized training env(s) from augmented training panel
    def _make_thunk(rank: int):
        def _init():
            cfg_i = EnvConfig(**{**asdict(env_cfg)})
            if cfg_i.seed is not None:
                cfg_i.seed = int(cfg_i.seed) + int(rank)
            
            # Check available data length and adjust max_steps accordingly
            train_dates = train_aug.index.get_level_values("date").unique()
            available_steps = len(train_dates)
            
            if episode_len is not None and episode_len > available_steps:
                print(f"[WARN] Requested episode_len={episode_len} but only {available_steps} steps available")
                cfg_i.max_steps = min(episode_len, available_steps - 1)
            elif episode_len is not None:
                cfg_i.max_steps = episode_len
            
            # Adjust random_reset_frac if needed
            if hasattr(cfg_i, "random_reset_frac") and available_steps < 10:
                cfg_i.random_reset_frac = 0.0  # Disable random reset for very short sequences
            e = make_env_from_panel(train_aug, **asdict(cfg_i))
            e = SafeRewardWrapper(e)
            e = SafeObsWrapper(e, clip=getattr(cfg_i, "obs_clip", 7.5) or 7.5)
            return e
        return _init

    thunks = [_make_thunk(i) for i in range(max(1, int(n_envs)))]
    if int(n_envs) <= 1 or str(vec_kind).lower() == "dummy":
        vec_env = DummyVecEnv(thunks)
    else:
        vec_env = SubprocVecEnv(thunks, start_method="spawn")

    # vec_env = VecCheckNan(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=ppo_cfg.gamma)

    # ---------------------- PPO instantiation ----------------------- #
    set_global_seed(seed=seed)
    policy_kwargs = _policy_kwargs_from_cfg(ppo_cfg)

    # Compute remaining timesteps in case of resume
    already_trained = 0
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if resume:
        cp, steps = _latest_checkpoint(ckpt_dir)
        if cp is not None:
            print(f"[INFO] Resuming fold {fold_i} seed {seed} from: {cp} ({steps} steps)")
            try:
                model = PPO.load(cp, env=vec_env, device="auto")
                already_trained = int(getattr(model, "num_timesteps", steps or 0))
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint; training fresh. Err={e}")

    if model is None:
        model = PPO(
            ppo_cfg.policy,
            vec_env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,  # Keep at 2048 or even increase to 4096
            batch_size=ppo_cfg.batch_size,  # Keep at 512 or increase to 1024
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            clip_range=ppo_cfg.clip_range,
            clip_range_vf=ppo_cfg.clip_range_vf,
            ent_coef=ppo_cfg.ent_coef,
            vf_coef=ppo_cfg.vf_coef,
            max_grad_norm=ppo_cfg.max_grad_norm,
            target_kl=ppo_cfg.target_kl,
            tensorboard_log=tb_dir,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
        )

    # --------------------------- learn ------------------------------ #
    remaining = max(0, int(ppo_cfg.total_timesteps) - int(already_trained))
    callbacks = []
    callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=ckpt_dir, name_prefix="ppo_checkpoint"))
    # callbacks.append(StopOnNaNCallback())
    metrics_json = os.path.join(results_dir, "training_metrics.json")
    callbacks.append(MetricsLoggerCallback(out_json_path=metrics_json, fold=fold_i, seed=seed))
    callback = CallbackList(callbacks)

    if remaining == 0:
        print(f"[INFO] Nothing left to train (remaining=0). Saving model and exiting.")
    else:
        print(f"[INFO] Training fold {fold_i} seed {seed} for {remaining:,} timesteps "
              f"(n_envs={n_envs}, n_steps_eff={max(8, int(ppo_cfg.n_steps // max(1, int(n_envs))))}, "
              f"batch_size_eff={max(64, int(ppo_cfg.batch_size // max(1, int(n_envs))))})")
        model.learn(total_timesteps=int(remaining), callback=callback, progress_bar=False)

    # ----------------------------- save -------------------------------- #
    final_model_path = os.path.join(model_dir, f"model_fold_{fold_i}_seed_{seed}.zip")
    model.save(final_model_path)
    try:
        # persist VecNormalize statistics alongside model for consistent eval
        vecnorm_path = os.path.join(model_dir, f"vecnorm_fold_{fold_i}_seed_{seed}.pkl")
        vec_env.save(vecnorm_path)
    except Exception:
        pass

    vec_env.close()

    return {
        "model_path": final_model_path,
        "tensorboard_dir": tb_dir,
        "fold": str(fold_i),
        "seed": str(seed),
    }


# --------------------------------- Main ------------------------------------ #

def _load_panel_for_universe(universe: str, data_dir: str) -> pd.DataFrame:
    """
    The panel should be a MultiIndex (date, debenture_id) dataframe with required columns.
    You should have run your data pipeline before training.
    """
    proc_path = os.path.join(data_dir, f"{universe}_processed.pkl")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"Processed panel not found: {proc_path}. Run data_final.py first.")
    df = pd.read_pickle(proc_path)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Panel index must be MultiIndex (date, debenture_id).")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    return df

def _load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train PPO with in-sample validation for reward parameters")
    parser.add_argument("--universe", type=str, choices=["cdi", "infra"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_base", type=str, default=".")
    parser.add_argument("--n_folds", type=int, default=9)
    parser.add_argument("--embargo_days", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--selection_metric",
        default="ir",
        choices=["ir","sharpe","sortino","calmar"],
        help="Metric used to pick reward penalties during validation",
    )
    parser.add_argument("--skip_finished", action="store_true",
                    help="Skip (fold,seed) if final model already exists")
    parser.add_argument("--resume", action="store_true",
                    help="Resume training from latest checkpoint if available")
    parser.add_argument("--n_envs", type=int, default=1, help="Parallel envs per PPO run (SubprocVecEnv when >1)")
    parser.add_argument("--episode_len", type=int, default=None, help="Optional max steps per episode during training")
    parser.add_argument("--reset_jitter_frac", type=float, default=0.9, help="Fraction of window eligible for random reset in training resets")
    parser.add_argument("--vec", type=str, default="subproc", choices=["dummy","subproc"], help="Vectorized env backend for training")

    args = parser.parse_args()

    # Load configuration from YAML file
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    universe = args.universe.lower()
    panel: pd.DataFrame = _load_panel_for_universe(universe, args.data_dir)

    dates = panel.index.get_level_values("date").unique().sort_values()
    fold_specs = folds_from_dates(dates, n_folds=args.n_folds, embargo_days=args.embargo_days)

    # PPO settings
    net_arch = config.get('net_arch', [256, 256])
    if isinstance(net_arch, list):
        net_arch = tuple(net_arch)

    lambda_grid = config.get('lambda_grid', {
        "lambda_turnover": [0.0001, 0.0002, 0.0004],
        "lambda_hhi": [0.005, 0.01, 0.02],
        "lambda_drawdown": [0.0025, 0.005, 0.01]
    })

    ppo_cfg = PPOConfig(
        total_timesteps=config.get('total_timesteps', 10_000),
        learning_rate=config.get('learning_rate', 5e-6),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 512),
        n_epochs=config.get('n_epochs', 5),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.075),
        clip_range_vf=config.get('clip_range_vf', 0.075),
        ent_coef=config.get('ent_coef', 0.0003),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.3),
        target_kl=config.get('target_kl', 0.125),
        net_arch=net_arch,
        activation=config.get('activation', 'tanh'),
        ortho_init=bool(config.get('ortho_init', True)),
    )

    # Env settings
    env_cfg = EnvConfig(
        rebalance_interval=config.get('rebalance_interval', 5),
        max_weight=config.get('max_weight', 0.10),   # <-- was per_name_cap
        allow_cash=config.get('allow_cash', True),
        transaction_cost_bps=config.get('transaction_cost_bps', 10.0),
        normalize_features=config.get('normalize_features', True),
        seed=None,
        obs_clip=config.get('obs_clip', 7.5),

        # these can be overwritten by validation:
        lambda_turnover=config.get('lambda_turnover', 0.0002),
        lambda_hhi=config.get('lambda_hhi', 0.01),
        lambda_drawdown=config.get('lambda_drawdown', 0.005),

        # optional extra reward weights supported by your env:
        weight_excess=config.get('weight_excess', 0.0),
        weight_alpha=config.get('weight_alpha', 1.0),
    )


    # Jobs
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    jobs: List[Tuple[Dict[str, str], int]] = []
    for fold in fold_specs:
        for sd in seeds:
            jobs.append((fold, sd))

    # Run (optionally parallel)
    out_base = args.out_base


    for fold, sd in jobs:
        set_global_seed(seed=sd)
        if args.skip_finished:
            final_model_path = os.path.join(
                args.out_base, "models", universe, "ppo", f"model_fold_{int(fold['fold'])}_seed_{sd}.zip"
            )
            if os.path.exists(final_model_path):
                print(f"[INFO] Skipping fold {fold['fold']} seed {sd} (final model exists).")
                continue

        train_one(
            universe, panel, fold_cfg=fold, seed=sd,
            ppo_cfg=ppo_cfg, env_cfg=env_cfg,
            out_base=out_base, lambda_grid=lambda_grid,
            save_freq=config.get('checkpoint_freq', 50_000),
            selection_metric=args.selection_metric,
            resume=args.resume,
            n_envs=args.n_envs,
            vec_kind=args.vec,
            episode_len=args.episode_len,
            reset_jitter_frac=args.reset_jitter_frac,
        )


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize
    import torch, torch.nn as nn
    main()
