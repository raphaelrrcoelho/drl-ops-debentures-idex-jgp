# train_final.py
"""
Train MaskablePPO with DISCRETE action space and fold-specific asset unions
Memory-efficient approach that maintains fixed dimensions per fold
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*get_schedule_fn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*constant_fn.*deprecated.*")

import os, json, time, argparse, random, yaml, re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Ensure reproducibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("PyTorch is required for training. Please install torch.") from e

# SB3 and MaskablePPO for discrete actions
try:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from sb3_contrib.common.wrappers import ActionMasker
except Exception as e:
    raise RuntimeError("sb3-contrib is required for MaskablePPO. Please install sb3-contrib.") from e

try:
    import gymnasium as gym
except Exception as e:
    raise RuntimeError("gymnasium is required.") from e

try:
    from env_final import EnvConfig, make_env_from_panel
except Exception as e:
    raise RuntimeError("env_final.py with discrete action space is required.") from e

# ------------------------------- Configs ----------------------------------- #

@dataclass
class PPOConfig:
    policy: str = "MultiInputPolicy"
    total_timesteps: int = 100_000
    learning_rate: float = 5e-6
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.075
    clip_range_vf: float = 0.075
    ent_coef: float = 0.0003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.3
    target_kl: Optional[float] = 0.125
    net_arch: tuple = (128, 128)
    ortho_init: bool = True
    activation: str = "tanh"

# -------------------------- Required Panel Columns ------------------------- #

REQUIRED_COLS = [
    "return", "spread", "duration", "time_to_maturity", "sector_id", "active",
    "risk_free", "index_return", "index_level",
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
    """Build expanding window folds."""
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
    
    cuts = np.linspace(0, T - 1, n_folds + 1, dtype=int)
    folds = []
    
    for i in range(n_folds):
        train_start = dates[0]
        if i == 0:
            train_end_idx = max(int(T / (n_folds + 1)), 10)
        else:
            train_end_idx = cuts[i]
        
        train_end = dates[min(train_end_idx, T - 1)]
        test_start_idx = min(train_end_idx + embargo_days + 1, T - 1)
        test_end_idx = cuts[i + 1] if i + 1 < len(cuts) else T - 1
        
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        
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

# ------------------------------ Panel Builders ----------------------------- #

def _date_level_map(panel: pd.DataFrame, col: str) -> pd.Series:
    if col not in panel.columns:
        return pd.Series(dtype=float)
    return panel[[col]].groupby(level="date").first()[col].sort_index()

def build_train_panel_with_fold_union(panel: pd.DataFrame, fold_cfg: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build training panel using union of assets that appear in THIS FOLD only.
    This is the key to memory efficiency - we only include assets relevant to this fold.
    """
    train_start = pd.to_datetime(fold_cfg["train_start"])
    train_end = pd.to_datetime(fold_cfg["train_end"])
    test_end = pd.to_datetime(fold_cfg["test_end"])
    
    # Get union of assets that appear anywhere from train_start to test_end
    fold_slice = panel.loc[
        (panel.index.get_level_values("date") >= train_start) &
        (panel.index.get_level_values("date") <= test_end)
    ]
    
    # Only include assets that are active at least once in the fold period
    active_mask = fold_slice["active"] > 0
    active_slice = fold_slice[active_mask]
    union_ids = active_slice.index.get_level_values("debenture_id").unique().tolist()
    
    print(f"  Fold union: {len(union_ids)} assets (from {fold_slice.index.get_level_values('debenture_id').nunique()} total)")
    
    # Build training panel with these IDs
    train_dates = panel.index.get_level_values("date").unique().sort_values()
    train_dates = train_dates[(train_dates >= train_start) & (train_dates <= train_end)]
    
    idx = pd.MultiIndex.from_product([train_dates, union_ids], names=["date", "debenture_id"])
    train_aug = panel.reindex(idx)
    
    # Safe fills
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

# ---------------------- Wrappers for Discrete Actions ---------------------- #

def mask_fn(env: gym.Env) -> np.ndarray:
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

class SafeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, clip: float = 7.5):
        super().__init__(env)
        self._clip = float(clip)

    def observation(self, obs):
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
            obs = np.nan_to_num(obs, nan=0.0, posinf=self._clip, neginf=-self._clip)
            return np.clip(obs, -self._clip, self._clip).astype(np.float32)

class SafeRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._warned = False

    def reward(self, rew):
        if not np.isfinite(rew):
            if not self._warned:
                print("[WARN] Non-finite reward encountered; replacing with 0.0")
                self._warned = True
            return 0.0
        return float(rew)

# ---------------------------- Callbacks ------------------------------------ #

class MetricsLoggerCallback(BaseCallback):
    def __init__(self, out_json_path: str, fold: int, seed: int, verbose: int = 0):
        super().__init__(verbose)
        self.out_json_path = out_json_path
        self.fold = int(fold)
        self.seed = int(seed)
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        elapsed = (time.time() - self.start_time) if self.start_time is not None else float("nan")
        rec = {
            "fold": self.fold,
            "seed": self.seed,
            "elapsed_sec": float(elapsed),
            "rss_mb": float(log_memory_usage_mb()),
        }
        
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
    save_freq: int = 50_000,
    resume: bool = False,
    n_envs: int = 16,
    vec_kind: str = "dummy",
    episode_len: Optional[int] = None,
) -> Dict[str, str]:
    """Train a MaskablePPO policy for discrete actions with fold-specific universe."""
    
    fold_i = int(fold_cfg["fold"])
    seed = int(seed)
    
    print(f"\n[START] Training fold {fold_i} seed {seed}")
    print(f"  Train dates: {fold_cfg['train_start']} to {fold_cfg['train_end']}")
    
    results_dir = os.path.join(out_base, "results", universe)
    model_dir = os.path.join(out_base, "models", universe, "ppo")
    tb_dir = os.path.join(out_base, "tb", universe)
    ensure_dirs(results_dir, model_dir, tb_dir)
    
    tb_dir = os.path.join(tb_dir, f"fold_{fold_i}_seed_{seed}")
    ckpt_dir = os.path.join(model_dir, f"fold_{fold_i}_seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Build training panel with FOLD-SPECIFIC union
    train_aug, union_ids = build_train_panel_with_fold_union(panel, fold_cfg)
    
    # Save union IDs for this fold (for evaluation)
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
    
    # Build environment
    def _make_thunk(rank: int):
        def _init():
            cfg_i = EnvConfig(**{**asdict(env_cfg)})
            if cfg_i.seed is not None:
                cfg_i.seed = int(cfg_i.seed) + int(rank)
            
            if episode_len is not None:
                cfg_i.max_steps = episode_len
            
            e = make_env_from_panel(train_aug, **asdict(cfg_i))
            e = ActionMasker(e, mask_fn)
            e = SafeRewardWrapper(e)
            e = SafeObsWrapper(e, clip=getattr(cfg_i, "obs_clip", 7.5) or 7.5)
            return e
        return _init
    
    thunks = [_make_thunk(i) for i in range(max(1, int(n_envs)))]
    
    if int(n_envs) <= 1 or str(vec_kind).lower() == "dummy":
        vec_env = DummyVecEnv(thunks)
    else:
        vec_env = SubprocVecEnv(thunks, start_method="forkserver")
    
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, 
                           clip_obs=10.0, clip_reward=10.0, gamma=ppo_cfg.gamma)
    
    # Create model
    set_global_seed(seed=seed)
    policy_kwargs = _policy_kwargs_from_cfg(ppo_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    already_trained = 0
    model = None
    
    if resume:
        cp, steps = _latest_checkpoint(ckpt_dir)
        if cp is not None:
            print(f"[INFO] Resuming fold {fold_i} seed {seed} from: {cp} ({steps} steps)")
            try:
                model = MaskablePPO.load(cp, env=vec_env, device="auto")
                already_trained = int(getattr(model, "num_timesteps", steps or 0))
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint; training fresh. Err={e}")
    
    if model is None:
        model = MaskablePPO(
            ppo_cfg.policy,
            vec_env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
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
    
    # Train
    remaining = max(0, int(ppo_cfg.total_timesteps) - int(already_trained))
    callbacks = []
    callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=ckpt_dir, 
                                       name_prefix="ppo_checkpoint"))
    metrics_json = os.path.join(results_dir, "training_metrics.json")
    callbacks.append(MetricsLoggerCallback(out_json_path=metrics_json, fold=fold_i, seed=seed))
    callback = CallbackList(callbacks)
    
    if remaining == 0:
        print(f"[INFO] Nothing left to train (remaining=0). Saving model and exiting.")
    else:
        print(f"[INFO] Training fold {fold_i} seed {seed} for {remaining:,} timesteps "
              f"(n_envs={n_envs}, discrete action space, {len(union_ids)} assets)")
        model.learn(total_timesteps=int(remaining), callback=callback, progress_bar=True)
    
    # Save
    final_model_path = os.path.join(model_dir, f"model_fold_{fold_i}_seed_{seed}.zip")
    model.save(final_model_path)
    
    try:
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

def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO with fold-specific asset unions")
    parser.add_argument("--universe", type=str, choices=["cdi", "infra"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_base", type=str, default=".")
    parser.add_argument("--n_folds", type=int, default=9)
    parser.add_argument("--embargo_days", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip_finished", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--episode_len", type=int, default=256)
    parser.add_argument("--vec", type=str, default="dummy", choices=["dummy","subproc"])
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    universe = args.universe.lower()
    panel = _load_panel_for_universe(universe, args.data_dir)
    
    dates = panel.index.get_level_values("date").unique().sort_values()
    fold_specs = folds_from_dates(dates, n_folds=args.n_folds, embargo_days=args.embargo_days)
    
    # Save fold specs
    results_dir = os.path.join(args.out_base, "results", universe)
    ensure_dirs(results_dir)
    with open(os.path.join(results_dir, "training_folds.json"), "w") as f:
        json.dump(fold_specs, f, indent=2)
    
    # PPO config
    net_arch = config.get('net_arch', [128, 128])
    if isinstance(net_arch, list):
        net_arch = tuple(net_arch)
    
    ppo_cfg = PPOConfig(
        policy=config.get('policy', 'MultiInputPolicy'),
        total_timesteps=config.get('total_timesteps', 100_000),
        learning_rate=config.get('learning_rate', 5e-6),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 256),
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
    
    # Env config
    env_cfg = EnvConfig(
        rebalance_interval=config.get('rebalance_interval', 5),
        max_weight=config.get('max_weight', 0.10),
        weight_blocks=config.get('weight_blocks', 100),
        allow_cash=config.get('allow_cash', True),
        cash_rate_as_rf=config.get('cash_rate_as_rf', True),
        on_inactive=config.get('on_inactive', 'to_cash'),
        transaction_cost_bps=config.get('transaction_cost_bps', 20.0),
        delist_extra_bps=config.get('delist_extra_bps', 20.0),
        normalize_features=config.get('normalize_features', True),
        obs_clip=config.get('obs_clip', 7.5),
        include_prev_weights=config.get('include_prev_weights', False),
        include_active_flag=config.get('include_active_flag', False),
        global_stats=config.get('global_stats', False),
        lambda_turnover=config.get('lambda_turnover', 0.0002),
        lambda_hhi=config.get('lambda_hhi', 0.01),
        lambda_drawdown=config.get('lambda_drawdown', 0.005),
        lambda_tail=config.get('lambda_tail', 0.0),
        tail_window=config.get('tail_window', 60),
        tail_q=config.get('tail_q', 0.05),
        dd_mode=config.get('dd_mode', 'incremental'),
        weight_excess=config.get('weight_excess', 0.0),
        weight_alpha=config.get('weight_alpha', 1.0),
        max_steps=config.get('max_steps', None),
        random_reset_frac=config.get('random_reset_frac', 0.9),
        seed=None,
    )
    
    # Save config
    with open(os.path.join(results_dir, "training_config.json"), "w") as f:
        json.dump({
            "ppo_config": asdict(ppo_cfg),
            "env_config": asdict(env_cfg),
            "seeds": args.seeds,
            "n_folds": args.n_folds,
        }, f, indent=2)
    
    # Train
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    
    for fold in fold_specs:
        for sd in seeds:
            set_global_seed(seed=sd)
            
            if args.skip_finished:
                final_model_path = os.path.join(
                    args.out_base, "models", universe, "ppo", 
                    f"model_fold_{int(fold['fold'])}_seed_{sd}.zip"
                )
                if os.path.exists(final_model_path):
                    print(f"[INFO] Skipping fold {fold['fold']} seed {sd} (exists).")
                    continue
            
            train_one(
                universe, panel, fold_cfg=fold, seed=sd,
                ppo_cfg=ppo_cfg, env_cfg=env_cfg,
                out_base=args.out_base,
                save_freq=config.get('checkpoint_freq', 50_000),
                resume=args.resume,
                n_envs=args.n_envs,
                vec_kind=args.vec,
                episode_len=args.episode_len,
            )

if __name__ == "__main__":
    main()