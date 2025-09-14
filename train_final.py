# train_final.py
"""
Train PPO with DISCRETE action space - fully config-driven version
All parameters are loaded from config.yaml with clear precedence rules
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*get_schedule_fn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*constant_fn.*deprecated.*")

import os, json, time, argparse, random, yaml, re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import gc

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

# SB3 and MaskablePPO
try:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except Exception as e:
    raise RuntimeError("sb3-contrib is required for MaskablePPO. Please install sb3-contrib.") from e

try:
    import gymnasium as gym
except Exception as e:
    raise RuntimeError("gymnasium is required.") from e

# Import optimized environment
try:
    from env_final import EnvConfig, SharedDataEnv, SharedDataProcessor, BatchedEnv
except Exception as e:
    raise RuntimeError("env_final.py with SharedDataEnv is required.") from e

# ------------------------------- Config Classes ----------------------------------- #

@dataclass
class PPOConfig:
    """PPO hyperparameters - all loaded from config"""
    policy: str = "MultiInputPolicy"
    total_timesteps: int = 100_000
    learning_rate: float = 3e-5
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01
    net_arch: List[int] = field(default_factory=lambda: [128, 128])
    ortho_init: bool = True
    activation: str = "tanh"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PPOConfig':
        """Create from dictionary, handling type conversions"""
        d = d.copy()
        # Convert net_arch to list if needed
        if 'net_arch' in d and isinstance(d['net_arch'], (list, tuple)):
            d['net_arch'] = list(d['net_arch'])
        # Handle None values
        for key in ['clip_range_vf', 'target_kl']:
            if key in d and d[key] == 'null':
                d[key] = None
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

@dataclass
class TrainingConfig:
    """Training structure parameters - all from config"""
    n_folds: int = 9
    embargo_days: int = 3
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    n_envs: int = 16
    use_batched: bool = False
    batch_env_size: int = 16
    checkpoint_freq: int = 50_000
    skip_finished: bool = True
    resume_from_checkpoint: bool = True
    vec_type: str = "subproc"  # "dummy" or "subproc"
    device: Optional[str] = None  # None for auto-detect
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary with type handling"""
        d = d.copy()
        # Convert seeds string to list of ints
        if 'seeds' in d:
            if isinstance(d['seeds'], str):
                d['seeds'] = [int(s.strip()) for s in d['seeds'].split(',') if s.strip()]
            elif isinstance(d['seeds'], list):
                d['seeds'] = [int(s) for s in d['seeds']]
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

@dataclass
class ValidationConfig:
    """Parameter validation config - all from config"""
    do_validation: bool = False
    validation_timesteps: int = 2000
    validation_split: float = 0.8
    selection_metric: str = "sharpe"
    lambda_grid: Optional[Dict[str, List[float]]] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ValidationConfig':
        """Create from dictionary"""
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})

@dataclass
class MasterConfig:
    """Master configuration container"""
    ppo: PPOConfig
    env: EnvConfig
    training: TrainingConfig
    validation: ValidationConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'MasterConfig':
        """Load all configs from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Load PPO config
        ppo_data = {}
        for key in PPOConfig.__annotations__:
            if key in data:
                ppo_data[key] = data[key]
        ppo = PPOConfig.from_dict(ppo_data)
        
        # Load environment config
        env_data = {}
        for key in EnvConfig.__annotations__:
            if key in data:
                env_data[key] = data[key]
        env = EnvConfig(**env_data)
        
        # Load training config
        training_data = {}
        for key in TrainingConfig.__annotations__:
            if key in data:
                training_data[key] = data[key]
        training = TrainingConfig.from_dict(training_data)
        
        # Load validation config
        val_section = data.get('validation', {})
        if isinstance(val_section, dict):
            val_data = val_section
        else:
            val_data = {}
        
        # Add top-level validation params
        for key in ['do_validation', 'selection_metric', 'lambda_grid']:
            if key in data:
                val_data[key] = data[key]
        
        validation = ValidationConfig.from_dict(val_data)
        
        return cls(ppo=ppo, env=env, training=training, validation=validation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving"""
        return {
            'ppo_config': asdict(self.ppo),
            'env_config': asdict(self.env),
            'training_config': asdict(self.training),
            'validation_config': asdict(self.validation),
        }

# -------------------------- Required Panel Columns ------------------------- #

REQUIRED_COLS = [
    "return", "spread", "duration", "time_to_maturity", "sector_id", "active",
    "risk_free", "index_return", "index_level",
]

# ------------------------------ Helper Utils ------------------------------- #

def _policy_kwargs_from_cfg(ppo_cfg: PPOConfig) -> dict:
    act_fn = nn.Tanh if str(ppo_cfg.activation).lower() == "tanh" else nn.ReLU
    return dict(
        net_arch=list(ppo_cfg.net_arch),
        activation_fn=act_fn,
        ortho_init=ppo_cfg.ortho_init
    )

def set_global_seed(seed: int = 0, torch_threads: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

def folds_from_dates(dates: pd.DatetimeIndex, config: TrainingConfig) -> List[Dict[str, str]]:
    """Build expanding window folds using config parameters"""
    dates = pd.to_datetime(pd.Index(dates).unique().sort_values())
    T = len(dates)
    n_folds = config.n_folds
    embargo_days = config.embargo_days
    
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

def build_train_panel_with_union(panel: pd.DataFrame, fold_cfg: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """Extract training window and union of asset IDs."""
    train_start = pd.to_datetime(fold_cfg["train_start"])
    train_end = pd.to_datetime(fold_cfg["train_end"])
    test_end = pd.to_datetime(fold_cfg["test_end"])

    # Get union of all assets that appear in train+test period
    union_slice = panel.loc[
        (panel.index.get_level_values("date") >= train_start) &
        (panel.index.get_level_values("date") <= test_end)
    ]
    union_ids = union_slice.index.get_level_values("debenture_id").unique().tolist()

    # Get training dates
    train_dates = panel.index.get_level_values("date").unique().sort_values()
    train_dates = train_dates[(train_dates >= train_start) & (train_dates <= train_end)]

    # Reindex to complete panel
    idx = pd.MultiIndex.from_product([train_dates, union_ids], names=["date", "debenture_id"])
    train_aug = panel.reindex(idx)

    # Basic fills
    train_aug["active"] = train_aug["active"].fillna(0).astype(np.int8)
    for c in ["return", "spread", "duration", "time_to_maturity"]:
        if c in train_aug.columns:
            train_aug[c] = train_aug[c].astype(float).fillna(0.0)
    if "sector_id" in train_aug.columns:
        train_aug["sector_id"] = train_aug["sector_id"].fillna(-1).astype(np.int16)

    return train_aug.sort_index(), union_ids

# ---------------------- Wrappers ---------------------- #

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
        self.last_log_time = None
        self.log_interval = 10  # Log every 10 seconds

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def _on_step(self) -> bool:
        # Log progress periodically
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            elapsed = current_time - self.start_time
            progress = self.num_timesteps / self.locals.get('total_timesteps', 1)
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            
            print(f"  Progress: {progress:.1%} | Steps: {self.num_timesteps:,} | "
                  f"FPS: {fps:.0f} | Elapsed: {elapsed:.1f}s | "
                  f"Memory: {log_memory_usage_mb():.0f}MB")
            
            self.last_log_time = current_time
        
        return True

    def _on_training_end(self) -> None:
        elapsed = (time.time() - self.start_time) if self.start_time is not None else float("nan")
        rec = {
            "fold": self.fold,
            "seed": self.seed,
            "elapsed_sec": float(elapsed),
            "fps": self.num_timesteps / elapsed if elapsed > 0 else 0,
            "total_timesteps": self.num_timesteps,
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

# ------------------------ Parameter Validation ------------------------ #

def validate_reward_params_fast(
    train_aug: pd.DataFrame,
    env_cfg: EnvConfig,
    val_cfg: ValidationConfig,
    ppo_cfg: PPOConfig,
    seed: int = 0,
) -> Dict[str, float]:
    """Fast grid-search using config parameters"""
    import dataclasses
    
    if not val_cfg.do_validation or not val_cfg.lambda_grid:
        return {
            "lambda_turnover": float(env_cfg.lambda_turnover),
            "lambda_hhi": float(env_cfg.lambda_hhi),
            "lambda_drawdown": float(env_cfg.lambda_drawdown),
        }
    
    TRADING_DAYS = 252
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Split data using config split ratio
    dates = train_aug.index.get_level_values("date").unique().sort_values()
    if dates.size < 10:
        return {
            "lambda_turnover": float(env_cfg.lambda_turnover),
            "lambda_hhi": float(env_cfg.lambda_hhi),
            "lambda_drawdown": float(env_cfg.lambda_drawdown),
        }
    
    cut = int(dates.size * val_cfg.validation_split)
    train_slice = train_aug[train_aug.index.get_level_values("date") <= dates[cut]]
    val_slice = train_aug[train_aug.index.get_level_values("date") > dates[cut]]
    
    if train_slice.empty or val_slice.empty:
        return {
            "lambda_turnover": float(env_cfg.lambda_turnover),
            "lambda_hhi": float(env_cfg.lambda_hhi),
            "lambda_drawdown": float(env_cfg.lambda_drawdown),
        }
    
    # Preprocess data once
    print("  Preprocessing validation data...")
    train_data = SharedDataProcessor.process_panel(train_slice, env_cfg)
    val_data = SharedDataProcessor.process_panel(val_slice, env_cfg)
    
    # Grid search from config
    grid_to = val_cfg.lambda_grid.get("lambda_turnover", [env_cfg.lambda_turnover])
    grid_hhi = val_cfg.lambda_grid.get("lambda_hhi", [env_cfg.lambda_hhi])
    grid_dd = val_cfg.lambda_grid.get("lambda_drawdown", [env_cfg.lambda_drawdown])
    
    best_score = -np.inf
    best = (float(env_cfg.lambda_turnover), float(env_cfg.lambda_hhi), float(env_cfg.lambda_drawdown))
    
    total_combos = len(grid_to) * len(grid_hhi) * len(grid_dd)
    combo_idx = 0
    
    for lam_to in grid_to:
        for lam_hhi in grid_hhi:
            for lam_dd in grid_dd:
                combo_idx += 1
                print(f"    Testing combo {combo_idx}/{total_combos}: λ_to={lam_to:.4f}, λ_hhi={lam_hhi:.4f}, λ_dd={lam_dd:.4f}")
                
                cfg_i = dataclasses.replace(
                    env_cfg,
                    lambda_turnover=float(lam_to),
                    lambda_hhi=float(lam_hhi),
                    lambda_drawdown=float(lam_dd),
                    seed=int(seed),
                )
                
                # Train with config timesteps
                SharedDataEnv.set_shared_data(train_data)
                
                def _make_train():
                    e = SharedDataEnv(cfg_i)
                    e = ActionMasker(e, mask_fn)
                    e = SafeRewardWrapper(e)
                    return e
                
                venv_tr = DummyVecEnv([_make_train])
                venv_tr = VecNormalize(venv_tr, norm_obs=True, norm_reward=True,
                                      clip_obs=10.0, clip_reward=10.0, gamma=ppo_cfg.gamma)
                
                # Quick training
                model = MaskablePPO(
                    "MultiInputPolicy",
                    venv_tr,
                    learning_rate=1e-4,
                    n_steps=min(64, len(train_slice) // 2),
                    batch_size=32,
                    n_epochs=2,
                    gamma=ppo_cfg.gamma,
                    gae_lambda=ppo_cfg.gae_lambda,
                    clip_range=0.2,
                    ent_coef=0.001,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    tensorboard_log=None,
                    policy_kwargs={"net_arch": [64, 64]},
                    verbose=0,
                    device=device,
                )
                
                model.learn(total_timesteps=val_cfg.validation_timesteps, progress_bar=False)
                
                # Validate
                SharedDataEnv.set_shared_data(val_data)
                
                def _make_val():
                    e = SharedDataEnv(cfg_i)
                    e = ActionMasker(e, mask_fn)
                    e = SafeRewardWrapper(e)
                    return e
                
                venv_va = DummyVecEnv([_make_val])
                venv_va = VecNormalize(venv_va, norm_obs=True, norm_reward=False,
                                      clip_obs=10.0, gamma=ppo_cfg.gamma)
                venv_va.training = False
                venv_va.norm_reward = False
                
                if hasattr(venv_tr, 'obs_rms'):
                    venv_va.obs_rms = venv_tr.obs_rms
                
                # Evaluate using config metric
                obs = venv_va.reset()
                returns = []
                for _ in range(len(val_data['dates'])):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, dones, infos = venv_va.step(action)
                    returns.append(infos[0].get('portfolio_return', 0.0))
                    if dones[0]:
                        break
                
                # Calculate score based on config metric
                if returns and len(returns) > 1:
                    r = np.array(returns)
                    if val_cfg.selection_metric == "sharpe":
                        score = (r.mean() / (r.std() + 1e-8)) * np.sqrt(TRADING_DAYS)
                    elif val_cfg.selection_metric == "ir":
                        score = r.mean() * TRADING_DAYS
                    else:
                        score = r.mean()
                    
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best = (float(lam_to), float(lam_hhi), float(lam_dd))
                        print(f"      New best {val_cfg.selection_metric}: {score:.4f}")
                
                # Cleanup
                venv_tr.close()
                venv_va.close()
                del model
                gc.collect()
    
    return {"lambda_turnover": best[0], "lambda_hhi": best[1], "lambda_drawdown": best[2]}

# ------------------------------ Train One ---------------------------------- #

def train_one(
    universe: str,
    panel: pd.DataFrame,
    fold_cfg: Dict[str, str],
    seed: int,
    config: MasterConfig,
    out_base: str,
) -> Dict[str, str]:
    """Train a single fold/seed using config parameters"""
    
    fold_i = int(fold_cfg["fold"])
    seed = int(seed)
    
    print(f"\n[START] Training fold {fold_i} seed {seed}")
    print(f"  Train dates: {fold_cfg['train_start']} to {fold_cfg['train_end']}")
    print(f"  Memory usage: {log_memory_usage_mb():.0f}MB")
    
    # Setup directories
    results_dir = os.path.join(out_base, "results", universe)
    model_dir = os.path.join(out_base, "models", universe, "ppo")
    tb_dir = os.path.join(out_base, "tb", universe)
    ensure_dirs(results_dir, model_dir, tb_dir)
    
    tb_dir = os.path.join(tb_dir, f"fold_{fold_i}_seed_{seed}")
    ckpt_dir = os.path.join(model_dir, f"fold_{fold_i}_seed_{seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Build training panel
    train_aug, union_ids = build_train_panel_with_union(panel, fold_cfg)
    print(f"  Union size: {len(union_ids)} assets")
    print(f"  Training samples: {len(train_aug.index.get_level_values('date').unique())} days")
    
    # Save union IDs
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
    
    # Validate lambda parameters if configured
    env_cfg = config.env
    if config.validation.do_validation and config.validation.lambda_grid:
        print(f"[INFO] Parameter validation ({config.validation.selection_metric})...")
        best = validate_reward_params_fast(
            train_aug, env_cfg, config.validation, config.ppo, seed
        )
        env_cfg = EnvConfig(**{**asdict(env_cfg), **best})
        print(f"[INFO] Selected λ values: {best}")
    
    # Preprocess data once
    print(f"[INFO] Preprocessing panel data (once for all {config.training.n_envs} environments)...")
    start_time = time.time()
    shared_data = SharedDataProcessor.process_panel(train_aug, env_cfg)
    SharedDataEnv.set_shared_data(shared_data)
    print(f"  Data preprocessing took {time.time() - start_time:.2f}s")
    print(f"  Memory after preprocessing: {log_memory_usage_mb():.0f}MB")
    
    # Create environment factory
    def _make_env(rank: int):
        def _init():
            cfg_i = EnvConfig(**asdict(env_cfg))
            cfg_i.seed = int(seed) + int(rank)
            
            # Create environment based on config
            if config.training.use_batched:
                from env_final import BatchedEnv
                e = SharedDataEnv(cfg_i)
                e = BatchedEnv(e, batch_size=config.training.batch_env_size)
            else:
                e = SharedDataEnv(cfg_i)
            
            e = ActionMasker(e, mask_fn)
            e = SafeRewardWrapper(e)
            return e
        return _init
    
    # Create vectorized environment based on config
    print(f"[INFO] Creating {config.training.n_envs} parallel environments...")
    thunks = [_make_env(i) for i in range(config.training.n_envs)]
    
    # Use vec_type from config
    if config.training.n_envs > 1 and config.training.vec_type == "subproc":
        try:
            vec_env = SubprocVecEnv(thunks, start_method="fork")
            print(f"  Using SubprocVecEnv with {config.training.n_envs} processes")
        except Exception as e:
            print(f"  SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
            vec_env = DummyVecEnv(thunks)
    else:
        vec_env = DummyVecEnv(thunks)
    
    vec_env = VecCheckNan(vec_env, raise_exception=True, warn_once=True, check_inf=True)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config.ppo.gamma
    )
    
    # Determine device from config
    if config.training.device:
        device = config.training.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # Check for checkpoint if resume is enabled
    already_trained = 0
    model = None
    
    if config.training.resume_from_checkpoint:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.zip')]
        if ckpt_files:
            ckpt_files.sort(key=lambda x: int(re.findall(r'(\d+)', x)[0]) if re.findall(r'(\d+)', x) else 0)
            latest_ckpt = os.path.join(ckpt_dir, ckpt_files[-1])
            print(f"[INFO] Resuming from checkpoint: {latest_ckpt}")
            
            try:
                model = MaskablePPO.load(latest_ckpt, env=vec_env, device=device)
                already_trained = model.num_timesteps
                print(f"  Resumed from {already_trained:,} timesteps")
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint: {e}")
    
    if model is None:
        # Create new model with config parameters
        adjusted_batch_size = min(
            config.ppo.batch_size * config.training.n_envs,
            config.ppo.n_steps * config.training.n_envs
        )
        
        print(f"[INFO] Creating new model")
        print(f"  Policy: {config.ppo.policy}")
        print(f"  Network: {config.ppo.net_arch}")
        print(f"  Learning rate: {config.ppo.learning_rate}")
        print(f"  n_steps: {config.ppo.n_steps} × {config.training.n_envs} envs")
        print(f"  batch_size: {adjusted_batch_size}")
        
        model = MaskablePPO(
            config.ppo.policy,
            vec_env,
            learning_rate=config.ppo.learning_rate,
            n_steps=config.ppo.n_steps,
            batch_size=adjusted_batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            clip_range_vf=config.ppo.clip_range_vf or config.ppo.clip_range,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            target_kl=config.ppo.target_kl,
            tensorboard_log=tb_dir,
            policy_kwargs=_policy_kwargs_from_cfg(config.ppo),
            verbose=1,
            device=device,
        )
    
    # Train
    remaining = max(0, config.ppo.total_timesteps - already_trained)
    
    if remaining == 0:
        print(f"[INFO] Training complete (already at {already_trained:,} timesteps)")
    else:
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback with config frequency
        callbacks.append(
            CheckpointCallback(
                save_freq=max(config.training.checkpoint_freq // config.training.n_envs, 100),
                save_path=ckpt_dir,
                name_prefix="ppo_checkpoint",
                save_replay_buffer=False,
                save_vecnormalize=True,
            )
        )
        
        # Metrics logger
        metrics_json = os.path.join(results_dir, "training_metrics.json")
        callbacks.append(MetricsLoggerCallback(out_json_path=metrics_json, fold=fold_i, seed=seed))
        
        callback = CallbackList(callbacks)
        
        print(f"[INFO] Training for {remaining:,} timesteps")
        print(f"  Approximately {remaining // (config.training.n_envs * config.ppo.n_steps)} PPO updates")
        
        start_train = time.time()
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=False,
        )
        
        train_time = time.time() - start_train
        fps = remaining / train_time if train_time > 0 else 0
        print(f"[INFO] Training complete in {train_time:.1f}s ({fps:.0f} FPS)")
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"model_fold_{fold_i}_seed_{seed}.zip")
    model.save(final_model_path)
    print(f"[INFO] Model saved to {final_model_path}")
    
    # Save VecNormalize
    vecnorm_path = os.path.join(model_dir, f"vecnorm_fold_{fold_i}_seed_{seed}.pkl")
    vec_env.save(vecnorm_path)
    print(f"[INFO] VecNormalize saved to {vecnorm_path}")
    
    # Cleanup
    vec_env.close()
    del model
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[DONE] Fold {fold_i} seed {seed} | Final memory: {log_memory_usage_mb():.0f}MB")
    
    return {
        "model_path": final_model_path,
        "vecnorm_path": vecnorm_path,
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
    parser = argparse.ArgumentParser(description="Train MaskablePPO - config-driven version")
    parser.add_argument("--universe", type=str, choices=["cdi", "infra"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_base", type=str, default=".")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    # Command-line overrides (optional - will override config values)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--skip_finished", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    
    args = parser.parse_args()
    
    # Load master config from YAML
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    print(f"\n{'='*60}")
    print(f"Loading configuration from: {args.config}")
    print(f"{'='*60}\n")
    
    config = MasterConfig.from_yaml(args.config)
    
    # Apply command-line overrides if provided
    if args.n_folds is not None:
        config.training.n_folds = args.n_folds
        print(f"[OVERRIDE] n_folds = {args.n_folds}")
    
    if args.seeds is not None:
        config.training.seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        print(f"[OVERRIDE] seeds = {config.training.seeds}")
    
    if args.n_envs is not None:
        config.training.n_envs = args.n_envs
        print(f"[OVERRIDE] n_envs = {args.n_envs}")
    
    if args.skip_finished is not None:
        config.training.skip_finished = args.skip_finished
        print(f"[OVERRIDE] skip_finished = {args.skip_finished}")
    
    if args.resume is not None:
        config.training.resume_from_checkpoint = args.resume
        print(f"[OVERRIDE] resume = {args.resume}")
    
    # Load universe data
    universe = args.universe.lower()
    panel = _load_panel_for_universe(universe, args.data_dir)
    
    # Generate folds using config
    dates = panel.index.get_level_values("date").unique().sort_values()
    fold_specs = folds_from_dates(dates, config.training)
    
    # Save fold specs and full config
    results_dir = os.path.join(args.out_base, "results", universe)
    ensure_dirs(results_dir)
    
    with open(os.path.join(results_dir, "training_folds.json"), "w") as f:
        json.dump(fold_specs, f, indent=2)
    
    with open(os.path.join(results_dir, "training_config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print(f"Training Configuration Summary")
    print(f"{'='*60}")
    print(f"Universe: {universe}")
    print(f"Folds: {config.training.n_folds}")
    print(f"Seeds: {config.training.seeds}")
    print(f"Parallel environments: {config.training.n_envs}")
    print(f"Total timesteps: {config.ppo.total_timesteps:,}")
    print(f"Learning rate: {config.ppo.learning_rate}")
    print(f"Network architecture: {config.ppo.net_arch}")
    print(f"Validation: {'Enabled' if config.validation.do_validation else 'Disabled'}")
    print(f"Device: {config.training.device or 'auto-detect'}")
    print(f"{'='*60}\n")
    
    # Train each fold/seed combination
    for fold in fold_specs:
        for seed in config.training.seeds:
            set_global_seed(seed=seed)
            
            if config.training.skip_finished:
                final_model_path = os.path.join(
                    args.out_base, "models", universe, "ppo",
                    f"model_fold_{int(fold['fold'])}_seed_{seed}.zip"
                )
                if os.path.exists(final_model_path):
                    print(f"[SKIP] Fold {fold['fold']} seed {seed} already exists")
                    continue
            
            train_one(
                universe=universe,
                panel=panel,
                fold_cfg=fold,
                seed=seed,
                config=config,
                out_base=args.out_base,
            )
            
            # Force garbage collection between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Config used: {args.config}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()