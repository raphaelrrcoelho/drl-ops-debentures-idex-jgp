# train_final_improved.py
"""
Improved PPO Training Script with Proper Configuration
======================================================
IMPROVEMENTS OVER ORIGINAL:
1. âœ… Simplified, cleaner code structure
2. âœ… Proper support for improved environment (lagged arrays)
3. âœ… Better memory management
4. âœ… Clearer logging and progress tracking
5. âœ… Proper seed management for reproducibility
6. âœ… Streamlined callbacks
7. âœ… Better error handling and validation
8. âœ… Removed unnecessary progressive/curriculum complexity
9. âœ… Optimized for the simplified configuration

Compatible with:
- env_final_improved.py (with lagged arrays)
- config_simplified.yaml (optimized hyperparameters)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*get_schedule_fn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*constant_fn.*deprecated.*")

import os
import json
import time
import argparse
import random
import yaml
import gc
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure reproducibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# PyTorch
try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("PyTorch is required. Install: pip install torch") from e

# Stable Baselines 3
try:
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except Exception as e:
    raise RuntimeError("sb3-contrib required. Install: pip install sb3-contrib") from e

try:
    import gymnasium as gym
except Exception as e:
    raise RuntimeError("gymnasium required. Install: pip install gymnasium") from e

# Import environment
try:
    from env_final import DebentureTradingEnv, EnvConfig
except Exception as e:
    raise RuntimeError("env_final.py (improved version) required.") from e

# ================================ CONFIGS ================================ #

@dataclass
class PPOConfig:
    """PPO hyperparameters"""
    policy: str = "MultiInputPolicy"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01
    net_arch: tuple = (128, 64)
    ortho_init: bool = True
    activation: str = "tanh"


# ================================ UTILITIES ================================ #

def set_global_seed(seed: int):
    """Set seed for all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*paths):
    """Create directories if they don't exist"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_memory_usage():
    """Get current memory usage in GB"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    return 0.0


def log_memory(prefix: str = ""):
    """Log current memory usage"""
    if HAS_PSUTIL:
        mem_gb = get_memory_usage()
        print(f"[MEMORY] {prefix}: {mem_gb:.2f} GB")


# ================================ DATA LOADING ================================ #

def load_panel(universe: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Load processed panel data for universe.
    
    Args:
        universe: 'cdi' or 'infra'
        data_dir: Data directory
    
    Returns:
        Panel DataFrame with multi-index (date, debenture_id)
    """
    pkl_path = os.path.join(data_dir, f"{universe}_processed.pkl")
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Processed data not found: {pkl_path}\n"
            f"Please run data_final.py first to generate processed data."
        )
    
    print(f"[DATA] Loading panel from {pkl_path}")
    panel = pd.read_pickle(pkl_path)
    
    # Validate required columns
    required = [
        "return", "spread", "duration", "sector_id", "active",
        "risk_free", "index_return", "index_weight",
    ]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"[DATA] Loaded panel: {len(panel)} rows, {panel.index.get_level_values('date').nunique()} dates")
    log_memory("After loading panel")
    
    return panel


# ================================ FOLD SPLITTING ================================ #

def create_walk_forward_folds(
    dates: pd.DatetimeIndex,
    n_folds: int = 9,
    embargo_days: int = 3,
) -> List[Dict[str, Any]]:
    """
    Create walk-forward validation folds.
    
    Args:
        dates: Sorted array of unique dates
        n_folds: Number of folds
        embargo_days: Days to skip between train/test
    
    Returns:
        List of fold specifications
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    n = len(dates)
    fold_size = n // (n_folds + 1)  # +1 for initial training period
    
    folds = []
    for i in range(n_folds):
        train_end_idx = (i + 1) * fold_size
        test_start_idx = min(train_end_idx + embargo_days, n - 1)
        test_end_idx = min(test_start_idx + fold_size, n)
        
        if test_end_idx <= test_start_idx:
            break
        
        fold_spec = {
            "fold": i,
            "train_start": dates[0].isoformat(),
            "train_end": dates[train_end_idx].isoformat(),
            "test_start": dates[test_start_idx].isoformat(),
            "test_end": dates[test_end_idx - 1].isoformat(),
            "train_days": train_end_idx,
            "test_days": test_end_idx - test_start_idx,
        }
        folds.append(fold_spec)
    
    return folds


# ================================ ENVIRONMENT CREATION ================================ #

def make_env(panel_subset: pd.DataFrame, cfg: EnvConfig, seed: int = 0):
    """Create a single environment instance"""
    def _init():
        env = DebentureTradingEnv(panel_subset, cfg)
        env.reset(seed=seed)
        
        # Wrap with action masker
        def mask_fn(env):
            return env.unwrapped._get_observation(env.unwrapped.t)[1]
        
        return ActionMasker(env, mask_fn)
    
    return _init


def make_vec_env(
    panel_subset: pd.DataFrame,
    env_cfg: EnvConfig,
    n_envs: int = 8,
    seed: int = 0,
    vec_type: str = "dummy",
):
    """
    Create vectorized environment.
    
    Args:
        panel_subset: Training data
        env_cfg: Environment configuration
        n_envs: Number of parallel environments
        seed: Base random seed
        vec_type: 'dummy' or 'subproc'
    
    Returns:
        Vectorized environment
    """
    print(f"[ENV] Creating {n_envs} parallel environments ({vec_type})")
    
    # Create environment factories
    env_fns = [
        make_env(panel_subset, env_cfg, seed + i)
        for i in range(n_envs)
    ]
    
    # Create vectorized environment
    if vec_type == "subproc":
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    
    log_memory("After creating environments")
    
    return vec_env


# ================================ CALLBACKS ================================ #

class ProgressCallback(BaseCallback):
    """Callback for logging training progress"""
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.start_time = None
        self.last_log_time = None
        self.log_interval = 60  # Log every 60 seconds
    
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        print("\n[TRAIN] Training started")
    
    def _on_step(self):
        # Log progress periodically
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            elapsed = current_time - self.start_time
            progress = self.num_timesteps / self.model.num_timesteps
            
            print(f"[TRAIN] Progress: {progress*100:.1f}% | "
                  f"Steps: {self.num_timesteps:,} / {self.model.num_timesteps:,} | "
                  f"Elapsed: {elapsed/60:.1f}m")
            
            log_memory("During training")
            self.last_log_time = current_time
        
        return True
    
    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        print(f"\n[TRAIN] Training completed in {elapsed/60:.1f} minutes")
        log_memory("After training")


# ================================ TRAINING ================================ #

def train_fold(
    universe: str,
    panel: pd.DataFrame,
    fold_spec: Dict[str, Any],
    seed: int,
    ppo_cfg: PPOConfig,
    env_cfg: EnvConfig,
    output_dir: str,
    n_envs: int = 8,
    vec_type: str = "dummy",
    checkpoint_freq: int = 100000,
    resume: bool = False,
) -> str:
    """
    Train PPO agent on a single fold.
    
    Args:
        universe: Universe name
        panel: Full panel data
        fold_spec: Fold specification
        seed: Random seed
        ppo_cfg: PPO hyperparameters
        env_cfg: Environment configuration
        output_dir: Output directory
        n_envs: Number of parallel environments
        vec_type: Vectorization type
        checkpoint_freq: Checkpoint frequency
        resume: Resume from checkpoint if available
    
    Returns:
        Path to saved model
    """
    fold_id = fold_spec["fold"]
    
    print("\n" + "="*70)
    print(f"TRAINING FOLD {fold_id} | SEED {seed}")
    print("="*70)
    print(f"Train period: {fold_spec['train_start']} to {fold_spec['train_end']}")
    print(f"Train days: {fold_spec['train_days']}")
    print(f"Config: {ppo_cfg.total_timesteps:,} timesteps, LR={ppo_cfg.learning_rate}")
    
    # Set seed
    set_global_seed(seed)
    
    # Extract training data
    train_start = pd.Timestamp(fold_spec["train_start"])
    train_end = pd.Timestamp(fold_spec["train_end"])
    
    train_panel = panel[
        (panel.index.get_level_values("date") >= train_start) &
        (panel.index.get_level_values("date") <= train_end)
    ].copy()
    
    print(f"[DATA] Training panel: {len(train_panel)} rows")
    
    # Create vectorized environment
    vec_env = make_vec_env(
        train_panel,
        env_cfg,
        n_envs=n_envs,
        seed=seed,
        vec_type=vec_type,
    )
    
    # Setup model directories
    model_dir = os.path.join(output_dir, "models", universe, "ppo")
    checkpoint_dir = os.path.join(model_dir, "checkpoints", f"fold_{fold_id}_seed_{seed}")
    ensure_dirs(model_dir, checkpoint_dir)
    
    # Check for existing checkpoint
    model_path = os.path.join(model_dir, f"model_fold_{fold_id}_seed_{seed}.zip")
    
    if resume and os.path.exists(model_path):
        print(f"[MODEL] Resuming from {model_path}")
        model = MaskablePPO.load(model_path, env=vec_env)
    else:
        # Create new model
        print(f"[MODEL] Creating new MaskablePPO model")
        print(f"  Policy: {ppo_cfg.policy}")
        print(f"  Network: {ppo_cfg.net_arch}")
        print(f"  Learning rate: {ppo_cfg.learning_rate}")
        print(f"  Batch size: {ppo_cfg.batch_size}")
        
        policy_kwargs = {
            "net_arch": dict(pi=list(ppo_cfg.net_arch), vf=list(ppo_cfg.net_arch)),
            "activation_fn": nn.Tanh if ppo_cfg.activation == "tanh" else nn.ReLU,
            "ortho_init": ppo_cfg.ortho_init,
        }
        
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
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
        )
    
    log_memory("After model creation")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"ppo_fold{fold_id}_seed{seed}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Progress callback
    progress_callback = ProgressCallback(verbose=1)
    callbacks.append(progress_callback)
    
    # Train
    print(f"\n[TRAIN] Starting training for {ppo_cfg.total_timesteps:,} timesteps")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=ppo_cfg.total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        raise
    
    elapsed = time.time() - start_time
    print(f"\n[TRAIN] Training completed in {elapsed/60:.1f} minutes")
    print(f"  Average: {elapsed/ppo_cfg.total_timesteps*1000:.2f} ms/step")
    
    # Save final model
    print(f"[MODEL] Saving to {model_path}")
    model.save(model_path)
    
    # Cleanup
    vec_env.close()
    del model, vec_env
    gc.collect()
    
    log_memory("After cleanup")
    
    return model_path


# ================================ MAIN ================================ #

def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO with improved environment and configuration"
    )
    parser.add_argument("--universe", type=str, choices=["cdi", "infra"], required=True,
                       help="Universe to train on")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default=".",
                       help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Configuration file (YAML)")
    parser.add_argument("--n_folds", type=int, default=9,
                       help="Number of walk-forward folds")
    parser.add_argument("--embargo_days", type=int, default=3,
                       help="Embargo days between train/test")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                       help="Comma-separated random seeds")
    parser.add_argument("--skip_finished", action="store_true",
                       help="Skip already trained folds")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoints if available")
    parser.add_argument("--n_envs", type=int, default=None,
                       help="Number of parallel environments (override config)")
    parser.add_argument("--vec_type", type=str, default=None, choices=["dummy", "subproc"],
                       help="Vectorization type (override config)")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("IMPROVED PPO TRAINING PIPELINE")
    print("="*70)
    print(f"Universe: {args.universe.upper()}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {args.output_dir}")
    
    # Load configuration
    print(f"\n[CONFIG] Loading from {args.config}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse network architecture
    net_arch = config.get('net_arch', [128, 64])
    if isinstance(net_arch, list):
        net_arch = tuple(net_arch)
    
    # Create PPO configuration
    ppo_cfg = PPOConfig(
        policy=config.get('policy', 'MultiInputPolicy'),
        total_timesteps=config.get('total_timesteps', 1_000_000),
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 512),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        clip_range_vf=config.get('clip_range_vf', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        target_kl=config.get('target_kl', 0.01),
        net_arch=net_arch,
        ortho_init=config.get('ortho_init', True),
        activation=config.get('activation', 'tanh'),
    )
    
    # Create environment configuration
    env_cfg = EnvConfig(
        rebalance_interval=config.get('rebalance_interval', 21),
        max_weight=config.get('max_weight', 0.10),
        weight_blocks=config.get('weight_blocks', 50),
        max_assets=config.get('max_assets', 50),
        allow_cash=config.get('allow_cash', True),
        cash_rate_as_rf=config.get('cash_rate_as_rf', True),
        on_inactive=config.get('on_inactive', 'to_cash'),
        transaction_cost_bps=config.get('transaction_cost_bps', 10.0),
        delist_extra_bps=config.get('delist_extra_bps', 10.0),
        normalize_features=config.get('normalize_features', True),
        obs_clip=config.get('obs_clip', 5.0),
        include_prev_weights=config.get('include_prev_weights', True),
        include_active_flag=config.get('include_active_flag', True),
        global_stats=config.get('global_stats', True),
        weight_alpha=config.get('weight_alpha', 1.0),
        lambda_turnover=config.get('lambda_turnover', 0.01),
        lambda_hhi=config.get('lambda_hhi', 0.0),
        lambda_drawdown=config.get('lambda_drawdown', 0.0),
        lambda_tail=config.get('lambda_tail', 0.0),
        tail_window=config.get('tail_window', 60),
        tail_q=config.get('tail_q', 0.05),
        dd_mode=config.get('dd_mode', 'incremental'),
        max_steps=config.get('max_steps', 252),
        random_reset_frac=config.get('random_reset_frac', 0.9),
        # Feature selection
        use_momentum_features=config.get('use_momentum_features', True),
        use_volatility_features=config.get('use_volatility_features', True),
        use_relative_value_features=config.get('use_relative_value_features', True),
        use_duration_features=config.get('use_duration_features', True),
        use_microstructure_features=config.get('use_microstructure_features', True),
        use_carry_features=config.get('use_carry_features', True),
        use_spread_dynamics=config.get('use_spread_dynamics', True),
        use_risk_adjusted_features=config.get('use_risk_adjusted_features', False),
        use_sector_curves=config.get('use_sector_curves', False),
        use_zscore_features=config.get('use_zscore_features', False),
        use_rolling_zscores=config.get('use_rolling_zscores', False),
    )
    
    # Print configuration summary
    print("\n[CONFIG] PPO Configuration:")
    print(f"  Total timesteps: {ppo_cfg.total_timesteps:,}")
    print(f"  Learning rate: {ppo_cfg.learning_rate}")
    print(f"  Network architecture: {ppo_cfg.net_arch}")
    print(f"  Batch size: {ppo_cfg.batch_size}")
    print(f"  n_steps: {ppo_cfg.n_steps}")
    
    print("\n[CONFIG] Environment Configuration:")
    print(f"  Max assets: {env_cfg.max_assets}")
    print(f"  Rebalance interval: {env_cfg.rebalance_interval} days")
    print(f"  Transaction costs: {env_cfg.transaction_cost_bps} bps")
    print(f"  Turnover penalty: {env_cfg.lambda_turnover}")
    
    # Load data
    print(f"\n[DATA] Loading {args.universe} universe")
    panel = load_panel(args.universe, args.data_dir)
    
    # Create folds
    print(f"\n[FOLDS] Creating {args.n_folds} walk-forward folds")
    dates = panel.index.get_level_values("date").unique().sort_values()
    fold_specs = create_walk_forward_folds(
        dates,
        n_folds=args.n_folds,
        embargo_days=args.embargo_days,
    )
    
    print(f"[FOLDS] Created {len(fold_specs)} folds")
    for i, fold in enumerate(fold_specs):
        print(f"  Fold {i}: {fold['train_days']} train days, {fold['test_days']} test days")
    
    # Save fold specifications
    results_dir = os.path.join(args.output_dir, "results", args.universe)
    ensure_dirs(results_dir)
    
    fold_spec_path = os.path.join(results_dir, "training_folds.json")
    with open(fold_spec_path, "w") as f:
        json.dump(fold_specs, f, indent=2, default=str)
    print(f"[FOLDS] Saved fold specifications to {fold_spec_path}")
    
    # Save configuration
    config_path = os.path.join(results_dir, "training_config.json")
    with open(config_path, "w") as f:
        training_config = {
            "ppo_config": asdict(ppo_cfg),
            "env_config": asdict(env_cfg),
            "n_folds": args.n_folds,
            "embargo_days": args.embargo_days,
            "seeds": args.seeds,
        }
        json.dump(training_config, f, indent=2)
    print(f"[CONFIG] Saved training configuration to {config_path}")
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    # Get training parameters
    n_envs = args.n_envs if args.n_envs else config.get('n_envs', 8)
    vec_type = args.vec_type if args.vec_type else config.get('vec', 'dummy')
    checkpoint_freq = config.get('checkpoint_freq', 100000)
    
    # Training schedule
    total_runs = len(fold_specs) * len(seeds)
    print(f"\n[SCHEDULE] Training Schedule:")
    print(f"  Folds: {len(fold_specs)}")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    print(f"  Total runs: {total_runs}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Vectorization: {vec_type}")
    
    # Estimate time
    updates_per_run = ppo_cfg.total_timesteps / (ppo_cfg.n_steps * n_envs)
    estimated_minutes_per_run = updates_per_run * 0.5  # Rough estimate: 0.5 min/update
    estimated_total_hours = (estimated_minutes_per_run * total_runs) / 60
    
    print(f"\n[ESTIMATE] Approximate training time:")
    print(f"  Per fold: {estimated_minutes_per_run:.0f} minutes")
    print(f"  Total: {estimated_total_hours:.1f} hours")
    
    # Train all folds
    completed = 0
    skipped = 0
    
    for fold_spec in fold_specs:
        for seed in seeds:
            fold_id = fold_spec["fold"]
            
            # Check if already trained
            if args.skip_finished:
                model_path = os.path.join(
                    args.output_dir, "models", args.universe, "ppo",
                    f"model_fold_{fold_id}_seed_{seed}.zip"
                )
                if os.path.exists(model_path):
                    print(f"\n[SKIP] Fold {fold_id} seed {seed} already trained")
                    skipped += 1
                    continue
            
            # Train
            try:
                train_fold(
                    universe=args.universe,
                    panel=panel,
                    fold_spec=fold_spec,
                    seed=seed,
                    ppo_cfg=ppo_cfg,
                    env_cfg=env_cfg,
                    output_dir=args.output_dir,
                    n_envs=n_envs,
                    vec_type=vec_type,
                    checkpoint_freq=checkpoint_freq,
                    resume=args.resume,
                )
                completed += 1
            except Exception as e:
                print(f"\n[ERROR] Failed to train fold {fold_id} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"Completed: {completed}/{total_runs}")
    print(f"Skipped: {skipped}/{total_runs}")
    print(f"Failed: {total_runs - completed - skipped}/{total_runs}")
    
    if completed > 0:
        print(f"\n[SUCCESS] Models saved to: {args.output_dir}/models/{args.universe}/ppo/")
        print("\nNext steps:")
        print(f"  1. Run evaluation: python evaluate_final.py --universe {args.universe}")
        print(f"  2. Run baselines: python baselines_final.py --universe {args.universe}")
        print(f"  3. Generate report: python generate_summary_report.py --universe {args.universe}")


if __name__ == "__main__":
    main()
