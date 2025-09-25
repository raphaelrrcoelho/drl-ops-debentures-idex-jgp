# train_final.py
"""
Train PPO with DISCRETE action space, ENHANCED features, and MAX ASSETS constraint
==================================================================================
OPTIMIZED VERSION with memory-efficient top-K panel construction:
1. Build panel with only assets that are ever top-K during training
2. Sparse matrix support for efficient memory usage
3. HDF5 caching with densify/sparsify on I/O
4. Removed deprecated/unused code paths

NEW: True top-K panel construction that includes:
- Assets currently in top-K by index_weight
- Assets that will be in top-K at any rebalance point
- No unnecessary assets that never enter top-K
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*get_schedule_fn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*constant_fn.*deprecated.*")

import os, json, time, argparse, random, yaml, re, gc
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import numpy as np
import pandas as pd
import csv
import numba
from numba import jit, prange
import pickle
import h5py

import io
io.DEFAULT_BUFFER_SIZE = 65536

# Memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Ensure reproducibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("PyTorch is required for training. Please install torch.") from e

# SB3 and MaskablePPO for discrete actions
try:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except Exception as e:
    raise RuntimeError("sb3-contrib is required for MaskablePPO. Please install sb3-contrib.") from e

try:
    import gymnasium as gym
except Exception as e:
    raise RuntimeError("gymnasium is required.") from e

# Import updated environment with max_assets
try:
    from env_final import EnvConfig, make_env_from_panel
except Exception as e:
    raise RuntimeError("env_final.py with max_assets support is required.") from e

# ------------------------------- Configs ----------------------------------- #

@dataclass
class PPOConfig:
    policy: str = "MultiInputPolicy"
    total_timesteps: int = 10_000
    learning_rate: float = 5e-6
    n_steps: int = 256
    batch_size: int = 124
    n_epochs: int = 10
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

# Memory efficiency parameters
SPARSE_THRESHOLD = 0.5  # Use sparse if >50% zeros/missing
MAX_MEMORY_GB = 8.0  # Max memory to use (GB)

# -------------------------- Required Panel Columns ------------------------- #

REQUIRED_COLS = [
    "return", "spread", "duration", "time_to_maturity", "sector_id", "active",
    "risk_free", "index_return", "index_level", "index_weight",
]

# ------------------------------ JIT Compiled Helpers ----------------------- #

@jit(nopython=True, cache=True, fastmath=True)
def compute_sharpe_ratio_jit(returns: np.ndarray, rf: np.ndarray, trading_days: int = 252) -> float:
    """JIT compiled Sharpe ratio calculation."""
    if returns.size == 0:
        return -np.inf
    excess = returns - rf
    std = np.std(excess)
    if std <= 0:
        return -np.inf
    return float(np.mean(excess) / std * np.sqrt(trading_days))

@jit(nopython=True, cache=True, fastmath=True)
def compute_information_ratio_jit(returns: np.ndarray, benchmark: np.ndarray, trading_days: int = 252) -> float:
    """JIT compiled Information ratio calculation."""
    if returns.size == 0:
        return -np.inf
    active = returns - benchmark
    std = np.std(active)
    if std <= 0:
        return -np.inf
    return float(np.mean(active) / std * np.sqrt(trading_days))

# ------------------------------ Memory Management -------------------------- #

@contextmanager
def memory_efficient_context():
    """Context manager for memory-efficient operations."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    yield
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def check_memory_usage():
    """Check current memory usage."""
    if HAS_PSUTIL:
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1024**3
        available_gb = psutil.virtual_memory().available / 1024**3
        return mem_gb, available_gb
    return None, None

# ======================= HDF5 CACHING ======================= #

class HDF5PanelCache:
    """Efficient HDF5-based panel storage with sparse support."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, universe: str, fold: int) -> str:
        return os.path.join(self.cache_dir, f"{universe}_fold_{fold}_topk.h5")
    
    def exists(self, universe: str, fold: int) -> bool:
        return os.path.exists(self.get_cache_path(universe, fold))
    
    def save_panel(self, panel: pd.DataFrame, universe: str, fold: int, 
                  union_ids: List[str], sparse_cols: Optional[List[str]] = None):
        """Save panel to HDF5, handling sparse columns."""
        cache_path = self.get_cache_path(universe, fold)
        print(f"    Caching panel to: {cache_path}")
        
        try:
            # Convert sparse columns to dense for HDF5 storage
            panel_to_save = panel.copy()
            sparse_info = {}
            
            if sparse_cols:
                for col in sparse_cols:
                    if hasattr(panel_to_save[col], 'sparse'):
                        sparse_info[col] = True
                        panel_to_save[col] = panel_to_save[col].sparse.to_dense()
            
            # Save panel
            panel_to_save.to_hdf(
                cache_path,
                key='panel',
                mode='w',
                complib='blosc',
                complevel=5,
                format="table",
            )
            
            # Save metadata
            with h5py.File(cache_path, 'a') as f:
                f.attrs['union_ids'] = json.dumps(union_ids)
                f.attrs['sparse_cols'] = json.dumps(sparse_info)
                f.attrs['n_rows'] = len(panel)
                f.attrs['n_cols'] = len(panel.columns)
        except Exception as e:
            print(f"    Warning: Cache save failed: {e}")
    
    def load_panel(self, universe: str, fold: int) -> Tuple[pd.DataFrame, List[str]]:
        """Load panel from HDF5 cache, restoring sparse columns."""
        cache_path = self.get_cache_path(universe, fold)
        print(f"    Loading cached panel from: {cache_path}")
        
        panel = pd.read_hdf(cache_path, key='panel')
        
        with h5py.File(cache_path, 'r') as f:
            union_ids = json.loads(f.attrs['union_ids'])
            sparse_cols = json.loads(f.attrs.get('sparse_cols', '{}'))
        
        # Restore sparse columns
        if sparse_cols:
            for col in sparse_cols:
                if col in panel.columns:
                    panel[col] = pd.arrays.SparseArray(panel[col], fill_value=0)
        
        return panel, union_ids

# ======================= CORE TOP-K COMPUTATION ======================= #

def compute_ever_topk_ids(panel: pd.DataFrame,
                          train_dates: pd.DatetimeIndex,
                          K: int = 50,
                          rebalance_interval: int = 5) -> List[str]:
    """
    Return the union of assets that are ever in the top-K on any rebalance date.
    This is the minimal set needed for the hold-until-rebalance strategy.
    """
    assert "index_weight" in panel.columns, "index_weight column required"
    assert "active" in panel.columns, "active column required"

    ever = set()
    
    # Only check rebalance dates (not every date)
    rb_indices = np.arange(0, len(train_dates), max(1, rebalance_interval))
    rb_dates = train_dates[rb_indices]
    
    print(f"  Checking {len(rb_dates)} rebalance dates for top-{K} assets...")
    
    for i, d in enumerate(rb_dates):
        try:
            # Get data for this date
            rows = panel.xs(d, level="date", drop_level=False)
        except KeyError:
            continue
            
        # Only consider active assets
        rows = rows[rows["active"] > 0]
        if rows.empty:
            continue
            
        # Get top-K by index_weight
        weights = rows["index_weight"].values
        if len(weights) > K:
            # Use argpartition for efficiency
            top_idx = np.argpartition(-weights, K-1)[:K]
            sel_ids = rows.iloc[top_idx].index.get_level_values("debenture_id")
        else:
            sel_ids = rows.index.get_level_values("debenture_id")
            
        ever.update(sel_ids.tolist())
        
        if (i + 1) % 20 == 0:
            print(f"    Processed {i+1}/{len(rb_dates)} rebalance dates, found {len(ever)} unique assets")
    
    result = sorted(ever)
    print(f"  Total ever-top-{K} assets: {len(result)}")
    return result

def build_topk_panel(panel: pd.DataFrame, 
                    fold_cfg: Dict[str, str],
                    env_cfg: EnvConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build minimal training panel with only ever-top-K assets.
    This dramatically reduces memory usage while maintaining correctness.
    """
    fold = int(fold_cfg.get("fold", 0))
    train_start = pd.to_datetime(fold_cfg["train_start"])
    train_end = pd.to_datetime(fold_cfg["train_end"])
    
    print(f"\n[TOP-K PANEL CONSTRUCTION - Fold {fold}]")
    print(f"  Train period: {train_start.date()} to {train_end.date()}")
    print(f"  Max assets per timestep: {env_cfg.max_assets}")
    print(f"  Rebalance interval: {env_cfg.rebalance_interval}")
    
    # Check cache first
    cache = HDF5PanelCache()
    if cache.exists("cdi", fold):
        try:
            return cache.load_panel("cdi", fold)
        except Exception as e:
            print(f"  Cache load failed: {e}, rebuilding...")
    
    with memory_efficient_context():
        # Get training dates
        all_dates = panel.index.get_level_values("date").unique()
        train_dates = all_dates[(all_dates >= train_start) & (all_dates <= train_end)].sort_values()
        print(f"  Train dates: {len(train_dates)}")
        
        # Compute ever-top-K assets
        ever_topk_ids = compute_ever_topk_ids(
            panel, 
            train_dates, 
            K=env_cfg.max_assets,
            rebalance_interval=env_cfg.rebalance_interval
        )
        
        # Build compact panel with only these assets
        print(f"  Building compact panel: {len(train_dates)} × {len(ever_topk_ids)} = {len(train_dates) * len(ever_topk_ids):,} obs")
        
        # Extract relevant subset
        date_mask = panel.index.get_level_values("date").isin(train_dates)
        asset_mask = panel.index.get_level_values("debenture_id").isin(ever_topk_ids)
        train_subset = panel.loc[date_mask & asset_mask].copy()
        
        # Create full index and reindex
        full_idx = pd.MultiIndex.from_product(
            [train_dates, ever_topk_ids],
            names=["date", "debenture_id"]
        )
        train_aug = train_subset.reindex(full_idx).sort_index()
        
        # Process missing values efficiently
        fill_specs = [
            ("active", 0, np.int8),
            ("return", 0.0, np.float32),
            ("spread", 0.0, np.float32),
            ("duration", 0.0, np.float32),
            ("time_to_maturity", 0.0, np.float32),
            ("sector_id", -1, np.int16),
            ("index_weight", 0.0, np.float32),
        ]
        
        for col, fill_val, dtype in fill_specs:
            if col in train_aug.columns:
                train_aug[col] = train_aug[col].fillna(fill_val).astype(dtype)
        
        # Broadcast date-level columns
        date_cols = ["risk_free", "index_return", "index_level"]
        date_maps = {}
        for name in date_cols:
            if name in panel.columns:
                date_map = panel.groupby(level="date", sort=False)[name].first()
                date_maps[name] = date_map
        
        if date_maps:
            train_aug = train_aug.reset_index()
            for name, mapping in date_maps.items():
                train_aug[name] = train_aug["date"].map(mapping).astype(np.float32).fillna(0.0)
            train_aug = train_aug.set_index(["date", "debenture_id"]).sort_index()
        
        # Check sparsity and convert if beneficial
        sparse_cols = []
        for col in train_aug.columns:
            if pd.api.types.is_numeric_dtype(train_aug[col]):
                zeros_ratio = (train_aug[col] == 0).sum() / len(train_aug)
                if zeros_ratio > SPARSE_THRESHOLD:
                    print(f"    Converting {col} to sparse ({zeros_ratio:.1%} zeros)")
                    train_aug[col] = pd.arrays.SparseArray(train_aug[col], fill_value=0)
                    sparse_cols.append(col)
        
        # Statistics
        feature_cols = [c for c in train_aug.columns if c.endswith("_lag1")]
        print(f"  Features: {len(feature_cols)} lagged features")
        
        active_per_day = (train_aug["active"] > 0).groupby(level="date").sum()
        print(f"  Active assets/day: {active_per_day.mean():.1f} (min: {active_per_day.min()}, max: {active_per_day.max()})")
        
        mem_usage_mb = train_aug.memory_usage(deep=True).sum() / 1024**2
        print(f"  Panel memory usage: {mem_usage_mb:.1f} MB")
        
        # Cache for future use
        try:
            cache.save_panel(train_aug, "cdi", fold, ever_topk_ids, sparse_cols)
        except Exception as e:
            print(f"  Cache save failed: {e}")
        
        return train_aug, ever_topk_ids

# ------------------------------ Helper Utils ------------------------------- #

def _policy_kwargs_from_cfg(ppo_cfg: PPOConfig) -> dict:
    act_fn = nn.Tanh if str(ppo_cfg.activation).lower() == "tanh" else nn.ReLU
    return dict(net_arch=ppo_cfg.net_arch, activation_fn=act_fn, ortho_init=ppo_cfg.ortho_init)

def set_global_seed(seed: int = 0):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ---------------------------- Optimized Fold Construction ------------------ #

def folds_from_dates(dates: pd.DatetimeIndex, n_folds: int = 9, embargo_days: int = 3) -> List[Dict[str, str]]:
    """Build expanding window folds."""
    dates = pd.to_datetime(pd.Index(dates).unique().sort_values())
    T = len(dates)
    
    print(f"\n[FOLD CONSTRUCTION]")
    print(f"  Total dates: {T} ({dates[0].date()} to {dates[-1].date()})")
    print(f"  Number of folds: {n_folds}")
    print(f"  Embargo days: {embargo_days}")
    
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
        train_end_idx = max(int(T / (n_folds + 1)), 10) if i == 0 else cuts[i]
        train_end = dates[min(train_end_idx, T - 1)]
        
        test_start_idx = min(train_end_idx + embargo_days + 1, T - 1)
        test_end_idx = cuts[i + 1] if i + 1 < len(cuts) else T - 1
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        
        if test_start_idx >= test_end_idx or train_end_idx < 5:
            continue
            
        fold = {
            "fold": int(i),
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        }
        folds.append(fold)
        
        train_days = (train_end - train_start).days
        test_days = (test_end - test_start).days
        print(f"  Fold {i}: Train {train_days}d, Test {test_days}d")
    
    return folds

# ---------------------- Wrappers for Discrete Actions ---------------------- #

def mask_fn(env: gym.Env) -> np.ndarray:
    """Action mask function for MaskablePPO."""
    if hasattr(env, 'get_action_masks'):
        masks = env.get_action_masks()
        if isinstance(masks, list):
            total_size = sum(len(m) for m in masks)
            result = np.empty(total_size, dtype=bool)
            idx = 0
            for m in masks:
                size = len(m)
                result[idx:idx+size] = m
                idx += size
            return result
        return masks
    else:
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'nvec'):
            return np.ones(sum(env.action_space.nvec), dtype=bool)
        return np.ones(100, dtype=bool)

class SafeObsWrapper(gym.ObservationWrapper):
    """Observation wrapper with NaN/Inf handling."""
    def __init__(self, env, clip: float = 7.5):
        super().__init__(env)
        self._clip = float(clip)

    def observation(self, obs):
        if isinstance(obs, dict):
            processed = {}
            for key, val in obs.items():
                if isinstance(val, np.ndarray):
                    if key == 'action_mask':
                        processed[key] = val
                    else:
                        val = np.nan_to_num(val.astype(np.float32), 
                                          nan=0.0, posinf=self._clip, neginf=-self._clip)
                        processed[key] = np.clip(val, -self._clip, self._clip)
                else:
                    processed[key] = val
            return processed
        else:
            obs = np.nan_to_num(obs.astype(np.float32), 
                              nan=0.0, posinf=self._clip, neginf=-self._clip)
            return np.clip(obs, -self._clip, self._clip)

class SafeRewardWrapper(gym.RewardWrapper):
    """Reward wrapper with NaN/Inf handling."""
    def reward(self, rew):
        if not np.isfinite(rew):
            return 0.0
        return float(rew)

# ---------------------------- Optimized Callbacks ------------------------- #

class DetailedMetricsLoggerCallback(BaseCallback):
    """Lightweight metrics logging callback."""
    def __init__(self, out_json_path: str, fold: int, seed: int, verbose: int = 1):
        super().__init__(verbose)
        self.out_json_path = out_json_path
        self.fold = int(fold)
        self.seed = int(seed)
        self.start_time = None
        self.episode_count = 0
        from collections import deque
        self.recent_rewards = deque(maxlen=100)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"\n[TRAINING STARTED - Fold {self.fold}, Seed {self.seed}]")
        print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if HAS_PSUTIL:
            mem_gb, _ = check_memory_usage()
            if mem_gb:
                print(f"  Memory usage: {mem_gb:.2f} GB")

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1
            info = self.locals.get("infos", [{}])[0]
            ep_reward = info.get("episode", {}).get("r", 0)
            self.recent_rewards.append(ep_reward)
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        print(f"\n[TRAINING COMPLETED - Fold {self.fold}, Seed {self.seed}]")
        print(f"  Total time: {elapsed/60:.1f} minutes")
        print(f"  Total episodes: {self.episode_count}")
        
        if self.recent_rewards:
            rewards_array = np.array(self.recent_rewards)
            final_perf = {
                "mean_reward": float(rewards_array.mean()),
                "std_reward": float(rewards_array.std()),
            }
            print(f"  Final performance: mean={final_perf['mean_reward']:.4f}")
        else:
            final_perf = {}
        
        rec = {
            "fold": self.fold,
            "seed": self.seed,
            "elapsed_sec": float(elapsed),
            "total_episodes": self.episode_count,
            "total_timesteps": self.num_timesteps,
            **final_perf
        }
        
        # Save metrics
        csv_path = self.out_json_path.replace('.json', '.csv')
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(rec)

# ------------------------ In-Sample Validation ------------------ #

def validate_reward_params(
    train_panel: pd.DataFrame,
    fold_cfg: Dict[str, str],
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    lambda_grid: Dict[str, List[float]],
    selection_metric: str = "ir",
    seed: int = 0,
    validation_timesteps: int = 1000,
) -> Dict[str, float]:
    """Fast grid-search for lambda penalties."""
    import dataclasses
    from dataclasses import asdict
    
    # Split train data
    dates = train_panel.index.get_level_values(0).unique().sort_values()
    cut = int(len(dates) * 0.8)
    
    if cut < 10:
        print("  ⚠ Not enough data for validation, using defaults")
        return {
            "lambda_turnover": float(env_cfg.lambda_turnover),
            "lambda_hhi": float(env_cfg.lambda_hhi),
            "lambda_drawdown": float(env_cfg.lambda_drawdown),
        }
    
    train_dates = dates[:cut]
    val_dates = dates[cut:]
    
    with memory_efficient_context():
        train_slice = train_panel.loc[train_panel.index.get_level_values(0).isin(train_dates)]
        val_slice = train_panel.loc[train_panel.index.get_level_values(0).isin(val_dates)]
        
        # Grid search
        grid_to = lambda_grid.get("lambda_turnover", [env_cfg.lambda_turnover])
        grid_hhi = lambda_grid.get("lambda_hhi", [env_cfg.lambda_hhi])
        grid_dd = lambda_grid.get("lambda_drawdown", [env_cfg.lambda_drawdown])
        
        from itertools import product
        combinations = list(product(grid_to, grid_hhi, grid_dd))
        
        print(f"\n[HYPERPARAMETER VALIDATION]")
        print(f"  Testing {len(combinations)} combinations...")
        
        best_score = -np.inf
        best = (float(env_cfg.lambda_turnover), float(env_cfg.lambda_hhi), float(env_cfg.lambda_drawdown))
        
        for i, (lam_to, lam_hhi, lam_dd) in enumerate(combinations, 1):
            cfg_i = dataclasses.replace(
                env_cfg,
                lambda_turnover=float(lam_to),
                lambda_hhi=float(lam_hhi),
                lambda_drawdown=float(lam_dd),
            )
            
            # Quick train
            def _make():
                e = make_env_from_panel(train_slice, **asdict(cfg_i))
                e = ActionMasker(e, mask_fn)
                e = SafeRewardWrapper(e)
                return e
            
            venv = DummyVecEnv([_make])
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=ppo_cfg.gamma)
            
            model = MaskablePPO(
                ppo_cfg.policy,
                venv,
                learning_rate=ppo_cfg.learning_rate,
                n_steps=min(64, ppo_cfg.n_steps),
                batch_size=min(64, ppo_cfg.batch_size),
                n_epochs=2,
                gamma=ppo_cfg.gamma,
                verbose=0,
                device="cpu",
                seed=seed,
            )
            
            model.learn(total_timesteps=validation_timesteps, progress_bar=False)
            
            # Evaluate on validation
            def _make_val():
                e = make_env_from_panel(val_slice, **asdict(cfg_i))
                e = ActionMasker(e, mask_fn)
                e = SafeRewardWrapper(e)
                return e
            
            venv_val = DummyVecEnv([_make_val])
            venv_val.obs_rms = venv.obs_rms
            venv_val.ret_rms = venv.ret_rms
            venv_val.training = False
            
            # Collect returns
            returns = []
            obs = venv_val.reset()
            for _ in range(min(100, len(val_dates))):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = venv_val.step(action)
                returns.append(infos[0].get("portfolio_return", 0.0))
                if dones[0]:
                    break
            
            # Score
            if returns and selection_metric == "ir":
                score = compute_information_ratio_jit(
                    np.array(returns, dtype=np.float32),
                    np.zeros(len(returns), dtype=np.float32),  # Simplified
                    252
                )
            else:
                score = np.mean(returns) if returns else -np.inf
            
            if score > best_score:
                best_score = score
                best = (float(lam_to), float(lam_hhi), float(lam_dd))
            
            if i % max(1, len(combinations) // 3) == 1:
                print(f"    Tested {i}/{len(combinations)}, best score: {best_score:.4f}")
            
            venv.close()
            venv_val.close()
            del model
            gc.collect()
    
    print(f"  Best lambdas: turnover={best[0]:.4f}, hhi={best[1]:.4f}, dd={best[2]:.4f}")
    return {"lambda_turnover": best[0], "lambda_hhi": best[1], "lambda_drawdown": best[2]}

# ------------------------------ Train One ----------------------- #

def train_one(
    universe: str,
    panel: pd.DataFrame,
    fold_cfg: Dict[str, str],
    seed: int,
    ppo_cfg: PPOConfig,
    env_cfg: EnvConfig,
    out_base: str,
    lambda_grid: Optional[Dict[str, List[float]]] = None,
    save_freq: int = 500_000,
    selection_metric: str = "ir",
    resume: bool = False,
    n_envs: int = 16,
    vec_kind: str = "subproc",
    episode_len: Optional[int] = 256,
    reset_jitter_frac: float = 0.9,
    validation_timesteps: int = 1000,
    do_validation: bool = False,
) -> Dict[str, str]:
    """Train a single fold/seed combination with top-K panel."""
    fold_i = int(fold_cfg["fold"])
    seed = int(seed)
    
    print("\n" + "="*60)
    print(f"TRAINING FOLD {fold_i} SEED {seed}")
    print("="*60)
    print(f"Universe: {universe.upper()}")
    print(f"Train: {fold_cfg['train_start']} to {fold_cfg['train_end']}")
    print(f"Test: {fold_cfg['test_start']} to {fold_cfg['test_end']}")
    print(f"Max assets: {env_cfg.max_assets}")
    
    # Setup directories
    results_dir = os.path.join(out_base, "results", universe)
    model_dir = os.path.join(out_base, "models", universe, "ppo")
    tb_dir = os.path.join(out_base, "tb", universe, f"fold_{fold_i}_seed_{seed}")
    ckpt_dir = os.path.join(model_dir, f"fold_{fold_i}_seed_{seed}")
    ensure_dirs(results_dir, model_dir, tb_dir, ckpt_dir)
    
    with memory_efficient_context():
        # Build top-K panel (memory efficient)
        train_panel, union_ids = build_topk_panel(panel, fold_cfg, env_cfg)
        
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
                json.dump(existing, f)
        except Exception as e:
            print(f"  ⚠ Could not save union ids: {e}")
        
        # Validate lambdas if requested
        if do_validation and lambda_grid:
            best = validate_reward_params(
                train_panel, fold_cfg, env_cfg, ppo_cfg,
                lambda_grid, selection_metric,
                validation_timesteps, seed
            )
            env_cfg = EnvConfig(**{**asdict(env_cfg), **best})
        
        # Create environments
        print(f"\n[ENVIRONMENT SETUP]")
        print(f"  Parallel environments: {n_envs}")
        print(f"  Vectorization: {vec_kind}")
        
        # Create base environment and get shared arrays
        cfg0 = EnvConfig(**{**asdict(env_cfg), "seed": seed * 1000})
        base_env = make_env_from_panel(train_panel, **asdict(cfg0))
        shared = base_env.export_shared_arrays()
        
        def _make_env(rank: int):
            def _init():
                cfg_i = EnvConfig(**{**asdict(env_cfg), "seed": seed * 1000 + rank})
                
                if episode_len is not None:
                    cfg_i.max_steps = episode_len
                if reset_jitter_frac is not None:
                    cfg_i.random_reset_frac = reset_jitter_frac
                
                if rank == 0:
                    e = base_env
                else:
                    e = make_env_from_panel(train_panel, **asdict(cfg_i), prebuilt=shared)
                
                e = ActionMasker(e, mask_fn)
                e = SafeRewardWrapper(e)
                e = SafeObsWrapper(e, clip=cfg_i.obs_clip)
                return e
            return _init
        
        # Create vectorized environment
        thunks = [_make_env(i) for i in range(n_envs)]
        if n_envs <= 1 or vec_kind == "dummy":
            vec_env = DummyVecEnv(thunks)
        else:
            vec_env = SubprocVecEnv(thunks, start_method="spawn")
        
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, clip_reward=10.0, gamma=ppo_cfg.gamma)
        
        # Create or load model
        set_global_seed(seed)
        
        print(f"\n[MODEL CONFIGURATION]")
        print(f"  Policy: {ppo_cfg.policy}")
        print(f"  Network: {ppo_cfg.net_arch}")
        print(f"  Learning rate: {ppo_cfg.learning_rate:.2e}")
        print(f"  Total timesteps: {ppo_cfg.total_timesteps:,}")
        
        already_trained = 0
        model = None
        
        # Check for checkpoint to resume
        if resume:
            ckpt_pattern = re.compile(r"ppo_checkpoint_(\d+)_steps\.zip")
            best_ckpt = None
            best_steps = -1
            
            if os.path.isdir(ckpt_dir):
                for fn in os.listdir(ckpt_dir):
                    match = ckpt_pattern.match(fn)
                    if match:
                        steps = int(match.group(1))
                        if steps > best_steps:
                            best_steps = steps
                            best_ckpt = os.path.join(ckpt_dir, fn)
            
            if best_ckpt:
                print(f"\n[RESUMING FROM CHECKPOINT]")
                print(f"  Path: {best_ckpt}")
                print(f"  Steps: {best_steps:,}")
                try:
                    model = MaskablePPO.load(best_ckpt, env=vec_env, device="cpu")
                    already_trained = best_steps
                except Exception as e:
                    print(f"  ⚠ Failed to load checkpoint: {e}")
        
        if model is None:
            policy_kwargs = _policy_kwargs_from_cfg(ppo_cfg)
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
                device="cpu",
                seed=seed,
            )
        
        # Train
        remaining = max(0, ppo_cfg.total_timesteps - already_trained)
        if remaining > 0:
            print(f"\n[TRAINING IN PROGRESS]")
            print(f"  Remaining timesteps: {remaining:,}")
            
            callbacks = [
                CheckpointCallback(save_freq=save_freq, save_path=ckpt_dir, name_prefix="ppo_checkpoint"),
                DetailedMetricsLoggerCallback(
                    os.path.join(results_dir, "training_metrics.json"),
                    fold_i, seed
                )
            ]
            
            model.learn(total_timesteps=remaining, 
                       callback=CallbackList(callbacks),
                       progress_bar=True)
        else:
            print("\n[TRAINING COMPLETE]")
            print("  No remaining timesteps")
        
        # Save VecNormalize stats
        try:
            vecnorm_path = os.path.join(model_dir, f"vecnorm_fold_{fold_i}_seed_{seed}.pkl")
            stats = {
                'obs_rms': vec_env.obs_rms,
                'ret_rms': vec_env.ret_rms,
                'clip_obs': vec_env.clip_obs,
                'clip_reward': vec_env.clip_reward,
                'gamma': vec_env.gamma,
            }
            
            with open(vecnorm_path, 'wb') as f:
                pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n[VECNORM SAVED]: {vecnorm_path}")
        except Exception as e:
            print(f"  ⚠ Could not save VecNormalize: {e}")
        
        # Clean up and save
        vec_env.close()
        del vec_env
        gc.collect()
        
        # Save final model
        final_path = os.path.join(model_dir, f"model_fold_{fold_i}_seed_{seed}.zip")
        model.save(final_path)
        print(f"[MODEL SAVED]: {final_path}")
    
    print("\n" + "="*60)
    print(f"FOLD {fold_i} SEED {seed} COMPLETE")
    print("="*60)
    
    return {
        "model_path": final_path,
        "tensorboard_dir": tb_dir,
        "fold": str(fold_i),
        "seed": str(seed),
    }

# --------------------------------- Main ------------------------------------ #

def _load_panel_for_universe(universe: str, data_dir: str) -> pd.DataFrame:
    """Load and validate panel data."""
    proc_path = os.path.join(data_dir, f"{universe}_processed.pkl")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"Processed panel not found: {proc_path}. Run data_final.py first.")
    
    print(f"\n[LOADING DATA]")
    print(f"  Path: {proc_path}")
    
    with memory_efficient_context():
        df = pd.read_pickle(proc_path)
        
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("Panel index must be MultiIndex (date, debenture_id).")
        
        dates = df.index.get_level_values("date").unique()
        assets = df.index.get_level_values("debenture_id").unique()
        features = [c for c in df.columns if c.endswith("_lag1")]
        
        print(f"  Dates: {len(dates)} ({dates.min().date()} to {dates.max().date()})")
        print(f"  Assets: {len(assets)}")
        print(f"  Features: {len(features)} lagged")
        print(f"  Total observations: {len(df):,}")
        
        # Check required columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Panel missing required columns: {missing}")
        
        # Ensure index_weight exists
        if "index_weight" not in df.columns:
            print("  ⚠ Creating synthetic index_weight")
            for date in dates:
                mask = df.index.get_level_values("date") == date
                active_mask = (df.loc[mask, "active"] > 0) if "active" in df.columns else pd.Series(True, index=df.loc[mask].index)
                n_active = active_mask.sum()
                df.loc[mask, "index_weight"] = active_mask.astype(float) / n_active if n_active > 0 else 0.0
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO with memory-efficient top-K constraint")
    parser.add_argument("--universe", type=str, choices=["cdi", "infra"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_base", type=str, default=".")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--n_folds", type=int, default=9)
    parser.add_argument("--embargo_days", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="0,1,2,4")
    parser.add_argument("--selection_metric", default="ir", choices=["ir","sharpe","sortino","calmar"])
    parser.add_argument("--skip_finished", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--vec", type=str, default="subproc", choices=["dummy","subproc"])
    parser.add_argument("--n_jobs", type=int, default=1)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MEMORY-EFFICIENT PPO TRAINING WITH TOP-K CONSTRAINT")
    print("="*60)
    print(f"Universe: {args.universe.upper()}")
    print(f"Configuration: {args.config}")
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    
    # Load data
    panel = _load_panel_for_universe(args.universe, args.data_dir)
    
    # Build folds
    dates = panel.index.get_level_values("date").unique().sort_values()
    fold_specs = folds_from_dates(dates, n_folds=args.n_folds, embargo_days=args.embargo_days)
    
    # Save fold specs
    results_dir = os.path.join(args.out_base, "results", args.universe)
    ensure_dirs(results_dir)
    
    fold_spec_path = os.path.join(results_dir, "training_folds.json")
    with open(fold_spec_path, "w") as f:
        json.dump(fold_specs, f, indent=2)
    
    # Create configs
    net_arch = config.get('net_arch', [256, 256])
    if isinstance(net_arch, list):
        net_arch = tuple(net_arch)
    
    ppo_cfg = PPOConfig(
        policy=config.get('policy', 'MultiInputPolicy'),
        total_timesteps=config.get('total_timesteps', 40960),
        learning_rate=config.get('learning_rate', 5e-6),
        n_steps=config.get('n_steps', 256),
        batch_size=config.get('batch_size', 124),
        n_epochs=config.get('n_epochs', 10),
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
    
    env_cfg = EnvConfig(
        rebalance_interval=config.get('rebalance_interval', 5),
        max_weight=config.get('max_weight', 0.10),
        weight_blocks=config.get('weight_blocks', 100),
        max_assets=config.get('max_assets', 50),
        allow_cash=config.get('allow_cash', True),
        cash_rate_as_rf=config.get('cash_rate_as_rf', True),
        on_inactive=config.get('on_inactive', 'to_cash'),
        transaction_cost_bps=config.get('transaction_cost_bps', 20.0),
        delist_extra_bps=config.get('delist_extra_bps', 20.0),
        normalize_features=config.get('normalize_features', True),
        obs_clip=config.get('obs_clip', 7.5),
        include_prev_weights=config.get('include_prev_weights', False),
        include_active_flag=config.get('include_active_flag', False),
        global_stats=config.get('global_stats', True),
        lambda_turnover=config.get('lambda_turnover', 0.0002),
        lambda_hhi=config.get('lambda_hhi', 0.01),
        lambda_drawdown=config.get('lambda_drawdown', 0.005),
        lambda_tail=config.get('lambda_tail', 0.001),
        tail_window=config.get('tail_window', 60),
        tail_q=config.get('tail_q', 0.05),
        dd_mode=config.get('dd_mode', 'incremental'),
        weight_alpha=config.get('weight_alpha', 1.0),
        max_steps=config.get('max_steps', 256),
        random_reset_frac=config.get('random_reset_frac', 0.9),
        use_momentum_features=config.get('use_momentum_features', True),
        use_volatility_features=config.get('use_volatility_features', False),
        use_relative_value_features=config.get('use_relative_value_features', True),
        use_duration_features=config.get('use_duration_features', True),
        use_microstructure_features=config.get('use_microstructure_features', False),
        use_carry_features=config.get('use_carry_features', True),
        use_spread_dynamics=config.get('use_spread_dynamics', True),
        use_risk_adjusted_features=config.get('use_risk_adjusted_features', False),
        use_sector_curves=config.get('use_sector_curves', True),
        use_zscore_features=config.get('use_zscore_features', False),
        use_rolling_zscores=config.get('use_rolling_zscores', False),
    )
    
    # Get validation config
    do_validation = config.get('do_validation', False)
    lambda_grid = config.get('lambda_grid', None) if do_validation else None
    validation_timesteps = config.get('validation_timesteps', 1000)
    
    # Save config
    config_path = os.path.join(results_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "ppo_config": asdict(ppo_cfg),
            "env_config": asdict(env_cfg),
            "lambda_grid": lambda_grid,
            "seeds": args.seeds,
            "n_folds": args.n_folds,
            "max_assets": env_cfg.max_assets,
        }, f, indent=2)
    
    # Train
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    print(f"\n[TRAINING SCHEDULE]")
    print(f"  Folds: {len(fold_specs)}")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    print(f"  Total runs: {len(fold_specs) * len(seeds)}")
    
    for fold in fold_specs:
        for seed in seeds:
            set_global_seed(seed)
            
            if args.skip_finished:
                final_path = os.path.join(
                    args.out_base, "models", args.universe, "ppo",
                    f"model_fold_{fold['fold']}_seed_{seed}.zip"
                )
                if os.path.exists(final_path):
                    print(f"\n[SKIP] Fold {fold['fold']} seed {seed} already exists")
                    continue
            
            train_one(
                args.universe, panel, fold, seed,
                ppo_cfg, env_cfg, args.out_base,
                lambda_grid,
                config.get('checkpoint_freq', 500000),
                args.selection_metric,
                args.resume,
                args.n_envs if args.n_envs else config.get('n_envs', 16),
                args.vec if args.vec else config.get('vec', 'subproc'),
                config.get('episode_len', 256),
                config.get('reset_jitter_frac', 0.9),
                validation_timesteps,
                do_validation,
            )
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()