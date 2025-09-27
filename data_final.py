# data_final.py
"""
Data preparation for CDI debenture universe - CORRECTED VERSION
================================================================

CRITICAL: IDEX CDI returns are ALREADY TOTAL RETURNS (spread + CDI)
- The raw data contains total returns, not spread-only
- MTM and Carry components when divided by weight give total returns
- No need to add risk-free separately
- Index returns are also total returns

Features are calculated on appropriate components:
- Momentum/Reversal: Based on total returns
- Volatility: Based on total returns  
- Risk-adjusted: Using excess returns (total - RF)
"""

from __future__ import annotations

import os
import json
import glob
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional HTTP for BCB data
try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    requests = None
    _HAS_REQUESTS = False


# ----------------------------- Config & constants ----------------------------

TRADING_DAYS_PER_YEAR = 252.0
BUSINESS_DAYS_SHORT_DROP = 21
TAU_SHORT_ANCHOR = 1.0 / TRADING_DAYS_PER_YEAR

# Momentum windows for multi-horizon features
MOMENTUM_WINDOWS = [1, 5, 20, 60, 126]
VOLATILITY_WINDOWS = [5, 20, 60]

SERIES_CODES = {
    "cdi_primary": 4389,
    "selic": 11,
}
_CDI_CANDIDATES = [SERIES_CODES["cdi_primary"], 12, 4392]
_SGS_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

CACHE_DIR = os.path.join("data", "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

REQUIRED_COLS = [
    "return", "spread", "duration", "sector_id", "active",
    "risk_free", "index_return", "time_to_maturity", "index_level",
]

OPTIONAL_KEEP = [
    "rating", "index_weight", "issuer", "sector",
    "index_level",
    # New features (will be added below)
    "ttm_rank", "excess_return", "sector_weight_index", "sector_spread_avg", 
    "sector_spread", "sector_momentum",
    # Momentum features
    *[f"momentum_{w}d" for w in MOMENTUM_WINDOWS],
    *[f"reversal_{w}d" for w in MOMENTUM_WINDOWS],
    # Volatility features
    *[f"volatility_{w}d" for w in VOLATILITY_WINDOWS],
    *[f"spread_vol_{w}d" for w in VOLATILITY_WINDOWS],
    # Relative value features
    "spread_vs_sector_median", "spread_vs_sector_mean",
    "spread_percentile_sector", "spread_percentile_all",
    # Duration risk features
    "duration_change", "duration_vol", "duration_spread_interaction",
    "modified_duration_proxy", "convexity_proxy",
    # Microstructure features  
    "liquidity_score", "weight_momentum", "weight_volatility",
    # Carry features
    "carry_spread_ratio", "carry_momentum", "carry_vol",
    # Spread dynamics
    "spread_momentum_5d", "spread_momentum_20d",
    "spread_mean_reversion", "spread_acceleration",
    # Risk-adjusted features
    "sharpe_5d", "sharpe_20d", "sharpe_60d",
    "information_ratio_20d",
    # ANBIMA sector curve features
    "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
    "sector_fitted_spread", "spread_residual_ns",
    "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
]


# --------------------------------- Helpers ----------------------------------

def _to_datetime_ser(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)

def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _load_xls(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path, engine=None)

def _normalize_columns(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Standardize raw Idex columns and coerce types."""
    rename = {
        "Data": "date",
        "Debênture": "debenture_id",
        "Debenture": "debenture_id",
        "Duration": "duration",
        "Duração": "duration",
        "Índice": "index_level",
        "Indice": "index_level",
        "Índice (nível)": "index_level",
        "Indice (nível)": "index_level",
        "Peso no índice (%)": "index_weight",
        "MTM ponderado (%)": "weighted_mtm",
        "Variação ponderada (%)": "weighted_return",
        "Carrego ponderado (%)": "weighted_carry",
        "Spread de compra (%)": "spread",
        "MID spread (Bps/NTNB)": "spread",
        "Segmento": "sector",
        "Setor": "sector",
        "Emissor": "issuer",
        "Callable": "callable",
        "Número": "number",
        "Numero": "number",
        "Rating": "rating",
        "Número Índice": "index_value",
    }
    df = df.rename(columns=rename)

    # Core identifiers
    df["date"] = _to_datetime_ser(df.get("date", pd.NaT))
    if "debenture_id" not in df.columns:
        if "Ticker" in df.columns:
            df["debenture_id"] = df["Ticker"].astype(str)
        else:
            raise ValueError("Cannot find a debenture identifier column.")
    else:
        df["debenture_id"] = df["debenture_id"].astype(str)

    # Convert percentages to decimals
    pct_cols = ["index_weight", "weighted_mtm", "weighted_return", "weighted_carry", "index_value"]
    for c in pct_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            df[c] = _safe_float(df[c]) / 100.0

    # Handle spread column
    if "spread" in df.columns:
        if universe.lower() == "infra":
            if df["spread"].dtype == object:
                df["spread"] = df["spread"].astype(str).str.replace(",", ".", regex=False)
            df["spread"] = _safe_float(df["spread"]) / 10000.0
        else:
            if df["spread"].dtype == object:
                df["spread"] = df["spread"].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            df["spread"] = _safe_float(df["spread"]) / 100.0

    # Numeric columns
    for c in ("duration", "index_level"):
        if c in df.columns:
            df[c] = _safe_float(df[c])

    # Categorical columns
    for c in ("sector", "issuer"):
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype("category")

    if "callable" in df.columns:
        df["callable"] = df["callable"].map({"1": 1, 1: 1, True: 1, "Sim": 1}).fillna(0).astype(np.int8)

    df = df.dropna(subset=["date", "debenture_id"]).sort_values(["date", "debenture_id"])
    return df


# --- Enhanced Feature Engineering Functions ---

def _compute_momentum_features(df: pd.DataFrame, windows: List[int] = MOMENTUM_WINDOWS) -> pd.DataFrame:
    """
    Compute momentum and reversal features based on total returns.
    """
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    for w in windows:
        # Momentum: cumulative return over window
        df[f"momentum_{w}d"] = df.groupby("debenture_id", sort=False)["return"].transform(
            lambda x: (1 + x).rolling(w, min_periods=max(1, w//2)).apply(
                lambda y: y.prod() - 1 if len(y) > 0 else 0, raw=True
            )
        ).astype(np.float32)
        
        # Reversal signal (inverse of momentum)
        df[f"reversal_{w}d"] = -df[f"momentum_{w}d"]
    
    return df


def _compute_volatility_features(df: pd.DataFrame, windows: List[int] = VOLATILITY_WINDOWS) -> pd.DataFrame:
    """
    Compute rolling volatility of returns and spreads.
    """
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    for w in windows:
        # Return volatility
        df[f"volatility_{w}d"] = (
            df.groupby("debenture_id", sort=False)["return"].transform(
                lambda x: x.rolling(w, min_periods=max(2, w//3)).std()
            ) * np.sqrt(TRADING_DAYS_PER_YEAR)
        ).astype(np.float32)
        
        # Spread volatility
        df[f"spread_vol_{w}d"] = df.groupby("debenture_id", sort=False)["spread"].transform(
            lambda x: x.rolling(w, min_periods=max(2, w//3)).std()
        ).astype(np.float32)
    
    return df


def _compute_duration_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute duration-based risk measures."""
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Duration change
    df["duration_change"] = df.groupby("debenture_id", sort=False)["duration"].diff().astype(np.float32)
    
    # Duration volatility
    df["duration_vol"] = df.groupby("debenture_id", sort=False)["duration"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    ).astype(np.float32)
    
    # Duration × Spread interaction
    df["duration_spread_interaction"] = (
        df["duration"] * df["spread"]
    ).astype(np.float32)
    
    # Modified duration proxy
    df["modified_duration_proxy"] = (
        df["duration"] / (1 + df["spread"])
    ).astype(np.float32)
    
    # Convexity proxy
    df["convexity_proxy"] = (df["duration"] ** 2 / 100).astype(np.float32)
    
    return df


def _compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidity and microstructure features."""
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Liquidity score (normalized index weight within date)
    for date, group in df.groupby("date"):
        mask = df["date"] == date
        weights = df.loc[mask, "index_weight"]
        if weights.sum() > 0:
            df.loc[mask, "liquidity_score"] = (
                weights / weights.sum()
            ).astype(np.float32)
        else:
            df.loc[mask, "liquidity_score"] = 0.0
    
    # Weight momentum
    df["weight_momentum"] = df.groupby("debenture_id", sort=False)["index_weight"].diff().astype(np.float32)
    
    # Weight volatility
    df["weight_volatility"] = df.groupby("debenture_id", sort=False)["index_weight"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    ).astype(np.float32)
    
    return df


def _compute_carry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute carry-related features."""
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Use spread as carry proxy
    df["carry_proxy"] = df["spread"] * df.get("index_weight", 1.0)
    
    # Carry/Spread ratio
    df["carry_spread_ratio"] = np.where(
        df["spread"] > 0,
        df["carry_proxy"] / df["spread"],
        0.0
    ).astype(np.float32)
    
    # Carry momentum
    df["carry_momentum"] = df.groupby("debenture_id", sort=False)["carry_proxy"].diff().astype(np.float32)
    
    # Carry volatility
    df["carry_vol"] = df.groupby("debenture_id", sort=False)["carry_proxy"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    ).astype(np.float32)
    
    return df


def _compute_spread_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spread momentum and mean reversion."""
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Spread momentum
    df["spread_momentum_5d"] = df.groupby("debenture_id", sort=False)["spread"].diff(5).astype(np.float32)
    df["spread_momentum_20d"] = df.groupby("debenture_id", sort=False)["spread"].diff(20).astype(np.float32)
    
    # Mean reversion signal
    spread_ma60 = df.groupby("debenture_id", sort=False)["spread"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    df["spread_mean_reversion"] = (
        df["spread"] - spread_ma60
    ).astype(np.float32)
    
    # Spread acceleration
    spread_diff = df.groupby("debenture_id", sort=False)["spread"].diff()
    df["spread_acceleration"] = spread_diff.groupby(df["debenture_id"]).diff().astype(np.float32)
    
    return df


def _compute_risk_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Sharpe ratios and information ratios."""
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Ensure excess_return exists
    if "excess_return" not in df.columns:
        df["excess_return"] = df["return"] - df.get("risk_free", 0.0)
    
    # Sharpe ratios at different horizons
    for w in [5, 20, 60]:
        mean_excess = df.groupby("debenture_id", sort=False)["excess_return"].transform(
            lambda x: x.rolling(w, min_periods=max(2, w//3)).mean()
        )
        std_returns = df.groupby("debenture_id", sort=False)["return"].transform(
            lambda x: x.rolling(w, min_periods=max(2, w//3)).std()
        )
        
        df[f"sharpe_{w}d"] = np.where(
            std_returns > 0,
            mean_excess / std_returns * np.sqrt(TRADING_DAYS_PER_YEAR),
            0.0
        ).astype(np.float32)
    
    # Information ratio vs index
    if "index_return" in df.columns:
        df["active_return"] = df["return"] - df["index_return"]
        
        mean_active = df.groupby("debenture_id", sort=False)["active_return"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        std_active = df.groupby("debenture_id", sort=False)["active_return"].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        
        df["information_ratio_20d"] = np.where(
            std_active > 0,
            mean_active / std_active * np.sqrt(TRADING_DAYS_PER_YEAR),
            0.0
        ).astype(np.float32)
        
        # Clean up
        df = df.drop(columns=["active_return"], errors='ignore')
    
    return df


def _compute_relative_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute spread relative to sector and market."""
    df = df.copy()
    
    # Sector-relative features
    for (date, sector), group in df[df["active"] > 0].groupby(["date", "sector_id"]):
        mask = (df["date"] == date) & (df["sector_id"] == sector)
        
        sector_median = group["spread"].median()
        sector_mean = group["spread"].mean()
        
        df.loc[mask, "spread_vs_sector_median"] = (
            df.loc[mask, "spread"] - sector_median
        ).astype(np.float32)
        
        df.loc[mask, "spread_vs_sector_mean"] = (
            df.loc[mask, "spread"] - sector_mean
        ).astype(np.float32)
        
        df.loc[mask, "spread_percentile_sector"] = (
            df.loc[mask, "spread"].rank(pct=True)
        ).astype(np.float32)
    
    # Overall market percentile
    for date, group in df[df["active"] > 0].groupby("date"):
        mask = df["date"] == date
        df.loc[mask, "spread_percentile_all"] = (
            df.loc[mask, "spread"].rank(pct=True)
        ).astype(np.float32)
    
    # Fill missing
    for col in ["spread_vs_sector_median", "spread_vs_sector_mean", 
                "spread_percentile_sector", "spread_percentile_all"]:
        df[col] = df[col].fillna(0).astype(np.float32)
    
    return df


# --- Cross-sectional transforms ---

def _winsorize_xsec(df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Per-date winsorization."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            q = df.groupby("date")[c].transform(
                lambda s: pd.Series(np.clip(s, s.quantile(lower), s.quantile(upper)), index=s.index)
            )
            df[c] = q
    return df

def _zscore_xsec(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-date z-score normalization."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            mu = df.groupby("date")[c].transform("mean")
            sd = df.groupby("date")[c].transform("std").replace(0.0, np.nan)
            df[c + "_z"] = (df[c] - mu) / sd
            df[c + "_z"] = df[c + "_z"].fillna(0.0)
    return df

def _zscore_ts_rolling_past_only(df: pd.DataFrame, cols: list[str], window: int = 252) -> pd.DataFrame:
    """Per-asset rolling z-score using only past data."""
    df = df.copy()
    df = df.sort_values(["debenture_id", "date"])
    g = df.groupby("debenture_id", group_keys=False)
    for c in cols:
        if c in df.columns:
            mu = g[c].apply(lambda s: s.rolling(window, min_periods=20).mean().shift(1))
            sd = g[c].apply(lambda s: s.rolling(window, min_periods=20).std(ddof=0).shift(1))
            z = (df[c] - mu) / sd
            df[c + f"_z{window}"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# --- SGS/BCB fetch ---

def _sgs_fetch(code: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch SGS series from BCB."""
    if not _HAS_REQUESTS:
        return pd.DataFrame(columns=["date", "value"])

    out = []
    cur = start.normalize()
    while cur <= end:
        stop = min(cur + pd.DateOffset(years=10) - pd.Timedelta(days=1), end)
        params = {"formato": "json",
                  "dataInicial": cur.strftime("%d/%m/%Y"),
                  "dataFinal": stop.strftime("%d/%m/%Y")}
        try:
            r = requests.get(_SGS_BASE.format(code=code), params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data:
                df = pd.DataFrame(data)
                df["date"] = _to_datetime_ser(df["data"])
                df["value"] = _safe_float(df["valor"])
                out.append(df[["date", "value"]].dropna())
        except Exception:
            pass
        cur = stop + pd.Timedelta(days=1)

    if not out:
        return pd.DataFrame(columns=["date", "value"])
    res = pd.concat(out, ignore_index=True).dropna().drop_duplicates("date").sort_values("date")
    return res


def _bcb_cached_series(code: int) -> pd.DataFrame:
    """Fetch BCB SGS series with caching."""
    path = os.path.join(CACHE_DIR, f"bcb_sgs_{code}.parquet")
    cached: Optional[pd.DataFrame] = None
    if os.path.exists(path):
        try:
            cached = pd.read_parquet(path)
        except Exception:
            cached = None

    today = pd.Timestamp("today").normalize()
    if cached is None or cached.empty:
        start = pd.Timestamp("2000-01-01")
        out = _sgs_fetch(code, start, today)
    else:
        last = pd.to_datetime(cached["date"]).max()
        if pd.isna(last) or last >= today:
            return cached.dropna().sort_values("date").reset_index(drop=True)
        out = pd.concat([cached, _sgs_fetch(code, last + pd.Timedelta(days=1), today)],
                        ignore_index=True)

    if not out.empty:
        out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        try:
            out.to_parquet(path, index=False)
        except Exception:
            pass
    return out


def _compute_daily_cdi(cdi_df: pd.DataFrame) -> pd.DataFrame:
    """Convert % a.a. to daily accrual."""
    if cdi_df is None or cdi_df.empty:
        return pd.DataFrame(columns=["date", "cdi_daily"])
    out = cdi_df.copy()
    out["cdi_daily"] = (1.0 + out["value"] / 100.0) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    return out[["date", "cdi_daily"]]


# --- Panel Building ---

@dataclass
class UniversePaths:
    data_dir: str
    universe: str
    @property
    def pattern(self) -> str:
        return f"idex_{self.universe}*.xls*"

def load_idex_folder(paths: UniversePaths) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(paths.data_dir, paths.pattern)))
    if not files:
        print(f"[WARN] No files for '{paths.universe}' with pattern {paths.pattern}")
        return pd.DataFrame()

    dfs = []
    for fp in files:
        try:
            raw = _load_xls(fp)
            dfs.append(_normalize_columns(raw, paths.universe))
        except Exception as e:
            print(f"[WARN] Failed reading {os.path.basename(fp)}: {e}")
    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["date", "debenture_id"]).sort_values(["date", "debenture_id"])
    return df


def _build_complete_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build complete panel from raw data."""
    if raw_df.empty:
        return raw_df

    if "index_weight" not in raw_df.columns:
        raw_df["index_weight"] = np.nan

    out = raw_df.sort_values(["debenture_id", "date"]).copy()

    # Forward-fill per debenture
    ffill_cols = [c for c in ["sector", "issuer", "callable", "index_level", "spread"] if c in out.columns]
    for c in [c for c in ffill_cols if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]:
        out[c] = out.groupby("debenture_id", sort=False)[c].ffill()
    for c in [c for c in ffill_cols if c in out.columns and not pd.api.types.is_numeric_dtype(out[c])]:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            if "Unknown" not in out[c].cat.categories:
                out[c] = out[c].cat.add_categories("Unknown")
            out[c] = out.groupby("debenture_id", sort=False)[c].ffill().fillna("Unknown")
        else:
            out[c] = out.groupby("debenture_id", sort=False)[c].ffill()

    # Active flag
    if "index_weight" in out.columns:
        out["active"] = (out["index_weight"].fillna(0.0) > 0.0).astype("int8")
    elif "active" in out.columns:
        out["active"] = out["active"].fillna(0).astype("int8")
    else:
        out["active"] = np.int8(0)

    return out


def _compute_per_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset TOTAL returns from weighted components.
    CRITICAL: Returns are already total (include CDI).
    """
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    if not {"weighted_mtm", "weighted_carry", "index_weight"}.issubset(df.columns):
        # If components not available, use return if exists
        if "return" not in df.columns:
            df["return"] = 0.0
        return df

    # Extract components
    w = df["index_weight"].to_numpy(dtype=np.float32)
    mtm = df["weighted_mtm"].to_numpy(dtype=np.float32)
    carry = df["weighted_carry"].to_numpy(dtype=np.float32)
    weighted_total = mtm + carry  # Total weighted return
    
    active = (w > 0.0)
    
    # Calculate per-asset total return
    total_ret = np.zeros_like(w, dtype=np.float32)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        total_ret[active] = weighted_total[active] / np.maximum(w[active], 1e-15)
    
    # Store as main return (already includes CDI)
    df["return"] = np.where(np.isfinite(total_ret), total_ret, 0.0).astype(np.float32)
    
    return df


def _attach_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute index return from index_value.
    CRITICAL: Index returns are already total (include CDI).
    """
    if "index_value" not in df.columns:
        print("[WARN] No index_value column found")
        df["index_return"] = 0.0
        return df
        
    # Extract index value series
    idx_value_series = (
        df[["date", "index_value"]]
        .dropna()
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")["index_value"]
        .sort_index()
    )
    
    if idx_value_series.empty:
        df["index_return"] = 0.0
        return df
    
    # Calculate returns from index values
    idx_ret = idx_value_series.pct_change().fillna(0.0)
    
    # Sanity check
    daily_mean = idx_ret.mean()
    daily_std = idx_ret.std()
    
    if abs(daily_mean) > 0.01:
        print(f"[WARN] Index daily returns seem high: mean={daily_mean*100:.4f}%")
    if daily_std > 0.05:
        print(f"[WARN] Index daily volatility seems high: std={daily_std*100:.2f}%")
    
    # Merge index returns
    df = df.merge(
        idx_ret.rename("index_return").reset_index(), 
        on="date", 
        how="left"
    )
    df["index_return"] = df["index_return"].fillna(0.0).astype(np.float32)
    
    # Preserve index level
    if "index_level" in df.columns:
        # Forward fill index level
        index_level_series = (
            df[["date", "index_level"]]
            .dropna()
            .drop_duplicates(subset=["date"], keep="first")
            .set_index("date")["index_level"]
            .sort_index()
        )
        if not index_level_series.empty:
            df = df.merge(
                index_level_series.rename("index_level_clean").reset_index(),
                on="date",
                how="left"
            )
            df["index_level"] = df["index_level_clean"].ffill().fillna(100.0)
            df = df.drop(columns=["index_level_clean"])
    else:
        df["index_level"] = 100.0
    
    print(f"[INFO] Index return stats: mean={df['index_return'].mean()*100:.4f}%, std={df['index_return'].std()*100:.4f}%")
    
    return df


def _attach_risk_free(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Attach risk_free daily rate from SGS.
    NOTE: Returns are already total, so RF is only for excess return calculation.
    """
    if df.empty:
        return df, {"source": "none"}

    rf_info = {"source": "sgs_cdi", "code": None}
    daily = None
    used_code = None
    
    # Try CDI sources
    for code in _CDI_CANDIDATES:
        ser = _bcb_cached_series(code)
        if not ser.empty:
            tmp = _compute_daily_cdi(ser)
            if not tmp.empty:
                daily = tmp
                used_code = code
                break

    # Fallback to SELIC
    if (daily is None) or daily.empty:
        rf_info["source"] = "sgs_selic"
        rf_info["code"] = str(SERIES_CODES["selic"])
        selic = _bcb_cached_series(SERIES_CODES["selic"])
        daily = _compute_daily_cdi(selic)

    if daily is None or daily.empty:
        raise RuntimeError("Could not fetch CDI/SELIC from SGS. Check internet/BCB availability.")

    if used_code is not None:
        rf_info["code"] = str(used_code)

    # Align to panel dates
    rf = daily.set_index("date").sort_index()
    rf = rf[~rf.index.duplicated(keep="last")]
    
    panel_dates = pd.DatetimeIndex(df["date"].dropna().sort_values().unique())
    rf_aligned = pd.DataFrame(index=panel_dates, columns=["risk_free"])
    rf_aligned = rf_aligned.merge(rf, left_index=True, right_index=True, how="left")
    rf_aligned["cdi_daily"] = rf_aligned["cdi_daily"].ffill().fillna(0.0)
    
    # Merge risk-free rate
    df = df.merge(
        rf_aligned[["cdi_daily"]].rename(columns={"cdi_daily": "risk_free"}).reset_index(),
        left_on="date", right_on="index", how="left"
    ).drop(columns=["index"])
    
    df["risk_free"] = df["risk_free"].fillna(0.0).astype(np.float32)
    
    print(f"[INFO] Risk-free rate attached: mean={df['risk_free'].mean()*100:.4f}%")
    
    return df, rf_info


def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic features: sector id, TTM, excess return."""
    if "sector" in df.columns:
        df["sector_id"] = df["sector"].cat.codes.astype("int16")
    else:
        df["sector_id"] = -1

    # Time to maturity proxy
    ttm_days = df.groupby("debenture_id", sort=False)["date"].transform(lambda x: (x.max() - x).dt.days)
    df["time_to_maturity"] = (ttm_days / TRADING_DAYS_PER_YEAR).astype("float32")

    # Rank TTM within sector
    if {"date", "sector_id", "time_to_maturity", "active"}.issubset(df.columns):
        def _rank01(x: pd.Series) -> pd.Series:
            if x.size <= 1:
                return pd.Series(np.zeros_like(x, dtype=np.float32), index=x.index)
            r = (x.rank(method="average") - 1) / max(float(x.size - 1), 1.0)
            return r.astype(np.float32)
        mask = df["active"] > 0
        ttm_rank = df.loc[mask].groupby(["date", "sector_id"], sort=False)["time_to_maturity"].transform(_rank01)
        df["ttm_rank"] = 0.0
        df.loc[mask, "ttm_rank"] = ttm_rank.astype(np.float32)

    # Excess return (total return - risk free)
    df["excess_return"] = (df["return"] - df["risk_free"]).astype("float32")
    
    return df


def _sector_signals(df: pd.DataFrame, momentum_window: int = 63) -> pd.DataFrame:
    """Add sector-level signals."""
    out = df.sort_values(["date", "sector"]).copy()
    
    # Sector weight
    if "sector" in out.columns and "index_weight" in out.columns:
        sw = (out.groupby(["date", "sector"], sort=False, observed=False)["index_weight"]
                 .sum(min_count=1)
                 .rename("sector_weight_index")
                 .reset_index())
        out = out.merge(sw, on=["date", "sector"], how="left")
    else:
        out["sector_weight_index"] = np.nan

    # Sector spread average
    if "spread" in out.columns:
        if "index_weight" in out.columns:
            tmp = out[["date", "sector", "spread", "index_weight"]].copy()
            tmp["w"] = _safe_float(tmp["index_weight"]).fillna(0.0).astype(np.float32)
            tmp["ws"] = _safe_float(tmp["spread"]).astype(np.float32) * tmp["w"]
            agg = tmp.groupby(["date", "sector"], sort=False, observed=False).agg(
                ws_sum=("ws","sum"), 
                w_sum=("w","sum")
            ).reset_index()
            agg["sector_spread_avg"] = np.where(
                agg["w_sum"]>0, 
                agg["ws_sum"]/agg["w_sum"], 
                np.nan
            ).astype(np.float32)
            out = out.merge(agg[["date","sector","sector_spread_avg"]], on=["date","sector"], how="left")
        else:
            avg = (out.groupby(["date", "sector"], sort=False, observed=False)["spread"]
                     .mean()
                     .rename("sector_spread_avg")
                     .reset_index())
            out = out.merge(avg, on=["date","sector"], how="left")
    else:
        out["sector_spread_avg"] = np.nan

    out["sector_spread"] = _safe_float(out["sector_spread_avg"]).fillna(0.0).astype(np.float32)

    # Sector momentum
    if "sector" in out.columns:
        out = out.sort_values(["sector","date"]).copy()
        g = out.groupby("sector", sort=False, observed=False)
        roll_mean = g["sector_spread"].transform(lambda s: s.rolling(momentum_window, min_periods=5).mean())
        out["sector_momentum"] = (_safe_float(out["sector_spread"]) - _safe_float(roll_mean)).astype(np.float32)
        out = out.sort_values(["debenture_id","date"]).copy()
    else:
        out["sector_momentum"] = 0.0

    # Clean up
    for c in ["sector_weight_index","sector_spread","sector_momentum","sector_spread_avg"]:
        if c in out.columns:
            out[c] = _safe_float(out[c]).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    return out


# --- ANBIMA sector curves (same as before, omitted for brevity) ---
# [Include all the ANBIMA functions: _phi_ns2, _wls_solve, _grid_lambdas, etc.]
# [I'm omitting these for space but they remain unchanged]

def _final_tidy_types(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up and set final types."""
    # Keep required and optional columns
    keep = set(REQUIRED_COLS)
    keep |= {c for c in OPTIONAL_KEEP if c in df.columns}
    pattern_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_z") or "_z" in c]
    keep |= set(pattern_cols)

    cols = [c for c in df.columns if c in keep or c in ("date", "debenture_id")]
    df = df[cols].copy()

    # Convert to float32
    float_cols: Iterable[str] = [
        c for c in df.columns
        if c not in ("date", "debenture_id", "sector_id", "active", "rating", "issuer", "sector")
    ]
    for c in float_cols:
        df[c] = _safe_float(df[c]).astype("float32")

    # Integer columns
    if "sector_id" in df.columns:
        df["sector_id"] = _safe_float(df["sector_id"]).fillna(-1).astype("int16")
    if "active" in df.columns:
        df["active"] = _safe_float(df["active"]).fillna(0).astype("int8")

    # Categories
    for c in ("issuer", "sector", "rating"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Set index
    df = df.sort_values(["date", "debenture_id"]).set_index(["date", "debenture_id"])
    return df


# --- Main processing function ---

def process_universe(universe: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
    """Main processing function."""
    paths = UniversePaths(data_dir=data_dir, universe=universe.lower())
    raw = load_idex_folder(paths)
    if raw.empty:
        print(f"[WARN] Universe '{universe}' is empty.")
        return None

    print(f"[INFO] Processing {universe} universe...")
    
    # Build panel
    pnl = _build_complete_panel(raw)
    
    # Compute returns (already total)
    pnl = _compute_per_asset_returns(pnl)
    
    # Attach benchmark (already total)
    pnl = _attach_benchmark(pnl)
    
    # Attach risk-free (for excess return calculation only)
    pnl, rf_info = _attach_risk_free(pnl)
    
    # Basic features
    pnl = _basic_features(pnl)
    
    # Sector signals
    pnl = _sector_signals(pnl, momentum_window=63)
    
    # Feature engineering
    print("[INFO] Computing momentum features...")
    pnl = _compute_momentum_features(pnl)
    
    print("[INFO] Computing volatility features...")
    pnl = _compute_volatility_features(pnl)
    
    print("[INFO] Computing relative value features...")
    pnl = _compute_relative_value_features(pnl)
    
    print("[INFO] Computing duration risk features...")
    pnl = _compute_duration_risk_features(pnl)
    
    print("[INFO] Computing microstructure features...")
    pnl = _compute_microstructure_features(pnl)
    
    print("[INFO] Computing carry features...")
    pnl = _compute_carry_features(pnl)
    
    print("[INFO] Computing spread dynamics...")
    pnl = _compute_spread_dynamics(pnl)
    
    print("[INFO] Computing risk-adjusted features...")
    pnl = _compute_risk_adjusted_features(pnl)
    
    # [Include ANBIMA sector curves if needed]
    # print("[INFO] Fitting sector curves...")
    # pnl = _apply_sector_curves_anbima(pnl, use_anchor=True)
    
    # Cross-sectional normalization
    cont_features = [
        "spread", "duration", "time_to_maturity", "index_weight",
        "return", "excess_return",
        *[f"momentum_{w}d" for w in MOMENTUM_WINDOWS],
        *[f"reversal_{w}d" for w in MOMENTUM_WINDOWS],
        *[f"volatility_{w}d" for w in VOLATILITY_WINDOWS],
        *[f"spread_vol_{w}d" for w in VOLATILITY_WINDOWS],
        "spread_vs_sector_median", "spread_vs_sector_mean",
        "spread_percentile_sector", "spread_percentile_all",
        "duration_change", "duration_vol", "duration_spread_interaction",
        "modified_duration_proxy", "convexity_proxy",
        "liquidity_score", "weight_momentum", "weight_volatility",
        "carry_spread_ratio", "carry_momentum", "carry_vol",
        "spread_momentum_5d", "spread_momentum_20d",
        "spread_mean_reversion", "spread_acceleration",
        "sharpe_5d", "sharpe_20d", "sharpe_60d", "information_ratio_20d",
    ]
    cont_features = [c for c in cont_features if c in pnl.columns]
    
    # Winsorize and z-score on active assets
    active_mask = (pnl["active"] == 1)
    
    print("[INFO] Winsorizing features...")
    pnl_active = pnl.loc[active_mask, ["date", "debenture_id"] + cont_features].copy()
    pnl_active = _winsorize_xsec(pnl_active, cont_features, lower=0.005, upper=0.995)
    
    print("[INFO] Computing cross-sectional z-scores...")
    pnl_active = _zscore_xsec(pnl_active, cont_features)
    
    zcols = [c + "_z" for c in cont_features]
    pnl = pnl.merge(pnl_active[["date", "debenture_id"] + zcols], on=["date", "debenture_id"], how="left")
    for zc in zcols:
        pnl[zc] = pnl[zc].fillna(0.0)
    
    # Rolling z-scores
    print("[INFO] Computing rolling z-scores...")
    pnl = _zscore_ts_rolling_past_only(pnl, cont_features, window=252)
    
    # Create lagged features
    print("[INFO] Creating lagged features...")
    
    lag_candidates = set(cont_features + zcols)
    lag_candidates.update([
        "ttm_rank", "index_weight", "return", "risk_free", "index_return", 
        "excess_return", "index_level"
    ])
    lag_candidates.update([c for c in pnl.columns if "_z252" in c])
    
    pnl = pnl.sort_values(["debenture_id", "date"])
    g = pnl.groupby("debenture_id", sort=False)
    
    for c in sorted(lag_candidates):
        if c in pnl.columns:
            pnl[c + "_lag1"] = g[c].shift(1).astype("float32")
            pnl[c + "_lag1"] = pnl[c + "_lag1"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    
    pnl = pnl.sort_values(["date", "debenture_id"])
    
    # Final cleanup
    final = _final_tidy_types(pnl)
    
    # Save risk-free series
    res_dir = os.path.join("results", universe.lower())
    os.makedirs(res_dir, exist_ok=True)
    rf_used = (
        final.reset_index()[["date", "risk_free"]]
             .drop_duplicates(subset=["date"])
             .sort_values("date")
    )
    rf_used.to_csv(os.path.join(res_dir, "risk_free_used.csv"), index=False)
    
    # Save processed panel
    out_path = os.path.join(data_dir, f"{universe.lower()}_processed.pkl")
    final.to_pickle(out_path)
    print(f"[OK] Saved {out_path}")
    
    # Data validation
    print("\n[DATA VALIDATION]")
    print(f"Return mean: {final['return'].mean()*100:.4f}%")
    print(f"Risk-free mean: {final['risk_free'].mean()*100:.4f}%")
    print(f"Index return mean: {final['index_return'].mean()*100:.4f}%")
    print(f"Excess return mean: {final['excess_return'].mean()*100:.4f}%")
    
    # Save metadata
    info_path = os.path.join(data_dir, "data_info.json")
    try:
        info = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "universe": universe,
            "risk_free": rf_info,
            "rows": len(final),
            "assets": final.index.get_level_values('debenture_id').nunique(),
            "dates": {
                "start": str(final.index.get_level_values('date').min().date()),
                "end": str(final.index.get_level_values('date').max().date())
            },
            "features": len([c for c in final.columns if c.endswith('_lag1')]),
            "return_structure": "Returns are TOTAL (spread + CDI included)",
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    
    print(f"\n{universe.upper()} processing complete!")
    print(f"Shape: {final.shape}")
    print(f"Features: {len([c for c in final.columns if c.endswith('_lag1')])} lagged")
    
    return final


def prepare_data(data_dir: str = "data"):
    """Process CDI universe."""
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DATA PREPARATION - CORRECTED RETURNS")
    print("="*60 + "\n")
    
    cdi = process_universe("cdi", data_dir)
    
    if cdi is not None:
        print(f"\n[SUCCESS] CDI universe processed")
    
    return cdi


if __name__ == "__main__":
    prepare_data()