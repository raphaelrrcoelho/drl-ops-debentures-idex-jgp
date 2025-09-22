# data_final.py
"""
Enhanced Data preparation for CDI debenture universe with advanced feature engineering
=====================================================================================

Key Enhancements:
1. Momentum/Reversal features at multiple horizons (1, 5, 20, 60 days)
2. Volatility features (rolling std at 5, 20, 60 days) for returns and spreads
3. Relative value features (spread vs sector median/mean, percentile ranks)
4. Duration-based risk measures (duration changes, duration×spread interactions)
5. Microstructure proxies (using index_weight as liquidity proxy)
6. Enhanced carry features (carry momentum, carry/spread ratio)
7. Spread dynamics (spread momentum, mean reversion, term structure)
8. Cross-sectional features within sectors
9. Risk-adjusted momentum (Sharpe-like ratios over rolling windows)

All features maintain strict causality with lag-1 versions for policy inputs.
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
    "risk_free", "index_return", "time_to_maturity", "index_level"
]

OPTIONAL_KEEP = [
    "rating", "index_weight", "issuer", "sector",
    # ANBIMA sector curve features
    "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
    "sector_fitted_spread", "spread_residual_ns",
    "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
    # New enhanced features (will be added below)
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
    pct_cols = ["index_weight", "weighted_mtm", "weighted_return", "weighted_carry"]
    for c in pct_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            df[c] = _safe_float(df[c]) / 100.0

    if "spread" in df.columns:
        if universe.lower() == "infra":
            if df["spread"].dtype == object:
                df["spread"] = df["spread"].astype(str).str.replace(",", ".", regex=False)
            df["spread"] = _safe_float(df["spread"]) / 10000.0
        else:
            if df["spread"].dtype == object:
                df["spread"] = df["spread"].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            df["spread"] = _safe_float(df["spread"]) / 100.0

    for c in ("duration", "index_level"):
        if c in df.columns:
            df[c] = _safe_float(df[c])

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
    Compute momentum and reversal features at multiple horizons.
    Momentum is cumulative return over window, reversal is -1 * momentum.
    """
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    for w in windows:
        # Use transform to maintain index alignment
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
        # Return volatility (annualized)
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
    """
    Compute duration-based risk measures.
    """
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Duration change
    df["duration_change"] = df.groupby("debenture_id", sort=False)["duration"].diff().astype(np.float32)
    
    # Duration volatility - use transform
    df["duration_vol"] = df.groupby("debenture_id", sort=False)["duration"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    ).astype(np.float32)
    
    # Duration × Spread interaction (risk measure)
    df["duration_spread_interaction"] = (
        df["duration"] * df["spread"]
    ).astype(np.float32)
    
    # Modified duration proxy (using available data)
    df["modified_duration_proxy"] = (
        df["duration"] / (1 + df["spread"])
    ).astype(np.float32)
    
    # Convexity proxy (using duration squared as simple proxy)
    df["convexity_proxy"] = (df["duration"] ** 2 / 100).astype(np.float32)
    
    return df


def _compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidity and microstructure features using index_weight as proxy.
    """
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
    
    # Weight momentum (change in index weight)
    df["weight_momentum"] = df.groupby("debenture_id", sort=False)["index_weight"].diff().astype(np.float32)
    
    # Weight volatility - use transform
    df["weight_volatility"] = df.groupby("debenture_id", sort=False)["index_weight"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    ).astype(np.float32)
    
    return df


def _compute_carry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced carry features using weighted_carry column.
    """
    if "weighted_carry" not in df.columns:
        # If no carry data, create synthetic carry from spread
        df["carry_proxy"] = df["spread"] * df.get("index_weight", 1.0)
    else:
        # Normalize carry by index weight to get per-bond carry
        df["carry_proxy"] = np.where(
            df["index_weight"] > 0,
            df["weighted_carry"] / df["index_weight"],
            df["weighted_carry"]
        )
    
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Carry/Spread ratio (attractiveness measure)
    df["carry_spread_ratio"] = np.where(
        df["spread"] > 0,
        df["carry_proxy"] / df["spread"],
        0.0
    ).astype(np.float32)
    
    # Carry momentum
    df["carry_momentum"] = df.groupby("debenture_id", sort=False)["carry_proxy"].diff().astype(np.float32)
    
    # Carry volatility - use transform
    df["carry_vol"] = df.groupby("debenture_id", sort=False)["carry_proxy"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    ).astype(np.float32)
    
    return df


def _compute_spread_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spread momentum, mean reversion, and acceleration.
    """
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Spread momentum at different horizons
    df["spread_momentum_5d"] = df.groupby("debenture_id", sort=False)["spread"].diff(5).astype(np.float32)
    df["spread_momentum_20d"] = df.groupby("debenture_id", sort=False)["spread"].diff(20).astype(np.float32)
    
    # Mean reversion signal (current vs 60-day mean) - use transform
    spread_ma60 = df.groupby("debenture_id", sort=False)["spread"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    df["spread_mean_reversion"] = (
        df["spread"] - spread_ma60
    ).astype(np.float32)
    
    # Spread acceleration (second derivative)
    spread_diff = df.groupby("debenture_id", sort=False)["spread"].diff()
    df["spread_acceleration"] = spread_diff.groupby(df["debenture_id"]).diff().astype(np.float32)
    
    return df


def _compute_risk_adjusted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Sharpe ratios and information ratios over rolling windows.
    """
    df = df.sort_values(["debenture_id", "date"]).copy()
    
    # Ensure excess_return exists
    if "excess_return" not in df.columns:
        df["excess_return"] = df["return"] - df.get("risk_free", 0.0)
    
    # Sharpe ratios at different horizons using transform only
    for w in [5, 20, 60]:
        # Calculate rolling mean and std using transform
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
    
    # Information ratio (vs index)
    # Create active return column temporarily
    df["active_return_temp"] = df["return"] - df["index_return"]
    
    mean_active = df.groupby("debenture_id", sort=False)["active_return_temp"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    std_active = df.groupby("debenture_id", sort=False)["active_return_temp"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    
    df["information_ratio_20d"] = np.where(
        std_active > 0,
        mean_active / std_active * np.sqrt(TRADING_DAYS_PER_YEAR),
        0.0
    ).astype(np.float32)
    
    # Clean up temporary column
    df = df.drop(columns=["active_return_temp"], errors='ignore')
    
    return df


def _compute_relative_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spread relative to sector and overall market.
    """
    df = df.copy()
    
    # Sector-relative features
    for (date, sector), group in df[df["active"] > 0].groupby(["date", "sector_id"]):
        mask = (df["date"] == date) & (df["sector_id"] == sector)
        
        # Spread vs sector median/mean
        sector_median = group["spread"].median()
        sector_mean = group["spread"].mean()
        
        df.loc[mask, "spread_vs_sector_median"] = (
            df.loc[mask, "spread"] - sector_median
        ).astype(np.float32)
        
        df.loc[mask, "spread_vs_sector_mean"] = (
            df.loc[mask, "spread"] - sector_mean
        ).astype(np.float32)
        
        # Percentile within sector
        df.loc[mask, "spread_percentile_sector"] = (
            df.loc[mask, "spread"].rank(pct=True)
        ).astype(np.float32)
    
    # Overall market percentile (per date)
    for date, group in df[df["active"] > 0].groupby("date"):
        mask = df["date"] == date
        df.loc[mask, "spread_percentile_all"] = (
            df.loc[mask, "spread"].rank(pct=True)
        ).astype(np.float32)
    
    # Fill missing values
    for col in ["spread_vs_sector_median", "spread_vs_sector_mean", 
                "spread_percentile_sector", "spread_percentile_all"]:
        df[col] = df[col].fillna(0).astype(np.float32)
    
    return df




# --- X-sec transforms (unchanged from original) ---

def _winsorize_xsec(df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Per-date winsorization: uses only cross-sectional info from the *same day*."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            q = df.groupby("date")[c].transform(
                lambda s: pd.Series(np.clip(s, s.quantile(lower), s.quantile(upper)), index=s.index)
            )
            df[c] = q
    return df

def _zscore_xsec(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-date zscore: (x - mean_t)/std_t using *only* same-day cross-section."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            mu = df.groupby("date")[c].transform("mean")
            sd = df.groupby("date")[c].transform("std").replace(0.0, np.nan)
            df[c + "_z"] = (df[c] - mu) / sd
            df[c + "_z"] = df[c + "_z"].fillna(0.0)
    return df

def _zscore_ts_rolling_past_only(df: pd.DataFrame, cols: list[str], window: int = 252) -> pd.DataFrame:
    """
    Per-asset rolling z-score using only data *up to t-1*.
    """
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


# --- SGS/BCB fetch (unchanged) ---

def _sgs_fetch(code: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch SGS (JSON) in ≤10y chunks and return [date,value]."""
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
    """Fetch BCB SGS series with on-disk parquet cache."""
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


# --- I/O and Panel Building ---

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
    """Build (date, debenture_id) panel on Idex trading days."""
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

    # Active if weight > 0
    if "index_weight" in out.columns:
        out["active"] = (out["index_weight"].fillna(0.0) > 0.0).astype("int8")
    elif "active" in out.columns:
        out["active"] = out["active"].fillna(0).astype("int8")
    else:
        out["active"] = np.int8(0)

    return out


def _compute_per_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Approx per-asset daily return from weighted components if present."""
    df = df.sort_values(["debenture_id", "date"]).copy()
    if not {"weighted_mtm", "weighted_carry", "index_weight"}.issubset(df.columns):
        if "return" not in df.columns:
            df["return"] = 0.0
        return df

    w = df["index_weight"].to_numpy(dtype=np.float32)
    mtm = df["weighted_mtm"].to_numpy(dtype=np.float32)
    carry = df["weighted_carry"].to_numpy(dtype=np.float32)
    active = (w > 0.0)
    ret = np.zeros_like(w, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[active] = (mtm[active] + carry[active]) / np.maximum(w[active], 1e-12)
    df["return"] = np.where(np.isfinite(ret), ret, 0.0).astype(np.float32)
    return df


def _attach_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Compute index_return from index_level (date-level) and broadcast."""
    if "index_level" not in df.columns:
        df["index_level"] = np.nan
    idx_series = (
        df[["date", "index_level"]]
        .dropna()
        .drop_duplicates(subset=["date"], keep="first")
        .set_index("date")["index_level"]
        .sort_index()
    )
    idx_ret = idx_series.pct_change().fillna(0.0)
    df = df.merge(idx_ret.rename("index_return").reset_index(), on="date", how="left")
    df["index_return"] = df["index_return"].fillna(0.0).astype(np.float32)

    df = df.merge(idx_series.rename("_idxlvl").reset_index(), on="date", how="left")
    df["index_level"] = _safe_float(df["index_level"]).fillna(df.pop("_idxlvl")).astype(np.float32)
    return df


def _attach_risk_free(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Attach risk_free daily from SGS."""
    if df.empty:
        return df, {"source": "none"}

    rf_info = {"source": "sgs_cdi", "code": None}
    daily = None
    used_code = None
    for code in _CDI_CANDIDATES:
        ser = _bcb_cached_series(code)
        if not ser.empty:
            tmp = _compute_daily_cdi(ser)
            if not tmp.empty:
                daily = tmp
                used_code = code
                break

    if (daily is None) or daily.empty:
        rf_info["source"] = "sgs_selic"
        rf_info["code"] = str(SERIES_CODES["selic"])
        selic = _bcb_cached_series(SERIES_CODES["selic"])
        daily = _compute_daily_cdi(selic)

    if daily is None or daily.empty:
        raise RuntimeError("Could not fetch CDI/SELIC from SGS. Check internet/BCB availability.")

    if used_code is not None:
        rf_info["code"] = str(used_code)

    rf = daily.set_index("date").sort_index()
    rf = rf[~rf.index.duplicated(keep="last")]
    
    panel_dates = pd.DatetimeIndex(df["date"].dropna().sort_values().unique())
    rf_aligned = pd.DataFrame(index=panel_dates, columns=["risk_free"])
    rf_aligned = rf_aligned.merge(rf, left_index=True, right_index=True, how="left")
    rf_aligned["cdi_daily"] = rf_aligned["cdi_daily"].ffill().fillna(0.0)
    
    df = df.merge(
        rf_aligned[["cdi_daily"]].rename(columns={"cdi_daily": "risk_free"}).reset_index(),
        left_on="date", right_on="index", how="left"
    ).drop(columns=["index"])
    
    df["risk_free"] = df["risk_free"].fillna(0.0).astype(np.float32)
    
    return df, rf_info


def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sector id, TTM proxy, within-sector rank, excess return."""
    if "sector" in df.columns:
        df["sector_id"] = df["sector"].cat.codes.astype("int16")
    else:
        df["sector_id"] = -1

    # TTM proxy in years
    ttm_days = df.groupby("debenture_id", sort=False)["date"].transform(lambda x: (x.max() - x).dt.days)
    df["time_to_maturity"] = (ttm_days / TRADING_DAYS_PER_YEAR).astype("float32")

    # Rank TTM within (date, sector) for active names
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

    if "return" in df.columns:
        if "risk_free" in df.columns:
            df["excess_return"] = (df["return"] - df["risk_free"]).astype("float32")
        else:
            df["excess_return"] = np.float32(0.0)
    return df


# --- Sector signals and curves (mostly unchanged) ---

def _sector_signals(df: pd.DataFrame, momentum_window: int = 63) -> pd.DataFrame:
    """Add sector-level signals per (date, sector) and broadcast to each bond row."""
    out = df.sort_values(["date", "sector"]).copy()
    
    if "sector" in out.columns and "index_weight" in out.columns:
        sw = (out.groupby(["date", "sector"], sort=False, observed=False)["index_weight"]
                 .sum(min_count=1)
                 .rename("sector_weight_index")
                 .reset_index())
        out = out.merge(sw, on=["date", "sector"], how="left")
    else:
        out["sector_weight_index"] = np.nan

    if "spread" in out.columns:
        if "index_weight" in out.columns:
            tmp = out[["date", "sector", "spread", "index_weight"]].copy()
            tmp["w"] = _safe_float(tmp["index_weight"]).fillna(0.0).astype(np.float32)
            tmp["ws"] = _safe_float(tmp["spread"]).astype(np.float32) * tmp["w"]
            agg = tmp.groupby(["date", "sector"], sort=False, observed=False).agg(ws_sum=("ws","sum"), w_sum=("w","sum")).reset_index()
            agg["sector_spread_avg"] = np.where(agg["w_sum"]>0, agg["ws_sum"]/agg["w_sum"], np.nan).astype(np.float32)
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

    if "sector" in out.columns:
        out = out.sort_values(["sector","date"]).copy()
        g = out.groupby("sector", sort=False, observed=False)
        roll_mean = g["sector_spread"].transform(lambda s: s.rolling(momentum_window, min_periods=5).mean())
        out["sector_momentum"] = (_safe_float(out["sector_spread"]) - _safe_float(roll_mean)).astype(np.float32)
        out = out.sort_values(["debenture_id","date"]).copy()
    else:
        out["sector_momentum"] = 0.0

    for c in ["sector_weight_index","sector_spread","sector_momentum","sector_spread_avg"]:
        if c in out.columns:
            out[c] = _safe_float(out[c]).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    return out


# --- ANBIMA sector curves (unchanged) ---

def _phi_ns2(tau: np.ndarray, lam: float) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    lam = float(max(lam, 1e-8))
    x = lam * np.maximum(tau, 0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        ratio = np.where(x > 1e-6, (1.0 - np.exp(-x)) / x, 1.0 - 0.5 * x + (x * x) / 6.0)
    return ratio


def _wls_solve(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    w = np.maximum(w, 0.0)
    sw = np.sqrt(w + 1e-12)
    Xw = X * sw[:, None]
    yw = y * sw
    XtX = Xw.T @ Xw + ridge * np.eye(X.shape[1])
    Xty = Xw.T @ yw
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def _grid_lambdas() -> np.ndarray:
    return np.exp(np.linspace(np.log(0.05), np.log(5.0), 25))


def _prepare_sector_day(df_day: pd.DataFrame, use_anchor: bool) -> pd.DataFrame:
    g = df_day.copy()
    if "time_to_maturity" not in g.columns or g["time_to_maturity"].isna().all():
        g["time_to_maturity"] = g.get("duration", pd.Series(index=g.index, data=np.nan)).astype(float)
    g = g[(g["time_to_maturity"] >= BUSINESS_DAYS_SHORT_DROP / TRADING_DAYS_PER_YEAR)]
    
    if use_anchor and not g.empty:
        rows: List[dict] = []
        for sid, sub in g.groupby("sector_id", sort=False):
            if sub.empty:
                continue
            tau_syn = TAU_SHORT_ANCHOR
            sub_sorted = sub.sort_values("time_to_maturity")
            k = max(1, int(0.25 * len(sub_sorted)))
            spr_syn = float(sub_sorted["spread"].iloc[:k].median())
            rows.append({
                "date": sub_sorted["date"].iloc[0],
                "debenture_id": f"__ANCHOR_SECTOR_{int(sid)}",
                "sector_id": int(sid),
                "spread": spr_syn,
                "duration": float(min(0.1, tau_syn)),
                "time_to_maturity": tau_syn,
                "active": 1,
                "_anchor": 1,
            })
        if rows:
            g = pd.concat([g, pd.DataFrame(rows)], ignore_index=True)
    else:
        g["_anchor"] = 0
    return g


def _outlier_mask_iqr(y: np.ndarray, k: float = 3.0) -> np.ndarray:
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (y >= lo) & (y <= hi)


def _fit_ns_sector_day(df_day: pd.DataFrame, use_anchor: bool = True) -> Dict[str, object]:
    """Joint WLS across sectors for the day: y = β0_sector + β1 φ(τ,λ)."""
    g = _prepare_sector_day(df_day, use_anchor=use_anchor)
    if g.empty:
        return {"beta0_sector": {}, "beta1": np.nan, "lambda": np.nan,
                "fitted": pd.Series(index=df_day.index, dtype=float), "used_mask": np.zeros(len(df_day), dtype=bool)}

    keep = _outlier_mask_iqr(g["spread"].to_numpy(dtype=float), k=3.0)
    g1 = g.loc[keep].copy()
    if g1.empty:
        g1 = g.copy()

    sectors = g1["sector_id"].astype(int).to_numpy()
    sec_vals = np.unique(sectors)
    sec_to_col = {s: i for i, s in enumerate(sec_vals)}
    S = len(sec_vals)

    tau = g1["time_to_maturity"].to_numpy(dtype=float)
    y = g1["spread"].to_numpy(dtype=float)
    w = np.clip(g1.get("duration", pd.Series(index=g1.index, data=np.nan)).to_numpy(dtype=float), 1e-4, 10.0)
    w[np.isnan(w)] = 1e-3

    best = (np.inf, None, None, None)
    for lam in _grid_lambdas():
        phi = _phi_ns2(tau, lam)
        X = np.empty((len(y), 1 + S), dtype=float)
        X[:, 0] = phi
        X[:, 1:] = 0.0
        for i, s in enumerate(sectors):
            X[i, 1 + sec_to_col[int(s)]] = 1.0

        beta = _wls_solve(X, y, w, ridge=1e-8)
        yhat = X @ beta
        sse = float(np.sum(w * (y - yhat) ** 2))
        if sse < best[0]:
            best = (sse, beta, lam, yhat)

    _, beta, lam_best, yhat = best
    if beta is None:
        return {"beta0_sector": {}, "beta1": np.nan, "lambda": np.nan,
                "fitted": pd.Series(index=df_day.index, dtype=float), "used_mask": np.zeros(len(df_day), dtype=bool)}

    beta1 = float(beta[0])
    beta0_sector = {int(s): float(beta[1 + j]) for s, j in sec_to_col.items()}

    # Robust pass
    resid = y - yhat
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    z = np.abs(resid - np.median(resid)) / mad
    keep2 = z <= 6.0
    if not keep2.all() and keep2.sum() >= max(3, S + 1):
        g2 = g1.loc[keep2].copy()
        sectors2 = g2["sector_id"].astype(int).to_numpy()
        sec_vals2 = np.unique(sectors2)
        sec_to_col2 = {s: i for i, s in enumerate(sec_vals2)}
        S2 = len(sec_vals2)
        tau2 = g2["time_to_maturity"].to_numpy(dtype=float)
        y2 = g2["spread"].to_numpy(dtype=float)
        w2 = np.clip(g2.get("duration", pd.Series(index=g2.index, data=np.nan)).to_numpy(dtype=float), 1e-4, 10.0)
        w2[np.isnan(w2)] = 1e-3
        phi2 = _phi_ns2(tau2, lam_best)
        X2 = np.empty((len(y2), 1 + S2), dtype=float)
        X2[:, 0] = phi2
        X2[:, 1:] = 0.0
        for i, s in enumerate(sectors2):
            X2[i, 1 + sec_to_col2[int(s)]] = 1.0
        beta2 = _wls_solve(X2, y2, w2, ridge=1e-8)
        beta1 = float(beta2[0])
        beta0_sector = {int(s): float(beta2[1 + j]) for s, j in sec_to_col2.items()}

    # Fitted values
    full_tau = df_day["time_to_maturity"].to_numpy(dtype=float)
    full_sec = df_day["sector_id"].astype(int).to_numpy()
    full_phi = _phi_ns2(full_tau, lam_best)
    fitted_full = np.array([beta0_sector.get(int(s), np.nan) + beta1 * ph for s, ph in zip(full_sec, full_phi)], dtype=float)
    fitted_ser = pd.Series(fitted_full, index=df_day.index)

    used_mask = df_day.index.isin(g1.index)
    return {"beta0_sector": beta0_sector, "beta1": beta1, "lambda": float(lam_best),
            "fitted": fitted_ser, "used_mask": used_mask}


def _apply_sector_curves_anbima(panel: pd.DataFrame, use_anchor: bool = True) -> pd.DataFrame:
    if panel.empty:
        return panel

    out = panel.copy()
    out["sector_ns_beta0"] = np.float32(np.nan)
    out["ns_beta1_common"] = np.float32(np.nan)
    out["ns_lambda_common"] = np.float32(np.nan)
    out["sector_fitted_spread"] = np.float32(np.nan)
    out["spread_residual_ns"] = np.float32(np.nan)
    out["sector_ns_level_1y"] = np.float32(np.nan)
    out["sector_ns_level_3y"] = np.float32(np.nan)
    out["sector_ns_level_5y"] = np.float32(np.nan)

    dates = panel["date"].sort_values().unique()
    for d in dates:
        day_idx = panel["date"] == d
        df_day = panel.loc[day_idx, ["date", "sector_id", "spread", "duration", "time_to_maturity", "active"]].copy()
        df_day = df_day[(df_day["active"] > 0) & np.isfinite(df_day["spread"]) & np.isfinite(df_day["time_to_maturity"])]
        if df_day.empty:
            continue

        res = _fit_ns_sector_day(df_day, use_anchor=use_anchor)
        beta0 = res["beta0_sector"]; b1 = res["beta1"]; lam = res["lambda"]; fitted = res["fitted"]

        if np.isfinite(b1) and np.isfinite(lam):
            sec_ids = panel.loc[day_idx, "sector_id"].astype(int)
            out.loc[day_idx, "sector_ns_beta0"] = sec_ids.map(beta0).astype(np.float32)
            out.loc[day_idx, "ns_beta1_common"] = np.float32(b1)
            out.loc[day_idx, "ns_lambda_common"] = np.float32(lam)
            out.loc[fitted.index, "sector_fitted_spread"] = fitted.astype(np.float32)
            obs = panel.loc[fitted.index, "spread"].astype(float)
            out.loc[fitted.index, "spread_residual_ns"] = (obs - fitted).astype(np.float32)

            phi_1y, phi_3y, phi_5y = _phi_ns2(1.0, lam), _phi_ns2(3.0, lam), _phi_ns2(5.0, lam)
            b0_series = sec_ids.map(beta0).astype(float)
            out.loc[day_idx, "sector_ns_level_1y"] = (b0_series + b1 * phi_1y).astype(np.float32)
            out.loc[day_idx, "sector_ns_level_3y"] = (b0_series + b1 * phi_3y).astype(np.float32)
            out.loc[day_idx, "sector_ns_level_5y"] = (b0_series + b1 * phi_5y).astype(np.float32)

    return out


def _final_tidy_types(df: pd.DataFrame) -> pd.DataFrame:
    """Keep required, optional, and engineered features."""
    keep = set(REQUIRED_COLS) | {c for c in OPTIONAL_KEEP if c in df.columns}
    pattern_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_z") or "_z" in c]
    keep |= set(pattern_cols)

    cols = [c for c in df.columns if c in keep or c in ("date", "debenture_id")]
    df = df[cols].copy()

    float_cols: Iterable[str] = [
        c for c in df.columns
        if c not in ("date", "debenture_id", "sector_id", "active", "rating", "issuer", "sector")
    ]
    for c in float_cols:
        df[c] = _safe_float(df[c]).astype("float32")

    if "sector_id" in df.columns:
        df["sector_id"] = _safe_float(df["sector_id"]).fillna(-1).astype("int16")
    if "active" in df.columns:
        df["active"] = _safe_float(df["active"]).fillna(0).astype("int8")

    for c in ("issuer", "sector", "rating"):
        if c in df.columns:
            df[c] = df[c].astype("category")

    df = df.sort_values(["date", "debenture_id"]).set_index(["date", "debenture_id"])
    return df


# ------------------------------- Public API ---------------------------------

def process_universe(universe: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
    """Main processing function with enhanced features."""
    paths = UniversePaths(data_dir=data_dir, universe=universe.lower())
    raw = load_idex_folder(paths)
    if raw.empty:
        print(f"[WARN] Universe '{universe}' is empty.")
        return None

    pnl = _build_complete_panel(raw)
    pnl = _compute_per_asset_returns(pnl)
    pnl = _attach_benchmark(pnl)

    # Risk-free (online) + provenance
    pnl, rf_info = _attach_risk_free(pnl)

    # Basic features
    pnl = _basic_features(pnl)
    
    # Sector signals
    pnl = _sector_signals(pnl, momentum_window=63)

    # === ENHANCED FEATURE ENGINEERING ===
    
    # 1. Momentum/Reversal features
    print("[INFO] Computing momentum/reversal features...")
    pnl = _compute_momentum_features(pnl, windows=MOMENTUM_WINDOWS)
    
    # 2. Volatility features
    print("[INFO] Computing volatility features...")
    pnl = _compute_volatility_features(pnl, windows=VOLATILITY_WINDOWS)
    
    # 3. Relative value features
    print("[INFO] Computing relative value features...")
    pnl = _compute_relative_value_features(pnl)
    
    # 4. Duration risk features
    print("[INFO] Computing duration risk features...")
    pnl = _compute_duration_risk_features(pnl)
    
    # 5. Microstructure features
    print("[INFO] Computing microstructure features...")
    pnl = _compute_microstructure_features(pnl)
    
    # 6. Carry features
    print("[INFO] Computing carry features...")
    pnl = _compute_carry_features(pnl)
    
    # 7. Spread dynamics
    print("[INFO] Computing spread dynamics...")
    pnl = _compute_spread_dynamics(pnl)
    
    # 8. Risk-adjusted features
    print("[INFO] Computing risk-adjusted features...")
    pnl = _compute_risk_adjusted_features(pnl)

    # ANBIMA sector curve features
    print("[INFO] Fitting sector curves...")
    pnl = _apply_sector_curves_anbima(pnl, use_anchor=True)

    # ---------------------------------------------------------------------
    # Cross-sectional feature cleaning (NO look-ahead)
    # ---------------------------------------------------------------------
    
    # Extended list of continuous features for normalization
    cont_candidates = [
        # Original features
        "spread", "duration", "time_to_maturity",
        "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
        "sector_fitted_spread", "spread_residual_ns",
        "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
        "index_level", "index_weight",
        # New features
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
    cont_features = [c for c in cont_candidates if c in pnl.columns]

    # Winsorize and z-score normalize on active assets
    active_mask = (pnl["active"] == 1) if "active" in pnl.columns else pd.Series(True, index=pnl.index)

    print("[INFO] Winsorizing features...")
    pnl_active = pnl.loc[active_mask, ["date", "debenture_id"] + cont_features].copy()
    pnl_active = _winsorize_xsec(pnl_active, cont_features, lower=0.005, upper=0.995)
    
    print("[INFO] Computing cross-sectional z-scores...")
    pnl_active = _zscore_xsec(pnl_active, cont_features)

    zcols = [c + "_z" for c in cont_features]
    pnl = pnl.merge(pnl_active[["date", "debenture_id"] + zcols], on=["date", "debenture_id"], how="left")
    for zc in zcols:
        pnl[zc] = pnl[zc].fillna(0.0)

    # Per-asset rolling z-scores
    print("[INFO] Computing rolling z-scores...")
    pnl = _zscore_ts_rolling_past_only(pnl, cont_features, window=252)

    # Diagnostics
    if "excess_return" not in pnl.columns and {"return", "risk_free"}.issubset(pnl.columns):
        pnl["excess_return"] = (pnl["return"] - pnl["risk_free"]).astype("float32")

    # ---------------------------------------------------------------------
    # Create lag-1 versions of all features (strictly causal)
    # ---------------------------------------------------------------------
    print("[INFO] Creating lagged features...")
    
    lag_candidates = set(cont_features + zcols)
    lag_candidates.update([
        "ttm_rank", "index_weight",
        "return", "risk_free", "index_return", "excess_return",
        "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
        "sector_fitted_spread", "spread_residual_ns",
        "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
        "index_level",
        # All new features
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
    ])
    
    # Add rolling zscore columns
    lag_candidates.update([c for c in pnl.columns if any(c.endswith(suf) for suf in ("_z252", "_z126", "_z63"))])

    pnl = pnl.sort_values(["debenture_id", "date"])
    g = pnl.groupby("debenture_id", sort=False)
    
    for c in sorted(lag_candidates):
        if c in pnl.columns:
            pnl[c + "_lag1"] = g[c].shift(1).astype("float32")
            pnl[c + "_lag1"] = pnl[c + "_lag1"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    # Ensure key lags exist
    if "excess_return_lag1" not in pnl.columns and "excess_return" in pnl.columns:
        pnl["excess_return_lag1"] = g["excess_return"].shift(1).astype("float32").fillna(0.0)
    if "return" in pnl.columns and "return_lag1" not in pnl.columns:
        pnl["return_lag1"] = g["return"].shift(1).astype("float32").fillna(0.0)
    if "index_return" in pnl.columns and "index_return_lag1" not in pnl.columns:
        pnl["index_return_lag1"] = g["index_return"].shift(1).astype("float32").fillna(0.0)
    if "risk_free" in pnl.columns and "risk_free_lag1" not in pnl.columns:
        pnl["risk_free_lag1"] = g["risk_free"].shift(1).astype("float32").fillna(0.0)

    pnl = pnl.sort_values(["date", "debenture_id"])

    # Final tidy + types + indexing
    final = _final_tidy_types(pnl)

    # Persist risk-free series
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

    # Provenance/context
    info_path = os.path.join(data_dir, "data_info.json")
    try:
        info = {}
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
        
        # Count features
        feature_count = len([c for c in final.columns if c.endswith("_lag1")])
        
        info.setdefault("risk_free", {})[universe.lower()] = {
            "source": rf_info.get("source", "unknown"),
            "code": rf_info.get("code", None),
            "rows": int(len(rf_used)),
            "first_date": rf_used["date"].min().strftime("%Y-%m-%d") if not rf_used.empty else None,
            "last_date": rf_used["date"].max().strftime("%Y-%m-%d") if not rf_used.empty else None,
            "cache_dir": CACHE_DIR,
        }
        info.update({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "versions": {"pandas": pd.__version__, "numpy": np.__version__},
            "columns_required": REQUIRED_COLS,
            "columns_optional": [c for c in OPTIONAL_KEEP if c in final.columns],
            "total_features": feature_count,
            "enhanced_features": {
                "momentum_windows": MOMENTUM_WINDOWS,
                "volatility_windows": VOLATILITY_WINDOWS,
                "feature_groups": [
                    "momentum_reversal",
                    "volatility",
                    "relative_value",
                    "duration_risk",
                    "microstructure",
                    "carry",
                    "spread_dynamics",
                    "risk_adjusted",
                    "sector_curves"
                ]
            },
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "short_end_exclusion_bd": BUSINESS_DAYS_SHORT_DROP,
            "ns_lambda_grid": list(map(float, _grid_lambdas())),
            "anbima_use_anchor": True,
            "wls_weight": "duration_years_clipped_[1e-4,10]",
            "feature_lag_policy": "all candidate predictors provided as *_lag1 per debenture; rewards use same-day return/rf/index_return",
        })
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    print(
        f"{universe.upper()} ready: rows={len(final):,} | "
        f"dates={final.index.get_level_values('date').min().date()}→"
        f"{final.index.get_level_values('date').max().date()} | "
        f"assets≈{final.index.get_level_values('debenture_id').nunique()} | "
        f"features≈{len([c for c in final.columns if c.endswith('_lag1')])}"
    )
    
    # Feature summary
    print("\n[FEATURE SUMMARY]")
    print(f"Total columns: {len(final.columns)}")
    print(f"Lagged features (_lag1): {len([c for c in final.columns if c.endswith('_lag1')])}")
    print(f"Z-score features (_z): {len([c for c in final.columns if c.endswith('_z')])}")
    print(f"Rolling z-scores (_z252): {len([c for c in final.columns if '_z252' in c])}")
    
    return final


def prepare_data(data_dir: str = "data"):
    """Process CDI and Infrastructure universes with enhanced features."""
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ENHANCED DATA PREPARATION WITH ADVANCED FEATURES")
    print("="*60 + "\n")
    
    cdi = process_universe("cdi", data_dir)
    
    if cdi is not None:
        print(f"\n[SUCCESS] CDI universe processed with enhanced features")
    
    print("\nData preparation complete.")
    return cdi


if __name__ == "__main__":
    prepare_data()