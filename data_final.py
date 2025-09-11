# data_final.py
"""
Data preparation for debenture universes (CDI and Infra), dissertation-ready.

Highlights
----------
1) Risk-free (daily) from BCB/SGS (online, no manual files):
   - Try CDI (multiple candidate SGS codes), fallback to SELIC if CDI unavailable.
   - Chunked fetching (≤10y windows), cached in data/_cache/*.parquet.
   - Persist the exact aligned series to results/<universe>/risk_free_used.csv
     and record provenance in data_info.json.

2) Sector credit curves (ANBIMA-style):
   S_sector(τ) = β0_sector + β1_common * φ(τ, λ_common),
   φ(τ,λ) = (1 - e^{-λτ}) / (λ τ)
   - Estimated per day by WLS (weights = duration years, clipped).
   - Drop short-end (<21 business days). Optional 1-day anchor per sector.
   - IQR filter + robust residual pass to avoid stale quotes.
   - Adds features:
       sector_ns_beta0, ns_beta1_common, ns_lambda_common,
       sector_fitted_spread, spread_residual_ns,
       sector_ns_level_1y / 3y / 5y

3) Panel construction follows Idex conventions:
   - Forward-fill per-bond spread when missing.
   - Index return from date-level index_level.
   - Optional decomposition columns preserved if present.

4) Causality: **all candidate features for the policy are provided as `_lag1`**
   (per-debenture shift by 1). Rewards remain on same-day data.
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

# Optional HTTP
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    requests = None
    _HAS_REQUESTS = False


# ----------------------------- Config & constants ----------------------------

TRADING_DAYS_PER_YEAR = 252.0
BUSINESS_DAYS_SHORT_DROP = 21
TAU_SHORT_ANCHOR = 1.0 / TRADING_DAYS_PER_YEAR

SERIES_CODES = {
    "cdi_primary": 4389,   # CDI Over, % a.a. (daily)
    "selic": 11,           # SELIC Over, % a.a. (daily)
}
# Candidate CDI codes (some docs/projects reference 12/4392)
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
    # diagnostics
    "ttm_rank", "excess_return", "sector_weight_index", "sector_spread_avg", "sector_spread", "sector_momentum"
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
        # spreads (CDI universe vs Infra universe)
        "Spread de compra (%)": "spread",      # CDI: % a.a.
        "MID spread (Bps/NTNB)": "spread",     # Infra: bps vs NTN-B
        # descriptors
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

    # Percent/BPS to decimals
    pct_cols = ["index_weight", "weighted_mtm", "weighted_return", "weighted_carry"]
    for c in pct_cols:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            df[c] = _safe_float(df[c]) / 100.0

    if "spread" in df.columns:
        if universe.lower() == "infra":
            # bps -> decimal
            if df["spread"].dtype == object:
                df["spread"] = df["spread"].astype(str).str.replace(",", ".", regex=False)
            df["spread"] = _safe_float(df["spread"]) / 10000.0
        else:
            # % -> decimal
            if df["spread"].dtype == object:
                df["spread"] = df["spread"].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
            df["spread"] = _safe_float(df["spread"]) / 100.0

    # Scalars
    for c in ("duration", "index_level"):
        if c in df.columns:
            df[c] = _safe_float(df[c])

    # Optionals as categories
    for c in ("sector", "issuer"):
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype("category")

    if "callable" in df.columns:
        df["callable"] = df["callable"].map({"1": 1, 1: 1, True: 1, "Sim": 1}).fillna(0).astype(np.int8)

    df = df.dropna(subset=["date", "debenture_id"]).sort_values(["date", "debenture_id"])
    return df

# --- X-sec safe transforms (no time look-ahead) -----------------------------

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

# --- Optional: trailing time-series scaling (strictly past-only) ------------

def _zscore_ts_rolling_past_only(df: pd.DataFrame, cols: list[str], window: int = 252) -> pd.DataFrame:
    """
    Per-asset rolling z-score using only data *up to t-1*.
    We .rolling(...).mean().shift(1) and .std().shift(1) to exclude the current day.
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


# ------------------------------- SGS fetch ----------------------------------

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
            r = requests.get(_SGS_BASE.format(code=code), params=params, timeout=30)  # type: ignore
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
    """
    Fetch BCB SGS series with on-disk parquet cache.
    - Updates incrementally from the last cached date to 'today'
    - Respects SGS window limits via chunking
    Returns: DataFrame[date, value] (% a.a. for CDI/SELIC)
    """
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


# ------------------------------- I/O: Idex ----------------------------------

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


# ------------------------------ Panel building ------------------------------

def _build_complete_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (date, debenture_id) panel on Idex trading days.
    Forward-fill per-debenture fields (e.g., spread) to mirror index conventions.
    """
    if raw_df.empty:
        return raw_df

    if "index_weight" not in raw_df.columns:
        raw_df["index_weight"] = np.nan

    out = raw_df.sort_values(["debenture_id", "date"]).copy()

    # Forward-fill per debenture
    ffill_cols = [c for c in ["sector", "issuer", "callable", "index_level", "spread"] if c in out.columns]
    # Numerics
    for c in [c for c in ffill_cols if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]:
        out[c] = out.groupby("debenture_id", sort=False)[c].ffill()
    # Categoricals/objects
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
    """
    Approx per-asset daily return from weighted components if present:
      return_i,t ≈ (weighted_mtm_t + weighted_carry_t) / index_weight_i,t
    """
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
    """
    Attach risk_free daily from SGS:
      1) Try CDI (candidate series)
      2) Fall back to SELIC if CDI is missing
    Returns (df_with_rf, info_dict) where info_dict notes source and code.
    """
    if df.empty:
        return df, {"source": "none"}

    # Try CDI candidates
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

    # Align to panel trading dates (exact match only)
    rf = daily.set_index("date").sort_index()
    rf = rf[~rf.index.duplicated(keep="last")]
    
    # Get unique trading dates from panel
    panel_dates = pd.DatetimeIndex(df["date"].dropna().sort_values().unique())
    
    # Merge with exact date matching
    rf_aligned = pd.DataFrame(index=panel_dates, columns=["risk_free"])
    rf_aligned = rf_aligned.merge(rf, left_index=True, right_index=True, how="left")
    
    # Forward fill only from past values (no future peeking)
    rf_aligned["cdi_daily"] = rf_aligned["cdi_daily"].ffill().fillna(0.0)
    
    # Merge back with panel
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

    # TTM proxy in years: last date per debenture - current
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
        # add excess return diagnostic if risk_free is already there; else later
        if "risk_free" in df.columns:
            df["excess_return"] = (df["return"] - df["risk_free"]).astype("float32")
        else:
            df["excess_return"] = np.float32(0.0)
    return df


# -------- ANBIMA-style sector curve (NS with shared β1, λ per day) ---------

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
    # Ensure τ exists
    if "time_to_maturity" not in g.columns or g["time_to_maturity"].isna().all():
        g["time_to_maturity"] = g.get("duration", pd.Series(index=g.index, data=np.nan)).astype(float)
    # Drop short-end
    g = g[(g["time_to_maturity"] >= BUSINESS_DAYS_SHORT_DROP / TRADING_DAYS_PER_YEAR)]
    # Optional anchor
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

    # IQR filter
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

    best = (np.inf, None, None, None)  # (sse, beta, lam, yhat)
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

    # Robust pass: drop extreme residuals and refit once
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

    # Fitted values for full df_day (no anchors)
    full_tau = df_day["time_to_maturity"].to_numpy(dtype=float)
    full_sec = df_day["sector_id"].astype(int).to_numpy()
    full_phi = _phi_ns2(full_tau, lam_best)
    fitted_full = np.array([beta0_sector.get(int(s), np.nan) + beta1 * ph for s, ph in zip(full_sec, full_phi)], dtype=float)
    fitted_ser = pd.Series(fitted_full, index=df_day.index)

    used_mask = df_day.index.isin(g1.index)
    return {"beta0_sector": beta0_sector, "beta1": beta1, "lambda": float(lam_best),
            "fitted": fitted_ser, "used_mask": used_mask}


def _sector_signals(df: pd.DataFrame, momentum_window: int = 63) -> pd.DataFrame:
    """
    Add sector-level signals per (date, sector) and broadcast to each bond row:
      - sector_weight_index: sum of index_weight within sector (per date)
      - sector_spread: sector average spread (weighted by index_weight if available)
      - sector_momentum: deviation of sector_spread from its rolling mean over `momentum_window` days
    Does NOT drop/alter any existing columns. Fills missing values with 0.0 where needed.
    """
    out = df.sort_values(["date", "sector"]).copy()
    # sector_weight_index
    if "sector" in out.columns and "index_weight" in out.columns:
        sw = (out.groupby(["date", "sector"], sort=False, observed=False)["index_weight"]
                 .sum(min_count=1)
                 .rename("sector_weight_index")
                 .reset_index())
        out = out.merge(sw, on=["date", "sector"], how="left")
    else:
        out["sector_weight_index"] = np.nan

    # sector average spread (weighted if weights available)
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

    # sector_spread = risk_free(date) + sector_spread_avg
    out["sector_spread"] = _safe_float(out["sector_spread_avg"]).fillna(0.0).astype(np.float32)

    # sector_momentum: per-sector rolling mean deviation of sector_spread
    if "sector" in out.columns:
        out = out.sort_values(["sector","date"]).copy()
        g = out.groupby("sector", sort=False, observed=False)
        roll_mean = g["sector_spread"].transform(lambda s: s.rolling(momentum_window, min_periods=5).mean())
        out["sector_momentum"] = (_safe_float(out["sector_spread"]) - _safe_float(roll_mean)).astype(np.float32)
        out = out.sort_values(["debenture_id","date"]).copy()
    else:
        out["sector_momentum"] = 0.0

    # Clean NA to 0.0 to keep shapes stable downstream
    for c in ["sector_weight_index","sector_spread","sector_momentum","sector_spread_avg"]:
        if c in out.columns:
            out[c] = _safe_float(out[c]).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    return out

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
    # Keep required, known optional, and any engineered *_z, *_z###, *_lag1 columns
    keep = set(REQUIRED_COLS) | {c for c in OPTIONAL_KEEP if c in df.columns}
    pattern_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_z") or "_z" in c]
    keep |= set(pattern_cols)

    cols = [c for c in df.columns if c in keep or c in ("date", "debenture_id")]
    df = df[cols].copy()

    # floats
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

    df = df.sort_values(["date", "debenture_id"]).set_index(["date", "debenture_id"])  # type: ignore
    return df


# ------------------------------- Public API ---------------------------------

def process_universe(universe: str, data_dir: str = "data") -> Optional[pd.DataFrame]:
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

    # Basic feats (uses risk_free if present)
    pnl = _basic_features(pnl)
    # NEW: sector signals (weight, spread, momentum)
    pnl = _sector_signals(pnl, momentum_window=63)


    # ANBIMA sector curve features
    pnl = _apply_sector_curves_anbima(pnl, use_anchor=True)

    # ---------------------------------------------------------------------
    # Cross-sectional feature cleaning (NO look-ahead across time) — column-based
    # Panel has 'date' / 'debenture_id' as COLUMNS here
    # ---------------------------------------------------------------------

    cont_candidates = [
        "spread", "duration", "time_to_maturity",
        "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
        "sector_fitted_spread", "spread_residual_ns",
        "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
        "index_level", "index_weight",
    ]
    cont_features = [c for c in cont_candidates if c in pnl.columns]

    # compute stats using only ACTIVE assets, then merge back by ['date','debenture_id']
    active_mask = (pnl["active"] == 1) if "active" in pnl.columns else pd.Series(True, index=pnl.index)

    pnl_active = pnl.loc[active_mask, ["date", "debenture_id"] + cont_features].copy()
    pnl_active = _winsorize_xsec(pnl_active, cont_features, lower=0.005, upper=0.995)
    pnl_active = _zscore_xsec(pnl_active, cont_features)

    zcols = [c + "_z" for c in cont_features]
    pnl = pnl.merge(pnl_active[["date", "debenture_id"] + zcols], on=["date", "debenture_id"], how="left")
    for zc in zcols:
        pnl[zc] = pnl[zc].fillna(0.0)

    # ---------------------------------------------------------------------
    # Optional: per-asset rolling z-score (strictly past-only) — column-based
    # (uses groupby('debenture_id') internally with .shift(1))
    # ---------------------------------------------------------------------
    pnl = _zscore_ts_rolling_past_only(pnl, cont_features, window=252)

    # Diagnostics: contemporaneous excess (reward component); safe as outcome
    if "excess_return" not in pnl.columns and {"return", "risk_free"}.issubset(pnl.columns):
        pnl["excess_return"] = (pnl["return"] - pnl["risk_free"]).astype("float32")

    # ---------------------------------------------------------------------
    # NEW: make lag-1 versions of all candidate features (strictly causal)
    # ---------------------------------------------------------------------
    # Start with core, curve, diagnostics, and x-sec/ts z-scores
    lag_candidates = set(cont_features + zcols)
    lag_candidates.update([
        "ttm_rank", "index_weight",
        "return", "risk_free", "index_return", "excess_return",
        "sector_ns_beta0", "ns_beta1_common", "ns_lambda_common",
        "sector_fitted_spread", "spread_residual_ns",
        "sector_ns_level_1y", "sector_ns_level_3y", "sector_ns_level_5y",
        "index_level"
    ])
    # add rolling zscore columns if present (e.g., *_z252)
    lag_candidates.update([c for c in pnl.columns if any(c.endswith(suf) for suf in ("_z252", "_z126", "_z63"))])

    # create per-debenture lag1 for each candidate; fill with 0.0 for missing
    pnl = pnl.sort_values(["debenture_id", "date"])
    g = pnl.groupby("debenture_id", sort=False)
    for c in sorted(lag_candidates):
        if c in pnl.columns:
            pnl[c + "_lag1"] = g[c].shift(1).astype("float32")
            pnl[c + "_lag1"] = pnl[c + "_lag1"].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    # Also provide an explicit lag for excess_return (if not already covered)
    if "excess_return_lag1" not in pnl.columns and "excess_return" in pnl.columns:
        pnl["excess_return_lag1"] = g["excess_return"].shift(1).astype("float32").fillna(0.0)

    # example: (kept) — if you want specific lags explicitly named
    if "return" in pnl.columns and "return_lag1" not in pnl.columns:
        pnl["return_lag1"] = g["return"].shift(1).astype("float32").fillna(0.0)
    if "index_return" in pnl.columns and "index_return_lag1" not in pnl.columns:
        pnl["index_return_lag1"] = g["index_return"].shift(1).astype("float32").fillna(0.0)
    if "risk_free" in pnl.columns and "risk_free_lag1" not in pnl.columns:
        pnl["risk_free_lag1"] = g["risk_free"].shift(1).astype("float32").fillna(0.0)

    # Back to (date, debenture_id) ordering
    pnl = pnl.sort_values(["date", "debenture_id"])

    # Final tidy + types + indexing
    final = _final_tidy_types(pnl)

    # Persist exact risk-free used (auditable) — same-day RF for rewards
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
        f"assets≈{final.index.get_level_values('debenture_id').nunique()}"
    )
    return final


def prepare_data(data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    cdi = process_universe("cdi", data_dir)
    print("Data preparation complete.")
    return cdi, infra


if __name__ == "__main__":
    prepare_data()
