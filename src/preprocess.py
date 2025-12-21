# src/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------- Shared helpers ----------

def month_to_season(month: int) -> str:
    """Map month (1..12) -> season name."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"

# ---------- Q2: pollution episodes ----------

def identify_episodes(
    group: pd.DataFrame,
    date_col: str = "date",
    polluted_col: str = "polluted",
    min_hours: int = 48,
) -> pd.DataFrame:
    """
    Identify continuous polluted episodes in a single group (e.g., one station).
    Expects group sorted by time (we sort inside to be safe).
    Returns DataFrame columns: start, end, duration_hours
    """
    g = group.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    g = g.dropna(subset=[date_col])
    g = g.sort_values(date_col)

    # Mark state changes
    g["change"] = g[polluted_col].ne(g[polluted_col].shift()).cumsum()

    episodes = g.groupby("change").agg(
        start=(date_col, "first"),
        end=(date_col, "last"),
        polluted=(polluted_col, "first"),
        duration_hours=(date_col, lambda x: (x.max() - x.min()).total_seconds() / 3600),
    )

    episodes = episodes[(episodes["polluted"]) & (episodes["duration_hours"] >= min_hours)]
    return episodes[["start", "end", "duration_hours"]].reset_index(drop=True)

# ---------- Preprocessing ----------

def check_missing(df):
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent (%)": missing_percent
    }).sort_values("missing_percent (%)", ascending=False)

    missing_df_nonzero = missing_df[missing_df["missing_count"] > 0]

    print(f"Total rows: {len(df):,}")
    print(f"Columns with missing values: {len(missing_df_nonzero):,} / {df.shape[1]}\n")

    return missing_df

def count_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask.sum(), mask.mean() * 100, lower, upper

# Cập nhật status dựa trên AQI, đảm bảo đồng nhất
def aqi_to_status(aqi):
    if pd.isna(aqi):
        return pd.NA
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for sensitive groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"
