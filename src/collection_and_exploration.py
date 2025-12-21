from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import Output, HBox
from IPython.display import HTML, display


def display_side_by_side(
    dfs: Sequence[pd.DataFrame],
    titles: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
) -> None:
    """
    Display multiple DataFrames side by side in a Jupyter notebook.
    """
    if titles is None:
        titles = [""] * len(dfs)

    html = ""
    for df_i, title in zip(dfs, titles):
        df_show = df_i.copy()
        if max_rows is not None:
            df_show = df_show.head(max_rows)
        if max_cols is not None and df_show.shape[1] > max_cols:
            df_show = df_show.iloc[:, :max_cols]

        html += "<div style='flex:1; padding: 0 10px;'>"
        if title:
            html += f"<h4 style='margin: 0 0 8px 0;'>{title}</h4>"
        html += df_show.to_html()
        html += "</div>"

    display(HTML(f"<div style='display:flex; align-items:flex-start;'>{html}</div>"))

        
def plot_hist(df: pd.DataFrame, col: str, bins: int = 50, figsize: tuple = (7, 4)) -> None:
    plt.figure(figsize=figsize)
    plt.hist(df[col].dropna(), bins=bins)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def show_hist(df, col, bins=50, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.hist(df[col].dropna(), bins=bins)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
def plot_box(df: pd.DataFrame, col: str, figsize: tuple = (6.5, 3.8)) -> None:
    plt.figure(figsize=figsize)
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()
    
def show_box(df, col, figsize=(6.5, 3.8), vert=False):
    plt.figure(figsize=figsize)
    plt.boxplot(df[col].dropna(), vert=vert)
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

def plot_top_bar(
    df: pd.DataFrame,
    col: str,
    top_n: int = 10,
    figsize: tuple = (6.5, 4),
    rotate_xticks: int = 45,
) -> None:
    vc = df[col].value_counts(dropna=False).head(top_n)
    plt.figure(figsize=figsize)
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(f"Top {top_n} categories of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    plt.show()


def plot_missing_percent_bar(
    missing_df_nonzero: pd.DataFrame,
    percent_col: str = "missing_percent (%)",
    title: str = "Missing Percentage by Column",
    figsize: tuple = (10, 4.5),
    rotate_xticks: int = 45,
) -> None:
    """
    missing_df_nonzero: index = column name, contains percent_col
    """
    if missing_df_nonzero is None or len(missing_df_nonzero) == 0:
        print("No missing values detected.")
        return

    plt.figure(figsize=figsize)
    plt.bar(missing_df_nonzero.index.astype(str), missing_df_nonzero[percent_col])
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Missing Percent (%)")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap_matplotlib(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap (Sample)",
    figsize: tuple = (10, 7),
    rotate_xticks: int = 60,
) -> None:
    """
    Pure matplotlib heatmap (không seaborn).
    """
    plt.figure(figsize=figsize)
    plt.imshow(corr_matrix.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=rotate_xticks, ha="right")
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    plt.title(title)
    plt.tight_layout()
    plt.show()



def show_scatter(
    df_corr: pd.DataFrame,
    x_col: str,
    y_col: str = "aqi",
    figsize: tuple = (6.2, 4),
    s: int = 6,
    alpha: float = 0.3,
    title: str | None = None,
) -> None:
    """
    Scatter plot y_col vs x_col.
    - An toàn với dữ liệu kiểu '-' / string: ép numeric errors='coerce' -> NaN rồi dropna.
    """
    if x_col not in df_corr.columns:
        raise KeyError(f"Column '{x_col}' not found in df_corr.")
    if y_col not in df_corr.columns:
        raise KeyError(f"Column '{y_col}' not found in df_corr.")

    tmp = df_corr[[y_col, x_col]].copy()

    # Ép kiểu số: '-' hoặc text -> NaN
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")

    tmp = tmp.dropna()

    plt.figure(figsize=figsize)
    plt.scatter(tmp[x_col], tmp[y_col], s=s, alpha=alpha)
    plt.title(title or f"{y_col.upper()} vs {x_col} (Sample)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

def plot_top_counties_mean_aqi(
    df: pd.DataFrame,
    county_col: str = "county",
    aqi_col: str = "aqi",
    top_n: int = 10,
    figsize: tuple = (7, 4),
    rotate_xticks: int = 45,
):
    """
    Returns Series mean_aqi by county (sorted desc). Also plots top_n bar.
    """
    if county_col not in df.columns:
        print(f"Column '{county_col}' not found.")
        return None

    aqi_by_county = (
        df[[county_col, aqi_col]]
        .dropna(subset=[county_col])
        .groupby(county_col)[aqi_col]
        .mean()
        .sort_values(ascending=False)
    )

    top = aqi_by_county.head(top_n)

    plt.figure(figsize=figsize)
    plt.bar(top.index.astype(str), top.values)
    plt.title(f"Top {top_n} Counties by Mean AQI")
    plt.xlabel(county_col)
    plt.ylabel(f"mean {aqi_col}")
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    plt.show()

    return aqi_by_county