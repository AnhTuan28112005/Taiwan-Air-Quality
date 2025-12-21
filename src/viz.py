# src/viz.py
from __future__ import annotations
from typing import List, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
# ==== CÂU 1 ====
def plot_diurnal_pattern(
    hourly_avg: pd.DataFrame,
    pollutants: Sequence[str],
    hour_col: str = "hour",
    title: str = "Diurnal Pattern of Pollutants",
    xlabel: str = "Hour of Day",
    ylabel: str = "Concentration",
    figsize: tuple = (10, 6),
    annotate_min_max: bool = True,
) -> None:
    """
    Plot diurnal (hourly) pattern for multiple pollutants.
    hourly_avg: DataFrame has columns [hour_col] + pollutants
    """
    plt.figure(figsize=figsize)

    for p in pollutants:
        plt.plot(hourly_avg[hour_col], hourly_avg[p], marker="o", label=p)

        if annotate_min_max:
            max_idx = hourly_avg[p].idxmax()
            min_idx = hourly_avg[p].idxmin()

            for idx in [max_idx, min_idx]:
                x = hourly_avg.loc[idx, hour_col]
                y = hourly_avg.loc[idx, p]
                plt.annotate(
                    f"{y:.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_weekly_pattern(
    weekly_avg: pd.DataFrame,
    pollutants: Sequence[str],
    weekday_col: str = "weekday",
    title: str = "Weekly Pattern of Pollutants",
    xlabel: str = "Day of Week (0=Mon)",
    ylabel: str = "Concentration",
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot weekly pattern as bar chart (uses pandas DataFrame.plot).
    weekly_avg: DataFrame has columns [weekday_col] + pollutants
    """
    ax = weekly_avg.plot(x=weekday_col, y=list(pollutants), kind="bar", figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def plot_seasonal_pattern(
    season_melted: pd.DataFrame,
    season_col: str = "season",
    value_col: str = "Average_Concentration",
    hue_col: str = "Pollutant",
    title: str = "Seasonal Variation of Main Pollutants",
    xlabel: str = "Season",
    ylabel: str = "Average Concentration",
    figsize: tuple = (10, 6),
    use_seaborn: bool = True,
    palette: Optional[str] = "viridis",
) -> None:
    """
    Plot seasonal pattern from a melted (long-form) DataFrame.
    season_melted columns: [season_col, hue_col, value_col]
    """
    plt.figure(figsize=figsize)

    if use_seaborn:
        import seaborn as sns  # import here so seaborn is optional

        sns.barplot(
            x=season_col,
            y=value_col,
            hue=hue_col,
            data=season_melted,
            palette=palette,
        )
    else:
        # fallback: matplotlib grouped bar from pivot table
        pivot = season_melted.pivot(index=season_col, columns=hue_col, values=value_col)
        pivot.plot(kind="bar", figsize=figsize)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=hue_col)
    plt.show()

# ==== CÂU 2: Pollution episode duration plots ====

def plot_top10_episode_duration(
    top10: pd.DataFrame,
    y_col: str = "avg_duration",
    title: str = "Top 10 Stations by Average Pollution Episode Duration",
    ylabel: str = "Average Duration (hours)",
    figsize: tuple = (10, 6),
    rotate_xticks: int = 45,
    use_seaborn: bool = True,
) -> None:
    """
    top10: DataFrame index = station/site, column y_col = avg duration.
    """
    plt.figure(figsize=figsize)

    if use_seaborn:
        import seaborn as sns
        sns.barplot(x=top10.index, y=top10[y_col])
    else:
        plt.bar(top10.index.astype(str), top10[y_col].values)

    plt.xticks(rotation=rotate_xticks)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_season_episode_duration(
    season_summary: pd.DataFrame,
    y_col: str = "avg_duration",
    title: str = "Average Pollution Episode Duration by Season",
    ylabel: str = "Average Duration (hours)",
    figsize: tuple = (8, 5),
    rotate_xticks: int = 0,
    use_seaborn: bool = True,
) -> None:
    """
    season_summary: DataFrame index = season, column y_col = avg duration.
    """
    plt.figure(figsize=figsize)

    if use_seaborn:
        import seaborn as sns
        sns.barplot(x=season_summary.index, y=season_summary[y_col])
    else:
        plt.bar(season_summary.index.astype(str), season_summary[y_col].values)

    plt.xticks(rotation=rotate_xticks)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==== CÂU 3: Correlation + status median + across-site consistency ====

def plot_topk_pearson_bar(
    corr_df: pd.DataFrame,
    top_k: int = 10,
    var_col: str = "variable",
    pearson_col: str = "pearson_r",
    title: str | None = None,
    xlabel: str = "Biến",
    ylabel: str = "Pearson r",
    figsize: tuple = (7, 4),
    rotate_xticks: int = 45,
):
    """
    Hiển thị layout giống notebook của bạn:
    - Trái: bảng thống kê corr_df
    - Phải: bar plot top_k theo pearson_r

    Trả về widgets.HBox để notebook display.
    """
    import ipywidgets as widgets
    from IPython.display import display

    if title is None:
        title = f"Top {top_k} biến tương quan Pearson với AQI"

    # lấy top_k để vẽ
    top_corr = corr_df.sort_values(pearson_col, ascending=False).head(top_k).copy()

    out_table = widgets.Output()
    out_plot = widgets.Output()

    with out_table:
        display(corr_df)

    with out_plot:
        plt.figure(figsize=figsize)
        plt.bar(top_corr[var_col].astype(str), top_corr[pearson_col].values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=rotate_xticks, ha="right")
        plt.tight_layout()
        plt.show()

    box = widgets.HBox([out_table, out_plot])
    return box



def plot_median_by_status_lines(
    med: pd.DataFrame,
    status_order: List[str],
    x_labels: Optional[List[str]] = None,
    title_prefix: str = "Median",
    figsize: Optional[tuple] = None,
) -> None:
    """
    Vẽ nhiều line plots dọc: mỗi biến 1 plot.
    med: DataFrame index = status (đã reindex đúng status_order), columns = top_vars
    status_order: list status theo thứ tự
    x_labels: nhãn hiển thị (ví dụ viết tắt)
    """
    top_vars = list(med.columns)

    if figsize is None:
        figsize = (10, 3.2 * len(top_vars))

    fig, axes = plt.subplots(
        nrows=len(top_vars),
        ncols=1,
        figsize=figsize,
        sharex=True
    )
    if len(top_vars) == 1:
        axes = [axes]

    x = list(range(len(status_order)))
    if x_labels is None:
        x_labels = status_order

    for ax, col in zip(axes, top_vars):
        ax.plot(x, med[col].values, marker="o")
        ax.set_title(f"{title_prefix} {col} theo AQI status")
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("status")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=0)

    plt.tight_layout()
    plt.show()


def plot_site_consistency_boxplot(
    site_scores_df: pd.DataFrame,
    top_vars: List[str],
    r_col: str = "pearson_r",
    var_col: str = "variable",
    title: str = "Across-site consistency: Distribution of Pearson r (AQI vs pollutant) by site",
    xlabel: str = "Variable",
    ylabel: str = "Pearson r (within each site)",
    figsize: tuple = (10, 5),
    show_fliers: bool = False,
    hline: Optional[float] = 0.5,
    rotate_xticks: int = 30,
) -> None:
    """
    Boxplot phân bố Pearson r theo site cho từng biến.
    site_scores_df columns: [var_col, 'siteid', r_col]
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = [
        site_scores_df.loc[site_scores_df[var_col] == v, r_col].dropna().values
        for v in top_vars
    ]
    ax.boxplot(data, labels=top_vars, showfliers=show_fliers)

    if hline is not None:
        ax.axhline(hline, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    plt.show()


def plot_site_consistency_heatmap(
    pivot: pd.DataFrame,
    title: str = "Across-site consistency heatmap (Pearson r per site)",
    cbar_label: str = "Pearson r",
    figsize: tuple = (12, 4),
    show_xticks: bool = False,
) -> None:
    """
    Heatmap từ pivot (index=variable, columns=siteid, values=pearson_r)
    pivot: DataFrame đã sort columns theo ý bạn ở notebook.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(title)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    if show_xticks:
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])

    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.show()

# ==== CÂU 4: Wind / Direction / Threshold plots ====


def plot_hexbin_wind_vs_aqi(
    w_plot,
    a_plot,
    bin_median_aqi: pd.DataFrame,
    wind_mid_col: str = "wind_mid",
    aqi_median_col: str = "aqi_median",
    wind_speed_col: str = "windspeed",
    target: str = "aqi",
    w_cap: float | None = None,
    a_cap: float | None = None,
    gridsize: int = 70,
    mincnt: int = 1,
    figsize: tuple = (8, 5),
    title: str = "Hexbin density: windspeed vs AQI (+ median trend by wind bin)",
) -> None:
    """
    Hexbin density plot windspeed vs AQI + overlay median trend by wind bins.
    - w_plot, a_plot: arrays/series used to plot hexbin (already capped/cleaned)
    - bin_median_aqi: df containing wind_mid and aqi_median
    """
    plt.figure(figsize=figsize)
    plt.hexbin(w_plot, a_plot, gridsize=gridsize, mincnt=mincnt)
    plt.plot(bin_median_aqi[wind_mid_col], bin_median_aqi[aqi_median_col], linewidth=2)

    xlab = wind_speed_col
    ylab = target
    if w_cap is not None:
        xlab += f" (capped at p99.5 ~ {w_cap:.2f})"
    if a_cap is not None:
        ylab += f" (capped at p99.5 ~ {a_cap:.2f})"

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("Count per hexbin")
    plt.tight_layout()
    plt.show()


def plot_median_aqi_vs_wind_bin(
    bin_median_aqi: pd.DataFrame,
    wind_mid_col: str = "wind_mid",
    aqi_median_col: str = "aqi_median",
    bin_width: float = 0.5,
    min_n_bin: int = 1000,
    figsize: tuple = (8, 4),
    title: str | None = None,
) -> None:
    """
    Line plot: median AQI vs windspeed (median per bin).
    """
    if title is None:
        title = f"Median AQI vs windspeed (bin_width={bin_width}, keep bins with n≥{min_n_bin})"

    plt.figure(figsize=figsize)
    plt.plot(bin_median_aqi[wind_mid_col], bin_median_aqi[aqi_median_col], marker="o")
    plt.xlabel(f"Windspeed (median per {bin_width} m/s bin)")
    plt.ylabel("Median AQI (within bin)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_polar_median_aqi_by_wind_direction(
    dir_summary_f: pd.DataFrame,
    dir_center_deg_col: str = "dir_center_deg",
    aqi_median_col: str = "aqi_median",
    figsize: tuple = (6.5, 6.5),
    title: str = "Median AQI theo hướng gió (16 sector)",
) -> None:
    """
    Polar plot: median AQI by wind direction sectors.
    dir_summary_f needs columns: dir_center_deg, aqi_median
    """
    theta = np.deg2rad(dir_summary_f[dir_center_deg_col].to_numpy())
    r = dir_summary_f[aqi_median_col].to_numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, r, marker="o")
    ax.set_title(title)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.tight_layout()
    plt.show()


def plot_p_bad_by_wind_direction(
    dir_summary_f: pd.DataFrame,
    sector_col: str = "dir_sector",
    p_bad_col: str = "p_bad",
    figsize: tuple = (9, 4),
    title: str = "Xác suất status xấu theo hướng gió (16 sector)",
) -> None:
    """
    Bar plot: P(status xấu) by wind direction sector.
    Only plots if p_bad_col exists.
    """
    if p_bad_col not in dir_summary_f.columns:
        return

    plt.figure(figsize=figsize)
    plt.bar(dir_summary_f[sector_col].astype(str), dir_summary_f[p_bad_col])
    plt.xlabel("Dir sector (0..15)")
    plt.ylabel("P(status xấu)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_threshold_separation_curves(
    thr_df: pd.DataFrame,
    t_col: str = "t",
    delta_median_col: str = "delta_median",
    delta_p_bad_col: str = "delta_p_bad",
    figsize: tuple = (8.5, 4),
    title_median: str = "Độ tách median AQI theo ngưỡng windspeed",
    title_pbad: str = "Độ tách rủi ro status xấu theo ngưỡng windspeed",
) -> None:
    """
    Plot:
    - delta_median vs threshold t
    - delta_p_bad vs threshold t (if exists)
    """
    # delta_median
    plt.figure(figsize=figsize)
    plt.plot(thr_df[t_col], thr_df[delta_median_col], marker="o")
    plt.xlabel("Threshold t (m/s)")
    plt.ylabel("delta_median = median(AQI|wind<t) - median(AQI|wind>=t)")
    plt.title(title_median)
    plt.tight_layout()
    plt.show()

    # delta_p_bad (optional)
    if delta_p_bad_col in thr_df.columns:
        plt.figure(figsize=figsize)
        plt.plot(thr_df[t_col], thr_df[delta_p_bad_col], marker="o")
        plt.xlabel("Threshold t (m/s)")
        plt.ylabel("delta_p_bad = P(bad|wind<t) - P(bad|wind>=t)")
        plt.title(title_pbad)
        plt.tight_layout()
        plt.show()
        
        
# ======= Câu 5 =========
def plot_aqi_yearly_trend(
    df: pd.DataFrame,
    year_col: str = "year",
    aqi_col: str = "aqi",
    figsize: tuple = (10, 5),
    title: str = "Biến động chỉ số AQI trung bình qua các năm (2016 - 2024)",
    xlabel: str = "Năm",
    ylabel: str = "AQI Trung bình",
    show_value_labels: bool = True,
    value_offset: float = 1.0,
    use_seaborn: bool = True,
    marker: str = "o",
    linewidth: float = 2.5,
    color: str | None = None,
) -> pd.DataFrame:
    """
    Yearly trend: mean AQI by year + lineplot.
    Returns yearly_trend DataFrame.
    """
    yearly_trend = df.groupby(year_col)[aqi_col].mean().reset_index()

    plt.figure(figsize=figsize)
    if use_seaborn:
        import seaborn as sns
        sns.lineplot(data=yearly_trend, x=year_col, y=aqi_col, marker=marker, linewidth=linewidth, color=color)
    else:
        plt.plot(yearly_trend[year_col], yearly_trend[aqi_col], marker=marker, linewidth=linewidth)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(yearly_trend[year_col])
    plt.grid(True, linestyle="--", alpha=0.7)

    if show_value_labels:
        for x, y in zip(yearly_trend[year_col], yearly_trend[aqi_col]):
            plt.text(x, y + value_offset, f"{y:.1f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.show()
    return yearly_trend


def plot_aqi_monthly_trend(
    df: pd.DataFrame,
    month_col: str = "month",
    aqi_col: str = "aqi",
    figsize: tuple = (10, 5),
    title: str = "Chỉ số AQI trung bình theo Tháng (Tính mùa vụ)",
    xlabel: str = "Tháng",
    ylabel: str = "AQI Trung bình",
    show_year_avg_line: bool = True,
    use_seaborn: bool = True,
    palette: str | None = None,
) -> pd.DataFrame:
    """
    Monthly trend: mean AQI by month (all years combined) + barplot.
    Returns monthly_trend DataFrame.
    """
    monthly_trend = df.groupby(month_col)[aqi_col].mean().reset_index()

    plt.figure(figsize=figsize)
    if use_seaborn:
        import seaborn as sns
        sns.barplot(data=monthly_trend, x=month_col, y=aqi_col, palette='coolwarm')
    else:
        plt.bar(monthly_trend[month_col].astype(str), monthly_trend[aqi_col].values)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if show_year_avg_line:
        plt.axhline(y=monthly_trend[aqi_col].mean(), color='red', linestyle="--", label="Trung bình năm")
        plt.legend()

    plt.tight_layout()
    plt.show()
    return monthly_trend


def plot_aqi_year_month_heatmap(
    df: pd.DataFrame,
    year_col: str = "year",
    month_col: str = "month",
    aqi_col: str = "aqi",
    figsize: tuple = (12, 6),
    title: str = "Heatmap: Mức độ ô nhiễm không khí theo Năm và Tháng",
    cmap: str = "RdYlGn_r",
    annot: bool = True,
    fmt: str = ".1f",
    linewidths: float = 0.5,
) -> pd.DataFrame:
    """
    Heatmap of mean AQI by (year x month).
    Returns pivot_table.
    """
    pivot_table = df.pivot_table(values=aqi_col, index=year_col, columns=month_col, aggfunc="mean")

    plt.figure(figsize=figsize)
    import seaborn as sns
    sns.heatmap(pivot_table, cmap=cmap, annot=annot, fmt=fmt, linewidths=linewidths)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Tháng")
    plt.ylabel("Năm")
    plt.tight_layout()
    plt.show()
    return pivot_table


def plot_monthly_cycle_overlay_by_year(
    df: pd.DataFrame,
    year_col: str = "year",
    month_col: str = "month",
    aqi_col: str = "aqi",
    figsize: tuple = (14, 7),
    title: str = "So sánh Chu kỳ Ô nhiễm AQI qua các năm (2016 - 2024)",
    xlabel: str = "Tháng",
    ylabel: str = "AQI Trung bình",
    use_seaborn: bool = True,
    palette: str | None = None,
    marker: str = "o",
    linewidth: float = 2.0,
    drop_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    Overlay lineplot: mean AQI by month for each year.
    Returns monthly_pattern DataFrame.
    """
    monthly_pattern = df.groupby([year_col, month_col])[aqi_col].mean().reset_index()

    if drop_years:
        monthly_pattern = monthly_pattern[~monthly_pattern[year_col].isin(drop_years)].copy()

    plt.figure(figsize=figsize)
    if use_seaborn:
        import seaborn as sns
        sns.lineplot(
            data=monthly_pattern,
            x=month_col,
            y=aqi_col,
            hue=year_col,
             palette='tab10',
            marker=marker,
            linewidth=linewidth,
        )
    else:
        # fallback: matplotlib loop
        for yr, g in monthly_pattern.groupby(year_col):
            plt.plot(g[month_col], g[aqi_col], marker=marker, linewidth=linewidth, label=str(yr))

    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(range(1, 13))
    plt.legend(title="Năm", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return monthly_pattern


def plot_monthly_cycle_facet_by_year(
    monthly_pattern: pd.DataFrame,
    overall_monthly_avg: pd.Series,
    year_col: str = "year",
    month_col: str = "month",
    aqi_col: str = "aqi",
    col_wrap: int = 3,
    height: float = 3.5,
    aspect: float = 1.5,
    sharey: bool = True,
    facet_linewidth: float = 2.0,
    facet_marker: str = "o",
    ref_line_alpha: float = 0.5,
    title: str = "Biến động chi tiết từng năm so với mức trung bình lịch sử (Đường đỏ)",
) -> None:
    """
    FacetGrid: each year a small lineplot + reference overall_monthly_avg (red dashed).
    monthly_pattern should have columns [year, month, aqi] (from plot_monthly_cycle_overlay_by_year return).
    overall_monthly_avg: Series index=month, values=mean AQI overall.
    """
    import seaborn as sns

    g = sns.FacetGrid(
        monthly_pattern,
        col=year_col,
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        sharey=sharey,
    )
    g.map(sns.lineplot, month_col, aqi_col, marker=facet_marker, color="#2c3e50", linewidth=facet_linewidth)

    for ax in g.axes.flat:
        ax.plot(
            overall_monthly_avg.index,
            overall_monthly_avg.values,
            color='red', 
            linestyle="--",
            alpha=ref_line_alpha,
            label="TB Lịch sử",
        )
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)

    g.set_titles("Năm {col_name}")
    g.set_axis_labels("Tháng", "AQI")
    plt.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    
#======= Câu 6 ========
def plot_mae_by_horizon(
    results_df: pd.DataFrame,
    horizon_col: str = "Horizon (Hours)",
    mae_col: str = "MAE",
    model_col: str = "Model",
    horizons: list[int] | None = None,
    figsize: tuple = (10, 6),
    title: str = "Biến động Sai số Dự báo (MAE) theo Khung thời gian",
    xlabel: str = "Dự báo trước (Giờ)",
    ylabel: str = "Sai số trung bình (MAE - µg/m3)",
    grid_alpha: float = 0.6,
    threshold: float | None = 10.0,
    threshold_label: str = "Ngưỡng chấp nhận được (Giả định)",
) -> None:
    """
    Lineplot: MAE vs horizon, hue = model.
    results_df cần có các cột: horizon_col, mae_col, model_col.
    """
    import seaborn as sns

    plt.figure(figsize=figsize)
    sns.lineplot(
        data=results_df,
        x=horizon_col,
        y=mae_col,
        hue=model_col,
        marker="o",
        linewidth=2.5,
    )

    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if horizons is not None:
        plt.xticks(horizons)

    plt.grid(True, linestyle="--", alpha=grid_alpha)

    if threshold is not None:
        plt.axhline(y=threshold, linestyle="--", label=threshold_label)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_pred_scatter(
    y_true,
    y_pred,
    h_focus: int = 24,
    model_name: str = "XGBoost",
    target_name: str = "PM2.5",
    figsize: tuple = (8, 8),
    title: str | None = None,
    alpha: float = 0.3,
    xlim: tuple[float, float] | None = (0, 100),
    ylim: tuple[float, float] | None = (0, 100),
    diag_line: bool = True,
) -> None:
    """
    Scatter: y_true vs y_pred, kèm đường 45 độ.
    """
    if title is None:
        title = f"Thực tế vs Dự báo ({model_name} - Sau {h_focus} giờ)"

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=alpha, color='purple')

    if diag_line:
        # đường chuẩn y=x theo range đang dùng
        if xlim is not None:
            lo, hi = xlim
        else:
            lo = float(np.nanmin([y_true.min(), y_pred.min()]))
            hi = float(np.nanmax([y_true.max(), y_pred.max()]))
        plt.plot([lo, hi], [lo, hi], "r--", lw=2)

    plt.title(title, fontsize=14)
    plt.xlabel(f"Giá trị Thực tế ({target_name})", fontsize=12)
    plt.ylabel("Giá trị Dự báo", fontsize=12)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
