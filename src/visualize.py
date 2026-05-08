from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, NUMERIC_COLUMNS, TARGET


PALETTE = ["#2a9d8f", "#e76f51"]


def _save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.countplot(data=df, x=TARGET, order=[0, 1], palette=PALETTE, hue=TARGET, legend=False, ax=ax)
    ax.set_title("Booking cancellation distribution")
    ax.set_xlabel("Canceled")
    ax.set_xticks([0, 1], labels=["No", "Yes"])
    ax.set_ylabel("Bookings")
    return _save(fig, "target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> Path:
    cols = [
        "lead_time",
        "adr",
        "total_nights",
        "total_guests",
        "booking_changes",
        "days_in_waiting_list",
        "previous_cancellations",
        "total_of_special_requests",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], kde=True, color="#006d77", ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    return _save(fig, "numeric_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    cols = [col for col in NUMERIC_COLUMNS if col in df.columns] + [
        "arrival_month_num",
        "total_nights",
        "total_guests",
        "has_children",
        "has_agent",
        "has_company",
        "adr_per_person",
        TARGET,
    ]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.3, ax=ax)
    ax.set_title("Correlation heatmap")
    return _save(fig, "correlation_heatmap.png")


def plot_cancellation_by_categories(df: pd.DataFrame) -> Path:
    categories = ["hotel", "market_segment", "deposit_type", "customer_type"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, col in zip(axes.ravel(), categories):
        proportions = pd.crosstab(df[col], df[TARGET], normalize="index")
        proportions = proportions.reindex(columns=[0, 1], fill_value=0)
        proportions.plot(kind="bar", stacked=True, color=PALETTE, ax=ax)
        ax.set_title(f"Cancellation mix by {col}")
        ax.set_xlabel("")
        ax.set_ylabel("Share")
        ax.legend(title="Canceled", labels=["No", "Yes"], loc="upper right")
    return _save(fig, "cancellation_by_categories.png")


def plot_boxplots_by_cancellation(df: pd.DataFrame) -> Path:
    cols = ["lead_time", "adr", "total_nights", "booking_changes", "total_of_special_requests"]
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 4.8))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=TARGET, y=col, order=[0, 1], palette=PALETTE, hue=TARGET, legend=False, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_xticks([0, 1], labels=["No", "Yes"])
    return _save(fig, "boxplots_by_cancellation.png")


def plot_monthly_cancellation(df: pd.DataFrame) -> Path:
    monthly = (
        df.groupby(["arrival_date_year", "arrival_month_num"], dropna=False)
        .agg(bookings=(TARGET, "size"), cancellation_rate=(TARGET, "mean"))
        .reset_index()
    )
    monthly["period"] = monthly["arrival_date_year"].astype(str) + "-" + monthly["arrival_month_num"].astype(int).astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=monthly, x="period", y="cancellation_rate", marker="o", color="#264653", ax=ax)
    ax.set_title("Monthly cancellation rate")
    ax.set_xlabel("Arrival month")
    ax.set_ylabel("Cancellation rate")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "monthly_cancellation_rate.png")


def save_all_figures(df: pd.DataFrame) -> dict[str, str]:
    paths = {
        "target_distribution": plot_target_distribution(df),
        "numeric_distributions": plot_numeric_distributions(df),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "cancellation_by_categories": plot_cancellation_by_categories(df),
        "boxplots_by_cancellation": plot_boxplots_by_cancellation(df),
        "monthly_cancellation_rate": plot_monthly_cancellation(df),
    }
    return {name: str(path) for name, path in paths.items()}
