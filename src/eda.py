from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import NUMERIC_COLUMNS, TABLES_DIR, TARGET


EDA_NUMERIC_COLUMNS = NUMERIC_COLUMNS + [
    "arrival_month_num",
    "total_nights",
    "total_guests",
    "has_children",
    "has_agent",
    "has_company",
    "adr_per_person",
]


def save_table(df: pd.DataFrame, filename: str, index: bool = True) -> Path:
    path = TABLES_DIR / filename
    df.to_csv(path, index=index)
    return path


def cancellation_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(bookings=(TARGET, "size"), cancellations=(TARGET, "sum"), cancellation_rate=(TARGET, "mean"))
        .sort_values("bookings", ascending=False)
    )
    grouped["cancellation_rate"] = grouped["cancellation_rate"].round(4)
    return grouped


def build_eda_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    numeric_cols = [col for col in EDA_NUMERIC_COLUMNS if col in df.columns]
    numeric_summary = df[numeric_cols].describe().T
    numeric_summary["missing"] = df[numeric_cols].isna().sum()
    numeric_summary["skew"] = df[numeric_cols].skew(numeric_only=True)

    categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    categorical_summary = []
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        categorical_summary.append(
            {
                "column": col,
                "unique": int(df[col].nunique(dropna=True)),
                "top": str(counts.index[0]),
                "top_count": int(counts.iloc[0]),
                "top_rate": round(float(counts.iloc[0] / len(df)), 4),
            }
        )

    target_counts = df[TARGET].value_counts().rename_axis(TARGET).to_frame("count")
    target_counts["rate"] = (target_counts["count"] / len(df)).round(4)

    monthly = (
        df.groupby(["arrival_date_year", "arrival_month_num"], dropna=False)
        .agg(bookings=(TARGET, "size"), cancellations=(TARGET, "sum"), cancellation_rate=(TARGET, "mean"))
        .reset_index()
        .sort_values(["arrival_date_year", "arrival_month_num"])
    )
    monthly["cancellation_rate"] = monthly["cancellation_rate"].round(4)

    corr_cols = [col for col in df.select_dtypes(include="number").columns if col != TARGET]
    correlation = df[corr_cols + [TARGET]].corr(numeric_only=True)
    cancellation_corr = (
        correlation[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False).to_frame("corr_with_cancel")
    )

    return {
        "numeric_summary": numeric_summary,
        "categorical_summary": pd.DataFrame(categorical_summary).set_index("column"),
        "target_counts": target_counts,
        "cancellation_by_hotel": cancellation_rate(df, "hotel"),
        "cancellation_by_market_segment": cancellation_rate(df, "market_segment"),
        "cancellation_by_deposit_type": cancellation_rate(df, "deposit_type"),
        "cancellation_by_customer_type": cancellation_rate(df, "customer_type"),
        "cancellation_by_lead_time_band": cancellation_rate(df, "lead_time_band"),
        "monthly_cancellation": monthly.set_index(["arrival_date_year", "arrival_month_num"]),
        "correlation": correlation,
        "cancellation_correlations": cancellation_corr,
        "cleaned_sample": df.head(1000),
    }


def save_eda_outputs(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    saved = {}
    for name, table in tables.items():
        saved[name] = str(save_table(table, f"{name}.csv"))
    return saved


def write_summary_json(summary: dict, path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
