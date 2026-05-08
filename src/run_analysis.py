from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from clean_data import clean_hotel_bookings, quality_checks, validate_schema
from config import DEFAULT_DATA_PATH, OUTPUT_DIR, REPORTS_DIR, TABLES_DIR, TARGET, ensure_directories
from eda import build_eda_tables, save_eda_outputs, write_summary_json
from features import add_features
from load_data import load_hotel_bookings, profile_raw_data
from model import train_cancellation_model
from visualize import save_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hotel booking cancellation analysis pipeline")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to input CSV")
    parser.add_argument("--skip-model", action="store_true", help="Refresh EDA outputs without retraining the model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    start = time.time()

    raw = load_hotel_bookings(args.data)
    raw_profile = profile_raw_data(raw)
    cleaned = clean_hotel_bookings(raw)
    featured = add_features(cleaned)

    schema = validate_schema(cleaned)
    quality = quality_checks(cleaned)
    quality_path = TABLES_DIR / "quality_checks.csv"
    quality.to_csv(quality_path, index=False)

    eda_tables = build_eda_tables(featured)
    table_paths = save_eda_outputs(eda_tables)
    figure_paths = save_all_figures(featured)
    model_metrics = None if args.skip_model else train_cancellation_model(featured)

    target_counts = eda_tables["target_counts"]
    top_corr = eda_tables["cancellation_correlations"].head(8)
    summary = {
        "data_path": str(args.data),
        "runtime_seconds": round(time.time() - start, 2),
        "raw_profile": raw_profile,
        "schema": schema,
        "rows_after_cleaning": int(featured.shape[0]),
        "columns_after_features": int(featured.shape[1]),
        "target_distribution": target_counts.to_dict(orient="index"),
        "top_correlations_with_cancellation": {
            idx: float(row["corr_with_cancel"]) for idx, row in top_corr.iterrows()
        },
        "quality_checks_path": str(quality_path),
        "tables": table_paths,
        "figures": figure_paths,
        "model": model_metrics,
        "dashboard": "streamlit run streamlit_app.py",
    }

    summary_path = OUTPUT_DIR / "analysis_summary.json"
    write_summary_json(summary, summary_path)
    write_markdown_report(summary)

    print(f"Completed hotel booking analysis in {summary['runtime_seconds']} seconds")
    print(f"Summary: {summary_path}")


def write_markdown_report(summary: dict) -> None:
    target_lines = []
    for label, values in summary["target_distribution"].items():
        target_lines.append(f"- {label}: {values['count']:,} rows ({values['rate']:.2%})")

    corr_lines = [
        f"- {name}: {value:.4f}" for name, value in summary["top_correlations_with_cancellation"].items()
    ]

    text = "\n".join(
        [
            "# Hotel Booking Cancellation Analysis",
            "",
            "## Dataset",
            f"- Rows: {summary['raw_profile']['rows']:,}",
            f"- Columns: {summary['raw_profile']['columns']:,}",
            f"- Missing cells: {summary['raw_profile']['missing_cells']:,}",
            f"- Duplicate rows: {summary['raw_profile']['duplicate_rows']:,}",
            "- Cleaning note: duplicate-looking rows are retained because the dataset has no unique booking ID.",
            "",
            "## Cancellation distribution",
            *target_lines,
            "",
            "## Strongest linear signals vs cancellation",
            *corr_lines,
            "",
            "## Prediction model",
            *model_lines(summary.get("model")),
            "",
            "## Dashboard",
            "Use the Streamlit dashboard for EDA, segment drilldown, and cancellation probability scoring:",
            "",
            "```powershell",
            "streamlit run streamlit_app.py",
            "```",
            "",
            "## Key artifacts",
            "- `outputs/analysis_summary.json`",
            "- `outputs/tables/numeric_summary.csv`",
            "- `outputs/tables/cancellation_correlations.csv`",
            "- `outputs/figures/monthly_cancellation_rate.png`",
            "- `outputs/models/hotel_cancellation_model.joblib`",
            "- `streamlit_app.py`",
            "",
        ]
    )
    (REPORTS_DIR / "analysis_report.md").write_text(text, encoding="utf-8")


def model_lines(metrics: dict | None) -> list[str]:
    if not metrics:
        return ["- Model training was skipped for this run."]
    return [
        f"- ROC AUC: {metrics['roc_auc']:.4f}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        "- Leakage control: `reservation_status`, `reservation_status_date`, and `assigned_room_type` are excluded from prediction features.",
    ]


if __name__ == "__main__":
    main()
