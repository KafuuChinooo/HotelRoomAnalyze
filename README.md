# Hotel Booking Cancellation Dashboard

Exploratory data analysis and prediction project for the Kaggle Hotel Booking Demand dataset.

The project analyzes booking demand, cancellation patterns, and segment-level risk. It also
trains a supervised model to estimate the probability that a booking will be canceled.
## Project history

This project was originally developed earlier as a personal data analysis project and was later reorganized into this public repository for portfolio presentation. The current version focuses on a cleaner project structure, reproducible setup, model evaluation, and dashboard presentation.

## Dataset

- Source: [Hotel Booking Demand on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- File used: `Data/hotel_bookings.csv`
- Original rows: 119,390
- Target: `is_canceled`

The dataset is originally from the Hotel Booking Demand Datasets article by Antonio, Almeida,
and Nunes. This repository keeps the CSV locally because it is small enough for normal GitHub
limits.

## Features

- Data validation, type cleanup, missing-value handling, and quality checks.
- Feature engineering for lead time, stay length, guest mix, agency/company flags, and ADR per guest.
- Reproducible EDA tables, figures, notebook, and markdown report.
- Cancellation prediction model with leakage columns excluded.
- Professional Streamlit dashboard with donut, bar, stacked bar, line, heatmap, treemap, scatter, box, and histogram charts.
- AI explanation modal: click a KPI card, chart segment, or chart point, then confirm whether AI should explain the selected dashboard item.
- Prediction form for booking-level cancellation probability.

## Leakage Control

The model excludes fields that directly reveal or are too close to the final outcome:

- `reservation_status`
- `reservation_status_date`
- `assigned_room_type`

The prediction target is `is_canceled`.

## Project Structure

```text
.
|-- Data/
|   `-- hotel_bookings.csv
|-- notebooks/
|   `-- hotel_booking_analysis.ipynb
|-- outputs/
|   |-- analysis_summary.json
|   |-- figures/
|   |-- models/
|   |   `-- hotel_cancellation_model.joblib
|   |-- reports/
|   `-- tables/
|-- src/
|   |-- config.py
|   |-- load_data.py
|   |-- clean_data.py
|   |-- features.py
|   |-- eda.py
|   |-- visualize.py
|   |-- model.py
|   |-- run_analysis.py
|   `-- build_notebook.py
|-- streamlit_app.py
|-- requirements.txt
`-- LICENSE
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline

```powershell
python src/run_analysis.py
python src/build_notebook.py
```

Use another CSV:

```powershell
python src/run_analysis.py --data path\to\hotel_bookings.csv
```

Skip retraining when only refreshing EDA:

```powershell
python src/run_analysis.py --skip-model
```

## Run Dashboard

```powershell
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

To enable AI explanations, create a local `.env` file with:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-1.5-flash
```

If no key is configured, the dashboard shows an isolated local mock explanation. If the API call fails, the modal shows a retryable error.

## Current Model Results

Latest local run:

- ROC AUC: 0.9506
- Accuracy: 0.8755
- Precision: 0.8594
- Recall: 0.7939
- F1: 0.8253

Exact metrics are saved in `outputs/models/model_metrics.json`.

## Main Outputs

- `outputs/analysis_summary.json`
- `outputs/reports/analysis_report.md`
- `outputs/tables/cancellation_correlations.csv`
- `outputs/tables/cancellation_by_market_segment.csv`
- `outputs/figures/monthly_cancellation_rate.png`
- `outputs/figures/cancellation_by_categories.png`
- `outputs/models/hotel_cancellation_model.joblib`
- `notebooks/hotel_booking_analysis.ipynb`

## GitHub Notes

- Do not commit `.env`, `venv/`, `.venv/`, cache folders, or log files.
- The raw CSV is about 17 MB and is safe for normal GitHub limits.
- Generated `outputs/` are included so the EDA and model results are reviewable without rerunning the pipeline.

## License

MIT License. See [LICENSE](LICENSE).
