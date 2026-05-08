# Hotel Booking Cancellation Analysis

## Dataset
- Rows: 119,390
- Columns: 32
- Missing cells: 129,425
- Duplicate rows: 31,994
- Cleaning note: duplicate-looking rows are retained because the dataset has no unique booking ID.

## Cancellation distribution
- 0: 75,166 rows (62.96%)
- 1: 44,224 rows (37.04%)

## Strongest linear signals vs cancellation
- lead_time: 0.2931
- total_of_special_requests: -0.2347
- required_car_parking_spaces: -0.1955
- booking_changes: -0.1444
- previous_cancellations: 0.1101
- has_agent: 0.1021
- has_company: -0.0993
- is_repeated_guest: -0.0848

## Prediction model
- ROC AUC: 0.9506
- Accuracy: 0.8755
- Precision: 0.8594
- Recall: 0.7939
- F1: 0.8253
- Leakage control: `reservation_status`, `reservation_status_date`, and `assigned_room_type` are excluded from prediction features.

## Dashboard
Use the Streamlit dashboard for EDA, segment drilldown, and cancellation probability scoring:

```powershell
streamlit run streamlit_app.py
```

## Key artifacts
- `outputs/analysis_summary.json`
- `outputs/tables/numeric_summary.csv`
- `outputs/tables/cancellation_correlations.csv`
- `outputs/figures/monthly_cancellation_rate.png`
- `outputs/models/hotel_cancellation_model.joblib`
- `streamlit_app.py`
