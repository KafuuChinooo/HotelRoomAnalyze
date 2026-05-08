from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "Data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

DEFAULT_DATA_PATH = DATA_DIR / "hotel_bookings.csv"
TARGET = "is_canceled"

MONTH_ORDER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

NUMERIC_COLUMNS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "agent",
    "company",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

CATEGORICAL_COLUMNS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "reservation_status",
    "reservation_status_date",
]

EXPECTED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET]
LEAKAGE_COLUMNS = [TARGET, "reservation_status", "reservation_status_date", "assigned_room_type"]
MODEL_FEATURE_COLUMNS = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_month_num",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "has_children",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "has_agent",
    "has_company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "adr_per_person",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reserved_room_type",
]


def ensure_directories() -> None:
    for path in [
        DATA_DIR,
        OUTPUT_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        NOTEBOOKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
