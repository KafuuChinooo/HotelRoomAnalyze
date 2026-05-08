from __future__ import annotations

import pandas as pd

from config import CATEGORICAL_COLUMNS, EXPECTED_COLUMNS, MONTH_ORDER, NUMERIC_COLUMNS, TARGET


def validate_schema(df: pd.DataFrame) -> dict:
    columns = set(col.strip().lower() for col in df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = sorted(expected - columns)
    extra = sorted(columns - expected)
    return {"missing_expected_columns": missing, "extra_columns": extra}


def clean_hotel_bookings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned = cleaned.reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "children" in cleaned.columns:
        cleaned["children"] = cleaned["children"].fillna(0)
    if "agent" in cleaned.columns:
        cleaned["agent"] = cleaned["agent"].fillna(0)
    if "company" in cleaned.columns:
        cleaned["company"] = cleaned["company"].fillna(0)
    if "country" in cleaned.columns:
        cleaned["country"] = cleaned["country"].replace({"nan": "Unknown", "": "Unknown"}).fillna("Unknown")

    if "arrival_date_month" in cleaned.columns:
        cleaned["arrival_month_num"] = cleaned["arrival_date_month"].map(MONTH_ORDER)

    date_cols = ["arrival_date_year", "arrival_month_num", "arrival_date_day_of_month"]
    if all(col in cleaned.columns for col in date_cols):
        cleaned["arrival_date"] = pd.to_datetime(
            {
                "year": cleaned["arrival_date_year"],
                "month": cleaned["arrival_month_num"],
                "day": cleaned["arrival_date_day_of_month"],
            },
            errors="coerce",
        )
    if "reservation_status_date" in cleaned.columns:
        cleaned["reservation_status_date"] = pd.to_datetime(cleaned["reservation_status_date"], errors="coerce")

    return cleaned


def clean_student_health(df: pd.DataFrame) -> pd.DataFrame:
    return clean_hotel_bookings(df)


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "is_canceled": (0, 1),
        "lead_time": (0, 800),
        "arrival_date_year": (2015, 2017),
        "arrival_date_week_number": (1, 53),
        "arrival_date_day_of_month": (1, 31),
        "stays_in_weekend_nights": (0, 20),
        "stays_in_week_nights": (0, 60),
        "adults": (0, 60),
        "children": (0, 10),
        "babies": (0, 10),
        "adr": (-10, 6000),
        "days_in_waiting_list": (0, 400),
        "required_car_parking_spaces": (0, 10),
        "total_of_special_requests": (0, 10),
    }
    rows = []
    for column, (low, high) in rules.items():
        if column not in df.columns:
            continue
        invalid = df[column].isna() | ~df[column].between(low, high)
        rows.append(
            {
                "column": column,
                "valid_min": low,
                "valid_max": high,
                "invalid_rows": int(invalid.sum()),
                "invalid_rate": round(float(invalid.mean()), 6),
            }
        )
    return pd.DataFrame(rows)
