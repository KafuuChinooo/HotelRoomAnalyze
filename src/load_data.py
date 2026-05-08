from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_hotel_bookings(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def load_student_health(path: str | Path) -> pd.DataFrame:
    return load_hotel_bookings(path)


def profile_raw_data(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {col: int(val) for col, val in df.isna().sum().items()},
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
