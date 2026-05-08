from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["total_nights"] = featured["stays_in_weekend_nights"] + featured["stays_in_week_nights"]
    featured["total_guests"] = featured["adults"] + featured["children"] + featured["babies"]
    featured["has_children"] = ((featured["children"] + featured["babies"]) > 0).astype(int)
    featured["has_agent"] = featured["agent"].fillna(0).gt(0).astype(int)
    featured["has_company"] = featured["company"].fillna(0).gt(0).astype(int)
    featured["adr_per_person"] = featured["adr"] / featured["total_guests"].replace(0, np.nan)
    featured["adr_per_person"] = featured["adr_per_person"].replace([np.inf, -np.inf], np.nan).fillna(0)

    featured["lead_time_band"] = pd.cut(
        featured["lead_time"],
        bins=[-np.inf, 7, 30, 90, 180, np.inf],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"],
    ).astype(str)
    featured["stay_length_band"] = pd.cut(
        featured["total_nights"],
        bins=[-np.inf, 1, 3, 7, 14, np.inf],
        labels=["0-1 nights", "2-3 nights", "4-7 nights", "8-14 nights", "15+ nights"],
    ).astype(str)
    featured["guest_mix"] = np.select(
        [
            featured["total_guests"].le(1),
            featured["has_children"].eq(1),
            featured["total_guests"].ge(3),
        ],
        ["solo", "family", "group"],
        default="couple",
    )

    return featured


def modeling_columns(df: pd.DataFrame, target: str, id_columns: list[str]) -> tuple[list[str], list[str]]:
    candidates = [col for col in df.columns if col not in id_columns + [target]]
    numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in candidates if col not in numeric]
    return numeric, categorical
