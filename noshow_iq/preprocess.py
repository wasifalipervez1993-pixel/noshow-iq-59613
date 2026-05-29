from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


COLUMN_RENAME_MAP: Dict[str, str] = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}


RAW_REQUIRED_COLUMNS: List[str] = [
    "Gender",
    "ScheduledDay",
    "AppointmentDay",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received",
]


FEATURE_COLUMNS: List[str] = [
    "gender",
    "age",
    "age_bin",
    "neighbourhood",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "has_handicap",
    "sms_received",
    "days_in_advance",
    "wait_time_bin",
    "appointment_weekday",
    "scheduled_weekday",
    "appointment_month",
    "is_weekend",
    "is_same_day",
    "is_child",
    "is_senior",
    "chronic_count",
    "has_chronic_condition",
    "sms_effective",
    "risk_group",
]


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Kaggle columns to clean snake_case names."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def validate_required_columns(df: pd.DataFrame, training: bool = True) -> None:
    """Validate raw or already-renamed required columns."""
    available = set(df.columns)
    missing = []

    for raw_col in RAW_REQUIRED_COLUMNS:
        renamed_col = COLUMN_RENAME_MAP.get(raw_col, raw_col)

        if raw_col not in available and renamed_col not in available:
            missing.append(raw_col)

    if training and "No-show" not in available and "no_show" not in available:
        missing.append("No-show")

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical text columns for stable model input."""
    df = df.copy()

    for col in ["gender", "neighbourhood"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
        )

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric fields safely and handle invalid numeric values."""
    df = df.copy()

    numeric_cols = [
        "age",
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "handicap",
        "sms_received",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["age"].notna()]
    df = df[df["age"] >= 0]
    df["age"] = df["age"].astype(int)

    binary_cols = [
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "sms_received",
    ]

    for col in binary_cols:
        df[col] = df[col].fillna(0).clip(lower=0, upper=1).astype(int)

    # In this dataset, Handcap contains 0, 1, 2, 3, and 4.
    # Therefore, it is kept as a numeric count/severity indicator.
    df["handicap"] = df["handicap"].fillna(0).clip(lower=0).astype(int)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust, dashboard-safe engineered features."""
    df = df.copy()

    df["scheduled_day"] = pd.to_datetime(
        df["scheduled_day"],
        errors="coerce",
        utc=True,
    )
    df["appointment_day"] = pd.to_datetime(
        df["appointment_day"],
        errors="coerce",
        utc=True,
    )

    df = df[df["scheduled_day"].notna()]
    df = df[df["appointment_day"].notna()]

    scheduled_dates = df["scheduled_day"].dt.normalize()
    appointment_dates = df["appointment_day"].dt.normalize()

    df["days_in_advance"] = (appointment_dates - scheduled_dates).dt.days
    df = df[df["days_in_advance"].notna()]
    df = df[df["days_in_advance"] >= 0]
    df["days_in_advance"] = df["days_in_advance"].astype(int)

    df["appointment_weekday"] = df["appointment_day"].dt.weekday.astype(int)
    df["scheduled_weekday"] = df["scheduled_day"].dt.weekday.astype(int)
    df["appointment_month"] = df["appointment_day"].dt.month.astype(int)

    df["is_weekend"] = df["appointment_weekday"].isin([5, 6]).astype(int)
    df["is_same_day"] = (df["days_in_advance"] == 0).astype(int)

    df["is_child"] = (df["age"] < 12).astype(int)
    df["is_senior"] = (df["age"] >= 60).astype(int)

    df["age_bin"] = pd.cut(
        df["age"],
        bins=[-1, 12, 18, 40, 60, 120],
        labels=["CHILD", "TEEN", "ADULT", "MIDDLE_AGE", "SENIOR"],
    ).astype(str)

    df["wait_time_bin"] = pd.cut(
        df["days_in_advance"],
        bins=[-1, 0, 3, 7, 30, 10_000],
        labels=["SAME_DAY", "SHORT", "MEDIUM", "LONG", "VERY_LONG"],
    ).astype(str)

    df["has_handicap"] = (df["handicap"] > 0).astype(int)

    chronic_cols = [
        "hypertension",
        "diabetes",
        "alcoholism",
        "has_handicap",
    ]

    df["chronic_count"] = df[chronic_cols].sum(axis=1).astype(int)
    df["has_chronic_condition"] = (df["chronic_count"] > 0).astype(int)

    df["sms_effective"] = (
        df["sms_received"] * (df["days_in_advance"] > 1)
    ).astype(int)

    df["risk_group"] = (
        df["is_senior"] + df["has_chronic_condition"] + df["is_child"]
    ).astype(int)

    return df


def clean_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Clean raw appointment data for training, API prediction, and dashboard input.

    Decisions:
    - Rename raw Kaggle columns.
    - Remove negative ages.
    - Keep age zero because it can represent infants.
    - Remove invalid dates and negative waiting times.
    - Keep handicap as a numeric count/severity field.
    - Do not use patient_id or appointment_id as model features.
    """
    validate_required_columns(df, training=training)

    df = fix_column_names(df)
    df = df.copy()

    if "appointment_id" in df.columns:
        df = df.drop_duplicates(subset=["appointment_id"])

    df = normalize_categorical_columns(df)
    df = convert_numeric_columns(df)
    df = add_engineered_features(df)

    if training:
        df["no_show"] = (
            df["no_show"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"no": 0, "yes": 1, "0": 0, "1": 1})
        )

        df = df[df["no_show"].notna()]
        df["no_show"] = df["no_show"].astype(int)

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split cleaned training data into model features and target."""
    cleaned = clean_data(df, training=True)
    X = cleaned[FEATURE_COLUMNS]
    y = cleaned["no_show"]
    return X, y


def clean_single_record(record: dict) -> pd.DataFrame:
    """Clean one API/dashboard input record for prediction."""
    raw = pd.DataFrame([record])
    cleaned = clean_data(raw, training=False)

    if cleaned.empty:
        raise ValueError(
            "Input record became invalid after cleaning. "
            "Check age and appointment dates."
        )

    missing = [col for col in FEATURE_COLUMNS if col not in cleaned.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return cleaned[FEATURE_COLUMNS]