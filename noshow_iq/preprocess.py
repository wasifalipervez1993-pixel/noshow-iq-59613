from __future__ import annotations

from typing import Dict, List

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


FEATURE_COLUMNS: List[str] = [
    "gender",
    "age",
    "neighbourhood",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "sms_received",
    "days_in_advance",
    "appointment_weekday",
    "scheduled_weekday",
    "is_same_day",
    "is_child",
    "is_senior",
    "has_chronic_condition",
]


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Kaggle columns to clean snake_case format."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features required for modeling."""
    df = df.copy()

    df["scheduled_day"] = pd.to_datetime(
        df["scheduled_day"], errors="coerce", utc=True
    )
    df["appointment_day"] = pd.to_datetime(
        df["appointment_day"], errors="coerce", utc=True
    )

    scheduled_dates = df["scheduled_day"].dt.date
    appointment_dates = df["appointment_day"].dt.date

    df["days_in_advance"] = (
        appointment_dates - scheduled_dates
    ).apply(lambda delta: delta.days)

    df["appointment_weekday"] = df["appointment_day"].dt.weekday
    df["scheduled_weekday"] = df["scheduled_day"].dt.weekday

    df["is_same_day"] = (df["days_in_advance"] == 0).astype(int)
    df["is_child"] = (df["age"] < 12).astype(int)
    df["is_senior"] = (df["age"] >= 60).astype(int)

    chronic_cols = ["hypertension", "diabetes", "alcoholism", "handicap"]
    df["has_chronic_condition"] = (
        df[chronic_cols].sum(axis=1) > 0
    ).astype(int)

    return df


def clean_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Clean dataset and prepare it for modeling.

    Key decisions:
    - Rename inconsistent column names
    - Remove invalid negative ages
    - Remove invalid date records (appointment before scheduling)
    - Keep age=0 (valid for infants)
    - Encode target variable for training
    """
    df = fix_column_names(df)
    df = df.copy()

    if "age" in df.columns:
        df = df[df["age"] >= 0]

    df = add_engineered_features(df)

    df = df[df["days_in_advance"] >= 0]

    if training and "no_show" in df.columns:
        df["no_show"] = df["no_show"].map({"No": 0, "Yes": 1}).astype(int)

    for col in ["gender", "neighbourhood"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    binary_cols = [
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "sms_received",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if "handicap" in df.columns:
        df["handicap"] = df["handicap"].astype(int)

    return df


def split_features_target(df: pd.DataFrame):
    """Split cleaned dataframe into X and y."""
    cleaned = clean_data(df, training=True)
    X = cleaned[FEATURE_COLUMNS]
    y = cleaned["no_show"]
    return X, y


def clean_single_record(record: dict) -> pd.DataFrame:
    """Prepare a single API input record for prediction."""
    raw = pd.DataFrame([record])
    cleaned = clean_data(raw, training=False)

    missing = [
        col for col in FEATURE_COLUMNS if col not in cleaned.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required feature columns after cleaning: {missing}"
        )

    return cleaned[FEATURE_COLUMNS]
