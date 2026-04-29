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
    "is_weekend",
    "risk_group",
    "sms_effective",
    "wait_time_bin",
]


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Kaggle columns to clean snake_case names."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add production-safe engineered features."""
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
        df[chronic_cols].sum(
            axis=1) > 0).astype(int)

    df["is_weekend"] = df["appointment_weekday"].isin([5, 6]).astype(int)

    df["risk_group"] = (
        df["is_senior"] + df["has_chronic_condition"] + df["is_child"]
    )

    df["sms_effective"] = (
        df["sms_received"] * (df["days_in_advance"] > 1)
    ).astype(int)

    df["wait_time_bin"] = pd.cut(
        df["days_in_advance"],
        bins=[-1, 0, 3, 7, 30, 10_000],
        labels=["same_day", "short", "medium", "long", "very_long"],
    ).astype(str)

    return df


def clean_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """Clean raw data for training or prediction."""
    df = fix_column_names(df)
    df = df.copy()

    df = df[df["age"] >= 0]

    df = add_engineered_features(df)

    df = df[df["days_in_advance"] >= 0]

    if training and "no_show" in df.columns:
        df["no_show"] = df["no_show"].map({"No": 0, "Yes": 1}).astype(int)

    for col in ["gender", "neighbourhood", "wait_time_bin"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    binary_cols = [
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "sms_received",
        "is_same_day",
        "is_child",
        "is_senior",
        "has_chronic_condition",
        "is_weekend",
        "sms_effective",
    ]

    for col in binary_cols:
        df[col] = df[col].astype(int)

    df["handicap"] = df["handicap"].astype(int)
    df["risk_group"] = df["risk_group"].astype(int)

    return df


def split_features_target(df: pd.DataFrame):
    """Split cleaned dataframe into model features and target."""
    cleaned = clean_data(df, training=True)
    X = cleaned[FEATURE_COLUMNS]
    y = cleaned["no_show"]
    return X, y


def clean_single_record(record: dict) -> pd.DataFrame:
    """Clean one API input record."""
    raw = pd.DataFrame([record])
    cleaned = clean_data(raw, training=False)

    missing = [col for col in FEATURE_COLUMNS if col not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return cleaned[FEATURE_COLUMNS]
