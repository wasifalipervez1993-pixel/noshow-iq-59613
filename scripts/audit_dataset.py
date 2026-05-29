from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DATA_PATH = Path("data/KaggleV2-May-2016.csv")
REPORT_PATH = Path("reports/dataset_audit_report.json")


def _safe_value_counts(series: pd.Series) -> dict[str, Any]:
    """Return value counts as plain JSON-safe dictionary."""
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).items()
    }


def main() -> None:
    """Audit the raw Kaggle no-show dataset column-by-column."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place KaggleV2-May-2016.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    scheduled = pd.to_datetime(
        df["ScheduledDay"],
        errors="coerce",
        utc=True,
    )
    appointment = pd.to_datetime(
        df["AppointmentDay"],
        errors="coerce",
        utc=True,
    )

    days_in_advance = (
        appointment.dt.normalize() - scheduled.dt.normalize()
    ).dt.days

    target_counts = df["No-show"].value_counts(dropna=False)
    target_percentages = (
        df["No-show"].value_counts(normalize=True, dropna=False)
        .round(4)
        .to_dict()
    )

    invalid_rows = df[
        (df["Age"] < 0) | (days_in_advance < 0)
    ].copy()

    report = {
        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "missing_values": {
            col: int(value) for col, value in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_appointment_ids": int(
            df["AppointmentID"].duplicated().sum()
        ),
        "repeated_patient_ids": int(df["PatientId"].duplicated().sum()),
        "unique_patient_ids": int(df["PatientId"].nunique()),
        "unique_appointment_ids": int(df["AppointmentID"].nunique()),
        "age_quality": {
            "minimum_age": int(df["Age"].min()),
            "maximum_age": int(df["Age"].max()),
            "negative_age_rows": int((df["Age"] < 0).sum()),
            "age_zero_rows": int((df["Age"] == 0).sum()),
            "age_above_100_rows": int((df["Age"] > 100).sum()),
        },
        "date_quality": {
            "invalid_scheduled_dates": int(scheduled.isna().sum()),
            "invalid_appointment_dates": int(appointment.isna().sum()),
            "appointment_before_scheduled_rows": int(
                (days_in_advance < 0).sum()
            ),
            "minimum_days_in_advance": int(days_in_advance.min()),
            "maximum_days_in_advance": int(days_in_advance.max()),
            "mean_days_in_advance": round(float(days_in_advance.mean()), 4),
            "median_days_in_advance": float(days_in_advance.median()),
        },
        "invalid_rows_removed_by_cleaning": int(len(invalid_rows)),
        "target_distribution": {
            str(key): int(value) for key, value in target_counts.items()
        },
        "target_distribution_percentage": {
            str(key): float(value)
            for key, value in target_percentages.items()
        },
        "gender_distribution": _safe_value_counts(df["Gender"]),
        "handcap_distribution": _safe_value_counts(df["Handcap"]),
        "handcap_above_one_rows": int((df["Handcap"] > 1).sum()),
        "neighbourhood_unique_count": int(df["Neighbourhood"].nunique()),
        "top_10_neighbourhoods": _safe_value_counts(
            df["Neighbourhood"].value_counts().head(10)
        ),
        "cleaning_decisions": {
            "remove_negative_age": True,
            "keep_age_zero": True,
            "remove_appointment_before_scheduled": True,
            "keep_repeated_patient_ids": True,
            "exclude_patient_id_from_features": True,
            "exclude_appointment_id_from_features": True,
            "keep_handcap_as_numeric_count": True,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
