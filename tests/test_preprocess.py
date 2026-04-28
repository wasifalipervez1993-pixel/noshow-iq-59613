import pandas as pd

from noshow_iq.preprocess import clean_data, clean_single_record


def sample_record():
    return {
        "PatientId": 29872499824296,
        "AppointmentID": 5642903,
        "Gender": "F",
        "ScheduledDay": "2016-04-29T18:38:08Z",
        "AppointmentDay": "2016-04-29T00:00:00Z",
        "Age": 62,
        "Neighbourhood": "JARDIM DA PENHA",
        "Scholarship": 0,
        "Hipertension": 1,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 0,
        "No-show": "No",
    }


def test_clean_data_renames_target_column():
    df = pd.DataFrame([sample_record()])
    cleaned = clean_data(df)
    assert "no_show" in cleaned.columns
    assert "hypertension" in cleaned.columns
    assert "handicap" in cleaned.columns


def test_days_in_advance_feature_exists():
    df = pd.DataFrame([sample_record()])
    cleaned = clean_data(df)
    assert "days_in_advance" in cleaned.columns


def test_negative_age_removed():
    row = sample_record()
    row["Age"] = -1
    df = pd.DataFrame([row])
    cleaned = clean_data(df)
    assert cleaned.empty


def test_clean_single_record_returns_features_only():
    row = sample_record()
    row.pop("No-show")
    features = clean_single_record(row)
    assert features.shape[0] == 1
    assert "days_in_advance" in features.columns
