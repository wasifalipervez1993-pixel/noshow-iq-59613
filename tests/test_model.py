import pandas as pd

from noshow_iq.model import build_pipeline
from noshow_iq.preprocess import split_features_target


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert pipeline.named_steps["classifier"].class_weight == "balanced"


def test_split_features_target_returns_x_y():
    df = pd.DataFrame(
        [
            {
                "PatientId": 1,
                "AppointmentID": 1,
                "Gender": "F",
                "ScheduledDay": "2016-04-29T18:38:08Z",
                "AppointmentDay": "2016-04-30T00:00:00Z",
                "Age": 30,
                "Neighbourhood": "CENTRO",
                "Scholarship": 0,
                "Hipertension": 0,
                "Diabetes": 0,
                "Alcoholism": 0,
                "Handcap": 0,
                "SMS_received": 1,
                "No-show": "Yes",
            }
        ]
    )

    X, y = split_features_target(df)
    assert X.shape[0] == 1
    assert y.iloc[0] == 1
