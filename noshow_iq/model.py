from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshow_iq.preprocess import FEATURE_COLUMNS, split_features_target


MODEL_PATH = Path("models/noshow_model.joblib")

CATEGORICAL_FEATURES = ["gender", "neighbourhood"]
NUMERIC_FEATURES = [
    "age",
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


def build_pipeline() -> Pipeline:
    """Build preprocessing + imbalance handling + classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train(
    csv_path: str = "data/KaggleV2-May-2016.csv",
    model_path: str | Path = MODEL_PATH,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train model, save artifact, and return metrics."""
    df = pd.read_csv(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["show", "no_show"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "imbalance_technique": "class_weight balanced",
        "classification_report": report,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return pipeline, metrics


def load_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    """Load saved model."""
    return joblib.load(model_path)


def predict(model: Pipeline, features: pd.DataFrame) -> Dict[str, Any]:
    """Predict risk probability, risk level, and recommendation."""
    probability = float(model.predict_proba(features)[0][1])

    if probability >= 0.65:
        risk_level = "high"
        recommendation = (
            "Call the patient and send a reminder. Consider overbooking carefully."
        )
    elif probability >= 0.40:
        risk_level = "medium"
        recommendation = "Send SMS reminder and confirm attendance before appointment."
    else:
        risk_level = "low"
        recommendation = "Standard reminder is enough."

    return {
        "risk_level": risk_level,
        "probability": round(probability, 4),
        "recommendation": recommendation,
    }


def evaluate(
    csv_path: str = "data/KaggleV2-May-2016.csv",
    model_path: str | Path = MODEL_PATH,
) -> Dict[str, Any]:
    """Evaluate saved model on a stratified holdout split."""
    df = pd.read_csv(csv_path)
    X, y = split_features_target(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = load_model(model_path)
    y_pred = model.predict(X_test)

    return classification_report(
        y_test,
        y_pred,
        target_names=["show", "no_show"],
        output_dict=True,
        zero_division=0,
    )