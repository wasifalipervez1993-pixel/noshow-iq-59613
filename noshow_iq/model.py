from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshow_iq.preprocess import FEATURE_COLUMNS, split_features_target


MODEL_PATH = Path("models/noshow_model.joblib")

CATEGORICAL_FEATURES = ["gender", "neighbourhood", "wait_time_bin"]

NUMERIC_FEATURES = [
    col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES
]


def ensure_wait_time_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure wait_time_bin exists for training and prediction."""
    if "wait_time_bin" not in df.columns:
        df = df.copy()
        df["wait_time_bin"] = pd.cut(
            df["days_in_advance"],
            bins=[-1, 1, 7, 30, 10_000],
            labels=["same_day", "short", "medium", "long"],
        ).astype(str)

    return df


def build_pipeline() -> Pipeline:
    """Build production-ready ML pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def find_best_threshold(y_true, y_probability) -> float:
    """Optimize threshold for no-show F1-score."""
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_probability,
    )

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_index = int(np.nanargmax(f1_scores[:-1]))

    return float(thresholds[best_index])


def get_top_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict[str, Any]] | str:
    """Return top permutation importances in MongoDB/JSON-safe format."""
    try:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )

        feature_names = list(X_test.columns)
        importance_pairs = list(zip(feature_names, result.importances_mean))
        importance_pairs.sort(key=lambda item: item[1], reverse=True)

        return [
            {
                "feature": str(feature),
                "importance": round(float(importance), 6),
            }
            for feature, importance in importance_pairs[:5]
        ]

    except Exception:
        return "not_available"


def train(
    csv_path: str = "data/KaggleV2-May-2016.csv",
    model_path: str | Path = MODEL_PATH,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train, optimize threshold, save model, and return metrics."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Provide dataset locally or disable training in CI."
        )

    df = pd.read_csv(csv_path)
    X, y = split_features_target(df)
    X = ensure_wait_time_bin(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_probability = pipeline.predict_proba(X_test)[:, 1]
    base_threshold = find_best_threshold(y_test, y_probability)

    threshold = min(0.6, base_threshold + 0.03)
    y_pred = (y_probability >= threshold).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["show", "no_show"],
        output_dict=True,
        zero_division=0,
    )

    top_features = get_top_feature_importance(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
    )

    model_bundle = {
        "model": pipeline,
        "threshold": threshold,
        "feature_columns": list(X.columns),
        "selected_model": "HistGradientBoostingClassifier",
    }

    metrics = {
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "selected_model": "HistGradientBoostingClassifier",
        "imbalance_technique": "class_weight balanced + threshold tuning",
        "decision_threshold": round(float(threshold), 4),
        "classification_report": report,
        "top_feature_importance": top_features,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    return model_bundle, metrics


def load_model(model_path: str | Path = MODEL_PATH) -> Dict[str, Any]:
    """Load saved model bundle."""
    return joblib.load(model_path)


def predict(model_bundle: Dict[str, Any],
            features: pd.DataFrame) -> Dict[str, Any]:
    """Predict risk level, probability, and clinic recommendation."""
    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.5))

    features = ensure_wait_time_bin(features)
    probability = float(model.predict_proba(features)[0][1])

    if probability >= threshold:
        risk_level = "high"
        recommendation = "Call patient and consider controlled overbooking."
    elif probability >= threshold * 0.65:
        risk_level = "medium"
        recommendation = "Send SMS reminder and confirm attendance."
    else:
        risk_level = "low"
        recommendation = "Standard reminder is enough."

    confidence = (
        "high"
        if probability > 0.7
        else "medium"
        if probability > 0.4
        else "low"
    )

    return {
        "risk_level": risk_level,
        "probability": round(probability, 4),
        "confidence": confidence,
        "recommendation": recommendation,
        "action_priority": "urgent" if risk_level == "high" else "normal",
    }


def evaluate(
    csv_path: str = "data/KaggleV2-May-2016.csv",
    model_path: str | Path = MODEL_PATH,
) -> Dict[str, Any]:
    """Evaluate saved model using its stored threshold."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(csv_path)
    X, y = split_features_target(df)
    X = ensure_wait_time_bin(X)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_bundle = load_model(model_path)
    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.5))

    y_probability = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probability >= threshold).astype(int)

    return classification_report(
        y_test,
        y_pred,
        target_names=["show", "no_show"],
        output_dict=True,
        zero_division=0,
    )
