from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from noshow_iq.preprocess import FEATURE_COLUMNS, split_features_target


MODEL_PATH = Path("models/noshow_model.joblib")

CATEGORICAL_FEATURES = [
    "gender",
    "age_bin",
    "neighbourhood",
    "wait_time_bin",
]

NUMERIC_FEATURES = [
    col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES
]


def build_pipeline() -> Pipeline:
    """Build production-ready preprocessing and classifier pipeline."""
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
        max_iter=300,
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
    """Choose threshold that maximizes F1-score for the no-show class."""
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_probability,
    )

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
    best_index = int(np.nanargmax(f1_scores[:-1]))

    return float(thresholds[best_index])


def build_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> Dict[str, Any]:
    """Create full JSON-safe evaluation report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=["show", "no_show"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "classification_report": report,
        "confusion_matrix": {
            "labels": ["show", "no_show"],
            "matrix": matrix,
        },
        "roc_auc": round(float(roc_auc_score(y_true, y_probability)), 4),
        "average_precision": round(
            float(average_precision_score(y_true, y_probability)),
            4,
        ),
    }


def get_top_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict[str, Any]] | str:
    """Return top permutation importances in JSON-safe format."""
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

        importance_pairs = list(zip(X_test.columns, result.importances_mean))
        importance_pairs.sort(key=lambda item: item[1], reverse=True)

        return [
            {
                "feature": str(feature),
                "importance": round(float(importance), 6),
            }
            for feature, importance in importance_pairs[:10]
        ]

    except Exception:
        return "not_available"


def train(
    csv_path: str = "data/KaggleV2-May-2016.csv",
    model_path: str | Path = MODEL_PATH,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train model, tune threshold on validation data, evaluate on test data,
    save model bundle, and return metrics.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Provide dataset locally or disable training in CI."
        )

    df = pd.read_csv(csv_path)
    X, y = split_features_target(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_probability = pipeline.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_probability)

    test_probability = pipeline.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= threshold).astype(int)

    evaluation = build_report(
        y_true=y_test,
        y_pred=test_prediction,
        y_probability=test_probability,
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
        "target_mapping": {
            "show": 0,
            "no_show": 1,
        },
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }

    metrics = {
        "training_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "selected_model": "HistGradientBoostingClassifier",
        "imbalance_technique": "class_weight balanced + validation threshold tuning",
        "decision_threshold": round(float(threshold), 4),
        "classification_report": evaluation["classification_report"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "roc_auc": evaluation["roc_auc"],
        "average_precision": evaluation["average_precision"],
        "top_feature_importance": top_features,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    return model_bundle, metrics


def load_model(model_path: str | Path = MODEL_PATH) -> Dict[str, Any]:
    """Load saved model bundle."""
    return joblib.load(model_path)


def predict(
    model_bundle: Dict[str, Any],
    features: pd.DataFrame,
) -> Dict[str, Any]:
    """Predict risk level, probability, confidence, and recommendation."""
    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.5))

    probability = float(model.predict_proba(features)[0][1])

    if probability >= threshold:
        risk_level = "high"
        recommendation = "Call patient and consider controlled overbooking."
        action_priority = "urgent"
    elif probability >= threshold * 0.65:
        risk_level = "medium"
        recommendation = "Send SMS reminder and confirm attendance."
        action_priority = "normal"
    else:
        risk_level = "low"
        recommendation = "Standard reminder is enough."
        action_priority = "normal"

    confidence = (
        "high"
        if probability >= 0.70
        else "medium"
        if probability >= 0.40
        else "low"
    )

    return {
        "risk_level": risk_level,
        "probability": round(probability, 4),
        "confidence": confidence,
        "recommendation": recommendation,
        "action_priority": action_priority,
    }


def evaluate(
    csv_path: str = "data/KaggleV2-May-2016.csv",
    model_path: str | Path = MODEL_PATH,
) -> Dict[str, Any]:
    """Evaluate saved model using the same final holdout strategy."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(csv_path)
    X, y = split_features_target(df)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model_bundle = load_model(model_path)
    model = model_bundle["model"]
    threshold = float(model_bundle.get("threshold", 0.5))

    y_probability = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probability >= threshold).astype(int)

    return build_report(
        y_true=y_test,
        y_pred=y_pred,
        y_probability=y_probability,
    )