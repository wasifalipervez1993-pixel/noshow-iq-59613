from __future__ import annotations

import json

from noshow_iq.database import insert_training_run
from noshow_iq.model import train


def main() -> None:
    """Train model and store metrics in MongoDB when available."""
    print("Starting NoShowIQ model training...")

    model_bundle, metrics = train()

    model = model_bundle["model"]

    # try:
    #     importances = model.named_steps["classifier"].feature_importances_
    #     print("\nTop Feature Importances:")
    #     print(sorted(importances, reverse=True)[:5])
    # except Exception:
    #     print("Feature importance not available for this model")

    threshold = model_bundle.get("threshold", 0.5)

    print("\n===== TRAINING COMPLETED =====")
    print(f"Selected model: {model_bundle.get('selected_model')}")
    print(f"Decision threshold: {threshold:.4f}")

    print("\n===== MODEL METRICS =====")
    print(json.dumps(metrics, indent=2))

    try:
        inserted_id = insert_training_run(metrics)
        print("\n===== MONGODB LOGGING =====")
        print(f"Training run stored in MongoDB: {inserted_id}")
    except Exception as exc:
        print("\n===== MONGODB LOGGING WARNING =====")
        print(f"Could not store training run in MongoDB: {exc}")


if __name__ == "__main__":
    main()
