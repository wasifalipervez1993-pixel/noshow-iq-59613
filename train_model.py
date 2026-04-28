from __future__ import annotations

import json

from noshow_iq.database import insert_training_run
from noshow_iq.model import train


def main():
    _, metrics = train()

    print(json.dumps(metrics, indent=2))

    try:
        inserted_id = insert_training_run(metrics)
        print(f"Training run stored in MongoDB: {inserted_id}")
    except Exception as exc:
        print(f"Warning: could not store training run in MongoDB: {exc}")


if __name__ == "__main__":
    main()