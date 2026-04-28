from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import MongoClient


def get_mongo_uri() -> str:
    """Read MongoDB URI from environment only."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable is not configured.")
    return mongo_uri


def get_database():
    """Return MongoDB database."""
    client = MongoClient(get_mongo_uri())
    return client["noshow_iq"]


def now_utc() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def insert_prediction(document: Dict[str, Any]) -> Optional[str]:
    """Insert one prediction document."""
    db = get_database()
    result = db["predictions"].insert_one(document)
    return str(result.inserted_id)


def insert_training_run(metrics: Dict[str, Any]) -> Optional[str]:
    """Insert one model training run document."""
    db = get_database()
    document = {
        "timestamp": now_utc(),
        **metrics,
    }
    result = db["training_runs"].insert_one(document)
    return str(result.inserted_id)


def get_last_predictions(limit: int = 20):
    """Return latest prediction records."""
    db = get_database()
    cursor = (
        db["predictions"]
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_stats():
    """
    Return dashboard stats using MongoDB aggregation only.
    No Python-side counting is used.
    """
    db = get_database()

    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]
                    }
                },
                "medium_risk_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$risk_level", "medium"]}, 1, 0]
                    }
                },
                "low_risk_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]
                    }
                },
                "average_probability": {"$avg": "$probability"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "total_predictions": 1,
                "high_risk_count": 1,
                "medium_risk_count": 1,
                "low_risk_count": 1,
                "average_probability": {"$round": ["$average_probability", 4]},
            }
        },
    ]

    stats = list(db["predictions"].aggregate(pipeline))
    base_stats = stats[0] if stats else {
        "total_predictions": 0,
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": 0,
    }

    training_pipeline = [
        {"$sort": {"timestamp": -1}},
        {"$limit": 1},
        {
            "$project": {
                "_id": 0,
                "last_trained": "$timestamp",
            }
        },
    ]

    training = list(db["training_runs"].aggregate(training_pipeline))
    base_stats["last_trained"] = training[0]["last_trained"] if training else None

    return base_stats