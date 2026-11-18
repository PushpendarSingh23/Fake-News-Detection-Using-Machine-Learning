"""Model Evaluation Loader & Reporter for Fake News Detection System.

Loads precomputed offline metrics from models/evaluation_metrics.json.
Guarantees:
- Never executes heavy training or vectorization during web requests.
- Handles missing, malformed, or incomplete JSON gracefully.
- Distinguishes genuine zero-valued metrics from missing/pending metrics (None).
- Renders transparent audit information for each classifier.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from config import Config

logger = logging.getLogger(__name__)


def load_precomputed_evaluation() -> Dict[str, Any]:
    """Read precomputed evaluation metrics from models/evaluation_metrics.json."""
    metrics_path = Path(Config.METRICS_PATH)

    if not metrics_path.exists():
        return {
            "status": "missing_metrics",
            "verification_status": "unverified",
            "note": "Evaluation metrics file (models/evaluation_metrics.json) does not exist yet. Run python train_model.py offline after acquiring the genuine ISOT dataset.",
            "models": {},
            "error": "Metrics file not found.",
        }

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as ex:
        logger.error("Evaluation metrics JSON is corrupted or invalid: %s", ex)
        return {
            "status": "corrupt_metrics",
            "verification_status": "error",
            "note": "Evaluation metrics file is corrupted or contains invalid JSON syntax.",
            "models": {},
            "error": f"JSON decode error: {ex}",
        }
    except Exception as ex:
        logger.error("Failed to read evaluation metrics file: %s", ex)
        return {
            "status": "read_error",
            "verification_status": "error",
            "note": "Failed to read evaluation metrics from disk.",
            "models": {},
            "error": str(ex),
        }

    return data


def get_model_evaluation_payload() -> Dict[str, Any]:
    """Prepare view-safe structured payload for templates and routes."""
    raw = load_precomputed_evaluation()

    models = raw.get("models", {})
    status = raw.get("status", "unknown")
    verification_status = raw.get("verification_status", "unverified")
    note = raw.get("note", "")
    error = raw.get("error")

    # Format models for template consumption
    formatted_models: Dict[str, Any] = {}
    for key, spec in models.items():
        is_verified = bool(spec.get("verified", False))
        acc = spec.get("accuracy")

        formatted_models[key] = {
            "model_key": key,
            "display_name": spec.get("display_name", key.replace("_", " ").title()),
            "artifact_path": spec.get("artifact_path", ""),
            "verified": is_verified,
            "status": spec.get("status", "unknown"),
            "accuracy": acc,
            "precision": spec.get("precision"),
            "recall": spec.get("recall"),
            "f1_score": spec.get("f1_score"),
            "roc_auc": spec.get("roc_auc"),
            "confusion_matrix": spec.get("confusion_matrix"),
            "classification_report": spec.get("classification_report"),
            "unverified_claimed_accuracy": spec.get("unverified_claimed_accuracy"),
        }

    return {
        "status": status,
        "verification_status": verification_status,
        "note": note,
        "models": formatted_models,
        "error": error,
    }
