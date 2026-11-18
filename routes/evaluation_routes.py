"""Evaluation View Controller for Fake News Detection System.

Renders precomputed multi-model evaluation metrics:
- Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC
- Non-blocking: reads directly from precomputed JSON cache.
"""

from __future__ import annotations

from flask import Blueprint, render_template

from utils.decorators import admin_required
from utils.model_evaluation import get_model_evaluation_payload

evaluation_bp = Blueprint("evaluation", __name__)


@evaluation_bp.route("/admin/evaluation")
@admin_required
def evaluation_page():
    """Render the precomputed model evaluation comparison dashboard for admin."""
    payload = get_model_evaluation_payload()
    return render_template(
        "evaluation.html",
        models=payload.get("models", {}),
        status=payload.get("status"),
        verification_status=payload.get("verification_status"),
        note=payload.get("note"),
        error=payload.get("error"),
    )


@evaluation_bp.route("/evaluation")
def public_evaluation_page():
    """Render the model evaluation comparison dashboard for users and reviewers."""
    payload = get_model_evaluation_payload()
    return render_template(
        "evaluation.html",
        models=payload.get("models", {}),
        status=payload.get("status"),
        verification_status=payload.get("verification_status"),
        note=payload.get("note"),
        error=payload.get("error"),
    )
