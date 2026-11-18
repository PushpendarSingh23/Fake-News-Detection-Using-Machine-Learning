from __future__ import annotations

import logging
from flask import Blueprint, render_template, session

from database import get_db
from predict import get_predictor
from utils.charts import build_history_chart, build_real_fake_pie_chart

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    # Public landing page for anonymous visitors
    if not session.get("user_id"):
        available_models = get_predictor().get_all_models_status()
        return render_template("index.html", available_models=available_models)

    user_id = session["user_id"]
    stats = {"total": 0, "fake_cnt": 0, "real_cnt": 0, "avg_confidence": 0.0}
    history = []
    chart = {"labels": [], "datasets": []}
    pie_chart = {"labels": ["Real", "Fake"], "datasets": []}
    total_users = 0
    dataset_total = 0

    try:
        db = get_db()
        cur = db.cursor(dictionary=True)

        # 1. Total, fake count, real count, and dynamic average confidence (case-insensitive & NULL-safe)
        cur.execute(
            """
            SELECT 
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN UPPER(TRIM(COALESCE(prediction_label, ''))) = 'FAKE' THEN 1 ELSE 0 END), 0) AS fake_cnt,
                COALESCE(SUM(CASE WHEN UPPER(TRIM(COALESCE(prediction_label, ''))) = 'REAL' THEN 1 ELSE 0 END), 0) AS real_cnt,
                COALESCE(AVG(COALESCE(confidence, 0.0)), 0.0) AS avg_confidence
            FROM prediction_history 
            WHERE user_id = %s
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if row:
            stats = {
                "total": int(row.get("total") or 0),
                "fake_cnt": int(row.get("fake_cnt") or 0),
                "real_cnt": int(row.get("real_cnt") or 0),
                "avg_confidence": round(float(row.get("avg_confidence") or 0.0), 2),
            }

        # 2. Recent predictions
        cur.execute(
            """
            SELECT id, model_name, prediction_label, confidence, created_at, input_filename 
            FROM prediction_history 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 10
            """,
            (user_id,),
        )
        history = cur.fetchall() or []
        cur.close()

        # 3. Chart data
        chart = build_history_chart(db, user_id)
        pie_chart = build_real_fake_pie_chart(db, user_id)

        # 4. Total dataset entries
        cur2 = db.cursor(dictionary=True)
        cur2.execute("SELECT COUNT(*) AS cnt FROM datasets")
        dataset_total_row = cur2.fetchone() or {}
        cur2.close()
        dataset_total = int(dataset_total_row.get("cnt") or 0)

        # 5. Total registered users
        cur3 = db.cursor(dictionary=True)
        cur3.execute("SELECT COUNT(*) AS cnt FROM users")
        total_users_row = cur3.fetchone() or {}
        cur3.close()
        total_users = int(total_users_row.get("cnt") or 0)

    except Exception as ex:
        logger.warning("Could not load full user dashboard stats from DB: %s", ex)

    return render_template(
        "user_dashboard.html",
        stats=stats,
        history=history,
        chart=chart,
        pie_chart=pie_chart,
        total_users=total_users,
        dataset_total=dataset_total,
    )


@dashboard_bp.route("/home")
def home():
    """Explicit landing page accessible to all users."""
    available_models = get_predictor().get_all_models_status()
    return render_template("index.html", available_models=available_models)
