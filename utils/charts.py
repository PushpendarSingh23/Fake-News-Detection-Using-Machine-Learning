"""Chart Data Builders for Fake News Detection System.

Builds structured Chart.js datasets for:
1. Historical prediction trend (Real vs Fake by day)
2. Overall Real vs Fake distribution pie chart

Features:
- Case-insensitive label normalization (handles 'REAL', 'real', 'FAKE', 'fake', etc.)
- Safe handling of unexpected, empty, or NULL labels
- Graceful empty dataset construction on database error
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_history_chart(db, user_id: int) -> dict[str, Any]:
    """Generate time series chart payload grouped by day and normalized label."""
    try:
        cur = db.cursor(dictionary=True)
        cur.execute(
            """
            SELECT DATE(created_at) AS day, 
                   UPPER(TRIM(COALESCE(prediction_label, ''))) AS label, 
                   COUNT(*) AS cnt 
            FROM prediction_history 
            WHERE user_id = %s 
            GROUP BY DATE(created_at), UPPER(TRIM(COALESCE(prediction_label, ''))) 
            ORDER BY day DESC LIMIT 14
            """,
            (user_id,),
        )
        rows = cur.fetchall() or []
        cur.close()
    except Exception as ex:
        logger.warning("Could not build history chart data: %s", ex)
        rows = []

    # Filter valid days and sort chronologically
    valid_days = sorted({str(r["day"]) for r in rows if r.get("day")})
    fake = [0] * len(valid_days)
    real = [0] * len(valid_days)

    day_index = {d: i for i, d in enumerate(valid_days)}
    for r in rows:
        d_str = str(r.get("day"))
        if d_str in day_index:
            idx = day_index[d_str]
            label = str(r.get("label") or "").strip().upper()
            cnt = int(r.get("cnt") or 0)
            if label == "FAKE":
                fake[idx] += cnt
            elif label == "REAL":
                real[idx] += cnt

    return {
        "labels": valid_days,
        "datasets": [
            {"label": "Real", "data": real, "borderColor": "#22c55e", "backgroundColor": "rgba(34,197,94,0.2)"},
            {"label": "Fake", "data": fake, "borderColor": "#ef4444", "backgroundColor": "rgba(239,68,68,0.2)"},
        ],
    }


def build_real_fake_pie_chart(db, user_id: int) -> dict[str, Any]:
    """Generate pie chart data with case-insensitive normalization."""
    try:
        cur = db.cursor(dictionary=True)
        cur.execute(
            """
            SELECT UPPER(TRIM(COALESCE(prediction_label, ''))) AS label, 
                   COUNT(*) AS cnt 
            FROM prediction_history 
            WHERE user_id = %s 
            GROUP BY UPPER(TRIM(COALESCE(prediction_label, '')))
            """,
            (user_id,),
        )
        rows = cur.fetchall() or []
        cur.close()
    except Exception as ex:
        logger.warning("Could not build pie chart data: %s", ex)
        rows = []

    real_cnt = 0
    fake_cnt = 0
    for r in rows:
        label = str(r.get("label") or "").strip().upper()
        cnt = int(r.get("cnt") or 0)
        if label == "REAL":
            real_cnt += cnt
        elif label == "FAKE":
            fake_cnt += cnt

    return {
        "labels": ["Real", "Fake"],
        "datasets": [
            {
                "label": "Real vs Fake",
                "data": [real_cnt, fake_cnt],
                "backgroundColor": ["rgba(34,197,94,0.8)", "rgba(239,68,68,0.8)"],
                "borderColor": ["#22c55e", "#ef4444"],
                "borderWidth": 1,
            }
        ],
    }
