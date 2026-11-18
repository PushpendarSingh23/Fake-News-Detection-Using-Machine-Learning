"""Phase 4 Automated Verification Suite: Route Bug Fixes & Evaluation View.

Tests:
1. Case-Insensitive REAL/FAKE label counting (uppercase, lowercase, mixed-case, whitespace, NULL)
2. Chart data builder label normalization (build_real_fake_pie_chart & build_history_chart)
3. Dynamic Average Confidence Calculation (populated vs empty history)
4. Evaluation Blueprint Registration & Non-blocking Precomputed Cache
5. Evaluation Route Authentication Guarding (@admin_required)
6. Robustness to Missing, Malformed, and Valid evaluation_metrics.json
7. Model Evaluation Integrity (strictly no fabricated metrics)
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import unittest
from unittest.mock import MagicMock, patch

from app import create_app
from utils.charts import build_history_chart, build_real_fake_pie_chart
from utils.model_evaluation import (
    load_precomputed_evaluation,
    get_model_evaluation_payload,
)


class TestDashboardLabelNormalization(unittest.TestCase):
    def test_pie_chart_handles_mixed_case_and_unexpected_labels(self):
        """Ensure pie chart sums REAL and FAKE regardless of casing or unexpected labels."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()

        # Database rows with mixed cases, whitespace, and null/unexpected values
        mock_cursor.fetchall.return_value = [
            {"label": "REAL", "cnt": 5},
            {"label": "real", "cnt": 3},
            {"label": "Real ", "cnt": 2},
            {"label": "FAKE", "cnt": 4},
            {"label": "fake", "cnt": 6},
            {"label": "Fake", "cnt": 1},
            {"label": None, "cnt": 99},          # Should be safely ignored
            {"label": "UNVERIFIED", "cnt": 10},  # Should be safely ignored
        ]
        mock_db.cursor.return_value = mock_cursor

        chart = build_real_fake_pie_chart(mock_db, user_id=1)

        labels = chart["labels"]
        data = chart["datasets"][0]["data"]

        self.assertEqual(labels, ["Real", "Fake"])
        # Real = 5 + 3 + 2 = 10
        self.assertEqual(data[0], 10, "Real count must sum all uppercase/lowercase/mixed-case variants")
        # Fake = 4 + 6 + 1 = 11
        self.assertEqual(data[1], 11, "Fake count must sum all uppercase/lowercase/mixed-case variants")

    def test_history_chart_label_normalization(self):
        """Ensure time series line chart sums Real and Fake per day regardless of casing."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchall.return_value = [
            {"day": "2026-11-18", "label": "real", "cnt": 4},
            {"day": "2026-11-18", "label": "REAL", "cnt": 2},
            {"day": "2026-11-18", "label": "FAKE", "cnt": 5},
            {"day": "2026-11-19", "label": "Fake", "cnt": 3},
            {"day": "2026-11-19", "label": "REAL", "cnt": 7},
        ]
        mock_db.cursor.return_value = mock_cursor

        chart = build_history_chart(mock_db, user_id=1)

        self.assertEqual(chart["labels"], ["2026-11-18", "2026-11-19"])
        real_series = chart["datasets"][0]["data"]
        fake_series = chart["datasets"][1]["data"]

        # Day 1: Real = 4 + 2 = 6, Fake = 5
        self.assertEqual(real_series[0], 6)
        self.assertEqual(fake_series[0], 5)

        # Day 2: Real = 7, Fake = 3
        self.assertEqual(real_series[1], 7)
        self.assertEqual(fake_series[1], 3)


class TestDynamicAverageConfidence(unittest.TestCase):
    def test_dynamic_confidence_calculation_populated(self):
        """Verify average confidence calculation with populated history records."""
        app = create_app()
        mock_db = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchone.return_value = {
            "total": 3,
            "fake_cnt": 1,
            "real_cnt": 2,
            "avg_confidence": 85.333333,
        }
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value = mock_cursor

        with app.test_request_context():
            with patch("routes.dashboard_routes.get_db", return_value=mock_db):
                with patch("routes.dashboard_routes.session", {"user_id": 1}):
                    from routes.dashboard_routes import index
                    # Call index view
                    resp = index()
                    # Verify rendered HTML contains formatted percentage: 85.33%
                    self.assertIn("85.33%", resp)
                    self.assertNotIn("0.00%", resp)

    def test_dynamic_confidence_calculation_empty_history(self):
        """Verify average confidence handles empty history cleanly without division-by-zero."""
        app = create_app()
        mock_db = MagicMock()
        mock_cursor = MagicMock()

        mock_cursor.fetchone.return_value = {
            "total": 0,
            "fake_cnt": 0,
            "real_cnt": 0,
            "avg_confidence": 0.0,
        }
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value = mock_cursor

        with app.test_request_context():
            with patch("routes.dashboard_routes.get_db", return_value=mock_db):
                with patch("routes.dashboard_routes.session", {"user_id": 1}):
                    from routes.dashboard_routes import index
                    resp = index()
                    self.assertIn("0.00%", resp)
                    self.assertIn("No predictions yet.", resp)


class TestEvaluationViewAndIntegrity(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_evaluation_blueprint_registered(self):
        """Ensure evaluation_bp is registered in Flask app."""
        self.assertIn("evaluation", self.app.blueprints)

    def test_evaluation_route_unauthenticated_guard(self):
        """Ensure /admin/evaluation requires admin session, redirecting guests to admin login."""
        resp = self.client.get("/admin/evaluation")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admin/login", resp.headers["Location"])

    def test_evaluation_route_admin_authenticated(self):
        """Ensure authenticated admin can view precomputed evaluation metrics instantly."""
        with self.client.session_transaction() as sess:
            sess["is_admin"] = True
            sess["admin_id"] = 1
            sess["admin_name"] = "admin"

        resp = self.client.get("/admin/evaluation")
        self.assertEqual(resp.status_code, 200)
        content = resp.get_data(as_text=True)

        self.assertIn("AI Model Evaluation & Benchmark Audit", content)
        self.assertIn("Logistic Regression", content)
        self.assertIn("Random Forest", content)
        self.assertIn("Support Vector Machine", content)

    def test_evaluation_json_missing_handling(self):
        """Ensure system handles missing evaluation_metrics.json gracefully without crashing."""
        with patch("utils.model_evaluation.Config.METRICS_PATH", "models/non_existent_file.json"):
            payload = get_model_evaluation_payload()
            self.assertEqual(payload["status"], "missing_metrics")
            self.assertIn("Metrics file not found", payload["error"])

    def test_evaluation_json_corrupt_handling(self):
        """Ensure system handles malformed JSON without crashing."""
        with NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp.write("{ invalid_json: true, broken... }")
            tmp_path = tmp.name

        try:
            with patch("utils.model_evaluation.Config.METRICS_PATH", tmp_path):
                payload = get_model_evaluation_payload()
                self.assertEqual(payload["status"], "corrupt_metrics")
                self.assertIn("JSON decode error", payload["error"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_distinguish_null_from_zero_metrics(self):
        """Ensure None (pending) is distinguished from genuine zero."""
        payload = get_model_evaluation_payload()
        rf = payload["models"]["random_forest"]

        # Because dataset training is pending, accuracy should be None, NOT 0.0
        self.assertIsNone(rf["accuracy"], "Pending model must have accuracy=None, never 0.0 or fabricated")
        self.assertFalse(rf["verified"])


if __name__ == "__main__":
    unittest.main()
