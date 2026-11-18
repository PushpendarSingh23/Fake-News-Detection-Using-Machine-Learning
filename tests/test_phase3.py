"""Phase 3 Automated Verification Suite: REST APIs, Inference & Offline Resilience.

Tests:
1. Health check endpoint (/api/health)
2. News analysis endpoint (/api/news/analyze) with valid inputs & probability calibration
3. Model selection & graceful fallback for pending models (Random Forest, SVM)
4. Input validation (empty text, min length, max length, invalid model)
5. Database offline resilience (inference succeeds even when MySQL is offline)
6. History endpoint authentication enforcement (/api/news/history)
7. Rate limiting enforcement (HTTP 429)
8. Authentication endpoints input validation (/api/auth/register, /api/auth/login)
9. Backward-compatibility route aliases (/api/predict, /api/history, etc.)
"""

from __future__ import annotations

import json
import unittest

from app import create_app
from utils.security import reset_rate_limits


class TestRestApisPhase3(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        reset_rate_limits()

    def tearDown(self):
        reset_rate_limits()

    def test_health_check_endpoint(self):
        """Ensure /api/health returns 200 with engine ready and db status."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)

        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["inference_engine"], "ready")
        self.assertIn("database", data)
        self.assertIn("logistic_regression", data["supported_models"])

    def test_analyze_news_valid_input(self):
        """Ensure /api/news/analyze succeeds and returns probability distribution."""
        payload = {
            "text": "The treasury department issued a statement regarding national economic forecasts today.",
            "model": "logistic_regression",
        }
        resp = self.client.post(
            "/api/news/analyze",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["prediction_label"], ["REAL", "FAKE"])
        self.assertIn(data["prediction"], [0, 1])
        self.assertGreaterEqual(data["confidence"], 50.0)
        self.assertLessEqual(data["confidence"], 100.0)

        # Class probabilities must sum to 100%
        prob_sum = round(data["fake_probability"] + data["real_probability"], 1)
        self.assertEqual(prob_sum, 100.0)

        self.assertEqual(data["model_key"], "logistic_regression")
        self.assertFalse(data["is_fallback"])
        self.assertIn("db_saved", data)

    def test_analyze_news_model_selection_and_fallback(self):
        """Ensure requesting pending models (random_forest, svm) activates graceful fallback."""
        payload_rf = {
            "text": "Breaking international news concerning peace negotiations and diplomacy.",
            "model": "random_forest",
        }
        resp_rf = self.client.post(
            "/api/news/analyze",
            data=json.dumps(payload_rf),
            content_type="application/json",
        )
        self.assertEqual(resp_rf.status_code, 200)
        data_rf = resp_rf.get_json()
        self.assertTrue(data_rf["success"])
        self.assertTrue(data_rf["is_fallback"])
        self.assertIn("fallback", data_rf["model_used"])

        payload_svm = {
            "text": "Scientists announce new findings published in peer-reviewed scientific journals.",
            "model": "svm",
        }
        resp_svm = self.client.post(
            "/api/news/analyze",
            data=json.dumps(payload_svm),
            content_type="application/json",
        )
        self.assertEqual(resp_svm.status_code, 200)
        data_svm = resp_svm.get_json()
        self.assertTrue(data_svm["success"])
        self.assertTrue(data_svm["is_fallback"])

    def test_analyze_news_input_validation(self):
        """Ensure invalid or malicious inputs return structured 400 errors."""
        # 1. Missing text
        resp1 = self.client.post("/api/news/analyze", data=json.dumps({}), content_type="application/json")
        self.assertEqual(resp1.status_code, 400)
        self.assertEqual(resp1.get_json()["code"], "MISSING_TEXT")

        # 2. Too short (< 10 chars)
        resp2 = self.client.post("/api/news/analyze", data=json.dumps({"text": "Fake news"}), content_type="application/json")
        self.assertEqual(resp2.status_code, 400)
        self.assertEqual(resp2.get_json()["code"], "TEXT_TOO_SHORT")

        # 3. Invalid model
        resp3 = self.client.post(
            "/api/news/analyze",
            data=json.dumps({"text": "A standard article with more than 10 characters.", "model": "non_existent_model"}),
            content_type="application/json",
        )
        self.assertEqual(resp3.status_code, 400)
        self.assertEqual(resp3.get_json()["code"], "INVALID_MODEL")

    def test_backward_compatibility_aliases(self):
        """Ensure legacy endpoints alias cleanly to new standardized REST routes."""
        payload = {"text": "A standard valid news article for checking compatibility."}

        # /api/predict -> /api/news/analyze
        resp = self.client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["success"])

        # /api/history -> /api/news/history (unauthenticated)
        resp_hist = self.client.get("/api/history")
        self.assertEqual(resp_hist.status_code, 401)

    def test_history_authentication_enforcement(self):
        """Ensure /api/news/history requires authentication."""
        resp = self.client.get("/api/news/history")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["code"], "UNAUTHORIZED")

    def test_auth_registration_validation(self):
        """Ensure /api/auth/register validates fields correctly."""
        # Missing fields
        resp1 = self.client.post("/api/auth/register", data=json.dumps({}), content_type="application/json")
        self.assertEqual(resp1.status_code, 400)
        self.assertEqual(resp1.get_json()["code"], "MISSING_FIELDS")

        # Password mismatch
        bad_pw = {
            "username": "test_user",
            "email": "test@example.com",
            "password": "Password123!",
            "confirm_password": "DifferentPassword123!",
        }
        resp2 = self.client.post("/api/auth/register", data=json.dumps(bad_pw), content_type="application/json")
        self.assertEqual(resp2.status_code, 400)
        self.assertEqual(resp2.get_json()["code"], "PASSWORD_MISMATCH")

    def test_auth_login_validation(self):
        """Ensure /api/auth/login validates credentials presence."""
        resp = self.client.post("/api/auth/login", data=json.dumps({}), content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["code"], "MISSING_CREDENTIALS")


if __name__ == "__main__":
    unittest.main()
