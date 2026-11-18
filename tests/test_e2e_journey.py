"""Phase 5 End-to-End User Journey & Integration Test Suite.

Verifies the complete lifecycle of the Fake News Detection application:
1. Landing Page presentation & components (/ and /home).
2. Prediction Page rendering with dynamic model status metadata.
3. Unauthenticated guest prediction execution (zero-barrier demonstration).
4. Model selection & graceful fallback for pending offline weights.
5. Authenticated user prediction workflow (session persistence & history link).
6. File upload analysis (.txt document extraction & classification).
7. User Dashboard & History views with dynamic statistics.
8. Model Evaluation views (public /evaluation and admin /admin/evaluation).
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

from app import create_app
from predict import get_predictor


class TestEndToEndUserJourney(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["WTF_CSRF_ENABLED"] = False
        self.client = self.app.test_client()

    def test_01_landing_page_rendering(self):
        """Verify landing page renders with complete system overview and quick analyzer."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("AI-Assisted Fake News Detection", html)
        self.assertIn("Quick News Authenticity Analyzer", html)
        self.assertIn("Classification Model", html)
        self.assertIn("Deterministic NLP Pipeline", html)
        self.assertIn("End-to-End Detection Pipeline", html)
        self.assertIn("User Portal", html)
        self.assertIn("Admin Portal", html)

    def test_02_explicit_home_route(self):
        """Verify /home route is accessible to all users."""
        resp = self.client.get("/home")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("AI-Assisted Fake News Detection", html)

    def test_03_prediction_page_model_selection(self):
        """Verify /predict renders with loaded model statuses."""
        resp = self.client.get("/predict")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("News Authenticity Analyzer", html)
        self.assertIn("Logistic Regression", html)
        self.assertIn("Random Forest", html)
        self.assertIn("Support Vector Machine", html)
        self.assertIn("Active", html)
        self.assertIn("Pending Dataset Training", html)

    def test_04_guest_prediction_success(self):
        """Verify unauthenticated visitors can perform predictions in guest mode."""
        news_sample = "Government agency updates consumer guidelines for digital identity verification following parliamentary review."
        resp = self.client.post(
            "/predict",
            data={"text": news_sample, "model": "logistic_regression"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Analysis Result & Classification", html)
        self.assertTrue("REAL NEWS" in html or "FAKE NEWS" in html)
        self.assertIn("Model Confidence", html)
        self.assertIn("Class Probability Distribution", html)
        self.assertIn("Academic & Inference Disclaimer", html)
        self.assertIn("Guest Notice", html)

    def test_05_guest_empty_input_validation(self):
        """Verify empty input triggers a user-friendly validation flash message."""
        resp = self.client.post(
            "/predict",
            data={"text": "   ", "model": "logistic_regression"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)
        self.assertIn("Please provide news text or upload a supported file.", html)

    def test_06_model_selection_fallback_flow(self):
        """Verify requesting a model pending training falls back safely to active baseline."""
        news_sample = "Local community organization hosts annual regional agricultural exhibition."
        resp = self.client.post(
            "/predict",
            data={"text": news_sample, "model": "random_forest"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Analysis Result & Classification", html)
        self.assertIn("Model Fallback Active", html)
        self.assertIn("Logistic Regression", html)

    def test_07_file_upload_prediction(self):
        """Verify plain text file upload extracts text and performs classification."""
        sample_file_content = b"International economic conference concludes in Geneva with multilateral trade commitments."
        data = {
            "text": "",
            "model": "logistic_regression",
            "file": (io.BytesIO(sample_file_content), "test_article.txt"),
        }
        resp = self.client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("Analysis Result & Classification", html)
        self.assertIn("TXT", html)

    def test_08_authenticated_user_prediction_flow(self):
        """Verify authenticated user predictions record history and render user links."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = 42
            sess["username"] = "testuser"

        news_sample = "National space exploration committee schedules satellite deployment for upcoming quarter."
        resp = self.client.post(
            "/predict",
            data={"text": news_sample, "model": "logistic_regression"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Authenticated user should NOT see guest notice, and should see history links
        self.assertNotIn("Guest Notice", html)
        self.assertIn("View Prediction History", html)
        self.assertIn("Dashboard Analytics", html)

    def test_09_history_route_authentication_guard(self):
        """Verify /history requires active user login."""
        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.headers["Location"])

    def test_10_authenticated_history_and_dashboard_view(self):
        """Verify authenticated user can view history and dashboard with dynamic statistics."""
        mock_db = MagicMock()
        mock_cur = MagicMock()

        # Mock dashboard stats
        mock_cur.fetchone.return_value = {
            "total": 5,
            "fake_cnt": 2,
            "real_cnt": 3,
            "avg_confidence": 91.20,
            "cnt": 1,
        }
        mock_cur.fetchall.return_value = [
            {
                "id": 1,
                "model_name": "logistic_regression",
                "prediction_label": "REAL",
                "confidence": 92.50,
                "created_at": "2026-11-19 00:00:00",
                "input_filename": None,
                "input_text": "Sample verified article text",
            }
        ]
        mock_db.cursor.return_value = mock_cur

        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "scholar"

        with patch("routes.dashboard_routes.get_db", return_value=mock_db):
            resp = self.client.get("/")
            self.assertEqual(resp.status_code, 200)
            dash_html = resp.get_data(as_text=True)
            self.assertIn("scholar", dash_html)
            self.assertIn("91.20%", dash_html)

        with patch("routes.prediction_routes.get_db", return_value=mock_db):
            resp = self.client.get("/history")
            self.assertEqual(resp.status_code, 200)
            hist_html = resp.get_data(as_text=True)
            self.assertIn("Prediction history", hist_html)
            self.assertIn("logistic_regression", hist_html)

    def test_11_public_evaluation_page(self):
        """Verify /evaluation is publicly accessible for students and viva examiners."""
        resp = self.client.get("/evaluation")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("AI Model Evaluation & Benchmark Audit", html)
        self.assertIn("Logistic Regression", html)
        self.assertIn("Random Forest", html)
        self.assertIn("Support Vector Machine", html)
        self.assertIn("ROC-AUC", html)
        self.assertIn("confusion matrix", html.lower())

    def test_12_admin_evaluation_guard(self):
        """Verify /admin/evaluation requires admin privileges."""
        resp = self.client.get("/admin/evaluation")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/admin/login", resp.headers["Location"])


if __name__ == "__main__":
    unittest.main()
