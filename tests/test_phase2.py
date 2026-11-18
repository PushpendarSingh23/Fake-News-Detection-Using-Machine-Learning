"""Phase 2 Automated Verification Suite: ML Engine, Deduplication & Inference.

Tests:
1. Text Preprocessing & Publisher Leakage Removal
2. Preprocessing Train-Serve Parity (zero-drift guarantee)
3. Dataset Availability Checker & Blocker Identification
4. Honesty of Evaluation Metadata (no fabricated metrics)
5. Predictor Loading & Logistic Regression Inference
6. Calibrated Probability & Confidence Constraints
7. Graceful Multi-Model Fallback when artifacts are pending
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from predict import get_predictor, MODEL_DISPLAY_NAMES
from train_model import check_dataset_availability
from utils.text_processing import (
    clean_raw_text,
    preprocess_text,
    _WIRE_SERVICE_PATTERN,
)


class TestTextPreprocessing(unittest.TestCase):
    def test_wire_service_leakage_removal(self):
        """Ensure wire service prefixes (Reuters, AP, Bloomberg) are cleanly stripped."""
        text_reuters = "WASHINGTON (Reuters) - Federal officials are investigating economic data."
        cleaned = clean_raw_text(text_reuters, strip_wire_tags=True)
        self.assertNotIn("reuters", cleaned)
        self.assertNotIn("washington", cleaned)
        self.assertTrue(cleaned.startswith("federal officials"))

        text_ap = "NEW YORK (AP) -- Stock markets rallied on Wednesday afternoon."
        cleaned_ap = clean_raw_text(text_ap, strip_wire_tags=True)
        self.assertNotIn("ap", cleaned_ap)
        self.assertTrue(cleaned_ap.startswith("stock markets"))

    def test_url_and_special_character_cleaning(self):
        """Ensure URLs, HTML tags, and non-alphanumeric characters are stripped."""
        raw = "Check out this link: https://example.com/breaking-news! <p>Breaking alert @user.</p>"
        cleaned = clean_raw_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertIn("breaking alert", cleaned)

    def test_lemmatization_and_stopwords(self):
        """Ensure stopwords are removed and words are properly lemmatized."""
        text = "The politicians were delivering speeches about various policies"
        processed = preprocess_text(text)
        # 'politicians' -> 'politician', 'speeches' -> 'speech', 'policies' -> 'policy'
        # 'the', 'were', 'about' (stopwords) removed
        self.assertIn("politician", processed)
        self.assertIn("speech", processed)
        self.assertIn("policy", processed)
        self.assertNotIn("the", processed.split())
        self.assertNotIn("were", processed.split())

    def test_train_serve_preprocessing_parity(self):
        """Ensure preprocessing used by Predictor is identical to offline training pipeline."""
        sample = "LONDON (Reuters) - Scientists discover new treatment options for patients."
        direct_output = preprocess_text(sample)

        predictor = get_predictor()
        # Verify predict internal preprocessing produces the exact same string
        res = predictor.predict(sample, model_key="logistic_regression")
        self.assertEqual(res["processed_preview"], direct_output[:200])


class TestDatasetAndEvaluationIntegrity(unittest.TestCase):
    def test_dataset_blocker_detection(self):
        """Ensure dataset availability returns False when genuine ISOT CSVs are absent."""
        available, fake_path, true_path = check_dataset_availability()
        # In current workspace, files are absent
        self.assertFalse(available, "Dataset should report False when genuine CSVs are not present")
        self.assertEqual(fake_path.name, "Fake.csv")
        self.assertEqual(true_path.name, "True.csv")

    def test_evaluation_metrics_no_fabricated_data(self):
        """Ensure evaluation_metrics.json does not claim verified accuracy when dataset is missing."""
        metrics_file = Path("models/evaluation_metrics.json")
        self.assertTrue(metrics_file.exists(), "evaluation_metrics.json must exist")

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("status"), "dataset_pending")
        self.assertEqual(data.get("verification_status"), "unverified")

        # Models should have accuracy=None (never fabricated)
        lr_spec = data["models"]["logistic_regression"]
        self.assertIsNone(lr_spec.get("accuracy"), "Unverified model must have null accuracy")
        self.assertFalse(lr_spec.get("verified"))

        rf_spec = data["models"]["random_forest"]
        self.assertEqual(rf_spec.get("status"), "pending_dataset")
        self.assertIsNone(rf_spec.get("accuracy"))

        svm_spec = data["models"]["svm"]
        self.assertEqual(svm_spec.get("status"), "pending_dataset")
        self.assertIsNone(svm_spec.get("accuracy"))


class TestPredictorInference(unittest.TestCase):
    def setUp(self):
        self.predictor = get_predictor()

    def test_predictor_loads_existing_artifacts(self):
        """Verify Predictor successfully loads existing Logistic Regression and TF-IDF artifacts."""
        available = self.predictor.get_available_models()
        self.assertIn("logistic_regression", available)
        self.assertEqual(available["logistic_regression"], "Logistic Regression")

    def test_prediction_output_contract(self):
        """Verify inference returns valid contract: label, confidence %, and probabilities."""
        text = "Congress passed a bipartisan budget bill late Thursday evening to fund federal agencies."
        res = self.predictor.predict(text, model_key="logistic_regression")

        self.assertIn(res["prediction_label"], ["REAL", "FAKE"])
        self.assertIn(res["prediction"], [0, 1])
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertLessEqual(res["confidence"], 100.0)

        # Probabilities sum to 100%
        prob_sum = round(res["fake_probability"] + res["real_probability"], 1)
        self.assertEqual(prob_sum, 100.0)

    def test_empty_input_validation(self):
        """Ensure empty or whitespace input raises ValueError."""
        with self.assertRaises(ValueError):
            self.predictor.predict("")
        with self.assertRaises(ValueError):
            self.predictor.predict("   \n\t  ")

    def test_graceful_fallback_for_pending_models(self):
        """Ensure requesting pending models (random_forest, svm) falls back cleanly without crashing."""
        text = "Breaking report on foreign policy and trade agreements."

        # Request random_forest
        res_rf = self.predictor.predict(text, model_key="random_forest")
        self.assertIn(res_rf["prediction_label"], ["REAL", "FAKE"])
        self.assertTrue(res_rf["is_fallback"])
        self.assertIn("fallback", res_rf["model_used"])

        # Request svm
        res_svm = self.predictor.predict(text, model_key="svm")
        self.assertIn(res_svm["prediction_label"], ["REAL", "FAKE"])
        self.assertTrue(res_svm["is_fallback"])
        self.assertIn("fallback", res_svm["model_used"])


if __name__ == "__main__":
    unittest.main()
