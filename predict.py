"""Inference Engine for AI Fake News Detection System.

Features:
- Unified NLP text preprocessing (matches training pipeline exactly).
- Multi-model inference supporting:
    - 'logistic_regression' (default)
    - 'random_forest'
    - 'svm'
- Calibrated confidence estimation and class probabilities.
- Graceful fallback if specific model weights are pending offline training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
from config import Config
from utils.text_processing import clean_raw_text, preprocess_text

logger = logging.getLogger(__name__)

# Backward-compatibility aliases for legacy scripts
clean_text = clean_raw_text
preprocess = lambda text, stop_words=None, lemmatizer=None: preprocess_text(text)

_PREDICTOR_SINGLETON: Predictor | None = None

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
}


class Predictor:
    """Loads ML artifacts and performs multi-model predictions."""

    def __init__(self):
        self.models_dir = Path(Config.MODELS_FOLDER)
        vectorizer_path = self.models_dir / "vectorizer.pkl"

        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found:\n{vectorizer_path}\nRun train_model.py first."
            )

        self.vectorizer = joblib.load(vectorizer_path)
        self.models: Dict[str, Any] = {}
        self.model_names: Dict[str, str] = {}

        # 1. Load Logistic Regression (model.pkl or model_lr.pkl)
        lr_path = self.models_dir / "model_lr.pkl"
        if not lr_path.exists():
            lr_path = self.models_dir / "model.pkl"

        if lr_path.exists():
            try:
                self.models["logistic_regression"] = joblib.load(lr_path)
                self.model_names["logistic_regression"] = MODEL_DISPLAY_NAMES["logistic_regression"]
            except Exception as e:
                logger.error("Failed to load Logistic Regression model: %s", e)

        # 2. Load Random Forest if available
        rf_path = self.models_dir / "model_rf.pkl"
        if rf_path.exists():
            try:
                self.models["random_forest"] = joblib.load(rf_path)
                self.model_names["random_forest"] = MODEL_DISPLAY_NAMES["random_forest"]
            except Exception as e:
                logger.warning("Random Forest model found but failed to load: %s", e)

        # 3. Load SVM if available
        svm_path = self.models_dir / "model_svm.pkl"
        if svm_path.exists():
            try:
                self.models["svm"] = joblib.load(svm_path)
                self.model_names["svm"] = MODEL_DISPLAY_NAMES["svm"]
            except Exception as e:
                logger.warning("SVM model found but failed to load: %s", e)

        # Fallback check
        if not self.models:
            raise FileNotFoundError("No model weights found in models directory.")

        # Load metadata
        metadata_file = self.models_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}

    def get_available_models(self) -> Dict[str, str]:
        """Return dictionary of currently loaded models."""
        return dict(self.model_names)

    def get_all_models_status(self) -> list[dict]:
        """Return status metadata for all known models for UI display."""
        loaded = self.models
        return [
            {
                "key": "logistic_regression",
                "name": "Logistic Regression",
                "is_available": "logistic_regression" in loaded,
                "badge": "Active" if "logistic_regression" in loaded else "Unavailable",
                "badge_class": "badge--real" if "logistic_regression" in loaded else "badge--fake",
                "description": "Baseline linear classification model using L2 regularization.",
            },
            {
                "key": "random_forest",
                "name": "Random Forest",
                "is_available": "random_forest" in loaded,
                "badge": "Active" if "random_forest" in loaded else "Pending Dataset Training",
                "badge_class": "badge--real" if "random_forest" in loaded else "badge--secondary",
                "description": "Ensemble of decision trees for non-linear feature interaction.",
            },
            {
                "key": "svm",
                "name": "Support Vector Machine (Linear SVM)",
                "is_available": "svm" in loaded,
                "badge": "Active" if "svm" in loaded else "Pending Dataset Training",
                "badge_class": "badge--real" if "svm" in loaded else "badge--secondary",
                "description": "Maximum-margin hyperplane classifier with Platt scaling calibration.",
            },
        ]

    def predict(self, text: str, model_key: str = "logistic_regression") -> dict:
        """Analyze news text and return classification and confidence metrics.

        Parameters:
            text: Raw input news text
            model_key: One of 'logistic_regression', 'random_forest', 'svm'
        """
        if not text or not str(text).strip():
            raise ValueError("Input text cannot be empty.")

        # 1. Preprocess input text using the single-source-of-truth pipeline
        processed = preprocess_text(text)
        if not processed:
            # Fallback for text that might be stripped entirely
            processed = clean_raw_text(text)

        # 2. Vectorize
        vector = self.vectorizer.transform([processed])

        # 3. Select requested model or fallback gracefully
        is_fallback = False
        selected_key = model_key.lower().strip() if model_key else "logistic_regression"

        if selected_key in self.models:
            model = self.models[selected_key]
            model_used = self.model_names[selected_key]
        else:
            # Fall back to logistic_regression
            fallback_key = "logistic_regression" if "logistic_regression" in self.models else next(iter(self.models))
            model = self.models[fallback_key]
            model_used = f"{self.model_names[fallback_key]} (fallback: {selected_key} pending training)"
            is_fallback = True

        # 4. Predict label
        prediction = model.predict(vector)[0]

        # 5. Calculate probabilities & confidence
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
        elif hasattr(model, "decision_function"):
            import scipy.special
            df_val = float(model.decision_function(vector)[0])
            prob_real = float(scipy.special.expit(df_val))
            prob_fake = 1.0 - prob_real
        else:
            prob_fake = 1.0 if int(prediction) == 0 else 0.0
            prob_real = 1.0 if int(prediction) == 1 else 0.0

        confidence = float(max(prob_fake, prob_real))
        label = "REAL" if int(prediction) == 1 else "FAKE"

        return {
            "prediction": int(prediction),
            "prediction_label": label,
            "confidence": round(confidence * 100, 2),
            "fake_probability": round(prob_fake * 100, 2),
            "real_probability": round(prob_real * 100, 2),
            "model_key": selected_key,
            "model_used": model_used,
            "is_fallback": is_fallback,
            "processed_preview": processed[:200],
            "metadata": self.metadata,
        }


def get_predictor() -> Predictor:
    """Retrieve or initialize the Predictor singleton."""
    global _PREDICTOR_SINGLETON
    if _PREDICTOR_SINGLETON is None:
        _PREDICTOR_SINGLETON = Predictor()
    return _PREDICTOR_SINGLETON
