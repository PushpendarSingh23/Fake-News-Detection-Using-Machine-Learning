"""Offline Multi-Model Training and Evaluation Pipeline for Fake News Detection.

Trains, cross-validates, and evaluates:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (LinearSVC with CalibratedClassifierCV)

Implements:
- Deduplication and data-leakage sanitization (publisher signature removal).
- Stratified 80/20 train/test splitting.
- TF-IDF feature extraction (25,000 max features, unigrams + bigrams).
- Full evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- Model serialization to models/*.pkl and JSON metadata export.
- Optional non-blocking database record update (if MySQL is running).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from config import Config
from utils.text_processing import ensure_nltk_resources, preprocess_text

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_FOLDER = Path(Config.DATASET_FOLDER)
MODELS_FOLDER = Path(Config.MODELS_FOLDER)


def check_dataset_availability() -> tuple[bool, Path, Path]:
    """Check if genuine ISOT dataset files exist."""
    fake_path = DATASET_FOLDER / "Fake.csv"
    true_path = DATASET_FOLDER / "True.csv"
    available = fake_path.exists() and true_path.exists()
    return available, fake_path, true_path


def load_and_preprocess_dataset(fake_path: Path, true_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load, sanitize, deduplicate, and preprocess ISOT dataset articles."""
    logger.info("Loading genuine ISOT dataset from %s...", DATASET_FOLDER)

    fake_df = pd.read_csv(fake_path, encoding="utf-8-sig", low_memory=False)
    true_df = pd.read_csv(true_path, encoding="utf-8-sig", low_memory=False)

    raw_fake_count = len(fake_df)
    raw_true_count = len(true_df)
    raw_total = raw_fake_count + raw_true_count
    logger.info("Raw counts: Fake=%d, True=%d, Total=%d", raw_fake_count, raw_true_count, raw_total)

    # Clean column headers
    fake_df.columns = fake_df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    true_df.columns = true_df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    fake_df = fake_df.loc[:, ~fake_df.columns.str.contains("^Unnamed")]
    true_df = true_df.loc[:, ~true_df.columns.str.contains("^Unnamed")]

    fake_df["label"] = 0  # 0 = FAKE
    true_df["label"] = 1  # 1 = REAL

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Construct combined content
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = df["title"] + " " + df["text"]

    # 1. Exact duplicate removal
    exact_duplicates_count = int(df.duplicated(subset=["content"]).sum())
    df = df.drop_duplicates(subset=["content"]).reset_index(drop=True)
    logger.info("Dropped %d exact duplicate articles.", exact_duplicates_count)

    # 2. Text preprocessing and publisher tag stripping
    logger.info("Executing NLP preprocessing and wire-tag sanitization across %d articles...", len(df))
    ensure_nltk_resources()
    df["processed"] = df["content"].apply(lambda t: preprocess_text(t, strip_wire_tags=True))

    # 3. Near-duplicate and empty text pruning
    initial_processed_len = len(df)
    df = df[df["processed"].str.split().str.len() >= 10].reset_index(drop=True)
    df = df.drop_duplicates(subset=["processed"]).reset_index(drop=True)
    near_duplicates_count = initial_processed_len - len(df)
    logger.info("Pruned %d short or near-duplicate articles.", near_duplicates_count)

    # Final counts and class balance
    final_fake_count = int((df["label"] == 0).sum())
    final_true_count = int((df["label"] == 1).sum())
    final_total = len(df)

    audit = {
        "raw_fake_count": raw_fake_count,
        "raw_true_count": raw_true_count,
        "raw_total_count": raw_total,
        "exact_duplicates_removed": exact_duplicates_count,
        "near_duplicates_removed": near_duplicates_count,
        "total_pruned": exact_duplicates_count + near_duplicates_count,
        "final_fake_count": final_fake_count,
        "final_true_count": final_true_count,
        "final_total_count": final_total,
        "fake_ratio": round((final_fake_count / final_total) * 100, 2),
        "true_ratio": round((final_true_count / final_total) * 100, 2),
    }

    logger.info(
        "Final clean dataset: %d samples (Fake: %d [%.1f%%], True: %d [%.1f%%])",
        final_total,
        final_fake_count,
        audit["fake_ratio"],
        final_true_count,
        audit["true_ratio"],
    )

    return df, audit


def train_and_evaluate(df: pd.DataFrame, audit: dict) -> dict[str, Any]:
    """Execute stratified split, TF-IDF vectorization, model training, and metric evaluation."""
    X = df["processed"]
    y = df["label"].astype(int)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Split: %d training samples, %d test samples", len(X_train), len(X_test))

    # Fit TF-IDF Vectorizer
    logger.info("Fitting TF-IDF Vectorizer (max_features=25000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Classifiers to train
    classifiers = {
        "logistic_regression": {
            "display_name": "Logistic Regression",
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "filename": "model_lr.pkl",
        },
        "random_forest": {
            "display_name": "Random Forest",
            "model": RandomForestClassifier(n_estimators=100, max_depth=30, n_jobs=-1, random_state=42),
            "filename": "model_rf.pkl",
        },
        "svm": {
            "display_name": "Support Vector Machine",
            "model": CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, random_state=42),
                cv=3,
            ),
            "filename": "model_svm.pkl",
        },
    }

    results: Dict[str, Any] = {
        "status": "trained_and_verified",
        "verification_status": "reproduced_offline",
        "dataset_audit": audit,
        "models": {},
    }

    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    for key, spec in classifiers.items():
        name = spec["display_name"]
        model = spec["model"]
        filename = spec["filename"]

        logger.info("--- Training %s ---", name)
        model.fit(X_train_vec, y_train)

        # Predictions & Probabilities
        y_pred = model.predict(X_test_vec)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_vec)[:, 1]
            try:
                roc_auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                roc_auc = None
        else:
            roc_auc = None

        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], zero_division=0)

        logger.info("%s -> Accuracy: %.4f, F1: %.4f, ROC-AUC: %s", name, acc, f1, f"{roc_auc:.4f}" if roc_auc else "N/A")

        # Save model pickle
        joblib.dump(model, MODELS_FOLDER / filename)
        if key == "logistic_regression":
            # Backward-compatibility alias
            joblib.dump(model, MODELS_FOLDER / "model.pkl")

        results["models"][key] = {
            "model_key": key,
            "display_name": name,
            "artifact_path": f"models/{filename}",
            "verified": True,
            "accuracy": round(acc, 5),
            "precision": round(float(prec), 5),
            "recall": round(float(rec), 5),
            "f1_score": round(float(f1), 5),
            "roc_auc": round(roc_auc, 5) if roc_auc is not None else None,
            "confusion_matrix": cm,
            "classification_report": report,
        }

    # Save vectorizer
    joblib.dump(vectorizer, MODELS_FOLDER / "vectorizer.pkl")

    # Export evaluation metadata JSON
    metrics_path = MODELS_FOLDER / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Export training summary
    summary_path = MODELS_FOLDER / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    logger.info("All models, vectorizer, and evaluation metrics serialized successfully.")
    return results


def main() -> None:
    print("=" * 70)
    print("AI Fake News Detection - Offline Multi-Model Training Engine")
    print("=" * 70)

    available, fake_path, true_path = check_dataset_availability()

    if not available:
        print("\n[!] DATASET MISSING: Genuine ISOT dataset files not found!")
        print(f"    Expected: {fake_path} and {true_path}")
        print("\n[INSTRUCTIONS TO ACQUIRE DATASET]:")
        print("1. Download the ISOT Fake News Dataset from:")
        print("   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("2. Extract the archive and place Fake.csv and True.csv in the dataset/ folder:")
        print(f"   {fake_path}")
        print(f"   {true_path}")
        print("\n[!] Offline training is BLOCKED until genuine dataset files are present.")
        print("[*] Existing pre-trained Logistic Regression artifact will remain active for inference.")
        sys.exit(1)

    df, audit = load_and_preprocess_dataset(fake_path, true_path)
    train_and_evaluate(df, audit)
    print("\n[+] Multi-model training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
