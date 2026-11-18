from __future__ import annotations

import logging
import os
import traceback
import uuid

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from config import Config
from database import get_db
from predict import get_predictor

from utils.decorators import login_required
from utils.files import allowed_file, secure_filename

logger = logging.getLogger(__name__)

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["GET", "POST"])
def predict_ui():
    """Predict news authenticity via Web UI.
    
    Supports both authenticated users (history saved to DB) and guest testers
    (unauthenticated demonstration). Supports model selection:
    - 'logistic_regression'
    - 'random_forest'
    - 'svm'
    """
    predictor = get_predictor()

    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        model_key = (request.form.get("model") or "logistic_regression").strip().lower()
        input_filename = None
        input_type = "TEXT"

        # File upload support
        uploaded = request.files.get("file")
        if uploaded and uploaded.filename:
            allowed = set(Config.ALLOWED_EXTENSIONS)

            if not allowed_file(uploaded.filename, tuple(allowed)):
                flash("Invalid file type. Upload .txt, .csv, or .json only.", "danger")
                return redirect(url_for("prediction.predict_ui"))

            fname = secure_filename(uploaded.filename)
            ext = os.path.splitext(fname)[1].lower()
            new_name = f"{uuid.uuid4().hex}{ext}"
            save_path = os.path.join(Config.UPLOAD_FOLDER, new_name)
            uploaded.save(save_path)
            input_filename = new_name
            input_type = ext.lstrip(".").upper()

            # Attempt to extract text from file
            try:
                with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                    file_text = f.read()

                if ext == ".csv":
                    lines = [l.strip() for l in file_text.splitlines() if l.strip()]
                    if lines:
                        if any("label" in lines[0].lower() for _ in [0]):
                            lines = lines[1:]
                        text = lines[0] if lines else text
                elif ext == ".json":
                    import json
                    obj = json.loads(file_text)
                    if isinstance(obj, dict) and "text" in obj:
                        text = str(obj["text"])
                    elif isinstance(obj, list) and obj:
                        text = str(obj[0].get("text") or obj[0].get("body") or obj[0])
                else:
                    text = file_text
            except Exception as fe:
                logger.warning("Failed to parse uploaded file: %s", fe)
                flash("Failed to read uploaded file.", "danger")
                return redirect(url_for("prediction.predict_ui"))

        if not text:
            flash("Please provide news text or upload a supported file.", "danger")
            return redirect(url_for("prediction.predict_ui"))

        try:
            result = predictor.predict(text, model_key=model_key)

            # Persist to database if user is logged in
            user_id = session.get("user_id")
            if user_id:
                try:
                    db = get_db()
                    cur = db.cursor()
                    cur.execute(
                        "INSERT INTO prediction_history (user_id, model_name, input_text, input_filename, prediction_label, confidence, input_type) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (
                            user_id,
                            result.get("model_key", model_key),
                            text,
                            input_filename,
                            result["prediction_label"],
                            result["confidence"],
                            input_type,
                        ),
                    )
                    db.commit()
                    cur.close()
                except Exception as dbe:
                    logger.warning("Could not save prediction to database (MySQL may be offline): %s", dbe)

            return render_template(
                "prediction_result.html",
                result=result,
                input_text_preview=text[:1000],
                input_filename=input_filename,
                input_type=input_type,
                is_guest=(user_id is None),
            )

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Prediction failed: %s\n%s", e, tb)
            flash("Prediction failed on the server. Please check input text.", "danger")
            return redirect(url_for("prediction.predict_ui"))

    # GET request: render prediction interface with available models
    available_models = predictor.get_all_models_status()
    return render_template("prediction.html", available_models=available_models)


@prediction_bp.route("/history")
@login_required
def history():
    page = max(int(request.args.get("page", "1") or 1), 1)
    per_page = 10
    offset = (page - 1) * per_page

    try:
        db = get_db()
        cur = db.cursor(dictionary=True)

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM prediction_history WHERE user_id=%s",
            (session["user_id"],),
        )
        total = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT id, model_name, prediction_label, confidence, created_at, input_filename, input_text FROM prediction_history "
            "WHERE user_id=%s ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (session["user_id"], per_page, offset),
        )
        rows = cur.fetchall() or []
        cur.close()

    except Exception as e:
        logger.warning("Database unavailable during history query: %s", e)
        flash("Database is temporarily unreachable. Historical records could not be loaded.", "warning")
        total = 0
        rows = []

    total_pages = max((total + per_page - 1) // per_page, 1)

    return render_template(
        "prediction_history.html",
        rows=rows,
        page=page,
        total_pages=total_pages,
        total=total,
    )


@prediction_bp.route("/history/delete/<int:pred_id>")
@login_required
def delete_history(pred_id: int):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            "DELETE FROM prediction_history WHERE id=%s AND user_id=%s",
            (pred_id, session["user_id"]),
        )
        db.commit()
        cur.close()
        flash("Prediction deleted.", "success")
    except Exception as e:
        logger.warning("Database unavailable during delete: %s", e)
        flash("Database is temporarily unreachable. Record could not be deleted.", "warning")

    return redirect(url_for("prediction.history"))

