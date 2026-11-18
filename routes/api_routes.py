"""Standardized REST API Blueprints for AI Fake News Detection System.

Endpoints:
- POST /api/news/analyze     : Predict news credibility (REAL / FAKE) with confidence and probabilities
- GET  /api/news/history     : Retrieve paginated prediction history for authenticated user
- POST /api/auth/register    : User registration endpoint
- POST /api/auth/login       : User authentication endpoint
- GET  /api/health           : Health check endpoint (API status & DB connectivity)

Backward-compatibility aliases:
- POST /api/predict          -> /api/news/analyze
- GET  /api/history          -> /api/news/history
- POST /api/register         -> /api/auth/register
- POST /api/login            -> /api/auth/login
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Dict

from flask import Blueprint, jsonify, request, session

from database import get_db
from predict import get_predictor, MODEL_DISPLAY_NAMES
from utils.security import (
    check_password,
    hash_password,
    rate_limit,
    verify_and_check_rehash,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 50000


def _get_json_data() -> Dict[str, Any]:
    """Safely extract JSON body from request."""
    if request.is_json:
        return request.get_json(silent=True) or {}
    return {}


# -------------------------------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------------------------------
@api_bp.route("/health", methods=["GET"])
def health():
    db_status = "offline"
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        db_status = "connected"
    except Exception:
        db_status = "offline"

    predictor_loaded = False
    try:
        p = get_predictor()
        predictor_loaded = bool(p.models)
    except Exception:
        predictor_loaded = False

    return jsonify({
        "status": "ok",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "database": db_status,
        "inference_engine": "ready" if predictor_loaded else "unavailable",
        "supported_models": list(MODEL_DISPLAY_NAMES.keys()),
    }), 200


# -------------------------------------------------------------------------
# News Analysis / Prediction Endpoints
# -------------------------------------------------------------------------
@api_bp.route("/news/analyze", methods=["POST"])
@api_bp.route("/predict", methods=["POST"])
@rate_limit(max_requests=60, window_seconds=60)
def analyze_news():
    data = _get_json_data()
    text = (data.get("text") or "").strip()
    model_key = (data.get("model") or "logistic_regression").strip().lower()

    # Input validation
    if not text:
        return jsonify({
            "success": False,
            "error": "News text is required.",
            "code": "MISSING_TEXT",
        }), 400

    if len(text) < MIN_TEXT_LENGTH:
        return jsonify({
            "success": False,
            "error": f"News text is too short (minimum {MIN_TEXT_LENGTH} characters).",
            "code": "TEXT_TOO_SHORT",
        }), 400

    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({
            "success": False,
            "error": f"News text exceeds maximum allowed length of {MAX_TEXT_LENGTH} characters.",
            "code": "TEXT_TOO_LONG",
        }), 400

    if model_key not in MODEL_DISPLAY_NAMES:
        return jsonify({
            "success": False,
            "error": f"Invalid model '{model_key}'. Supported models: {', '.join(MODEL_DISPLAY_NAMES.keys())}.",
            "code": "INVALID_MODEL",
        }), 400

    # Execute core ML inference
    try:
        predictor = get_predictor()
        result = predictor.predict(text, model_key=model_key)
    except Exception as e:
        logger.error("Inference execution failed: %s", e)
        return jsonify({
            "success": False,
            "error": "Prediction inference failed. Please check server logs.",
            "code": "INFERENCE_ERROR",
        }), 500

    # Attempt database storage (gracefully skipped if MySQL is offline)
    db_saved = False
    user_id = session.get("user_id")

    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            """
            INSERT INTO prediction_history
            (user_id, model_name, input_text, input_filename, prediction_label, confidence, input_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                result["model_used"],
                text,
                None,
                result["prediction_label"],
                result["confidence"],
                "API",
            ),
        )
        db.commit()
        cur.close()
        db_saved = True
    except Exception as ex:
        # Non-blocking: failure to save history does NOT break the prediction response
        logger.warning("Could not persist prediction to MySQL (DB offline or error): %s", ex)
        db_saved = False

    return jsonify({
        "success": True,
        "prediction_label": result["prediction_label"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "fake_probability": result["fake_probability"],
        "real_probability": result["real_probability"],
        "model_key": result["model_key"],
        "model_used": result["model_used"],
        "is_fallback": result["is_fallback"],
        "db_saved": db_saved,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }), 200


# -------------------------------------------------------------------------
# Prediction History Endpoint
# -------------------------------------------------------------------------
@api_bp.route("/news/history", methods=["GET"])
@api_bp.route("/history", methods=["GET"])
def api_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({
            "success": False,
            "error": "Authentication required. Please log in first.",
            "code": "UNAUTHORIZED",
        }), 401

    try:
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(max(int(request.args.get("per_page", 10)), 1), 100)
    except (TypeError, ValueError):
        page, per_page = 1, 10

    offset = (page - 1) * per_page

    try:
        db = get_db()
        cur = db.cursor(dictionary=True)

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM prediction_history WHERE user_id=%s",
            (user_id,),
        )
        total = (cur.fetchone() or {}).get("cnt", 0)

        cur.execute(
            """
            SELECT id, model_name, prediction_label, confidence, input_type,
                   created_at, LEFT(input_text, 120) AS preview
            FROM prediction_history
            WHERE user_id=%s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, per_page, offset),
        )
        rows = cur.fetchall() or []
        cur.close()

        total_pages = (total + per_page - 1) // per_page if total > 0 else 1

        return jsonify({
            "success": True,
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "rows": rows,
        }), 200

    except Exception as ex:
        logger.warning("Database query failed for history (DB offline): %s", ex)
        return jsonify({
            "success": False,
            "error": "Database service is currently unreachable.",
            "code": "DATABASE_OFFLINE",
            "rows": [],
            "total": 0,
        }), 503


# -------------------------------------------------------------------------
# User Registration Endpoint
# -------------------------------------------------------------------------
@api_bp.route("/auth/register", methods=["POST"])
@api_bp.route("/register", methods=["POST"])
@rate_limit(max_requests=10, window_seconds=60)
def api_register():
    data = _get_json_data()
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    confirm = data.get("confirm_password") or ""

    if not username or not email or not password or not confirm:
        return jsonify({
            "success": False,
            "error": "All fields (username, email, password, confirm_password) are required.",
            "code": "MISSING_FIELDS",
        }), 400

    if not re.match(r"^[a-zA-Z0-9_]{3,30}$", username):
        return jsonify({
            "success": False,
            "error": "Username must be 3-30 characters (letters, numbers, underscore only).",
            "code": "INVALID_USERNAME",
        }), 400

    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        return jsonify({
            "success": False,
            "error": "Invalid email format.",
            "code": "INVALID_EMAIL",
        }), 400

    if password != confirm:
        return jsonify({
            "success": False,
            "error": "Passwords do not match.",
            "code": "PASSWORD_MISMATCH",
        }), 400

    if len(password) < 8:
        return jsonify({
            "success": False,
            "error": "Password must be at least 8 characters.",
            "code": "PASSWORD_TOO_SHORT",
        }), 400

    try:
        db = get_db()
        cur = db.cursor(dictionary=True)

        cur.execute("SELECT id FROM users WHERE username=%s LIMIT 1", (username,))
        if cur.fetchone():
            cur.close()
            return jsonify({
                "success": False,
                "error": "Username already taken.",
                "code": "USERNAME_EXISTS",
            }), 409

        cur.execute("SELECT id FROM users WHERE email=%s LIMIT 1", (email,))
        if cur.fetchone():
            cur.close()
            return jsonify({
                "success": False,
                "error": "Email already registered.",
                "code": "EMAIL_EXISTS",
            }), 409

        password_hash = hash_password(password)

        cur.execute(
            """
            INSERT INTO users (username, email, password_hash, role, is_active)
            VALUES (%s, %s, %s, 'user', 1)
            """,
            (username, email, password_hash),
        )
        db.commit()
        cur.close()

        return jsonify({
            "success": True,
            "message": "User registered successfully. You may now log in.",
        }), 201

    except Exception as ex:
        logger.warning("Database registration failed: %s", ex)
        return jsonify({
            "success": False,
            "error": "Database service is currently unreachable.",
            "code": "DATABASE_OFFLINE",
        }), 503


# -------------------------------------------------------------------------
# User Authentication / Login Endpoint
# -------------------------------------------------------------------------
@api_bp.route("/auth/login", methods=["POST"])
@api_bp.route("/login", methods=["POST"])
@rate_limit(max_requests=15, window_seconds=60)
def api_login():
    data = _get_json_data()
    login_id = (data.get("username") or data.get("email") or "").strip()
    password = data.get("password") or ""

    if not login_id or not password:
        return jsonify({
            "success": False,
            "error": "Username/Email and password are required.",
            "code": "MISSING_CREDENTIALS",
        }), 400

    try:
        db = get_db()
        cur = db.cursor(dictionary=True)

        cur.execute(
            """
            SELECT id, username, email, password_hash, role, is_active
            FROM users
            WHERE username=%s OR LOWER(email)=%s
            LIMIT 1
            """,
            (login_id, login_id.lower()),
        )
        user = cur.fetchone()

        if not user or not user.get("is_active", 1):
            cur.close()
            return jsonify({
                "success": False,
                "error": "Invalid username or password.",
                "code": "INVALID_CREDENTIALS",
            }), 401

        is_valid, should_rehash = verify_and_check_rehash(user["password_hash"], password)

        if not is_valid:
            cur.close()
            return jsonify({
                "success": False,
                "error": "Invalid username or password.",
                "code": "INVALID_CREDENTIALS",
            }), 401

        # Transparently rehash to bcrypt if legacy hash was used
        if should_rehash:
            try:
                new_hash = hash_password(password)
                cur2 = db.cursor()
                cur2.execute("UPDATE users SET password_hash=%s WHERE id=%s", (new_hash, user["id"]))
                db.commit()
                cur2.close()
            except Exception:
                pass

        cur.close()

        # Set session for cookie-based authentication
        session["user_id"] = int(user["id"])
        session["username"] = user["username"]
        session["role"] = user["role"]

        return jsonify({
            "success": True,
            "message": "Login successful.",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
            },
        }), 200

    except Exception as ex:
        logger.warning("Database login query failed: %s", ex)
        return jsonify({
            "success": False,
            "error": "Database service is currently unreachable.",
            "code": "DATABASE_OFFLINE",
        }), 503
