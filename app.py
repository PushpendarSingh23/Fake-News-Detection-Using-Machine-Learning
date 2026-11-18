from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, session

from config import Config
from database import init_db, close_db
from routes.auth_routes import auth_bp
from routes.dashboard_routes import dashboard_bp
from routes.prediction_routes import prediction_bp
from routes.admin_routes import admin_bp
from routes.admin_login_routes import admin_login_bp
from routes.api_routes import api_bp
from routes.evaluation_routes import evaluation_bp


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load configuration
    app.config.from_object(Config)

    # Ensure project runtime directories exist
    for d in [
        app.config["UPLOAD_FOLDER"],
        app.config["DATASET_FOLDER"],
        app.config["MODELS_FOLDER"],
        app.config["IMAGES_FOLDER"],
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Initialize database tables and migrations (fails gracefully if DB offline)
    init_db(app)

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(admin_login_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(evaluation_bp)

    # Guard against accidental admin session leakage
    @app.before_request
    def _admin_session_sanity_check():
        if session.get("is_admin") and not session.get("admin_id"):
            session.pop("is_admin", None)

    # Database teardown per request
    app.teardown_appcontext(close_db)

    # Secret key & Session security configuration
    app.secret_key = app.config["SECRET_KEY"]
    app.config["SESSION_COOKIE_HTTPONLY"] = app.config.get("SESSION_COOKIE_HTTPONLY", True)
    app.config["SESSION_COOKIE_SAMESITE"] = app.config.get("SESSION_COOKIE_SAMESITE", "Lax")
    app.config["SESSION_COOKIE_SECURE"] = app.config.get("SESSION_COOKIE_SECURE", False)

    return app


app = create_app()


if __name__ == "__main__":
    model_files = [
        Path(Config.MODELS_FOLDER) / "model.pkl",
        Path(Config.MODELS_FOLDER) / "vectorizer.pkl",
        Path(Config.MODELS_FOLDER) / "metadata.json",
    ]

    missing = [str(p) for p in model_files if not p.exists()]
    if missing:
        print("[WARN] ML artifacts missing:")
        for file in missing:
            print(file)

    port = int(os.environ.get("PORT", 5000))
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=port,
        debug=app.config.get("DEBUG", True),
    )
