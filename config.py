import os
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _load_secret_key() -> str:
    """Load SECRET_KEY securely.

    - In production, SECRET_KEY must come from environment.
    - In development, a random fallback is generated if missing.
    """
    secret = os.environ.get("SECRET_KEY")
    if secret:
        return secret

    debug_env = os.environ.get("DEBUG", "true").lower() in ("1", "true", "yes")
    if debug_env:
        import secrets
        return secrets.token_urlsafe(32)

    raise RuntimeError(
        "SECRET_KEY environment variable is required in non-debug environments."
    )


class Config:
    # ---------------- Flask Core ----------------
    SECRET_KEY = _load_secret_key()
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

    # ---------------- Session & Cookie Hardening ----------------
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("1", "true", "yes")

    # ---------------- MySQL Database ----------------
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "ai_fake_news")
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))

    # ---------------- Uploads & Folders ----------------
    ALLOWED_EXTENSIONS = ("txt", "csv", "json")
    MAX_PDF_UPLOAD_MB = 10
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
    MODEL_FOLDER = os.path.join(BASE_DIR, "models")
    MODELS_FOLDER = MODEL_FOLDER
    IMAGES_FOLDER = os.path.join(BASE_DIR, "images")

    # ---------------- ML Artifact Paths ----------------
    MODEL_PATH = os.path.join(MODEL_FOLDER, "model.pkl")
    MODEL_LR_PATH = os.path.join(MODEL_FOLDER, "model_lr.pkl")
    MODEL_RF_PATH = os.path.join(MODEL_FOLDER, "model_rf.pkl")
    MODEL_SVM_PATH = os.path.join(MODEL_FOLDER, "model_svm.pkl")
    VECTORIZER_PATH = os.path.join(MODEL_FOLDER, "vectorizer.pkl")
    METRICS_PATH = os.path.join(MODEL_FOLDER, "evaluation_metrics.json")
