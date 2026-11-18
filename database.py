from __future__ import annotations

import logging
import re
import threading
from typing import Any

import mysql.connector
from flask import current_app, g
from mysql.connector import pooling

logger = logging.getLogger(__name__)

_POOL: pooling.MySQLConnectionPool | None = None
_POOL_LOCK = threading.Lock()


def get_db_pool(app) -> pooling.MySQLConnectionPool:
    """Retrieve or initialize the thread-safe MySQL connection pool singleton."""
    global _POOL
    if _POOL is not None:
        return _POOL

    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL

        cfg = app.config
        pool_name = "fake_news_pool"
        pool_size = int(cfg.get("DB_POOL_SIZE", 10))

        try:
            # Attempt to create the singleton pool
            _POOL = pooling.MySQLConnectionPool(
                pool_name=pool_name,
                pool_size=pool_size,
                host=cfg["DB_HOST"],
                port=cfg["DB_PORT"],
                user=cfg["DB_USER"],
                password=cfg["DB_PASSWORD"],
                database=cfg["DB_NAME"],
                autocommit=True,
            )
        except pooling.PoolError:
            # If pool already exists in MySQL connector registry, retrieve it
            _POOL = pooling.MySQLConnectionPool(pool_name=pool_name)

        return _POOL


def get_db():
    """Get a per-request MySQL connection from the singleton pool."""
    if "db" not in g:
        app = current_app
        pool = get_db_pool(app)
        g.db = pool.get_connection()
    return g.db


def close_db(_exc=None):
    """Release the connection back to the pool at request teardown."""
    db = g.pop("db", None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass


def _column_exists(cur, table: str, column: str) -> bool:
    """Check idempotently if a column exists in a table."""
    cur.execute(f"SHOW COLUMNS FROM `{table}` LIKE %s", (column,))
    return bool(cur.fetchone())


def _index_exists(cur, table: str, index_name: str) -> bool:
    """Check idempotently if an index exists on a table."""
    cur.execute(f"SHOW INDEX FROM `{table}` WHERE Key_name = %s", (index_name,))
    return bool(cur.fetchone())


def init_db(app):
    """Run idempotent schema migrations.

    - Creates required tables if not present.
    - Idempotently adds missing columns and views.
    - Resolves admin username migration with collision handling.
    - Preserves all existing user, admin, and prediction records.
    - Fails gracefully if MySQL is temporarily unreachable.
    """
    with app.app_context():
        try:
            db = get_db()
        except Exception as e:
            app.logger.warning("[WARN] Database init skipped (MySQL unreachable): %s", e)
            return

        cur = db.cursor(dictionary=True)

        try:
            # 1. Users table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(80) NOT NULL UNIQUE,
                    email VARCHAR(120) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    role ENUM('admin','user') NOT NULL DEFAULT 'user',
                    is_active TINYINT(1) NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # 2. Admins table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(80) NULL,
                    fullname VARCHAR(120) NOT NULL,
                    email VARCHAR(120) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # Idempotently ensure `username` exists in `admins`
            if not _column_exists(cur, "admins", "username"):
                cur.execute("ALTER TABLE admins ADD COLUMN username VARCHAR(80) NULL AFTER id")

            # Safe admin username generation & collision resolution
            cur.execute("SELECT id, fullname, username FROM admins")
            existing_admins = cur.fetchall() or []
            used_usernames = {a["username"].lower() for a in existing_admins if a.get("username")}

            for admin in existing_admins:
                if not admin.get("username"):
                    base_slug = re.sub(r"[^a-zA-Z0-9_]", "_", (admin.get("fullname") or "admin").strip().lower())
                    base_slug = re.sub(r"_+", "_", base_slug).strip("_") or "admin"
                    candidate = base_slug
                    counter = 2
                    while candidate in used_usernames:
                        candidate = f"{base_slug}_{counter}"
                        counter += 1

                    cur.execute("UPDATE admins SET username=%s WHERE id=%s", (candidate, admin["id"]))
                    used_usernames.add(candidate)

            # Ensure unique index on admins.username
            if not _index_exists(cur, "admins", "uq_admins_username"):
                try:
                    cur.execute("CREATE UNIQUE INDEX uq_admins_username ON admins (username)")
                except Exception as ex:
                    logger.warning("[WARN] Index uq_admins_username skipped: %s", ex)

            # 3. Prediction history table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NULL,
                    model_name VARCHAR(80) NOT NULL DEFAULT 'logistic_regression',
                    input_text LONGTEXT NOT NULL,
                    input_filename VARCHAR(255) NULL,
                    prediction_label VARCHAR(50) NOT NULL,
                    confidence DECIMAL(10,6) NOT NULL DEFAULT 0.0,
                    input_type VARCHAR(10) NULL DEFAULT 'TEXT',
                    extracted_text LONGTEXT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_created (user_id, created_at),
                    CONSTRAINT fk_pred_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # Idempotently add missing columns in prediction_history
            if not _column_exists(cur, "prediction_history", "model_name"):
                cur.execute(
                    "ALTER TABLE prediction_history ADD COLUMN model_name VARCHAR(80) NOT NULL DEFAULT 'logistic_regression' AFTER user_id"
                )

            if not _column_exists(cur, "prediction_history", "input_type"):
                cur.execute("ALTER TABLE prediction_history ADD COLUMN input_type VARCHAR(10) NULL DEFAULT 'TEXT'")

            if not _column_exists(cur, "prediction_history", "extracted_text"):
                cur.execute("ALTER TABLE prediction_history ADD COLUMN extracted_text LONGTEXT NULL")

            if not _column_exists(cur, "prediction_history", "confidence"):
                cur.execute("ALTER TABLE prediction_history ADD COLUMN confidence DECIMAL(10,6) NOT NULL DEFAULT 0.0")

            # 4. Model info table (tracks ML algorithms and performance metrics)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_info (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    model_name VARCHAR(80) NOT NULL UNIQUE,
                    display_name VARCHAR(120) NOT NULL,
                    accuracy DECIMAL(8,5) NOT NULL,
                    precision_score DECIMAL(8,5) NOT NULL,
                    recall_score DECIMAL(8,5) NOT NULL,
                    f1_score DECIMAL(8,5) NOT NULL,
                    roc_auc DECIMAL(8,5) NULL,
                    is_active TINYINT(1) NOT NULL DEFAULT 1,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # 5. Datasets table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    text LONGTEXT NOT NULL,
                    label VARCHAR(50) NOT NULL,
                    created_by INT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_label_created (label, created_at),
                    CONSTRAINT fk_dataset_user FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # 6. Reports table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NULL,
                    report_type VARCHAR(80) NOT NULL,
                    payload JSON NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_report_type_created (report_type, created_at),
                    CONSTRAINT fk_reports_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # 7. Create/Update NewsAnalysis view for specification compliance
            try:
                cur.execute("CREATE OR REPLACE VIEW news_analysis AS SELECT * FROM prediction_history")
            except Exception as ex:
                logger.warning("[WARN] View news_analysis creation skipped: %s", ex)

        finally:
            cur.close()
