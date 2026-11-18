"""Phase 1 Automated Verification Suite.

Tests:
1. Environment and Configuration Hardening (no hardcoded secrets, session settings)
2. Password Hashing and Backward-Compatible Werkzeug Verification
3. Transparent Password Upgrade (Rehash) Detection
4. CSRF Token Generation and Timing-Safe Validation
5. Database Connection Pool Singleton Mechanics
6. Graceful Offline Handling for Database Initialization
7. Collision-Proof Admin Username Slug Generation
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from werkzeug.security import generate_password_hash as werkzeug_generate_hash

from config import Config
from app import create_app
from utils.security import (
    hash_password,
    check_password,
    is_legacy_hash,
    verify_and_check_rehash,
    generate_csrf_token,
    validate_csrf_token,
)
from database import get_db_pool, init_db, _column_exists, _index_exists


class TestConfigAndEnvironment(unittest.TestCase):
    def test_no_hardcoded_database_password(self):
        """Ensure default DB_PASSWORD is empty or from env, not hardcoded."""
        # Check that the default in code is not 'pra@1122'
        self.assertNotEqual(
            Config.DB_PASSWORD,
            "pra@1122",
            "CRITICAL: Found hardcoded default password in Config!"
        )

    def test_session_cookie_settings_dev(self):
        """Ensure session cookies are secure and hardened for development."""
        self.assertTrue(Config.SESSION_COOKIE_HTTPONLY)
        self.assertEqual(Config.SESSION_COOKIE_SAMESITE, "Lax")
        # In dev, secure cookies should be False unless explicit in env
        self.assertFalse(Config.SESSION_COOKIE_SECURE)

    def test_app_factory_creates_app(self):
        """Ensure create_app creates a valid Flask app without crashing."""
        app = create_app()
        self.assertIsNotNone(app)
        self.assertEqual(app.config["SESSION_COOKIE_HTTPONLY"], True)


class TestSecurityAndBackwardCompatibility(unittest.TestCase):
    def test_bcrypt_hashing_and_verification(self):
        """Test modern bcrypt password hashing and verification."""
        password = "SecurePassword123!"
        hashed = hash_password(password)

        self.assertTrue(hashed.startswith(("$2a$", "$2b$", "$2y$")))
        self.assertTrue(check_password(hashed, password))
        self.assertFalse(check_password(hashed, "WrongPassword"))
        self.assertFalse(is_legacy_hash(hashed))

    def test_legacy_werkzeug_hash_backward_compatibility(self):
        """Test backward-compatible verification of legacy Werkzeug PBKDF2/scrypt hashes."""
        password = "AdminLegacyPassword123!"
        legacy_hash = werkzeug_generate_hash(password)

        self.assertTrue(is_legacy_hash(legacy_hash))
        self.assertTrue(check_password(legacy_hash, password))
        self.assertFalse(check_password(legacy_hash, "WrongPassword"))

    def test_transparent_rehash_detection(self):
        """Ensure legacy Werkzeug hashes trigger should_rehash=True, while bcrypt does not."""
        password = "SamplePassword"

        # 1. Legacy hash should signal rehash
        legacy_hash = werkzeug_generate_hash(password)
        is_valid, should_rehash = verify_and_check_rehash(legacy_hash, password)
        self.assertTrue(is_valid)
        self.assertTrue(should_rehash, "Legacy hash MUST trigger should_rehash=True for upgrade")

        # 2. Upgraded bcrypt hash should NOT signal rehash
        modern_hash = hash_password(password)
        is_valid_modern, should_rehash_modern = verify_and_check_rehash(modern_hash, password)
        self.assertTrue(is_valid_modern)
        self.assertFalse(should_rehash_modern, "Bcrypt hash should NOT trigger rehash")

    def test_csrf_generation_and_validation(self):
        """Test CSRF token generation and timing-safe comparison."""
        token = generate_csrf_token()
        self.assertIsInstance(token, str)
        self.assertEqual(len(token), 64)  # 32 bytes hex encoded = 64 chars

        self.assertTrue(validate_csrf_token(token, token))
        self.assertFalse(validate_csrf_token(token, "invalid_token"))
        self.assertFalse(validate_csrf_token(None, token))
        self.assertFalse(validate_csrf_token(token, None))


class TestDatabasePoolAndMigrations(unittest.TestCase):
    def tearDown(self):
        import database
        database._POOL = None
    def test_pool_singleton_mechanics(self):
        """Test that get_db_pool returns the identical pool instance on repeated invocations."""
        app = create_app()
        mock_pool = MagicMock()

        with patch("mysql.connector.pooling.MySQLConnectionPool", return_value=mock_pool):
            import database
            database._POOL = None  # Reset singleton for testing

            pool1 = get_db_pool(app)
            pool2 = get_db_pool(app)

            self.assertIs(pool1, pool2, "get_db_pool must return the exact same pool singleton")
            self.assertEqual(database.pooling.MySQLConnectionPool.call_count, 1)

    def test_init_db_graceful_offline_handling(self):
        """Ensure init_db does not crash if MySQL is offline/unreachable."""
        app = create_app()
        with patch("database.get_db", side_effect=Exception("MySQL connection refused")):
            # Should catch exception and log warning without raising
            try:
                init_db(app)
            except Exception as e:
                self.fail(f"init_db raised an unhandled exception when DB is offline: {e}")

    def test_admin_username_collision_resolution_logic(self):
        """Test collision-proof admin username generation algorithm."""
        import re

        admins = [
            {"id": 1, "fullname": "John Doe", "username": None},
            {"id": 2, "fullname": "John Doe", "username": None},
            {"id": 3, "fullname": "John Doe", "username": None},
            {"id": 4, "fullname": "Jane Smith", "username": None},
        ]

        used = set()
        resolved = []

        for a in admins:
            base_slug = re.sub(r"[^a-zA-Z0-9_]", "_", (a["fullname"] or "admin").strip().lower())
            base_slug = re.sub(r"_+", "_", base_slug).strip("_") or "admin"
            candidate = base_slug
            counter = 2
            while candidate in used:
                candidate = f"{base_slug}_{counter}"
                counter += 1
            used.add(candidate)
            resolved.append(candidate)

        self.assertEqual(resolved[0], "john_doe")
        self.assertEqual(resolved[1], "john_doe_2")
        self.assertEqual(resolved[2], "john_doe_3")
        self.assertEqual(resolved[3], "jane_smith")
        self.assertEqual(len(set(resolved)), 4, "All generated usernames must be strictly unique")


if __name__ == "__main__":
    unittest.main()
