from __future__ import annotations

import hmac
import secrets
import threading
import time
from functools import wraps

import bcrypt
from flask import jsonify, request
from werkzeug.security import check_password_hash as werkzeug_check_password_hash


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with 12 rounds."""
    if not password:
        raise ValueError("Password cannot be empty.")
    password_b = password.encode("utf-8") if isinstance(password, str) else password
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password_b, salt).decode("utf-8")


def is_legacy_hash(password_hash: str) -> bool:
    """Check if the stored hash is a legacy Werkzeug hash (pbkdf2 / scrypt)."""
    if not password_hash:
        return False
    return password_hash.startswith(("pbkdf2:", "scrypt:"))


def check_password(password_hash: str, password: str) -> bool:
    """Verify password against stored hash, supporting both bcrypt and legacy Werkzeug hashes."""
    if not password_hash or not password:
        return False

    # Check for bcrypt hash prefix ($2a$, $2b$, $2y$)
    if password_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception:
            return False

    # Backward-compatible check for legacy Werkzeug PBKDF2 / scrypt hashes
    if is_legacy_hash(password_hash):
        try:
            return werkzeug_check_password_hash(password_hash, password)
        except Exception:
            return False

    return False


def verify_and_check_rehash(password_hash: str, password: str) -> tuple[bool, bool]:
    """Verify password and report if it should be transparently upgraded to bcrypt.

    Returns:
        (is_valid, should_rehash)
    """
    if not check_password(password_hash, password):
        return False, False

    should_rehash = is_legacy_hash(password_hash)
    return True, should_rehash


def generate_csrf_token() -> str:
    """Generate a cryptographically secure 32-byte CSRF token."""
    return secrets.token_hex(32)


def validate_csrf_token(submitted_token: str | None, expected_token: str | None) -> bool:
    """Validate submitted CSRF token using constant-time comparison."""
    if not submitted_token or not expected_token:
        return False
    return hmac.compare_digest(str(submitted_token), str(expected_token))


# -------------------------------------------------------------------------
# In-Memory Sliding Window Rate Limiter
# -------------------------------------------------------------------------
_RATE_LIMIT_STORE: dict[str, list[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def rate_limit(max_requests: int = 30, window_seconds: int = 60):
    """In-memory sliding window rate limiter decorator per IP."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = request.headers.get("X-Forwarded-For", request.remote_addr or "127.0.0.1")
            ip = ip.split(",")[0].strip()
            now = time.time()
            cutoff = now - window_seconds

            with _RATE_LIMIT_LOCK:
                timestamps = _RATE_LIMIT_STORE.get(ip, [])
                timestamps = [t for t in timestamps if t > cutoff]

                if len(timestamps) >= max_requests:
                    retry_after = int(timestamps[0] + window_seconds - now) + 1
                    return jsonify({
                        "success": False,
                        "error": "Rate limit exceeded. Please wait before submitting more requests.",
                        "code": "RATE_LIMIT_EXCEEDED",
                        "retry_after": max(retry_after, 1),
                    }), 429

                timestamps.append(now)
                _RATE_LIMIT_STORE[ip] = timestamps

            return fn(*args, **kwargs)
        return wrapper
    return decorator


def reset_rate_limits():
    """Reset rate limiter store (useful for testing)."""
    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_STORE.clear()
