from __future__ import annotations

import re
from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from database import get_db
from utils.security import hash_password, verify_and_check_rehash

admin_login_bp = Blueprint("admin_login", __name__)


# -------------------------
# Admin Registration
# -------------------------
@admin_login_bp.route("/admin/register", methods=["GET", "POST"])
def admin_register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        fullname = request.form.get("fullname", "").strip() or username
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return render_template("admin_register.html")

        if not re.match(r"^[a-zA-Z0-9_]{3,30}$", username):
            flash("Username must be 3-30 characters (letters, numbers, underscore only).", "danger")
            return render_template("admin_register.html")

        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
            flash("Invalid email address.", "danger")
            return render_template("admin_register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("admin_register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return render_template("admin_register.html")

        db = get_db()
        cur = db.cursor(dictionary=True)

        cur.execute("SELECT id FROM admins WHERE email=%s OR username=%s LIMIT 1", (email, username))
        existing = cur.fetchone()
        if existing:
            cur.close()
            flash("An admin with that email or username already exists.", "danger")
            return render_template("admin_register.html")

        password_hash = hash_password(password)

        cur.execute(
            """
            INSERT INTO admins (username, fullname, email, password_hash)
            VALUES (%s, %s, %s, %s)
            """,
            (username, fullname, email, password_hash),
        )

        db.commit()
        cur.close()

        flash("Admin registered successfully. Please login.", "success")
        return redirect(url_for("admin_login.admin_login"))

    return render_template("admin_register.html")


# -------------------------
# Admin Login
# -------------------------
@admin_login_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("is_admin"):
        return redirect(url_for("admin.dashboard"))

    if request.method == "POST":
        login_id = (request.form.get("email") or request.form.get("username") or "").strip().lower()
        password = request.form.get("password", "")

        if not login_id or not password:
            flash("Email/Username and password are required.", "danger")
            return render_template("admin_login.html")

        db = get_db()
        cur = db.cursor(dictionary=True)

        cur.execute(
            """
            SELECT id, username, fullname, email, password_hash
            FROM admins
            WHERE LOWER(email)=%s OR LOWER(username)=%s
            LIMIT 1
            """,
            (login_id, login_id),
        )

        admin = cur.fetchone()

        if admin is None:
            cur.close()
            flash("Invalid credentials.", "danger")
            return render_template("admin_login.html")

        is_valid, should_rehash = verify_and_check_rehash(admin["password_hash"], password)

        if not is_valid:
            cur.close()
            flash("Invalid credentials.", "danger")
            return render_template("admin_login.html")

        # Transparently upgrade legacy Werkzeug PBKDF2/scrypt hash to bcrypt
        if should_rehash:
            try:
                new_hash = hash_password(password)
                cur2 = db.cursor()
                cur2.execute("UPDATE admins SET password_hash=%s WHERE id=%s", (new_hash, admin["id"]))
                db.commit()
                cur2.close()
            except Exception:
                pass

        cur.close()

        session.clear()
        session["admin_id"] = admin["id"]
        session["admin_name"] = admin["username"] or admin["fullname"]
        session["is_admin"] = True
        session["role"] = "admin"

        flash("Welcome Admin!", "success")
        return redirect(url_for("admin.dashboard"))

    return render_template("admin_login.html")


# -------------------------
# Admin Logout
# -------------------------
@admin_login_bp.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("admin_login.admin_login"))


# -------------------------
# Dashboard Redirect
# -------------------------
@admin_login_bp.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("is_admin"):
        flash("Please login first.", "danger")
        return redirect(url_for("admin_login.admin_login"))

    return redirect(url_for("admin.dashboard"))
