"""Database Backup Utility for AI Fake News Detection System.

Executes a full SQL dump of existing database tables and records.
Preserves user accounts, admin accounts, datasets, and prediction history.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config


def backup_database():
    import mysql.connector

    cfg = Config
    backup_dir = Path(cfg.BASE_DIR) / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"backup_{cfg.DB_NAME}_{timestamp}.sql"

    print(f"[*] Attempting database backup for '{cfg.DB_NAME}' on {cfg.DB_HOST}:{cfg.DB_PORT}...")

    try:
        conn = mysql.connector.connect(
            host=cfg.DB_HOST,
            port=cfg.DB_PORT,
            user=cfg.DB_USER,
            password=cfg.DB_PASSWORD,
            database=cfg.DB_NAME,
        )
    except mysql.connector.Error as err:
        print(f"[!] Unable to connect to MySQL: {err}")
        print(f"[!] Please ensure MySQL server is running on {cfg.DB_HOST}:{cfg.DB_PORT}.")
        return False, str(err)

    cur = conn.cursor(dictionary=True)

    try:
        cur.execute("SHOW TABLES")
        tables = [list(row.values())[0] for row in cur.fetchall()]

        with open(backup_file, "w", encoding="utf-8") as f:
            f.write(f"-- MySQL Database Backup: {cfg.DB_NAME}\n")
            f.write(f"-- Timestamp: {timestamp}\n")
            f.write("-- --------------------------------------------------------\n\n")

            for table in tables:
                f.write(f"-- Table structure for table `{table}`\n")
                cur.execute(f"SHOW CREATE TABLE `{table}`")
                create_table_sql = cur.fetchone()["Create Table"]
                f.write(f"{create_table_sql};\n\n")

                cur.execute(f"SELECT * FROM `{table}`")
                rows = cur.fetchall()
                if rows:
                    f.write(f"-- Dumping data for table `{table}` ({len(rows)} records)\n")
                    for row in rows:
                        cols = ", ".join([f"`{k}`" for k in row.keys()])
                        vals = []
                        for val in row.values():
                            if val is None:
                                vals.append("NULL")
                            elif isinstance(val, (int, float)):
                                vals.append(str(val))
                            else:
                                escaped = str(val).replace("\\", "\\\\").replace("'", "\\'")
                                vals.append(f"'{escaped}'")
                        vals_str = ", ".join(vals)
                        f.write(f"INSERT INTO `{table}` ({cols}) VALUES ({vals_str});\n")
                    f.write("\n")

        print(f"[+] Backup successfully completed: {backup_file}")
        print(f"[+] Total tables backed up: {len(tables)} ({', '.join(tables)})")
        return True, str(backup_file)

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    success, result = backup_database()
    if not success:
        sys.exit(1)
