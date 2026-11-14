"""One-off admin seed: ``python -m services.user_service.seed_admin``.

Reads ADMIN_EMAIL / ADMIN_PASSWORD from the environment, creates the
admin user if that email is not registered yet. Idempotent — safe to
re-run. Refuses to run without both values set, and never prints the
password. Real credentials go in the environment, never in the repo.
"""

from __future__ import annotations

import sys

from services.user_service.auth import hash_password
from services.user_service.config import settings
from services.user_service.db import SessionLocal, init_db
from services.user_service.models import User


def main() -> int:
    email = settings.admin_email.strip().lower()
    password = settings.admin_password
    if not email or not password:
        print("ADMIN_EMAIL and ADMIN_PASSWORD must both be set.", file=sys.stderr)
        return 2
    if len(password) < 8:
        print("ADMIN_PASSWORD must be at least 8 characters.", file=sys.stderr)
        return 2
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing is not None:
            if existing.role != "admin":
                existing.role = "admin"
                db.commit()
                print(f"Promoted existing user to admin: {email}")
            else:
                print(f"Admin already exists: {email}")
            return 0
        user = User(
            email=email,
            password_hash=hash_password(password),
            role="admin",
        )
        db.add(user)
        db.commit()
        print(f"Created admin user: {email}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
