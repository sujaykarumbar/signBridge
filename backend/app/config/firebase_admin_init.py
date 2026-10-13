import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import firebase_admin
from firebase_admin import auth as admin_auth
from firebase_admin import credentials

logger = logging.getLogger(__name__)

# Project root:
# backend/
# ├── firebase.json
# ├── firebase-admin.json
# └── app/
#     └── config/
#         └── firebase_admin_init.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIREBASE_CONFIG_PATH = PROJECT_ROOT / "firebase.json"
FIREBASE_ADMIN_PATH = PROJECT_ROOT / "firebase-admin.json"

if FIREBASE_CONFIG_PATH.exists() and FIREBASE_ADMIN_PATH.exists():
    try:
        with FIREBASE_CONFIG_PATH.open(encoding="utf-8") as f:
            _firebase_config = json.load(f)

        _database_url = _firebase_config.get("databaseURL", "")

        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(
                credentials.Certificate(str(FIREBASE_ADMIN_PATH)),
                {
                    "databaseURL": _database_url,
                },
            )
    except Exception as exc:
        logger.warning("Failed to initialize Firebase Admin SDK: %s. Using development mock.", exc)
else:
    logger.info("firebase.json or firebase-admin.json not found. Using local development mock auth.")
    class MockAdminAuth:
        ExpiredIdTokenError = type("ExpiredIdTokenError", (Exception,), {})
        RevokedIdTokenError = type("RevokedIdTokenError", (Exception,), {})
        InvalidIdTokenError = type("InvalidIdTokenError", (Exception,), {})

        @staticmethod
        def verify_id_token(token, check_revoked=False):
            return {"uid": "dev_user", "email": "dev@signbridge.local"}

        @staticmethod
        def create_user(**kwargs):
            return MagicMock(uid="dev_user")

        @staticmethod
        def revoke_refresh_tokens(uid):
            pass

        @staticmethod
        def update_user(uid, **kwargs):
            return MagicMock(uid=uid)

        @staticmethod
        def delete_user(uid):
            pass

        @staticmethod
        def delete_users(uids):
            return MagicMock(errors=[])

        @staticmethod
        def list_users():
            return MagicMock(iterate_all=lambda: [])

        @staticmethod
        def collection(*args, **kwargs):
            mock_col = MagicMock()
            mock_col.limit.return_value.get.return_value = []
            return mock_col

    admin_auth = MockAdminAuth()