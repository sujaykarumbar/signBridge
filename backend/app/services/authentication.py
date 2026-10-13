"""
Firebase Authentication Service.

Responsibilities:
- User registration
- User login
- Logout / token revocation
- Password management
- Email verification
- User deletion
- Test-user cleanup

Firebase Admin SDK is initialized by:
    app.config.firebase_admin_init.admin_auth
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Final, Pattern

import requests
from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError

from app.config.firebase_admin_init import admin_auth


logger = logging.getLogger(__name__)


# Configuration

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FIREBASE_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "firebase.json"

if FIREBASE_CONFIG_PATH.exists():
    with FIREBASE_CONFIG_PATH.open(encoding="utf-8") as file:
        _firebase_config = json.load(file)
    FIREBASE_API_KEY: Final[str] = _firebase_config.get("apiKey", "dev_api_key")
    IDENTITY_URL: Final[str] = _firebase_config.get("identityURL", "https://identitytoolkit.googleapis.com/v1")
    del _firebase_config
else:
    FIREBASE_API_KEY: Final[str] = "dev_api_key"
    IDENTITY_URL: Final[str] = "https://identitytoolkit.googleapis.com/v1"


# Firebase bulk-delete API limit.
MAX_DELETE_BATCH_SIZE: Final[int] = 1000

# Test accounts created by the live benchmark suite.
TEST_USER_EMAIL_PATTERN: Final[Pattern[str]] = re.compile(
    r"^libe-test-[A-Za-z][0-9]@example\.com$",
)


# Internal helpers

def _identity_request(
    endpoint: str,
    payload: dict,
) -> dict | None:
    """
    Execute a Firebase Identity Toolkit REST request.

    Returns:
        Response JSON on success.
        None when the request fails or response is invalid.
    """
    try:
        response = requests.post(
            f"{IDENTITY_URL}/{endpoint}",
            params={"key": FIREBASE_API_KEY},
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        err_msg = None
        if hasattr(e, "response") and e.response is not None:
            try:
                err_data = e.response.json()
                err_msg = err_data.get("error", {}).get("message")
            except Exception:
                pass
        logger.warning(
            "Firebase Identity API request failed: endpoint=%s error=%s",
            endpoint,
            err_msg or e,
        )
        if err_msg:
            return {"_error": err_msg}
        return None

    except ValueError:
        logger.exception(
            "Firebase Identity API returned invalid JSON: endpoint=%s",
            endpoint,
        )
        return None


# Authentication

def login_account(
    email: str,
    password: str,
) -> dict | None:
    """
    Authenticate a Firebase user and return its UID and ID token.
    """
    response = _identity_request(
        "accounts:signInWithPassword",
        {
            "email": email,
            "password": str(password),
            "returnSecureToken": True,
        },
    )

    if not response:
        if FIREBASE_API_KEY == "dev_api_key":
            logger.info("Dev mode login fallback for %s", email)
            return {
                "id": "dev_user",
                "token": "dev_token",
            }
        return None

    if "_error" in response:
        friendly = {
            "EMAIL_NOT_FOUND": "Account not found. Please register first.",
            "INVALID_PASSWORD": "Incorrect password.",
            "INVALID_LOGIN_CREDENTIALS": "Invalid email or password.",
            "USER_DISABLED": "This account has been disabled.",
        }.get(response["_error"], response["_error"].replace("_", " ").capitalize())
        return {"error": friendly}

    try:
        return {
            "id": response["localId"],
            "token": response["idToken"],
        }

    except (KeyError, TypeError):
        logger.exception("Invalid Firebase login response")
        return None


def register_account(
    email: str,
    password: str,
) -> dict | None:
    """
    Create a Firebase user and return an ID token by signing in afterwards.
    """
    # First try client-side Firebase Auth REST API (supports web API key directly)
    signup_resp = _identity_request(
        "accounts:signUp",
        {
            "email": email,
            "password": str(password),
            "returnSecureToken": True,
        },
    )
    if signup_resp and "idToken" in signup_resp:
        return {
            "id": signup_resp.get("localId", ""),
            "token": signup_resp["idToken"],
        }
    if signup_resp and "_error" in signup_resp:
        friendly = {
            "EMAIL_EXISTS": "This email is already registered. Please log in.",
            "OPERATION_NOT_ALLOWED": "Email/password signup is disabled in your Firebase console.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
            "WEAK_PASSWORD : Password should be at least 6 characters": "Password must be at least 6 characters.",
        }.get(signup_resp["_error"], signup_resp["_error"].replace("_", " ").capitalize())
        return {"error": friendly}

    try:
        admin_auth.create_user(
            email=email,
            password=str(password),
        )

        return login_account(email, password)

    except auth.EmailAlreadyExistsError:
        logger.warning("Registration attempted for existing email")
        return None

    except FirebaseError:
        logger.exception("Firebase error during registration")
        return None

    except Exception:
        if FIREBASE_API_KEY == "dev_api_key":
            logger.info("Dev mode registration fallback for %s", email)
            return {
                "id": "dev_user",
                "token": "dev_token",
            }
        logger.exception("Unexpected registration failure")
        return None


def logout_user(user_id: str) -> bool:
    """
    Revoke refresh tokens for a Firebase user.
    """
    try:
        admin_auth.revoke_refresh_tokens(user_id)
        return True

    except FirebaseError:
        logger.exception("Failed to revoke refresh tokens")
        return False

    except Exception:
        logger.exception("Unexpected logout failure")
        return False


# Password management

def forgot_password(email: str) -> bool:
    """
    Send a password-reset email.

    Always returns True to avoid revealing whether an account exists.
    """
    _identity_request(
        "accounts:sendOobCode",
        {
            "requestType": "PASSWORD_RESET",
            "email": email,
        },
    )

    return True


def update_password(
    user_id: str,
    password: str,
) -> bool:
    """
    Update a user's password and revoke existing refresh tokens.
    """
    try:
        admin_auth.update_user(
            user_id,
            password=str(password),
        )

    except auth.UserNotFoundError:
        logger.warning(
            "Password update attempted for nonexistent user"
        )
        return False

    except auth.InvalidArgumentError:
        logger.warning("Invalid password update request")
        return False

    except FirebaseError:
        logger.exception("Firebase error while updating password")
        return False

    except Exception:
        logger.exception("Unexpected password update failure")
        return False

    try:
        admin_auth.revoke_refresh_tokens(user_id)

    except Exception:
        # Password was already changed. Do not claim the operation
        # completely failed, but report the token-revocation problem.
        logger.exception(
            "Password changed but refresh-token revocation failed"
        )
        return False

    return True


# Email verification

def email_verify(id_token: str) -> bool:
    """
    Send a Firebase email-verification message.
    """
    response = _identity_request(
        "accounts:sendOobCode",
        {
            "requestType": "VERIFY_EMAIL",
            "idToken": id_token,
        },
    )

    return response is not None


# User deletion

def delete_user(user_id: str) -> bool | None:
    """
    Delete one Firebase user.

    Returns:
        True  -> user deleted
        False -> user did not exist
        None  -> Firebase/unknown error
    """
    try:
        admin_auth.delete_user(user_id)
        return True

    except auth.UserNotFoundError:
        logger.warning(
            "Delete attempted for nonexistent user"
        )
        return False

    except FirebaseError:
        logger.exception("Firebase error while deleting user")
        return None

    except Exception:
        logger.exception("Unexpected user deletion failure")
        return None


def delete_users(
    user_ids: list[str],
    batch_size: int = MAX_DELETE_BATCH_SIZE,
) -> dict:
    """
    Delete multiple Firebase users in batches.

    Firebase allows a maximum of 1,000 UIDs per bulk deletion request.

    Returns:
        {
            "requested_count": int,
            "success_count": int,
            "failure_count": int,
            "errors": [
                {
                    "uid": str,
                    "reason": str,
                }
            ],
            "deleted_uids": list[str],
        }
    """
    if not user_ids:
        return {
            "requested_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "errors": [],
            "deleted_uids": [],
        }

    if not 1 <= batch_size <= MAX_DELETE_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be between 1 and "
            f"{MAX_DELETE_BATCH_SIZE}"
        )

    # Deduplicate while preserving order.
    unique_uids = list(dict.fromkeys(user_ids))

    result = {
        "requested_count": len(unique_uids),
        "success_count": 0,
        "failure_count": 0,
        "errors": [],
        "deleted_uids": [],
    }

    for start in range(0, len(unique_uids), batch_size):
        batch = unique_uids[start:start + batch_size]

        try:
            deletion_result = admin_auth.delete_users(batch)

        except Exception:
            logger.exception(
                "Firebase bulk deletion failed: batch_start=%d",
                start,
            )

            result["failure_count"] += len(batch)
            result["errors"].extend(
                {
                    "uid": uid,
                    "reason": "batch deletion failed",
                }
                for uid in batch
            )
            continue

        failed_uids = {
            batch[error.index]: error.reason
            for error in deletion_result.errors
        }

        result["success_count"] += deletion_result.success_count
        result["failure_count"] += deletion_result.failure_count

        result["errors"].extend(
            {
                "uid": uid,
                "reason": reason,
            }
            for uid, reason in failed_uids.items()
        )

        result["deleted_uids"].extend(
            uid
            for uid in batch
            if uid not in failed_uids
        )

    return result


# Test-user cleanup

def find_test_users() -> list[tuple[str, str]]:
    """
    Find benchmark accounts matching the exact test-user pattern.

    Returns:
        [(uid, email), ...]
    """
    matches: list[tuple[str, str]] = []

    try:
        for user in admin_auth.list_users().iterate_all():
            email = user.email

            if (
                email
                and TEST_USER_EMAIL_PATTERN.fullmatch(email)
            ):
                matches.append((user.uid, email))

    except Exception:
        logger.exception(
            "Failed to list Firebase users for test cleanup"
        )
        raise

    return matches


def delete_test_users(
    dry_run: bool = False,
) -> dict:
    """
    Delete benchmark-created Firebase users.

    Only accounts matching:
        libe-test-[A-Za-z][0-9]@example.com

    are eligible for deletion.

    dry_run=True performs discovery only.
    """
    matches = find_test_users()

    result = {
        "dry_run": dry_run,
        "matched": len(matches),
        "emails": [email for _, email in matches],
    }

    if dry_run or not matches:
        result.update(
            {
                "requested_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "errors": [],
                "deleted_uids": [],
            }
        )
        return result

    deletion_result = delete_users(
        [uid for uid, _ in matches]
    )

    result.update(deletion_result)

    return result