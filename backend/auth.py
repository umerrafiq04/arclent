"""Password hashing, opaque server-side sessions, and FastAPI auth dependencies.

Sessions are DB-backed opaque tokens (secrets.token_urlsafe), not JWT/signed cookies —
that gives genuine server-side logout/revocation (delete the row), which a stateless
signed token can't do, at negligible cost for this app's scale (one sqlite lookup per
request). The cookie itself carries only the token; nothing sensitive ever reaches the
frontend/JS.
"""

import time
from datetime import datetime, timedelta
from secrets import token_urlsafe

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response

from backend.config import COOKIE_SECURE
from backend.database import (
    create_auth_session,
    delete_auth_session,
    get_auth_session_user,
    get_user_by_email,
)

COOKIE_NAME = "arclent_session"  # deliberately not "session_id" — that name already means
# the LangGraph thread_id everywhere else in this codebase.
SESSION_TTL = timedelta(days=14)

_hasher = PasswordHasher()
# Fixed hash used to keep an "unknown email" login attempt taking roughly as long as a
# "wrong password for a real account" attempt — otherwise the response-time difference
# leaks which emails are registered.
_DUMMY_HASH = _hasher.hash("not-a-real-password-just-for-timing")

# Simple in-process rate limit on sign-in attempts, keyed by lower-cased email. Resets on
# restart and does not coordinate across multiple worker processes — acceptable for the
# current single-process deployment; documented as a limitation, not silently glossed over.
_RATE_LIMIT_WINDOW_SECONDS = 15 * 60
_RATE_LIMIT_MAX_ATTEMPTS = 5
_signin_attempts: dict[str, list[float]] = {}


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def check_signin_rate_limit(email: str) -> None:
    now = time.monotonic()
    attempts = [t for t in _signin_attempts.get(email, []) if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    if len(attempts) >= _RATE_LIMIT_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many sign-in attempts. Please try again later.")
    _signin_attempts[email] = attempts


def record_signin_attempt(email: str) -> None:
    _signin_attempts.setdefault(email, []).append(time.monotonic())


def clear_signin_attempts(email: str) -> None:
    _signin_attempts.pop(email, None)


def verify_login(email: str, password: str) -> dict | None:
    """Timing-safe: always runs a full Argon2 verify, even for an unknown email, so the
    response time doesn't reveal whether the address is registered.
    """
    user = get_user_by_email(email.strip().lower())
    if user is None:
        try:
            _hasher.verify(_DUMMY_HASH, password)
        except VerifyMismatchError:
            pass
        return None
    if not verify_password(user["password_hash"], password):
        return None
    return user


def create_session(user_id: int) -> str:
    token = token_urlsafe(32)
    expires_at = (datetime.utcnow() + SESSION_TTL).strftime("%Y-%m-%d %H:%M:%S")
    create_auth_session(user_id, token, expires_at)
    return token


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def get_current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not signed in.")
    user = get_auth_session_user(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Session expired. Please sign in again.")
    return user


def get_current_recruiter(user: dict = Depends(get_current_user)) -> dict:
    if user.get("company_id") is None:
        raise HTTPException(status_code=403, detail="This account has no company profile.")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user
