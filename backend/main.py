import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.auth import hash_password
from backend.config import (
    BOOTSTRAP_ADMIN_EMAIL,
    BOOTSTRAP_ADMIN_NAME,
    BOOTSTRAP_ADMIN_PASSWORD,
    FRONTEND_ORIGIN,
)
from backend.database import create_admin_or_attached_user, get_user_by_email, init_db
from backend.routes import admin, auth, chat, company, jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Recruitment Platform API")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort safety net so a bug in one route never leaks a raw traceback to the client."""
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred. Please try again."})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bootstrap_admin_if_configured() -> None:
    """Creates a single admin account from env vars on first boot — for environments with no
    DB shell access (e.g. this app on Railway without a working SSH path). There's no public
    signup-as-admin endpoint by design, so this is the only other way one gets made. Idempotent:
    skips silently once an account with that email already exists, never overwrites a password.
    """
    if not (BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD):
        return
    email = BOOTSTRAP_ADMIN_EMAIL.strip().lower()
    if get_user_by_email(email):
        return
    create_admin_or_attached_user(
        email=email,
        password_hash=hash_password(BOOTSTRAP_ADMIN_PASSWORD),
        name=BOOTSTRAP_ADMIN_NAME,
        company_id=None,
        role="admin",
    )
    logger.info("Bootstrapped admin account for %s", email)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    _bootstrap_admin_if_configured()


app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(company.router)
app.include_router(jobs.router)
app.include_router(jobs.public_router)
app.include_router(admin.router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/auth")
def auth_page() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "auth.html"))


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/auth")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
