import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

# The active LLM provider for backend/agent/llm.py — "deepseek" or "mistral". DeepSeek is the
# default now (own dedicated account/quota, not shared with anything else hitting the Mistral
# key) — Mistral stays fully wired so this is a one-variable rollback, not a code change, if
# ever needed.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

APP_DB_PATH = str(BASE_DIR / os.getenv("APP_DB_PATH", "database/recruitment.db"))
CHECKPOINT_DB_PATH = str(BASE_DIR / os.getenv("CHECKPOINT_DB_PATH", "database/checkpoints.sqlite"))

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8000")

# Whether the session cookie requires HTTPS. Defaults to on only when FRONTEND_ORIGIN itself
# is https — local dev over plain http needs this off, or the browser will refuse to store it.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "").lower() in ("1", "true", "yes") or FRONTEND_ORIGIN.startswith("https://")

# Optional one-time admin bootstrap for environments with no DB shell access (e.g. Railway
# when SSH isn't available) — see main.py's startup hook. Unset in normal operation; there is
# no public signup-as-admin endpoint, this is the only other way an admin account gets made.
BOOTSTRAP_ADMIN_EMAIL = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "")
BOOTSTRAP_ADMIN_PASSWORD = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")
BOOTSTRAP_ADMIN_NAME = os.getenv("BOOTSTRAP_ADMIN_NAME", "Admin")

# Same "no DB shell access" problem, for the one-off demo data seed (see
# scripts/seed_demo_jobs.py and main.py's startup hook). Unset in normal operation.
BOOTSTRAP_DEMO_JOBS = os.getenv("BOOTSTRAP_DEMO_JOBS", "").lower() in ("1", "true", "yes")
