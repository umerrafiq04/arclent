import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

APP_DB_PATH = str(BASE_DIR / os.getenv("APP_DB_PATH", "database/recruitment.db"))
CHECKPOINT_DB_PATH = str(BASE_DIR / os.getenv("CHECKPOINT_DB_PATH", "database/checkpoints.sqlite"))

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8000")

# Whether the session cookie requires HTTPS. Defaults to on only when FRONTEND_ORIGIN itself
# is https — local dev over plain http needs this off, or the browser will refuse to store it.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "").lower() in ("1", "true", "yes") or FRONTEND_ORIGIN.startswith("https://")
