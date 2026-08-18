"""
EXISTING CODE (core tables + CRUD) — adapted from backend/database.py + backend/job_id.py,
trimmed to exactly what the Recruiter Dashboard / AI job-creation workflow needs.

INTEGRATION NOTE: the full project also has `users` and `auth_sessions` tables (backend/auth.py,
backend/routes/auth.py) providing its own email/password session-cookie auth. Those are almost
certainly NOT what you want to reuse in a larger platform that already has its own auth system —
this file omits them. Everywhere below that references `company_id` or `owner_user_id` is
expecting integer IDs from YOUR platform's own auth/tenant system; adapt the type/lookup as
needed, but keep the column semantics (company_id scopes every query — see the multi-tenant note
on each list function).

Database: SQLite, accessed via the stdlib `sqlite3` module — no ORM. Two separate SQLite files
are used in the original system:
  1. This one (`recruitment.db`) — the tables below. Authoritative for anything queried in bulk
     (job lists, public listings) and for published-job content.
  2. A SEPARATE LangGraph checkpoint database (`checkpoints.sqlite`) — NOT covered in this file;
     see handoff/agent_state.py and Part 6 of the handoff notebook. That one is authoritative for
     an IN-PROGRESS conversation's full state (message history, phase, etc.); THIS file's `jobs`
     table is only a write-through mirror of job_state, kept current enough for list/detail
     queries without needing to touch LangGraph state.
"""

import json
import sqlite3
import string
from contextlib import contextmanager
from pathlib import Path

# INTEGRATION: point this at wherever your platform wants the SQLite file (or replace this whole
# file's connection layer if you're moving to Postgres/MySQL — the SQL here is plain ANSI-ish
# SQLite and will need adaptation, e.g. AUTOINCREMENT -> SERIAL, TEXT -> proper JSON columns).
APP_DB_PATH = "recruitment.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS company_profile (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name       TEXT NOT NULL UNIQUE,
    industry           TEXT,
    company_overview   TEXT,
    website            TEXT,
    headquarters       TEXT,
    company_culture    TEXT,
    benefits           TEXT,
    work_life_balance  TEXT,
    why_join_us        TEXT,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One row per chat session (= one job draft, eventually published or not). session_id is the
-- SAME value as the LangGraph thread_id — it's the join key between the two databases.
CREATE TABLE IF NOT EXISTS jobs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                   TEXT UNIQUE,               -- e.g. "AM0001" — NULL until published
    company_id               INTEGER NOT NULL REFERENCES company_profile(id),
    owner_user_id            INTEGER,                   -- INTEGRATION: FK into YOUR users table
    session_id               TEXT,
    job_title                TEXT,
    job_category             TEXT,
    experience                TEXT,
    location                 TEXT,
    work_mode                TEXT,
    employment_type          TEXT,
    required_skills          TEXT,                      -- JSON-encoded list[str]
    preferred_skills         TEXT,                      -- JSON-encoded list[str]
    education                TEXT,
    responsibilities         TEXT,                      -- JSON-encoded list[str]
    salary                   TEXT,
    deadline                 TEXT,
    additional_information   TEXT,
    company_overrides        TEXT,                      -- JSON-encoded dict[str, str]
    jd_version_1              TEXT,                      -- JSON-encoded JobDescriptionDraft
    jd_version_2              TEXT,                      -- unused in practice (see notes) but present in schema
    selected_jd               TEXT,                      -- JSON-encoded JobDescriptionDraft, set at publish time
    selected_version          TEXT,                      -- "1"
    jd_stale                  BOOLEAN DEFAULT 0,
    jd_generated_at           TIMESTAMP,
    status                    TEXT DEFAULT 'draft',       -- 'draft' | 'published'
    accepting_applications    INTEGER NOT NULL DEFAULT 1, -- independent of status; published-only toggle
    created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at              TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_session_id ON jobs(session_id);

-- Per-company Job ID sequence, e.g. company "Amazon" -> prefix "AM" -> AM0001, AM0002, ...
CREATE TABLE IF NOT EXISTS company_sequences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL UNIQUE REFERENCES company_profile(id),
    prefix        TEXT NOT NULL UNIQUE,
    last_number   INTEGER NOT NULL DEFAULT 0,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pure ownership/authorization record, created the moment a session_id is minted — BEFORE a
-- `jobs` row necessarily exists (upsert_job_draft only creates one once job_title is known).
-- INTEGRATION: user_id should reference YOUR platform's own user table.
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     INTEGER,
    company_id  INTEGER NOT NULL REFERENCES company_profile(id),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_company_id ON chat_sessions(company_id);
"""

JSON_LIST_FIELDS = ("required_skills", "preferred_skills", "responsibilities")
JSON_DICT_FIELDS = ("company_overrides",)
JSON_OPTIONAL_FIELDS = ("jd_version_1", "jd_version_2", "selected_jd")

COMPANY_PROFILE_EDITABLE_FIELDS = (
    "company_name", "industry", "company_overview", "website", "headquarters",
    "company_culture", "benefits", "work_life_balance", "why_join_us",
)


def init_db(db_path: str = APP_DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True) if Path(db_path).parent != Path(".") else None
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_connection(db_path: str = APP_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    for field in JSON_LIST_FIELDS:
        if field in data:
            data[field] = json.loads(data[field]) if data[field] else []
    for field in JSON_DICT_FIELDS:
        if field in data:
            data[field] = json.loads(data[field]) if data[field] else {}
    for field in JSON_OPTIONAL_FIELDS:
        if field in data:
            data[field] = json.loads(data[field]) if data[field] else None
    return data


# ---------------------------------------------------------------------------
# company_profile
# ---------------------------------------------------------------------------

def get_company_profile_by_id(company_id: int, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM company_profile WHERE id = ?", (company_id,)).fetchone()
        return _row_to_dict(row)


def get_company_profile_by_name(company_name: str, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM company_profile WHERE company_name = ?", (company_name,)).fetchone()
        return _row_to_dict(row)


def create_company_profile(company_name: str, fields: dict, db_path: str = APP_DB_PATH) -> dict:
    """INTEGRATION: in the original app this happens inside signup (create_company_and_recruiter,
    which also creates the first user row atomically). Split out here since your platform likely
    already has its own company/tenant creation flow — call this once you have a company_id-worthy
    record to attach jobs to.
    """
    payload = {k: v for k, v in fields.items() if k in COMPANY_PROFILE_EDITABLE_FIELDS}
    payload["company_name"] = company_name
    with get_connection(db_path) as conn:
        columns = list(payload.keys())
        placeholders = ", ".join("?" for _ in columns)
        cur = conn.execute(
            f"INSERT INTO company_profile ({', '.join(columns)}) VALUES ({placeholders})",
            list(payload.values()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM company_profile WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_dict(row)


def update_company_profile(company_id: int, updates: dict, db_path: str = APP_DB_PATH) -> dict | None:
    payload = {k: v for k, v in updates.items() if k in COMPANY_PROFILE_EDITABLE_FIELDS}
    if not payload:
        return get_company_profile_by_id(company_id, db_path)
    with get_connection(db_path) as conn:
        set_clause = ", ".join(f"{col} = ?" for col in payload.keys())
        conn.execute(
            f"UPDATE company_profile SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            [*payload.values(), company_id],
        )
        conn.commit()
        row = conn.execute("SELECT * FROM company_profile WHERE id = ?", (company_id,)).fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# chat_sessions — pure ownership record
# ---------------------------------------------------------------------------

def create_chat_session(session_id: str, user_id, company_id: int, db_path: str = APP_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, user_id, company_id) VALUES (?, ?, ?)",
            (session_id, user_id, company_id),
        )
        conn.commit()


def get_chat_session_owner(session_id: str, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# jobs — draft write-through, JD storage, publish
# ---------------------------------------------------------------------------

def get_job_by_session_id(session_id: str, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def upsert_job_draft(
    session_id: str,
    company_id: int,
    job_state: dict,
    jd_stale: bool = False,
    owner_user_id=None,
    db_path: str = APP_DB_PATH,
) -> dict:
    """Write-through the working JobState (see handoff/models.py) into the jobs table.

    Creates the row on first call (once job_title is known), updates it on every subsequent
    call. Never touches job_id/status/published_at here — those are publish-time concerns
    (see finalize_publish). owner_user_id is stamped only at creation (audit trail only —
    authorization should always check company_id, never this).
    """
    payload = {
        "job_title": job_state.get("job_title"),
        "job_category": job_state.get("job_category"),
        "experience": job_state.get("experience"),
        "location": job_state.get("location"),
        "work_mode": job_state.get("work_mode"),
        "employment_type": job_state.get("employment_type"),
        "required_skills": json.dumps(job_state.get("required_skills", [])),
        "preferred_skills": json.dumps(job_state.get("preferred_skills", [])),
        "education": job_state.get("education"),
        "responsibilities": json.dumps(job_state.get("responsibilities", [])),
        "salary": job_state.get("salary"),
        "deadline": job_state.get("deadline"),
        "additional_information": job_state.get("additional_information"),
        "company_overrides": json.dumps(job_state.get("company_overrides", {})),
        "jd_stale": 1 if jd_stale else 0,
    }
    with get_connection(db_path) as conn:
        existing = conn.execute("SELECT id FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        if existing is None:
            columns = ["session_id", "company_id", "owner_user_id", *payload.keys()]
            placeholders = ", ".join("?" for _ in columns)
            values = [session_id, company_id, owner_user_id, *payload.values()]
            conn.execute(f"INSERT INTO jobs ({', '.join(columns)}) VALUES ({placeholders})", values)
        else:
            set_clause = ", ".join(f"{col} = ?" for col in payload.keys())
            conn.execute(
                f"UPDATE jobs SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                [*payload.values(), session_id],
            )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def save_jd_versions(session_id: str, jd_versions: dict, db_path: str = APP_DB_PATH) -> dict | None:
    """Persist a freshly generated JD draft, resetting staleness/selection."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET jd_version_1 = ?, jd_version_2 = ?, jd_stale = 0,
                jd_generated_at = CURRENT_TIMESTAMP, selected_version = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
            """,
            (json.dumps(jd_versions.get("1")), json.dumps(jd_versions.get("2")), session_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def save_selected_version(session_id: str, version: str, db_path: str = APP_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET selected_version = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (version, session_id),
        )
        conn.commit()


def save_refined_jd(session_id: str, version: str, jd: dict, db_path: str = APP_DB_PATH) -> dict | None:
    column = "jd_version_1" if version == "1" else "jd_version_2"
    with get_connection(db_path) as conn:
        conn.execute(
            f"UPDATE jobs SET {column} = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (json.dumps(jd), session_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


# --- Job ID allocation (originally backend/job_id.py) ---

_JOB_ID_NUMBER_WIDTH = 4


def _candidate_prefixes(company_name: str) -> list[str]:
    letters = [c for c in company_name.upper() if c.isalpha()]
    words = [w for w in company_name.upper().split() if w and w[0].isalpha()]
    candidates = []
    if len(letters) >= 2:
        candidates.append(letters[0] + letters[1])
    if len(letters) >= 3:
        candidates.append(letters[0] + letters[2])
    if len(words) >= 2:
        candidates.append(words[0][0] + words[1][0])
    if len(letters) >= 4:
        candidates.append(letters[0] + letters[3])
    seen: set[str] = set()
    unique = []
    for cand in candidates:
        if len(cand) == 2 and cand not in seen:
            unique.append(cand)
            seen.add(cand)
    return unique


def _next_free_prefix(company_name: str, used_prefixes: set[str]) -> str:
    for cand in _candidate_prefixes(company_name):
        if cand not in used_prefixes:
            return cand
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            cand = a + b
            if cand not in used_prefixes:
                return cand
    raise RuntimeError("Exhausted all 2-letter company prefixes")  # practically unreachable


def _format_job_id(prefix: str, number: int) -> str:
    return f"{prefix}{number:0{_JOB_ID_NUMBER_WIDTH}d}"


def _get_or_create_sequence_prefix(conn: sqlite3.Connection, company_id: int, company_name: str) -> str:
    row = conn.execute("SELECT prefix FROM company_sequences WHERE company_id = ?", (company_id,)).fetchone()
    if row:
        return row["prefix"]
    used = {r["prefix"] for r in conn.execute("SELECT prefix FROM company_sequences").fetchall()}
    prefix = _next_free_prefix(company_name, used)
    conn.execute(
        "INSERT INTO company_sequences (company_id, prefix, last_number) VALUES (?, ?, 0)",
        (company_id, prefix),
    )
    return prefix


def finalize_publish(
    session_id: str,
    company_id: int,
    company_name: str,
    selected_jd: dict,
    selected_version: str,
    db_path: str = APP_DB_PATH,
) -> dict | None:
    """Atomically allocates the Job ID and marks the job published — single transaction, so a
    Job ID is never allocated without the job actually being published (or vice versa).
    """
    with get_connection(db_path) as conn:
        # Grabs the write lock before the very first read, not just before the UPDATE, so a
        # concurrent publish for the same company can't read last_number before this one commits.
        conn.execute("BEGIN IMMEDIATE")
        prefix = _get_or_create_sequence_prefix(conn, company_id, company_name)
        conn.execute(
            "UPDATE company_sequences SET last_number = last_number + 1, updated_at = CURRENT_TIMESTAMP "
            "WHERE company_id = ?",
            (company_id,),
        )
        number = conn.execute(
            "SELECT last_number FROM company_sequences WHERE company_id = ?", (company_id,)
        ).fetchone()["last_number"]
        job_id = _format_job_id(prefix, number)
        conn.execute(
            """
            UPDATE jobs
            SET job_id = ?, selected_jd = ?, selected_version = ?, status = 'published',
                published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
            """,
            (job_id, json.dumps(selected_jd), selected_version, session_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def finalize_edit(
    session_id: str, job_state: dict, selected_jd: dict, selected_version: str, db_path: str = APP_DB_PATH
) -> dict | None:
    """Applies a confirmed edit to an ALREADY-published job — content columns + selected_jd only.
    Never touches job_id/status/created_at/published_at, so the Job ID and publish history are
    preserved exactly.
    """
    payload = {
        "job_title": job_state.get("job_title"),
        "job_category": job_state.get("job_category"),
        "experience": job_state.get("experience"),
        "location": job_state.get("location"),
        "work_mode": job_state.get("work_mode"),
        "employment_type": job_state.get("employment_type"),
        "required_skills": json.dumps(job_state.get("required_skills", [])),
        "preferred_skills": json.dumps(job_state.get("preferred_skills", [])),
        "education": job_state.get("education"),
        "responsibilities": json.dumps(job_state.get("responsibilities", [])),
        "salary": job_state.get("salary"),
        "deadline": job_state.get("deadline"),
        "additional_information": job_state.get("additional_information"),
        "company_overrides": json.dumps(job_state.get("company_overrides", {})),
        "selected_jd": json.dumps(selected_jd),
        "selected_version": selected_version,
        "jd_stale": 0,
    }
    with get_connection(db_path) as conn:
        set_clause = ", ".join(f"{col} = ?" for col in payload.keys())
        conn.execute(
            f"UPDATE jobs SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            [*payload.values(), session_id],
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def set_accepting_applications(session_id: str, accepting: bool, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET accepting_applications = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (1 if accepting else 0, session_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def delete_job(session_id: str, db_path: str = APP_DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# List/detail queries — these are ALL plain SQL against the `jobs` mirror table, completely
# independent of any LangGraph/checkpoint state (see Part 11 of the handoff notebook).
# Every one of these is scoped by company_id — never remove that filter, it's the multi-tenant
# boundary.
# ---------------------------------------------------------------------------

_JOB_LIST_COLUMNS = """
    jobs.id, jobs.job_id, jobs.session_id, jobs.company_id, jobs.job_title, jobs.job_category,
    jobs.experience, jobs.location, jobs.work_mode, jobs.employment_type, jobs.required_skills,
    jobs.preferred_skills, jobs.status, jobs.accepting_applications, jobs.deadline, jobs.created_at,
    jobs.updated_at, jobs.published_at,
    company_profile.company_name AS company_name
"""


def list_jobs_for_company(company_id: int, db_path: str = APP_DB_PATH) -> list[dict]:
    """Recruiter Dashboard's "Recent Jobs" list — this recruiter's own company, draft + published."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT {_JOB_LIST_COLUMNS} FROM jobs
            JOIN company_profile ON company_profile.id = jobs.company_id
            WHERE jobs.company_id = ? AND jobs.job_title IS NOT NULL
            ORDER BY jobs.updated_at DESC
            """,
            (company_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def list_published_jobs(db_path: str = APP_DB_PATH) -> list[dict]:
    """Public jobs listing — published jobs only, across ALL companies."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT {_JOB_LIST_COLUMNS} FROM jobs
            JOIN company_profile ON company_profile.id = jobs.company_id
            WHERE jobs.status = 'published'
            ORDER BY jobs.published_at DESC
            """
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_published_job_by_job_id(job_id: str, db_path: str = APP_DB_PATH) -> dict | None:
    """Public job detail page — full row (including selected_jd) for one published job."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT jobs.*, company_profile.company_name AS company_name,
                   company_profile.industry AS company_industry
            FROM jobs
            JOIN company_profile ON company_profile.id = jobs.company_id
            WHERE jobs.job_id = ? AND jobs.status = 'published'
            """,
            (job_id,),
        ).fetchone()
        return _row_to_dict(row)
