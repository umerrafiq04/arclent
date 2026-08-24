import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.config import APP_DB_PATH

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

CREATE TABLE IF NOT EXISTS jobs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                   TEXT UNIQUE,
    company_id               INTEGER NOT NULL REFERENCES company_profile(id),
    session_id               TEXT,
    job_title                TEXT,
    job_category             TEXT,
    experience               TEXT,
    location                 TEXT,
    work_mode                TEXT,
    employment_type          TEXT,
    required_skills          TEXT,
    preferred_skills         TEXT,
    education                TEXT,
    responsibilities         TEXT,
    custom_questions         TEXT,
    platforms                TEXT,
    salary                   TEXT,
    additional_information   TEXT,
    company_overrides        TEXT,
    jd_version_1             TEXT,
    jd_version_2             TEXT,
    selected_jd              TEXT,
    selected_version         TEXT,
    jd_stale                 BOOLEAN DEFAULT 0,
    jd_generated_at          TIMESTAMP,
    status                   TEXT DEFAULT 'draft',
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at             TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_session_id ON jobs(session_id);

CREATE TABLE IF NOT EXISTS company_sequences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id    INTEGER NOT NULL UNIQUE REFERENCES company_profile(id),
    prefix        TEXT NOT NULL UNIQUE,
    last_number   INTEGER NOT NULL DEFAULT 0,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

JSON_LIST_FIELDS = ("required_skills", "preferred_skills", "responsibilities", "custom_questions", "platforms")
JSON_DICT_FIELDS = ("company_overrides",)
JSON_OPTIONAL_FIELDS = ("jd_version_1", "jd_version_2", "selected_jd")


def init_db(db_path: str = APP_DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
    run_migrations(db_path)


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


# ---------------------------------------------------------------------------
# Migrations — the SCHEMA string above stays untouched (plain CREATE TABLE IF
# NOT EXISTS, safe to re-run forever). Everything added after the app already
# had live data (multi-tenant auth) goes through this versioned runner instead,
# so it's applied exactly once and never silently no-ops against an existing
# populated database.
# ---------------------------------------------------------------------------


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def _migration_001_users(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            email           TEXT NOT NULL UNIQUE,
            password_hash   TEXT NOT NULL,
            name            TEXT NOT NULL,
            company_id      INTEGER REFERENCES company_profile(id),
            role            TEXT NOT NULL DEFAULT 'recruiter' CHECK (role IN ('recruiter','admin')),
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_users_company_id ON users(company_id);
        """
    )


def _migration_002_auth_sessions(conn: sqlite3.Connection) -> None:
    # Named auth_sessions, not "sessions" — session_id already means the LangGraph
    # thread_id everywhere else in this codebase; reusing the word would be confusing.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token       TEXT PRIMARY KEY,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            expires_at  TIMESTAMP NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
        """
    )


def _migration_003_chat_sessions(conn: sqlite3.Connection) -> None:
    # Tracks who owns a chat/job-draft session from the moment session_id is minted —
    # upsert_job_draft only creates a jobs row once job_title is known, so without this,
    # an early-conversation turn (e.g. an advice question) would have no ownership record.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id  TEXT PRIMARY KEY,
            user_id     INTEGER REFERENCES users(id),
            company_id  INTEGER NOT NULL REFERENCES company_profile(id),
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_company_id ON chat_sessions(company_id);
        """
    )


def _migration_004_jobs_owner(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "jobs", "owner_user_id"):
        conn.execute("ALTER TABLE jobs ADD COLUMN owner_user_id INTEGER REFERENCES users(id)")


def _migration_005_backfill_chat_sessions(conn: sqlite3.Connection) -> None:
    # Existing jobs (created before chat_sessions existed) get a matching ownerless
    # (user_id=NULL) chat_sessions row so they're queryable the same way going forward.
    conn.execute(
        """
        INSERT INTO chat_sessions (session_id, user_id, company_id)
        SELECT session_id, NULL, company_id FROM jobs
        WHERE session_id IS NOT NULL
          AND session_id NOT IN (SELECT session_id FROM chat_sessions)
        """
    )


def _migration_006_accepting_applications(conn: sqlite3.Connection) -> None:
    # Whether a PUBLISHED job is still open to new applicants — independent of status
    # ('draft'/'published' stays purely about the publish workflow). Defaults to 1 so every
    # existing job (and every future publish) starts out open, matching current behavior.
    if not _column_exists(conn, "jobs", "accepting_applications"):
        conn.execute("ALTER TABLE jobs ADD COLUMN accepting_applications INTEGER NOT NULL DEFAULT 1")


def _migration_007_deadline(conn: sqlite3.Connection) -> None:
    if not _column_exists(conn, "jobs", "deadline"):
        conn.execute("ALTER TABLE jobs ADD COLUMN deadline TEXT")


def _migration_008_custom_questions(conn: sqlite3.Connection) -> None:
    # Recruiter-authored screening questions, manually added in the draft panel — never AI-touched
    # (see JobState.custom_questions in models.py). Defaults every existing row to an empty list,
    # same as required_skills/preferred_skills/responsibilities.
    if not _column_exists(conn, "jobs", "custom_questions"):
        conn.execute("ALTER TABLE jobs ADD COLUMN custom_questions TEXT")


def _migration_009_applications(conn: sqlite3.Connection) -> None:
    # A candidate's submission on a published job's Apply form — name/email plus their answers to
    # that job's own custom_questions (if any), keyed by question text since custom_questions has
    # no stable id of its own and can be edited by the recruiter after a job is published.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS applications (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id            TEXT NOT NULL REFERENCES jobs(job_id),
            applicant_name    TEXT NOT NULL,
            applicant_email   TEXT NOT NULL,
            answers           TEXT,
            submitted_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_applications_job_id ON applications(job_id);
        """
    )


def _migration_010_platforms(conn: sqlite3.Connection) -> None:
    # Which platform(s) this role is hiring for (Facebook, YouTube, Instagram, TikTok, Vimeo,
    # Twitch, Discord) — a multi-select checklist item, same list-field treatment as
    # required_skills/preferred_skills/responsibilities/custom_questions.
    if not _column_exists(conn, "jobs", "platforms"):
        conn.execute("ALTER TABLE jobs ADD COLUMN platforms TEXT")


MIGRATIONS = {
    1: _migration_001_users,
    2: _migration_002_auth_sessions,
    3: _migration_003_chat_sessions,
    4: _migration_004_jobs_owner,
    5: _migration_005_backfill_chat_sessions,
    6: _migration_006_accepting_applications,
    7: _migration_007_deadline,
    8: _migration_008_custom_questions,
    9: _migration_009_applications,
    10: _migration_010_platforms,
}


def run_migrations(db_path: str = APP_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)")
        row = conn.execute("SELECT version FROM schema_meta").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_meta (version) VALUES (0)")
            current = 0
        else:
            current = row["version"]
        for version in sorted(MIGRATIONS):
            if version > current:
                MIGRATIONS[version](conn)
                conn.execute("UPDATE schema_meta SET version = ?", (version,))
                conn.commit()


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


def get_company_profile_by_name(company_name: str, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM company_profile WHERE company_name = ?", (company_name,)
        ).fetchone()
        return _row_to_dict(row)


def get_company_profile_by_id(company_id: int, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM company_profile WHERE id = ?", (company_id,)).fetchone()
        return _row_to_dict(row)


COMPANY_PROFILE_EDITABLE_FIELDS = (
    "company_name", "industry", "company_overview", "website", "headquarters",
    "company_culture", "benefits", "work_life_balance", "why_join_us",
)


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


def get_job_by_session_id(session_id: str, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE session_id = ?", (session_id,)
        ).fetchone()
        return _row_to_dict(row)


def upsert_job_draft(
    session_id: str,
    company_id: int,
    job_state: dict,
    jd_stale: bool = False,
    owner_user_id: int | None = None,
    db_path: str = APP_DB_PATH,
) -> dict:
    """Write-through the working JobState into the jobs table as a draft row.

    Creates the row on first call (once job_title is known), updates it on every
    subsequent call. Never touches job_id/status/published_at here — those are
    publish-time concerns (job_id.py / publish_job node). owner_user_id is stamped
    only at creation (audit trail — authorization always checks company_id, never this).
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
        "custom_questions": json.dumps(job_state.get("custom_questions", [])),
        "platforms": json.dumps(job_state.get("platforms", [])),
        "salary": job_state.get("salary"),
        "deadline": job_state.get("deadline"),
        "additional_information": job_state.get("additional_information"),
        "company_overrides": json.dumps(job_state.get("company_overrides", {})),
        "jd_stale": 1 if jd_stale else 0,
    }
    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM jobs WHERE session_id = ?", (session_id,)
        ).fetchone()
        if existing is None:
            columns = ["session_id", "company_id", "owner_user_id", *payload.keys()]
            placeholders = ", ".join("?" for _ in columns)
            values = [session_id, company_id, owner_user_id, *payload.values()]
            conn.execute(
                f"INSERT INTO jobs ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
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
    """Persist freshly generated JD drafts, resetting staleness/selection for the new pair."""
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


def save_selected_version(session_id: str, version: str, db_path: str = APP_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET selected_version = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (version, session_id),
        )
        conn.commit()


def _get_or_create_sequence_prefix(conn: sqlite3.Connection, company_id: int, company_name: str) -> str:
    row = conn.execute("SELECT prefix FROM company_sequences WHERE company_id = ?", (company_id,)).fetchone()
    if row:
        return row["prefix"]
    from backend.job_id import next_free_prefix

    used = {r["prefix"] for r in conn.execute("SELECT prefix FROM company_sequences").fetchall()}
    prefix = next_free_prefix(company_name, used)
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
    """Atomically allocate the Job ID and mark the job published — single transaction,
    so a Job ID is never allocated without the job actually being published (or vice versa).
    """
    from backend.job_id import format_job_id

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
        job_id = format_job_id(prefix, number)

        # job_title here is the drafted document's own (deliberately more polished/specific)
        # headline, not the recruiter's plain chip-selected job_state.job_title — the public
        # listing and admin views read this column directly, so the enhanced title actually
        # shows up where a candidate would see it, not just inside the chat JD card. Falls back
        # to whatever was already stored (the plain title, from the earlier draft upsert) on the
        # rare chance the generated document is missing one.
        published_title = selected_jd.get("job_title")
        conn.execute(
            """
            UPDATE jobs
            SET job_id = ?, selected_jd = ?, selected_version = ?, status = 'published',
                published_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
                job_title = COALESCE(?, job_title)
            WHERE session_id = ?
            """,
            (job_id, json.dumps(selected_jd), selected_version, published_title, session_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def finalize_edit(session_id: str, job_state: dict, selected_jd: dict, selected_version: str, db_path: str = APP_DB_PATH) -> dict | None:
    """Apply a confirmed edit to an already-published job — content columns + selected_jd only.

    Never touches job_id/status/created_at/published_at, so the Job ID and publish history
    are preserved exactly. This is the only place edit-mode changes reach the live row; until
    called, edits live only in the LangGraph checkpoint (see apply_updates' write-through guard).
    """
    payload = {
        # Prefer the drafted document's own (deliberately more polished/specific) headline over
        # the recruiter's plain chip-selected job_state.job_title — same reasoning as
        # finalize_publish above, so an edited-and-republished listing keeps showing the enhanced
        # title, not a regression back to the plain one.
        "job_title": selected_jd.get("job_title") or job_state.get("job_title"),
        "job_category": job_state.get("job_category"),
        "experience": job_state.get("experience"),
        "location": job_state.get("location"),
        "work_mode": job_state.get("work_mode"),
        "employment_type": job_state.get("employment_type"),
        "required_skills": json.dumps(job_state.get("required_skills", [])),
        "preferred_skills": json.dumps(job_state.get("preferred_skills", [])),
        "education": job_state.get("education"),
        "responsibilities": json.dumps(job_state.get("responsibilities", [])),
        "custom_questions": json.dumps(job_state.get("custom_questions", [])),
        "platforms": json.dumps(job_state.get("platforms", [])),
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
    """Closes or reopens a published job to new applicants — never touches status/job_id, so
    the job stays exactly where it is in every listing, just visibly marked as closed.
    """
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET accepting_applications = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (1 if accepting else 0, session_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM jobs WHERE session_id = ?", (session_id,)).fetchone()
        return _row_to_dict(row)


def delete_job(session_id: str, db_path: str = APP_DB_PATH) -> bool:
    """Removes a job (draft or published) entirely. No soft-delete — nothing else in the
    schema references a job row by foreign key once it's gone. Returns whether a row existed.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM jobs WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0


def _application_row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    data = dict(row)
    data["answers"] = json.loads(data["answers"]) if data.get("answers") else {}
    return data


def create_application(job_id: str, answers: dict, db_path: str = APP_DB_PATH) -> dict:
    """A candidate's submission on a published job's Apply form — just their answers to that job's
    own custom_questions, no name/email collected (an explicit founder decision — the applicant_*
    columns stay NOT NULL and are simply stored empty rather than migrating them away, avoiding a
    destructive schema change for two columns that may be reintroduced later). answers is keyed by
    the exact question text (custom_questions has no stable id of its own) — {question: answer}.
    """
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO applications (job_id, applicant_name, applicant_email, answers) VALUES (?, ?, ?, ?)",
            (job_id, "", "", json.dumps(answers)),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return _application_row_to_dict(row)


def list_applications_for_job(job_id: str, db_path: str = APP_DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM applications WHERE job_id = ? ORDER BY submitted_at DESC", (job_id,)
        ).fetchall()
        return [_application_row_to_dict(r) for r in rows]


_JOB_LIST_COLUMNS = """
    jobs.id, jobs.job_id, jobs.session_id, jobs.company_id, jobs.job_title, jobs.job_category,
    jobs.experience, jobs.location, jobs.work_mode, jobs.employment_type, jobs.required_skills,
    jobs.preferred_skills, jobs.platforms, jobs.status, jobs.accepting_applications, jobs.deadline,
    jobs.created_at, jobs.updated_at, jobs.published_at,
    company_profile.company_name AS company_name
"""


def list_jobs_for_company(company_id: int, db_path: str = APP_DB_PATH) -> list[dict]:
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


def list_all_jobs_admin(db_path: str = APP_DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT {_JOB_LIST_COLUMNS} FROM jobs
            JOIN company_profile ON company_profile.id = jobs.company_id
            WHERE jobs.job_title IS NOT NULL
            ORDER BY jobs.updated_at DESC
            """
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_admin_job_by_id(internal_id: int, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT jobs.*, company_profile.company_name AS company_name,
                   company_profile.industry AS company_industry
            FROM jobs
            JOIN company_profile ON company_profile.id = jobs.company_id
            WHERE jobs.id = ?
            """,
            (internal_id,),
        ).fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Auth: users, auth_sessions, chat_sessions
# ---------------------------------------------------------------------------


def get_user_by_email(email: str, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        return _row_to_dict(row)


def get_user_by_id(user_id: int, db_path: str = APP_DB_PATH) -> dict | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_dict(row)


def create_admin_or_attached_user(
    email: str,
    password_hash: str,
    name: str,
    company_id: int | None,
    role: str = "recruiter",
    db_path: str = APP_DB_PATH,
) -> dict:
    """Used only by scripts/create_user.py — never reachable from a public API route.
    Raises sqlite3.IntegrityError on a duplicate email.
    """
    with get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, name, company_id, role) VALUES (?, ?, ?, ?, ?)",
            (email, password_hash, name, company_id, role),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_dict(row)


def create_company_and_recruiter(
    company_name: str,
    company_fields: dict,
    email: str,
    password_hash: str,
    name: str,
    db_path: str = APP_DB_PATH,
) -> dict:
    """Signup: create a brand-new company_profile + its first recruiter user, atomically.
    Never attaches to an existing company by name-match — a name collision must be a clean
    error, not a silent takeover of another tenant's data. Raises sqlite3.IntegrityError on a
    duplicate company_name or email; the caller maps that to a 409, never the generic 500.
    """
    payload = {k: v for k, v in company_fields.items() if k in COMPANY_PROFILE_EDITABLE_FIELDS}
    payload["company_name"] = company_name
    with get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        columns = list(payload.keys())
        placeholders = ", ".join("?" for _ in columns)
        cur = conn.execute(
            f"INSERT INTO company_profile ({', '.join(columns)}) VALUES ({placeholders})",
            list(payload.values()),
        )
        company_id = cur.lastrowid
        conn.execute(
            "INSERT INTO users (email, password_hash, name, company_id, role) VALUES (?, ?, ?, ?, 'recruiter')",
            (email, password_hash, name, company_id),
        )
        conn.commit()
        user_row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        company_row = conn.execute("SELECT * FROM company_profile WHERE id = ?", (company_id,)).fetchone()
        return {"user": _row_to_dict(user_row), "company": _row_to_dict(company_row)}


def create_auth_session(user_id: int, token: str, expires_at: str, db_path: str = APP_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO auth_sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at),
        )
        conn.commit()


def get_auth_session_user(token: str, db_path: str = APP_DB_PATH) -> dict | None:
    """Returns the session's user if the token is valid and unexpired; otherwise deletes the
    (now-useless) row so expired sessions don't silently accumulate, and returns None.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT users.* FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token = ? AND auth_sessions.expires_at > CURRENT_TIMESTAMP
            """,
            (token,),
        ).fetchone()
        if row is None:
            conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
            conn.commit()
            return None
        return _row_to_dict(row)


def delete_auth_session(token: str, db_path: str = APP_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))
        conn.commit()


def create_chat_session(session_id: str, user_id: int, company_id: int, db_path: str = APP_DB_PATH) -> None:
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
