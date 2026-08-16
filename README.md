# Arclent

An AI-powered recruitment module where a recruiter describes a hiring need in plain language and
the AI turns it into a structured job, a generated job description, and a published listing —
without a form in sight.

Authentication is **out of scope** — this module assumes an already-authenticated recruiter and a
configured company context, and is meant to be mounted behind another team's auth layer later.

## The core idea (Page 2)

> "I don't need to fill out a form. I can simply tell the AI what kind of person I want to hire."

The recruiter chats naturally. The AI extracts every field it can from a single message, reuses
the company's stored profile instead of asking for it again, asks only about the one or two things
that actually matter, and shows the structured job being built in real time on the right. Once
enough information exists, it summarizes, generates two distinct job description drafts, refines
them on request, and only publishes on explicit confirmation.

## Architecture

```
Browser (static HTML/CSS/vanilla JS)
        │  fetch() → JSON only, no direct DB access
        ▼
FastAPI  ──►  routes/company.py, jobs.py, admin.py  ──►  database.py  ──►  SQLite (recruitment.db)
        │
        └──►  routes/chat.py  ──►  agent/graph.py (LangGraph)
                                        │
                          ┌─────────────┴─────────────┐
                          │                            │
                Conversation state              Job content state
             (LangGraph SqliteSaver          (write-through upsert
              checkpoints.sqlite)             into `jobs` table)
                          │                            │
                          └───────────┬────────────────┘
                                      ▼
                     langchain-mistralai (ChatMistralAI)
                        .with_structured_output(...)
                                      ▼
                                 Mistral API
```

Two SQLite databases, kept deliberately separate:
- **`database/recruitment.db`** — durable application data (`company_profile`, `jobs`, `company_sequences`). This is what Pages 1/3/4 query directly.
- **`database/checkpoints.sqlite`** — LangGraph's own conversation/checkpoint store (message history, in-progress job state, control flags). A browser refresh recovers the conversation from here; it's never touched by Pages 1/3/4.

The LangGraph workflow (`backend/agent/`) is a single stateful graph, not a multi-agent system:

```
load_context → analyze_turn → apply_updates → {generate_jd | refine_jd | publish_job | publish_edit | END}
```

`analyze_turn` is one structured Mistral call per turn that fuses intent detection, field
extraction, and the natural-language reply — extraction/intent/response are never split into
separate calls. JD generation, refinement, and publishing are separate specialized steps that only
fire when actually requested, so a plain "I prefer 2" selection costs zero extra LLM calls.

### Editing an already-published job

Clicking **Edit Job** anywhere (Page 1, Page 4) reopens the job's *original* chat thread — the
same `session_id`/LangGraph checkpoint used to create it, whether that was a real conversation or
a job seeded straight into the database (in which case the checkpoint is hydrated from the DB row
on first touch). Editing a live job never mutates the public record immediately: field changes
accumulate only in the checkpoint (`phase: "editing"`) until the recruiter explicitly clicks
**Publish Edit**, at which point `publish_edit` writes the change through — updating content and
the selected JD, but never the Job ID, `status`, or `published_at`. A stale JD (job facts changed
since it was generated) blocks Publish Edit the same way it blocks first-time publishing.

## Project structure

```
backend/
  main.py, config.py, database.py, models.py, schemas.py, job_id.py
  routes/        chat.py, jobs.py, company.py, admin.py
  agent/         graph.py, state.py, nodes.py, prompts.py, llm.py, sufficiency.py
frontend/
  create-job.html   Page 2 — the AI recruiter workflow (primary deliverable)
  recruiter.html    Page 1 — dashboard + company profile editor
  jobs.html         Page 3 — public job listings
  admin.html        Page 4 — admin table, company drilldown, job detail
  css/, js/         js/api.js (fetch wrapper), dates.js, toast.js, nav.js are shared across pages
database/          recruitment.db, checkpoints.sqlite (created at runtime)
notebooks/
  database_setup.ipynb   creates the schema + seeds demo data — no application logic
```

## Setup

**Requirements:** Python 3.11+, a Mistral API key.

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run the seed notebook

# 2. Configure environment
cp .env.example .env
# edit .env and set MISTRAL_API_KEY

# 3. Initialize + seed the database
jupyter nbconvert --to notebook --execute --inplace notebooks/database_setup.ipynb
# (or open the notebook in Jupyter and run all cells)

# 4. Run the API (also serves the frontend as static files)
uvicorn backend.main:app --reload --port 8000

# 5. Open the app
# http://localhost:8000/recruiter.html
```

The seed notebook is idempotent — re-running it skips companies/jobs that already exist, so it's
safe to run again after using the app manually.

### Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `MISTRAL_API_KEY` | Your Mistral API key. Read only by the backend — never sent to the browser. |
| `MISTRAL_MODEL` | Model used for structured-output calls (default `mistral-large-latest`). |
| `APP_DB_PATH` | Path to the application SQLite database. |
| `CHECKPOINT_DB_PATH` | Path to the LangGraph checkpoint SQLite database. |
| `DEMO_COMPANY_NAME` | Which seeded company Page 1/2 act as (single-tenant demo scope; must match a `company_name` in `company_profile`). |
| `FRONTEND_ORIGIN` | CORS origin for local dev. |

## Demo workflow (~90 seconds)

1. Open `recruiter.html` — see the dashboard, company profile, and existing demo jobs.
2. Click **+ Post a New Job**.
3. Say:
   > "We need a Data Analyst with 0-1 years, hybrid in Bangalore, SQL mandatory. These are the only details."
4. Watch the right-hand panel fill in live — no follow-up questions needed, since everything essential was already provided.
5. Say "Generate the JD." — two distinct drafts appear.
6. Say "I prefer 2, make it more professional." — version 2 refines; version 1 is untouched.
7. Say "Yes, publish it." — a unique Job ID (e.g. `ZA0004`) is assigned and the job is published.
8. Click **Public Jobs** — the new listing is there immediately, with an **Apply Now** placeholder.
9. Click **Admin** — the same job appears with full structured detail; click the company name to see every job that company has posted.
10. Back on **Dashboard**, click **Edit Job** on the job you just published, say "Change the location to Delhi," regenerate the JD, then **Publish Edit** — the Job ID stays the same everywhere it's shown.

No manual database editing required at any point.

## Notes on scope and design decisions

- **Job IDs** are assigned only at publish time (not at draft creation), via a per-company sequence table, so abandoned drafts never burn a visible ID gap. Format is always `<2-letter prefix><4-digit number>` (e.g. `ZA0198`); prefix collisions are resolved by trying alternate letter pairs derived from the company name, never by changing the ID's shape.
- **Job-specific overrides** of company information (e.g. "for this job, emphasize our engineering culture instead") live in `job_state.company_overrides`, keyed to the company-profile field they override, and are never written back to `company_profile`.
- **A stale JD** (job facts changed after generation) is flagged (`jd_stale`) and blocks publishing (or Publish Edit) until regenerated/refined — it's never silently republished with contradictory information.
- **JD role enrichment**: the JD-generation prompt adds role-standard requirements beyond the recruiter's literal words (e.g. "SQL mandatory" for a Data Analyst still yields a full minimum-requirements section), but keeps recruiter-stated musts in `required_skills`/`minimum_requirements` and its own suggestions clearly hedged in `preferred_qualifications` — and never invents company-specific facts (locations, headcounts, clients, etc.) beyond the stored company profile.
- **Multi-job resume**: every job has its own `session_id`/checkpoint/DB row from the moment it's created, so several drafts can be in progress at once with zero cross-talk. Page 1 links to each by `?session_id=`; `localStorage` only remembers the single most-recently-touched one as a refresh convenience.
- Pages 1, 3, and 4 are intentionally lightweight, per the brief — they read from the same `jobs`/`company_profile` tables Page 2 writes to, with no independent data model of their own.
- Not implemented (by design, out of scope for this module): authentication, applicant tracking, candidates/applications, interviews.
