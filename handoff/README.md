# Recruiter Dashboard AI Workflow — Developer Handoff

## One-page plain-English explanation

> "If I throw away the existing frontend and build my own Recruiter Dashboard, exactly what API do I call, what do I send, what does the AI receive, what does it return, what gets stored, and what do I need to display?"

**You call `POST /api/chat`** with a JSON body `{"session_id": null, "message": "Data Analyst"}` (`session_id` is `null` only on the very first message of a new job — the backend mints one and hands it back to you; send that same value on every later call for this job).

**The response is not a single JSON object — it's a Server-Sent Events (SSE) stream** over that one POST request. You'll receive zero or more `{"type": "status", "label": "..."}` frames (live progress text, safe to ignore or show as a spinner label), followed by exactly one `{"type": "result", ...}` frame, then the stream closes. Strip the `"type"` key off that final frame and you have the full `ChatResponse` object — see `handoff/schemas.py::ChatResponse`.

**Behind that endpoint**: a LangGraph pipeline runs one Mistral LLM call (`ChatMistralAI`, structured output forced into the `TurnAnalysis` Pydantic schema — see `handoff/models.py`) that in a single shot classifies intent, extracts job facts, and writes the next chat reply plus quick-reply chip suggestions. The extracted facts get merged into an in-memory `job_state` dict, which is: (a) checkpointed by LangGraph into its own SQLite file keyed by `session_id`, and (b) mirrored into a `jobs` SQL table row (via `upsert_job_draft`) for cheap list/detail queries.

**The AI never writes the actual job description during chat.** That only happens via a **separate, direct** `POST /api/chat/{session_id}/generate` call — no LLM call happens on a normal chat turn beyond the conversational one. Likewise, publishing only happens via a direct `POST /api/chat/{session_id}/publish` call. Both are ordinary request/response JSON (not SSE), and both return the same `ChatResponse` shape so your UI updates the same way either way.

**What you display, from one `ChatResponse` object**: `messages` (the chat transcript), `job_state` (every field collected so far — title, skills, experience, location, etc.), `jd_versions`/`selected_version` (the generated job description once one exists), `phase` + `completeness_pct` (status indicators), and `suggested_options`/`asking_about_field` (quick-reply chips + whether a "Skip this" affordance applies). Full field-by-field breakdown: **Part 10** of the notebook.

**What gets stored, where**: two separate SQLite databases. One (`checkpoints.sqlite`, via LangGraph's `SqliteSaver`) holds the authoritative in-progress conversation state. The other (`recruitment.db`, plain `sqlite3`) holds a denormalized `jobs` table row per session — kept current via write-through on every turn, used for dashboard list/detail queries so they never need to touch LangGraph state. See **Part 6**.

## What's in this folder

| File | What it is |
|---|---|
| `models.py` | **EXISTING** — the AI's exact input/output contract (`TurnAnalysis`, `JobDescriptionDraft`, `JobState`, `Intent`). Copy-paste-ready, zero project-specific dependencies. |
| `schemas.py` | **EXISTING** — HTTP request/response bodies (`ChatRequest`, `ChatResponse`, etc.). This IS the frontend↔backend contract. |
| `agent_state.py` | **EXISTING** — the LangGraph `GraphState` TypedDict (what gets checkpointed per session). |
| `database.py` | **EXISTING** (core tables trimmed to this component) + **INTEGRATION NOTEs** inline — schema + CRUD for `company_profile`/`jobs`/`company_sequences`/`chat_sessions`. Verified to run standalone (see the notebook, Part 7). |
| `frontend_example.html` | **INTEGRATION EXAMPLE** — a self-contained, framework-free HTML/JS page demonstrating the full contract (SSE chat turn + the three direct action endpoints). Not part of, and not dependent on, the existing frontend. |

The full walkthrough — architecture, every model input/output field explained, the complete request-to-database-to-response lifecycle for a concrete example, and a developer integration checklist — is in `../notebooks/recruiter_dashboard_handoff.ipynb`.

## What is NOT in this folder (by design)

- The `users`/`auth_sessions` tables and cookie-session auth logic (`backend/auth.py`, `backend/routes/auth.py`) — a larger platform almost certainly has its own auth system; every function here takes plain `company_id`/`user_id` integers and doesn't care where they came from.
- The FastAPI route wiring itself (`backend/routes/chat.py`) — reproduced and explained in the notebook (Part 4) rather than duplicated here, since it depends on the LangGraph graph object and FastAPI's dependency injection, both of which need to be wired into your platform's own app, not copy-pasted verbatim.
- The system prompt text and LangGraph node functions (`backend/agent/prompts.py`, `backend/agent/nodes.py`) — these implement the actual conversational behavior and are the "preserve as-is" core of PART 14 of the spec; they're documented and quoted in full in the notebook (Parts 2, 5, 11) rather than re-hosted here, since they must stay wired to the real `backend/` package to keep working.
