"""
EXISTING CODE — copied from backend/schemas.py (only the ListOperation import path differs).

These are the exact request/response bodies that cross the HTTP boundary for the Recruiter
Dashboard AI workflow. This IS the frontend<->backend contract — a new frontend must send/parse
exactly these shapes for the existing backend logic to keep working unmodified.
"""

from pydantic import BaseModel, Field

from handoff.models import ListOperation  # INTEGRATION: adjust import path in the new platform


class ChatRequest(BaseModel):
    """Body of POST /api/chat and (as multipart form fields) POST /api/chat/upload."""
    session_id: str | None = None  # null on the very first turn of a new job; echoed back after
    message: str


class ChatMessage(BaseModel):
    """One entry in the `messages` array of a ChatResponse."""
    role: str  # "user" | "assistant"
    content: str
    jd_document: dict | None = None  # present when this message IS a JD version, not plain text
    jd_version: str | None = None  # "1" (this system only ever produces one version — see notes)
    is_selected: bool = False  # whether jd_version is the currently selected one


class ChatResponse(BaseModel):
    """THE response shape returned by every chat-related endpoint:
    POST /api/chat (as the final SSE "result" event), POST /api/chat/upload (same),
    GET /api/chat/{session_id}, PATCH /api/chat/{session_id}/job-state,
    POST /api/chat/{session_id}/generate, POST /api/chat/{session_id}/skip-field,
    POST /api/chat/{session_id}/publish.

    A new frontend can build its ENTIRE Recruiter Dashboard/Draft UI from this one schema —
    see Part 10 (Recruiter Dashboard Data Contract) for a field-by-field usage guide.
    """
    session_id: str
    phase: str  # "collecting" | "summary" | "jd_selection" | "publish_confirm" | "published" | "editing"
    assistant_message: str  # convenience copy of the latest assistant text (also last item of `messages`)
    messages: list[ChatMessage]
    job_state: dict  # current JobState, as a plain dict (see handoff/models.py JobState)
    completeness_pct: int  # 0-100, UX-only progress indicator, never a gate on anything
    missing_essential: list[str]  # fields still blocking hard-floor sufficiency, if any
    jd_versions: dict | None = None  # {"1": JobDescriptionDraft-shaped dict} once generated
    selected_version: str | None = None  # "1" once a JD exists (this system never produces "2")
    jd_stale: bool  # true once a job_state field changed after jd_versions was generated
    job_record: dict | None = None  # fresh read of the `jobs` SQL row (see handoff/database.py)
    asking_about_field: str | None = None  # optional field the latest AI question is about, if any
    suggested_options: list[str] = Field(default_factory=list)  # quick-reply chip labels, if any
    options_multi_select: bool = False  # whether several suggested_options can be picked at once


class JobStatePatch(BaseModel):
    """Body of PATCH /api/chat/{session_id}/job-state — a DIRECT, silent edit to the draft (no
    chat message, no LLM call, no bot reply). Used for form-field edits: title, experience,
    salary, skill add/remove, company-context overrides, or a hand-edit to the generated JD's
    own text fields.
    """
    field_updates: dict[str, str] = Field(default_factory=dict)  # scalar JobState fields, see SCALAR_JOB_FIELDS
    list_operations: list[ListOperation] = Field(default_factory=list)  # required/preferred_skills, responsibilities
    company_overrides: dict[str, str] = Field(default_factory=dict)  # see COMPANY_OVERRIDE_FIELDS
    jd_text_updates: dict[str, str] = Field(default_factory=dict)  # direct edits to the CURRENT JD draft's own text


class CompanyProfileUpdate(BaseModel):
    """Body of PUT /api/company-profile."""
    company_name: str | None = None
    industry: str | None = None
    company_overview: str | None = None
    website: str | None = None
    headquarters: str | None = None
    company_culture: str | None = None
    benefits: str | None = None
    work_life_balance: str | None = None
    why_join_us: str | None = None


class AcceptingApplicationsUpdate(BaseModel):
    """Body of PUT /api/jobs/{session_id}/accepting-applications."""
    accepting_applications: bool
