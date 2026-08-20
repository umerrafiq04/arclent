from pydantic import BaseModel, Field

from backend.models import ListOperation


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    jd_document: dict | None = None  # present when this message IS a JD version, not plain text
    jd_version: str | None = None  # "1" | "2", set alongside jd_document
    is_selected: bool = False  # whether jd_version is the currently selected one (for the button)


class ChatResponse(BaseModel):
    session_id: str
    phase: str
    assistant_message: str
    messages: list[ChatMessage]
    job_state: dict
    completeness_pct: int
    missing_essential: list[str]
    jd_versions: dict | None = None
    selected_version: str | None = None
    jd_stale: bool
    job_record: dict | None = None
    asking_about_field: str | None = None  # optional field the latest AI question is about, if any
    suggested_options: list[str] = Field(default_factory=list)  # chip labels for the latest AI question, if any
    options_multi_select: bool = False  # whether the recruiter can tap several suggested_options before sending
    # Deterministic sufficiency signal (hard floor + the standard checklist resolved) — the ONLY
    # thing that should gate the "Generate Full Description"/"Regenerate" button's enabled state.
    # Never infer readiness from the LLM's own judgment or from the button simply being clickable.
    ready_to_generate: bool = False


class JobStatePatch(BaseModel):
    """Direct, silent job_state edit — the draft form uses this for field edits (title,
    experience, salary, skills add/remove, company-context overrides, etc.) instead of sending a
    chat message, so editing the draft never triggers a bot reply or an LLM call.
    jd_text_updates edits the CURRENT job description draft's own text fields directly (e.g. the
    summary) — same principle, a hand-edit shouldn't need an LLM refinement call either.
    """
    field_updates: dict[str, str] = Field(default_factory=dict)
    list_operations: list[ListOperation] = Field(default_factory=list)
    company_overrides: dict[str, str] = Field(default_factory=dict)
    jd_text_updates: dict[str, str] = Field(default_factory=dict)


class CompanyProfileUpdate(BaseModel):
    company_name: str | None = None
    industry: str | None = None
    company_overview: str | None = None
    website: str | None = None
    headquarters: str | None = None
    company_culture: str | None = None
    benefits: str | None = None
    work_life_balance: str | None = None
    why_join_us: str | None = None


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    company_name: str
    industry: str | None = None
    company_overview: str | None = None
    website: str | None = None
    headquarters: str | None = None
    company_culture: str | None = None
    benefits: str | None = None
    work_life_balance: str | None = None
    why_join_us: str | None = None


class SigninRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    company_id: int | None = None
    company_name: str | None = None


class AcceptingApplicationsUpdate(BaseModel):
    accepting_applications: bool
