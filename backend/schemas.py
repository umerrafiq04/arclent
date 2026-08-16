from pydantic import BaseModel


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
