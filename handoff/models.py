"""
EXISTING CODE — copied verbatim (only the import path differs) from backend/models.py.

Pydantic models for: the job being drafted (JobState), the single structured LLM output per
recruiter turn (TurnAnalysis), the generated job description (JobDescriptionDraft), and the
refinement output (JDRefinementOutput). This is the actual "AI contract" of the system — the
exact shape the LLM is forced to produce, and the exact shape a job posting takes once written.

No project-specific dependencies beyond pydantic itself — this file can be copied into the new
platform unchanged.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CompanyProfile(BaseModel):
    id: int | None = None
    company_name: str
    industry: str | None = None
    company_overview: str | None = None
    website: str | None = None
    headquarters: str | None = None
    company_culture: str | None = None
    benefits: str | None = None
    work_life_balance: str | None = None
    why_join_us: str | None = None


# JobState holds only fields that describe the position itself. Reusable company-level
# content (overview/culture/benefits/work-life-balance/why-join-us) is never duplicated
# here — it either comes from CompanyProfile as-is, or is overridden per-job via
# company_overrides, which is keyed to CompanyProfile column names.
class JobState(BaseModel):
    job_title: str | None = None
    job_category: str | None = None
    experience: str | None = None
    location: str | None = None
    work_mode: str | None = None
    employment_type: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    education: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    salary: str | None = None
    deadline: str | None = None
    additional_information: str | None = None
    company_overrides: dict[str, str] = Field(default_factory=dict)


SCALAR_JOB_FIELDS = {
    "job_title", "job_category", "experience", "location", "work_mode",
    "employment_type", "education", "salary", "deadline", "additional_information",
}
LIST_JOB_FIELDS = {"required_skills", "preferred_skills", "responsibilities"}
COMPANY_OVERRIDE_FIELDS = {
    "company_overview", "company_culture", "benefits",
    "work_life_balance", "why_join_us",
}

# Fields a recruiter can always decline to answer without blocking the conversation — everything
# except job_title and required_skills/responsibilities, which the "hard floor" (see
# sufficiency.py) never lets go missing. A turn may only offer a "Skip this" affordance for a
# field in this set, and only when that field isn't currently flagged essential for this
# specific role.
OPTIONAL_SKIPPABLE_FIELDS = {
    "job_category", "experience", "location", "work_mode", "employment_type",
    "education", "salary", "deadline", "additional_information", "preferred_skills",
}


class Intent(str, Enum):
    PROVIDE_INFORMATION = "PROVIDE_INFORMATION"
    CORRECT_INFORMATION = "CORRECT_INFORMATION"
    FINISH_COLLECTING = "FINISH_COLLECTING"
    REQUEST_JD_GENERATION = "REQUEST_JD_GENERATION"
    SELECT_JD = "SELECT_JD"
    REQUEST_REFINEMENT = "REQUEST_REFINEMENT"
    CONFIRM_PUBLISH = "CONFIRM_PUBLISH"
    CHITCHAT_OR_UNCLEAR = "CHITCHAT_OR_UNCLEAR"
    ADVICE_REQUEST = "ADVICE_REQUEST"  # recruiter asked for role advice — never auto-applied
    OFF_TOPIC = "OFF_TOPIC"  # unrelated request — politely redirect, never touches JobState
    DOCUMENT_REVIEW = "DOCUMENT_REVIEW"  # uploaded JD summarized — never auto-applied


# Off-topic / document-review turns carry no confirmed job content at all — block every kind
# of JobState change regardless of what the model returned (defense in depth, not just prompting).
INTENTS_BLOCK_ALL_JOBSTATE_CHANGES = {
    Intent.OFF_TOPIC,
    Intent.DOCUMENT_REVIEW,
}

# Advice turns may still state a real job fact alongside the question (e.g. "I want a GenAI
# Engineer — what should a candidate have?" states the title while asking for advice), so scalar
# field_updates still apply. But the advice itself is always skill/requirement-shaped, so
# list_operations (required_skills/preferred_skills/responsibilities) and company_overrides must
# never be auto-applied from an advice turn — only an explicit later confirmation does that.
INTENTS_BLOCK_LIST_AND_OVERRIDE_CHANGES = {
    Intent.ADVICE_REQUEST,
}


class ListOperation(BaseModel):
    field: Literal["required_skills", "preferred_skills", "responsibilities"]
    operation: Literal["ADD", "REMOVE", "REPLACE"]
    values: list[str]


class TurnAnalysis(BaseModel):
    """THE single structured Mistral output per recruiter turn — this is the entire "AI model
    output contract" for the conversational half of the system.

    Fuses intent detection, field extraction, sufficiency judgment, and the
    natural-language reply into ONE call — never split into separate calls.
    """
    intent: Intent
    field_updates: dict[str, str] = Field(default_factory=dict)
    list_operations: list[ListOperation] = Field(default_factory=list)
    company_overrides: dict[str, str] = Field(default_factory=dict)
    enough_information: bool
    missing_essential: list[str] = Field(default_factory=list)
    selected_version: Literal["1", "2"] | None = None
    asking_about_field: str | None = None
    suggested_options: list[str] = Field(default_factory=list)
    # True when suggested_options are choices the recruiter can combine (e.g. picking several
    # skills: Python + SQL + React), false when only one answer makes sense (e.g. work mode,
    # experience band, yes/no). Governs whether the UI lets the recruiter tap multiple chips
    # before sending, or sends immediately on the first tap.
    options_multi_select: bool = False
    response: str


class JobDescriptionDraft(BaseModel):
    """THE structured output of the SEPARATE generation call (see generate_jd) — the actual
    job description content. Distinct schema/call from TurnAnalysis; the model never writes
    JD prose as part of a normal conversational turn.
    """
    job_title: str | None = None
    requisition_id: str | None = None  # always left null by the model — assigned at publish time
    job_category: str | None = None
    employment_type: str | None = None
    location: str | None = None
    work_mode: str | None = None
    deadline: str | None = None
    company_overview: str | None = None
    job_summary: str | None = None
    about_role: str | None = None
    major_accountabilities: list[str] = Field(default_factory=list)
    minimum_requirements: list[str] = Field(default_factory=list)
    required_qualifications: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    stand_out: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    why_company: str | None = None


class JDRefinementOutput(BaseModel):
    updated_jd: JobDescriptionDraft
    change_summary: str
