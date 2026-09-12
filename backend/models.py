from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field


def _none_as_empty_list(v):
    return v if v is not None else []


def _none_as_empty_dict(v):
    return v if v is not None else {}


# Every structured-output list field the chat/generation LLM populates is typed with one of these
# instead of a bare `list[T]` — verified live against Groq's openai/gpt-oss-120b: when a turn
# genuinely has nothing for a list field (e.g. no suggested_options because the reply isn't a
# question), the model sometimes emits `null` instead of `[]`. A bare `list[T]` schema has no
# `null` variant, so Groq's OWN server-side tool-call validation rejects the whole structured-
# output call outright (a 400, not a parse error langchain could recover from) — and since the
# model does this consistently for a given turn shape, every retry fails the identical way,
# exhausting call_structured's retry budget and falling through to the exception fallback.
# Annotated + a BeforeValidator makes the exported JSON schema explicitly nullable
# (`anyOf: [array, null]`, so Groq accepts `null`) while still normalizing it straight to `[]` at
# parse time — every existing call site keeps getting a real list, never None.
NullableStrList = Annotated[list[str] | None, BeforeValidator(_none_as_empty_list)]

# Same reasoning as NullableStrList, for dict-typed structured-output fields (field_updates,
# company_overrides) — confirmed live against Groq: it emits `null` for these too when a turn has
# nothing to put there, and a bare `dict[str, str]` schema has no `null` variant either.
NullableStrDict = Annotated[dict[str, str] | None, BeforeValidator(_none_as_empty_dict)]


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
    # Recruiter-authored screening questions — manually typed in the draft panel only, never
    # touched by the AI in any way: excluded from every LLM prompt (chat, generation, refinement —
    # see prompts.py) and deterministically stripped from the chat LLM's own list_operations in
    # apply_updates before they're ever applied, even if a future prompt change somehow got the
    # model to try. The one path allowed to write it is the direct, non-LLM job-state PATCH
    # endpoint the draft panel's Custom Questions section calls.
    custom_questions: list[str] = Field(default_factory=list)
    # Which platform(s) this role is hiring for (Facebook, YouTube, Instagram, TikTok, Vimeo,
    # Twitch, Discord) — a proactively-asked, multi-select checklist item (see PLATFORM_OPTIONS/
    # _CHECKLIST_ORDER in nodes.py), unlike custom_questions this one IS AI-touched (the chat LLM
    # sets it via list_operations same as required_skills) and skippable, not mandatory.
    platforms: list[str] = Field(default_factory=list)


SCALAR_JOB_FIELDS = {
    "job_title", "job_category", "experience", "location", "work_mode",
    "employment_type", "education", "salary", "deadline", "additional_information",
}
LIST_JOB_FIELDS = {"required_skills", "preferred_skills", "responsibilities", "custom_questions", "platforms"}
COMPANY_OVERRIDE_FIELDS = {
    "company_overview", "company_culture", "benefits",
    "work_life_balance", "why_join_us",
}

# Fields a recruiter can decline to answer without blocking the conversation. job_title, location,
# salary, and platforms are never in here — sufficiency.py's hard floor never lets any of them go
# missing (an explicit founder decision: the guided flow only proactively asks a small handful of
# questions at all, so none of them should have a Skip affordance — platforms was originally here,
# moved out per a later correction).
# required_skills/responsibilities ARE in here even though the hard floor requires at least one of
# the two: apply_updates only ever lets the "Skip this" affordance through for whichever of the
# pair is NOT the one currently satisfying the hard floor (checked against the
# "required_skills_or_responsibilities" sentinel in missing_essential), so the floor itself can
# never actually be skipped away — only the redundant second ask once the first already covers it.
OPTIONAL_SKIPPABLE_FIELDS = {
    "job_category", "experience", "work_mode", "employment_type",
    "education", "deadline", "additional_information", "preferred_skills",
    "required_skills", "responsibilities",
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


# Shared by TurnAnalysis.list_operations (the chat LLM's own structured output) AND the draft
# panel's direct, non-LLM job-state PATCH endpoint (see JobStatePatch in schemas.py) — the same
# validation model serves both call sites. custom_questions is included here so the direct-patch
# path can write it, but apply_updates (nodes.py) deterministically drops any list_operations
# entry targeting custom_questions BEFORE applying the chat LLM's own output, so this type
# allowing it is not itself a guarantee the AI can never touch it — see apply_updates for the
# actual enforcement.
class ListOperation(BaseModel):
    field: Literal["required_skills", "preferred_skills", "responsibilities", "custom_questions", "platforms"]
    operation: Literal["ADD", "REMOVE", "REPLACE"]
    values: list[str]


# See NullableStrList above — same reasoning, for TurnAnalysis.list_operations specifically.
NullableListOperationList = Annotated[list[ListOperation] | None, BeforeValidator(_none_as_empty_list)]


# The generated job description's OWN list fields (distinct from JobState's — see ListOperation
# above) — what the draft panel's "Full job description detail" editor uses to add/remove items
# directly on the drafted document, the same way ListOperation lets the recruiter edit
# required_skills/preferred_skills/responsibilities on job_state. Only stand_out/benefits live on
# the document itself now — required_skills/preferred_skills/responsibilities were removed from
# JobDescriptionDraft entirely (see the comment there) so ListOperation above is what edits those,
# even from inside the "Full job description detail" section.
# REPLACE (send the whole updated list) is what powers in-place item editing in the draft panel —
# rewriting one existing item's text sends the full array back with that one entry changed, which
# preserves its position; ADD always appends and REMOVE only deletes, neither can edit in place.
class JDListOperation(BaseModel):
    field: Literal["stand_out", "benefits"]
    operation: Literal["ADD", "REMOVE", "REPLACE"]
    values: list[str]


class TurnAnalysis(BaseModel):
    """The single structured Mistral output per recruiter turn.

    Fuses intent detection, field extraction, sufficiency judgment, and the
    natural-language reply into one call — never split into separate calls.
    """
    intent: Intent
    field_updates: NullableStrDict = Field(default_factory=dict)
    list_operations: NullableListOperationList = Field(default_factory=list)
    company_overrides: NullableStrDict = Field(default_factory=dict)
    enough_information: bool
    missing_essential: NullableStrList = Field(default_factory=list)
    selected_version: Literal["1", "2"] | None = None
    asking_about_field: str | None = None
    suggested_options: NullableStrList = Field(default_factory=list)
    # True when suggested_options are choices the recruiter can combine (e.g. picking several
    # skills: Python + SQL + React), false when only one answer makes sense (e.g. work mode,
    # experience band, yes/no). Governs whether the UI lets the recruiter tap multiple chips
    # before sending, or sends immediately on the first tap.
    options_multi_select: bool = False
    response: str


# required_skills/preferred_skills/responsibilities: JobState (see below) is still the single
# source of truth for these — the draft panel, the public listing, and the published DB row all
# read from there, never from this document. Earlier this document ALSO independently generated
# its own major_accountabilities/minimum_requirements/required_qualifications/preferred_qualifications
# as a second, independently-enriched copy of the same three concepts — a reported duplication bug
# (the recruiter saw "Responsibilities" and "Major Accountabilities" as two separate, sometimes
# inconsistent sections) — those four fields are gone for good.
#
# These three ARE still fields here, though, for a narrower reason: a PROOFREAD MIRROR, not a
# second copy. See JD_GENERATION_PROMPT_TEMPLATE's PROOFREAD MIRROR section — the model corrects
# spelling/grammar in JobState's own lists (same item count, same order, same meaning, e.g. "manage
# smalllll team" -> "Manage small team"), and generate_jd writes the correction directly back into
# job_state itself (never stored on the saved JD document — see the length-match guard in
# generate_jd, which discards any output that doesn't preserve item count, so this can never
# silently add/remove/reorder content, only fix how existing items are spelled).
class JobDescriptionDraft(BaseModel):
    job_title: str | None = None
    requisition_id: str | None = None
    job_category: str | None = None
    employment_type: str | None = None
    location: str | None = None
    work_mode: str | None = None
    deadline: str | None = None
    company_overview: str | None = None
    job_summary: str | None = None
    about_role: str | None = None
    required_skills: NullableStrList = Field(default_factory=list)
    preferred_skills: NullableStrList = Field(default_factory=list)
    responsibilities: NullableStrList = Field(default_factory=list)
    stand_out: NullableStrList = Field(default_factory=list)
    benefits: NullableStrList = Field(default_factory=list)
    why_company: str | None = None


class JDRefinementOutput(BaseModel):
    updated_jd: JobDescriptionDraft
    change_summary: str
