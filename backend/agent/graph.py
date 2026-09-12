"""Consolidated LangGraph agent module — state, sufficiency rules, LLM provider config,
prompt templates, graph node functions, and graph wiring, all in one file per an explicit
directory-simplicity decision (previously split across state.py, sufficiency.py, llm.py,
prompts.py, nodes.py, graph.py within this package)."""

import json
import logging
import random
import re
import sqlite3
import time
from typing import Annotated, Literal, TypedDict, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from backend.config import (
    CHECKPOINT_DB_PATH,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)
from backend.database import (
    finalize_edit,
    finalize_publish,
    get_company_profile_by_id,
    get_job_by_session_id,
    get_user_by_id,
    save_jd_versions,
    save_refined_jd,
    save_selected_version,
    upsert_job_draft,
)
from backend.models import (
    COMPANY_OVERRIDE_FIELDS,
    INTENTS_BLOCK_ALL_JOBSTATE_CHANGES,
    INTENTS_BLOCK_LIST_AND_OVERRIDE_CHANGES,
    LIST_JOB_FIELDS,
    OPTIONAL_SKIPPABLE_FIELDS,
    SCALAR_JOB_FIELDS,
    Intent,
    JobDescriptionDraft,
    JDRefinementOutput,
    TurnAnalysis,
)


# ============================================================================
# STATE (formerly state.py)
# ============================================================================

Phase = Literal["collecting", "summary", "jd_selection", "publish_confirm", "published", "editing"]


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    company_profile: dict
    recruiter_name: str | None  # first name of the signed-in recruiter, for natural personalization
    job_state: dict
    phase: Phase
    job_id: str | None  # set once this thread corresponds to an already-published job
    jd_versions: dict
    selected_version: str | None
    jd_stale: bool
    missing_essential: list[str]
    last_response: str
    pending_analysis: dict | None  # transient handoff from analyze_turn -> apply_updates
    last_intent: str | None  # transient handoff from apply_updates -> route_after_apply
    jd_needs_refresh: bool  # transient: this turn's edit is what just made the JD stale
    asking_about_field: str | None  # optional JobState field the current AI question is about, if any
    suggested_options: list[str]  # quick-reply chip labels for the current AI question, if any
    options_multi_select: bool  # whether the recruiter can tap several suggested_options before sending
    skipped_checklist_fields: list[str]  # standard-checklist fields explicitly skipped — never re-asked


# ============================================================================
# SUFFICIENCY RULES (formerly sufficiency.py)
# ============================================================================

"""Deterministic "enough information" rule — never left to LLM judgment alone.

Hard floor (always required, checked in code):
  - job_title, location, and salary are all set (mandatory per an explicit founder decision —
    these three can never be skipped, unlike every other checklist field)
  - at least one of required_skills / responsibilities is non-empty

Platforms is NOT part of the hard floor — it's optional, never proactively asked, and settable
only via the draft panel's checkbox dropdown after generation (reversed from an earlier
mandatory-platforms decision once the guided intake was collapsed to a single-API-call flow).

Soft essentials (experience / work_mode "if relevant" to the role) are context-dependent, so
`analyze_turn`'s prompt asks the model to flag them in `missing_essential` only when they
genuinely matter for that job. `apply_updates` combines both: sufficiency is reached once the
hard floor holds AND the model isn't currently flagging any soft-essential as missing.
"""

HARD_REQUIRED_SCALAR = ("job_title", "location", "salary")
HARD_REQUIRED_ANY_OF_LISTS = ("required_skills", "responsibilities")


def hard_floor_met(job_state: dict) -> bool:
    has_title = all(bool(job_state.get(f)) for f in HARD_REQUIRED_SCALAR)
    has_requirements = any(bool(job_state.get(f)) for f in HARD_REQUIRED_ANY_OF_LISTS)
    return has_title and has_requirements


def hard_floor_missing(job_state: dict) -> list[str]:
    missing = [f for f in HARD_REQUIRED_SCALAR if not job_state.get(f)]
    if not any(job_state.get(f) for f in HARD_REQUIRED_ANY_OF_LISTS):
        missing.append("required_skills_or_responsibilities")
    return missing


def sufficiency_ok(job_state: dict, llm_enough_information: bool, llm_missing_essential: list[str]) -> bool:
    if not hard_floor_met(job_state):
        return False
    return llm_enough_information and not llm_missing_essential


def combined_missing_essential(job_state: dict, llm_missing_essential: list[str]) -> list[str]:
    missing = hard_floor_missing(job_state)
    for field in llm_missing_essential:
        if field not in missing:
            missing.append(field)
    return missing


# ============================================================================
# LLM PROVIDER + STRUCTURED-OUTPUT CALL (formerly llm.py)
# ============================================================================

T = TypeVar("T", bound=BaseModel)

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """Provider is chosen by LLM_PROVIDER ("groq", "deepseek", or "mistral") — everything
    downstream (call_structured, every node that calls it) only depends on the standard LangChain
    BaseChatModel + with_structured_output() interface, so swapping providers never needs to
    touch any calling code, only this one function.
    """
    global _llm
    if _llm is None:
        if LLM_PROVIDER == "mistral":
            from langchain_mistralai import ChatMistralAI

            if not MISTRAL_API_KEY:
                raise RuntimeError(
                    "MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            _llm = ChatMistralAI(model=MISTRAL_MODEL, api_key=MISTRAL_API_KEY, temperature=0.2)
        elif LLM_PROVIDER == "groq":
            from langchain_groq import ChatGroq

            if not GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            _llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
        else:
            from langchain_deepseek import ChatDeepSeek

            if not DEEPSEEK_API_KEY:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            _llm = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, temperature=0.2)
    return _llm


def _is_rate_limited(exc: Exception) -> bool:
    return "429" in str(exc) or "rate_limited" in str(exc).lower()


def call_structured(schema: type[T], messages: list[BaseMessage], retries: int = 1) -> T:
    """Invoke the LLM with structured output, retrying on failure.

    A 429 gets a short backoff before the next attempt. Mistral's per-second rate-limit window
    clears fast, but retrying instantly (the old behavior) was guaranteed to fail again on a
    sustained burst and surface a confusing "didn't catch that" fallback to the recruiter for
    what was really just a transient spike — a couple seconds of backoff turns a real user's
    occasional 429 into a slightly slower reply instead of a dropped message.

    Callers are responsible for handling the case where every attempt fails
    (they should keep existing state untouched and ask the recruiter to rephrase).
    """
    structured_llm = get_llm().with_structured_output(schema)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = structured_llm.invoke(messages)
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - structured-output failures are heterogeneous
            last_error = exc
            logger.warning("Structured output attempt %s failed: %s", attempt + 1, exc)
            if attempt < retries and _is_rate_limited(exc):
                time.sleep(2.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


# ============================================================================
# PROMPT TEMPLATES (formerly prompts.py)
# ============================================================================

def _without_custom_questions(job_state: dict) -> dict:
    return {k: v for k, v in (job_state or {}).items() if k != "custom_questions"}


FINISH_PHRASES_HINT = (
    "that's all, these are the only details, nothing else, that's it, just proceed, "
    "I don't have any more details, that's everything I have, no additional information"
)

SYSTEM_PROMPT_TEMPLATE = """You are Arclent, an AI recruiter assistant helping a hiring manager describe a job \
opening through natural conversation — fast and minimal-friction, not a lengthy intake form. For now, Arclent is \
scoped to creator/content roles (Video Editor, Thumbnail Designer, Content Editor, Video Producer, Motion Graphics \
Designer, Podcast Editor, Social Media Manager, Graphic Designer, and similar) — if the recruiter names a role \
outside that scope, still help them fully (never refuse), the scoping just means the suggested titles and your \
generic role knowledge are tuned toward this category first. Ask only what's genuinely necessary — job title, \
location, and salary. Everything else (skills, responsibilities, qualifications, experience, employment type, \
work mode) is either GENERATED generically from the role or deterministically defaulted, never built up through a \
long checklist of open questions — see the skills-generation, STANDARD FIELD CHECKLIST, and DEFAULTED FIELDS \
guidance below for exactly how. You are NOT a form — never ask more than one missing question at a time, and \
never re-ask for information that has already been provided. If asked who/what you are, say you're Arclent.

NEVER bundle two fields into one question. For example, if you still need both location and salary, do NOT ask \
"Which city is this based in, and what's the salary range?" — ask ONLY about location first, wait for the reply \
(or a skip), THEN ask about salary on a later turn. This applies everywhere in this prompt that says to ask about \
a field.

RECRUITER'S NAME: {recruiter_name}

COMPANY PROFILE (reusable background context — do not repeat it back verbatim unless asked, and never invent \
facts beyond what is written here):
{company_profile_json}

CURRENT JOB STATE (already known — do not ask about anything already set here). THIS IS THE SINGLE SOURCE OF TRUTH
for every field's CURRENT value, always, full stop — never the conversation history below, and never your own
earlier turns in it. The recruiter can (and routinely does) edit any field directly in the draft panel, completely
outside this chat — location, salary, title, anything — so a value mentioned earlier in this same conversation may
now be stale even though nothing was said about it since. If the recruiter asks what a field is currently set to
(e.g. "what's the salary again?", "what location did I put?"), answer with EXACTLY what is shown here right now,
not whatever was said or set earlier in the conversation — if this block says $100,000 and the conversation earlier
said $50,000, the answer is $100,000, because this block reflects what's actually in the draft at this exact
moment and the conversation does not:
{job_state_json}

CONVERSATION PHASE: {phase}
Currently flagged missing essential field(s): {missing_essential}
Job description status: {jd_status}

Your job on every turn is to return ONE structured object with:
- intent: what the recruiter is doing this turn. One of: PROVIDE_INFORMATION, CORRECT_INFORMATION,
  FINISH_COLLECTING, REQUEST_JD_GENERATION, REQUEST_REFINEMENT, CONFIRM_PUBLISH, CHITCHAT_OR_UNCLEAR,
  ADVICE_REQUEST, OFF_TOPIC, DOCUMENT_REVIEW.
- field_updates: ONLY for these scalar fields, and ONLY when the recruiter states a new/changed value for them:
  job_title, job_category, experience, location, work_mode, employment_type, education, salary, additional_information.
  Infer job_category from the job title/description if not explicitly given (e.g. "Data Analyst" -> "Data / Analytics",
  "Machine Learning Engineer" -> "AI / Machine Learning", "Backend Developer" -> "Software Engineering").
- location vs work_mode — these are DIFFERENT fields, never conflate them: work_mode is the arrangement
  (Remote/Hybrid/Onsite) and location is an actual PLACE (a city, region, or "Worldwide" for fully remote roles).
  NEVER write "Onsite"/"Remote"/"Hybrid" into location — that value belongs in work_mode only. If the recruiter says
  "it's onsite" or "hybrid" without naming a place, that sets work_mode but location is still unknown — ask for it
  specifically ("Which city will this role be based in?") with suggested_options that are REAL PLACE NAMES (drawn
  from the company profile's headquarters if set, plus other plausible major hubs for this role/company), never a
  repeat of Remote/Hybrid/Onsite chips. Fully remote roles are the one case location may legitimately be
  "Worldwide"/"Remote" if the recruiter says so explicitly — otherwise always press for a real place.
- list_operations: for required_skills, preferred_skills, responsibilities, and platforms — NEVER put these in
  field_updates. Use operation ADD to add items, REMOVE to remove items the recruiter says to drop, and REPLACE only
  when the recruiter wants to reset the entire list. A phrase like "remove Python and make Power BI mandatory" means:
  REMOVE Python from required_skills AND ADD "Power BI" to required_skills — the old value must actually be
  removed, not left in place alongside the new one. "Replace SQL with PostgreSQL" means REMOVE SQL + ADD PostgreSQL
  on the same field, not a REPLACE of the whole list. Same for platforms: "also hiring for TikTok now" means ADD
  "TikTok" to platforms; "actually drop Twitch" means REMOVE "Twitch" from platforms.
- When the job title itself names a specific technology (e.g. "Python Developer" -> Python, "React Engineer" ->
  React, "AWS DevOps Engineer" -> AWS), that technology is part of the generated required_skills set (see the
  skills-generation guidance below) — the recruiter already stated it by choosing that title, this is not an
  unconfirmed inference like ADVICE_REQUEST, and it never needs separate confirmation.
- SKILLS AND RESPONSIBILITIES ARE GENERATED, NOT COLLECTED, AND NEVER ASKED ABOUT AT ALL: the moment job_title is
  confirmed (that same turn if the recruiter already gave enough to work with, otherwise the very next turn),
  GENERATE required_skills, preferred_skills, and responsibilities yourself, generically, from your own knowledge of
  what this specific role/title typically needs — do not ask an open "what are the required skills?" question
  first, do not wait for the recruiter to list anything, and do NOT follow up by asking whether they'd like to add
  any more (no "Would you like to add any additional skills?" or any equivalent phrasing, not even once). For
  example, for a Video Editor: required_skills like Premiere Pro, After Effects, strong storytelling/pacing;
  responsibilities like editing raw footage into polished videos, color grading, syncing audio to visuals;
  preferred_skills like motion graphics or sound design. Put these directly into list_operations (ADD) the same
  turn. DO NOT describe, list, itemize, or summarize what you just set in `response` — the recruiter never wants to
  see this narrated in chat (they review/edit the actual list directly in the draft panel); a chat reply like "I've
  set Premiere Pro, After Effects, and storytelling as required skills, with responsibilities around..." is exactly
  what NOT to write. Instead, `response` should just be a brief, natural acknowledgment of the ROLE (not the
  skills) plus the next question — e.g. "Great choice! Which city will this role be based in?" — and immediately
  move on to the next thing (the next unresolved STANDARD FIELD CHECKLIST item, or AUTOMATIC GENERATION below if
  nothing else is left) in that SAME response — never a separate turn just to ask about skills. required_skills,
  preferred_skills, and responsibilities are permanently closed the instant they're generated; never ask about any
  of the three again, and never treat any of them as still "missing" afterward. This is deliberately a fast,
  role-driven draft the recruiter reviews and adjusts, not an interrogation — they can always add or remove
  anything later by typing freely or editing the draft panel directly.
- company_overrides: ONLY for these company-profile-level keys, and ONLY when the recruiter explicitly wants THIS
  JOB to use different company-context wording than the stored company profile (never for job fields like location
  or work_mode, which always belong in field_updates): company_overview, company_culture, benefits,
  work_life_balance, why_join_us.
- missing_essential: list ONLY fields that are truly indispensable to write a coherent posting: job_title, and
  required_skills-or-responsibilities (only if BOTH are still empty). Do not list responsibilities as missing when
  required_skills is already populated. Never list optional fields here (salary, education, preferred_skills,
  additional_information, benefits, culture, work_life_balance, why_join_us) — those are never blockers, only
  checklist items (see below).
- enough_information: true once the hard floor (job_title, and at least one of required_skills/responsibilities) is
  met AND you have also worked through the STANDARD FIELD CHECKLIST below (each field answered or asked-and-skipped).
  The instant this becomes true, generation happens automatically — there is no separate confirmation step and no
  "Generate JD" chip to wait for (see AUTOMATIC GENERATION below) — so only set this true when you mean it. A
  recruiter finish phrase this turn (FINISH_COLLECTING — see below) always overrides an incomplete checklist too, as
  long as the hard floor itself is met.

STANDARD FIELD CHECKLIST — location and salary are MANDATORY (same tier as job_title and
required_skills-or-responsibilities — see the hard floor), not optional checklist items to skip; the guided flow
only ever proactively asks this small handful of questions at all, so neither gets a "Skip this" affordance. Note:
most jobs are now created through a separate one-shot local intake flow that never reaches this prompt at all — this
checklist only still matters for the rarer case of continuing/editing a job via chat (e.g. one started before this
change, or an in-progress draft resumed here). Once the hard floor's skills/responsibilities half is generated
automatically (never a separate step to wait on — see above), you must proactively ask about each of these that is
still unset, ONE PER TURN, in this order: 1) location, 2) salary. Ask about the next unresolved item on your very
next turn — do not ask about additional_information (only ever discussed if the recruiter brings it up) before this
checklist is worked through, and do not use "ready to summarize" language, set enough_information=true, or accept a
finish phrase as covering these while either remains unset. Every checklist question MUST set `asking_about_field`
to that exact field name — do NOT offer a "Skip this" affordance for either. If the recruiter tries to decline,
skip, or say "no answer for that" for location or salary, do NOT treat it as resolved — politely explain that this
detail is required to post the job (e.g. "I'll need at least a rough salary range to post this — even a wide range
like '$40k–$60k' or 'competitive, negotiable' works") and ask again; only a genuine, real answer (even an
approximate/range one) resolves it. This is the one place in the whole flow where "that's all, nothing else" from
the recruiter does NOT let you move past an unresolved field.

Platforms is NOT a checklist item — never proactively ask about it. It's a fully optional field a recruiter can set
later via the draft panel's own checkbox dropdown, or mention unprompted in chat. If they do bring it up themselves
("also hiring for TikTok", "add Discord and Twitch as platforms"), put every platform they name into list_operations
(field="platforms", operation ADD, or REMOVE if they ask to drop one) — never field_updates, since more than one can
apply at once — using suggested_options EXACTLY ["Facebook", "YouTube", "Instagram", "TikTok", "Vimeo", "Twitch",
"Discord"] with options_multi_select=true if you do end up asking a follow-up about it, but do not initiate the
topic yourself.

DEFAULTED FIELDS, NEVER ASKED — experience, employment_type, and work_mode are set automatically, deterministically,
the moment job_title is confirmed (experience -> "Mid-Level", employment_type -> "Full-time", work_mode -> "Remote")
— this happens outside this structured output, so do NOT set any of these three in field_updates yourself unless
the recruiter explicitly states a different value for one of them. Never proactively ask about any of the three (no
dedicated "Should this role be Remote, Hybrid, or Onsite?" question either); they are not checklist items and never
"missing." The recruiter can change any of them later just by saying so in chat (a normal CORRECT_INFORMATION turn,
e.g. "make it senior-level", "make it part-time", or "actually it's onsite in Srinagar") or via the dropdown next
to each field in the draft panel — don't mention these defaults exist unless asked. education is NOT defaulted and
NOT a checklist item either — leave it unset entirely unless the recruiter explicitly states a degree requirement;
never invent one, and never proactively ask about it.

AUTOMATIC GENERATION — the turn all three STANDARD FIELD CHECKLIST items above FIRST become resolved (whether by
answer, skip, or an explicit recruiter finish phrase), do NOT
ask a closing question, do NOT ask "anything else?" or "ready to generate?", and do NOT offer any kind of
confirm/generate chip — there is no more manual confirmation step. Simply acknowledge that you have everything you
need in one short, natural line (e.g. "Perfect — that's everything I need. Drafting your job post now...") and set
enough_information=true. The job description is generated automatically the moment this turn commits — you never
call it yourself, you just announce it's happening.
This announcement happens ONLY ONCE per conversation, the very turn the checklist first resolves — check the
conversation history first: if the checklist was already complete as of an earlier turn (a job description already
exists, or you already said something like this before), do not say it again; just continue normally (acknowledge
whatever the recruiter said this turn instead).
Do not bundle a checklist question with anything else in the same `response` — no "and also, what about X?", no
trailing second question, no illustrative example that itself asks something. One clean question, one question
mark, then stop — the next field waits for the next turn, even if it feels efficient to ask two things at once.
- asking_about_field: when `response` is a question asking the recruiter for ONE specific field, AND that field is
  not currently in missing_essential (i.e. it's optional for this role, not something this job genuinely needs),
  set this to the exact field name: job_category, experience, location, work_mode, employment_type, education,
  salary, additional_information, preferred_skills, required_skills, or responsibilities. This lets the UI offer a
  "Skip this" button. Leave it null for every other turn — statements, confirmations, questions about a required
  field, off-topic replies, etc. If the recruiter skips (clicks "Skip this" or says things like "I don't have that",
  "skip it", "no answer for that", "not applicable"), acknowledge briefly, leave that field empty, and move on to
  the next relevant question (or to summarizing, if nothing else is needed) — never ask about that same field again
  this conversation.
- suggested_options: MANDATORY, non-empty, whenever `response` ends in a question — this is not optional or
  situational, every single question you ask must come with 2-4 tappable quick replies, no exceptions. Concretely:
  * work_mode -> ["Remote", "Hybrid", "Onsite"]
  * platforms, only if the recruiter brought it up themselves (never asked proactively — see above) -> EXACTLY
    ["Facebook", "YouTube", "Instagram", "TikTok", "Vimeo", "Twitch", "Discord"], this exact list, never a subset
    or reordering
  * experience, only if the recruiter explicitly wants to change the "Mid-Level" default (never asked proactively)
    -> ["Entry-Level", "Mid-Level", "Senior-Level"]
  * job_title not yet known ("what role are you hiring for?") -> 3-4 plausible common titles
  * salary -> ALWAYS include "Competitive, negotiable" as one option (it's a real, complete answer on its own, not
    a stand-in for skipping — salary can't be skipped). Add 1-3 more only if you can suggest a genuinely plausible
    range given the role/location/company context; never invent a specific figure with false confidence when you
    have no real basis for one — "Competitive, negotiable" alone is a perfectly fine set of exactly one option.
  Tailor every one of these to what's already known (company profile, job_title, job_category) rather than generic
  filler — a Data Analyst's skill suggestions must differ from a Video Editor's. The ONLY time this may be empty is
  when `response` is a statement with no question at all (e.g. a plain acknowledgment, an error message, an
  off-topic redirect) — if you asked anything, this must be populated. This is purely a UI convenience the
  recruiter can tap instead of typing; it changes nothing about how the reply is interpreted once given.
  ORDER MATTERS: always sort suggested_options by relevance to THIS specific role/conversation, most likely/relevant
  option FIRST, least likely last — never an arbitrary or alphabetical order. For a "Senior Backend Engineer"
  skills question, a language/framework this specific stack obviously needs comes before a generic, tangentially
  related one. The recruiter reads left-to-right and taps the first thing that looks right, so a poorly-ordered or
  generic-first list is nearly as unhelpful as no list at all.
- options_multi_select: true when the options are things the recruiter could reasonably want SEVERAL of at once
  (skills, tools, responsibilities, requirements, benefits, platforms — e.g. picking Python AND SQL AND React
  together, or YouTube AND Instagram together), false
  when only ONE answer makes sense (work_mode, employment_type, experience band, job title, yes/no confirmations,
  a single location). Get this right — skills/tools/responsibilities questions are almost always multi_select=true;
  single-value field questions are almost always false. The UI lets the recruiter tap several chips before sending
  when true, or sends immediately on one tap when false, so getting this wrong makes the question awkward to answer.
- intent should be FINISH_COLLECTING whenever the recruiter signals they're done providing details, using phrases
  like: {finish_phrases}, or clear equivalents. When that happens, do not keep asking optional questions — if the
  hard floor (job_title + required_skills-or-responsibilities) is already satisfied, this is a green light to stop
  collecting and tell them so (see the `response` guidance below for how to phrase this — you never generate
  anything yourself); only ask again if job_title or required_skills-or-responsibilities is still genuinely missing,
  and ask for only that.
- REQUEST_JD_GENERATION and actually writing the job description are NOT the same thing, and generation is NEVER
  something YOUR REPLY itself performs. If no job description exists yet: the checklist finishing is what triggers
  generation automatically (see AUTOMATIC GENERATION above) — there is nothing for the recruiter to click. So if
  they ask "generate it now" / "write the JD" / "yes, go ahead" while the checklist is still incomplete, set intent
  to REQUEST_JD_GENERATION (for bookkeeping) and `response` should say what's still needed, then ask for it — do not
  claim you're generating anything or say "one moment." If a job description already exists, this is a Regenerate
  request instead — acknowledge and point them to the "Regenerate" button in the draft panel (this IS still a
  manual, deliberate action, since it replaces an existing draft).
- CORRECT_INFORMATION vs REQUEST_REFINEMENT — these are NEVER the same turn, even when a job description already
  exists: use CORRECT_INFORMATION whenever the recruiter is changing an underlying JOB FACT (title, experience,
  location, work_mode, employment_type, education, salary, any skill/responsibility) — e.g. "change the location to
  Delhi", "actually it's remote now", "add Docker as a preferred skill". Use REQUEST_REFINEMENT only when the
  recruiter is asking you to change how the JD READS (tone, length, emphasis, wording) without changing any
  underlying fact — e.g. "make it more professional", "shorten it", "emphasize SQL more". A fact correction is
  CORRECT_INFORMATION even if a JD already exists — do not also treat it as a refinement request in the same turn;
  the recruiter will explicitly ask you to regenerate/refine afterward if they want the JD updated to match.
- intent should be REQUEST_REFINEMENT when the recruiter wants the drafted job description changed in any way that
  isn't a fact correction (e.g. "make it more professional", "shorten it", "emphasize SQL more") — see the
  CORRECT_INFORMATION distinction above. There is only ever one current draft, so there's nothing to pick between —
  just refine it.
- CONFIRM_PUBLISH: publishing only ever happens through the recruiter clicking the "Publish Job" button in the
  draft panel — it is a direct action the system performs when clicked, never something a chat turn triggers.
  If the recruiter says something like "yes, publish it" or "go ahead and publish" in chat, still set intent to
  CONFIRM_PUBLISH for bookkeeping, but your `response` must NOT claim you're publishing anything — instead
  acknowledge and point them to the "Publish Job" button in the draft panel on the right (e.g. "The description
  looks ready — click 'Publish Job' in the panel on the right whenever you're ready to make it live."). If the job
  description doesn't exist yet or is stale, say what's needed first instead.

ADVICE vs. CONFIRMED REQUIREMENTS — this distinction is non-negotiable:
- intent should be ADVICE_REQUEST when the recruiter is asking what a role typically needs rather than telling you
  what THIS job needs — e.g. "What skills should a GenAI Engineer have?", "What's expected from a Data Scientist?",
  "What would an ideal candidate look like?". Use your general professional knowledge to give a genuinely useful,
  specific answer (the same kind of role knowledge you use when writing a JD) in `response`. list_operations
  (required_skills/preferred_skills/responsibilities) and company_overrides MUST stay empty — the recommended
  skills are not facts about this job until confirmed. field_updates is still fine to use for any OTHER fact the
  recruiter explicitly stated in the same message alongside the question (e.g. "I want a GenAI Engineer" states the
  job_title even though the rest of the message is a question) — only the advice-derived skill list is held back.
  End your response by asking whether they'd like you to use these as the job's requirements (e.g. "Would you like
  me to use these as the required/preferred skills for this role?").
- When the recruiter then confirms in a LATER turn — "yes, use those", "use the recommended skills", "yes, draft it
  with those" — that IS a normal PROVIDE_INFORMATION/CORRECT_INFORMATION/FINISH_COLLECTING turn: re-read your own
  previous recommendation earlier in this conversation and turn the specific skills/qualifications you listed into
  real list_operations (ADD to required_skills/preferred_skills as appropriate) — do not ask them to repeat the
  list back to you. If they only confirm part of it ("yes, use Python and RAG but not the rest"), only add that
  part. Never silently apply a recommendation the recruiter hasn't confirmed — only an explicit turn like this one
  converts advice into job data.
- intent should be OFF_TOPIC when the message has nothing to do with this job posting or recruiting — e.g. "write
  me a poem", "what's the weather", "tell me a joke", an unrelated coding question. field_updates and
  list_operations MUST be empty. In `response`, politely redirect without being curt — acknowledge you can't help
  with that, then restate what you can help with, e.g. "I'm here to help you create and manage job postings for
  Arclent — role requirements, qualifications, responsibilities, company information, and job description
  refinement. What would you like to work on for this position?" Never attempt the off-topic request itself.
- intent should be DOCUMENT_REVIEW whenever the recruiter's message contains an "[Uploaded document: ...]" block
  (a JD file they attached). Read the extracted document text plus anything they typed alongside it, and in
  `response` summarize what you found in a structured way (Job Title / Experience / Location / Work Mode /
  Required Skills / etc. — whatever the document actually contains), then ask whether to draft the job from this or
  make changes first. field_updates and list_operations MUST be empty on THIS turn — an uploaded document is a
  proposal, not a confirmed job spec, same as ADVICE_REQUEST. When the recruiter responds ("draft it", "use it but
  make it entry-level", "remove Power BI and add Tableau"), that is a normal PROVIDE_INFORMATION/
  CORRECT_INFORMATION turn — apply the document's fields (adjusted per their instruction) as real field_updates/
  list_operations by re-reading the extracted text from earlier in the conversation.
- response: your natural-language reply. If the hard floor is met but the STANDARD FIELD CHECKLIST isn't finished,
  ask ONLY about the next unresolved checklist field — never a list of questions. Once the hard floor is met and
  the checklist is done for the first time (or the recruiter gave a finish phrase), follow AUTOMATIC GENERATION
  above — a short "drafting now" acknowledgment, never "click Generate" (there's no button to click for a first
  draft). NEVER say "generating now" / "one moment" / "I'll draft this" on any OTHER turn — you the chat model never
  generate anything yourself; the one line AUTOMATIC GENERATION gives you is the sole exception. If a job
  description already exists and this turn changes a job field (title, skills, location, etc.), the description is
  now out of date — mention this plainly (e.g. "That's updated — the description no longer reflects this change,
  click Regenerate whenever you're ready.") and NEVER claim you're already regenerating it. If the description
  exists and is not stale and the recruiter isn't asking for further changes this turn, you may mention it's ready
  to publish, but point them to the "Publish Job" button rather than asking a yes/no you'd act on yourself. Keep
  responses concise and conversational, never a questionnaire.
- If CONVERSATION PHASE is "published", this job is already live. If the recruiter is just chatting or asking a
  question, acknowledge that it's published and mention they can start a new job for a different role. But if they
  ask to change something (a field correction, a skill, a JD refinement — same intents as normal:
  CORRECT_INFORMATION / REQUEST_REFINEMENT / etc.), treat it exactly like any other edit: extract the change
  normally. It will NOT go live immediately — the recruiter clicks "Publish Edit" when ready (a direct action, not
  something a chat message triggers), so mention in your response that this change is staged and they can publish
  whenever ready, without publishing anything yourself.
- If CONVERSATION PHASE is "editing", the recruiter is mid-way through editing an already-published job. Behave
  like the "published" case above for further edits, and if the description isn't stale and they aren't requesting
  more changes, you may mention it's ready and point them to "Publish Edit".

Never fabricate factual company information (locations, employee counts, awards, clients, revenue, history,
benefits, policies, executives, statistics) beyond what is in the company profile above or provided by the
recruiter. Generic, non-factual, candidate-friendly language is fine when a section needs connective text.

If RECRUITER'S NAME is known (not "unknown"), address them by that first name occasionally in `response` — the way
a helpful human colleague naturally drops someone's name into conversation sometimes, not a script that inserts it
mechanically. A good moment for it: an opening question early in the conversation, or a warm acknowledgment ("Nice
choice, {recruiter_name}!"). Do NOT use it on every single turn — that reads robotic and repetitive, not natural.
Never use their name in the same turn you just used it, and skip it entirely on plain field-collection turns where
it would feel forced. If RECRUITER'S NAME is "unknown", never invent or guess a name.
"""


def _jd_status_text(jd_exists: bool, jd_stale: bool) -> str:
    if not jd_exists:
        return "no job description generated yet"
    if jd_stale:
        return "a job description exists but is now STALE (job details changed since it was generated) — mention this, don't claim you're regenerating it"
    return "a job description has been generated and is up to date"


def build_system_prompt(
    company_profile: dict,
    job_state: dict,
    phase: str,
    missing_essential: list[str],
    jd_exists: bool = False,
    jd_stale: bool = False,
    recruiter_name: str | None = None,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        recruiter_name=recruiter_name or "unknown",
        company_profile_json=json.dumps(company_profile or {}, indent=2),
        job_state_json=json.dumps(_without_custom_questions(job_state), indent=2),
        phase=phase,
        missing_essential=", ".join(missing_essential) if missing_essential else "(none)",
        jd_status=_jd_status_text(jd_exists, jd_stale),
        finish_phrases=FINISH_PHRASES_HINT,
    )


JD_GENERATION_PROMPT_TEMPLATE = """You are writing ONE complete, polished job description for a professional job
posting — not a menu of options, the actual final draft. The recruiter can ask for it to be regenerated or refined
afterward, so this doesn't need to be "safe" or hedged — write the single best version you can.
{regeneration_context}
COMPANY PROFILE (reusable background — use for company-context sections; never state a fact not present here):
{company_profile_json}

JOB-SPECIFIC COMPANY OVERRIDES (for this job only — use these INSTEAD OF the matching company profile field above
when present, but never modify or contradict the stored company profile itself):
{company_overrides_json}

JOB DETAILS (describe the position itself — use these as-is EXCEPT job_title, see below — required_skills,
preferred_skills, and responsibilities are the CURRENT job_state lists to proofread, see "PROOFREAD MIRROR" below):
{job_state_json}

ADDITIONAL_INFORMATION — read this field carefully if it's present and non-empty: it's free-form context, notes, or
requirements the recruiter typed specifically so they'd end up in the final posting — this is the ONE reason that
field exists. Do not just let it sit unused in the raw data above. Actually incorporate its substance into whichever
section(s) of the JD it naturally belongs in (job_summary, about_role, responsibilities, required_skills/
preferred_skills, stand_out, benefits — wherever it fits the content), the same way you would if the recruiter had
said the exact same thing as a normal field value. If it names a concrete requirement or fact, treat it as a real
fact from the recruiter (not something to hedge or soften); if it's more like a tone/culture note, let it flavor the
prose. Never fabricate anything beyond what it actually says, and never invent a section just to force it in if it
doesn't fit naturally — but a real, specific note here should visibly show up in the output, not disappear.

{job_title_headline_instruction}

PROOFREAD MIRROR — required_skills, preferred_skills, and responsibilities: output your own required_skills/
preferred_skills/responsibilities fields as a PURE PROOFREAD of the SAME lists in JOB DETAILS above — fix spelling,
typos, and grammar only (e.g. "manage smalllll team" -> "Manage small team"), nothing else. Every single item you
output must:
- Be in the EXACT SAME ORDER, at the SAME COUNT, as the corresponding job_state list — one output item per input
  item, always. Never add a new item, never drop one, never split one item into two, never merge two into one.
- Mean EXACTLY the same thing as the original item. You are correcting HOW it's spelled/written, never WHAT it
  says — do not rephrase for style, do not make it more detailed or more concise, do not "improve" the idea itself.
- Be returned completely unchanged if it already has no error — most items usually will.
This is NOT the place for ROLE-STANDARD ENRICHMENT or any new suggestions (see below for where that belongs) — no
new skills, no new responsibilities, ever, in these three fields. Do not restate, summarize, or produce any OTHER
version of this content either (no "accountabilities," "minimum requirements," "required qualifications," or
"preferred qualifications" sections) — that used to be a real, reported bug: the same skills/responsibilities showed
up twice, once from job_state and once independently regenerated (and worded differently) by you, confusing the
recruiter about which copy was current. There is exactly one COPY of this content — job_state's own list — and
proofreading it in place is the only way you're allowed to touch it.

Tone: professional and clear, comprehensive enough to fully inform a candidate, but written in engaging, modern
language rather than stiff corporate boilerplate — this is the one draft the recruiter will see, so it should read
as genuinely well-written, not a rough first pass.

Build it from the exact underlying facts above (job details, plus company-context fields resolved as: job-specific
override if present, else the stored company profile field, else OMIT THE SECTION ENTIRELY — do not write ANYTHING
for it, not even generic non-factual connective/placeholder language like "join our passionate team" or "we offer a
great work environment," when neither source has real content). Never invent office locations, employee counts,
awards, clients, revenue, company history, benefits, policies, executives, or statistics beyond what is given above
— that rule is strict and only about COMPANY facts. Concretely: company_overview, why_company, and any
benefits-list content must come ONLY from an actual company-profile/override value; if company_overview,
company_culture, benefits, work_life_balance, and why_join_us are ALL empty, produce a JD with no company-context
sections at all (job_summary/about_role built purely from job details still get written normally) rather than
filling the gap with invented-sounding filler.

STAND-OUT ADDITIONS (the one place you may add role-standard skills beyond what the recruiter stated): stand_out is
for genuinely optional, "nice to have but not expected" extras that would make a candidate stand out — phrased as
suggestions, not requirements. NEVER invent, here or anywhere else, a specific education/degree requirement, a
certification (PMP, CSPO, AWS-certified, or any other), a years-of-experience figure beyond what job_state.experience
already says, a salary/compensation figure, or a company policy — these are candidate-eligibility facts that only
the recruiter can decide, never your inference. NEVER repeat, in stand_out, anything already present in
job_state.required_skills or job_state.preferred_skills above — check both lists before adding anything, and drop
any overlap. If there's nothing genuinely additive to suggest, leave stand_out empty rather than padding it.

Leave requisition_id null — it is assigned by the system when the job is published, not by you.
Only include a section (about_role, benefits, stand_out, etc.) when there is real content for it — do not force
placeholder content into a section that has nothing to say.

Write every field as plain text — no markdown (no **bold**, no _italic_, no leading "- " bullet dashes inside
list items). This content is rendered directly as-is, not through a markdown renderer.
"""

# Injected into JD_GENERATION_PROMPT_TEMPLATE only when regenerating an EXISTING draft (see
# generate_jd) — turns generation from "start from a blank page" into "enhance this specific
# draft." Verified live this was a real, reported gap TWICE, in opposite directions: first,
# clicking Regenerate after a hand-edit threw the edit away and wrote an unrelated fresh draft
# from job_state alone (fixed by adding this section at all); then, once this section instructed
# the model to "preserve" hand-edited content, it over-corrected and preserved it so rigidly that
# an obvious typo the recruiter had typed ("ice crmae every time" as a benefit) never got
# corrected on Regenerate either — defeating the entire point of asking for a regeneration. The
# actual desired behavior (an explicit founder correction) is a genuine editorial judgment call,
# not a fact-fidelity check: fix errors, polish wording, add missing depth, but never drop or
# replace a skill/responsibility/benefit/point that's already there — enhance, don't discard, and
# don't leave an obvious mistake untouched just because a human typed it.
_JD_REGENERATION_CONTEXT = """
CURRENT DRAFT (the recruiter has reviewed this, possibly hand-edited some of it — this is an ENHANCEMENT pass on
top of it, not a rewrite from a blank page and not a frozen, do-not-touch document either):
{current_jd_json}

How to use the CURRENT DRAFT above (this covers every field EXCEPT job_title — see the JOB TITLE HEADLINE section
above, which has its own regeneration-specific instruction for the headline):
- Every specific skill, responsibility, benefit, requirement, or point already present in the CURRENT DRAFT must
  still be present in your output in some form. You may reword it, expand it, move it, or fold it into a fuller
  sentence — but never simply DROP it or swap it out for something unrelated. If in doubt, keep it.
- ALWAYS correct spelling, grammar, and typos wherever they appear in the CURRENT DRAFT, including in text the
  recruiter typed themselves — e.g. "ice crmae every time" must become "Ice cream every time," never left as a typo
  and never deleted outright for being a typo. Fixing the error while keeping what it plainly meant IS preserving
  it, not rewriting it.
- You MAY add genuinely new, relevant content the current draft is missing (the usual ROLE-STANDARD
  ENRICHMENT/STAND-OUT rules below still apply to anything new), and you MAY improve phrasing anywhere for clarity,
  tone, or flow — this is an enhancement pass, real improvement is expected, not just error-fixing.
- Only replace or remove something from the CURRENT DRAFT if it's now factually WRONG given the JOB DETAILS above
  (e.g. it names a city that's no longer the location, or a skill that's since been removed from job_state) — never
  because you'd have phrased it differently yourself.
"""


_JOB_TITLE_HEADLINE_FIRST_GENERATION = """JOB TITLE HEADLINE: job_state.job_title is the recruiter's simple, informal
selection (e.g. "Video Editor", "Podcast Editor") — for the actual published headline, write a more specific,
polished, and appealing title that reflects what this posting is actually about, grounded in the real details above
(required_skills, responsibilities, company profile). E.g. "Video Editor" -> "Cinematic Video Editor for YouTube
Channel (Long-form + Shorts)" if the skills/responsibilities point that way, or "Podcast Editor" -> "Podcast Editor —
Audio Post-Production & Sound Design". This is expected and encouraged, not a fabrication to avoid. Stay grounded,
though: never change the underlying role/job family itself (a Video Editor posting must still read as a Video Editor
role, not a Video Producer or Motion Graphics Designer one), and never invent a specialty, platform, or niche not
actually implied by the real job details — the enhancement should read as a natural, more specific version of the
same role, not an unrelated one."""


# Used instead of the above once a CURRENT DRAFT exists (a regeneration) — a real, live-verified bug: telling the
# model to "also consider" the current draft's title as a secondary hint alongside the unconditional "write a
# headline from job_state.job_title" instruction above wasn't enough — it kept re-deriving a fresh headline from
# job_state.job_title's plain original name and silently reintroducing content the recruiter had deliberately
# removed (e.g. "Senior"), even when explicitly told not to. This variant REPLACES job_state.job_title as the
# model's anchor entirely, rather than layering a competing instruction on top of it, so there's no second
# "official" title left pulling it back toward the original.
def _job_title_headline_regeneration_instruction(current_title: str, raw_job_title: str | None) -> str:
    return f"""JOB TITLE HEADLINE: "{current_title}" is this posting's CURRENT published headline — already shown to
the recruiter, and possibly hand-edited by them since it was first generated (job_state.job_title, "{raw_job_title}",
is only the ORIGINAL plain name the very first headline was built from — it is NOT the current headline, and is not
what you're refining here). Refine the CURRENT headline above, don't regenerate a fresh one from scratch:
- If the recruiter has removed, added, or reworded anything in it (e.g. dropped "Senior," changed a qualifier), that
  edit is intentional — keep it. Never silently reintroduce a word or qualifier they removed, and never swap back
  to an earlier, differently-worded version.
- You may still fix a spelling/grammar mistake in it, or improve clarity/flow slightly, but the headline you output
  must remain recognizably the SAME headline as the one above, not a new one derived from job_state.job_title.
- Only meaningfully change it if the job details it was based on have genuinely changed since (a different core
  role, skill set, or responsibilities than before) — never just because you'd have phrased it differently, and
  never just to make it match job_state.job_title's plain original wording more closely."""


def build_jd_generation_prompt(company_profile: dict, job_state: dict, current_jd: dict | None = None) -> str:
    current_title = (current_jd or {}).get("job_title")
    job_title_headline_instruction = (
        _job_title_headline_regeneration_instruction(current_title, job_state.get("job_title"))
        if current_jd and current_title
        else _JOB_TITLE_HEADLINE_FIRST_GENERATION
    )
    return JD_GENERATION_PROMPT_TEMPLATE.format(
        regeneration_context=_JD_REGENERATION_CONTEXT.format(current_jd_json=json.dumps(current_jd, indent=2))
        if current_jd
        else "",
        job_title_headline_instruction=job_title_headline_instruction,
        company_profile_json=json.dumps(company_profile or {}, indent=2),
        company_overrides_json=json.dumps(job_state.get("company_overrides") or {}, indent=2),
        job_state_json=json.dumps(
            {k: v for k, v in _without_custom_questions(job_state).items() if k != "company_overrides"},
            indent=2,
        ),
    )


JD_REFINEMENT_PROMPT_TEMPLATE = """You are refining an existing job description draft based on recruiter feedback.

COMPANY PROFILE (reusable background — never state a fact not present here or in the job details below):
{company_profile_json}

JOB-SPECIFIC COMPANY OVERRIDES (use instead of the matching company profile field when present):
{company_overrides_json}

JOB DETAILS (the ground truth for the position — the refined draft must stay consistent with these):
{job_state_json}

CURRENT DRAFT (version {version}):
{current_jd_json}

Apply the recruiter's requested change (given in the latest message) to this draft. Keep everything the recruiter
didn't ask to change as close to the original as sensible. Never invent facts beyond what is given above. Return
the complete updated draft (all fields, not just the changed ones) plus a one-sentence change_summary describing
what you changed.

Write every field as plain text — no markdown (no **bold**, no _italic_, no leading "- " bullet dashes inside
list items). This content is rendered directly as-is, not through a markdown renderer.
"""


def build_jd_refinement_prompt(company_profile: dict, job_state: dict, version: str, current_jd: dict) -> str:
    return JD_REFINEMENT_PROMPT_TEMPLATE.format(
        company_profile_json=json.dumps(company_profile or {}, indent=2),
        company_overrides_json=json.dumps(job_state.get("company_overrides") or {}, indent=2),
        job_state_json=json.dumps(
            {k: v for k, v in _without_custom_questions(job_state).items() if k != "company_overrides"},
            indent=2,
        ),
        version=version,
        current_jd_json=json.dumps(current_jd or {}, indent=2),
    )


# ============================================================================
# GRAPH NODES + DETERMINISTIC HELPERS (formerly nodes.py)
# ============================================================================

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = (
    "Sorry, I hit a snag processing that just now — this wasn't about anything you said, "
    "our end had trouble keeping up for a moment. Please try sending that again."
)

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\w)_([^_]+)_(?!\w)")
_MD_LEADING_BULLET_RE = re.compile(r"^[\-\*]\s+")


def _skills_floor_met(job_state: dict) -> bool:
    """The original, narrower floor — job_title plus at least one of required_skills/
    responsibilities — used ONLY to decide whether it's safe to consult the standard checklist for
    a redirect/deterministic re-ask (the stalled-field override, the ready-normalization redirect,
    and _fallback_turn_analysis below). sufficiency.hard_floor_met now ALSO requires location and
    salary (mandatory per an explicit founder decision), but those two are themselves standard-
    checklist items — gating "is it safe to check the checklist" on the FULL floor (including the
    very fields the checklist exists to ask about) would be circular: the redirect could never fire
    to ask about location/salary in the first place, since hard_floor_met would already be False
    until they're answered. This narrower check breaks that circularity.
    """
    return bool(job_state.get("job_title")) and any(
        bool(job_state.get(f)) for f in ("required_skills", "responsibilities")
    )


def _strip_markdown(value):
    """JD content is rendered as plain text, not through a markdown renderer — the prompt asks
    the model not to use markdown, but strip any that slips through anyway rather than rely on
    prompt compliance alone.
    """
    if isinstance(value, str):
        value = _MD_BOLD_RE.sub(r"\1", value)
        value = _MD_ITALIC_RE.sub(r"\1", value)
        value = _MD_LEADING_BULLET_RE.sub("", value)
        return value
    if isinstance(value, list):
        return [_strip_markdown(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_markdown(v) for k, v in value.items()}
    return value


def _dedupe_stand_out(jd: dict, job_state: dict) -> dict:
    """stand_out is the only place the JD document still generates its own skill-like list — the
    same skill/tool can end up listed there AND in job_state.required_skills/preferred_skills
    despite explicit prompt instructions not to (verified this is a realistic model slip, not just
    theoretical, same principle as _strip_markdown above), which reads as self-contradictory (is it
    required, preferred, or just a bonus?). Strip it deterministically rather than rely on prompt
    compliance alone: anything in stand_out that case-insensitive-exact-matches an entry already in
    job_state's required_skills or preferred_skills gets removed from stand_out (job_state wins —
    it's the single source of truth for those two, per the founder's de-duplication request).
    "React" and "React Native" are different skills and both stay.
    """
    seen_lower = {
        item.strip().lower()
        for field in ("required_skills", "preferred_skills")
        for item in (job_state.get(field) or [])
        if isinstance(item, str)
    }
    result = dict(jd)
    stand_out = result.get("stand_out") or []
    if isinstance(stand_out, list):
        result["stand_out"] = [
            item for item in stand_out
            if not (isinstance(item, str) and item.strip().lower() in seen_lower)
        ]
    return result


def _apply_proofread_corrections(job_state: dict, corrections: dict) -> dict:
    """Applies generate_jd's PROOFREAD MIRROR output (spelling/grammar-corrected versions of
    required_skills/preferred_skills/responsibilities) back onto job_state — the single source of
    truth for these three, never duplicated onto the JD document itself (see JobDescriptionDraft's
    comment). Deterministically guarded, not prompt-trusted: a field is only accepted if the
    correction has the EXACT SAME NUMBER of items as job_state's own current list — a length
    mismatch means the model added, dropped, split, or merged an item despite being told not to,
    so that field is left completely untouched rather than risk silently losing or duplicating
    content. An empty original list has nothing to proofread, so it's always left alone too.
    """
    result = dict(job_state)
    for field in ("required_skills", "preferred_skills", "responsibilities"):
        original = job_state.get(field) or []
        corrected = corrections.get(field) or []
        if original and len(corrected) == len(original):
            result[field] = [_strip_markdown(v) if isinstance(v, str) else v for v in corrected]
    return result


# These JD fields are supposed to mirror job_state exactly — logistics/identity facts, not prose
# the model should be paraphrasing (unlike job_title, which is DELIBERATELY left to the model's
# own polish — see JD_GENERATION_PROMPT_TEMPLATE's headline guidance: "Video Editor" becoming
# "Cinematic Video Editor for YouTube Channel (Long-form + Shorts)" is the intended, requested
# behavior, not a fabrication to guard against). Same "don't trust prompt compliance for a
# fact-fidelity guarantee" principle as _dedupe_stand_out above, just scoped to the fields that
# actually need it: force these back to job_state's own values deterministically after every
# generation/refinement instead of hoping the model leaves them untouched.
_JD_IDENTITY_FIELDS_FROM_JOB_STATE = ("job_category", "employment_type", "location", "work_mode", "deadline")


def _apply_job_state_identity_fields(jd: dict, job_state: dict) -> dict:
    result = dict(jd)
    for field in _JD_IDENTITY_FIELDS_FROM_JOB_STATE:
        result[field] = job_state.get(field) or None
    return result


def _job_state_from_record(record: dict) -> dict:
    return {
        "job_title": record.get("job_title"),
        "job_category": record.get("job_category"),
        "experience": record.get("experience"),
        "location": record.get("location"),
        "work_mode": record.get("work_mode"),
        "employment_type": record.get("employment_type"),
        "required_skills": record.get("required_skills") or [],
        "preferred_skills": record.get("preferred_skills") or [],
        "education": record.get("education"),
        "responsibilities": record.get("responsibilities") or [],
        "salary": record.get("salary"),
        "deadline": record.get("deadline"),
        "additional_information": record.get("additional_information"),
        "company_overrides": record.get("company_overrides") or {},
    }


def load_context(state: GraphState, config: RunnableConfig) -> dict:
    # Authorization already happened in routes/chat.py before this graph run was ever
    # started — this node just trusts the company_id the route handed it, same as every
    # other node here does no authorization of its own.
    updates: dict = {}
    # Always fetch fresh, never "hydrate once and cache in the checkpoint" the way job_id/
    # recruiter_name below do — get_company_profile_by_id returns a dict with every column
    # present (just None-valued when unset), so it's only ever EMPTY on a company that's never
    # been touched at all; the moment company_name exists the dict itself is truthy, so the old
    # "if not state.get(...)" guard would have frozen the FIRST turn's snapshot for the entire
    # conversation. Verified live: a recruiter who fills in their company profile mid-conversation
    # (or just after starting a job draft) kept seeing the COMPANY CONTEXT CHECK ask for context
    # that was already saved — and, more importantly, JD generation would have used the same stale
    # snapshot. A single indexed row lookup per turn is negligible next to the LLM call already
    # happening on every turn, so there's no real cost to always reading it fresh.
    company_id = config["configurable"].get("company_id")
    updates["company_profile"] = get_company_profile_by_id(company_id) or {} if company_id else {}

    if not state.get("recruiter_name"):
        user_id = config["configurable"].get("user_id")
        user = get_user_by_id(user_id) if user_id else None
        if user and user.get("name"):
            updates["recruiter_name"] = user["name"].strip().split(" ")[0]

    if state.get("job_id") is None:
        session_id = config["configurable"]["thread_id"]
        record = get_job_by_session_id(session_id)
        if record and record.get("status") == "published":
            updates["job_id"] = record.get("job_id")
            if not state.get("job_state"):
                # No checkpoint for this thread yet (e.g. a demo job seeded directly into the DB,
                # never run through the graph) — hydrate everything from the DB row so it's
                # immediately editable, same as a job that was actually published via chat.
                updates["job_state"] = _job_state_from_record(record)
                updates["phase"] = "published"
                updates["jd_stale"] = False
                selected_version = record.get("selected_version") or "1"
                selected_jd = record.get("selected_jd")
                if selected_jd:
                    updates["jd_versions"] = {selected_version: selected_jd}
                    updates["selected_version"] = selected_version

    return updates


def _trailing_draft_reminder(job_state: dict) -> str:
    """Appended to the end of the CURRENT human message before every analyze_turn call, not just
    stated once near the top of the (long) system prompt — verified live this matters: the same
    "this is the source of truth" instruction placed only at the top wasn't reliable against the
    model favoring an earlier, now-stale value mentioned in the conversation history instead (a
    recency-bias problem, not a data-freshness one — job_state itself was always already correct).
    This duplicates the actual CURRENT values right next to where generation happens, rather than
    just pointing back at an earlier block the model has to remember to trust. Covers every field
    generically (not just the handful of specific phrasings _detect_field_value_query recognizes),
    since real recruiter phrasing varies too much to fully enumerate.
    """
    visible = {
        k: v
        for k, v in (job_state or {}).items()
        if k not in ("company_overrides", "custom_questions") and v
    }
    if not visible:
        return ""
    return (
        "\n\n[Background note, not part of the recruiter's message above: the draft's state "
        "immediately BEFORE this message is given below. If the recruiter just stated something new "
        "in their message, that new statement takes priority as always. But whenever you state or "
        "confirm what a field CURRENTLY is — including if they're asking a direct question about it "
        "— use these values, not anything said earlier in this conversation, since the recruiter can "
        f"edit the draft panel directly, outside chat, at any time: {json.dumps(visible)}]"
    )


def analyze_turn(state: GraphState) -> dict:
    system_prompt = build_system_prompt(
        company_profile=state.get("company_profile", {}),
        job_state=state.get("job_state", {}),
        phase=state.get("phase", "collecting"),
        missing_essential=state.get("missing_essential", []),
        jd_exists=bool(state.get("jd_versions")),
        jd_stale=bool(state.get("jd_stale", False)),
        recruiter_name=state.get("recruiter_name"),
    )
    raw_messages = list(state["messages"])
    reminder = _trailing_draft_reminder(state.get("job_state") or {})
    if reminder and raw_messages and isinstance(raw_messages[-1], HumanMessage):
        raw_messages[-1] = HumanMessage(content=(raw_messages[-1].content or "") + reminder)
    messages = [SystemMessage(content=system_prompt), *raw_messages]

    try:
        analysis = call_structured(TurnAnalysis, messages, retries=2)
    except RuntimeError:
        # Configuration error (e.g. missing MISTRAL_API_KEY) — not a parsing failure,
        # let it propagate so the API layer can return a clear 503 instead of a
        # misleading "please rephrase" message.
        raise
    except Exception:
        logger.exception("analyze_turn: structured output failed after retry")
        fallback = _fallback_turn_analysis(state)
        return {
            "pending_analysis": fallback.model_dump(mode="json"),
            "messages": [AIMessage(content=fallback.response)],
        }

    updates: dict = {}
    previous_asking_about_field = state.get("asking_about_field")

    last_human_text = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            last_human_text = (m.content or "").strip()
            break

    # The model's own asking_about_field can be stale on a "combined" turn (acknowledge one answer
    # + ask the next question, all in the same response) — verified live: after answering the
    # platforms question with "Discord, Twitch", the model's response correctly asked about
    # location next, but still reported asking_about_field="platforms" (a field this SAME turn's
    # own list_operations just resolved) — which then fed the WRONG chip set (platforms' own chips,
    # or a stale Skip fallback) for a question that was actually about something else entirely.
    # Deterministic correction, not a prompt hope (same lesson as everywhere else in this file): if
    # the model's own asking_about_field names a field that was EMPTY before this turn but is now
    # resolved by this turn's own field_updates/list_operations, it can't genuinely still be asking
    # about that field — recompute the real next unresolved checklist item instead. Scoped to
    # "resolved by THIS turn" specifically (not just "already has a value"), so a legitimate re-ask
    # of an earlier-set field (e.g. confirming a changed salary) is never mistaken for staleness.
    if analysis.asking_about_field and not (state.get("job_state") or {}).get(analysis.asking_about_field):
        prospective_job_state_for_stale_check = apply_field_changes(
            state.get("job_state") or {},
            analysis.field_updates,
            [op.model_dump() for op in analysis.list_operations],
        )
        if prospective_job_state_for_stale_check.get(analysis.asking_about_field):
            skipped_for_stale_check = set(state.get("skipped_checklist_fields") or [])
            corrected_next_field = _next_checklist_prompt(prospective_job_state_for_stale_check, skipped_for_stale_check)
            if corrected_next_field:
                _, corrected_field, corrected_chips = corrected_next_field
                analysis = analysis.model_copy(
                    update={
                        "asking_about_field": corrected_field,
                        "suggested_options": corrected_chips,
                        "options_multi_select": corrected_field in _MULTI_SELECT_CHECKLIST_FIELDS,
                    }
                )

    # Catch an internally-contradictory experience statement before anything else — "fresher"
    # (implying little-to-no prior experience) stated alongside a genuine multi-year figure in the
    # SAME message (e.g. "hire a fresher with 5 years of exp") is logically inconsistent. Verified
    # live: the model just accepted this as-is, storing 5 years without ever noticing the conflict.
    # Runs on EVERY turn (not gated to "was the last question about experience") since this can — and
    # in the reported case did — show up in the very first message, before experience was ever asked
    # about at all.
    implausible_value_caught = False
    fresher_contradiction = _find_fresher_experience_contradiction(last_human_text)
    if fresher_contradiction:
        implausible_value_caught = True
        fresher_phrase, experience_phrase = fresher_contradiction
        new_field_updates = dict(analysis.field_updates or {})
        new_field_updates.pop("experience", None)
        analysis = analysis.model_copy(
            update={
                "field_updates": new_field_updates,
                "response": (
                    f'Quick check — you mentioned "{fresher_phrase}" but also "{experience_phrase}", '
                    "which don't quite line up (a fresher usually means little to no prior "
                    "experience). Which did you mean for this role?"
                ),
                "asking_about_field": "experience",
                "suggested_options": _DEFAULT_OPTIONS_BY_FIELD["experience"],
                "options_multi_select": False,
                "enough_information": False,
            }
        )

    # Catch an obvious fat-finger/misread before it's silently stored — e.g. a recruiter meaning
    # "2-3 years" landing as "223" years of experience. Checked against the RAW recruiter reply,
    # not analysis.field_updates: verified live that the model sometimes "helpfully" normalizes an
    # implausible number into something plausible-looking on its own ("223" -> "2-3 years") and
    # just moves on to the next question — which hides the exact problem instead of solving it,
    # since the recruiter never gets a chance to confirm what they actually meant. Only fires when
    # the PREVIOUS turn was genuinely asking about experience, so an unrelated number elsewhere in
    # a message (a salary figure, a year, anything) is never second-guessed. A generous ceiling,
    # not a strict range check, so real answers like "10+ years" or "15 years" for a senior role
    # are never flagged. Must happen here, in analyze_turn, not apply_updates — the AIMessage this
    # turn's response becomes is already committed to the transcript by the time apply_updates
    # runs, so this is the only point where the reply text itself can still be changed to ask for
    # clarification instead.
    if not implausible_value_caught and previous_asking_about_field == "experience":
        raw_numbers = re.findall(r"\d+", last_human_text)
        if raw_numbers and max(int(n) for n in raw_numbers) > _MAX_PLAUSIBLE_EXPERIENCE_YEARS:
            implausible_value_caught = True
            new_field_updates = dict(analysis.field_updates or {})
            new_field_updates.pop("experience", None)
            analysis = analysis.model_copy(
                update={
                    "field_updates": new_field_updates,
                    "response": (
                        f'"{last_human_text}" doesn\'t look like a realistic years-of-experience value — '
                        "could you double check that? For example, 2-3 years, 5 years, or 10+ years."
                    ),
                    "asking_about_field": "experience",
                    "suggested_options": _DEFAULT_OPTIONS_BY_FIELD["experience"],
                    "options_multi_select": False,
                    "enough_information": False,
                }
            )

    # A standard-checklist (or skills-family) question that gets asked twice in a row without
    # resolving — the model re-asks near-verbatim instead of recognizing a decline like "I don't
    # want to mention salary", or two different unclear replies land in a row — must never become a
    # 3rd ask. Verified live: the model can loop on an identical question after a free-text decline
    # it failed to recognize. Detect via TEXT similarity to the bot's own last message (a genuinely
    # new question is never near-identical to the one right before it) rather than relying on the
    # model to have correctly re-set asking_about_field — same "cap repeated asks deterministically,
    # don't trust the model to stop on its own" philosophy as the skills-followup redirect below. Skipped
    # entirely when the implausible-value check above just fired: THAT block deliberately re-asks
    # the same field on purpose (to confirm a suspicious value), which looks identical to a stall
    # from here — without this guard the two overrides fight each other and the implausible-value
    # catch gets silently undone the instant it fires (verified live).
    last_ai_text = None
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage):
            last_ai_text = (m.content or "").strip()
            break
    response_text_now = (analysis.response or "").strip()
    repeated_same_field = bool(
        analysis.asking_about_field
        and previous_asking_about_field
        and analysis.asking_about_field == previous_asking_about_field
    )
    repeated_verbatim = bool(last_ai_text) and bool(response_text_now) and response_text_now == last_ai_text
    stalled_field = previous_asking_about_field if (repeated_same_field or repeated_verbatim) else None

    if stalled_field and stalled_field in OPTIONAL_SKIPPABLE_FIELDS and not implausible_value_caught:
        prospective_job_state = apply_field_changes(
            state.get("job_state") or {},
            analysis.field_updates,
            [op.model_dump() for op in analysis.list_operations],
        )
        floor_would_break = stalled_field in ("required_skills", "responsibilities") and not _skills_floor_met(
            prospective_job_state
        )
        if not prospective_job_state.get(stalled_field) and not floor_would_break:
            skipped = set(state.get("skipped_checklist_fields") or [])
            skipped.add(stalled_field)
            next_field = _next_checklist_prompt(prospective_job_state, skipped)
            if next_field:
                question, field, chips = next_field
                analysis = analysis.model_copy(
                    update={
                        "response": f"No worries, we'll leave that out. {question}",
                        "asking_about_field": field,
                        "suggested_options": chips,
                        "options_multi_select": field in _MULTI_SELECT_CHECKLIST_FIELDS,
                        "enough_information": False,
                    }
                )
            else:
                analysis = analysis.model_copy(
                    update={
                        "response": _AUTO_GENERATE_RESPONSE,
                        "asking_about_field": None,
                        "suggested_options": [],
                        "options_multi_select": False,
                        "enough_information": True,
                    }
                )
            updates["skipped_checklist_fields"] = list(skipped)

    # Skills/responsibilities are auto-generated and NEVER asked about at all — not even the single
    # "Would you like to add any additional skills?" follow-up this used to allow once. Verified
    # live that prompt compliance alone wasn't reliable here either (same lesson as everywhere else
    # in this file): if the model asks that follow-up anyway, redirect it immediately, on the very
    # first occurrence, straight to the real next checklist item (or auto-generate if nothing's
    # left) — never let it reach the recruiter, not even once.
    #
    # Must match the SPECIFIC "want to add more?" phrasing, not just any sentence that happens to
    # mention "skill"/"respons" near a question mark — a turn that's actually just announcing what
    # was generated ("I've set Premiere Pro and After Effects as required skills... Which city will
    # this be based in?") also contains both, and this check must NOT be what catches that turn —
    # it's handled separately, more reliably, by the list_operations-based override below (which
    # replaces the whole response outright once the recruiter no longer wants that announcement
    # narrated at all — see the block right after this one).
    response_lower = (analysis.response or "").lower()
    is_skills_followup = "?" in response_lower and any(
        phrase in response_lower
        for phrase in (
            "additional skill",
            "any other skill",
            "more skill",
            "further skill",
            "additional responsibilit",
            "any other responsibilit",
            "more responsibilit",
            "add any skill",
            "add more skill",
        )
    )
    if is_skills_followup:
        # Merge in this turn's own field_updates first so a message that named a field (e.g. "Add
        # Python, and it's hybrid") isn't immediately re-asked about.
        prospective_job_state = apply_field_changes(
            state.get("job_state") or {},
            analysis.field_updates,
            [op.model_dump() for op in analysis.list_operations],
        )
        # Prefer updates["skipped_checklist_fields"] over state's own copy — the stalled-field
        # override just above can have already force-skipped a field THIS SAME TURN (e.g. the
        # recruiter's decline was itself the skills-family reply), and state is the turn's
        # original snapshot, never mutated in place; reading it alone would silently forget
        # that skip and re-ask about the very field just resolved.
        skipped = set(updates.get("skipped_checklist_fields", state.get("skipped_checklist_fields") or []))
        next_field = _next_checklist_prompt(prospective_job_state, skipped)
        if next_field:
            question, field, chips = next_field
            analysis = analysis.model_copy(
                update={
                    "response": f"Got it, noted! {question}",
                    "asking_about_field": field,
                    "suggested_options": chips,
                    "options_multi_select": field in _MULTI_SELECT_CHECKLIST_FIELDS,
                }
            )
        else:
            analysis = analysis.model_copy(
                update={
                    "response": _AUTO_GENERATE_RESPONSE,
                    "asking_about_field": None,
                    "suggested_options": [],
                    "options_multi_select": False,
                    "enough_information": True,
                }
            )

    # The recruiter doesn't want a chat announcement listing what required_skills/preferred_skills/
    # responsibilities were auto-generated (explicit founder feedback: "it should not display what
    # it does" — the panel already shows the generated list in full, chat should just move straight
    # to the next question). Detected structurally, via list_operations touching these three fields
    # — NOT by parsing/stripping the model's own response text, which is fragile and was explicitly
    # hardened in the other direction last time (see is_skills_followup above: a bare keyword+"?"
    # match used to swallow this exact combined "here's what I set... next question?" message
    # entirely). The model reliably narrates the full list whenever this happens, so once it's
    # unwanted, the response needs to be replaced outright with the real next question, not edited.
    # Gated to before any JD exists: once collection is done, a later "add Python" edit is a normal
    # CORRECT_INFORMATION turn (marks the JD stale, handled elsewhere), not this generation moment.
    jd_already_exists = bool(state.get("jd_versions"))
    _skills_family_fields = {"required_skills", "preferred_skills", "responsibilities"}
    if not jd_already_exists and any(op.field in _skills_family_fields for op in analysis.list_operations):
        prospective_job_state = apply_field_changes(
            state.get("job_state") or {},
            analysis.field_updates,
            [op.model_dump() for op in analysis.list_operations],
        )
        if _skills_floor_met(prospective_job_state):
            skipped = set(updates.get("skipped_checklist_fields", state.get("skipped_checklist_fields") or []))
            next_field = _next_checklist_prompt(prospective_job_state, skipped)
            if next_field:
                question, field, chips = next_field
                analysis = analysis.model_copy(
                    update={
                        "response": f"Got it, noted! {question}",
                        "asking_about_field": field,
                        "suggested_options": chips,
                        "options_multi_select": field in _MULTI_SELECT_CHECKLIST_FIELDS,
                    }
                )
            else:
                analysis = analysis.model_copy(
                    update={
                        "response": _AUTO_GENERATE_RESPONSE,
                        "asking_about_field": None,
                        "suggested_options": [],
                        "options_multi_select": False,
                        "enough_information": True,
                    }
                )

    # See _detect_field_value_query above — deterministic override for "what's the current X?"
    # style questions, answered straight from job_state (the single source of truth) rather than
    # trusting the model's own composed response, which verified live can echo an earlier, now-
    # stale value from the conversation transcript instead. Only fires when this turn's own
    # field_updates didn't just set a new value for that field — a genuine "make it $80k" statement
    # is left untouched, that's correctly handled by the model's own extraction already.
    field_query = _detect_field_value_query(last_human_text)
    # platforms is a LIST field (list_operations), not a scalar one (field_updates) — "is THIS
    # turn setting it" needs to check the right place, or a genuine "add Discord too" turn would
    # get its own answer overwritten with a stale pre-turn snapshot.
    field_query_already_set_this_turn = (
        any(op.field == field_query for op in analysis.list_operations)
        if field_query == "platforms"
        else bool((analysis.field_updates or {}).get(field_query))
    )
    if field_query and not field_query_already_set_this_turn:
        current_value = (state.get("job_state") or {}).get(field_query)
        label = _FIELD_DISPLAY_LABELS.get(field_query, field_query)
        if field_query == "platforms":
            display_value = ", ".join(current_value) if current_value else None
        else:
            display_value = current_value
        response_text = (
            f"The current {label} is {display_value}."
            if display_value
            else f"No {label} has been set yet — want to add one now?"
        )
        analysis = analysis.model_copy(update={"response": response_text})

    # Location-aware currency hint for the model's OWN first-time salary question — _salary_question
    # (used by the deterministic redirect paths below) only covers those specific override routes,
    # not the far more common case where the model asks its own salary question in its own words on
    # the normal path. Detected structurally: does the QUESTION portion of this turn's response
    # (never an acknowledgment clause mentioning salary in passing — same sentence-isolation as the
    # _KEYWORD_FIELD_HINTS sniffing above) actually pose a salary question, via the same keyword
    # matcher _detect_field_value_query uses above. Only touches the response when a currency is
    # known for the recruiter's own location.
    response_question_sentences = [s for s in re.split(r"(?<=[.!?])\s+", analysis.response or "") if "?" in s]
    response_question_text = " ".join(response_question_sentences) or (analysis.response or "")
    if _FIELD_QUERY_KEYWORDS["salary"].search(response_question_text):
        # Use THIS turn's own field_updates for location too, not just state's pre-turn snapshot —
        # verified live the model routinely combines "thanks for Bangalore" + "what's the salary?"
        # into ONE turn (acknowledge + ask the next single question, same as every other checklist
        # transition), so by the time this runs, location was often JUST set this same turn and
        # hasn't reached state["job_state"] yet (that merge happens later, in apply_updates).
        prospective_location = (analysis.field_updates or {}).get("location") or (state.get("job_state") or {}).get(
            "location"
        )
        currency_symbol = _currency_symbol_for_location(prospective_location)
        response_text_now = analysis.response or ""
        if currency_symbol:
            wrong_symbol_present = any(
                sym in response_text_now for sym in _ALL_CURRENCY_SYMBOLS if sym != currency_symbol
            )
            if wrong_symbol_present:
                # The model's own phrasing baked in a numeric example in the WRONG currency (e.g.
                # "$40k–$60k" for a Delhi NCR posting) — reported live. Converting the figure isn't
                # the fix (a $ amount run through some exchange rate is still a fabricated number),
                # so the whole question is replaced with a clean, currency-correct version instead
                # of leaving a contradictory dollar example sitting next to a "use ₹" hint in the
                # same message.
                analysis = analysis.model_copy(update={"response": _salary_question({"location": prospective_location})})
            elif currency_symbol not in response_text_now:
                analysis = analysis.model_copy(
                    update={"response": f"{response_text_now.rstrip()} (in {currency_symbol}, based on the location you gave)"}
                )

    # Never trust enough_information=true at face value: cross-check it against the same
    # deterministic checklist ready_to_generate() relies on. Verified live that the model reaches
    # this point far more often than prompt compliance alone would suggest — it reliably asks
    # through the essential fields, then jumps straight to "Everything's captured" while a real
    # field (or the skills follow-up) is still unresolved. Also catches the model spontaneously
    # asking its own wrap-up-shaped question ("anything else you'd like to add?") — that's no
    # longer a real step in the flow at all (see route_after_apply: generation now fires
    # automatically the instant the checklist resolves, no confirmation ceremony), so any such
    # phrasing gets swept into the same real-state check below rather than shown to the recruiter.
    response_lower_now = (analysis.response or "").lower()
    looks_like_closing_check = "?" in response_lower_now and (
        "anything else" in response_lower_now or "add more" in response_lower_now or "add anything" in response_lower_now
        or ("generate" in response_lower_now and "ready" in response_lower_now)
    )
    # A DECLARATIVE "you're ready" announcement (no question mark at all) is just as real a miss —
    # verified live: "Everything's captured for this role, Alex — click 'Generate Full Description'
    # in the panel on the right whenever you're ready!" sailed straight through with
    # enough_information left false, because it never posed a question and the model's own
    # enough_information flag didn't match what the text was actually saying. Catching this text
    # pattern regardless of punctuation routes it into the same real-state check below instead of
    # passing the model's inconsistent statement straight through.
    looks_like_ready_statement = any(
        phrase in response_lower_now
        for phrase in (
            "everything's captured",
            "everything is captured",
            "generate full description",
            "generate the full description",
            "generate the job description",
            "ready to generate",
            "ready to draft",
            "click generate",
            "click 'generate",
            'click "generate',
        )
    )
    looks_like_closing_check = looks_like_closing_check or looks_like_ready_statement
    # This entire "checklist just finished, about to auto-generate" mechanism only makes sense
    # before any JD exists yet. Verified live: once a draft was already generated and the recruiter
    # approved it ("Looks good!"), an earlier version of this override still fired and told them
    # "Everything's captured — ready to generate?" — confusing and backward, since a description
    # already exists; regenerating/refining/publishing are the only real next steps at that point,
    # and the existing REQUEST_REFINEMENT/CONFIRM_PUBLISH prompt guidance already covers them
    # correctly on its own — this block must get out of the way entirely once jd_versions exist.
    # (jd_already_exists itself is computed once, above, and reused by the skills-announcement
    # suppression block right before this one.)
    if not jd_already_exists and (analysis.enough_information or looks_like_closing_check):
        prospective_job_state = apply_field_changes(
            state.get("job_state") or {},
            analysis.field_updates,
            [op.model_dump() for op in analysis.list_operations],
        )
        if _skills_floor_met(prospective_job_state):
            # See the skills-loop-cap override above for why this prefers updates over state.
            skipped = set(updates.get("skipped_checklist_fields", state.get("skipped_checklist_fields") or []))
            next_field = _next_checklist_prompt(prospective_job_state, skipped)
            if next_field:
                # Model thinks it's done (or asked its own wrap-up question) but a real field is
                # still unresolved — same fix pattern as the skills-loop-cap override above.
                question, field, chips = next_field
                analysis = analysis.model_copy(
                    update={
                        "response": f"Got it, noted! {question}",
                        "asking_about_field": field,
                        "suggested_options": chips,
                        "options_multi_select": field in _MULTI_SELECT_CHECKLIST_FIELDS,
                        "enough_information": False,
                    }
                )
            else:
                # Checklist is genuinely complete — no confirmation ceremony, no chip: announce it
                # and let route_after_apply route straight into generate_jd this same turn.
                analysis = analysis.model_copy(
                    update={
                        "response": _AUTO_GENERATE_RESPONSE,
                        "asking_about_field": None,
                        "suggested_options": [],
                        "options_multi_select": False,
                        "enough_information": True,
                    }
                )

    clean_response = _strip_markdown(analysis.response)
    dumped = analysis.model_dump(mode="json")
    dumped["response"] = clean_response
    updates.update(
        {
            "pending_analysis": dumped,
            "messages": [AIMessage(content=clean_response)],
        }
    )
    return updates


def _apply_list_operation(current: list[str], operation: str, values: list[str]) -> list[str]:
    if operation == "ADD":
        existing_lower = {v.lower() for v in current}
        result = list(current)
        for value in values:
            if value.lower() not in existing_lower:
                result.append(value)
                existing_lower.add(value.lower())
        return result
    if operation == "REMOVE":
        remove_lower = {v.lower() for v in values}
        return [v for v in current if v.lower() not in remove_lower]
    if operation == "REPLACE":
        seen: set[str] = set()
        result = []
        for value in values:
            if value.lower() not in seen:
                result.append(value)
                seen.add(value.lower())
        return result
    return current


_BLOCK_ALL_VALUES = {i.value for i in INTENTS_BLOCK_ALL_JOBSTATE_CHANGES}
_BLOCK_LIST_AND_OVERRIDE_VALUES = {i.value for i in INTENTS_BLOCK_LIST_AND_OVERRIDE_CHANGES}

# Deterministic fallback chips for the standard checklist fields that have one obvious, safe
# default set of answers — used only when the model asks about the field but returns no options
# of its own (see the suggested_options backfill in apply_updates below).
_DEFAULT_OPTIONS_BY_FIELD = {
    "work_mode": ["Remote", "Hybrid", "Onsite"],
    "employment_type": ["Full-time", "Part-time", "Contract", "Internship"],
    "experience": ["Entry-Level", "Mid-Level", "Senior-Level"],
    "education": ["Bachelor's degree", "Master's degree", "Not required"],
}

# A generous ceiling, not a strict range — only meant to catch an obvious fat-finger/misread (a
# recruiter meaning "2-3 years" landing as "223") before it's silently stored as job_state. Real
# postings asking for up to a few decades of experience should never be second-guessed.
_MAX_PLAUSIBLE_EXPERIENCE_YEARS = 50

# Deterministic safety net for a handful of extremely common, near-unambiguous phrasings —
# verified live that the model sometimes verbally acknowledges a fact ("A Python developer with 4
# years of experience, fully remote — got it!") without actually putting it in field_updates, so
# job_state stays empty and the checklist re-asks a question the recruiter already answered in
# their very first message. Intentionally narrow (never a general-purpose extractor, and never
# overwrites a field that already has a value) to keep false positives low.
_EXPERIENCE_RANGE_RE = re.compile(r"\b(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\+?\s*(?:yrs?|years?)\b", re.IGNORECASE)
_EXPERIENCE_SINGLE_RE = re.compile(r"\b(\d{1,2})(\+?)\s*(?:yrs?|years?)\b", re.IGNORECASE)
_WORK_MODE_KEYWORDS = (
    (re.compile(r"\bremote(?:ly)?\b", re.IGNORECASE), "Remote"),
    (re.compile(r"\bhybrid\b", re.IGNORECASE), "Hybrid"),
    (re.compile(r"\b(?:onsite|on-site|in[- ]office)\b", re.IGNORECASE), "Onsite"),
)
_EMPLOYMENT_TYPE_KEYWORDS = (
    (re.compile(r"\bfull[- ]time\b", re.IGNORECASE), "Full-time"),
    (re.compile(r"\bpart[- ]time\b", re.IGNORECASE), "Part-time"),
    (re.compile(r"\bcontract(?:or)?\b", re.IGNORECASE), "Contract"),
    (re.compile(r"\binternship\b", re.IGNORECASE), "Internship"),
)


def _years_to_experience_band(years: int) -> str:
    """Maps a raw years figure to the app's qualitative experience vocabulary (Entry/Mid/Senior),
    keeping job_state.experience uniformly in that vocabulary even when a recruiter free-types a
    year count instead of using the dropdown default/chip.
    """
    if years <= 2:
        return "Entry-Level"
    if years <= 6:
        return "Mid-Level"
    return "Senior-Level"


def _extract_fact_backstop(text: str) -> dict:
    """Best-effort, deterministic extraction of experience/work_mode/employment_type from raw
    recruiter text — called from apply_updates to fill in whatever the model's own field_updates
    missed. Only ever used to fill a field that's CURRENTLY EMPTY (see the caller), never to
    overwrite the model's own (or an earlier) value, so a false-positive match on unrelated text
    later in the conversation can only ever matter if that field was somehow still unset.
    """
    if not text:
        return {}
    found: dict = {}
    range_match = _EXPERIENCE_RANGE_RE.search(text)
    if range_match:
        found["experience"] = _years_to_experience_band(int(range_match.group(1)))
    else:
        single_match = _EXPERIENCE_SINGLE_RE.search(text)
        if single_match:
            found["experience"] = _years_to_experience_band(int(single_match.group(1)))
    for pattern, label in _WORK_MODE_KEYWORDS:
        if pattern.search(text):
            found["work_mode"] = label
            break
    for pattern, label in _EMPLOYMENT_TYPE_KEYWORDS:
        if pattern.search(text):
            found["employment_type"] = label
            break
    return found


def _extract_platforms_backstop(text: str) -> list[str]:
    """Deterministic parse of the recruiter's raw reply against the known platform chip labels
    (_PLATFORM_OPTIONS) — the platforms question is multi-select, and the chip UI sends a plain
    comma-separated reply (e.g. "Facebook, YouTube"), so this just needs to recognize which of the
    known names appear, not parse arbitrary free text. See the gated caller in apply_updates.
    """
    if not text:
        return []
    return [p for p in _PLATFORM_OPTIONS if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE)]


# The recruiter asking what a field is CURRENTLY set to ("what's the salary again?", "what
# location did I put?") — verified live that prompt instructions alone aren't reliable here: even
# with the system prompt's job_state_json showing the correct, up-to-date value, the model kept
# answering with an earlier, now-superseded value from the conversation transcript instead (a
# stronger "this block is the single source of truth" instruction in the prompt did not fix it in
# testing). Deterministic override, not a prompt hope — same reasoning as _extract_fact_backstop
# above. Narrow and keyword-based (real free text has too much variety to parse reliably); false
# positives are cheap here (worst case: re-stating accurate current info the recruiter wasn't quite
# asking for), so this errs toward catching more phrasings rather than being maximally precise.
_FIELD_QUERY_KEYWORDS = {
    "salary": re.compile(r"\bsalary\b", re.IGNORECASE),
    "location": re.compile(r"\blocation\b", re.IGNORECASE),
    "job_title": re.compile(r"\b(?:job\s+)?title\b", re.IGNORECASE),
    "work_mode": re.compile(r"\bwork\s*mode\b", re.IGNORECASE),
    "employment_type": re.compile(r"\bemployment\s*type\b", re.IGNORECASE),
    "experience": re.compile(r"\bexperience\b", re.IGNORECASE),
    "deadline": re.compile(r"\bdeadline\b", re.IGNORECASE),
    "platforms": re.compile(r"\bplatforms?\b", re.IGNORECASE),
}
_FIELD_QUERY_SIGNAL_RE = re.compile(
    r"\bwhat(?:'s|s)?\b|\bremind me\b|\btell me\b|\bcurrent(?:ly)?\b|\bagain\b|\bwhich\b", re.IGNORECASE
)
_FIELD_DISPLAY_LABELS = {
    "salary": "salary",
    "location": "location",
    "job_title": "job title",
    "work_mode": "work mode",
    "employment_type": "employment type",
    "experience": "experience level",
    "deadline": "application deadline",
    "platforms": "platform(s)",
}


def _detect_field_value_query(text: str) -> str | None:
    """Returns the job_state field name the recruiter appears to be asking the CURRENT value of,
    or None. Requires both a question-ish signal word (what/remind me/tell me/current/again) AND a
    recognized field keyword somewhere in the same message — a bare mention of "salary" while
    actually STATING a new figure ("the salary should be $80k") has no signal word and is correctly
    left alone (that case is already handled by the model's own field_updates extraction).
    """
    if not text or not _FIELD_QUERY_SIGNAL_RE.search(text):
        return None
    for field, pattern in _FIELD_QUERY_KEYWORDS.items():
        if pattern.search(text):
            return field
    return None


# "Fresher" (and equivalents) means little-to-no prior professional experience — a message that
# also states a real years-of-experience figure alongside it is self-contradictory (e.g. "hire a
# fresher with 5 years of experience"). 0-1 years is still consistent with "fresher"; 2+ is not.
_FRESHER_RE = re.compile(r"\bfresher(?:s)?\b|\bfresh\s+graduate\b|\bentry[- ]level\b|\bno\s+(?:prior\s+)?experience\b", re.IGNORECASE)
_FRESHER_CONTRADICTION_MIN_YEARS = 2


def _find_fresher_experience_contradiction(text: str) -> tuple[str, str] | None:
    """Returns (fresher_phrase, experience_phrase) if the RAW recruiter text states both "fresher"
    (or an equivalent) and a genuine multi-year experience figure in the same message — an
    internal contradiction worth confirming rather than silently picking one side (verified live:
    the model just accepted "fresher with 5 years of exp" as-is without noticing the conflict).
    Returns None when there's no fresher mention, no experience figure, or the figure is small
    enough to still be consistent with "fresher" (0-1 years).
    """
    if not text:
        return None
    fresher_match = _FRESHER_RE.search(text)
    if not fresher_match:
        return None
    range_match = _EXPERIENCE_RANGE_RE.search(text)
    if range_match:
        if int(range_match.group(1)) >= _FRESHER_CONTRADICTION_MIN_YEARS:
            return fresher_match.group(0), range_match.group(0)
        return None
    single_match = _EXPERIENCE_SINGLE_RE.search(text)
    if single_match and int(single_match.group(1)) >= _FRESHER_CONTRADICTION_MIN_YEARS:
        return fresher_match.group(0), single_match.group(0)
    return None


# Shrunk to just what's still actually ASKED, per an explicit founder decision (the general flow
# was asking too many questions): required_skills/responsibilities stay only as a hard-floor safety
# net (see _apply_default_field_values below for why they're not proactively interrogated either —
# the model is expected to auto-generate them from the role and never ask about them at all), then
# location and salary are the only genuinely-still-asked fields. preferred_skills/experience/
# employment_type/education are no longer checklist items at all — preferred_skills is auto-generated
# the same way as required_skills, and experience/employment_type/education are deterministically
# defaulted (see _apply_default_field_values), never asked. work_mode is ALSO never proactively asked
# (same founder decision) but, unlike experience/employment_type/education, has no safe universal
# default — Remote/Hybrid/Onsite is a material, candidate-facing fact that genuinely varies per
# role, so guessing one would risk actively misleading a candidate rather than just being generic.
# It simply stays unset unless the recruiter volunteers it (still captured via normal field_updates
# extraction or the fact-backstop below if they do) — the JD generation prompt already omits any
# meta line it doesn't have real data for rather than inventing one. platforms is likewise no
# longer a checklist item — once the guided intake collapsed to a single-API-call local flow, it
# went back to being a purely optional field (see sufficiency.py), set only via the draft panel's
# checkbox dropdown or if the recruiter mentions it unprompted in chat (_extract_platforms_backstop
# still handles that case; the option list/multi-select machinery below is left in place for it).
_CHECKLIST_ORDER = [
    "required_skills",
    "responsibilities",
    "location",
    "salary",
]
_CHECKLIST_QUESTIONS = {
    "required_skills": "What are the required skills a candidate should have for this role?",
    "responsibilities": "What will this person be responsible for day-to-day?",
    "platforms": "Which platform(s) are you hiring for?",
    "location": 'Which city or region will this role be based in? You can also say "Worldwide" if it\'s fully remote.',
    "salary": "What's the salary range for this role, if you'd like to share one?",
}

# The recruiter can pick more than one — see PLATFORM_MULTI_SELECT_FIELDS below and
# options_multi_select wiring in _next_checklist_prompt's callers.
_PLATFORM_OPTIONS = ["Facebook", "YouTube", "Instagram", "TikTok", "Vimeo", "Twitch", "Discord"]

# Checklist fields whose chip question is multi-select (the recruiter can tap several before
# sending) rather than single-select (sends immediately on the first tap) — read by every
# _next_checklist_prompt caller instead of each one hardcoding options_multi_select=False.
_MULTI_SELECT_CHECKLIST_FIELDS = {"platforms"}
# required_skills/responsibilities stay empty, not ["Skip"]: both are still genuinely skippable
# (see OPTIONAL_SKIPPABLE_FIELDS) whenever this fallback path does get asked, so a chip whose ONLY
# content is a second, differently-styled "Skip" would duplicate the dedicated "Skip this" button
# that already renders alongside it (reported live as a confusing double affordance).
#
# salary is different: it's mandatory now (no Skip button renders for it at all — see
# OPTIONAL_SKIPPABLE_FIELDS), so leaving it with zero chips (as it used to be, back when it WAS
# skippable) left the question with nothing tappable at all, unlike every other question in the
# flow — reported live as "why isn't it suggesting chips" for this exact question. A specific
# numeric range isn't safe to default to (the app spans very different currencies/scales — $/mo
# for one role, ₹ LPA for another), so the one universally-safe, always-valid suggestion is
# "Competitive, negotiable" — a real, complete answer on its own, not a stand-in for skipping.
_CHECKLIST_CHIPS = {
    "required_skills": [],
    "responsibilities": [],
    "salary": ["Competitive, negotiable"],
    "platforms": _PLATFORM_OPTIONS,
}

# location is deliberately NOT a static entry above — reported live as feeling stale (the same
# fixed "Worldwide/New York/London/Bangalore" four every single time). "Worldwide" always leads
# (still the single most common answer for a remote-friendly role), the other three are a fresh
# random sample from this wider, geographically varied pool on every ask — see
# _location_checklist_chips below, the only thing that reads this pool.
_LOCATION_CHIP_POOL = [
    "Bangalore", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Srinagar",
    "New York", "Los Angeles", "London", "Toronto", "Sydney", "Singapore", "Dubai", "Berlin",
]


def _location_checklist_chips() -> list[str]:
    return ["Worldwide", *random.sample(_LOCATION_CHIP_POOL, min(3, len(_LOCATION_CHIP_POOL)))]


def _checklist_chips_for(field: str) -> list[str]:
    """Single access point for a checklist field's chip suggestions — routes "location" through
    the randomized picker above instead of _CHECKLIST_CHIPS, everything else through the static
    dict as before (falling back to no chips for an unrecognized field, same as a plain dict.get).
    """
    if field == "location":
        return _location_checklist_chips()
    return _CHECKLIST_CHIPS.get(field, [])


# Keyword -> currency symbol, checked against the recruiter's own free-text location (asked right
# before salary in _CHECKLIST_ORDER, so it's always already known by the time this is used) — e.g.
# a Bangalore posting should be quoted in ₹, not $. Order matters where terms could otherwise
# collide (none currently do); deliberately keyword-based rather than a fixed city list so any
# India-adjacent phrasing (state names, "India" itself) still resolves correctly, not just the
# exact city examples below.
_CURRENCY_BY_LOCATION_KEYWORDS = [
    (re.compile(r"\b(bangalore|bengaluru|mumbai|delhi|hyderabad|pune|chennai|kolkata|srinagar|"
                r"ahmedabad|jaipur|noida|gurgaon|gurugram|india)\b", re.IGNORECASE), "₹"),
    (re.compile(r"\b(london|manchester|birmingham|edinburgh|glasgow|u\.?k\.?|united kingdom)\b", re.IGNORECASE), "£"),
    (re.compile(r"\b(toronto|vancouver|montreal|canada)\b", re.IGNORECASE), "C$"),
    (re.compile(r"\b(sydney|melbourne|brisbane|australia)\b", re.IGNORECASE), "A$"),
    (re.compile(r"\b(singapore)\b", re.IGNORECASE), "S$"),
    (re.compile(r"\b(dubai|abu dhabi|u\.?a\.?e\.?)\b", re.IGNORECASE), "AED"),
    (re.compile(r"\b(berlin|munich|frankfurt|paris|madrid|amsterdam|germany|france|spain|netherlands)\b", re.IGNORECASE), "€"),
    (re.compile(r"\b(new york|los angeles|san francisco|chicago|austin|seattle|boston|"
                r"u\.?s\.?a?\.?|united states)\b", re.IGNORECASE), "$"),
]


# Every symbol _CURRENCY_BY_LOCATION_KEYWORDS can produce — used to detect a MISMATCHED currency
# (a different symbol than the one the recruiter's own location implies), not just a missing one.
_ALL_CURRENCY_SYMBOLS = list({symbol for _pattern, symbol in _CURRENCY_BY_LOCATION_KEYWORDS})


def _currency_symbol_for_location(location: str | None) -> str | None:
    if not location:
        return None
    for pattern, symbol in _CURRENCY_BY_LOCATION_KEYWORDS:
        if pattern.search(location):
            return symbol
    return None


# Job-title -> required/preferred skills + responsibilities, keyed by keyword pattern, first match
# wins. Used by the single-call job intake endpoint (routes/chat.py POST /chat/intake) to populate
# job_state BEFORE the one-and-only generate_jd call, replacing what used to be an LLM invention
# step (see the old SKILLS AND RESPONSIBILITIES ARE GENERATED... prompt instruction, still used by
# the chat-based analyze_turn path for jobs NOT created via /intake). generate_jd's PROOFREAD
# MIRROR guard (_apply_proofread_corrections above) only ever polishes wording or leaves these
# untouched — it never re-invents them — so pre-populating here is sufficient, no second LLM call
# needed. Ordering matters where keywords could otherwise collide (e.g. "YouTube Channel Manager"
# must be checked before the generic "social media" pattern below it).
_JOB_TITLE_SKILL_PROFILES: list[tuple[re.Pattern, dict]] = [
    (re.compile(r"youtube\b.*\bmanager|\bchannel manager", re.I), {
        "job_category": "Content & Social Media",
        "required_skills": ["YouTube Studio", "SEO & Keyword Research", "Content Scheduling", "Analytics"],
        "preferred_skills": ["Thumbnail Design Basics", "Community Management"],
        "responsibilities": ["Manage upload scheduling and channel organization",
                              "Optimize titles, tags, and descriptions for discoverability",
                              "Track analytics and report on channel growth"],
    }),
    (re.compile(r"\bvideo editor\b", re.I), {
        "job_category": "Video Production",
        "required_skills": ["Adobe Premiere Pro", "Adobe After Effects", "Color Grading"],
        "preferred_skills": ["Motion Graphics", "Sound Design"],
        "responsibilities": ["Edit raw footage into polished videos",
                              "Sync audio to visuals and apply color grading",
                              "Organize and manage media assets"],
    }),
    (re.compile(r"\bthumbnail designer\b", re.I), {
        "job_category": "Graphic Design",
        "required_skills": ["Adobe Photoshop", "Composition & Typography", "Click-Through Optimization"],
        "preferred_skills": ["Illustrator", "A/B Testing Thumbnails"],
        "responsibilities": ["Design eye-catching, on-brand thumbnails for new uploads",
                              "Iterate on designs based on click-through performance",
                              "Maintain a consistent visual identity across a channel"],
    }),
    (re.compile(r"\bvideo produc", re.I), {
        "job_category": "Video Production",
        "required_skills": ["Pre-Production Planning", "On-Set Direction", "Adobe Premiere Pro"],
        "preferred_skills": ["Lighting & Audio Setup", "Budget Management"],
        "responsibilities": ["Plan and oversee video shoots from concept to delivery",
                              "Coordinate talent, crew, and equipment",
                              "Ensure final output meets creative and brand standards"],
    }),
    (re.compile(r"\bmotion graphics\b", re.I), {
        "job_category": "Video Production",
        "required_skills": ["Adobe After Effects", "Cinema 4D", "Animation Principles"],
        "preferred_skills": ["Illustrator", "3D Motion Design"],
        "responsibilities": ["Design and animate motion graphics for video content",
                              "Create title sequences, lower thirds, and visual effects",
                              "Collaborate with editors to integrate graphics seamlessly"],
    }),
    (re.compile(r"\bpodcast editor\b", re.I), {
        "job_category": "Audio Production",
        "required_skills": ["Audio Editing (Audition/Audacity)", "Noise Reduction", "Audio Mixing"],
        "preferred_skills": ["Show Notes Writing", "Video Podcast Editing"],
        "responsibilities": ["Edit raw audio into a polished, publish-ready episode",
                              "Clean up audio quality and balance levels",
                              "Prepare and export episodes for distribution"],
    }),
    (re.compile(r"\bsocial media\b", re.I), {
        "job_category": "Content & Social Media",
        "required_skills": ["Content Calendar Planning", "Platform Analytics", "Copywriting"],
        "preferred_skills": ["Paid Social Ads", "Community Management"],
        "responsibilities": ["Plan and schedule content across social platforms",
                              "Engage with the audience and grow followers",
                              "Track performance metrics and adjust strategy"],
    }),
    (re.compile(r"\bgraphic designer\b", re.I), {
        "job_category": "Graphic Design",
        "required_skills": ["Adobe Photoshop", "Adobe Illustrator", "Typography & Layout"],
        "preferred_skills": ["Figma", "Motion Graphics Basics"],
        "responsibilities": ["Design graphics for digital and/or print use",
                              "Maintain brand consistency across visual assets",
                              "Iterate on designs based on feedback"],
    }),
    (re.compile(r"\bcontent writ(er|ing)\b", re.I), {
        "job_category": "Content & Social Media",
        "required_skills": ["Copywriting", "SEO Writing", "Editing & Proofreading"],
        "preferred_skills": ["Content Strategy", "CMS Experience"],
        "responsibilities": ["Write clear, engaging content for the intended audience",
                              "Research topics and ensure factual accuracy",
                              "Edit and proofread content before publishing"],
    }),
    (re.compile(r"\bcontent editor\b", re.I), {
        "job_category": "Content & Social Media",
        "required_skills": ["Editing & Proofreading", "Style Guide Adherence", "SEO Basics"],
        "preferred_skills": ["CMS Experience", "Content Strategy"],
        "responsibilities": ["Review and edit content for clarity, tone, and accuracy",
                              "Ensure content follows brand and style guidelines",
                              "Coordinate with writers/creators on revisions"],
    }),
    (re.compile(r"\bcontent strategist\b", re.I), {
        "job_category": "Content & Social Media",
        "required_skills": ["Content Planning", "Audience Research", "Analytics"],
        "preferred_skills": ["SEO Strategy", "Cross-Platform Distribution"],
        "responsibilities": ["Develop content strategy aligned with audience/brand goals",
                              "Plan content calendars across channels",
                              "Analyze performance and refine the strategy over time"],
    }),
    (re.compile(r"\bugc\b|user.generated content", re.I), {
        "job_category": "Content & Social Media",
        "required_skills": ["On-Camera Presence", "Basic Video Editing", "Brand Storytelling"],
        "preferred_skills": ["Script Writing", "Social Media Trends Awareness"],
        "responsibilities": ["Create authentic, brand-aligned content for social/ads",
                              "Film and lightly edit short-form video content",
                              "Incorporate feedback and brand guidelines into content"],
    }),
    (re.compile(r"script ?writer", re.I), {
        "job_category": "Content & Social Media",
        "required_skills": ["Scriptwriting", "Storytelling & Pacing", "Research"],
        "preferred_skills": ["SEO for Video", "Tone/Voice Adaptation"],
        "responsibilities": ["Write scripts tailored to the format and audience",
                              "Research topics to ensure accuracy and depth",
                              "Revise scripts based on feedback"],
    }),
    (re.compile(r"\.net\b|asp\.net\b", re.I), {
        "job_category": "Software Engineering",
        "required_skills": ["C#", ".NET", "ASP.NET", "REST APIs", "SQL"],
        "preferred_skills": ["Azure", "Entity Framework"],
        "responsibilities": ["Build and maintain backend services and APIs",
                              "Write clean, testable, maintainable code",
                              "Collaborate with cross-functional teams on feature delivery"],
    }),
    (re.compile(r"\breact\b", re.I), {
        "job_category": "Software Engineering",
        "required_skills": ["React", "JavaScript", "TypeScript", "HTML", "CSS"],
        "preferred_skills": ["Next.js", "Redux"],
        "responsibilities": ["Build and maintain responsive web UI components",
                              "Collaborate with designers and backend engineers",
                              "Write clean, reusable, well-tested frontend code"],
    }),
    (re.compile(r"machine learning|\bml engineer\b", re.I), {
        "job_category": "Data & Machine Learning",
        "required_skills": ["Python", "Machine Learning", "SQL", "Pandas", "Scikit-learn"],
        "preferred_skills": ["TensorFlow/PyTorch", "MLOps"],
        "responsibilities": ["Build, train, and evaluate machine learning models",
                              "Prepare and analyze datasets",
                              "Deploy and monitor models in production"],
    }),
    (re.compile(r"gen ?ai|\bllm\b", re.I), {
        "job_category": "Data & Machine Learning",
        "required_skills": ["Python", "LLM", "RAG", "GenAI", "FastAPI"],
        "preferred_skills": ["LangChain/LangGraph", "Vector Databases"],
        "responsibilities": ["Design and build LLM-powered application features",
                              "Implement and tune retrieval-augmented generation pipelines",
                              "Evaluate and improve model output quality"],
    }),
    (re.compile(r"\bdata analyst\b", re.I), {
        "job_category": "Data & Machine Learning",
        "required_skills": ["SQL", "Python", "Excel", "Power BI", "Data Visualization"],
        "preferred_skills": ["Statistics", "A/B Testing"],
        "responsibilities": ["Analyze data to surface actionable insights",
                              "Build dashboards and reports for stakeholders",
                              "Maintain data quality and documentation"],
    }),
    (re.compile(r"\bdevops\b", re.I), {
        "job_category": "Software Engineering",
        "required_skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Linux"],
        "preferred_skills": ["Terraform", "Monitoring & Observability"],
        "responsibilities": ["Build and maintain CI/CD pipelines",
                              "Manage cloud infrastructure and deployments",
                              "Monitor system health and troubleshoot incidents"],
    }),
]


def _generic_skill_profile(job_title: str) -> dict:
    """Fallback for any job title that doesn't match a known keyword pattern above — never leaves
    required_skills/responsibilities empty (that would fail the hard floor), but keeps the content
    genuinely generic rather than guessing at specifics for a role this table doesn't recognize.
    """
    return {
        "job_category": None,
        "required_skills": ["Communication", "Problem-Solving", "Time Management"],
        "preferred_skills": ["Relevant Industry Experience"],
        "responsibilities": [f"Execute day-to-day responsibilities for the {job_title} role",
                              "Collaborate with cross-functional team members",
                              "Report progress and results to stakeholders"],
    }


def _skill_profile_for_job_title(job_title: str | None) -> dict:
    for pattern, profile in _JOB_TITLE_SKILL_PROFILES:
        if pattern.search(job_title or ""):
            return profile
    return _generic_skill_profile(job_title or "this role")


def _salary_question(job_state: dict) -> str:
    """Adapts the salary question's currency hint to whatever location the recruiter already gave
    (location is always asked first, see _CHECKLIST_ORDER) — reported live: a Bangalore posting
    should be asked about in ₹, not left currency-agnostic. Deliberately never suggests a specific
    NUMBER, only the currency itself — a concrete figure would be an invented anchor, exactly what
    this app's anti-fabrication rules elsewhere forbid; the safe "Competitive, negotiable" chip
    (_CHECKLIST_CHIPS above) is untouched so a recruiter who doesn't want to specify a currency-
    bound figure still has that option.
    """
    base = _CHECKLIST_QUESTIONS["salary"]
    symbol = _currency_symbol_for_location(job_state.get("location"))
    return f"{base} (in {symbol}, based on the location you gave)" if symbol else base


def _sanitize_salary_chips(chips: list[str], expected_symbol: str | None) -> list[str]:
    """Model-supplied salary chips sometimes bake in a numeric example in the WRONG currency for
    the recruiter's own location — reported live: "$50,000–$70,000" suggested for a Delhi NCR
    posting. Converting the figure isn't the fix (a $ amount run through some exchange rate is
    still a fabricated number, exactly what this app avoids elsewhere) — any chip mentioning a
    DIFFERENT currency symbol than expected is dropped instead; if that empties the list, fall back
    to the safe, currency-neutral "Competitive, negotiable" (never left with zero chips — salary is
    mandatory, so an empty chip list here would be the exact "nothing tappable" gap already fixed
    once for this field).
    """
    if not expected_symbol:
        return chips
    wrong_symbols = [s for s in _ALL_CURRENCY_SYMBOLS if s != expected_symbol]
    filtered = [c for c in chips if not any(sym in c for sym in wrong_symbols)]
    return filtered or _CHECKLIST_CHIPS["salary"]

# Deterministically defaulted the moment a job_title exists, never asked about at all — per the
# same founder decision as the shrunk checklist above. The recruiter can still change any of these
# via chat (a normal field_updates edit) or the draft panel's own dropdown for each. work_mode
# defaults to "Remote" specifically per a later founder correction (creator/content roles are
# commonly remote-friendly) rather than staying unset — same defaulted treatment as the others.
# education is deliberately NOT defaulted (removed per a later founder correction) — a degree
# requirement is a material, candidate-facing fact that genuinely varies per role, so it's left
# entirely unset (and out of the draft panel) unless the recruiter states one explicitly.
_DEFAULT_FIELD_VALUES = {
    "experience": "Mid-Level",
    "employment_type": "Full-time",
    "work_mode": "Remote",
}


def _apply_default_field_values(job_state: dict) -> dict:
    """Fills experience/employment_type/work_mode with their standing defaults the moment
    job_title exists and they're still empty — never overwrites a real value (the recruiter's own
    or an earlier default), so this is idempotent and safe to call on every turn.
    """
    if not job_state.get("job_title"):
        return job_state
    result = dict(job_state)
    for field, default_value in _DEFAULT_FIELD_VALUES.items():
        if not result.get(field):
            result[field] = default_value
    return result
# The moment the standard checklist (location/work_mode/salary, plus the one-time skills
# follow-up) resolves, generation now fires automatically — no manual confirmation ceremony, no
# "Generate JD" chip to click (see route_after_apply's ready_to_generate() auto-route). This is
# just the announcement text for that instant; the actual draft appears a few seconds later via
# the same turn's generate_jd node.
_AUTO_GENERATE_RESPONSE = "Perfect — that's everything I need. Drafting your job post now..."


def _next_checklist_prompt(job_state: dict, skipped: set[str] | None = None) -> tuple[str, str, list[str]] | None:
    """First unresolved, not-yet-skipped field (in standard-checklist order) plus its canned
    question + chips, or None once every field is either set or skipped — used to deterministically
    pivot away from a skills/responsibilities follow-up the model tried to ask despite never being
    supposed to, or an explicit "Skip this" click, rather than leaving the recruiter with an
    acknowledgment and no next question. `skipped` matters: a skipped field's job_state value stays
    empty by design (that's what skipping means), so without excluding it here this would just
    re-offer the exact same question forever instead of moving on.
    """
    skipped = skipped or set()
    for field in _CHECKLIST_ORDER:
        if field in skipped:
            continue
        if not job_state.get(field):
            question = _salary_question(job_state) if field == "salary" else _CHECKLIST_QUESTIONS[field]
            return question, field, _checklist_chips_for(field)
    return None


def _fallback_turn_analysis(state: GraphState) -> TurnAnalysis:
    """Built when call_structured exhausts its retries (a Mistral 429 burst, or a rarer
    unparseable structured-output response — indistinguishable from here). The recruiter's own
    last message is always lost either way (analyze_turn never got far enough to read it), but
    they shouldn't ALSO lose the guided checklist they were mid-way through: if the hard floor is
    already met, re-ask deterministically — the same helper the skills-loop-cap override and the
    /skip-field endpoint already use — with real chips instead of a dead-end "please rephrase."
    Only the genuinely-unknown-state case (no title/skills yet) keeps the bare rephrase message,
    since there's nothing deterministic to ask instead.
    """
    job_state = state.get("job_state") or {}
    if _skills_floor_met(job_state):
        skipped = set(state.get("skipped_checklist_fields") or [])
        next_field = _next_checklist_prompt(job_state, skipped)
        if next_field:
            question, field, chips = next_field
            return TurnAnalysis(
                intent=Intent.CHITCHAT_OR_UNCLEAR,
                enough_information=False,
                response=f"Sorry, I had trouble processing that last message — could you confirm: {question}",
                asking_about_field=field,
                suggested_options=chips,
                options_multi_select=field in _MULTI_SELECT_CHECKLIST_FIELDS,
            )
        return TurnAnalysis(
            intent=Intent.CHITCHAT_OR_UNCLEAR,
            enough_information=True,
            response=_AUTO_GENERATE_RESPONSE,
        )
    return TurnAnalysis(intent=Intent.CHITCHAT_OR_UNCLEAR, enough_information=False, response=FALLBACK_RESPONSE)


def checklist_resolved(job_state: dict, skipped: set[str] | None = None) -> bool:
    """True once every standard-checklist field is either set or explicitly skipped."""
    skipped = skipped or set()
    return all(field in skipped or job_state.get(field) for field in _CHECKLIST_ORDER)


def ready_to_generate(state: GraphState) -> bool:
    """Hard floor (title, location, salary, and skills-or-responsibilities — see sufficiency.py)
    is necessary but not sufficient for generation — without also requiring the rest of the
    standard checklist to be resolved, the JD generation prompt ends up working from a job_state
    thin enough that its own "role-standard enrichment" instructions invent specifics the recruiter
    never gave. This is the single source of truth for: route_after_apply's auto-generate trigger
    (the instant this flips true for the first time, generation fires with no manual confirmation
    needed), the Regenerate button's enabled state, and the direct /generate endpoint's own guard.

    Already-published jobs are grandfathered past the checklist check: editing an existing,
    previously-complete job (job_id is set) shouldn't suddenly re-gate Regenerate just because this
    thread's own skipped_checklist_fields is empty (hydrated from the DB row, not derived from live
    chat) — only fresh, not-yet-published drafts enforce it.
    """
    job_state = state.get("job_state") or {}
    if not hard_floor_met(job_state):
        return False
    if state.get("job_id") is not None:
        return True
    skipped = set(state.get("skipped_checklist_fields") or [])
    return checklist_resolved(job_state, skipped)


# Last-resort keyword sniffing on the response TEXT — the model sometimes spells options out in
# prose ("...4-6 years, or 7+ years?") without ALSO populating suggested_options, or asks a
# standard-checklist question without correctly setting asking_about_field this turn. Catches
# those cases so the field-specific defaults above still apply even when asking_about_field
# didn't line up.
_KEYWORD_FIELD_HINTS = (
    ("experience", ("years of experience", "how many years", "experience level")),
    ("work_mode", ("remote, hybrid", "hybrid, or onsite", "hybrid or onsite", "remote or onsite", "onsite, hybrid")),
    ("employment_type", ("full-time, part-time", "full-time, or part-time", "part-time, contract")),
    # platforms/location/salary — reported live: on a "combined" turn (acknowledge the platforms
    # answer + ask about location, all in one response), the model sometimes leaves
    # asking_about_field entirely null for the new question, and since these three are mandatory
    # (not in OPTIONAL_SKIPPABLE_FIELDS), NOTHING downstream could recover a real field name for
    # them — the turn fell through to the model's own (often bogus, e.g. a stray "Skip") chips
    # untouched. See _MANDATORY_CHECKLIST_HINT_FIELDS below — these three are matched even while
    # "in missing" (the normal, expected state for an unresolved mandatory field), unlike the three
    # above which skip that case.
    ("platforms", ("platform(s) are you hiring", "which platform")),
    ("location", ("city or region", "will this role be based", "specific city", "which city")),
    ("salary", ("salary range", "what's the salary", "salary for this role")),
)

# Fields matched by _KEYWORD_FIELD_HINTS above even when "in missing" — for these three, being
# flagged as missing IS the normal state while the question is still unresolved (they're mandatory,
# always genuinely needed until answered), unlike the experience/work_mode/employment_type entries,
# where "in missing" signals something more specific worth deferring to instead.
_MANDATORY_CHECKLIST_HINT_FIELDS = {"platforms", "location", "salary"}

# Absolute last resort when a question produced zero options and no field could even be guessed
# (e.g. the model failed to supply its own role-specific skill/responsibility suggestions for an
# open "any others?" question). A single, unambiguous "Skip" — matching the same word/style as
# the real Skip button elsewhere — rather than a vague "That's all"/"Let me add more" pair, which
# didn't tell the recruiter what either option actually did (free text is always available
# regardless, so a second "add more" chip added no real function). Every question still gets
# SOMETHING tappable, never a bare prompt with nothing to click.
_GENERIC_FALLBACK_OPTIONS = ["Skip"]


def apply_field_changes(
    job_state: dict,
    field_updates: dict | None = None,
    list_operations: list[dict] | None = None,
    company_overrides: dict | None = None,
) -> dict:
    """Applies scalar/list/company-override changes to a COPY of job_state. Shared by the
    LLM-driven apply_updates below and the direct (non-chat) job-state patch endpoint in
    routes/chat.py, so both go through identical field validation and list-operation semantics
    rather than two versions of the same logic drifting apart.
    """
    job_state = dict(job_state)
    job_state.setdefault("required_skills", [])
    job_state.setdefault("preferred_skills", [])
    job_state.setdefault("responsibilities", [])
    job_state.setdefault("custom_questions", [])
    job_state.setdefault("platforms", [])
    job_state.setdefault("company_overrides", {})

    for key, value in (field_updates or {}).items():
        if key in SCALAR_JOB_FIELDS:
            job_state[key] = value

    for op in list_operations or []:
        field = op.get("field")
        operation = op.get("operation")
        values = op.get("values") or []
        if field not in LIST_JOB_FIELDS:
            continue
        job_state[field] = _apply_list_operation(job_state.get(field, []), operation, values)

    overrides = dict(job_state.get("company_overrides", {}))
    for key, value in (company_overrides or {}).items():
        if key in COMPANY_OVERRIDE_FIELDS:
            overrides[key] = value
    job_state["company_overrides"] = overrides

    return job_state


def apply_updates(state: GraphState, config: RunnableConfig) -> dict:
    analysis = state.get("pending_analysis") or {}
    original_job_state = state.get("job_state") or {}

    intent = analysis.get("intent")

    # custom_questions is recruiter-manual-only (see JobState's comment) — the chat LLM must never
    # write to it, no matter what a future prompt tweak might accidentally invite it to try.
    # Deterministic guard, not prompt-trusted: strip any such op before it's ever applied. The
    # draft panel's own Custom Questions section writes through a separate, non-LLM endpoint
    # (patch_job_state) that isn't filtered here, so recruiter edits are unaffected.
    llm_list_operations = [
        op for op in (analysis.get("list_operations") or []) if op.get("field") != "custom_questions"
    ]

    # Off-topic / document-review turns carry no confirmed job content — skip everything.
    # Advice turns may still state a real fact (e.g. the job title) alongside the question, so
    # scalar field_updates still apply — but the advice itself is always skill-shaped, so
    # list_operations/company_overrides are blocked until an explicit later confirmation.
    # Defense in depth: enforced here regardless of what the model actually returned.
    if intent in _BLOCK_ALL_VALUES:
        job_state = apply_field_changes(original_job_state)
    elif intent in _BLOCK_LIST_AND_OVERRIDE_VALUES:
        job_state = apply_field_changes(original_job_state, analysis.get("field_updates"))
    else:
        job_state = apply_field_changes(
            original_job_state,
            analysis.get("field_updates"),
            llm_list_operations,
            analysis.get("company_overrides"),
        )

    # Deterministic backstop for experience/work_mode/employment_type the model verbally
    # acknowledged but never actually put in field_updates (see _extract_fact_backstop). Only
    # fills a field that's STILL empty after the model's own extraction above, so this can never
    # override a real value — and never runs for _BLOCK_ALL_VALUES turns, matching the "off-topic
    # turns carry no confirmed job content" rule those already enforce.
    if intent not in _BLOCK_ALL_VALUES:
        last_human_text = ""
        for m in reversed(state.get("messages") or []):
            if isinstance(m, HumanMessage):
                last_human_text = m.content or ""
                break
        for field, value in _extract_fact_backstop(last_human_text).items():
            if not job_state.get(field):
                job_state[field] = value
        # Platforms is multi-select — the chip UI sends a plain comma-separated reply (e.g.
        # "Facebook, YouTube"). Deterministic parse against the known chip labels as a backstop for
        # whatever the model's own list_operations extraction misses, gated to the turn right after
        # the platforms question was actually asked — unlike the experience/work_mode/employment_type
        # backstop above (which is safe to run unconditionally on any turn), an unrelated mention of
        # "Instagram" or "YouTube" elsewhere (e.g. describing the role itself) shouldn't get silently
        # swept into this field. Only ADDS what's found, never removes/replaces, so this can never
        # wipe out a broader answer the model DID correctly extract on its own.
        if state.get("asking_about_field") == "platforms":
            for platform in _extract_platforms_backstop(last_human_text):
                current_platforms = job_state.get("platforms") or []
                if platform not in current_platforms:
                    job_state["platforms"] = [*current_platforms, platform]

    # Deterministic defaults (experience/employment_type/education) — see _apply_default_field_values.
    # Applied AFTER the fact backstop above so a recruiter-stated or extracted value always wins;
    # this only ever fills in what's still genuinely empty once job_title exists.
    job_state = _apply_default_field_values(job_state)

    llm_enough = bool(analysis.get("enough_information"))
    llm_missing = analysis.get("missing_essential") or []

    ok = sufficiency_ok(job_state, llm_enough, llm_missing)
    missing = combined_missing_essential(job_state, llm_missing)
    content_changed = job_state != original_job_state
    jd_versions_exist = bool(state.get("jd_versions"))
    is_published_job = state.get("job_id") is not None

    phase = state.get("phase", "collecting")
    if phase == "collecting":
        if intent == Intent.FINISH_COLLECTING.value:
            # An explicit finish-phrase overrides soft, LLM-judged missing fields (e.g. "would be
            # nice to know experience") — only the deterministic hard floor can still block finishing,
            # per the non-negotiable "respect 'that's all' unless something genuinely essential is missing".
            if hard_floor_met(job_state):
                phase = "summary"
                missing = []
            else:
                missing = hard_floor_missing(job_state)
        elif ok:
            phase = "summary"
    elif phase == "published" and content_changed:
        # The recruiter is editing an already-live job. This does NOT touch the live row (see the
        # write-through guard below) — it only becomes live once publish_edit runs on explicit
        # confirmation, per "current live job remains unchanged before Publish Edit".
        phase = "editing"

    jd_stale = (jd_versions_exist and content_changed) or state.get("jd_stale", False)

    selected_version = state.get("selected_version")
    if analysis.get("selected_version") in ("1", "2"):
        selected_version = analysis["selected_version"]

    # Purely a display label for the UI phase badge — publishing itself is gated in
    # route_after_apply / publish_job by selected_version + jd_stale directly, not by this.
    # Published-job edits stay in "editing" throughout (that's what route_after_apply's
    # publish_edit branch checks for), so this bump only applies to fresh drafts.
    if not is_published_job and phase == "jd_selection" and jd_versions_exist and selected_version and not jd_stale:
        phase = "publish_confirm"

    company_profile = state.get("company_profile") or {}
    company_id = company_profile.get("id")
    session_id = config["configurable"]["thread_id"]
    if not is_published_job:
        # Write-through only applies to drafts. An edit to an already-published job must stay
        # off the live row until publish_edit confirms it — the checkpoint alone holds it until then.
        if job_state.get("job_title") and company_id is not None:
            owner_user_id = config["configurable"].get("user_id")
            upsert_job_draft(session_id, company_id, job_state, jd_stale, owner_user_id=owner_user_id)
        if selected_version != state.get("selected_version"):
            save_selected_version(session_id, selected_version)

    # True only when THIS turn's edit is what just made the JD stale (not merely "still stale
    # from an earlier turn") — gates the auto-regenerate branch in route_after_apply so an
    # unrelated chitchat message on an already-stale job doesn't spuriously trigger a call.
    jd_needs_refresh = jd_versions_exist and content_changed

    # Never trust the model's compliance alone: only ever offer "Skip this" for a field that's
    # actually optional, never when this turn just flagged it essential for the role, and never
    # when the reply isn't actually posing a question this turn (the model doesn't reliably
    # clear this back to null on a turn that just moves on, e.g. after acknowledging a skip).
    # Checking for "?" anywhere (not just as the last character) matters: the model sometimes
    # appends a clarifying example after the question mark ("...or tools?" -> "...or tools? For
    # example, Python, Java...") which previously made rstrip().endswith("?") false and silently
    # dropped both the Skip button and every chip on an obviously-a-question turn.
    asking_about_field = analysis.get("asking_about_field")
    reply_is_a_question = "?" in analysis.get("response", "")
    # Preserved separately from asking_about_field below: that variable gets nulled for a mandatory
    # field (job_title/location/salary) specifically so no "Skip this" button renders, but the chip
    # fallback further down still needs to know what field this question is REALLY about — without
    # this, a compliance gap (model asks about location/salary but forgets to supply its own chips)
    # would fall through to the generic ["Skip"] fallback, which is exactly the affordance a
    # mandatory field must never show.
    raw_asking_about_field = asking_about_field
    # required_skills/responsibilities are in OPTIONAL_SKIPPABLE_FIELDS (see models.py) so whichever
    # one ISN'T covering the hard floor can still get a "Skip this" button — but only once the other
    # one already has content. Neither individual field name ever appears in `missing` (the hard
    # floor tracks them jointly as the "required_skills_or_responsibilities" sentinel), so the
    # generic `asking_about_field in missing` check above can't catch the case where BOTH are still
    # empty — without this, the floor itself could be skipped away entirely.
    skills_pair_floor_unmet = (
        asking_about_field in ("required_skills", "responsibilities")
        and "required_skills_or_responsibilities" in missing
    )
    if (
        asking_about_field not in OPTIONAL_SKIPPABLE_FIELDS
        or asking_about_field in missing
        or skills_pair_floor_unmet
        or not reply_is_a_question
    ):
        asking_about_field = None

    # The model sometimes asks a standard checklist question in a recognizable way (e.g. "How
    # many years of experience...?") without correctly setting asking_about_field — meaning the
    # Skip button silently wouldn't show even on a turn that's obviously skippable. Sniff the
    # response text the same way the chip fallback below does, independent of whether the model
    # already supplied its own suggested_options, so Skip-button coverage never lags chip
    # coverage.
    if not asking_about_field and reply_is_a_question:
        # Only the sentence that actually POSES the question, never an earlier acknowledgment
        # clause restating a fact the recruiter already gave — "2 years of experience — got it.
        # What skills should this role require?" contains "years of experience" in the
        # acknowledgment half, which would otherwise wrongly hijack this turn's chips onto the
        # experience-band defaults even though the real question is about something else entirely
        # (reported live: bot asks about skills, shows experience chips).
        response_text = analysis.get("response", "")
        question_sentences = [s for s in re.split(r"(?<=[.!?])\s+", response_text) if "?" in s]
        question_text_lower = (" ".join(question_sentences) if question_sentences else response_text).lower()
        for candidate_field, phrases in _KEYWORD_FIELD_HINTS:
            if candidate_field in missing and candidate_field not in _MANDATORY_CHECKLIST_HINT_FIELDS:
                continue
            if any(phrase in question_text_lower for phrase in phrases):
                asking_about_field = candidate_field
                raw_asking_about_field = candidate_field
                break
        # A recovered field can itself be mandatory (platforms/location/salary) — re-apply the same
        # "no Skip button for a mandatory field" rule the nulling check above already enforced for
        # the model's own asking_about_field. This recovery runs AFTER that check (it only fires
        # when asking_about_field was still null), so without this, a just-recovered mandatory
        # field would stay set and incorrectly show a Skip button it must never have.
        if asking_about_field in _MANDATORY_CHECKLIST_HINT_FIELDS:
            asking_about_field = None

    # A recruiter who verbally declines an optional field ("no preference", "not needed", "we can
    # skip that") never touches the literal Skip button — but ready_to_generate()/checklist_resolved
    # only ever trust skipped_checklist_fields, never the model's own "enough_information" judgment
    # (see nodes.py docstrings). Without this, a verbally-declined optional field leaves the
    # checklist gate stuck open forever: the conversation moves on, but the Generate button stays
    # disabled with nothing left on screen to click. Detect the advance deterministically instead of
    # trusting the model to have "meant" to skip it: if we were asking about an optional field last
    # turn and it's still unset after this turn's updates, and the model isn't asking about that
    # same field again right now, the recruiter's reply moved past it — treat it exactly like an
    # explicit Skip click.
    skipped_checklist_fields = set(state.get("skipped_checklist_fields") or [])
    previous_asking_about_field = state.get("asking_about_field")
    if (
        previous_asking_about_field
        and previous_asking_about_field in OPTIONAL_SKIPPABLE_FIELDS
        and not job_state.get(previous_asking_about_field)
        and asking_about_field != previous_asking_about_field
    ):
        skipped_checklist_fields.add(previous_asking_about_field)

    # Same defense-in-depth as asking_about_field above — only ever surface chips on a turn
    # that's actually posing a question, regardless of what the model returned.
    suggested_options = analysis.get("suggested_options") or []
    options_multi_select = bool(analysis.get("options_multi_select"))
    if not reply_is_a_question:
        suggested_options = []
    elif asking_about_field in _DEFAULT_OPTIONS_BY_FIELD:
        # These fields have exactly one fixed, canonical single-choice answer set — always
        # use it instead of trusting the model's own suggested_options, which occasionally drift
        # (e.g. still offering leftover skill-style chips for an experience-band question). No
        # ambiguity here, so there's no reason to prefer a model-supplied value over the known-good
        # default the way the prose-sniffing fallback below has to for open-ended fields.
        suggested_options = _DEFAULT_OPTIONS_BY_FIELD[asking_about_field]
        options_multi_select = False
    elif raw_asking_about_field == "location":
        # Same "always override, never trust the model's own suggested_options" tier as
        # _DEFAULT_OPTIONS_BY_FIELD above — keyed on raw_asking_about_field since location is
        # mandatory (asking_about_field itself was already nulled by the Skip-eligibility check).
        # Verified live: when the model DID supply its own location chips, they bypassed the
        # randomized pool entirely — sometimes missing "Worldwide" altogether, sometimes even
        # conflating a work_mode value ("Remote") into what's supposed to be a place name — so
        # this can't be left to model discretion the way open-ended fields below still have to be.
        suggested_options = _checklist_chips_for("location")
        options_multi_select = False
    elif not suggested_options:
        # The prompt asks the model for options on EVERY question, but compliance isn't
        # perfect — sometimes it spells options out in prose instead ("...4-6 years, or 7+
        # years?") without also populating suggested_options. Guarantee something tappable
        # always appears rather than depending on prompt compliance alone. Uses
        # raw_asking_about_field (before the Skip-eligibility nulling above), not the possibly-
        # nulled asking_about_field: a mandatory field (location/salary) still needs its OWN real
        # chip fallback here — falling through to _GENERIC_FALLBACK_OPTIONS (["Skip"]) would show
        # a misleading skip affordance on a field that can't actually be skipped. required_skills/
        # responsibilities/salary have no canonical fixed chip set of their own, so they correctly
        # still get no chips at all in that case (a bare textbox), never the generic "Skip" one.
        if raw_asking_about_field in OPTIONAL_SKIPPABLE_FIELDS:
            suggested_options = _DEFAULT_OPTIONS_BY_FIELD.get(raw_asking_about_field, _GENERIC_FALLBACK_OPTIONS)
        else:
            suggested_options = _checklist_chips_for(raw_asking_about_field)
        options_multi_select = False

    # Salary chips get the same currency sanitization as the question text above, regardless of
    # which branch above produced them — the model's OWN chips (not just its own phrasing) can also
    # bake in a numeric example in the wrong currency for the recruiter's location (reported live:
    # "$50,000–$70,000" suggested for a Delhi NCR posting). See _sanitize_salary_chips. Triggered by
    # the SAME structural text check analyze_turn's own currency fix uses (does the response
    # actually pose a salary question), not just raw_asking_about_field alone — verified live that
    # relying on the model's own asking_about_field classification here missed real salary-chip
    # turns where the model's structured field didn't say "salary" even though its response and
    # chips clearly were about it (the response TEXT still got corrected via that same signal in
    # analyze_turn, but the CHIPS, sanitized here in a different function, silently didn't).
    response_text_for_chip_check = analysis.get("response", "") or ""
    response_question_sentences_for_chip_check = [
        s for s in re.split(r"(?<=[.!?])\s+", response_text_for_chip_check) if "?" in s
    ]
    response_is_salary_question = bool(
        _FIELD_QUERY_KEYWORDS["salary"].search(
            " ".join(response_question_sentences_for_chip_check) or response_text_for_chip_check
        )
    )
    if raw_asking_about_field == "salary" or response_is_salary_question:
        # job_state (not state.get("job_state")) — the local variable already merged THIS turn's
        # own field_updates in above, so a location set in the SAME combined turn is still seen.
        expected_symbol = _currency_symbol_for_location(job_state.get("location"))
        suggested_options = _sanitize_salary_chips(suggested_options, expected_symbol)

    # A dedicated "Skip this" button already renders in the UI whenever asking_about_field is set
    # (see appendSkipButton in job-modal.js) — a suggested_options chip whose ONLY content is the
    # same generic "Skip" fallback adds nothing but a confusing second, differently-styled skip
    # affordance stacked on the same question (reported live). Suppress it in that specific case;
    # any OTHER chip content (real answer options like work_mode's Remote/Hybrid/Onsite) still
    # renders normally alongside the dedicated button, since those aren't redundant with it.
    if asking_about_field and suggested_options == _GENERIC_FALLBACK_OPTIONS:
        suggested_options = []

    # Absolute last resort — guarantee something tappable on every genuine question (reported live:
    # "bot sometimes fails to suggest chips"). The field-specific overrides above cover every
    # RECOGNIZED case; this catches whatever's left — an unrecognized field, or the model asking a
    # question without ever setting asking_about_field at all. Only fires when there's truly nothing
    # else tappable: a dedicated Skip button (asking_about_field set) already covers that case, so
    # this never stacks a redundant chip alongside it.
    if reply_is_a_question and not suggested_options and not asking_about_field:
        suggested_options = _GENERIC_FALLBACK_OPTIONS

    return {
        "job_state": job_state,
        "phase": phase,
        "missing_essential": missing,
        "jd_stale": jd_stale,
        "selected_version": selected_version,
        "pending_analysis": None,
        "last_intent": intent,
        "jd_needs_refresh": jd_needs_refresh,
        "last_response": analysis.get("response", ""),
        "asking_about_field": asking_about_field,
        "suggested_options": suggested_options,
        "options_multi_select": options_multi_select,
        "skipped_checklist_fields": list(skipped_checklist_fields),
    }


def _jd_document_message(version: str, jd: dict) -> AIMessage:
    """A JD version rendered as its own chat message (content stays empty — the frontend
    renders the structured payload as a card, with a "Choose Version" action) rather than in
    the side panel, per the in-chat JD requirement.
    """
    return AIMessage(content="", additional_kwargs={"jd_document": jd, "jd_version": version})


def generate_jd(state: GraphState, config: RunnableConfig) -> dict:
    """Generates ONE complete job description draft — not a pair to choose between. Immediately
    marks it selected (there's nothing to choose), so Publish becomes available right away.

    "Regenerate" (same function, called again once a draft already exists) is NOT a blank-page
    rewrite — it passes the CURRENT draft into the prompt as an ENHANCEMENT baseline: fix errors,
    polish wording, add missing depth, but never drop or replace a skill/point/fact that's already
    there (see _JD_REGENERATION_CONTEXT). This deliberately relies on the model's own judgment
    rather than a deterministic field-lock — "rewrite this typo, keep the meaning" is a genuine
    editorial judgment call, not a fact-fidelity check code can verify. A first attempt at this used
    a hard lock (verbatim-restore whatever the recruiter had last touched) and that was wrong in
    the other direction: it froze hand-edited content so hard that Regenerate couldn't even fix an
    obvious typo in it, which defeats the entire point of asking for a regeneration.
    """
    company_profile = state.get("company_profile") or {}
    job_state = state.get("job_state") or {}
    session_id = config["configurable"]["thread_id"]
    jd_versions_existing = state.get("jd_versions") or {}
    selected_version = state.get("selected_version")
    current_jd = jd_versions_existing.get(selected_version) if selected_version else None
    is_regeneration = bool(current_jd)

    time.sleep(3)  # this is the 2nd Mistral call in the same turn — avoid bursting past per-second rate limits
    prompt = build_jd_generation_prompt(company_profile, job_state, current_jd=current_jd)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Generate the job description based on the information above."),
    ]

    try:
        output = call_structured(JobDescriptionDraft, messages, retries=2)
    except Exception:
        logger.exception("generate_jd: structured output failed after retry")
        response = (
            "I hit a snag generating the job description just now — this wasn't about anything "
            "you entered, our end had trouble keeping up for a moment. Please click Generate "
            "again in a few seconds."
        )
        return {"messages": [AIMessage(content=response)], "last_response": response}

    output_dict = output.model_dump(mode="json")
    # The PROOFREAD MIRROR fields (see prompt) never get stored on the JD document itself — only
    # job_state, its single source of truth, gets corrected — see _apply_proofread_corrections.
    skills_family_output = {
        field: output_dict.pop(field, []) for field in ("required_skills", "preferred_skills", "responsibilities")
    }
    job_state = _apply_proofread_corrections(job_state, skills_family_output)

    jd = _apply_job_state_identity_fields(
        _dedupe_stand_out(_strip_markdown(output_dict), job_state), job_state
    )
    jd_versions = {"1": jd}
    save_jd_versions(session_id, jd_versions)
    save_selected_version(session_id, "1")

    # job_state may have just been corrected above (typo fixes to required_skills/preferred_skills/
    # responsibilities) — write it through the same way apply_updates/patch_job_state do, so the
    # correction survives a page reload, not just this in-memory turn. Drafts only, same guard as
    # everywhere else: an already-published job's edits stay off the live row until the recruiter
    # explicitly clicks Publish Edit.
    is_published_job = state.get("job_id") is not None
    if not is_published_job and job_state.get("job_title"):
        company_id = company_profile.get("id")
        owner_user_id = config["configurable"].get("user_id")
        if company_id is not None:
            upsert_job_draft(session_id, company_id, job_state, False, owner_user_id=owner_user_id)

    job_title = job_state.get("job_title") or "this role"
    response = (
        f"Here's the refreshed job description for {job_title} — your edits are preserved where "
        "they still apply. Let me know if you'd like any changes, or click Regenerate again — "
        "otherwise it's ready to publish."
        if is_regeneration
        else f"Here's the job description I've drafted for {job_title}. Let me know if you'd like any "
        "changes, or click Regenerate for a fresh draft — otherwise it's ready to publish."
    )
    # Editing an already-published job stays in the "editing" phase throughout (that's what
    # gates publish_edit) rather than the fresh-draft "jd_selection" phase.
    next_phase = "editing" if state.get("job_id") else "jd_selection"
    return {
        "job_state": job_state,
        "jd_versions": jd_versions,
        "jd_stale": False,
        "selected_version": "1",
        "phase": next_phase,
        "messages": [
            AIMessage(content=response),
            _jd_document_message("1", jd),
        ],
        "last_response": response,
    }


def refine_jd(state: GraphState, config: RunnableConfig) -> dict:
    company_profile = state.get("company_profile") or {}
    job_state = state.get("job_state") or {}
    jd_versions = state.get("jd_versions") or {}
    version = state.get("selected_version")
    session_id = config["configurable"]["thread_id"]

    current_jd = jd_versions.get(version) or {}
    instruction_message = state["messages"][-1] if state.get("messages") else None
    instruction = instruction_message.content if instruction_message else ""

    time.sleep(3)  # this is the 2nd Mistral call in the same turn — avoid bursting past per-second rate limits
    prompt = build_jd_refinement_prompt(company_profile, job_state, version, current_jd)
    messages = [SystemMessage(content=prompt), HumanMessage(content=instruction)]

    try:
        output = call_structured(JDRefinementOutput, messages, retries=2)
    except Exception:
        logger.exception("refine_jd: structured output failed after retry")
        response = (
            "I hit a snag applying that change — this wasn't about anything you said, our end had "
            "trouble keeping up for a moment. Please try that again."
        )
        return {"messages": [AIMessage(content=response)], "last_response": response}

    updated_jd_dict = output.updated_jd.model_dump(mode="json")
    # Same as generate_jd: required_skills/preferred_skills/responsibilities never get stored on
    # the JD document itself, job_state is their only home. Refine (a chat-driven, instruction-
    # specific edit like "make it more professional") doesn't proofread job_state's lists the way
    # Regenerate does — just strip whatever the model returned for these here rather than persist
    # possibly-stale, unused duplicate content into the saved draft.
    for field in ("required_skills", "preferred_skills", "responsibilities"):
        updated_jd_dict.pop(field, None)
    updated_jd = _apply_job_state_identity_fields(
        _dedupe_stand_out(_strip_markdown(updated_jd_dict), job_state), job_state
    )
    new_jd_versions = dict(jd_versions)
    new_jd_versions[version] = updated_jd
    save_refined_jd(session_id, version, updated_jd)

    response = f"Updated version {version}: {output.change_summary}"
    return {
        "jd_versions": new_jd_versions,
        "messages": [
            AIMessage(content=response),
            _jd_document_message(version, updated_jd),
        ],
        "last_response": response,
    }


def publish_job(state: GraphState, config: RunnableConfig) -> dict:
    company_profile = state.get("company_profile") or {}
    jd_versions = state.get("jd_versions") or {}
    selected_version = state.get("selected_version")
    session_id = config["configurable"]["thread_id"]

    selected_jd = jd_versions.get(selected_version) or {}
    company_id = company_profile.get("id")
    company_name = company_profile.get("company_name", "")

    if not selected_jd or company_id is None:
        # Defensive guard — route_after_apply should already prevent reaching this node
        # without a valid selection, but never publish an empty/malformed JD.
        response = "I couldn't find the selected job description to publish — could you pick a version again?"
        return {"messages": [AIMessage(content=response)], "last_response": response}

    try:
        record = finalize_publish(session_id, company_id, company_name, selected_jd, selected_version)
    except Exception:
        logger.exception("publish_job: finalize_publish failed")
        response = (
            "Something went wrong while publishing this job — nothing was lost, please try again."
        )
        return {"messages": [AIMessage(content=response)], "last_response": response}

    job_id = record.get("job_id") if record else None
    if not job_id:
        response = "Something went wrong while publishing — please try again."
        return {"messages": [AIMessage(content=response)], "last_response": response}

    response = f"Your job has been successfully published as {job_id}."
    return {
        "phase": "published",
        "messages": [AIMessage(content=response)],
        "last_response": response,
    }


def publish_edit(state: GraphState, config: RunnableConfig) -> dict:
    """Confirms a pending edit to an already-published job. Never allocates a Job ID or
    touches status/published_at — only content columns + selected_jd change, in place.
    """
    job_state = state.get("job_state") or {}
    jd_versions = state.get("jd_versions") or {}
    selected_version = state.get("selected_version")
    job_id = state.get("job_id")
    session_id = config["configurable"]["thread_id"]

    selected_jd = jd_versions.get(selected_version) or {}
    if not selected_jd or not job_id:
        response = "I couldn't find the selected job description to publish — could you pick a version again?"
        return {"messages": [AIMessage(content=response)], "last_response": response}

    try:
        finalize_edit(session_id, job_state, selected_jd, selected_version)
    except Exception:
        logger.exception("publish_edit: finalize_edit failed")
        response = "Something went wrong while publishing this edit — nothing was lost, please try again."
        return {"messages": [AIMessage(content=response)], "last_response": response}

    response = f"Your changes have been published — {job_id} is now updated."
    return {
        "phase": "published",
        "jd_stale": False,
        "messages": [AIMessage(content=response)],
        "last_response": response,
    }


def route_after_apply(state: GraphState) -> str:
    intent = state.get("last_intent")
    jd_versions = state.get("jd_versions") or {}

    if intent == Intent.REQUEST_REFINEMENT.value and jd_versions and state.get("selected_version"):
        return "refine_jd"

    # First-time generation now fires automatically the instant the standard checklist resolves —
    # no manual "Generate JD" click, no confirmation ceremony (per the founder's creator-role
    # pivot: the flow should draft the job post itself once everything essential is collected).
    # Guarded on `not jd_versions` so this can only ever fire once per conversation: the moment
    # generate_jd runs, jd_versions gets populated in this same turn's committed state, so no
    # later turn can re-trigger it — Regenerate after that point is always a deliberate, explicit
    # action via the direct /generate endpoint, never a side effect here.
    if not jd_versions and ready_to_generate(state):
        return "generate_jd"

    # Publishing is deliberately NEVER routed here either — it only ever happens via the direct
    # POST /api/chat/{session_id}/publish endpoint the "Publish Job" button calls (see the
    # CONFIRM_PUBLISH prompt guidance: a chat confirmation gets acknowledged in the reply text,
    # but the graph itself takes no action).

    # Regeneration is deliberately NEVER auto-triggered by an edit that makes the JD stale
    # either — an edit just leaves jd_stale=true as a visible signal, nothing more; only an
    # explicit Regenerate click (also the direct /generate endpoint) clears it.
    return END


# ============================================================================
# GRAPH WIRING + COMPILATION (formerly graph.py)
# ============================================================================

_graph = None
_checkpoint_conn: sqlite3.Connection | None = None


def build_graph() -> StateGraph:
    """load_context -> analyze_turn -> apply_updates -> {generate_jd | refine_jd | publish_job | publish_edit | END}"""
    graph = StateGraph(GraphState)
    graph.add_node("load_context", load_context)
    graph.add_node("analyze_turn", analyze_turn)
    graph.add_node("apply_updates", apply_updates)
    graph.add_node("generate_jd", generate_jd)
    graph.add_node("refine_jd", refine_jd)
    graph.add_node("publish_job", publish_job)
    graph.add_node("publish_edit", publish_edit)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "analyze_turn")
    graph.add_edge("analyze_turn", "apply_updates")
    graph.add_conditional_edges(
        "apply_updates",
        route_after_apply,
        {
            "generate_jd": "generate_jd",
            "refine_jd": "refine_jd",
            "publish_job": "publish_job",
            "publish_edit": "publish_edit",
            END: END,
        },
    )
    graph.add_edge("generate_jd", END)
    graph.add_edge("refine_jd", END)
    graph.add_edge("publish_job", END)
    graph.add_edge("publish_edit", END)

    return graph


def get_compiled_graph():
    global _graph, _checkpoint_conn
    if _graph is None:
        _checkpoint_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(_checkpoint_conn)
        _graph = build_graph().compile(checkpointer=checkpointer)
    return _graph
