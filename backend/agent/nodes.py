import logging
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from backend.agent.llm import call_structured
from backend.agent.prompts import build_jd_generation_prompt, build_jd_refinement_prompt, build_system_prompt
from backend.agent.state import GraphState
from backend.agent.sufficiency import (
    combined_missing_essential,
    hard_floor_met,
    hard_floor_missing,
    sufficiency_ok,
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

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = (
    "Sorry, I hit a snag processing that just now — this wasn't about anything you said, "
    "our end had trouble keeping up for a moment. Please try sending that again."
)

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\w)_([^_]+)_(?!\w)")
_MD_LEADING_BULLET_RE = re.compile(r"^[\-\*]\s+")


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
    if not state.get("company_profile"):
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
    messages = [SystemMessage(content=system_prompt), *state["messages"]]

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
    response_lower = (analysis.response or "").lower()
    is_skills_followup = "?" in response_lower and ("skill" in response_lower or "respons" in response_lower)
    followup_count = state.get("skills_followup_count", 0)
    if is_skills_followup:
        if followup_count >= _SKILLS_FOLLOWUP_CAP:
            # Cap already hit — deterministically override the response instead of asking about
            # skills/responsibilities yet again. Merge in this turn's own field_updates first so a
            # message that named a field (e.g. "Add Python, and it's hybrid") isn't immediately
            # re-asked about.
            prospective_job_state = dict(state.get("job_state") or {})
            prospective_job_state.update(analysis.field_updates or {})
            skipped = set(state.get("skipped_checklist_fields") or [])
            next_field = _next_checklist_prompt(prospective_job_state, skipped)
            if next_field:
                question, field, chips = next_field
                analysis = analysis.model_copy(
                    update={
                        "response": f"Got it, noted! {question}",
                        "asking_about_field": field,
                        "suggested_options": chips,
                        "options_multi_select": False,
                    }
                )
            else:
                analysis = analysis.model_copy(
                    update={
                        "response": _READY_TO_GENERATE_RESPONSE,
                        "asking_about_field": None,
                        "suggested_options": [],
                        "options_multi_select": False,
                        "enough_information": True,
                    }
                )
        else:
            updates["skills_followup_count"] = followup_count + 1

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
    "experience": ["0-1 years", "2-3 years", "4-6 years", "7+ years"],
    "education": ["Bachelor's degree", "Master's degree", "Not required"],
}

# Prompt-only compliance for "ask about skills/responsibilities at most once" proved unreliable in
# practice — the model kept re-asking "anything else?" after every single skill the recruiter
# named, sometimes 10+ times in a row (each one a Mistral call, compounding rate-limit risk on top
# of the frustration). SKILLS_FOLLOWUP_CAP deterministically cuts that off in analyze_turn below:
# once this many skills/responsibilities-flavored questions have been asked across the whole
# conversation, any further one gets swapped for a canned pivot to the next unresolved standard
# checklist field instead of trusting the model to stop on its own.
_SKILLS_FOLLOWUP_CAP = 2

_CHECKLIST_ORDER = ["experience", "location", "work_mode", "employment_type"]
_CHECKLIST_QUESTIONS = {
    "experience": "How many years of experience should this role require?",
    "location": 'Which city or region will this role be based in? You can also say "Worldwide" if it\'s fully remote.',
    "work_mode": "Should this role be Remote, Hybrid, or Onsite?",
    "employment_type": "Should this be a Full-time, Part-time, Contract, or Internship position?",
}
_CHECKLIST_CHIPS = {
    "experience": _DEFAULT_OPTIONS_BY_FIELD["experience"],
    "location": ["Worldwide", "New York", "London", "Bangalore"],
    "work_mode": _DEFAULT_OPTIONS_BY_FIELD["work_mode"],
    "employment_type": _DEFAULT_OPTIONS_BY_FIELD["employment_type"],
}
_READY_TO_GENERATE_RESPONSE = (
    'Everything\'s captured for this role! Click "Generate Full Description" in the panel on the '
    "right whenever you're ready."
)


def _next_checklist_prompt(job_state: dict, skipped: set[str] | None = None) -> tuple[str, str, list[str]] | None:
    """First unresolved, not-yet-skipped field (in standard-checklist order) plus its canned
    question + chips, or None once every field is either set or skipped — used to deterministically
    pivot away from a capped-out skills/responsibilities follow-up loop (see _SKILLS_FOLLOWUP_CAP)
    or an explicit "Skip this" click, rather than leaving the recruiter with an acknowledgment and
    no next question. `skipped` matters: a skipped field's job_state value stays empty by design
    (that's what skipping means), so without excluding it here this would just re-offer the exact
    same question forever instead of moving on.
    """
    skipped = skipped or set()
    for field in _CHECKLIST_ORDER:
        if field in skipped:
            continue
        if not job_state.get(field):
            return _CHECKLIST_QUESTIONS[field], field, _CHECKLIST_CHIPS[field]
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
    if hard_floor_met(job_state):
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
                options_multi_select=False,
            )
        return TurnAnalysis(
            intent=Intent.CHITCHAT_OR_UNCLEAR,
            enough_information=True,
            response=_READY_TO_GENERATE_RESPONSE,
        )
    return TurnAnalysis(intent=Intent.CHITCHAT_OR_UNCLEAR, enough_information=False, response=FALLBACK_RESPONSE)


def checklist_resolved(job_state: dict, skipped: set[str] | None = None) -> bool:
    """True once every standard-checklist field is either set or explicitly skipped."""
    skipped = skipped or set()
    return all(field in skipped or job_state.get(field) for field in _CHECKLIST_ORDER)


def ready_to_generate(state: GraphState) -> bool:
    """Hard floor (title + skills-or-responsibilities) is necessary but not sufficient for
    generation — without also requiring the standard checklist to be resolved, the JD generation
    prompt ends up working from a job_state thin enough that its own "role-standard enrichment"
    instructions invent specifics the recruiter never gave (a degree requirement, a certification,
    a year count). This is the single source of truth for both the "Generate Full Description"
    button's enabled state and the direct /generate endpoint's own guard — never rely on the LLM's
    own judgment (or the UI simply being clickable) as proof enough information was collected.

    Already-published jobs are grandfathered past the checklist re-check: editing an existing,
    previously-complete job (job_id is set) shouldn't suddenly re-gate Regenerate just because this
    thread's own skipped_checklist_fields is empty (it was hydrated from the DB row, not derived
    from live chat) — only fresh, not-yet-published drafts enforce the checklist.
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
)

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
            analysis.get("list_operations"),
            analysis.get("company_overrides"),
        )

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
    if (
        asking_about_field not in OPTIONAL_SKIPPABLE_FIELDS
        or asking_about_field in missing
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
        response_lower = analysis.get("response", "").lower()
        for candidate_field, phrases in _KEYWORD_FIELD_HINTS:
            if candidate_field in missing:
                continue
            if any(phrase in response_lower for phrase in phrases):
                asking_about_field = candidate_field
                break

    # Same defense-in-depth as asking_about_field above — only ever surface chips on a turn
    # that's actually posing a question, regardless of what the model returned.
    suggested_options = analysis.get("suggested_options") or []
    options_multi_select = bool(analysis.get("options_multi_select"))
    if not reply_is_a_question:
        suggested_options = []
    elif asking_about_field in _DEFAULT_OPTIONS_BY_FIELD:
        # These four fields have exactly one fixed, canonical single-choice answer set — always
        # use it instead of trusting the model's own suggested_options, which occasionally drift
        # (e.g. still offering leftover skill-style chips for an experience-band question). No
        # ambiguity here, so there's no reason to prefer a model-supplied value over the known-good
        # default the way the prose-sniffing fallback below has to for open-ended fields.
        suggested_options = _DEFAULT_OPTIONS_BY_FIELD[asking_about_field]
        options_multi_select = False
    elif not suggested_options:
        # The prompt asks the model for options on EVERY question, but compliance isn't
        # perfect — sometimes it spells options out in prose instead ("...4-6 years, or 7+
        # years?") without also populating suggested_options. Guarantee something tappable
        # always appears rather than depending on prompt compliance alone — reuse whatever
        # asking_about_field resolved to above (model-supplied or text-sniffed), and fall back
        # to a generic pair as an absolute last resort.
        suggested_options = _DEFAULT_OPTIONS_BY_FIELD.get(asking_about_field, _GENERIC_FALLBACK_OPTIONS)
        options_multi_select = False

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
    }


def _jd_document_message(version: str, jd: dict) -> AIMessage:
    """A JD version rendered as its own chat message (content stays empty — the frontend
    renders the structured payload as a card, with a "Choose Version" action) rather than in
    the side panel, per the in-chat JD requirement.
    """
    return AIMessage(content="", additional_kwargs={"jd_document": jd, "jd_version": version})


def generate_jd(state: GraphState, config: RunnableConfig) -> dict:
    """Generates ONE complete job description draft — not a pair to choose between. Immediately
    marks it selected (there's nothing to choose), so Publish becomes available right away;
    "Regenerate" (same REQUEST_JD_GENERATION intent, called again) replaces this single draft
    with a fresh one, it never creates a second one to pick between.
    """
    company_profile = state.get("company_profile") or {}
    job_state = state.get("job_state") or {}
    session_id = config["configurable"]["thread_id"]

    time.sleep(3)  # this is the 2nd Mistral call in the same turn — avoid bursting past per-second rate limits
    prompt = build_jd_generation_prompt(company_profile, job_state)
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

    jd = _strip_markdown(output.model_dump(mode="json"))
    jd_versions = {"1": jd}
    save_jd_versions(session_id, jd_versions)
    save_selected_version(session_id, "1")

    job_title = job_state.get("job_title") or "this role"
    response = (
        f"Here's the job description I've drafted for {job_title}. Let me know if you'd like any "
        "changes, or click Regenerate for a fresh draft — otherwise it's ready to publish."
    )
    # Editing an already-published job stays in the "editing" phase throughout (that's what
    # gates publish_edit) rather than the fresh-draft "jd_selection" phase.
    next_phase = "editing" if state.get("job_id") else "jd_selection"
    return {
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

    updated_jd = _strip_markdown(output.updated_jd.model_dump(mode="json"))
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

    # Generation is deliberately NEVER routed here, regardless of intent — collecting job
    # details is the bot's job, but actually calling the model to WRITE the description only
    # ever happens via the direct POST /api/chat/{session_id}/generate endpoint the "Generate
    # Full Description"/"Regenerate" button calls, never as a side effect of a chat turn (see
    # the REQUEST_JD_GENERATION / FINISH_COLLECTING prompt guidance: a chat "generate it now" or
    # a finish phrase gets acknowledged and pointed at the button, but the graph itself takes no
    # action). Same principle as publishing below — the recruiter presses a real button for both
    # of the two consequential, hard-to-undo-cheaply actions in this flow.

    # Publishing is deliberately NEVER routed here either — it only ever happens via the direct
    # POST /api/chat/{session_id}/publish endpoint the "Publish Job" button calls (see the
    # CONFIRM_PUBLISH prompt guidance: a chat confirmation gets acknowledged in the reply text,
    # but the graph itself takes no action).

    # Regeneration is deliberately NEVER auto-triggered by an edit that makes the JD stale
    # either — an edit just leaves jd_stale=true as a visible signal, nothing more; only an
    # explicit Regenerate click (also the direct /generate endpoint) clears it.
    return END
