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
    JDGenerationOutput,
    JDRefinementOutput,
    TurnAnalysis,
)

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = (
    "Sorry, I didn't quite catch that — could you rephrase, or tell me a bit more "
    "about the role you're hiring for?"
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
        selected_version=state.get("selected_version"),
    )
    messages = [SystemMessage(content=system_prompt), *state["messages"]]

    try:
        analysis = call_structured(TurnAnalysis, messages, retries=1)
    except RuntimeError:
        # Configuration error (e.g. missing MISTRAL_API_KEY) — not a parsing failure,
        # let it propagate so the API layer can return a clear 503 instead of a
        # misleading "please rephrase" message.
        raise
    except Exception:
        logger.exception("analyze_turn: structured output failed after retry")
        fallback = TurnAnalysis(
            intent=Intent.CHITCHAT_OR_UNCLEAR,
            enough_information=False,
            response=FALLBACK_RESPONSE,
        )
        return {
            "pending_analysis": fallback.model_dump(mode="json"),
            "messages": [AIMessage(content=fallback.response)],
        }

    clean_response = _strip_markdown(analysis.response)
    dumped = analysis.model_dump(mode="json")
    dumped["response"] = clean_response
    return {
        "pending_analysis": dumped,
        "messages": [AIMessage(content=clean_response)],
    }


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


def apply_updates(state: GraphState, config: RunnableConfig) -> dict:
    analysis = state.get("pending_analysis") or {}
    original_job_state = state.get("job_state") or {}
    job_state = dict(original_job_state)
    job_state.setdefault("required_skills", [])
    job_state.setdefault("preferred_skills", [])
    job_state.setdefault("responsibilities", [])
    job_state.setdefault("company_overrides", {})

    intent = analysis.get("intent")

    # Off-topic / document-review turns carry no confirmed job content — skip everything.
    # Advice turns may still state a real fact (e.g. the job title) alongside the question, so
    # scalar field_updates still apply — but the advice itself is always skill-shaped, so
    # list_operations/company_overrides are blocked until an explicit later confirmation.
    # Defense in depth: enforced here regardless of what the model actually returned.
    if intent not in _BLOCK_ALL_VALUES:
        for key, value in (analysis.get("field_updates") or {}).items():
            if key in SCALAR_JOB_FIELDS:
                job_state[key] = value

        if intent not in _BLOCK_LIST_AND_OVERRIDE_VALUES:
            for op in analysis.get("list_operations") or []:
                field = op.get("field")
                operation = op.get("operation")
                values = op.get("values") or []
                if field not in LIST_JOB_FIELDS:
                    continue
                job_state[field] = _apply_list_operation(job_state.get(field, []), operation, values)

            overrides = dict(job_state.get("company_overrides", {}))
            for key, value in (analysis.get("company_overrides") or {}).items():
                if key in COMPANY_OVERRIDE_FIELDS:
                    overrides[key] = value
            job_state["company_overrides"] = overrides

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

    # Same defense-in-depth as asking_about_field above — only ever surface chips on a turn
    # that's actually posing a question, regardless of what the model returned.
    suggested_options = analysis.get("suggested_options") or []
    if not reply_is_a_question:
        suggested_options = []

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
    }


def _jd_document_message(version: str, jd: dict) -> AIMessage:
    """A JD version rendered as its own chat message (content stays empty — the frontend
    renders the structured payload as a card, with a "Choose Version" action) rather than in
    the side panel, per the in-chat JD requirement.
    """
    return AIMessage(content="", additional_kwargs={"jd_document": jd, "jd_version": version})


def generate_jd(state: GraphState, config: RunnableConfig) -> dict:
    company_profile = state.get("company_profile") or {}
    job_state = state.get("job_state") or {}
    session_id = config["configurable"]["thread_id"]

    time.sleep(3)  # this is the 2nd Mistral call in the same turn — avoid bursting past per-second rate limits
    prompt = build_jd_generation_prompt(company_profile, job_state)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Generate two job description drafts based on the information above."),
    ]

    try:
        output = call_structured(JDGenerationOutput, messages, retries=1)
    except Exception:
        logger.exception("generate_jd: structured output failed after retry")
        response = (
            "I ran into a problem generating the job description just now — could you ask me to "
            "generate it again?"
        )
        return {"messages": [AIMessage(content=response)], "last_response": response}

    jd_versions = {
        "1": _strip_markdown(output.version_1.model_dump(mode="json")),
        "2": _strip_markdown(output.version_2.model_dump(mode="json")),
    }
    save_jd_versions(session_id, jd_versions)

    job_title = job_state.get("job_title") or "this role"
    response = (
        f"Based on the information we've collected, I've prepared two versions for {job_title}. "
        "Let me know which you'd prefer (\"I prefer 1\" or \"2\"), or how you'd like either one refined."
    )
    # Editing an already-published job stays in the "editing" phase throughout (that's what
    # gates publish_edit) rather than the fresh-draft "jd_selection" phase.
    next_phase = "editing" if state.get("job_id") else "jd_selection"
    return {
        "jd_versions": jd_versions,
        "jd_stale": False,
        "selected_version": None,
        "phase": next_phase,
        "messages": [
            AIMessage(content=response),
            _jd_document_message("1", jd_versions["1"]),
            _jd_document_message("2", jd_versions["2"]),
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
        output = call_structured(JDRefinementOutput, messages, retries=1)
    except Exception:
        logger.exception("refine_jd: structured output failed after retry")
        response = "I had trouble applying that change — could you rephrase what you'd like adjusted?"
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
    is_published_job = state.get("job_id") is not None

    if intent == Intent.REQUEST_JD_GENERATION.value:
        job_state = state.get("job_state") or {}
        if hard_floor_met(job_state) and not (state.get("missing_essential") or []):
            return "generate_jd"
        return END

    # The recruiter just gave a finish phrase ("that's all", etc.) and apply_updates already
    # moved phase to "summary" because the hard floor is met — analyze_turn's reply text
    # promises to summarize and generate right now (per the FINISH_COLLECTING prompt guidance),
    # so this must actually fire this turn rather than silently ending and leaving that promise
    # unfulfilled until the recruiter sends a separate "generate" message. apply_updates only
    # ever sets phase to "summary" from within the "collecting" branch, so this can't misfire on
    # an edit to an already-published job (that path stays in "editing"/"published" instead).
    if (
        intent == Intent.FINISH_COLLECTING.value
        and state.get("phase") == "summary"
        and not (state.get("missing_essential") or [])
    ):
        return "generate_jd"

    if intent == Intent.REQUEST_REFINEMENT.value and jd_versions and state.get("selected_version"):
        return "refine_jd"

    if intent == Intent.CONFIRM_PUBLISH.value and jd_versions and state.get("selected_version") and not state.get("jd_stale", False):
        if is_published_job and state.get("phase") == "editing":
            return "publish_edit"
        if not is_published_job and state.get("phase") != "published":
            return "publish_job"

    # An edit just made an existing JD stale — refresh it immediately rather than making the
    # recruiter ask again in a follow-up turn (analyze_turn's reply already says this is
    # happening now, so the behavior needs to match).
    if state.get("jd_needs_refresh") and jd_versions:
        return "generate_jd"

    return END
