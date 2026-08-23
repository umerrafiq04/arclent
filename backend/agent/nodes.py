import logging
import random
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
    previous_asking_about_field = state.get("asking_about_field")

    last_human_text = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            last_human_text = (m.content or "").strip()
            break

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
                        "options_multi_select": False,
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
                    "options_multi_select": False,
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
                        "options_multi_select": False,
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
    if field_query and not (analysis.field_updates or {}).get(field_query):
        current_value = (state.get("job_state") or {}).get(field_query)
        label = _FIELD_DISPLAY_LABELS.get(field_query, field_query)
        response_text = (
            f"The current {label} is {current_value}."
            if current_value
            else f"No {label} has been set yet — want to add one now?"
        )
        analysis = analysis.model_copy(update={"response": response_text})

    # Location-aware currency hint for the model's OWN first-time salary question — _salary_question
    # (used by the deterministic redirect paths below) only covers those specific override routes,
    # not the far more common case where the model asks its own salary question in its own words on
    # the normal path. Detected structurally: does the QUESTION portion of this turn's response
    # (never an acknowledgment clause mentioning salary in passing — same sentence-isolation as the
    # _KEYWORD_FIELD_HINTS sniffing above) actually pose a salary question, via the same keyword
    # matcher _detect_field_value_query uses above. Only appends when a currency is known for the
    # recruiter's own location AND the model didn't already include that symbol itself.
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
        if currency_symbol and currency_symbol not in (analysis.response or ""):
            analysis = analysis.model_copy(
                update={"response": f"{(analysis.response or '').rstrip()} (in {currency_symbol}, based on the location you gave)"}
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
                        "options_multi_select": False,
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
}
_FIELD_QUERY_SIGNAL_RE = re.compile(
    r"\bwhat(?:'s|s)?\b|\bremind me\b|\btell me\b|\bcurrent(?:ly)?\b|\bagain\b", re.IGNORECASE
)
_FIELD_DISPLAY_LABELS = {
    "salary": "salary",
    "location": "location",
    "job_title": "job title",
    "work_mode": "work mode",
    "employment_type": "employment type",
    "experience": "experience level",
    "deadline": "application deadline",
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
# meta line it doesn't have real data for rather than inventing one.
_CHECKLIST_ORDER = [
    "required_skills",
    "responsibilities",
    "location",
    "salary",
]
_CHECKLIST_QUESTIONS = {
    "required_skills": "What are the required skills a candidate should have for this role?",
    "responsibilities": "What will this person be responsible for day-to-day?",
    "location": 'Which city or region will this role be based in? You can also say "Worldwide" if it\'s fully remote.',
    "salary": "What's the salary range for this role, if you'd like to share one?",
}
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


def _currency_symbol_for_location(location: str | None) -> str | None:
    if not location:
        return None
    for pattern, symbol in _CURRENCY_BY_LOCATION_KEYWORDS:
        if pattern.search(location):
            return symbol
    return None


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
                options_multi_select=False,
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
    job_state.setdefault("custom_questions", [])
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
            if candidate_field in missing:
                continue
            if any(phrase in question_text_lower for phrase in phrases):
                asking_about_field = candidate_field
                raw_asking_about_field = candidate_field
                break

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

    # A dedicated "Skip this" button already renders in the UI whenever asking_about_field is set
    # (see appendSkipButton in job-modal.js) — a suggested_options chip whose ONLY content is the
    # same generic "Skip" fallback adds nothing but a confusing second, differently-styled skip
    # affordance stacked on the same question (reported live). Suppress it in that specific case;
    # any OTHER chip content (real answer options like work_mode's Remote/Hybrid/Onsite) still
    # renders normally alongside the dedicated button, since those aren't redundant with it.
    if asking_about_field and suggested_options == _GENERIC_FALLBACK_OPTIONS:
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
