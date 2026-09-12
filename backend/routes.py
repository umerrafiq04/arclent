"""Consolidated FastAPI route handlers — auth, chat/job-creation, company profile, jobs
(recruiter + public), and admin, all in one file per an explicit directory-simplicity decision
(previously split across backend/routes/{auth,chat,company,jobs,admin}.py)."""

import csv
import io
import json
import logging
import sqlite3
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

from backend.agent.graph import (
    _apply_default_field_values,
    _apply_list_operation,
    _AUTO_GENERATE_RESPONSE,
    _CHECKLIST_QUESTIONS,
    _job_state_from_record,
    _next_checklist_prompt,
    _salary_question,
    _skill_profile_for_job_title,
    apply_field_changes,
    generate_jd,
    get_compiled_graph,
    hard_floor_met,
    publish_edit,
    publish_job,
    ready_to_generate,
    route_after_apply,
)
from backend.auth import (
    COOKIE_NAME,
    check_signin_rate_limit,
    clear_session_cookie,
    clear_signin_attempts,
    create_session,
    get_current_recruiter,
    get_current_user,
    hash_password,
    record_signin_attempt,
    require_admin,
    set_session_cookie,
    verify_login,
)
from backend.database import (
    create_application,
    create_chat_session,
    create_company_and_recruiter,
    delete_auth_session,
    delete_job,
    get_admin_job_by_id,
    get_chat_session_owner,
    get_company_profile_by_id,
    get_company_profile_by_name,
    get_job_by_session_id,
    get_published_job_by_job_id,
    get_user_by_email,
    get_user_by_id,
    list_all_jobs_admin,
    list_applications_for_job,
    list_jobs_for_company,
    list_published_jobs,
    save_refined_jd,
    set_accepting_applications,
    update_company_profile,
    upsert_job_draft,
)
from backend.document_extract import DocumentExtractError, extract_text
from backend.models import OPTIONAL_SKIPPABLE_FIELDS
from backend.schemas import (
    AcceptingApplicationsUpdate,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompanyProfileUpdate,
    JobApplicationRequest,
    JobIntakeRequest,
    JobStatePatch,
    SigninRequest,
    SignupRequest,
    UserResponse,
)


# ============================================================================
# AUTH (formerly routes/auth.py)
# ============================================================================

auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

MIN_PASSWORD_LENGTH = 8


def _user_response(user: dict) -> UserResponse:
    company_name = None
    if user.get("company_id") is not None:
        profile = get_company_profile_by_id(user["company_id"])
        company_name = profile["company_name"] if profile else None
    return UserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
        company_id=user.get("company_id"),
        company_name=company_name,
    )


@auth_router.post("/signup", response_model=UserResponse, status_code=201)
def signup(body: SignupRequest, response: Response) -> UserResponse:
    email = body.email.strip().lower()
    name = body.name.strip()
    company_name = body.company_name.strip()

    if len(body.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=422, detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if not name:
        raise HTTPException(status_code=422, detail="Name is required.")
    if not company_name:
        raise HTTPException(status_code=422, detail="Company name is required.")

    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="That email is already registered.")
    if get_company_profile_by_name(company_name):
        raise HTTPException(status_code=409, detail="That company name is already registered.")

    company_fields = body.model_dump(exclude={"name", "email", "password", "company_name"}, exclude_unset=True)
    password_hash = hash_password(body.password)
    try:
        created = create_company_and_recruiter(company_name, company_fields, email, password_hash, name)
    except sqlite3.IntegrityError:
        # Race with another signup for the same email/company_name between the checks above
        # and the insert — rare, but must still surface as a clean error, not a 500.
        raise HTTPException(status_code=409, detail="That email or company name is already registered.")

    user = created["user"]
    token = create_session(user["id"])
    set_session_cookie(response, token)
    return _user_response(user)


@auth_router.post("/signin", response_model=UserResponse)
def signin(body: SigninRequest, response: Response) -> UserResponse:
    email = body.email.strip().lower()
    check_signin_rate_limit(email)

    user = verify_login(email, body.password)
    if user is None:
        record_signin_attempt(email)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    clear_signin_attempts(email)
    token = create_session(user["id"])
    set_session_cookie(response, token)
    return _user_response(user)


@auth_router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        delete_auth_session(token)
    clear_session_cookie(response)


@auth_router.get("/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)) -> UserResponse:
    return _user_response(user)


# ============================================================================
# CHAT / JOB CREATION + EDITING (formerly routes/chat.py)
# ============================================================================

chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


def _resolve_and_authorize_session(session_id: str | None, user: dict) -> str:
    """Establishes/verifies ownership BEFORE any graph turn runs — a company-hopping
    recruiter shouldn't even trigger an LLM call against another company's conversation.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        create_chat_session(session_id, user["id"], user["company_id"])
        return session_id
    owner = get_chat_session_owner(session_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="No conversation found for this session_id")
    if owner["company_id"] != user["company_id"]:
        raise HTTPException(status_code=403, detail="You don't have access to this conversation.")
    return session_id

# Maps a REAL predicted next graph node (via the actual route_after_apply function — never a
# separate guess that could drift from what really happens) to an honest status label. Only
# emitted when that node is genuinely about to run; nothing here is decorative.
ROUTE_STATUS_LABELS = {
    "generate_jd": "Creating your job description...",
    "refine_jd": "Updating the selected job description...",
    "publish_job": "Publishing your job...",
    "publish_edit": "Publishing your changes...",
}

# UX-only indicator (never a gate on the conversation) — see Section 38B of the spec.
_COMPLETENESS_FIELDS = (
    "job_title", "job_category", "experience", "location", "work_mode",
    "employment_type", "education", "salary",
)

# Fixed display order for the draft panel's "add these details" alert — COMPANY_OVERRIDE_FIELDS
# itself is a set (membership checks elsewhere), so this keeps missing_company_fields stable
# across responses instead of set-iteration order, which Python doesn't guarantee.
_COMPANY_ALERT_FIELDS_ORDER = (
    "company_overview", "company_culture", "benefits", "work_life_balance", "why_join_us",
)


def _completeness_pct(job_state: dict) -> int:
    total = len(_COMPLETENESS_FIELDS) + 2  # + required_skills, responsibilities
    filled = sum(1 for f in _COMPLETENESS_FIELDS if job_state.get(f))
    filled += 1 if job_state.get("required_skills") else 0
    filled += 1 if job_state.get("responsibilities") else 0
    return round(100 * filled / total)


def _to_chat_messages(raw_messages: list, selected_version: str | None) -> list[ChatMessage]:
    chat_messages = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            chat_messages.append(ChatMessage(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            jd_document = msg.additional_kwargs.get("jd_document")
            jd_version = msg.additional_kwargs.get("jd_version")
            chat_messages.append(
                ChatMessage(
                    role="assistant",
                    content=msg.content,
                    jd_document=jd_document,
                    jd_version=jd_version,
                    is_selected=bool(jd_document) and jd_version == selected_version,
                )
            )
    return chat_messages


def _to_response(session_id: str, state: dict) -> ChatResponse:
    job_state = state.get("job_state") or {}
    job_record = get_job_by_session_id(session_id)
    company_profile = state.get("company_profile") or {}
    missing_company_fields = [f for f in _COMPANY_ALERT_FIELDS_ORDER if not company_profile.get(f)]
    return ChatResponse(
        session_id=session_id,
        phase=state.get("phase", "collecting"),
        assistant_message=state.get("last_response", ""),
        messages=_to_chat_messages(state.get("messages", []), state.get("selected_version")),
        job_state=job_state,
        completeness_pct=_completeness_pct(job_state),
        missing_essential=state.get("missing_essential", []),
        jd_versions=state.get("jd_versions") or None,
        selected_version=state.get("selected_version"),
        jd_stale=state.get("jd_stale", False),
        job_record=job_record,
        asking_about_field=state.get("asking_about_field"),
        suggested_options=state.get("suggested_options") or [],
        options_multi_select=bool(state.get("options_multi_select")),
        ready_to_generate=ready_to_generate(state),
        missing_company_fields=missing_company_fields,
    )


def _authorize_session(session_id: str, user: dict) -> None:
    owner = get_chat_session_owner(session_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="No conversation found for this session_id")
    if owner["company_id"] != user["company_id"]:
        raise HTTPException(status_code=403, detail="You don't have access to this conversation.")


def _sse(event_type: str, payload: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


def stream_graph_turn(session_id: str, human_message: HumanMessage, initial_label: str, user: dict):
    """Runs one graph turn via LangGraph's incremental .stream() (not .invoke()), emitting an
    honest SSE status event only when a real, identifiable step is genuinely about to run —
    the "next node" prediction reuses the actual route_after_apply function, so a shown label
    can never drift from what the graph really does next. Ends with a "result" event carrying
    the same ChatResponse shape the JSON endpoints have always returned.

    company_id/user_id ride along in "configurable" purely for load_context to hydrate
    company_profile on turn 1 of a thread (it's already in checkpointed state afterward) —
    ownership itself was already verified by the caller before this function is ever invoked.
    """
    graph = get_compiled_graph()
    config = {
        "configurable": {
            "thread_id": session_id,
            "company_id": user["company_id"],
            "user_id": user["id"],
        }
    }

    yield _sse("status", {"label": initial_label})

    try:
        prior_state = dict(graph.get_state(config).values)
    except Exception:
        prior_state = {}
    had_company_profile = bool(prior_state.get("company_profile"))
    accumulated = dict(prior_state)

    try:
        for update in graph.stream({"messages": [human_message]}, config, stream_mode="updates"):
            for node_name, node_output in update.items():
                if not isinstance(node_output, dict):
                    continue
                accumulated.update(node_output)
                if node_name == "load_context" and not had_company_profile and accumulated.get("company_profile"):
                    yield _sse("status", {"label": "Reviewing your company profile..."})
                elif node_name == "apply_updates":
                    next_node = route_after_apply(accumulated)
                    label = ROUTE_STATUS_LABELS.get(next_node)
                    if label:
                        yield _sse("status", {"label": label})
    except RuntimeError as exc:
        # e.g. MISTRAL_API_KEY missing/misconfigured
        yield _sse("error", {"detail": str(exc), "status_code": 503})
        return
    except Exception:
        # Nothing from this turn was committed, so prior progress is safe.
        logger.exception("stream_graph_turn: unexpected error running graph turn")
        yield _sse(
            "error",
            {
                "detail": "Something went wrong processing that message. Your previous progress is safe — please try again.",
                "status_code": 500,
            },
        )
        return

    final_state = graph.get_state(config).values
    response = _to_response(session_id, final_state)
    yield _sse("result", response.model_dump())


@chat_router.post("")
def post_chat(body: ChatRequest, user: dict = Depends(get_current_recruiter)) -> StreamingResponse:
    session_id = _resolve_and_authorize_session(body.session_id, user)
    return StreamingResponse(
        stream_graph_turn(session_id, HumanMessage(content=body.message), "Understanding your request...", user),
        media_type="text/event-stream",
    )


@chat_router.post("/upload")
async def post_chat_upload(
    session_id: str | None = Form(None),
    message: str = Form(""),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_recruiter),
) -> StreamingResponse:
    session_id = _resolve_and_authorize_session(session_id, user)
    content = await file.read()

    try:
        extracted = extract_text(file.filename or "upload", content)
    except DocumentExtractError as exc:
        # A clean, immediate JSON error — no graph turn was started, nothing to stream.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    combined = (
        f"[Uploaded document: {file.filename}]\n{extracted}\n\n"
        f"Recruiter's message: {message.strip() or '(none)'}"
    )
    return StreamingResponse(
        stream_graph_turn(session_id, HumanMessage(content=combined), "Reading the uploaded document...", user),
        media_type="text/event-stream",
    )


@chat_router.post("/intake", response_model=ChatResponse)
def post_job_intake(body: JobIntakeRequest, user: dict = Depends(get_current_recruiter)) -> ChatResponse:
    """The single combined call behind the local "Post a Job" pre-flow (job title -> location ->
    salary -> additional details, all collected client-side with zero LLM involvement — see
    frontend/js/job-modal.js's local intake flow). Always mints a brand-new session, deterministically
    fills required_skills/preferred_skills/responsibilities/job_category from
    _skill_profile_for_job_title (a hardcoded, job-title-keyword lookup — never an LLM call), then
    makes exactly ONE LLM call: generate_jd. This is the only path where a fresh job goes from zero
    to a generated description in a single request; the existing chat-based flow (POST /api/chat,
    still reachable for continuing/editing a job) is completely untouched.
    """
    session_id = _resolve_and_authorize_session(None, user)

    profile = _skill_profile_for_job_title(body.job_title)
    job_state = apply_field_changes(
        {},
        field_updates={
            "job_title": body.job_title,
            "job_category": profile.get("job_category"),
            "location": body.location,
            "salary": body.salary,
            "additional_information": body.additional_information,
        },
        list_operations=[
            {"field": "required_skills", "operation": "ADD", "values": profile["required_skills"]},
            {"field": "preferred_skills", "operation": "ADD", "values": profile["preferred_skills"]},
            {"field": "responsibilities", "operation": "ADD", "values": profile["responsibilities"]},
        ],
    )
    job_state = _apply_default_field_values(job_state)

    if not hard_floor_met(job_state):
        raise HTTPException(status_code=400, detail="Job title, location, and salary are all required.")

    company_profile = get_company_profile_by_id(user["company_id"]) or {}
    recruiter_user = get_user_by_id(user["id"])
    recruiter_name = recruiter_user["name"].strip().split(" ")[0] if recruiter_user and recruiter_user.get("name") else None

    # Synthesized, fully deterministic transcript (zero LLM) so reopening this draft later shows a
    # faithful history — mirrors exactly what a real chat conversation asking these same questions
    # would have produced.
    salary_question = _salary_question({"location": body.location})
    seed_messages = [
        HumanMessage(content=body.job_title),
        AIMessage(content=_CHECKLIST_QUESTIONS["location"]),
        HumanMessage(content=body.location),
        AIMessage(content=salary_question),
        HumanMessage(content=body.salary),
        AIMessage(content="Anything else you'd like to add? (optional — company culture, specific requirements, etc.)"),
        HumanMessage(content=body.additional_information or "Skip"),
    ]

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id, "company_id": user["company_id"], "user_id": user["id"]}}
    seed = {
        "company_profile": company_profile,
        "job_state": job_state,
        "phase": "collecting",
        "job_id": None,
        "jd_versions": {},
        "selected_version": None,
        "jd_stale": False,
        "missing_essential": [],
        "asking_about_field": None,
        "suggested_options": [],
        "options_multi_select": False,
        "skipped_checklist_fields": [],
        "messages": seed_messages,
    }
    if recruiter_name:
        seed["recruiter_name"] = recruiter_name
    graph.update_state(config, seed)

    state = graph.get_state(config).values
    result = generate_jd(state, config)  # the one and only LLM call in this whole flow
    graph.update_state(config, result)

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)


@chat_router.get("/{session_id}", response_model=ChatResponse)
def get_chat(session_id: str, user: dict = Depends(get_current_recruiter)) -> ChatResponse:
    _authorize_session(session_id, user)

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    snapshot = graph.get_state(config)

    if snapshot.values.get("messages"):
        return _to_response(session_id, snapshot.values)

    # No checkpoint for this thread (e.g. a demo job seeded directly into the DB, never run
    # through the graph). If it's a published job, synthesize a read-only view straight from
    # the DB row — no graph invocation, no LLM call. The first real edit message will seed a
    # proper checkpoint via load_context's hydration.
    record = get_job_by_session_id(session_id)
    if record and record.get("status") == "published":
        job_state = _job_state_from_record(record)
        selected_version = record.get("selected_version") or "1"
        selected_jd = record.get("selected_jd")
        greeting = f"This job is already published as {record.get('job_id')}. What would you like to change?"
        synthetic_state = {
            "job_state": job_state,
            "phase": "published",
            "jd_stale": False,
            "missing_essential": [],
            "last_response": greeting,
            "messages": [AIMessage(content=greeting)],
        }
        if selected_jd:
            synthetic_state["jd_versions"] = {selected_version: selected_jd}
            synthetic_state["selected_version"] = selected_version
        return _to_response(session_id, synthetic_state)

    raise HTTPException(status_code=404, detail="No conversation found for this session_id")


@chat_router.patch("/{session_id}/job-state", response_model=ChatResponse)
def patch_job_state(session_id: str, body: JobStatePatch, user: dict = Depends(get_current_recruiter)) -> ChatResponse:
    """Direct, silent edit to the draft — no chat message, no LLM call, no bot reply. This is
    what the draft form's field edits (title, experience, salary, skills add/remove, company
    context overrides, etc.) call, reusing apply_field_changes so the exact same validation and
    list-operation semantics apply as when the bot makes an edit from a chat turn.
    """
    _authorize_session(session_id, user)

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id, "company_id": user["company_id"], "user_id": user["id"]}}
    state = graph.get_state(config).values
    if not state:
        raise HTTPException(status_code=404, detail="No conversation found for this session_id")

    original_job_state = state.get("job_state") or {}
    job_state = apply_field_changes(
        original_job_state,
        body.field_updates,
        [op.model_dump() for op in body.list_operations],
        body.company_overrides,
    )
    # ANY change (including custom_questions) counts for the phase transition below — an edit that
    # hasn't been pushed to the live published row yet must always re-enable "Publish Edit", or a
    # recruiter who only added a screening question would see the button stuck on the disabled
    # "Published ✓" state with no way to actually publish their change (a real, reported bug).
    content_changed = job_state != original_job_state
    # jd_stale is a narrower, JD-specific signal — custom_questions never appears anywhere in the
    # generated JD (see JobState's comment), so a change to it alone shouldn't flag the JD as out of
    # date, same reasoning as jd_text_updates/jd_list_operations below not marking stale for a
    # direct fix to the drafted content itself.
    content_changed_for_jd = (
        {k: v for k, v in job_state.items() if k != "custom_questions"}
        != {k: v for k, v in original_job_state.items() if k != "custom_questions"}
    )
    jd_versions_exist = bool(state.get("jd_versions"))
    jd_stale = (jd_versions_exist and content_changed_for_jd) or state.get("jd_stale", False)

    phase = state.get("phase", "collecting")
    if phase == "published" and content_changed:
        phase = "editing"

    update = {"job_state": job_state, "jd_stale": jd_stale, "phase": phase}

    # Hand-editing the current draft's own text/list fields directly (e.g. the summary, about the
    # role, stand-out list) — same principle as job_state fields: no LLM refinement call needed
    # for a literal edit. Only overwrites keys that already exist on the draft (defense-in-depth
    # allowlist), and doesn't mark it stale — a direct fix to the drafted content itself isn't
    # "out of date with job_state" the way an unrelated job_state field change is.
    selected_version = state.get("selected_version")
    jd_versions = state.get("jd_versions") or {}
    if (body.jd_text_updates or body.jd_list_operations) and selected_version and jd_versions.get(selected_version):
        jd = dict(jd_versions[selected_version])
        changed = False
        for key, value in body.jd_text_updates.items():
            if key in jd:
                jd[key] = value
                changed = True
        for op in body.jd_list_operations:
            if op.field in jd:
                jd[op.field] = _apply_list_operation(jd.get(op.field) or [], op.operation, op.values)
                changed = True
        if changed:
            new_jd_versions = dict(jd_versions)
            new_jd_versions[selected_version] = jd
            update["jd_versions"] = new_jd_versions
            save_refined_jd(session_id, selected_version, jd)

    graph.update_state(config, update)

    is_published_job = state.get("job_id") is not None
    if not is_published_job and job_state.get("job_title"):
        # Mirrors apply_updates' write-through guard — drafts only, a published job's edits stay
        # off the live row until the recruiter explicitly clicks Publish Edit.
        upsert_job_draft(session_id, user["company_id"], job_state, jd_stale, owner_user_id=user["id"])

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)


@chat_router.post("/{session_id}/generate", response_model=ChatResponse)
def generate_session(session_id: str, user: dict = Depends(get_current_recruiter)) -> ChatResponse:
    """The ONLY path that actually calls the model to WRITE the job description — a direct
    action the "Generate Full Description"/"Regenerate" button calls, never a side effect of a
    chat message (see the REQUEST_JD_GENERATION/FINISH_COLLECTING prompt guidance: the bot's job
    in chat is collecting details and saying when it's ready, never generating itself). Reuses
    the exact generate_jd node function the graph itself uses, just invoked directly — no fake
    "Please generate..." user message is added to the transcript, this doesn't go through
    analyze_turn at all.
    """
    _authorize_session(session_id, user)

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id, "company_id": user["company_id"], "user_id": user["id"]}}
    state = graph.get_state(config).values
    if not state:
        raise HTTPException(status_code=404, detail="No conversation found for this session_id")

    if not ready_to_generate(state):
        job_state = state.get("job_state") or {}
        if not hard_floor_met(job_state):
            raise HTTPException(
                status_code=400,
                detail="Add at least a job title and one required skill or responsibility before generating.",
            )
        raise HTTPException(
            status_code=400,
            detail="A few more details are needed before generating — location, work mode, and salary.",
        )

    result = generate_jd(state, config)
    graph.update_state(config, result)

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)


@chat_router.post("/{session_id}/skip-field", response_model=ChatResponse)
def skip_field(session_id: str, user: dict = Depends(get_current_recruiter)) -> ChatResponse:
    """Direct, silent action the "Skip this" button calls. There's nothing for the recruiter to
    have "said", so unlike every other chat action this adds NO user message to the transcript
    at all — just a new bot message picking up the conversation. Skipping one of the four
    standard-checklist fields never needs an LLM call either: the next question is chosen
    deterministically via the exact same _next_checklist_prompt helper analyze_turn's
    skills-loop-cap override uses, so this is instant and immune to Mistral rate limits — which
    matters here specifically, since a 429 on a skip used to surface as a confusing "Sorry, I
    didn't catch that" for an action that isn't really a message at all.
    """
    _authorize_session(session_id, user)

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id, "company_id": user["company_id"], "user_id": user["id"]}}
    state = graph.get_state(config).values
    if not state:
        raise HTTPException(status_code=404, detail="No conversation found for this session_id")

    job_state = state.get("job_state") or {}
    currently_asking = state.get("asking_about_field")
    # Defense-in-depth: the UI never renders a "Skip this" button for a mandatory field (job_title/
    # location/salary — see OPTIONAL_SKIPPABLE_FIELDS) since asking_about_field itself never gets
    # set to one of those in the first place, but this direct endpoint is still reachable on its
    # own, so it must refuse rather than silently honor a skip that shouldn't be possible.
    if currently_asking and currently_asking not in OPTIONAL_SKIPPABLE_FIELDS:
        raise HTTPException(status_code=400, detail=f"{currently_asking} is required and can't be skipped.")

    # The field being skipped is whatever question is currently on screen — a skipped field's
    # job_state value stays empty by design, so without remembering it here _next_checklist_prompt
    # would just re-offer the exact same question forever instead of moving on.
    skipped = set(state.get("skipped_checklist_fields") or [])
    if currently_asking:
        skipped.add(currently_asking)

    next_field = _next_checklist_prompt(job_state, skipped)
    if next_field:
        question, field, chips = next_field
        response_text = f"No problem — skipping that. {question}"
        update = {
            "asking_about_field": field,
            "suggested_options": chips,
            "options_multi_select": False,
            "skipped_checklist_fields": list(skipped),
        }
        update["messages"] = [AIMessage(content=response_text)]
        update["last_response"] = response_text
        graph.update_state(config, update)
        final_state = graph.get_state(config).values
        return _to_response(session_id, final_state)

    # The checklist just became fully resolved via this skip — auto-generate immediately, same as
    # the chat path (see route_after_apply): no closing-check ceremony, no chip to click. This
    # direct, LLM-free endpoint has its own copy of that "what's next" decision, so it triggers
    # generation itself rather than relying on a graph turn that never runs here.
    update = {
        "asking_about_field": None,
        "suggested_options": [],
        "options_multi_select": False,
        "skipped_checklist_fields": list(skipped),
        "messages": [AIMessage(content=_AUTO_GENERATE_RESPONSE)],
        "last_response": _AUTO_GENERATE_RESPONSE,
    }
    graph.update_state(config, update)
    state = graph.get_state(config).values

    result = generate_jd(state, config)
    graph.update_state(config, result)

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)


@chat_router.post("/{session_id}/publish", response_model=ChatResponse)
def publish_session(session_id: str, user: dict = Depends(get_current_recruiter)) -> ChatResponse:
    """The ONLY path that actually publishes a job or a published-job edit — a direct action the
    "Publish Job"/"Publish Edit" button calls, never a side effect of a chat message (see the
    CONFIRM_PUBLISH prompt guidance in prompts.py). Reuses the exact same publish_job/publish_edit
    node functions the graph itself uses, just invoked directly instead of via routing, so the
    publish logic itself (Job ID allocation, DB writes) isn't duplicated anywhere.
    """
    _authorize_session(session_id, user)

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id, "company_id": user["company_id"], "user_id": user["id"]}}
    state = graph.get_state(config).values
    if not state:
        raise HTTPException(status_code=404, detail="No conversation found for this session_id")

    jd_versions = state.get("jd_versions") or {}
    selected_version = state.get("selected_version")
    if not (jd_versions and selected_version and jd_versions.get(selected_version)):
        raise HTTPException(status_code=400, detail="Generate a job description before publishing.")
    # Publishing is deliberately allowed even when jd_stale is true (an explicit founder decision):
    # once a JD exists, the recruiter can publish it as-is no matter how many further edits they've
    # made since — Regenerate is offered, never required. jd_stale itself still exists purely as an
    # informational signal (the chat prompt mentions it, see _jd_status_text in prompts.py), it's
    # just no longer a hard gate on this endpoint.

    is_published_job = state.get("job_id") is not None
    if is_published_job and state.get("phase") != "editing":
        raise HTTPException(status_code=400, detail="This job is already published.")

    node = publish_edit if is_published_job else publish_job
    result = node(state, config)
    graph.update_state(config, result)

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)


# ============================================================================
# COMPANY PROFILE (formerly routes/company.py)
# ============================================================================

company_router = APIRouter(prefix="/api/company-profile", tags=["company"])


@company_router.get("")
def get_company_profile(user: dict = Depends(get_current_recruiter)) -> dict:
    profile = get_company_profile_by_id(user["company_id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Company profile not found")
    return profile


@company_router.put("")
def put_company_profile(body: CompanyProfileUpdate, user: dict = Depends(get_current_recruiter)) -> dict:
    # company_id always comes from the authenticated session, never the request body —
    # there is nothing here for a client to tamper with to reach another company's row.
    updated = update_company_profile(user["company_id"], body.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update company profile")
    return updated


# ============================================================================
# JOBS — recruiter + public (formerly routes/jobs.py)
# ============================================================================

jobs_router = APIRouter(prefix="/api/jobs", tags=["jobs"])
jobs_public_router = APIRouter(prefix="/api/public/jobs", tags=["public-jobs"])


@jobs_router.get("")
def get_jobs_for_recruiter(user: dict = Depends(get_current_recruiter)) -> list[dict]:
    """Jobs (draft + published) for the authenticated recruiter's own company only."""
    return list_jobs_for_company(user["company_id"])


@jobs_router.get("/report")
def get_jobs_report(user: dict = Depends(get_current_recruiter)) -> Response:
    """CSV export of this company's own jobs — real fields only (title, status, location,
    employment type, posted date, accepting-applications). No proposal/hire counts: there is
    no applications-tracking table in this schema yet, so those numbers don't exist to export.
    """
    jobs = list_jobs_for_company(user["company_id"])
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["Job ID", "Title", "Status", "Employment Type", "Location", "Work Mode", "Posted Date", "Accepting Applications"]
    )
    for job in jobs:
        writer.writerow(
            [
                job.get("job_id") or "",
                job.get("job_title") or "",
                job.get("status") or "",
                job.get("employment_type") or "",
                job.get("location") or "",
                job.get("work_mode") or "",
                job.get("published_at") or job.get("created_at") or "",
                "Yes" if job.get("accepting_applications") else "No",
            ]
        )
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=arclent-jobs-report.csv"},
    )


@jobs_router.put("/{session_id}/accepting-applications")
def put_accepting_applications(
    session_id: str,
    body: AcceptingApplicationsUpdate,
    user: dict = Depends(get_current_recruiter),
) -> dict:
    """Closes/reopens a published job to new applicants — the job stays published and stays
    listed publicly either way; this only toggles whether it's still accepting applicants.
    """
    job = get_job_by_session_id(session_id)
    if not job or job["company_id"] != user["company_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "published":
        raise HTTPException(status_code=400, detail="Only a published job can be opened or closed to applications.")
    return set_accepting_applications(session_id, body.accepting_applications)


@jobs_router.delete("/{session_id}")
def delete_job_route(session_id: str, user: dict = Depends(get_current_recruiter)) -> dict:
    """Permanently removes a job (draft or published) belonging to the recruiter's own company."""
    job = get_job_by_session_id(session_id)
    if not job or job["company_id"] != user["company_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    delete_job(session_id)
    return {"deleted": True}


@jobs_public_router.get("")
def get_public_jobs() -> list[dict]:
    """Published jobs only, across all companies — Page 3's public listing."""
    return list_published_jobs()


@jobs_public_router.get("/{job_id}")
def get_public_job(job_id: str) -> dict:
    job = get_published_job_by_job_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@jobs_public_router.post("/{job_id}/apply")
def apply_to_job(job_id: str, body: JobApplicationRequest) -> dict:
    """A candidate's submission on the public Apply form — no auth, anyone can apply."""
    job = get_published_job_by_job_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("accepting_applications"):
        raise HTTPException(status_code=400, detail="This job is no longer accepting applications.")
    # Only keep answers for questions that actually exist on this job right now — the recruiter may
    # have edited custom_questions since the candidate loaded the page, so the submitted keys aren't
    # trusted as-is.
    current_questions = set(job.get("custom_questions") or [])
    answers = {q: a for q, a in body.answers.items() if q in current_questions}
    return create_application(job_id, answers)


@jobs_router.get("/{session_id}/applications")
def get_applications_for_job(session_id: str, user: dict = Depends(get_current_recruiter)) -> list[dict]:
    """Submitted applications for one of the recruiter's own jobs, newest first."""
    job = get_job_by_session_id(session_id)
    if not job or job["company_id"] != user["company_id"]:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.get("job_id"):
        return []
    return list_applications_for_job(job["job_id"])


# ============================================================================
# ADMIN (formerly routes/admin.py)
# ============================================================================

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


@admin_router.get("/jobs")
def get_admin_jobs(admin: dict = Depends(require_admin)) -> list[dict]:
    """All jobs, all companies, all statuses — Page 4's admin table."""
    return list_all_jobs_admin()


@admin_router.get("/jobs/{job_id}")
def get_admin_job(job_id: int, admin: dict = Depends(require_admin)) -> dict:
    job = get_admin_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@admin_router.get("/companies/{company_id}")
def get_admin_company(company_id: int, admin: dict = Depends(require_admin)) -> dict:
    """Company profile + every job posted by that company — the admin drilldown view."""
    profile = get_company_profile_by_id(company_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Company not found")
    return {"company": profile, "jobs": list_jobs_for_company(company_id)}
