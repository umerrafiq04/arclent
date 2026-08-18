import json
import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

from backend.agent.graph import get_compiled_graph
from backend.agent.nodes import (
    _job_state_from_record,
    _next_checklist_prompt,
    _READY_TO_GENERATE_RESPONSE,
    apply_field_changes,
    generate_jd,
    publish_edit,
    publish_job,
    route_after_apply,
)
from backend.agent.sufficiency import hard_floor_met
from backend.auth import get_current_recruiter
from backend.database import (
    create_chat_session,
    get_chat_session_owner,
    get_job_by_session_id,
    save_refined_jd,
    upsert_job_draft,
)
from backend.document_extract import DocumentExtractError, extract_text
from backend.schemas import ChatMessage, ChatRequest, ChatResponse, JobStatePatch

router = APIRouter(prefix="/api/chat", tags=["chat"])


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


@router.post("")
def post_chat(body: ChatRequest, user: dict = Depends(get_current_recruiter)) -> StreamingResponse:
    session_id = _resolve_and_authorize_session(body.session_id, user)
    return StreamingResponse(
        stream_graph_turn(session_id, HumanMessage(content=body.message), "Understanding your request...", user),
        media_type="text/event-stream",
    )


@router.post("/upload")
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


@router.get("/{session_id}", response_model=ChatResponse)
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


@router.patch("/{session_id}/job-state", response_model=ChatResponse)
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
    content_changed = job_state != original_job_state
    jd_versions_exist = bool(state.get("jd_versions"))
    jd_stale = (jd_versions_exist and content_changed) or state.get("jd_stale", False)

    phase = state.get("phase", "collecting")
    if phase == "published" and content_changed:
        phase = "editing"

    update = {"job_state": job_state, "jd_stale": jd_stale, "phase": phase}

    # Hand-editing the current draft's own text (e.g. the summary) directly — same principle as
    # job_state fields: no LLM refinement call needed for a literal edit. Only overwrites keys
    # that already exist on the draft (defense-in-depth allowlist), and doesn't mark it stale —
    # a direct text fix isn't "out of date with job_state" the way an unrelated field change is.
    selected_version = state.get("selected_version")
    jd_versions = state.get("jd_versions") or {}
    if body.jd_text_updates and selected_version and jd_versions.get(selected_version):
        jd = dict(jd_versions[selected_version])
        changed = False
        for key, value in body.jd_text_updates.items():
            if key in jd:
                jd[key] = value
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


@router.post("/{session_id}/generate", response_model=ChatResponse)
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

    job_state = state.get("job_state") or {}
    if not hard_floor_met(job_state):
        raise HTTPException(
            status_code=400,
            detail="Add at least a job title and one required skill or responsibility before generating.",
        )

    result = generate_jd(state, config)
    graph.update_state(config, result)

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)


@router.post("/{session_id}/skip-field", response_model=ChatResponse)
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
    # The field being skipped is whatever question is currently on screen — a skipped field's
    # job_state value stays empty by design, so without remembering it here _next_checklist_prompt
    # would just re-offer the exact same question forever instead of moving on.
    skipped = set(state.get("skipped_checklist_fields") or [])
    currently_asking = state.get("asking_about_field")
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
        }
    else:
        response_text = _READY_TO_GENERATE_RESPONSE
        update = {
            "asking_about_field": None,
            "suggested_options": [],
            "options_multi_select": False,
        }
    update["skipped_checklist_fields"] = list(skipped)
    update["messages"] = [AIMessage(content=response_text)]
    update["last_response"] = response_text
    graph.update_state(config, update)

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)


@router.post("/{session_id}/publish", response_model=ChatResponse)
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
    if state.get("jd_stale"):
        raise HTTPException(status_code=400, detail="The job description is out of date — regenerate it before publishing.")

    is_published_job = state.get("job_id") is not None
    if is_published_job and state.get("phase") != "editing":
        raise HTTPException(status_code=400, detail="This job is already published.")

    node = publish_edit if is_published_job else publish_job
    result = node(state, config)
    graph.update_state(config, result)

    final_state = graph.get_state(config).values
    return _to_response(session_id, final_state)
