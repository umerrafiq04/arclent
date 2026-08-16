import json
import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

from backend.agent.graph import get_compiled_graph
from backend.agent.nodes import _job_state_from_record, route_after_apply
from backend.auth import get_current_recruiter
from backend.database import create_chat_session, get_chat_session_owner, get_job_by_session_id
from backend.document_extract import DocumentExtractError, extract_text
from backend.schemas import ChatMessage, ChatRequest, ChatResponse

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
    )


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
    owner = get_chat_session_owner(session_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="No conversation found for this session_id")
    if owner["company_id"] != user["company_id"]:
        raise HTTPException(status_code=403, detail="You don't have access to this conversation.")

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
