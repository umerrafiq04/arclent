"""
EXISTING CODE — copied verbatim from backend/agent/state.py.

This is the LangGraph state schema — the object that flows through every node in the graph and
is what LangGraph's SqliteSaver checkpoints (persists) per session_id/thread_id. It is NOT the
same object as JobState (handoff/models.py) — job_state is one FIELD inside this larger
GraphState, alongside conversation history, phase, and several bookkeeping fields.

Depends only on langchain_core / langgraph — copy-paste-ready as-is.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Phase = Literal["collecting", "summary", "jd_selection", "publish_confirm", "published", "editing"]


class GraphState(TypedDict):
    # Full conversation history (HumanMessage/AIMessage). `add_messages` is a LangGraph reducer:
    # each node's partial update APPENDS to this list rather than replacing it.
    messages: Annotated[list[BaseMessage], add_messages]

    company_profile: dict  # hydrated once on turn 1 from the company_profile table (see load_context)
    recruiter_name: str | None  # first name of the signed-in recruiter, for natural personalization

    job_state: dict  # the JobState fields being collected — see handoff/models.py

    phase: Phase
    job_id: str | None  # set once this thread corresponds to an already-published job

    jd_versions: dict  # {"1": JobDescriptionDraft-shaped dict} once generated
    selected_version: str | None  # "1" once a JD exists
    jd_stale: bool  # true once job_state changed after jd_versions was generated

    missing_essential: list[str]  # hard-floor-blocking fields still missing, if any
    last_response: str  # the latest assistant reply text (mirrors ChatResponse.assistant_message)

    pending_analysis: dict | None  # transient handoff from analyze_turn -> apply_updates (one graph run)
    last_intent: str | None  # transient handoff from apply_updates -> route_after_apply
    jd_needs_refresh: bool  # transient: this turn's edit is what just made the JD stale

    asking_about_field: str | None  # optional JobState field the current AI question is about, if any
    suggested_options: list[str]  # quick-reply chip labels for the current AI question, if any
    options_multi_select: bool  # whether the recruiter can tap several suggested_options before sending

    skills_followup_count: int  # how many "add any other skills/responsibilities?" questions asked so far
    skipped_checklist_fields: list[str]  # standard-checklist fields explicitly skipped — never re-asked
