from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

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
