import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import (
    analyze_turn,
    apply_updates,
    generate_jd,
    load_context,
    publish_edit,
    publish_job,
    refine_jd,
    route_after_apply,
)
from backend.agent.state import GraphState
from backend.config import CHECKPOINT_DB_PATH

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
