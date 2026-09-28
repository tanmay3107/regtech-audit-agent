from langgraph.graph import StateGraph, START, END
from src.agents.state import AuditState
from src.agents.nodes import planner_node, retriever_node, verifier_node

def route_next_step(state: AuditState) -> str:
    """Routes execution conditionally back to retrieval or terminates."""
    # Low confidence triggers re-retrieval (up to 3 attempts)
    if state["verification_confidence"] < 0.70 and state["iteration"] < 3:
        return "retriever"
    
    # Process remaining sub-tasks
    if state["current_task_idx"] < len(state.get("audit_tasks", [])):
        return "retriever"
    
    return END

def build_audit_graph():
    builder = StateGraph(AuditState)

    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("verifier", verifier_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "verifier")

    builder.add_conditional_edges(
        "verifier",
        route_next_step,
        {
            "retriever": "retriever",
            END: END
        }
    )

    return builder.compile()