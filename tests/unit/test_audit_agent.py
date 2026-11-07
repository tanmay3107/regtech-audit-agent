import pytest
from unittest.mock import MagicMock
from src.agents.state import AuditState
from src.agents.nodes import planner_node, retriever_node, verifier_node
from src.agents.graph import build_audit_graph

@pytest.fixture
def base_state() -> AuditState:
    return {
        "user_query": "Audit liquid capital reserves against FCA Handbook thresholds.",
        "audit_tasks": [],
        "current_task_idx": 0,
        "retrieved_docs": [],
        "audit_findings": [],
        "verification_confidence": 1.0,
        "iteration": 0,
        "status": "initiated"
    }

def test_planner_node_decompounds_tasks(base_state, mocker):
    """Test that the planner node successfully parses tasks from LLM output."""
    mock_response = MagicMock()
    mock_response.content = '["Verify liquid capital reserves", "Check SM&CR senior management assignment"]'
    
    # Mock LLM invocation inside the node
    mocker.patch("src.agents.nodes.llm.invoke", return_value=mock_response)

    updated_state = planner_node(base_state)

    assert len(updated_state["audit_tasks"]) == 2
    assert updated_state["status"] == "planned"
    assert updated_state["current_task_idx"] == 0

def test_retriever_node_fetches_context(base_state):
    """Test that the retriever node populates regulatory context chunks."""
    updated_state = retriever_node(base_state)

    assert len(updated_state["retrieved_docs"]) > 0
    assert updated_state["retrieved_docs"][0]["id"] == "FCA-COND-1.2"
    assert updated_state["status"] == "retrieved"

def test_verifier_node_produces_finding(base_state, mocker):
    """Test that the verifier node evaluates compliance and outputs structured findings."""
    base_state["audit_tasks"] = ["Verify liquid capital reserves"]
    base_state["retrieved_docs"] = [{
        "id": "FCA-COND-1.2",
        "text": "Firms must maintain liquid capital reserves equivalent to 3 months opex.",
        "source": "FCA Handbook"
    }]

    mock_response = MagicMock()
    mock_response.content = '{"finding": "Adequate liquid reserves found.", "clause_cited": "FCA-COND-1.2", "compliant": true, "confidence": 0.95}'
    
    mocker.patch("src.agents.nodes.llm.invoke", return_value=mock_response)

    updated_state = verifier_node(base_state)

    assert len(updated_state["audit_findings"]) == 1
    assert updated_state["audit_findings"][0]["compliant"] is True
    assert updated_state["verification_confidence"] == 0.95
    assert updated_state["iteration"] == 1

def test_graph_compilation_and_execution():
    """Test that the LangGraph workflow compiles correctly and executes without error."""
    graph = build_audit_graph()
    assert graph is not None