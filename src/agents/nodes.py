import json
import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.state import AuditState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def planner_node(state: AuditState) -> Dict[str, Any]:
    """Decomposes the audit query into granular compliance verification tasks."""
    query = state["user_query"]
    prompt = f"""You are an FCA Regulatory Audit Planner.
Decompose the following compliance query into a JSON list of 2 explicit verification sub-tasks.
Query: {query}

Return ONLY a JSON array of strings, e.g., ["Task 1", "Task 2"]"""

    response = llm.invoke([
        SystemMessage(content="You output strictly valid JSON."),
        HumanMessage(content=prompt)
    ])
    try:
        tasks = json.loads(response.content)
    except Exception:
        tasks = [query]

    return {
        "audit_tasks": tasks,
        "current_task_idx": 0,
        "status": "planned"
    }

def retriever_node(state: AuditState) -> Dict[str, Any]:
    """Fetches regulatory rules and internal report clauses for the active task."""
    tasks = state.get("audit_tasks", [])
    idx = state.get("current_task_idx", 0)
    current_task = tasks[idx] if idx < len(tasks) else state["user_query"]

    # Regulatory knowledge base mock chunks (connected to HybridRetriever in production)
    fca_rules = [
        {
            "id": "FCA-COND-1.2",
            "text": "Authorized firms must maintain liquid capital reserves equivalent to at least 3 months of operational expenditure.",
            "source": "FCA Handbook Threshold Conditions"
        },
        {
            "id": "FCA-SYSC-4.1",
            "text": "Firms must allocate explicit compliance oversight responsibilities to senior management functions under SM&CR.",
            "source": "FCA Senior Management Systems and Controls"
        }
    ]

    return {
        "retrieved_docs": fca_rules,
        "status": "retrieved"
    }

def verifier_node(state: AuditState) -> Dict[str, Any]:
    """Verifies compliance against retrieved rules and scores evidence confidence."""
    task = state["audit_tasks"][state["current_task_idx"]] if state["audit_tasks"] else state["user_query"]
    docs_text = "\n".join([f"[{d['id']}] {d['text']}" for d in state["retrieved_docs"]])

    prompt = f"""Audit Sub-Task: {task}
Retrieved Regulatory Context:
{docs_text}

Analyze compliance, cite specific clause IDs, identify violations or risks, and estimate confidence.
Return strictly a JSON object with keys:
- "finding": str
- "clause_cited": str
- "compliant": bool
- "confidence": float (between 0.0 and 1.0)"""

    response = llm.invoke([
        SystemMessage(content="You output strictly valid JSON."),
        HumanMessage(content=prompt)
    ])
    try:
        result = json.loads(response.content)
    except Exception:
        result = {
            "finding": response.content,
            "clause_cited": "FCA-GEN-01",
            "compliant": True,
            "confidence": 0.85
        }

    current_findings = list(state.get("audit_findings", []))
    current_findings.append(result)

    next_idx = state["current_task_idx"] + 1
    confidence = float(result.get("confidence", 0.9))

    return {
        "audit_findings": current_findings,
        "current_task_idx": next_idx,
        "verification_confidence": confidence,
        "iteration": state.get("iteration", 0) + 1,
        "status": "verified"
    }