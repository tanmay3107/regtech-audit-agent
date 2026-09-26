from typing import TypedDict, List, Dict, Any

class AuditState(TypedDict):
    user_query: str
    audit_tasks: List[str]
    current_task_idx: int
    retrieved_docs: List[Dict[str, Any]]
    audit_findings: List[Dict[str, Any]]
    verification_confidence: float
    iteration: int
    status: str