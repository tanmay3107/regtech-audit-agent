from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agents.graph import build_audit_graph

load_dotenv()

app = FastAPI(title="RegTech FCA Compliance Audit Agent")
audit_graph = build_audit_graph()

class AuditRequest(BaseModel):
    query: str

class AuditResponse(BaseModel):
    user_query: str
    audit_findings: list
    total_iterations: int
    status: str

@app.post("/api/v1/audit", response_model=AuditResponse)
async def run_audit(request: AuditRequest):
    initial_state = {
        "user_query": request.query,
        "audit_tasks": [],
        "current_task_idx": 0,
        "retrieved_docs": [],
        "audit_findings": [],
        "verification_confidence": 1.0,
        "iteration": 0,
        "status": "initiated"
    }
    
    try:
        final_state = await audit_graph.ainvoke(initial_state)
        return AuditResponse(
            user_query=final_state["user_query"],
            audit_findings=final_state["audit_findings"],
            total_iterations=final_state["iteration"],
            status="completed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)