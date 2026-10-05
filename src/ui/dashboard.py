import asyncio
import json
import streamlit as st
import pandas as pd
from src.agents.graph import build_audit_graph

# Page Configuration
st.set_page_config(
    page_title="FCA RegTech Audit Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish
st.markdown("""
<style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .badge-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_graph():
    """Compiles and caches the LangGraph state machine."""
    return build_audit_graph()


async def execute_audit(query: str):
    """Executes the agent graph asynchronously and returns the final state."""
    graph = get_graph()
    initial_state = {
        "user_query": query,
        "audit_tasks": [],
        "current_task_idx": 0,
        "retrieved_docs": [],
        "audit_findings": [],
        "verification_confidence": 1.0,
        "iteration": 0,
        "status": "initiated"
    }
    return await graph.ainvoke(initial_state)


# Header Section
st.title("⚖️ Automated RegTech Audit & Compliance Agent")
st.caption("Production-grade Multi-Agent Pipeline for FCA Regulatory Verification")

# Sidebar Controls
with st.sidebar:
    st.header("📋 Presets & Config")
    
    sample_queries = [
        "Audit whether our liquid capital reserves and SM&CR oversight assignment satisfy FCA rules.",
        "Check compliance regarding operational disruption reporting thresholds under FCA guidelines.",
        "Verify if our customer onboarding data retention policy complies with FCA and UK GDPR standards."
    ]
    
    selected_preset = st.selectbox("Load Sample Query:", ["Custom..."] + sample_queries)
    
    st.divider()
    st.markdown("### 🛠️ Architecture Stack")
    st.markdown("""
    - **Orchestration**: LangGraph State Machine
    - **LLM**: GPT-4o-mini *(Ollama Llama 3.2 Fallback)*
    - **Vector Index**: Qdrant (HNSW + BM25 RRF)
    - **Eval Engine**: RAGAS Evaluation Suite
    """)

# Main Query Input Box
if selected_preset != "Custom...":
    default_query = selected_preset
else:
    default_query = ""

user_query = st.text_area(
    "Enter Regulatory Compliance Audit Query:",
    value=default_query,
    height=100,
    placeholder="e.g., Audit our liquid capital reserves against FCA Handbook threshold conditions..."
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_button = st.button("🚀 Run Audit", type="primary", use_container_width=True)

# Main Execution & Output Section
if run_button and user_query.strip():
    with st.spinner("Executing Multi-Agent Audit Pipeline..."):
        try:
            # Run the asynchronous graph execution inside Streamlit
            final_state = asyncio.run(execute_audit(user_query.strip()))
            
            st.success("Audit Completed Successfully!")
            st.divider()
            
            # --- Executive Summary Metrics ---
            findings = final_state.get("audit_findings", [])
            total_tasks = len(final_state.get("audit_tasks", []))
            compliant_count = sum(1 for f in findings if f.get("compliant", False))
            avg_confidence = (
                sum(f.get("confidence", 0.0) for f in findings) / len(findings)
                if findings else 0.0
            )
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sub-Tasks Processed", total_tasks)
            m2.metric("Compliant Checks", f"{compliant_count}/{len(findings)}")
            m3.metric("Verification Confidence", f"{avg_confidence * 100:.1f}%")
            m4.metric("Agent Iterations", final_state.get("iteration", 0))
            
            st.divider()
            
            # --- Detailed Tabs for Execution Transparency ---
            tab_findings, tab_reasoning, tab_docs = st.tabs([
                "📊 Audit Findings", 
                "🤖 Agent Task Decomposition", 
                "📚 Context & Evidence Cites"
            ])
            
            # Tab 1: Structured Findings
            with tab_findings:
                st.subheader("Final Compliance Verification Report")
                for idx, finding in enumerate(findings, 1):
                    is_compliant = finding.get("compliant", True)
                    badge = (
                        '<span class="badge-pass">COMPLIANT</span>' 
                        if is_compliant 
                        else '<span class="badge-fail">NON-COMPLIANT</span>'
                    )
                    
                    with st.expander(f"Finding #{idx}: Clause [{finding.get('clause_cited', 'N/A')}]", expanded=True):
                        st.markdown(f"**Status**: {badge}", unsafe_allow_html=True)
                        st.markdown(f"**Finding**: {finding.get('finding', 'No details provided.')}")
                        st.progress(float(finding.get("confidence", 0.85)), text=f"Confidence Score: {finding.get('confidence', 0.85):.2f}")

            # Tab 2: Planner Output
            with tab_reasoning:
                st.subheader("Planner Node Execution")
                st.json({
                    "original_query": final_state.get("user_query"),
                    "generated_tasks": final_state.get("audit_tasks"),
                    "status": final_state.get("status")
                })

            # Tab 3: Retrieved Documents / Evidence
            with tab_docs:
                st.subheader("Retrieved Regulatory Context")
                retrieved_docs = final_state.get("retrieved_docs", [])
                if retrieved_docs:
                    df = pd.DataFrame(retrieved_docs)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No documents retrieved.")

            # --- Export Section ---
            st.divider()
            report_json = json.dumps(final_state, indent=2)
            st.download_button(
                label="📥 Download Structured Audit Report (JSON)",
                data=report_json,
                file_name="fca_audit_report.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Audit Execution Failed: {str(e)}")

elif run_button:
    st.warning("Please enter a valid audit query before proceeding.")