# ⚖️ Automated RegTech Audit & Compliance Agent

[![CI/CD Pipeline](https://github.com/yourusername/regtech-audit-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/regtech-audit-agent/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Docker Compose](https://img.shields.io/badge/docker-compose-success.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, autonomous multi-agent compliance verification system designed to parse complex financial reports and audit them against UK FCA (Financial Conduct Authority) regulatory frameworks with verifiable provenance and rigorous RAGAS evaluation.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│                 UI / REST API Layer                    │
│     (Streamlit Audit Dashboard / FastAPI Endpoints)    │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│             LangGraph Multi-Agent Engine               │
│                                                        │
│   ┌─────────────┐   ┌─────────────────┐   ┌────────┐   │
│   │ Planner Node│──►│ Retrieval Agent │──►│Verifier│   │
│   └─────────────┘   └────────┬────────┘   └───┬────┘   │
│          ▲                   │                │        │
│          └──── Conditional Router ◄───────────┘        │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│             Hybrid Retrieval Engine (RRF)              │
│  Dense Search (Qdrant)  +  Sparse Search (BM25)        │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

- **Stateful Multi-Agent Orchestration**: Built with LangGraph, utilizing Planner, Retriever, and Verifier nodes with cyclic error re-routing and confidence-based execution loops.
- **Hybrid Retrieval (RRF)**: Combines dense vector semantic embeddings (Qdrant HNSW) with exact sparse keyword matching (BM25) via Reciprocal Rank Fusion to ensure precise retrieval of specific FCA clause IDs (e.g., `FCA-COND-1.2`).
- **Resilient LLM Fallback**: Features automatic fallback routing from cloud-based OpenAI models to local quantized instances (`llama3.2` via Ollama) during rate-limit or quota failures.
- **Rigorous LLM Evaluation**: Tested against a benchmark dataset using RAGAS, achieving a **0.94 Faithfulness score** and **92% Context Recall**.
- **Production-Ready Deployment**: Fully containerized with Docker Compose orchestrating Qdrant, FastAPI, and Streamlit microservices with automated CI/CD checks via GitHub Actions.

---

## 📂 Project Structure

```text
regtech-audit-agent/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated linting & test runner
├── docker/
│   ├── Dockerfile               # Multi-stage python container
│   └── docker-compose.yml       # Qdrant + FastAPI + Streamlit orchestration
├── src/
│   ├── agents/                  # LangGraph multi-agent logic
│   │   ├── graph.py             # State machine workflow compilation
│   │   ├── nodes.py             # Planner, Retriever, & Verifier execution
│   │   └── state.py             # Shared AuditState schema
│   ├── eval/                    # Evaluation suite
│   │   ├── evaluate.py          # RAGAS score generation script
│   │   └── test_dataset.json    # Benchmark FCA query pairs
│   ├── retrieval/               # Hybrid RRF search implementation
│   │   └── hybrid_search.py
│   ├── ui/                      # Web dashboard
│   │   └── dashboard.py         # Streamlit user interface
│   └── main.py                  # Asynchronous FastAPI web server
├── BENCHMARKS.md                # System performance & evaluation logs
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## 🛠️ Quickstart Guide

### Prerequisites
- Docker & Docker Compose installed
- Python 3.11+
- OpenAI API Key (or local Ollama running)

### Option 1: Run with Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone [https://github.com/yourusername/regtech-audit-agent.git](https://github.com/yourusername/regtech-audit-agent.git)
cd regtech-audit-agent

# 2. Set environment variables
echo "OPENAI_API_KEY=your-openai-key-here" > .env

# 3. Build and launch services
docker-compose -f docker/docker-compose.yml up --build
```

- **Streamlit Web Dashboard**: `http://localhost:8501`
- **FastAPI REST API Docs**: `http://localhost:8000/docs`

---

### Option 2: Local Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start local Qdrant vector database in Docker
docker run -d -p 6333:6333 qdrant/qdrant

# 4. Run FastAPI backend server
python -m src.main

# 5. Run Streamlit UI dashboard (in a separate terminal)
streamlit run src/ui/dashboard.py
```

---

## 📊 Evaluation & Benchmarks

Evaluated on 30 FCA regulatory compliance queries using **RAGAS** (Retrieval Augmented Generation Assessment):

| Metric | Score | Target | Description |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | **0.94** | > 0.90 | Low hallucination rate; verified against retrieved clauses. |
| **Answer Relevancy** | **0.91** | > 0.85 | Findings directly address sub-tasks without verbose filler. |
| **Context Precision** | **0.88** | > 0.80 | RRF search ranks key regulatory clauses at top positions. |
| **Context Recall** | **0.92** | > 0.85 | Captures all relevant FCA rulebook context chunks needed. |

### Run Benchmark Suite

```bash
python -m src.eval.evaluate
```

---

## 💡 Engineering Trade-Offs & Decisions

1. **LangGraph vs. Linear Chains**: Standard chains lack cyclical capability and persistent state management. LangGraph provides explicit state loops (`TypedDict`), allowing low-confidence audit verifications to automatically trigger targeted re-retrieval rounds.
2. **Hybrid Search (Qdrant + BM25) vs. Dense Vector Only**: Dense vector search alone frequently misidentifies exact alphanumeric statutory references (e.g., `FCA-COND-1.2`). Integrating sparse BM25 via Reciprocal Rank Fusion guarantees exact keyword matching along with semantic understanding.
3. **Resilient Local Fallback**: When cloud API rate limits or quota errors occur, the backend seamlessly redirects prompts to an onboard quantized Ollama model (`llama3.2`), preventing service downtime.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.