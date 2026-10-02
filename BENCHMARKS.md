# RegTech Audit Agent — Performance & Evaluation Benchmarks

## Evaluation Setup
- **Benchmark Suite**: 30 Curated FCA Regulatory Compliance Test Queries
- **Evaluation Framework**: RAGAS (Retrieval Augmented Generation Assessment)
- **Base LLM**: OpenAI GPT-4o-mini
- **Vector DB**: Qdrant (Dense HNSW + BM25 RRF Hybrid Retrieval)

## RAGAS Metric Summary

| Metric | Score | Target | Description |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | **0.94** | > 0.90 | Low hallucination rate; responses adhere strictly to retrieved FCA rules. |
| **Answer Relevancy** | **0.91** | > 0.85 | Findings directly address the sub-audit tasks without verbose filler. |
| **Context Precision** | **0.88** | > 0.80 | RRF search effectively places relevant FCA clauses at rank 1–2. |
| **Context Recall** | **0.92** | > 0.85 | Retrieves all necessary regulatory context clauses required for verification. |

## System Latency & Cost Metrics
- **Average Query Latency**: 2.14 seconds (Multi-agent loop with 2 sub-tasks)
- **Average Cost per Audit Query**: ~$0.0032 USD