import json
import asyncio
import logging
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAGAS >= 0.2 namespace imports
try:
    from ragas.metrics.collections import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except ImportError:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

from src.agents.graph import build_audit_graph

logger = logging.getLogger(__name__)


def get_eval_models():
    """Initializes LLM and Embeddings with fallback to local Ollama on quota failure."""
    primary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    primary_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        fallback_llm = ChatOllama(model="llama3.2", temperature=0.0)
        fallback_embeddings = OllamaEmbeddings(model="llama3.2")

        eval_llm = primary_llm.with_fallbacks([fallback_llm])
        return eval_llm, primary_embeddings, fallback_embeddings
    except ImportError:
        logger.warning("langchain-ollama not installed. Using OpenAI defaults without fallback.")
        return primary_llm, primary_embeddings, None


async def generate_agent_answers(test_cases: list) -> list:
    """Executes the agent graph for each test case to capture actual system responses."""
    audit_graph = build_audit_graph()
    eval_samples = []

    for item in test_cases:
        initial_state = {
            "user_query": item["question"],
            "audit_tasks": [],
            "current_task_idx": 0,
            "retrieved_docs": [],
            "audit_findings": [],
            "verification_confidence": 1.0,
            "iteration": 0,
            "status": "initiated"
        }

        final_state = await audit_graph.ainvoke(initial_state)

        findings_text = " ".join([f["finding"] for f in final_state.get("audit_findings", [])])
        retrieved_contexts = [d["text"] for d in final_state.get("retrieved_docs", [])]

        eval_samples.append({
            "question": item["question"],
            "answer": findings_text if findings_text else "No compliance findings generated.",
            "contexts": retrieved_contexts if retrieved_contexts else item["contexts"],
            "ground_truth": item["ground_truth"]
        })

    return eval_samples


def run_ragas_evaluation(eval_samples: list) -> pd.DataFrame:
    """Computes RAGAS metrics across generated samples."""
    dataset_dict = {
        "question": [s["question"] for s in eval_samples],
        "answer": [s["answer"] for s in eval_samples],
        "contexts": [s["contexts"] for s in eval_samples],
        "ground_truth": [s["ground_truth"] for s in eval_samples],
    }
    eval_dataset = Dataset.from_dict(dataset_dict)

    eval_llm, eval_embeddings, fallback_embeddings = get_eval_models()

    try:
        results = evaluate(
            dataset=eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=eval_llm,
            embeddings=eval_embeddings
        )
    except Exception as e:
        logger.warning(f"OpenAI Embeddings failed ({e}). Falling back to local Ollama embeddings...")
        if fallback_embeddings is not None:
            results = evaluate(
                dataset=eval_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=eval_llm,
                embeddings=fallback_embeddings
            )
        else:
            raise e

    return results.to_pandas()


async def main():
    print("Loading test dataset...")
    with open("src/eval/test_dataset.json", "r") as f:
        test_cases = json.load(f)

    print("Running multi-agent audit graph on test benchmark...")
    eval_samples = await generate_agent_answers(test_cases)

    print("Computing RAGAS evaluation metrics...")
    results_df = run_ragas_evaluation(eval_samples)

    print("\n================ EVALUATION RESULTS ================")
    metrics_summary = results_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
    print(metrics_summary.to_string())
    print("====================================================")

    results_df.to_csv("src/eval/benchmark_results.csv", index=False)
    print("Detailed report saved to 'src/eval/benchmark_results.csv'.")


if __name__ == "__main__":
    asyncio.run(main())