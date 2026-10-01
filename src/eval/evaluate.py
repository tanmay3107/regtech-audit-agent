import json
import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.agents.graph import build_audit_graph

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

        # Extract generated findings and retrieved context
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
    # Convert list of dicts to Hugging Face Dataset format
    dataset_dict = {
        "question": [s["question"] for s in eval_samples],
        "answer": [s["answer"] for s in eval_samples],
        "contexts": [s["contexts"] for s in eval_samples],
        "ground_truth": [s["ground_truth"] for s in eval_samples],
    }
    eval_dataset = Dataset.from_dict(dataset_dict)

    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Run RAGAS evaluation
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

    df = results.to_pandas()
    return df


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

    # Save detailed evaluation log
    results_df.to_csv("src/eval/benchmark_results.csv", index=False)
    print("Detailed report saved to 'src/eval/benchmark_results.csv'.")


if __name__ == "__main__":
    asyncio.run(main())