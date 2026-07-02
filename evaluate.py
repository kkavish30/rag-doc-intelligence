# evaluate.py
import os
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval import retrieve_and_rerank
from generation import generate_answer
from guardrail import check_relevance
from eval_dataset import EVAL_QA_PAIRS
from dotenv import load_dotenv

load_dotenv()

# RAGAS needs its own LLM and embeddings to compute metrics
ragas_llm = LangchainLLMWrapper(
    ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

# Metrics must be instantiated objects in ragas 0.4.x, not bare references
faithfulness_metric = Faithfulness(llm=ragas_llm)
answer_relevancy_metric = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_precision_metric = LLMContextPrecisionWithReference(llm=ragas_llm)


def run_pipeline_for_eval(question: str) -> dict:
    """
    Runs full RAG pipeline (without memory) and returns RAGAS-compatible output.
    """
    docs = retrieve_and_rerank(question)
    guard = check_relevance(docs)

    if not guard["pass"]:
        return {
            "answer": "I could not find relevant information in the documents.",
            "contexts": [],
            "guardrail_passed": False
        }

    result = generate_answer(question, docs, [])
    contexts = [doc["text"] for doc in docs]

    return {
        "answer": result["answer"],
        "contexts": contexts,
        "guardrail_passed": True
    }


def build_ragas_dataset(qa_pairs: list[dict]) -> tuple[Dataset, list[dict]]:
    """
    Runs the pipeline on each QA pair and builds a HuggingFace Dataset.
    Only includes pairs where ground_truth is not None (excludes out-of-scope
    queries, since RAGAS metrics require a ground truth to compare against).

    Returns (dataset, guardrail_log) where guardrail_log tracks pass/fail
    for ALL pairs including out-of-scope ones.
    """
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    guardrail_log = []
    total = len(qa_pairs)

    for i, pair in enumerate(qa_pairs):
        print(f"Running eval {i+1}/{total}: {pair['question'][:60]}...")

        output = run_pipeline_for_eval(pair["question"])

        guardrail_log.append({
            "question": pair["question"],
            "expected_in_scope": pair["ground_truth"] is not None,
            "guardrail_passed": output["guardrail_passed"]
        })

        # Only feed in-scope QA pairs to RAGAS — it can't score against None
        if pair["ground_truth"] is not None:
            data["question"].append(pair["question"])
            data["answer"].append(output["answer"])
            data["contexts"].append(output["contexts"])
            data["ground_truth"].append(pair["ground_truth"])

        time.sleep(1)  # avoid rate limits

    return Dataset.from_dict(data), guardrail_log


def print_guardrail_report(guardrail_log: list[dict]):
    print("\n=== GUARDRAIL BEHAVIOR REPORT ===")
    correct = 0
    for entry in guardrail_log:
        expected = "SHOULD PASS" if entry["expected_in_scope"] else "SHOULD REJECT"
        actual = "PASSED" if entry["guardrail_passed"] else "REJECTED"
        is_correct = (entry["expected_in_scope"] == entry["guardrail_passed"])
        correct += is_correct
        marker = "OK" if is_correct else "XX"
        print(f"  {marker} [{expected:14}] -> [{actual:8}] | {entry['question'][:60]}")

    accuracy = correct / len(guardrail_log) * 100
    print(f"\nGuardrail accuracy: {correct}/{len(guardrail_log)} ({accuracy:.1f}%)")


def run_evaluation():
    print("Building evaluation dataset and running pipeline on all QA pairs...")
    dataset, guardrail_log = build_ragas_dataset(EVAL_QA_PAIRS)

    print_guardrail_report(guardrail_log)

    print(f"\nRunning RAGAS evaluation on {len(dataset)} in-scope QA pairs...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric, answer_relevancy_metric, context_precision_metric]
    )

    results_df = results.to_pandas()

    # Use .mean() on dataframe columns — ragas 0.4.3 returns per-question scores
    faithfulness_score = results_df['faithfulness'].mean()
    answer_relevancy_score = results_df['answer_relevancy'].mean()
    context_precision_score = results_df['llm_context_precision_with_reference'].mean()
    overall = (faithfulness_score + answer_relevancy_score + context_precision_score) / 3

    print("\n=== RAGAS EVALUATION RESULTS (averages) ===")
    print(f"Faithfulness:       {faithfulness_score:.4f}")
    print(f"Answer Relevancy:   {answer_relevancy_score:.4f}")
    print(f"Context Precision:  {context_precision_score:.4f}")
    print(f"\nOverall mean:       {overall:.4f}")

    with open("eval_results.txt", "w", encoding="utf-8") as f:
        f.write("RAGAS Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Faithfulness:       {faithfulness_score:.4f}\n")
        f.write(f"Answer Relevancy:   {answer_relevancy_score:.4f}\n")
        f.write(f"Context Precision:  {context_precision_score:.4f}\n")
        f.write(f"Overall mean:       {overall:.4f}\n")
        f.write(f"In-scope QA pairs:  {len(dataset)}\n")
        f.write(f"Total QA pairs:     {len(EVAL_QA_PAIRS)}\n")
        f.write("\nGuardrail Report:\n")
        correct = sum(e["expected_in_scope"] == e["guardrail_passed"] for e in guardrail_log)
        f.write(f"Guardrail accuracy: {correct}/{len(guardrail_log)}\n")

    results_df.to_csv("eval_results_detailed.csv", index=False)

    print("\nResults saved to eval_results.txt and eval_results_detailed.csv")
    return results_df


if __name__ == "__main__":
    run_evaluation()