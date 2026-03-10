"""
RAGAS Evaluation — Financial Document Intelligence
---------------------------------------------------
Evaluates the RAG pipeline on a curated set of SEC filing questions.

Metrics:
  - Faithfulness       : Does the answer stay grounded in the retrieved context?
  - Answer Relevancy   : Does the answer actually address the question?
  - Context Precision  : Are the top-ranked chunks the most relevant ones?
  - Context Recall     : Does the retrieved context cover the ground truth?

Usage:
    python evaluate/evaluate.py
    python evaluate/evaluate.py --output results/ragas_results.json
    python evaluate/evaluate.py --ticker AAPL --n_questions 5
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(".")

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq

from rag.chain import ask


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation dataset
# Curated question + ground truth answer pairs covering key SEC filing topics.
# Ground truths are concise factual statements a good RAG system should cover.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    # Risk Factors
    {
        "question": "What are Apple's main supply chain risk factors?",
        "ground_truth": "Apple relies on a limited number of suppliers, many located in China and Asia, for key components and manufacturing. Disruptions from geopolitical tensions, natural disasters, or trade restrictions could materially impact production and revenue.",
        "ticker": "AAPL",
        "section": "risk_factors",
    },
    {
        "question": "What regulatory risks does Apple face regarding the App Store?",
        "ground_truth": "Apple faces antitrust scrutiny and regulatory pressure globally over its App Store policies, including the 30% commission and restrictions on alternative payment methods. Regulations such as the EU Digital Markets Act could force Apple to allow third-party app stores and alternative payment systems.",
        "ticker": "AAPL",
        "section": "risk_factors",
    },
    {
        "question": "What are Microsoft's key risk factors related to cybersecurity?",
        "ground_truth": "Microsoft faces significant cybersecurity risks including nation-state attacks, ransomware, and vulnerabilities in its cloud infrastructure. Security breaches could damage customer trust, result in regulatory penalties, and materially harm Azure and Office 365 revenue.",
        "ticker": "MSFT",
        "section": "risk_factors",
    },
    {
        "question": "What risks does Tesla identify related to manufacturing and production?",
        "ground_truth": "Tesla faces risks from production ramp challenges at new factories, supply chain constraints for battery materials like lithium and cobalt, and dependence on a limited number of manufacturing facilities. Production delays can materially impact delivery targets and financial results.",
        "ticker": "TSLA",
        "section": "risk_factors",
    },
    # MD&A
    {
        "question": "How did Apple describe its revenue performance in its most recent annual filing?",
        "ground_truth": "Apple reported strong Services revenue growth driven by the App Store, Apple Music, and iCloud, while iPhone revenue remained the largest segment. Management highlighted the growing installed base of active devices as a driver of future Services monetization.",
        "ticker": "AAPL",
        "section": "mda",
    },
    {
        "question": "What did Microsoft's management say about Azure cloud revenue growth?",
        "ground_truth": "Microsoft's management highlighted Azure as a primary revenue growth driver, with strong demand from enterprise customers migrating workloads to the cloud. AI-powered services including Copilot were cited as incremental revenue opportunities within the Azure platform.",
        "ticker": "MSFT",
        "section": "mda",
    },
    {
        "question": "How did Tesla explain changes in its gross margin?",
        "ground_truth": "Tesla management discussed gross margin pressures from price reductions implemented to stimulate demand, partially offset by manufacturing cost improvements and higher energy generation revenue. The company highlighted continued cost reduction efforts at Gigafactories.",
        "ticker": "TSLA",
        "section": "mda",
    },
    # Business Overview
    {
        "question": "How does Apple describe its business segments?",
        "ground_truth": "Apple operates through product segments including iPhone, Mac, iPad, and Wearables, Home and Accessories, as well as a Services segment. Services include the App Store, Apple Music, iCloud, Apple TV+, Apple Pay, and licensing revenue.",
        "ticker": "AAPL",
        "section": "business",
    },
    {
        "question": "How does Google describe its core advertising business?",
        "ground_truth": "Google's advertising revenue is generated through Google Search, YouTube ads, and the Google Network. Advertisers bid on keywords and placements through an auction system, with revenue driven by search query volume, ad click-through rates, and cost-per-click pricing.",
        "ticker": "GOOGL",
        "section": "business",
    },
    # Cross-cutting
    {
        "question": "What do Apple and Microsoft say about artificial intelligence in their filings?",
        "ground_truth": "Apple focuses on on-device AI through Apple Intelligence integrated into its hardware and operating systems, emphasizing privacy. Microsoft highlights its OpenAI partnership and Copilot integration across Microsoft 365, Azure, and GitHub as key growth drivers.",
        "ticker": None,
        "section": None,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Run RAG pipeline for each question
# ─────────────────────────────────────────────────────────────────────────────
def build_ragas_dataset(questions: list, n_results: int = 6) -> dict:
    """Run RAG pipeline on all questions and collect answers + contexts."""
    print(f"\n{'='*60}")
    print(f"Running RAG pipeline on {len(questions)} questions...")
    print(f"{'='*60}\n")

    all_questions, all_answers, all_contexts, all_ground_truths = [], [], [], []

    for i, item in enumerate(questions):
        q = item["question"]
        print(f"[{i+1}/{len(questions)}] {q[:70]}...")

        try:
            result = ask(
                query=q,
                ticker=item.get("ticker"),
                section=item.get("section"),
                n_results=n_results,
                verbose=False,
            )
            answer = result["answer"]
            # RAGAS expects a list of context strings per question
            contexts = [s["text"] for s in result.get("source_chunks", [])]

            # Fallback: use source metadata as context if no text chunks returned
            if not contexts:
                contexts = [
                    f"{s['ticker']} {s['form']} {s['filing_date']} {s['section']}"
                    for s in result.get("sources", [])
                ]

            print(f"    ✓ Answer: {answer[:80]}...")
            print(f"    ✓ Contexts: {len(contexts)} chunks retrieved\n")

        except Exception as e:
            print(f"    ✗ Error: {e}\n")
            answer = ""
            contexts = [""]

        all_questions.append(q)
        all_answers.append(answer)
        all_contexts.append(contexts)
        all_ground_truths.append(item["ground_truth"])

    return {
        "question": all_questions,
        "answer": all_answers,
        "contexts": all_contexts,
        "ground_truth": all_ground_truths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(data: dict) -> dict:
    print(f"\\n{'='*60}")
    print("Running RAGAS evaluation...")
    print(f"{'='*60}\\n")

    dataset = Dataset.from_dict(data)

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=2048,
    max_retries=3,
    n=1,  # explicitly tell Groq to only return 1 generation
    )
    wrapped_llm = LangchainLLMWrapper(groq_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=wrapped_llm,
    embeddings=wrapped_embeddings,
    raise_exceptions=False,
    batch_size=2,  # evaluate 2 questions at a time instead of all at once
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Format and save results
# ─────────────────────────────────────────────────────────────────────────────
def format_results(results, questions: list) -> dict:
    """Convert RAGAS results to a clean dict for saving and display."""
    scores_df = results.to_pandas()

    output = {
        "run_timestamp": datetime.now().isoformat(),
        "n_questions": len(questions),
        "aggregate_scores": {
            "faithfulness": round(float(scores_df["faithfulness"].mean()), 4),
            "answer_relevancy": round(float(scores_df["answer_relevancy"].mean()), 4),
            "context_precision": round(float(scores_df["context_precision"].mean()), 4),
            "context_recall": round(float(scores_df["context_recall"].mean()), 4),
        },
        "per_question": [],
    }

    for i, row in scores_df.iterrows():
        output["per_question"].append({
            "question": questions[i]["question"],
            "ticker": questions[i].get("ticker"),
            "section": questions[i].get("section"),
            "faithfulness": round(float(row.get("faithfulness", 0)), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 4),
            "context_precision": round(float(row.get("context_precision", 0)), 4),
            "context_recall": round(float(row.get("context_recall", 0)), 4),
        })

    return output


def print_results(output: dict):
    """Pretty print evaluation results to console."""
    print(f"\n{'='*60}")
    print("RAGAS EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Timestamp : {output['run_timestamp']}")
    print(f"Questions : {output['n_questions']}")
    print()

    agg = output["aggregate_scores"]
    print("AGGREGATE SCORES")
    print(f"  Faithfulness       : {agg['faithfulness']:.4f}  {'🟢' if agg['faithfulness'] > 0.7 else '🟡' if agg['faithfulness'] > 0.5 else '🔴'}")
    print(f"  Answer Relevancy   : {agg['answer_relevancy']:.4f}  {'🟢' if agg['answer_relevancy'] > 0.7 else '🟡' if agg['answer_relevancy'] > 0.5 else '🔴'}")
    print(f"  Context Precision  : {agg['context_precision']:.4f}  {'🟢' if agg['context_precision'] > 0.7 else '🟡' if agg['context_precision'] > 0.5 else '🔴'}")
    print(f"  Context Recall     : {agg['context_recall']:.4f}  {'🟢' if agg['context_recall'] > 0.7 else '🟡' if agg['context_recall'] > 0.5 else '🔴'}")
    print()

    print("PER-QUESTION BREAKDOWN")
    print(f"{'Question':<55} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Recall':>6}")
    print("-" * 85)
    for q in output["per_question"]:
        label = q["question"][:52] + "..." if len(q["question"]) > 55 else q["question"]
        print(f"{label:<55} {q['faithfulness']:>6.3f} {q['answer_relevancy']:>6.3f} {q['context_precision']:>6.3f} {q['context_recall']:>6.3f}")

    print(f"\n{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline")
    parser.add_argument("--output", type=str, default="results/ragas_results.json",
                        help="Path to save JSON results (default: results/ragas_results.json)")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Filter eval questions to a specific ticker (e.g. AAPL)")
    parser.add_argument("--n_questions", type=int, default=None,
                        help="Limit number of questions to evaluate (useful for quick tests)")
    parser.add_argument("--n_chunks", type=int, default=6,
                        help="Number of chunks to retrieve per question (default: 6)")
    args = parser.parse_args()

    # Filter questions
    questions = EVAL_QUESTIONS
    if args.ticker:
        questions = [q for q in questions if q.get("ticker") == args.ticker or q.get("ticker") is None]
    if args.n_questions:
        questions = questions[:args.n_questions]

    # Run pipeline
    data = build_ragas_dataset(questions, n_results=args.n_chunks)

    # Evaluate
    results = run_evaluation(data)

    # Format and display
    output = format_results(results, questions)
    print_results(output)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()