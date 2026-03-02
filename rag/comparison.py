"""
Multi-Company Comparison
------------------------
Queries multiple companies in parallel and synthesizes a side-by-side comparison.

Usage:
    python rag/comparison.py --query "What are the main AI risks?" --tickers AAPL MSFT GOOGL
"""

import sys
import os
import argparse
sys.path.append(".")

from dotenv import load_dotenv
from pathlib import Path
from groq import Groq

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from embeddings.embedder import get_embedding_model, get_chroma_collection, query_similar_chunks

GROQ_MODEL = "llama-3.3-70b-versatile"
N_CHUNKS_PER_COMPANY = 4


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
COMPARISON_SYSTEM_PROMPT = """You are a senior financial analyst. You are given excerpts from SEC filings 
of multiple companies and asked to compare them on a specific question.

Your response must:
1. Address each company individually first with a short paragraph
2. Then provide a clear COMPARISON section highlighting key differences and similarities
3. End with a VERDICT — which company appears strongest on this dimension and why
4. Always cite which company and filing year you're drawing from
5. Be analytical, not just descriptive — point out what the differences actually mean

Be concise but insightful. Avoid generic statements."""


def retrieve_per_company(
    query: str,
    tickers: list[str],
    model,
    collection,
    section: str = None,
    year: str = None,
    n_chunks: int = N_CHUNKS_PER_COMPANY,
) -> dict[str, list[dict]]:
    """
    Retrieves relevant chunks separately for each company.
    Returns a dict of ticker → list of chunks.
    """
    results = {}
    for ticker in tickers:
        chunks = query_similar_chunks(
            query=query,
            model=model,
            collection=collection,
            n_results=n_chunks,
            ticker=ticker,
            section=section,
            year=year,
        )
        # Only keep relevant chunks
        results[ticker] = [c for c in chunks if c["similarity"] >= 0.25]
        print(f"   {ticker}: {len(results[ticker])} relevant chunks found")

    return results


def build_comparison_context(chunks_by_company: dict[str, list[dict]]) -> str:
    """
    Builds a structured context string grouping excerpts by company.
    """
    sections = []
    for ticker, chunks in chunks_by_company.items():
        if not chunks:
            sections.append(f"### {ticker}\nNo relevant information found in filings.")
            continue

        company_name = chunks[0]["metadata"].get("company", ticker)
        header = f"### {ticker} — {company_name}"
        excerpts = []
        for i, c in enumerate(chunks):
            m = c["metadata"]
            excerpts.append(
                f"[{ticker} Excerpt {i+1}] "
                f"{m['form']} {m['filing_date']} | {m['section_label']} | "
                f"Relevance: {c['similarity']:.2f}\n{c['text']}"
            )
        sections.append(header + "\n\n" + "\n\n---\n\n".join(excerpts))

    return "\n\n" + ("=" * 60) + "\n\n".join(sections)


def compare(
    query: str,
    tickers: list[str],
    section: str = None,
    year: str = None,
    verbose: bool = True,
) -> dict:
    """
    Main comparison function. Retrieves chunks per company and
    synthesizes a structured comparison using the LLM.

    Args:
        query:   The comparison question e.g. "How do these companies approach AI?"
        tickers: List of tickers e.g. ["AAPL", "MSFT", "GOOGL"]
        section: Optional section filter
        year:    Optional year filter
        verbose: Print retrieval details

    Returns:
        Dict with answer, per-company sources, and chunks_used
    """
    if verbose:
        print(f"\n🔍 Comparing {', '.join(tickers)} on: \"{query}\"")

    model = get_embedding_model()
    collection = get_chroma_collection()

    # Retrieve chunks per company
    chunks_by_company = retrieve_per_company(
        query=query,
        tickers=tickers,
        model=model,
        collection=collection,
        section=section,
        year=year,
    )

    # Check we have data for at least one company
    total_chunks = sum(len(v) for v in chunks_by_company.values())
    if total_chunks == 0:
        return {
            "answer": "No relevant information found for any of the selected companies.",
            "chunks_by_company": {},
            "total_chunks": 0,
            "query": query,
            "tickers": tickers,
        }

    # Build context
    context = build_comparison_context(chunks_by_company)

    user_message = f"""Compare the following companies on this question: "{query}"

Filing excerpts by company:
{context}

Please provide:
1. A paragraph for each company summarizing their position
2. A COMPARISON section with key similarities and differences  
3. A VERDICT on which company appears strongest on this dimension

Tickers to compare: {', '.join(tickers)}"""

    if verbose:
        print(f"\n🤖 Synthesizing comparison with {GROQ_MODEL}...")

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content

    # Compile sources per company
    sources_by_company = {}
    for ticker, chunks in chunks_by_company.items():
        seen = set()
        sources = []
        for c in chunks:
            m = c["metadata"]
            key = f"{m['filing_date']}_{m['section']}"
            if key not in seen:
                sources.append({
                    "ticker":      ticker,
                    "company":     m.get("company", ticker),
                    "form":        m["form"],
                    "filing_date": m["filing_date"],
                    "section":     m["section_label"],
                    "similarity":  c["similarity"],
                })
                seen.add(key)
        sources_by_company[ticker] = sources

    return {
        "answer":             answer,
        "chunks_by_company":  {t: len(c) for t, c in chunks_by_company.items()},
        "sources_by_company": sources_by_company,
        "total_chunks":       total_chunks,
        "query":              query,
        "tickers":            tickers,
    }


def print_comparison(result: dict):
    """Pretty prints a comparison result."""
    print(f"\n{'='*70}")
    print(f"⚖️  Comparison: {', '.join(result['tickers'])}")
    print(f"❓ Question: {result['query']}")
    print(f"{'='*70}")
    print(f"\n{result['answer']}")
    print(f"\n📎 Sources used:")
    for ticker, sources in result.get("sources_by_company", {}).items():
        print(f"\n  {ticker}:")
        for s in sources:
            print(f"    • {s['form']} {s['filing_date']} | {s['section']} [{s['similarity']:.2f}]")
    print(f"\n{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple companies using SEC filings")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--tickers", type=str, nargs="+", required=True, help="e.g. --tickers AAPL MSFT GOOGL")
    parser.add_argument("--section", type=str, default=None)
    parser.add_argument("--year", type=str, default=None)
    args = parser.parse_args()

    result = compare(
        query=args.query,
        tickers=args.tickers,
        section=args.section,
        year=args.year,
    )
    print_comparison(result)