"""
RAG Chain
---------
Wires together ChromaDB retrieval + Groq LLM to answer questions
about SEC filings with source citations.

Usage:
    python rag/chain.py --query "What are Apple's main risk factors in 2024?"
    python rag/chain.py --query "How did revenue grow?" --ticker AAPL --year 2024
"""

import os
import argparse
from dotenv import load_dotenv
from groq import Groq

# Import our own modules
import sys
sys.path.append(".")
from embeddings.embedder import get_embedding_model, get_chroma_collection, query_similar_chunks

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"   # Best free model on Groq
N_RETRIEVAL_CHUNKS = 6                    # How many chunks to retrieve per query
MIN_SIMILARITY = 0.25                     # Below this score, chunk is too irrelevant


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions about SEC filings (10-K and 10-Q reports).

You are given retrieved excerpts from real SEC filings. Your job is to:
1. Answer the question accurately using ONLY the provided context
2. Always cite your sources (company, filing date, section)
3. If the context doesn't contain enough information, say so clearly — never hallucinate
4. Be concise but thorough
5. When discussing numbers, be precise

Format your response as:
- A direct answer to the question
- Key supporting details from the filings
- Source citations at the end like: [Source: AAPL 10-K 2024-11-01, Risk Factors]
"""

def build_context_string(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a context string for the LLM prompt.
    Each chunk is labelled with its source metadata so the LLM can cite it.
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        m = chunk["metadata"]
        header = (
            f"[Excerpt {i+1}] "
            f"{m['company']} | {m['form']} | {m['filing_date']} | "
            f"{m['section_label']} | Relevance: {chunk['similarity']:.2f}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def ask(
    query: str,
    ticker: str = None,
    section: str = None,
    year: str = None,
    n_results: int = N_RETRIEVAL_CHUNKS,
    verbose: bool = True,
) -> dict:
    """
    Main RAG function. Given a natural language query:
    1. Embeds the query
    2. Retrieves top N relevant chunks from ChromaDB
    3. Filters out low-relevance chunks
    4. Sends context + query to Groq LLM
    5. Returns answer with sources

    Args:
        query:    Natural language question
        ticker:   Optional — filter to specific company e.g. "AAPL"
        section:  Optional — filter to section e.g. "risk_factors"
        year:     Optional — filter to year e.g. "2024"
        n_results: Number of chunks to retrieve
        verbose:  Print retrieval details

    Returns:
        Dict with keys: answer, sources, chunks_used, query
    """
    # Step 1: Load embedding model and ChromaDB
    model = get_embedding_model()
    collection = get_chroma_collection()

    if collection.count() == 0:
        return {
            "answer": "❌ No documents in the database yet. Run embedder.py first.",
            "sources": [],
            "chunks_used": 0,
            "query": query,
        }

    # Step 2: Retrieve relevant chunks
    if verbose:
        print(f"\n🔍 Retrieving chunks for: \"{query}\"")
        if ticker: print(f"   Filter: ticker={ticker}")
        if section: print(f"   Filter: section={section}")
        if year: print(f"   Filter: year={year}")

    chunks = query_similar_chunks(
        query=query,
        model=model,
        collection=collection,
        n_results=n_results,
        ticker=ticker,
        section=section,
        year=year,
    )

    # Step 3: Filter low-relevance chunks
    relevant_chunks = [c for c in chunks if c["similarity"] >= MIN_SIMILARITY]

    if not relevant_chunks:
        return {
            "answer": "I couldn't find relevant information in the filings to answer this question.",
            "sources": [],
            "chunks_used": 0,
            "query": query,
        }

    if verbose:
        print(f"✅ Retrieved {len(relevant_chunks)} relevant chunks:")
        for c in relevant_chunks:
            m = c["metadata"]
            print(f"   [{c['similarity']:.3f}] {m['ticker']} {m['filing_date']} — {m['section_label']}")

    # Step 4: Build context and call Groq
    context = build_context_string(relevant_chunks)

    user_message = f"""Context from SEC Filings:
{context}

Question: {query}

Please answer based on the filing excerpts above. Cite your sources."""

    if verbose:
        print(f"\n🤖 Calling {GROQ_MODEL}...")

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,      # Low temperature = more factual, less creative
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    # Step 5: Compile sources
    sources = []
    seen = set()
    for c in relevant_chunks:
        m = c["metadata"]
        source_key = f"{m['ticker']}_{m['filing_date']}_{m['section']}"
        if source_key not in seen:
            sources.append({
                "ticker":       m["ticker"],
                "company":      m["company"],
                "form":         m["form"],
                "filing_date":  m["filing_date"],
                "section":      m["section_label"],
                "similarity":   c["similarity"],
            })
            seen.add(source_key)

    return {
        "answer":       answer,
        "sources":      sources,
        "chunks_used":  len(relevant_chunks),
        "query":        query,
    }


def print_answer(result: dict):
    """Pretty prints a RAG result."""
    print(f"\n{'='*70}")
    print(f"❓ Question: {result['query']}")
    print(f"{'='*70}")
    print(f"\n💬 Answer:\n{result['answer']}")
    print(f"\n📎 Sources ({result['chunks_used']} chunks used):")
    for s in result["sources"]:
        print(f"   • {s['company']} | {s['form']} | {s['filing_date']} | {s['section']} [{s['similarity']:.3f}]")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions about SEC filings")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    parser.add_argument("--ticker", type=str, help="Filter by ticker e.g. AAPL")
    parser.add_argument("--section", type=str, help="Filter by section e.g. risk_factors")
    parser.add_argument("--year", type=str, help="Filter by year e.g. 2024")
    parser.add_argument("--n_results", type=int, default=N_RETRIEVAL_CHUNKS)
    args = parser.parse_args()

    result = ask(
        query=args.query,
        ticker=args.ticker,
        section=args.section,
        year=args.year,
        n_results=args.n_results,
    )
    print_answer(result)