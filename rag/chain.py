"""
RAG Chain — Financial Document Intelligence
Updated to return source_chunks with raw text for RAGAS evaluation.
"""
import sys, os, argparse
sys.path.append(".")
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
from groq import Groq
from embeddings.embedder import get_embedding_model, get_chroma_collection

GROQ_MODEL = "llama-3.3-70b-versatile"
N_RESULTS_DEFAULT = 6
SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings.
Answer questions based ONLY on the provided context excerpts from 10-K and 10-Q filings.
Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. Always cite which company and filing section your answer comes from.
3. If the context does not contain enough information, say so clearly.
4. Be precise and factual. Avoid speculation.
5. Structure longer answers with clear paragraphs or bullet points."""

def ask(query, ticker=None, section=None, year=None, n_results=N_RESULTS_DEFAULT, verbose=False):
    model = get_embedding_model()
    collection = get_chroma_collection()
    query_embedding = model.encode(query).tolist()

    filters = []
    if ticker: filters.append({"ticker": {"$eq": ticker}})
    if section: filters.append({"section": {"$eq": section}})
    if year: filters.append({"year": {"$eq": year}})
    where_filter = None
    if len(filters) == 1: where_filter = filters[0]
    elif len(filters) > 1: where_filter = {"$and": filters}

    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where_filter: query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        return {"answer": "No relevant content found.", "sources": [], "source_chunks": [], "chunks_used": 0}

    context_parts, sources, source_chunks = [], [], []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = round(1 - dist, 4)
        label = f"[{meta.get('ticker','?')} | {meta.get('form','?')} | {meta.get('filing_date','?')} | {meta.get('section','?')}]"
        context_parts.append(f"{label}\n{doc}")
        sources.append({"ticker": meta.get("ticker",""), "form": meta.get("form",""),
                        "filing_date": meta.get("filing_date",""), "section": meta.get("section",""), "similarity": similarity})
        source_chunks.append({"text": doc, "ticker": meta.get("ticker",""), "form": meta.get("form",""),
                               "filing_date": meta.get("filing_date",""), "section": meta.get("section",""), "similarity": similarity})

    context = "\n\n---\n\n".join(context_parts)
    if verbose:
        print(f"\n{'='*60}\nQUERY: {query}\nCHUNKS: {len(documents)}")
        for s in sources: print(f"  {s['ticker']:6} | {s['section']:25} | {s['similarity']:.3f}")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context from SEC filings:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer based only on the context above."}
        ],
        temperature=0.1, max_tokens=1024,
    )
    return {"answer": response.choices[0].message.content.strip(),
            "sources": sources, "source_chunks": source_chunks, "chunks_used": len(documents)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--section", type=str, default=None)
    parser.add_argument("--year", type=str, default=None)
    parser.add_argument("--n_results", type=int, default=N_RESULTS_DEFAULT)
    args = parser.parse_args()
    result = ask(args.query, args.ticker, args.section, args.year, args.n_results, verbose=True)
    print("\n📄 ANSWER\n" + "─"*60)
    print(result["answer"])
    print("\n📎 SOURCES\n" + "─"*60)
    for s in result["sources"]:
        print(f"  {s['ticker']:6} | {s['form']:5} | {s['filing_date']:12} | {s['section']:25} | {s['similarity']:.3f}")