"""
Embeddings & Vector Store
--------------------------
Embeds chunks using sentence-transformers and stores them in ChromaDB.

This is the core of the RAG pipeline — converting text into vectors
so we can later retrieve the most semantically relevant chunks for any query.

Usage:
    python embeddings/embedder.py --input_dir data/chunks
    python embeddings/embedder.py --query "What are Apple's main risk factors?"
"""

import json
import time
import argparse
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# all-MiniLM-L6-v2: fast, lightweight, great for retrieval tasks
# Upgrade to BAAI/bge-large-en-v1.5 later for better quality (but slower)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_DIR = "data/chroma_db"       # Where ChromaDB persists to disk
COLLECTION_NAME = "sec_filings"     # ChromaDB collection name
BATCH_SIZE = 64                     # Embed this many chunks at once


def get_embedding_model() -> SentenceTransformer:
    """
    Loads the sentence-transformer embedding model.
    Downloads automatically on first run (~90MB), cached locally after.
    """
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Model loaded — embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_chroma_collection(chroma_dir: str = CHROMA_DIR) -> chromadb.Collection:
    """
    Initialises ChromaDB with persistent storage and returns the collection.
    Creates the collection if it doesn't exist yet.
    """
    print(f"\n🗄️  Connecting to ChromaDB at: {chroma_dir}")
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_dir)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",   # Cosine similarity — standard for text embeddings
            "description": "SEC 10-K and 10-Q filing chunks with section metadata",
        }
    )

    print(f"✅ Collection '{COLLECTION_NAME}' ready — {collection.count()} chunks already stored")
    return collection


def embed_and_store_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    collection: chromadb.Collection,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Embeds a list of chunk dicts and upserts them into ChromaDB.

    Uses upsert (not add) so re-running won't create duplicates —
    safe to run multiple times on the same data.

    Args:
        chunks:     List of chunk dicts from chunker.py
        model:      Loaded SentenceTransformer model
        collection: ChromaDB collection to store into
        batch_size: How many chunks to embed at once (GPU memory tradeoff)

    Returns:
        Number of chunks stored
    """
    if not chunks:
        print("⚠️  No chunks to embed")
        return 0

    # Filter out chunks already in the DB (by chunk_id) to avoid re-embedding
    existing_ids = set()
    try:
        existing = collection.get(ids=[c["chunk_id"] for c in chunks[:100]])
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print(f"⏭️  All {len(chunks)} chunks already in DB — skipping")
        return 0

    print(f"\n🔢 Embedding {len(new_chunks)} chunks (batch size {batch_size})...")
    total_stored = 0
    start_time = time.time()

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]

        # Extract just the text for embedding
        texts = [c["text"] for c in batch]

        # Embed — returns a numpy array of shape (batch_size, embedding_dim)
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        # ChromaDB expects: ids, embeddings, documents, metadatas
        ids = [c["chunk_id"] for c in batch]
        documents = texts
        metadatas = [
            {
                # Store everything EXCEPT text (already in documents) and chunk_id (in ids)
                "section":      c.get("section", ""),
                "section_label": c.get("section_label", ""),
                "ticker":       c.get("ticker", ""),
                "company":      c.get("company", ""),
                "form":         c.get("form", ""),
                "filing_date":  c.get("filing_date", ""),
                "year":         c.get("year", ""),
                "chunk_index":  c.get("chunk_index", 0),
                "char_count":   c.get("char_count", 0),
            }
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        total_stored += len(batch)
        elapsed = time.time() - start_time
        progress = total_stored / len(new_chunks) * 100
        print(f"   [{progress:5.1f}%] {total_stored}/{len(new_chunks)} chunks "
              f"| {elapsed:.1f}s elapsed", end="\r")

    print(f"\n✅ Stored {total_stored} chunks in {time.time() - start_time:.1f}s")
    return total_stored


def load_chunks_from_dir(chunks_dir: str = "data/chunks") -> list[dict]:
    """
    Loads all chunk JSON files from a directory into a flat list.
    """
    chunk_files = list(Path(chunks_dir).glob("*_chunks.json"))

    if not chunk_files:
        print(f"⚠️  No chunk files found in {chunks_dir}")
        return []

    print(f"📂 Loading chunks from {len(chunk_files)} files...")
    all_chunks = []

    for filepath in sorted(chunk_files):
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        print(f"   → {filepath.name}: {len(chunks)} chunks")

    print(f"✅ Total chunks loaded: {len(all_chunks)}")
    return all_chunks


def query_similar_chunks(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    n_results: int = 5,
    ticker: str = None,
    section: str = None,
    year: str = None,
) -> list[dict]:
    """
    Retrieves the most semantically similar chunks for a query.

    Supports optional metadata filtering:
        ticker="AAPL"          → only Apple filings
        section="risk_factors" → only Risk Factors sections
        year="2024"            → only 2024 filings

    Args:
        query:     Natural language question
        model:     Embedding model
        collection: ChromaDB collection
        n_results: Number of chunks to return
        ticker:    Optional filter by company ticker
        section:   Optional filter by section name
        year:      Optional filter by filing year

    Returns:
        List of result dicts with text, metadata, and similarity distance
    """
    # Embed the query
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    # Build metadata filter
    where_clause = {}
    filters = []
    if ticker:
        filters.append({"ticker": {"$eq": ticker}})
    if section:
        filters.append({"section": {"$eq": section}})
    if year:
        filters.append({"year": {"$eq": year}})

    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    # Query ChromaDB
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_clause:
        query_kwargs["where"] = where_clause

    results = collection.query(**query_kwargs)

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "chunk_id":     results["ids"][0][i],
            "text":         results["documents"][0][i],
            "metadata":     results["metadatas"][0][i],
            "distance":     results["distances"][0][i],
            "similarity":   round(1 - results["distances"][0][i], 4),
        })

    return formatted


def print_search_results(results: list[dict], query: str):
    """Pretty prints search results."""
    print(f"\n🔍 Query: \"{query}\"")
    print(f"{'='*70}")
    for i, r in enumerate(results):
        m = r["metadata"]
        print(f"\n[{i+1}] Similarity: {r['similarity']:.4f}")
        print(f"    {m['ticker']} | {m['form']} | {m['filing_date']} | {m['section_label']}")
        print(f"    {r['text'][:300]}...")
    print(f"\n{'='*70}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed SEC filing chunks into ChromaDB")
    parser.add_argument("--input_dir", type=str, default="data/chunks", help="Directory of chunk JSON files")
    parser.add_argument("--chroma_dir", type=str, default=CHROMA_DIR, help="ChromaDB persistence directory")
    parser.add_argument("--query", type=str, help="Test query to run after embedding")
    parser.add_argument("--ticker", type=str, help="Filter query by ticker")
    parser.add_argument("--section", type=str, help="Filter query by section e.g. risk_factors")
    parser.add_argument("--year", type=str, help="Filter query by year e.g. 2024")
    parser.add_argument("--n_results", type=int, default=5)
    args = parser.parse_args()

    # Load model and DB
    model = get_embedding_model()
    collection = get_chroma_collection(args.chroma_dir)

    if args.query:
        # Search mode — just run a query against existing DB
        results = query_similar_chunks(
            query=args.query,
            model=model,
            collection=collection,
            n_results=args.n_results,
            ticker=args.ticker,
            section=args.section,
            year=args.year,
        )
        print_search_results(results, args.query)
    else:
        # Embed mode — load chunks and store in ChromaDB
        chunks = load_chunks_from_dir(args.input_dir)
        if chunks:
            stored = embed_and_store_chunks(chunks, model, collection)
            print(f"\n🎉 Done! ChromaDB now contains {collection.count()} total chunks")

            # Run a quick test query to confirm everything works
            print("\n🧪 Running test query to confirm retrieval works...")
            test_results = query_similar_chunks(
                query="What are the main risks facing the company?",
                model=model,
                collection=collection,
                n_results=3,
            )
            print_search_results(test_results, "What are the main risks facing the company?")