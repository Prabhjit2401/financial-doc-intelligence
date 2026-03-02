"""
Section-Aware Chunker
---------------------
Splits parsed SEC filing sections into overlapping chunks with rich metadata.

Unlike naive character-based chunking, this:
- Chunks WITHIN sections (never splits across section boundaries)
- Attaches section name, company, ticker, year to every chunk
- Uses overlap so context isn't lost at chunk boundaries
- Skips boilerplate-heavy sections that add noise (exhibits, covers)

Usage:
    python ingestion/chunker.py --filepath data/parsed/AAPL_10K_2024-11-01_parsed.json
    python ingestion/chunker.py --input_dir data/parsed
"""

import re
import json
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Chunking config
# These values are tunable — we'll experiment with them during RAGAS evaluation
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800        # Target characters per chunk (~200 tokens, good for retrieval)
CHUNK_OVERLAP = 150     # Overlap between consecutive chunks (preserves cross-boundary context)
MIN_CHUNK_SIZE = 100    # Skip chunks shorter than this (likely boilerplate remnants)

# Sections to skip — high noise, low signal for Q&A
SKIP_SECTIONS = {
    "exhibits",
    "mine_safety",
    "unresolved_comments",
    "security_ownership",
    "accountant_fees",
}

# Human-readable section labels for display and prompts
SECTION_LABELS = {
    "business":             "Business Overview",
    "risk_factors":         "Risk Factors",
    "properties":           "Properties",
    "legal_proceedings":    "Legal Proceedings",
    "market_info":          "Market Information",
    "selected_data":        "Selected Financial Data",
    "mda":                  "Management Discussion & Analysis",
    "market_risk":          "Quantitative Market Risk",
    "financial_statements": "Financial Statements",
    "controls":             "Controls & Procedures",
    "other_info":           "Other Information",
    "directors":            "Directors & Officers",
    "compensation":         "Executive Compensation",
    "relationships":        "Related Party Relationships",
    "full_text":            "Full Document",
}


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using punctuation patterns.
    More natural than hard character splits — chunks end at sentence boundaries.
    """
    # Split on period/exclamation/question followed by whitespace and capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    # Also split on double newlines (paragraph breaks)
    result = []
    for s in sentences:
        parts = s.split("\n\n")
        result.extend(p.strip() for p in parts if p.strip())
    return result


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a block of text into overlapping chunks at sentence boundaries.

    Strategy:
    1. Split into sentences first
    2. Greedily accumulate sentences until chunk_size is reached
    3. Start next chunk CHUNK_OVERLAP characters back (overlap)
    4. Ensures chunks end cleanly at sentence boundaries

    Args:
        text:       Input text block
        chunk_size: Target max characters per chunk
        overlap:    How many chars to repeat at start of next chunk

    Returns:
        List of text chunk strings
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, force-split it
        if sentence_len > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            # Hard split the long sentence
            for i in range(0, sentence_len, chunk_size - overlap):
                chunks.append(sentence[i:i + chunk_size])
            continue

        # If adding this sentence exceeds chunk_size, save current and start new
        if current_len + sentence_len > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunks.append(chunk_text_str)

            # Overlap: carry back last N characters worth of sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_chunk = overlap_sentences
            current_len = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_len += sentence_len

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_filing(parsed: dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Takes a parsed filing dict (output of pdf_parser.py) and produces
    a flat list of chunk dicts, each with full metadata attached.

    Args:
        parsed:     Dict from pdf_parser.py with 'metadata' and 'sections' keys
        chunk_size: Target chunk size in characters
        overlap:    Overlap between consecutive chunks

    Returns:
        List of chunk dicts, each containing:
        {
            "chunk_id":     unique identifier,
            "text":         the chunk text,
            "section":      section key e.g. "risk_factors",
            "section_label": human readable e.g. "Risk Factors",
            "ticker":       e.g. "AAPL",
            "company":      e.g. "Apple Inc.",
            "form":         e.g. "10K",
            "filing_date":  e.g. "2024-11-01",
            "year":         e.g. "2024",
            "chunk_index":  position within section,
            "total_chunks_in_section": total chunks for this section,
            "char_count":   length of this chunk,
        }
    """
    metadata = parsed.get("metadata", {})
    sections = parsed.get("sections", {})

    ticker = metadata.get("ticker", "UNKNOWN")
    company = metadata.get("company", ticker)
    form = metadata.get("form", "UNKNOWN")
    filing_date = metadata.get("filing_date", "UNKNOWN")
    year = filing_date[:4] if len(filing_date) >= 4 else "UNKNOWN"

    all_chunks = []
    total_sections_processed = 0
    total_chunks_created = 0

    for section_key, section_text in sections.items():

        # Skip low-value sections
        if section_key in SKIP_SECTIONS:
            continue

        if not section_text or len(section_text.strip()) < MIN_CHUNK_SIZE:
            continue

        section_label = SECTION_LABELS.get(section_key, section_key.replace("_", " ").title())

        # Split section into chunks
        text_chunks = chunk_text(section_text, chunk_size, overlap)

        # Filter out chunks that are too short
        text_chunks = [c for c in text_chunks if len(c.strip()) >= MIN_CHUNK_SIZE]

        total_in_section = len(text_chunks)

        for idx, chunk_str in enumerate(text_chunks):
            chunk_id = f"{ticker}_{form}_{year}_{section_key}_{idx:04d}"

            all_chunks.append({
                "chunk_id":                   chunk_id,
                "text":                       chunk_str.strip(),
                "section":                    section_key,
                "section_label":              section_label,
                "ticker":                     ticker,
                "company":                    company,
                "form":                       form,
                "filing_date":                filing_date,
                "year":                       year,
                "chunk_index":                idx,
                "total_chunks_in_section":    total_in_section,
                "char_count":                 len(chunk_str.strip()),
            })

        total_sections_processed += 1
        total_chunks_created += total_in_section
        print(f"   ✂️  {section_label:35s} → {total_in_section:3d} chunks")

    print(f"\n   📊 {total_sections_processed} sections → {total_chunks_created} total chunks")
    return all_chunks


def process_parsed_file(filepath: str, output_dir: str = "data/chunks") -> list[dict]:
    """
    Loads a parsed JSON file and produces chunks. Saves output as JSON.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"⚙️  Chunking: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    chunks = chunk_filing(parsed)

    # Save chunks
    stem = Path(filepath).stem.replace("_parsed", "")
    out_path = Path(output_dir) / f"{stem}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {len(chunks)} chunks → {out_path}")
    return chunks


def process_all_parsed_files(input_dir: str = "data/parsed", output_dir: str = "data/chunks") -> list[dict]:
    """
    Chunks all parsed JSON files in a directory.
    """
    json_files = list(Path(input_dir).glob("*_parsed.json"))

    if not json_files:
        print(f"⚠️  No parsed JSON files found in {input_dir}")
        return []

    print(f"📂 Found {len(json_files)} parsed filings to chunk")

    all_chunks = []
    for filepath in sorted(json_files):
        chunks = process_parsed_file(str(filepath), output_dir)
        all_chunks.extend(chunks)

    print(f"\n🎉 Done! Total chunks across all filings: {len(all_chunks)}")

    # Print a summary table
    print("\n📋 Summary by filing:")
    by_filing = {}
    for c in all_chunks:
        key = f"{c['ticker']} {c['form']} {c['filing_date']}"
        by_filing[key] = by_filing.get(key, 0) + 1
    for filing, count in sorted(by_filing.items()):
        print(f"   {filing:35s} → {count:4d} chunks")

    return all_chunks


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk parsed SEC filings into retrieval-ready pieces")
    parser.add_argument("--filepath", type=str, help="Path to a single *_parsed.json file")
    parser.add_argument("--input_dir", type=str, default="data/parsed", help="Directory of parsed JSON files")
    parser.add_argument("--output_dir", type=str, default="data/chunks", help="Where to save chunk JSON files")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help=f"Target chars per chunk (default {CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap chars between chunks (default {CHUNK_OVERLAP})")
    args = parser.parse_args()

    if args.filepath:
        process_parsed_file(args.filepath, args.output_dir)
    else:
        process_all_parsed_files(args.input_dir, args.output_dir)