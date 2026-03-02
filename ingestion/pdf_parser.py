"""
SEC Filing Parser
-----------------
Parses downloaded 10-K/10-Q HTML filings into clean, section-aware text.

Instead of returning one giant blob of text, this script detects standard
SEC sections (Risk Factors, MD&A, Financial Statements etc.) and returns
a structured dict — enabling smarter, context-preserving chunking later.

Usage:
    python ingestion/pdf_parser.py --filepath data/raw/AAPL_10K_2024-11-01.htm
"""

import re
import json
import argparse
from pathlib import Path
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────────────────────────
# Standard 10-K section headers as defined by SEC
# We match these case-insensitively and allow for minor formatting variations
# ─────────────────────────────────────────────────────────────────────────────
SECTION_PATTERNS = [
    ("business",              r"item\s*1[\.\s]+business"),
    ("risk_factors",          r"item\s*1a[\.\s]+risk\s*factors"),
    ("unresolved_comments",   r"item\s*1b[\.\s]+unresolved\s*staff\s*comments"),
    ("properties",            r"item\s*2[\.\s]+properties"),
    ("legal_proceedings",     r"item\s*3[\.\s]+legal\s*proceedings"),
    ("mine_safety",           r"item\s*4[\.\s]+mine\s*safety"),
    ("market_info",           r"item\s*5[\.\s]+market"),
    ("selected_data",         r"item\s*6[\.\s]+selected"),
    ("mda",                   r"item\s*7[\.\s]+management"),
    ("market_risk",           r"item\s*7a[\.\s]+quantitative"),
    ("financial_statements",  r"item\s*8[\.\s]+financial\s*statements"),
    ("controls",              r"item\s*9a[\.\s]+controls"),
    ("other_info",            r"item\s*9b[\.\s]+other\s*information"),
    ("directors",             r"item\s*10[\.\s]+directors"),
    ("compensation",          r"item\s*11[\.\s]+executive\s*compensation"),
    ("security_ownership",    r"item\s*12[\.\s]+security\s*ownership"),
    ("relationships",         r"item\s*13[\.\s]+certain\s*relationships"),
    ("accountant_fees",       r"item\s*14[\.\s]+principal\s*account"),
    ("exhibits",              r"item\s*15[\.\s]+exhibits"),
]


def clean_html(filepath: str) -> str:
    """
    Step 1: Load the .htm file and strip all HTML, returning clean plain text.
    - Removes scripts, styles, and inline XBRL tags
    - Collapses excessive whitespace
    - Preserves paragraph breaks
    """
    print(f"📄 Loading: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_html = f.read()

    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove noise elements that add no text value
    for tag in soup(["script", "style", "meta", "link", "ix:header", "ix:hidden"]):
        tag.decompose()

    # Replace block elements with newlines to preserve paragraph structure
    for tag in soup.find_all(["p", "div", "tr", "br", "h1", "h2", "h3", "h4", "li"]):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text()

    # Clean up whitespace — collapse multiple blank lines into one
    text = re.sub(r"\n[ \t]+", "\n", text)       # strip leading spaces on lines
    text = re.sub(r"[ \t]{2,}", " ", text)         # collapse inline spaces
    text = re.sub(r"\n{3,}", "\n\n", text)         # max 2 consecutive newlines

    print(f"✅ Extracted {len(text):,} characters of clean text")
    return text


def detect_sections(text: str) -> dict:
    """
    Step 2: Find standard SEC 10-K section boundaries in the cleaned text.

    Strategy:
    - Scan for Item N header patterns using regex
    - Record the character position of each match
    - Slice text between consecutive section starts
    - Skip the Table of Contents matches (usually appear twice — TOC + actual section)
    """
    print("\n🔍 Detecting sections...")

    # Find ALL matches for each section pattern, recording their positions
    matches = []  # list of (position, section_key, matched_text)

    for section_key, pattern in SECTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((match.start(), section_key, match.group()))

    # Sort all matches by position in document
    matches.sort(key=lambda x: x[0])

    if not matches:
        print("⚠️  No sections detected — returning full text as single section")
        return {"full_text": text}

    # De-duplicate: SEC filings list items twice (Table of Contents + actual content)
    # The real content section is always the SECOND occurrence — skip the first
    seen = {}
    deduped = []
    for pos, key, matched in matches:
        if key not in seen:
            seen[key] = 0
        seen[key] += 1
        # Keep only the second+ occurrence (skip TOC entries)
        if seen[key] >= 2:
            deduped.append((pos, key, matched))

    # If dedup removed everything, fall back to all matches
    if not deduped:
        deduped = matches

    print(f"✅ Found {len(deduped)} sections")
    for pos, key, matched in deduped:
        print(f"   → [{pos:>8,}]  {key:25s}  \"{matched[:50]}\"")

    # Slice text between section boundaries
    sections = {}
    for i, (pos, key, _) in enumerate(deduped):
        start = pos
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        section_text = text[start:end].strip()

        # Skip very short sections (likely TOC artifacts < 200 chars)
        if len(section_text) > 200:
            sections[key] = section_text

    return sections


def parse_filing(filepath: str, output_dir: str = "data/parsed") -> dict:
    """
    Main entry point. Parses a single filing .htm file into structured sections.

    Args:
        filepath:   Path to the downloaded .htm filing
        output_dir: Where to save the parsed JSON output

    Returns:
        Dict with keys:
        - metadata: company, ticker, form, filing_date (parsed from filename)
        - sections: dict of section_name → clean text
        - full_text: complete clean text (fallback)
        - char_count: total characters
        - section_count: number of sections found
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Parse metadata from filename e.g. "AAPL_10K_2024-11-01.htm"
    stem = Path(filepath).stem  # "AAPL_10K_2024-11-01"
    parts = stem.split("_")
    metadata = {
        "ticker": parts[0] if len(parts) > 0 else "UNKNOWN",
        "form": parts[1] if len(parts) > 1 else "UNKNOWN",
        "filing_date": parts[2] if len(parts) > 2 else "UNKNOWN",
        "source_file": str(filepath),
    }

    # Step 1: clean HTML → plain text
    full_text = clean_html(filepath)

    # Step 2: detect and slice sections
    sections = detect_sections(full_text)

    result = {
        "metadata": metadata,
        "sections": sections,
        "full_text": full_text,
        "char_count": len(full_text),
        "section_count": len(sections),
    }

    # Save parsed output as JSON
    out_filename = f"{stem}_parsed.json"
    out_path = Path(output_dir) / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved parsed output: {out_path}")
    print(f"   Sections: {list(sections.keys())}")
    print(f"   Total chars: {len(full_text):,}")

    return result


def parse_all_filings(input_dir: str = "data/raw", output_dir: str = "data/parsed") -> list:
    """
    Convenience function — parses all .htm files in a directory.
    """
    htm_files = list(Path(input_dir).glob("*.htm"))

    if not htm_files:
        print(f"⚠️  No .htm files found in {input_dir}")
        return []

    print(f"📂 Found {len(htm_files)} filings to parse\n")
    results = []

    for filepath in sorted(htm_files):
        print(f"\n{'='*60}")
        result = parse_filing(str(filepath), output_dir)
        results.append(result)

    print(f"\n🎉 Done! Parsed {len(results)} filings → {output_dir}/")
    return results


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse SEC filing HTML into structured sections")
    parser.add_argument("--filepath", type=str, help="Path to a single .htm filing")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Parse all .htm files in this dir")
    parser.add_argument("--output_dir", type=str, default="data/parsed", help="Where to save parsed JSON")
    args = parser.parse_args()

    if args.filepath:
        # Parse a single file
        parse_filing(args.filepath, args.output_dir)
    else:
        # Parse all files in input_dir
        parse_all_filings(args.input_dir, args.output_dir)