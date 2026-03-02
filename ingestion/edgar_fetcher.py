"""
SEC EDGAR Fetcher
-----------------
Fetches 10-K and 10-Q filings for any public company using its ticker symbol.
Uses the free SEC EDGAR REST API — no API key required.

Usage:
    python ingestion/edgar_fetcher.py --ticker AAPL --filing_type 10-K --max_filings 3
"""

import os
import time
import argparse
import requests
from pathlib import Path

# SEC requires a User-Agent header — mandatory, else requests get blocked
HEADERS = {
    "User-Agent": "financial-doc-intelligence prabhjitsingh2401@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}


def get_cik_from_ticker(ticker: str) -> str:
    """
    Converts a ticker symbol (e.g. 'AAPL') to a zero-padded 10-digit CIK number.
    """
    print(f"🔍 Looking up CIK for ticker: {ticker.upper()}")

    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(tickers_url, headers=HEADERS)
    response.raise_for_status()

    for entry in response.json().values():
        if entry["ticker"] == ticker.upper():
            cik_padded = str(entry["cik_str"]).zfill(10)
            print(f"✅ Found CIK: {cik_padded} for {entry['title']}")
            return cik_padded

    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")


def get_filings_metadata(cik: str, filing_type: str = "10-K", max_filings: int = 5) -> list:
    """
    Fetches metadata for recent filings of a given type.
    """
    print(f"\n📋 Fetching {filing_type} filings for CIK: {cik}")

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    data = response.json()
    company_name = data.get("name", "Unknown")
    print(f"🏢 Company: {company_name}")

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    primary_documents = filings.get("primaryDocument", [])

    # CIK without leading zeros — used in the archive URL path
    cik_stripped = str(int(cik))

    results = []
    for i, form in enumerate(forms):
        if form == filing_type and len(results) < max_filings:
            accession_clean = accession_numbers[i].replace("-", "")
            results.append({
                "company": company_name,
                "ticker": data.get("tickers", [""])[0],
                "cik": cik,
                "cik_stripped": cik_stripped,
                "form": form,
                "filing_date": filing_dates[i],
                "accession_number": accession_numbers[i],
                "accession_clean": accession_clean,
                "primary_document": primary_documents[i],
            })

    print(f"✅ Found {len(results)} {filing_type} filings")
    for f in results:
        print(f"   → {f['filing_date']}  |  {f['accession_number']}")

    return results


def download_filing(filing: dict, output_dir: str = "data/raw") -> str:
    """
    Downloads the primary document for a filing.
    Correct SEC archive URL format:
        https://www.sec.gov/Archives/edgar/data/<CIK>/<accession_nodashes>/<filename>
    Note the /data/ segment — this is required and was the source of previous 404s.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ticker = filing["ticker"]
    date = filing["filing_date"]
    form = filing["form"].replace("-", "")
    filename = f"{ticker}_{form}_{date}.htm"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        print(f"⏭️  Already downloaded: {filename}")
        return filepath

    print(f"\n⬇️  Downloading: {filename}")

    cik_stripped = filing["cik_stripped"]
    accession_clean = filing["accession_clean"]
    primary_doc = filing["primary_document"]

    # ✅ Correct URL format — /Archives/edgar/data/<CIK>/...
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_stripped}/{accession_clean}/{primary_doc}"
    )

    print(f"   URL: {url}")

    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code == 200:
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(response.text)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"✅ Saved: {filepath} ({size_kb:.1f} KB)")
        time.sleep(0.5)  # Be polite to SEC servers
        return filepath
    else:
        raise Exception(f"HTTP {response.status_code} for {url}")


def fetch_company_filings(
    ticker: str,
    filing_type: str = "10-K",
    max_filings: int = 3,
    output_dir: str = "data/raw"
) -> list:
    """
    Main entry point. Fetches and downloads recent SEC filings for a ticker.

    Args:
        ticker:       Stock ticker e.g. 'AAPL', 'MSFT', 'TSLA'
        filing_type:  '10-K' (annual) or '10-Q' (quarterly)
        max_filings:  How many recent filings to download
        output_dir:   Local folder to save files

    Returns:
        List of filing dicts with local_path added
    """
    cik = get_cik_from_ticker(ticker)
    filings = get_filings_metadata(cik, filing_type, max_filings)

    downloaded = []
    for filing in filings:
        try:
            local_path = download_filing(filing, output_dir)
            filing["local_path"] = local_path
            downloaded.append(filing)
        except Exception as e:
            print(f"❌ Failed: {filing['accession_number']}: {e}")

    print(f"\n🎉 Done! Downloaded {len(downloaded)}/{len(filings)} filings to '{output_dir}/'")
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR filings by ticker")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker e.g. AAPL")
    parser.add_argument("--filing_type", type=str, default="10-K", choices=["10-K", "10-Q"])
    parser.add_argument("--max_filings", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()

    results = fetch_company_filings(
        ticker=args.ticker,
        filing_type=args.filing_type,
        max_filings=args.max_filings,
        output_dir=args.output_dir,
    )

    print("\n📁 Downloaded files:")
    for r in results:
        print(f"   {r['filing_date']}  →  {r.get('local_path', 'FAILED')}")