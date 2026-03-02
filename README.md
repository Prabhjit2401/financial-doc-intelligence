---
title: Financial Doc Intelligence
emoji: 📊
colorFrom: green
colorTo: teal
sdk: streamlit
sdk_version: 1.32.0
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# 📊 Financial Document Intelligence

Query SEC 10-K & 10-Q filings using natural language. Built with ChromaDB, Groq LLaMA 3.3, and Streamlit.

## What it does
- Downloads SEC filings directly from EDGAR for any public company
- Parses and chunks filings with section-aware splitting (Risk Factors, MD&A, etc.)
- Embeds chunks using sentence-transformers and stores in ChromaDB
- Answers natural language questions with source citations
- Compares multiple companies side by side

## Tech Stack
- **Retrieval:** ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
- **LLM:** Groq LLaMA 3.3 70B
- **Frontend:** Streamlit
- **Data:** SEC EDGAR REST API (free, no key needed)

## Run locally
```bash
git clone https://github.com/Prabhjit2401/financial-doc-intelligence
cd financial-doc-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Fetch and index filings
python ingestion/edgar_fetcher.py --ticker AAPL --filing_type 10-K --max_filings 3
python ingestion/pdf_parser.py
python ingestion/chunker.py
python embeddings/embedder.py

# Run the app
streamlit run app/streamlit_app.py
```