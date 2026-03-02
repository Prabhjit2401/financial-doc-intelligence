"""
Financial Document Intelligence — Streamlit App
------------------------------------------------
Run with:
    streamlit run app/streamlit_app.py
"""

import sys
sys.path.append(".")

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from rag.chain import ask
from embeddings.embedder import get_embedding_model, get_chroma_collection


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Doc Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');
    .stApp { background-color: #0D0F14; color: #E2E8F0; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .answer-box {
        background: #111520;
        border: 1px solid #1E2432;
        border-left: 4px solid #00C9A7;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        font-size: 15px;
        line-height: 1.8;
        color: #E2E8F0;
    }
    .source-chip {
        display: inline-block;
        background: #1A1F2E;
        border: 1px solid #2D3748;
        border-radius: 6px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 12px;
        color: #94A3B8;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-card {
        background: #111520;
        border: 1px solid #1E2432;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: #00C9A7; }
    .metric-label { font-size: 12px; color: #64748B; margin-top: 4px; }
    .stButton > button {
        background: #00C9A7 !important;
        color: #0D0F14 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSidebar"] { background-color: #0A0C10 !important; border-right: 1px solid #1E2432; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None  # Set when sample button clicked


# ─────────────────────────────────────────────────────────────────────────────
# Cache heavy resources
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = get_embedding_model()
    collection = get_chroma_collection()
    return model, collection


def run_query(query: str, ticker, section, year, n_chunks):
    """Runs the RAG pipeline and appends result to history."""
    with st.spinner("🔍 Retrieving excerpts and generating answer..."):
        try:
            result = ask(
                query=query,
                ticker=None if ticker == "All" else ticker,
                section=None if section == "All" else section,
                year=None if year == "All" else year,
                n_results=n_chunks,
                verbose=False,
            )
            st.session_state.history.append({
                "query": query,
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_used": result["chunks_used"],
            })
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Financial Doc Intelligence")
    st.markdown("<p style='color:#64748B; font-size:13px;'>Query SEC 10-K & 10-Q filings using natural language</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🔧 Filters")
    ticker_filter = st.selectbox("Company", ["All", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"])
    section_filter = st.selectbox(
        "Section",
        ["All", "risk_factors", "mda", "financial_statements", "business", "controls"],
        format_func=lambda x: {
            "All": "All Sections",
            "risk_factors": "Risk Factors",
            "mda": "MD&A",
            "financial_statements": "Financial Statements",
            "business": "Business Overview",
            "controls": "Controls & Procedures",
        }.get(x, x),
    )
    year_filter = st.selectbox("Year", ["All", "2025", "2024", "2023", "2022"])
    n_chunks = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=6)

    st.divider()
    try:
        _, collection = load_resources()
        count = collection.count()
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{count:,}</div>
            <div class='metric-label'>chunks in database</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.warning("Database not loaded")

    st.divider()
    st.markdown("<p style='color:#2D3748; font-size:11px;'>Built with ChromaDB · Groq · Streamlit</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#E2E8F0; margin-bottom:4px;'>Ask your <span style='color:#00C9A7'>10-K</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:24px;'>Query SEC filings with natural language. Answers grounded in real filing excerpts.</p>", unsafe_allow_html=True)

# ── Sample question buttons ──
# These bypass the text input entirely — they set pending_query and rerun
st.markdown("**Try asking:**")
sample_questions = [
    "What are Apple's main risk factors?",
    "How has Apple's revenue trended?",
    "What did management say about competition?",
    "What are the risks related to AI?",
    "How does Apple describe its business?",
]

cols = st.columns(len(sample_questions))
for i, (col, q) in enumerate(zip(cols, sample_questions)):
    with col:
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_query = q

# ── If a sample was clicked, run it immediately (no text input needed) ──
if st.session_state.pending_query:
    query_to_run = st.session_state.pending_query
    st.session_state.pending_query = None  # Clear so it doesn't loop
    run_query(query_to_run, ticker_filter, section_filter, year_filter, n_chunks)

st.markdown("---")

# ── Manual text input + Ask button ──
with st.form(key="query_form", clear_on_submit=True):
    query_input = st.text_input(
        "Your question",
        placeholder="e.g. What risks does Apple face in its supply chain?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask →")

# Using st.form with clear_on_submit=True means:
# - Input clears after submission (clean UX)
# - submitted is only True on actual form submit, not on sample button clicks
if submitted and query_input.strip():
    run_query(query_input.strip(), ticker_filter, section_filter, year_filter, n_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Display latest answer
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.history:
    latest = st.session_state.history[-1]

    st.markdown("### 💬 Answer")
    st.markdown(f"<div class='answer-box'>{latest['answer']}</div>", unsafe_allow_html=True)

    if latest["sources"]:
        st.markdown("### 📎 Sources")
        source_html = ""
        for s in latest["sources"]:
            source_html += (
                f"<span class='source-chip'>"
                f"📄 {s['ticker']} · {s['form']} · {s['filing_date']} · {s['section']} [{s['similarity']:.2f}]"
                f"</span>"
            )
        st.markdown(source_html, unsafe_allow_html=True)

    # Previous questions
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 🕓 Previous Questions")
        for item in reversed(st.session_state.history[:-1]):
            with st.expander(f"❓ {item['query']}"):
                st.markdown(f"<div class='answer-box'>{item['answer']}</div>", unsafe_allow_html=True)
                source_html = ""
                for s in item["sources"]:
                    source_html += f"<span class='source-chip'>📄 {s['ticker']} · {s['filing_date']} · {s['section']}</span>"
                st.markdown(source_html, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center; padding: 60px 0;'>
        <div style='font-size:48px; margin-bottom:16px;'>📂</div>
        <div style='font-family: Syne, sans-serif; font-size:18px; color:#4A5568;'>Ask a question to get started</div>
        <div style='font-size:13px; margin-top:8px; color:#2D3748;'>Answers grounded in real SEC filing excerpts</div>
    </div>
    """, unsafe_allow_html=True)