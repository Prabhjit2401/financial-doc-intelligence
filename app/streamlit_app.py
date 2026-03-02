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
from rag.comparison import compare
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
        white-space: pre-wrap;
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
    .ticker-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 4px 12px;
        margin: 2px;
        font-size: 12px;
        font-weight: 700;
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
    .stTabs [data-baseweb="tab-list"] { background-color: #111520; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #64748B; font-family: 'IBM Plex Mono', monospace; }
    .stTabs [aria-selected="true"] { color: #00C9A7 !important; background: #1A1F2E !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "comparison_history" not in st.session_state:
    st.session_state.comparison_history = []


@st.cache_resource
def load_resources():
    model = get_embedding_model()
    collection = get_chroma_collection()
    return model, collection


TICKER_COLORS = {
    "AAPL": "#64B5F6",
    "MSFT": "#81C784",
    "TSLA": "#FF8A65",
    "GOOGL": "#FFD54F",
    "AMZN": "#CE93D8",
}

ALL_TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Financial Doc Intelligence")
    st.markdown("<p style='color:#64748B; font-size:13px;'>Query SEC 10-K & 10-Q filings using natural language</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🔧 Filters")
    ticker_filter = st.selectbox("Company (single query)", ["All"] + ALL_TICKERS)
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
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#E2E8F0; margin-bottom:4px;'>Ask your <span style='color:#00C9A7'>10-K</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:24px;'>Query SEC filings with natural language. Answers grounded in real filing excerpts.</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Single Company Query", "⚖️ Compare Companies"])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Single Query
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("**Try asking:**")
    sample_questions = [
        "What are the main risk factors?",
        "How has revenue trended recently?",
        "What did management say about competition?",
        "What are the risks related to AI?",
        "How does the company describe its business?",
    ]
    cols = st.columns(len(sample_questions))
    for i, (col, q) in enumerate(zip(cols, sample_questions)):
        with col:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_query = q

    if st.session_state.pending_query:
        query_to_run = st.session_state.pending_query
        st.session_state.pending_query = None
        with st.spinner("🔍 Retrieving excerpts and generating answer..."):
            try:
                result = ask(
                    query=query_to_run,
                    ticker=None if ticker_filter == "All" else ticker_filter,
                    section=None if section_filter == "All" else section_filter,
                    year=None if year_filter == "All" else year_filter,
                    n_results=n_chunks,
                    verbose=False,
                )
                st.session_state.history.append({
                    "query": query_to_run,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "chunks_used": result["chunks_used"],
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    with st.form(key="query_form", clear_on_submit=True):
        query_input = st.text_input(
            "Your question",
            placeholder="e.g. What risks does Apple face in its supply chain?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →")

    if submitted and query_input.strip():
        with st.spinner("🔍 Retrieving excerpts and generating answer..."):
            try:
                result = ask(
                    query=query_input.strip(),
                    ticker=None if ticker_filter == "All" else ticker_filter,
                    section=None if section_filter == "All" else section_filter,
                    year=None if year_filter == "All" else year_filter,
                    n_results=n_chunks,
                    verbose=False,
                )
                st.session_state.history.append({
                    "query": query_input.strip(),
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "chunks_used": result["chunks_used"],
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display latest answer
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

        if len(st.session_state.history) > 1:
            st.markdown("---")
            st.markdown("### 🕓 Previous Questions")
            for item in reversed(st.session_state.history[:-1]):
                with st.expander(f"❓ {item['query']}"):
                    st.markdown(f"<div class='answer-box'>{item['answer']}</div>", unsafe_allow_html=True)
                    for s in item["sources"]:
                        st.markdown(f"<span class='source-chip'>📄 {s['ticker']} · {s['filing_date']} · {s['section']}</span>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 60px 0;'>
            <div style='font-size:48px; margin-bottom:16px;'>📂</div>
            <div style='font-family: Syne, sans-serif; font-size:18px; color:#4A5568;'>Ask a question to get started</div>
            <div style='font-size:13px; margin-top:8px; color:#2D3748;'>Answers grounded in real SEC filing excerpts</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Comparison Mode
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("Select 2–4 companies and ask a question to compare them side by side.")
    st.markdown("---")

    # Company selector
    st.markdown("**Select companies to compare:**")
    ticker_cols = st.columns(len(ALL_TICKERS))
    selected_tickers = []
    for i, (col, ticker) in enumerate(zip(ticker_cols, ALL_TICKERS)):
        with col:
            color = TICKER_COLORS.get(ticker, "#94A3B8")
            checked = st.checkbox(ticker, value=(ticker in ["AAPL", "MSFT"]), key=f"cmp_{ticker}")
            if checked:
                selected_tickers.append(ticker)

    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 companies to compare.")
    else:
        # Show selected badges
        badge_html = ""
        for t in selected_tickers:
            color = TICKER_COLORS.get(t, "#94A3B8")
            badge_html += f"<span class='ticker-badge' style='background:{color}22; border:1px solid {color}; color:{color};'>{t}</span>"
        st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("")

    # Sample comparison questions
    st.markdown("**Try comparing:**")
    comp_samples = [
        "How do these companies approach AI and machine learning?",
        "Compare the main risk factors across these companies",
        "How does revenue growth compare?",
        "What do these companies say about competition?",
    ]
    comp_cols = st.columns(len(comp_samples))
    for i, (col, q) in enumerate(zip(comp_cols, comp_samples)):
        with col:
            if st.button(q, key=f"comp_sample_{i}", use_container_width=True):
                st.session_state["pending_comparison"] = q

    st.markdown("")

    with st.form(key="comparison_form", clear_on_submit=True):
        comp_query = st.text_input(
            "Comparison question",
            placeholder="e.g. How do these companies manage supply chain risk?",
            label_visibility="collapsed",
        )
        comp_year = st.selectbox("Year filter", ["All", "2025", "2024", "2023"])
        comp_submitted = st.form_submit_button("Compare →")

    # Handle pending sample click
    if "pending_comparison" in st.session_state and st.session_state.pending_comparison:
        pending = st.session_state.pending_comparison
        st.session_state.pending_comparison = None
        if len(selected_tickers) >= 2:
            with st.spinner(f"⚖️ Comparing {', '.join(selected_tickers)}..."):
                try:
                    result = compare(
                        query=pending,
                        tickers=selected_tickers,
                        year=None if comp_year == "All" else comp_year,
                        verbose=False,
                    )
                    st.session_state.comparison_history.append(result)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Handle form submit
    if comp_submitted and comp_query.strip() and len(selected_tickers) >= 2:
        with st.spinner(f"⚖️ Comparing {', '.join(selected_tickers)}..."):
            try:
                result = compare(
                    query=comp_query.strip(),
                    tickers=selected_tickers,
                    year=None if comp_year == "All" else comp_year,
                    verbose=False,
                )
                st.session_state.comparison_history.append(result)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Display latest comparison
    if st.session_state.comparison_history:
        latest = st.session_state.comparison_history[-1]

        st.markdown(f"### ⚖️ Comparison: {' vs '.join(latest['tickers'])}")
        st.markdown(f"*{latest['query']}*")

        st.markdown(f"<div class='answer-box'>{latest['answer']}</div>", unsafe_allow_html=True)

        # Sources per company
        st.markdown("### 📎 Sources by Company")
        src_cols = st.columns(len(latest["tickers"]))
        for col, ticker in zip(src_cols, latest["tickers"]):
            with col:
                color = TICKER_COLORS.get(ticker, "#94A3B8")
                st.markdown(f"<span class='ticker-badge' style='background:{color}22; border:1px solid {color}; color:{color}; font-size:14px;'>{ticker}</span>", unsafe_allow_html=True)
                sources = latest.get("sources_by_company", {}).get(ticker, [])
                for s in sources:
                    st.markdown(f"<span class='source-chip'>📄 {s['form']} · {s['filing_date']}<br>{s['section']}</span>", unsafe_allow_html=True)

        # Previous comparisons
        if len(st.session_state.comparison_history) > 1:
            st.markdown("---")
            st.markdown("### 🕓 Previous Comparisons")
            for item in reversed(st.session_state.comparison_history[:-1]):
                label = f"⚖️ {' vs '.join(item['tickers'])}: {item['query']}"
                with st.expander(label):
                    st.markdown(f"<div class='answer-box'>{item['answer']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 60px 0;'>
            <div style='font-size:48px; margin-bottom:16px;'>⚖️</div>
            <div style='font-family: Syne, sans-serif; font-size:18px; color:#4A5568;'>Select companies and ask a question</div>
            <div style='font-size:13px; margin-top:8px; color:#2D3748;'>Side-by-side analysis across SEC filings</div>
        </div>
        """, unsafe_allow_html=True)