"""
Financial Document Intelligence — HuggingFace Demo
---------------------------------------------------
This is a UI showcase. Run locally with the full stack for live queries.
See: https://github.com/Prabhjit2401/financial-doc-intelligence
"""

import streamlit as st

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
        background: #111520; border: 1px solid #1E2432;
        border-left: 4px solid #00C9A7; border-radius: 8px;
        padding: 20px 24px; margin: 16px 0; font-size: 15px;
        line-height: 1.8; color: #E2E8F0; white-space: pre-wrap;
    }
    .source-chip {
        display: inline-block; background: #1A1F2E; border: 1px solid #2D3748;
        border-radius: 6px; padding: 4px 12px; margin: 4px; font-size: 12px;
        color: #94A3B8; font-family: 'IBM Plex Mono', monospace;
    }
    .ticker-badge {
        display: inline-block; border-radius: 6px; padding: 4px 12px;
        margin: 2px; font-size: 12px; font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    .demo-banner {
        background: #1A1A2E; border: 1px solid #FFD700; border-left: 4px solid #FFD700;
        border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; line-height: 1.9;
    }
    .stat-card {
        background: #111520; border: 1px solid #1E2432; border-radius: 10px;
        padding: 20px; text-align: center; margin: 4px;
    }
    .stat-value { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800; color: #00C9A7; }
    .stat-label { font-size: 12px; color: #64748B; margin-top: 4px; }
    .step-box {
        background: #111520; border: 1px solid #1E2432; border-radius: 8px;
        padding: 16px 20px; margin: 8px 0; font-size: 13px; color: #94A3B8;
    }
    .step-num { color: #00C9A7; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
    .stButton > button { background: #00C9A7 !important; color: #0D0F14 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; }
    div[data-testid="stSidebar"] { background-color: #0A0C10 !important; border-right: 1px solid #1E2432; }
    .stTabs [data-baseweb="tab-list"] { background-color: #111520; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #64748B; font-family: 'IBM Plex Mono', monospace; }
    .stTabs [aria-selected="true"] { color: #00C9A7 !important; background: #1A1F2E !important; border-radius: 6px; }
    code { background: #1A1F2E !important; color: #00C9A7 !important; padding: 2px 6px !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

TICKER_COLORS = {"AAPL": "#64B5F6", "MSFT": "#81C784", "TSLA": "#FF8A65", "GOOGL": "#FFD54F", "AMZN": "#CE93D8"}
ALL_TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]

# Sample answers to show in demo mode
DEMO_ANSWERS = {
    "risk": {
        "query": "What are Apple's main risk factors?",
        "answer": """Apple faces several significant risk factors across its business:

**Supply Chain Concentration**: Apple relies heavily on suppliers in China and Asia for component manufacturing and final assembly. Geopolitical tensions, trade disputes, or natural disasters could disrupt production significantly.

**Market Concentration**: A substantial portion of Apple's revenue comes from iPhone sales. Any slowdown in smartphone demand or failure of a flagship product could materially impact financial results.

**Regulatory & Legal Risk**: Apple faces increasing antitrust scrutiny globally, particularly around the App Store's 30% commission model. The EU's Digital Markets Act and similar regulations in other jurisdictions could force significant business model changes.

**Macroeconomic Sensitivity**: As a premium-priced consumer electronics company, Apple is sensitive to consumer spending slowdowns during economic downturns or periods of high inflation.

**Competitive Pressure**: Intensifying competition from Samsung, Google, and Chinese manufacturers in international markets continues to pressure market share and margins.""",
        "sources": [
            {"ticker": "AAPL", "form": "10-K", "filing_date": "2024-11-01", "section": "Risk Factors", "similarity": 0.89},
            {"ticker": "AAPL", "form": "10-K", "filing_date": "2023-11-03", "section": "Risk Factors", "similarity": 0.84},
        ]
    },
    "comparison": {
        "query": "How do AAPL and MSFT approach AI and machine learning?",
        "tickers": ["AAPL", "MSFT"],
        "answer": """**Apple (AAPL)**
Apple's AI strategy centers on on-device intelligence and privacy-preserving machine learning. In their 2024 10-K, Apple emphasizes Apple Intelligence — a suite of AI features integrated across iOS, macOS, and hardware through their custom silicon (M-series and A-series chips). Apple differentiates by processing AI workloads locally rather than in the cloud, positioning privacy as a competitive advantage.

**Microsoft (MSFT)**
Microsoft has made AI the centerpiece of its growth strategy, with a multi-billion dollar partnership with OpenAI underpinning products across the entire portfolio. Their filings highlight Copilot integration across Microsoft 365, Azure AI services, and GitHub Copilot as key revenue drivers. Microsoft is betting on cloud-delivered AI at enterprise scale.

**COMPARISON**
The strategies are fundamentally different: Apple pursues on-device, privacy-first AI while Microsoft pursues cloud-scale, enterprise AI. Apple's approach has hardware margin implications and tighter control; Microsoft's has faster iteration and broader reach but higher infrastructure costs.

**VERDICT**
Microsoft appears stronger on near-term AI revenue monetization given Copilot's direct integration into its existing enterprise subscription base. Apple holds a long-term advantage in consumer trust and hardware-AI integration as AI moves to the edge.""",
        "sources": {
            "AAPL": [{"form": "10-K", "filing_date": "2024-11-01", "section": "Business Overview", "similarity": 0.87}],
            "MSFT": [{"form": "10-K", "filing_date": "2024-07-30", "section": "MD&A", "similarity": 0.91}],
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Financial Doc Intelligence")
    st.markdown("<p style='color:#64748B; font-size:13px;'>Query SEC 10-K & 10-Q filings using natural language</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🔧 Filters")
    st.selectbox("Company", ["All"] + ALL_TICKERS)
    st.selectbox("Section", ["All Sections", "Risk Factors", "MD&A", "Financial Statements", "Business Overview"])
    st.selectbox("Year", ["All", "2025", "2024", "2023", "2022"])
    st.slider("Chunks to retrieve", 3, 10, 6)
    st.divider()
    st.markdown("""
    <div class='stat-card'>
        <div class='stat-value'>3,200+</div>
        <div class='stat-label'>chunks indexed locally</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**📂 Source Code**")
    st.markdown("[GitHub →](https://github.com/Prabhjit2401/financial-doc-intelligence)")
    st.divider()
    st.markdown("<p style='color:#2D3748; font-size:11px;'>Built with ChromaDB · Groq LLaMA 3.3 · Streamlit</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#E2E8F0; margin-bottom:4px;'>Ask your <span style='color:#00C9A7'>10-K</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:16px;'>Query SEC 10-K & 10-Q filings with natural language. Answers grounded in real filing excerpts with citations.</p>", unsafe_allow_html=True)

# Demo banner
st.markdown("""
<div class='demo-banner'>
    <div style='color:#FFD700; font-size:15px; font-weight:700; margin-bottom:10px;'>⚠️ Hosted Demo — Showing sample outputs</div>
    <div style='color:#94A3B8; font-size:13px;'>
        The vector database requires ~2GB RAM which exceeds HuggingFace's free tier limit.
        This demo shows real sample outputs from a local run over 3,200+ indexed chunks from AAPL, MSFT, TSLA, GOOGL, and AMZN 10-K filings.<br><br>
        To run fully: &nbsp;<code>git clone https://github.com/Prabhjit2401/financial-doc-intelligence</code><br>
        Then follow the README to index filings and query locally.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, "5", "companies indexed"), (c2, "3,200+", "chunks in DB"),
    (c3, "15", "10-K filings"), (c4, "LLaMA 3.3", "LLM backbone")
]:
    with col:
        st.markdown(f"<div class='stat-card'><div class='stat-value' style='font-size:22px;'>{val}</div><div class='stat-label'>{label}</div></div>", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Single Company Query", "⚖️ Compare Companies", "🏗️ How It Works"])

# ══════════════════════════════
# TAB 1 — Demo query
# ══════════════════════════════
with tab1:
    st.markdown("**Try asking:**")
    sample_qs = ["What are Apple's main risk factors?", "How has revenue trended?",
        "What did management say about competition?", "What are risks related to AI?", "Describe the business segments"]
    cols = st.columns(len(sample_qs))
    for i, (col, q) in enumerate(zip(cols, sample_qs)):
        with col:
            st.button(q, key=f"sq_{i}", use_container_width=True)

    st.markdown("---")
    with st.form("qform"):
        qi = st.text_input("Question", placeholder="e.g. What supply chain risks does Apple face?", label_visibility="collapsed")
        st.form_submit_button("Ask →")

    # Always show the demo answer
    demo = DEMO_ANSWERS["risk"]
    st.markdown("### 💬 Sample Answer")
    st.markdown(f"*Query: \"{demo['query']}\"*")
    st.markdown(f"<div class='answer-box'>{demo['answer']}</div>", unsafe_allow_html=True)
    st.markdown("### 📎 Sources")
    for s in demo["sources"]:
        st.markdown(f"<span class='source-chip'>📄 {s['ticker']} · {s['form']} · {s['filing_date']} · {s['section']} [{s['similarity']:.2f}]</span>", unsafe_allow_html=True)

# ══════════════════════════════
# TAB 2 — Demo comparison
# ══════════════════════════════
with tab2:
    st.markdown("Select 2–4 companies and ask a question to compare them side by side.")
    st.markdown("---")
    st.markdown("**Select companies:**")
    tcols = st.columns(len(ALL_TICKERS))
    for col, ticker in zip(tcols, ALL_TICKERS):
        with col:
            st.checkbox(ticker, value=(ticker in ["AAPL", "MSFT"]), key=f"cmp_{ticker}")

    st.markdown("")
    badge_html = "".join(
        f"<span class='ticker-badge' style='background:{TICKER_COLORS[t]}22;border:1px solid {TICKER_COLORS[t]};color:{TICKER_COLORS[t]};'>{t}</span>"
        for t in ["AAPL", "MSFT"]
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown("")
    csamps = ["How do they approach AI?", "Compare main risk factors", "Compare revenue growth", "What do they say about competition?"]
    ccols = st.columns(len(csamps))
    for i, (col, q) in enumerate(zip(ccols, csamps)):
        with col:
            st.button(q, key=f"cs_{i}", use_container_width=True)

    st.markdown("")
    with st.form("cform"):
        st.text_input("Comparison question", placeholder="e.g. How do these companies manage debt risk?", label_visibility="collapsed")
        st.form_submit_button("Compare →")

    demo = DEMO_ANSWERS["comparison"]
    st.markdown(f"### ⚖️ AAPL vs MSFT")
    st.markdown(f"*Query: \"{demo['query']}\"*")
    st.markdown(f"<div class='answer-box'>{demo['answer']}</div>", unsafe_allow_html=True)

    st.markdown("### 📎 Sources by Company")
    scols = st.columns(2)
    for col, ticker in zip(scols, ["AAPL", "MSFT"]):
        with col:
            color = TICKER_COLORS[ticker]
            st.markdown(f"<span class='ticker-badge' style='background:{color}22;border:1px solid {color};color:{color};font-size:14px;'>{ticker}</span>", unsafe_allow_html=True)
            for s in demo["sources"][ticker]:
                st.markdown(f"<span class='source-chip'>📄 {s['form']} · {s['filing_date']}<br>{s['section']} [{s['similarity']:.2f}]</span>", unsafe_allow_html=True)

# ══════════════════════════════
# TAB 3 — Architecture
# ══════════════════════════════
with tab3:
    st.markdown("### 🏗️ How It Works")
    st.markdown("<p style='color:#64748B;'>A production-grade RAG pipeline over SEC filings.</p>", unsafe_allow_html=True)
    st.markdown("")

    steps = [
        ("01", "SEC EDGAR Ingestion", "edgar_fetcher.py downloads 10-K/10-Q filings directly from the SEC EDGAR REST API for any public company. No API key required."),
        ("02", "HTML Parsing + Section Detection", "pdf_parser.py strips XBRL/HTML tags and uses regex patterns to detect standard SEC sections: Risk Factors, MD&A, Financial Statements, Business Overview, and more."),
        ("03", "Section-Aware Chunking", "chunker.py splits text into ~800 character chunks that never cross section boundaries. Each chunk carries rich metadata: ticker, company, filing date, section, form type."),
        ("04", "Embedding + Vector Store", "embedder.py encodes chunks using sentence-transformers (all-MiniLM-L6-v2) and stores them in ChromaDB with cosine similarity indexing. Supports metadata filtering."),
        ("05", "RAG Chain", "chain.py embeds the user query, retrieves the top-N most relevant chunks with optional metadata filters (ticker, section, year), and sends them to Groq's LLaMA 3.3 70B with a citation-enforcing system prompt."),
        ("06", "Multi-Company Comparison", "comparison.py retrieves chunks independently per company, then sends a structured comparison prompt to the LLM asking for per-company paragraphs, a comparison section, and a verdict."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class='step-box'>
            <span class='step-num'>{num}</span> &nbsp;
            <strong style='color:#E2E8F0;'>{title}</strong><br>
            <span style='margin-left: 32px; display:block; margin-top:4px;'>{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 🛠️ Tech Stack")
    cols = st.columns(3)
    stack = [
        ("Data", ["SEC EDGAR API", "BeautifulSoup4", "Section-aware chunking"]),
        ("Retrieval", ["sentence-transformers", "ChromaDB", "Cosine similarity"]),
        ("Generation", ["Groq LLaMA 3.3 70B", "Citation enforcement", "Metadata filtering"]),
    ]
    for col, (category, items) in zip(cols, stack):
        with col:
            items_html = "".join(f"<div style='color:#94A3B8; font-size:12px; margin:4px 0;'>• {item}</div>" for item in items)
            st.markdown(f"""
            <div class='stat-card' style='text-align:left;'>
                <div style='color:#00C9A7; font-size:13px; font-weight:700; margin-bottom:8px;'>{category}</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 🚀 Run Locally")
    st.code("""git clone https://github.com/Prabhjit2401/financial-doc-intelligence
cd financial-doc-intelligence
pip install -r requirements.txt

# Index filings for any company
python ingestion/edgar_fetcher.py --ticker AAPL --filing_type 10-K --max_filings 3
python ingestion/pdf_parser.py
python ingestion/chunker.py
python embeddings/embedder.py

# Ask questions
python rag/chain.py --query "What are the main risk factors?" --ticker AAPL

# Run the web app
streamlit run app/streamlit_app.py""", language="bash")