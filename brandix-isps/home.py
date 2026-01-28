"""
Brandix ISPS - Intelligent Strategic Planning Synchronization System
Main Homepage
"""

import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Brandix ISPS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Compatible CSS
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    /* Metric boxes */
    .stMetric, [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
        font-size: 0.875rem !important;
    }
    
    /* Headers */
    h1 {
        color: #4da6ff !important;
        font-weight: 600 !important;
    }
    
    h2, h3, h4 {
        color: #66b3ff !important;
    }
    
    /* Feature boxes */
    .feature-box {
        background-color: rgba(28, 131, 225, 0.08);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(28, 131, 225, 0.2);
        margin: 10px 0;
    }
    
    .feature-box h4 {
        color: #4da6ff !important;
        margin-bottom: 15px;
    }
    
    .feature-box ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .feature-box li {
        color: #e0e0e0;
        padding: 5px 0;
        padding-left: 20px;
        position: relative;
    }
    
    .feature-box li:before {
        content: "▸";
        position: absolute;
        left: 0;
        color: #4da6ff;
    }
    
    /* Info/Success/Warning boxes */
    div[data-baseweb="notification"] {
        background-color: rgba(28, 131, 225, 0.15) !important;
        border-left: 4px solid #1c83e1 !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }
    
    .element-container div[data-testid="stMarkdown"] div[data-baseweb="notification"] {
        background-color: rgba(28, 131, 225, 0.15) !important;
    }
    
    /* Success messages */
    [kind="success"] {
        background-color: rgba(76, 175, 80, 0.15) !important;
        border-left: 4px solid #4caf50 !important;
    }
    
    /* Warning messages */
    [kind="warning"] {
        background-color: rgba(255, 152, 0, 0.15) !important;
        border-left: 4px solid #ff9800 !important;
    }
    
    /* Text visibility */
    p, span, label, li {
        color: #e0e0e0 !important;
    }
    
    /* Strong/bold text */
    strong {
        color: #ffffff !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 20px 0 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: rgba(28, 131, 225, 0.2);
        color: #e0e0e0;
        border: 1px solid rgba(28, 131, 225, 0.3);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: rgba(28, 131, 225, 0.3);
        border: 1px solid rgba(28, 131, 225, 0.5);
        transform: translateY(-2px);
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Links */
    a {
        color: #4da6ff !important;
    }
    
    a:hover {
        color: #66b3ff !important;
    }
    
    /* Code blocks */
    code {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #4da6ff !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Caption text */
    .stCaption {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🎯 Brandix ISPS")
st.markdown("### Intelligent Strategic Planning Synchronization System")
st.markdown("**AI-Powered Strategic Alignment Analysis for Brandix Lanka Limited**")
st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome to Brandix ISPS
    
    The **Intelligent Strategic Planning Synchronization System** is an advanced AI-powered platform 
    designed to analyze and optimize the alignment between Brandix's 5-year Strategic Plan (2025-2030) 
    and annual Action Plans.
    
    #### Key Features:
    
    **📤 Document Upload & Management**
    - Upload and manage strategic plans and action plans by year
    - Support for multi-year analysis (2026-2030)
    - Version control and document tracking
    
    **⚙️ AI-Powered Analysis**
    - Advanced NLP embeddings using Sentence Transformers
    - FAISS vector similarity search
    - Local LLM integration (Phi-3 Mini via Ollama)
    - Comprehensive synchronization scoring
    
    **📊 Detailed Results & Insights**
    - Overall alignment metrics and KPIs
    - Strategy-wise synchronization analysis
    - Gap identification and severity assessment
    - Pillar-level performance breakdown
    
    **🤖 AI-Generated Improvements**
    - Intelligent recommendations for weak alignments
    - New action suggestions
    - KPI enhancements
    - Timeline and resource recommendations
    
    **📋 Executive Summaries**
    - LLM-generated professional reports
    - Key findings and critical gaps
    - Strategic recommendations
    - Risk assessment and next steps
    
    **📈 Multi-Year Comparison**
    - Track alignment progress across years
    - Trend analysis and projections
    - Year-over-year improvements
    - 2030 goal tracking
    """)

with col2:
    st.markdown("""
    ### System Status
    """)
    
    # Check system components
    UPLOAD_BASE = Path("data/uploaded")
    OUTPUTS_BASE = Path("outputs")
    
    UPLOAD_BASE.mkdir(parents=True, exist_ok=True)
    OUTPUTS_BASE.mkdir(parents=True, exist_ok=True)
    
    # Count uploaded years
    AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
    uploaded_years = []
    analyzed_years = []
    
    for year in AVAILABLE_YEARS:
        year_path = UPLOAD_BASE / year
        strategic_exists = (year_path / "strategic_plan.docx").exists()
        action_exists = (year_path / "action_plan.docx").exists()
        
        if strategic_exists and action_exists:
            uploaded_years.append(year)
        
        # Check if analyzed
        results_file = OUTPUTS_BASE / year / "synchronization_report.json"
        if results_file.exists():
            analyzed_years.append(year)
    
    st.metric("Years Uploaded", len(uploaded_years))
    st.metric("Years Analyzed", len(analyzed_years))
    
    if uploaded_years:
        st.success("✅ Ready")
    else:
        st.warning("⏳ Upload Documents")
    
    st.markdown("---")
    
    st.markdown("""
    ### Quick Start
    
    **Step 1:** Upload Documents
    - Go to 📤 Admin Upload
    - Select year
    - Upload plans
    
    **Step 2:** Run Analysis
    - Go to ⚙️ Run Analysis
    - Click Start
    - Wait for completion
    
    **Step 3:** View Results
    - Go to 📊 View Results
    - Explore insights
    - Download reports
    """)

st.markdown("---")

# System Architecture
st.markdown("### 🏗️ System Architecture")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
    <h4>📄 Document Processing</h4>
    <ul>
    <li>DOCX text extraction</li>
    <li>Strategic objective parsing</li>
    <li>Action item identification</li>
    <li>Pillar categorization</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
    <h4>🧮 Embedding & Vector Store</h4>
    <ul>
    <li>Sentence-Transformers (MiniLM)</li>
    <li>384-dimensional embeddings</li>
    <li>FAISS similarity search</li>
    <li>Cosine similarity scoring</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
    <h4>🤖 LLM Integration</h4>
    <ul>
    <li>Ollama + Phi-3 Mini</li>
    <li>RAG pipeline</li>
    <li>Executive summaries</li>
    <li>Improvement suggestions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Navigation Guide
st.markdown("### 🧭 Navigation Guide")

navigation_data = {
    "Page": ["📤 Admin Upload", "⚙️ Run Analysis", "📊 View Results", "📋 Executive Summary", "📈 Multi-Year Comparison"],
    "Purpose": [
        "Upload and manage strategic and action plans by year",
        "Execute AI analysis pipeline (embeddings → alignment → insights)",
        "View detailed synchronization results and metrics",
        "Read LLM-generated executive summaries and recommendations",
        "Compare alignment trends across multiple years"
    ],
    "When to Use": [
        "First time setup, adding new years",
        "After uploading documents",
        "After analysis completes",
        "For high-level strategic insights",
        "When 2+ years are analyzed"
    ]
}

import pandas as pd
df_nav = pd.DataFrame(navigation_data)
st.dataframe(df_nav, use_container_width=True, hide_index=True)

st.markdown("---")

# Technical Details
with st.expander("📚 Technical Specifications"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Technologies Used:**
        - **Frontend:** Streamlit 1.28+
        - **NLP:** Sentence-Transformers
        - **Vector DB:** FAISS
        - **LLM:** Phi-3 Mini (Ollama)
        - **Visualization:** Plotly
        - **Data Processing:** Pandas, NumPy
        - **Document Parsing:** python-docx
        """)
    
    with col2:
        st.markdown("""
        **System Requirements:**
        - Python 3.8+
        - 8GB RAM minimum
        - Ollama installed and running
        - Phi-3 Mini model downloaded
        - 2GB storage for data/models
        
        **Performance:**
        - Analysis time: 2-3 minutes per year
        - Supports up to 100 objectives
        - Handles 50+ action items
        - LLM generation: 30-60 seconds
        """)

st.markdown("---")

# Footer
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📧 Support**
    
    For technical issues or questions:
    - Check documentation
    - Review coursework brief
    - Contact developer
    """)

with col2:
    st.markdown("""
    **🎓 Academic Project**
    
    MSc Computer Science
    Information Retrieval Module
    Brandix Case Study
    """)

with col3:
    st.markdown("""
    **📅 Version Info**
    
    Version: 1.0
    Released: January 2026
    Status: Production Ready
    """)

st.markdown("---")
st.caption("🎯 Brandix ISPS - Intelligent Strategic Planning Synchronization System | Powered by AI")