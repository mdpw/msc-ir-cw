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

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f4788;
        font-weight: 600;
    }
    h2, h3 {
        color: #2c5aa0;
    }
    .feature-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
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