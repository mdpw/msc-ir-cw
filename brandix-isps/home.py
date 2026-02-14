"""
Brandix ISPS - Intelligent Strategic Planning Synchronization System
Main Homepage
"""

import streamlit as st
from pathlib import Path
import base64

# Helper function to encode image
def get_base64_image(image_path):
    """Convert image to base64 for HTML embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

# Page config
st.set_page_config(
    page_title="Brandix ISPS",
    page_icon="Brandix_Logo_Small.png",  # Use image instead of emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Compatible CSS + Font Awesome Icons
st.markdown("""
    <style>
    /* Import Font Awesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    /* Icon styling */
    .fa-icon {
        color: #4da6ff;
        margin-right: 8px;
    }
    
    .fa-icon-large {
        font-size: 1.2em;
        color: #4da6ff;
        margin-right: 10px;
    }
    
    .fa-icon-small {
        font-size: 0.9em;
        color: #4da6ff;
        margin-right: 6px;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .logo-img {
        height: 55px;
        margin-right: 15px;
        border-radius: 5px;
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
    
    /* Sidebar explicit dark mode */
    [data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebarNav"] span {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    
    /* Top Header */
    header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
        background: transparent !important;
    }
    
    /* Dropdown/Selectbox dark mode */
    div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    ul[data-testid="stSelectboxVirtualList"] {
        background-color: #1e2129 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header with Logo
logo_base64 = get_base64_image("pages/brandix_logo.png")

if logo_base64:
    st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" class="logo-img">
            <h1 style="margin: 0;">
                Brandix ISPS
            </h1>
        </div>
    """, unsafe_allow_html=True)
else:
    # Fallback if logo not found
    st.markdown('<h1><i class="fas fa-bullseye fa-icon-large"></i>Brandix ISPS</h1>', unsafe_allow_html=True)

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
    
    #### Key Features
    """)
    
    # 2-column grid for features
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-upload fa-icon-small"></i>Document Management</h4>
        <ul>
        <li>Upload strategic & action plans</li>
        <li>Multi-year analysis support</li>
        <li>Version control & document tracking</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-chart-bar fa-icon-small"></i>Results & Insights</h4>
        <ul>
        <li>Alignment metrics & KPIs</li>
        <li>Strategy-wise sync analysis</li>
        <li>Gap & severity assessment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-clipboard fa-icon-small"></i>Executive Summaries</h4>
        <ul>
        <li>Professional strategic reports</li>
        <li>Critical gap identification</li>
        <li>Leadership recommendations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-network-wired fa-icon-small"></i>Knowledge Graph</h4>
        <ul>
        <li>Interactive network visuals</li>
        <li>Relationship discovery</li>
        <li>Similarity threshold filtering</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with f_col2:
        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-cog fa-icon-small"></i>AI-Powered Analysis</h4>
        <ul>
        <li>Sentence Transformer NLP</li>
        <li>FAISS vector similarity search</li>
        <li>Local LLM (Phi-3 Mini)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-robot fa-icon-small"></i>AI Improvements</h4>
        <ul>
        <li>Intelligent gap solutions</li>
        <li>New action & KPI suggestions</li>
        <li>Resource & risk mitigation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-chart-line fa-icon-small"></i>Multi-Year Comparison</h4>
        <ul>
        <li>Track progress across years</li>
        <li>Trend analysis & projections</li>
        <li>2030 goal tracking</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-box">
        <h4><i class="fas fa-vial fa-icon-small"></i>Testing & Evaluation</h4>
        <ul>
        <li>Ground truth comparison</li>
        <li>Precision/Recall/F1 metrics</li>
        <li>Expert validation framework</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

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
        st.success("Ready")
    else:
        st.warning("Upload Documents")
    
    st.markdown("---")
    
    st.markdown("""
    ### Quick Start
    
    **Step 1:** Upload Documents
    - Go to <i class="fas fa-upload fa-icon-small"></i>Admin Upload
    
    **Step 2:** Run Analysis
    - Go to <i class="fas fa-cog fa-icon-small"></i>Run Analysis
    
    **Step 3:** View Results
    - Go to <i class="fas fa-chart-bar fa-icon-small"></i>View Results
    - Explore <i class="fas fa-network-wired fa-icon-small"></i>Knowledge Graph
    
    **Step 4:** Evaluate
    - Go to <i class="fas fa-vial fa-icon-small"></i>Testing & Evaluation
    
    **Step 5:** Strategic Insights
    - Go to <i class="fas fa-clipboard fa-icon-small"></i>Executive Summary
    
    **Step 6:** Track Progress
    - Go to <i class="fas fa-chart-line fa-icon-small"></i>Multi-Year Comparison
    """, unsafe_allow_html=True)

st.markdown("---")

# System Architecture
st.markdown('<h3><i class="fas fa-building fa-icon"></i>System Architecture</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
    <h4><i class="fas fa-file-alt fa-icon-small"></i>Document Processing</h4>
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
    <h4><i class="fas fa-calculator fa-icon-small"></i>Embedding & Vector Store</h4>
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
    <h4><i class="fas fa-robot fa-icon-small"></i>LLM Integration</h4>
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
st.markdown('<h3><i class="fas fa-compass fa-icon"></i>Navigation Guide</h3>', unsafe_allow_html=True)

navigation_data = {
    "Page": [
        '<i class="fas fa-upload fa-icon-small"></i>Admin Upload',
        '<i class="fas fa-cog fa-icon-small"></i>Run Analysis',
        '<i class="fas fa-chart-bar fa-icon-small"></i>View Results',
        '<i class="fas fa-network-wired fa-icon-small"></i>Knowledge Graph',
        '<i class="fas fa-vial fa-icon-small"></i>Testing & Evaluation',
        '<i class="fas fa-clipboard fa-icon-small"></i>Executive Summary',
        '<i class="fas fa-chart-line fa-icon-small"></i>Multi-Year Comparison'
    ],
    "Purpose": [
        "Upload and manage strategic and action plans by year",
        "Execute AI analysis pipeline (embeddings → alignment → insights)",
        "View detailed synchronization results and metrics",
        "Interactive visualization of strategic networks",
        "System evaluation and metrics (Precision/Recall)",
        "Read LLM-generated executive summaries and recommendations",
        "Compare alignment trends across multiple years"
    ],
    "When to Use": [
        "First time setup, adding new years",
        "After uploading documents",
        "After analysis completes",
        "Visualizing complex relationships",
        "Validating system accuracy",
        "For high-level strategic insights",
        "When 2+ years are analyzed"
    ]
}

import pandas as pd
df_nav = pd.DataFrame(navigation_data)
st.markdown(df_nav.to_html(escape=False, index=False), unsafe_allow_html=True)

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
        - **Graphing:** NetworkX, Plotly
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
st.markdown('<p style="text-align: center; color: rgba(255, 255, 255, 0.5);"><i class="fas fa-bullseye fa-icon-small"></i>Brandix ISPS - Intelligent Strategic Planning Synchronization System | Powered by AI</p>', unsafe_allow_html=True)
