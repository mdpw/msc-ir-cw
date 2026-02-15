"""
Brandix ISPS - Executive Summary Page
LLM-generated professional executive summaries by year
"""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Executive Summary", page_icon=None, layout="wide")

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
    
    /* Info boxes */
    .info-box {
        background-color: rgba(28, 131, 225, 0.08);
        border: 1px solid rgba(28, 131, 225, 0.2);
        border-left: 4px solid #4da6ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #e0e0e0;
    }
    
    /* Summary Section Styling */
    .summary-section {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 20px 0;
    }
    
    .summary-section h3 {
        margin-top: 0;
        color: #4da6ff !important;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
    }
    
    /* Headers */
    h1 {
        color: #4da6ff !important;
        font-weight: 600 !important;
    }
    
    h2, h3, h4 {
        color: #66b3ff !important;
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
    
    /* Buttons */
    .stButton>button {
        width: 100%;
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
    
    /* Primary buttons */
    .stButton>button[kind="primary"] {
        background-color: rgba(28, 131, 225, 0.4);
        border: 1px solid #1c83e1;
    }
    
    /* Text colors */
    p, span, label, li {
        color: #e0e0e0 !important;
    }
    
    strong {
        color: #ffffff !important;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
OUTPUTS_BASE = Path(__file__).parent.parent / "outputs"

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# Header
st.markdown('<h1><i class="fas fa-clipboard-list fa-icon-large"></i>Executive Summary</h1>', unsafe_allow_html=True)
st.markdown("### AI-Generated Strategic Insights")
st.markdown("---")

# Year Selection
col1, col2 = st.columns([1, 3])

with col1:
    selected_year = st.selectbox(
        "Select Year",
        AVAILABLE_YEARS,
        index=AVAILABLE_YEARS.index(st.session_state.selected_year),
        key='exec_year_selector'
    )
    
    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()

with col2:
    st.info(f"Executive Summary for **Year {selected_year}**")

# Load Data
@st.cache_data
def load_executive_summary(year):
    """Load executive summary for a specific year"""
    summary_file = OUTPUTS_BASE / year / 'executive_summary.json'
    
    if not summary_file.exists():
        return None
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load summary
summary = load_executive_summary(selected_year)

if summary is None:
    st.error(f"No executive summary found for year {selected_year}")
    st.info("Go to **'Run Analysis'** page to generate executive summary")
    st.stop()

st.success(f"Executive summary available for {selected_year}")

st.markdown("---")

# Display Summary Sections with Font Awesome Icons
sections = [
    ("fas fa-chart-line", "Executive Overview", "overview"),
    ("fas fa-search", "Key Findings", "key_findings"),
    ("fas fa-exclamation-triangle", "Critical Gaps", "critical_gaps"),
    ("fas fa-lightbulb", "Strategic Recommendations", "recommendations"),
    ("fas fa-shield-alt", "Risk Assessment", "risk_assessment"),
    ("fas fa-forward", "Next Steps", "next_steps")
]

for icon, title, key in sections:
    icon_color = "#f44336" if key == "critical_gaps" else "#4da6ff"
    
    st.markdown(f"""
    <div class="summary-section">
    <h3><i class="{icon}" style="color: {icon_color}; margin-right: 15px;"></i>{title}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    content = summary.get(key, "Not available")
    st.markdown(content)
    st.markdown("---")

# Download Section
st.markdown('<h3><i class="fas fa-download fa-icon"></i>Download Executive Summary</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # JSON download
    json_data = json.dumps(summary, indent=2)
    st.download_button(
        label="Download as JSON",
        data=json_data,
        file_name=f"executive_summary_{selected_year}.json",
        mime="application/json",
        use_container_width=True
    )

with col2:
    # Markdown download
    md_file = OUTPUTS_BASE / selected_year / "executive_summary.md"
    if md_file.exists():
        with open(md_file, 'r', encoding='utf-8') as f:
            md_data = f.read()
        
        st.download_button(
            label="Download as Markdown",
            data=md_data,
            file_name=f"executive_summary_{selected_year}.md",
            mime="text/markdown",
            use_container_width=True
        )

# Navigation
st.markdown("---")
st.markdown('<h3><i class="fas fa-directions fa-icon"></i>Next Steps</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("View Detailed Results", use_container_width=True):
        st.switch_page("pages/03_View_Results.py")

with col2:
    if st.button("Compare Years", use_container_width=True):
        st.switch_page("pages/05_Multi_Year_Comparison.py")

st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-microchip fa-icon-small"></i><strong>Generated by:</strong> Phi-3 Mini via Ollama | <strong>Technology:</strong> Local LLM with RAG Pipeline
</div>
""", unsafe_allow_html=True)
