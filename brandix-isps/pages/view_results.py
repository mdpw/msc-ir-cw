"""
Brandix ISPS - View Results Page
Year-specific analysis results display
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="View Results",
    page_icon="📊",
    layout="wide"
)

# Custom CSS (same as your original)
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
    h2 {
        color: #2c5aa0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Configuration
# ============================================================

AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
OUTPUTS_BASE = Path("outputs")

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# ============================================================
# Year Selection
# ============================================================

st.title("📊 Strategic Alignment Results")
st.markdown("Detailed synchronization analysis results")
st.markdown("---")

# Year selector
col1, col2 = st.columns([1, 3])

with col1:
    selected_year = st.selectbox(
        "Select Year",
        AVAILABLE_YEARS,
        index=AVAILABLE_YEARS.index(st.session_state.selected_year),
        key='results_year_selector'
    )
    
    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()

with col2:
    st.info(f"📅 Viewing results for **Year {selected_year}**")

# ============================================================
# Load Data for Selected Year
# ============================================================

@st.cache_data
def load_year_data(year):
    """Load all data for a specific year"""
    output_dir = OUTPUTS_BASE / year
    
    data = {}
    
    # Load synchronization report
    report_path = output_dir / 'synchronization_report.json'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            data['report'] = json.load(f)
    else:
        return None
    
    # Load improvements (optional)
    improvements_path = output_dir / 'improvements.json'
    if improvements_path.exists():
        with open(improvements_path, 'r', encoding='utf-8') as f:
            data['improvements'] = json.load(f)
    
    # Load executive summary (optional)
    summary_path = output_dir / 'executive_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            data['executive_summary'] = json.load(f)
    
    return data

# Load data
year_data = load_year_data(selected_year)

if year_data is None:
    st.error(f"⚠️ No analysis results found for year {selected_year}")
    st.info("👉 Go to **'Run Analysis'** page to analyze this year")
    st.stop()

report = year_data['report']
improvements_data = year_data.get('improvements')
executive_summary = year_data.get('executive_summary')

# Extract key data
overall = report['overall_alignment']
objectives = report['objective_details']
gaps = report['gaps']
pillar_stats = report['pillar_stats']

st.success(f"✅ Analysis completed for {selected_year}")

st.markdown("---")

# ============================================================
# Navigation Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard",
    "🔍 Detailed Analysis",
    "💡 Gap Analysis",
    "🤖 AI Improvements",
    "📋 Executive Summary",
    "📈 Pillar View",
    "ℹ️ About"
])

# ============================================================
# TAB 1: Dashboard (Your original dashboard page)
# ============================================================

with tab1:
    st.subheader("📊 Strategic Plan Synchronization Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Alignment",
            f"{overall['overall_score']:.1f}%",
            help="Average similarity across all objective-action pairs"
        )
    
    with col2:
        st.metric(
            "Mean Max Similarity",
            f"{overall['mean_max_similarity']:.1f}%",
            delta=f"{overall['classification']}",
            help="Average of best matches for each objective"
        )
    
    with col3:
        st.metric(
            "Well-Covered",
            f"{overall['distribution']['strong'] + overall['distribution']['moderate']}",
            delta=f"{overall['coverage_rate']:.0f}% coverage",
            help="Objectives with ≥50% alignment"
        )
    
    with col4:
        st.metric(
            "Gaps Found",
            overall['distribution']['weak'],
            delta="Needs attention",
            delta_color="inverse",
            help="Objectives with <50% alignment"
        )
    
    st.markdown("---")
    
    # YOUR ORIGINAL DASHBOARD CODE CONTINUES HERE
    # (All the visualizations from your original dashboard page)
    
    # Alignment Distribution
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📊 Alignment Distribution")
        
        dist_data = pd.DataFrame({
            'Category': ['Strong\n(≥70%)', 'Moderate\n(50-70%)', 'Weak\n(<50%)'],
            'Count': [
                overall['distribution']['strong'],
                overall['distribution']['moderate'],
                overall['distribution']['weak']
            ],
            'Color': ['#2ecc71', '#f39c12', '#e74c3c']
        })
        
        fig_dist = go.Figure(data=[
            go.Bar(
                x=dist_data['Category'],
                y=dist_data['Count'],
                marker_color=dist_data['Color'],
                text=dist_data['Count'],
                textposition='auto',
                textfont=dict(size=16, color='white')
            )
        ])
        
        fig_dist.update_layout(
            height=350,
            xaxis_title="Alignment Strength",
            yaxis_title="Number of Objectives",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Alignment Status")
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Strong', 'Moderate', 'Weak'],
            values=[
                overall['distribution']['strong'],
                overall['distribution']['moderate'],
                overall['distribution']['weak']
            ],
            hole=0.5,
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        
        fig_donut.update_layout(
            height=350,
            showlegend=True,
            annotations=[dict(text=f"{overall['coverage_rate']:.0f}%<br>Coverage", 
                             x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    # Continue with rest of dashboard...
    # (Include all your other visualization code here)

# ============================================================
# TAB 2: Detailed Analysis (Your original detailed analysis)
# ============================================================

with tab2:
    # YOUR ORIGINAL DETAILED ANALYSIS CODE
    st.subheader("🔍 Strategy-wise Synchronization Analysis")
    # ... (rest of your code)

# Continue for all other tabs...