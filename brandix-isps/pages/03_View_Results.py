"""
Brandix ISPS - View Results Page
Year-specific detailed analysis results display
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="View Results", page_icon=None, layout="wide")

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
    
    ul[data-testid="stSelectboxVirtualList"] {
        background-color: #1e2129 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Metric boxes */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #e0e0e0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(77, 166, 255, 0.1) !important;
        border-bottom: 2px solid #4da6ff !important;
        color: #4da6ff !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
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
OUTPUTS_BASE = Path("outputs")

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# Header
st.markdown('<h1><i class="fas fa-chart-bar fa-icon-large"></i>Strategic Alignment Results</h1>', unsafe_allow_html=True)
st.markdown("### Detailed Synchronization Analysis")
st.markdown("---")

# Year Selection
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
    st.info(f"Viewing results for **Year {selected_year}**")

# Load Data
@st.cache_data
def load_year_data(year):
    """Load all data for a specific year"""
    output_dir = OUTPUTS_BASE / year
    
    # Load synchronization report (required)
    report_path = output_dir / 'synchronization_report.json'
    if not report_path.exists():
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    return report

# Load data
report = load_year_data(selected_year)

if report is None:
    st.error(f"No analysis results found for year {selected_year}")
    st.info("Go to **'Run Analysis'** page to analyze this year")
    st.stop()

st.success(f"Analysis completed for {selected_year}")

# Extract data
overall = report['overall_alignment']
objectives = report['objective_details']
gaps = report['gaps']
pillar_stats = report['pillar_stats']

st.markdown("---")

# Navigation Tabs with icons
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Detailed Analysis",
    "Gap Analysis",
    "Pillar View"
])

# ============================================================
# TAB 1: Dashboard
# ============================================================
with tab1:
    st.markdown('<h3><i class="fas fa-tachometer-alt fa-icon"></i>Strategic Synchronization Dashboard</h3>', unsafe_allow_html=True)
    
    # Key Metrics
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
    
    # New Row for Scope Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Objectives", overall.get('total_objectives', len(objectives)))
    with col2:
        st.metric("Total Actions", overall.get('total_actions', 0))
    with col3:
        st.metric("Strategic Pillars", len(pillar_stats))
    with col4:
        # Calculate comparison density
        total_objs = overall.get('total_objectives', len(objectives))
        total_acts = overall.get('total_actions', 0)
        comparisons = total_objs * total_acts
        st.metric("Total Comparisons", f"{comparisons:,}")
    
    st.markdown("---")
    
    # Alignment Distribution
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<h3><i class="fas fa-chart-bar fa-icon"></i>Alignment Distribution</h3>', unsafe_allow_html=True)
        
        dist_data = pd.DataFrame({
            'Category': ['Strong (≥70%)', 'Moderate (50-70%)', 'Weak (<50%)'],
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
            font=dict(color='#e0e0e0'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown('<h3><i class="fas fa-bullseye fa-icon"></i>Alignment Status</h3>', unsafe_allow_html=True)
        
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
            textfont=dict(size=14, color='#e0e0e0')
        )])
        
        fig_donut.update_layout(
            height=350,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            annotations=[dict(
                text=f"{overall['coverage_rate']:.0f}%<br>Coverage", 
                x=0.5, y=0.5, font_size=20, showarrow=False, font_color='#ffffff'
            )]
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    st.markdown("---")
    
    # Pillar Performance
    st.markdown('<h3><i class="fas fa-chart-area fa-icon"></i>Pillar Performance Overview</h3>', unsafe_allow_html=True)
    
    pillar_data = []
    for pillar, stats in pillar_stats.items():
        pillar_data.append({
            'Pillar': pillar,
            'Objectives': stats['count'],
            'Avg Score': stats['average_score'],
            'Status': stats['pillar_status']
        })
    
    df_pillars = pd.DataFrame(pillar_data)
    
    fig_pillars = px.bar(
        df_pillars,
        x='Pillar',
        y='Avg Score',
        color='Status',
        color_discrete_map={'Strong': '#2ecc71', 'Moderate': '#f39c12', 'Weak': '#e74c3c'},
        text='Avg Score',
        title="Average Alignment Score by Pillar"
    )
    
    fig_pillars.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_pillars.update_layout(
        height=400, 
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    
    st.plotly_chart(fig_pillars, use_container_width=True)

# ============================================================
# TAB 2: Detailed Analysis
# ============================================================
with tab2:
    st.markdown('<h3><i class="fas fa-search-plus fa-icon"></i>Strategy-wise Synchronization Analysis</h3>', unsafe_allow_html=True)
    
    # Objective selector
    obj_options = [f"{obj['objective_id']}: {obj['objective'][:60]}..." for obj in objectives]
    selected_obj = st.selectbox("Select Objective to Analyze", obj_options)
    
    # Get selected objective
    obj_idx = obj_options.index(selected_obj)
    obj_detail = objectives[obj_idx]
    
    st.markdown("---")
    
    # Objective Info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Alignment Score", f"{obj_detail['alignment_score']:.1f}%")
    
    with col2:
        st.metric("Coverage", obj_detail['coverage'])
    
    with col3:
        st.metric("Strong Matches", obj_detail['num_strong_matches'])
    
    st.markdown(f"**Pillar:** {obj_detail['pillar']}")
    st.markdown(f"**Objective:** {obj_detail['objective']}")
    
    st.markdown("---")
    
    # Top Matching Actions
    st.markdown('<h4><i class="fas fa-check-double fa-icon-small"></i>Top Matching Actions</h4>', unsafe_allow_html=True)
    
    matches_data = []
    for match in obj_detail['matched_actions'][:10]:
        matches_data.append({
            'Action ID': match['action_id'],
            'Title': match['title'],
            'Pillar': match['pillar'],
            'Similarity': f"{match['similarity']:.1%}",
            'Strength': match['alignment_strength']
        })
    
    df_matches = pd.DataFrame(matches_data)
    # Style status symbols
    def style_strength(val):
        color = '#2ecc71' if val == 'Strong' else '#f39c12' if val == 'Moderate' else '#e74c3c'
        return f'<span style="color: {color}; font-weight: bold;">{val}</span>'
    
    df_matches['Strength'] = df_matches['Strength'].apply(style_strength)
    st.markdown(df_matches.to_html(escape=False, index=False), unsafe_allow_html=True)

# ============================================================
# TAB 3: Gap Analysis
# ============================================================
with tab3:
    st.markdown('<h3><i class="fas fa-exclamation-circle fa-icon"></i>Gap Analysis</h3>', unsafe_allow_html=True)
    
    # Weak Objectives
    st.markdown('### <i class="fas fa-exclamation-triangle fa-icon-small" style="color: #f44336;"></i> Weak Objectives (< 50% Alignment)', unsafe_allow_html=True)
    
    if gaps['weak_objectives']:
        weak_data = []
        for gap in gaps['weak_objectives']:  # Show all gaps
            weak_data.append({
                'ID': gap['objective_id'],
                'Objective': gap['objective'][:80],
                'Pillar': gap['pillar'],
                'Score': f"{gap['alignment_score']:.1f}%",
                'Severity': gap['severity']
            })
        
        df_weak = pd.DataFrame(weak_data)
        st.dataframe(df_weak, use_container_width=True, hide_index=True)
    else:
        st.success("No weak objectives found!")
    
    st.markdown("---")
    
    # Orphan Actions
    st.markdown('### <i class="fas fa-link-slash fa-icon-small" style="color: #4da6ff;"></i> Orphan Actions (Weakly Linked to Strategy)', unsafe_allow_html=True)
    
    if gaps['orphan_actions']:
        orphan_data = []
        for orphan in gaps['orphan_actions']: # Show all orphan actions
            orphan_data.append({
                'Action ID': orphan['action_id'],
                'Title': orphan['title'][:60],
                'Pillar': orphan['pillar'],
                'Max Similarity': f"{orphan['max_similarity']:.1%}"
            })
        
        df_orphan = pd.DataFrame(orphan_data)
        st.dataframe(df_orphan, use_container_width=True, hide_index=True)
    else:
        st.success("No orphan actions found!")

# ============================================================
# TAB 4: Pillar View
# ============================================================
with tab4:
    st.markdown('<h3><i class="fas fa-layer-group fa-icon"></i>Pillar-Level Performance</h3>', unsafe_allow_html=True)
    
    for pillar, stats in pillar_stats.items():
        status_emoji = '🟢' if stats['pillar_status'] == 'Strong' else '🟡' if stats['pillar_status'] == 'Moderate' else '🔴'
        with st.expander(f"{status_emoji} **{pillar}** - {stats['pillar_status']} ({stats['average_score']:.1f}%)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Objectives", stats['count'])
            with col2:
                st.metric("Average Score", f"{stats['average_score']:.1f}%")
            with col3:
                st.metric("Status", stats['pillar_status'])
            
            # Objective list
            st.markdown('**<i class="fas fa-list-ul fa-icon-small"></i>Objectives in this pillar:**', unsafe_allow_html=True)
            for i, (obj_id, obj_text) in enumerate(zip(stats['objective_ids'], stats['objectives']), 1):
                score = stats['scores'][i-1]
                st.markdown(f"{i}. `{obj_id}` - {obj_text} - <span style='color: #4da6ff; font-weight: bold;'>{score:.1f}%</span>", unsafe_allow_html=True)

# Download section
st.markdown("---")
st.markdown('<h3><i class="fas fa-file-export fa-icon"></i>Download Reports</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    results_file = OUTPUTS_BASE / selected_year / "synchronization_report.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            st.download_button(
                label="Download Full Analysis (JSON)",
                data=f.read(),
                file_name=f"analysis_{selected_year}.json",
                mime="application/json",
                use_container_width=True
            )

with col2:
    if st.button("View Executive Summary →", type="primary", use_container_width=True):
        st.switch_page("pages/04_Executive_Summary.py")

st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-lightbulb fa-icon-small"></i><strong>Tip:</strong> Use the tabs above to explore different aspects of the analysis
</div>
""", unsafe_allow_html=True)
