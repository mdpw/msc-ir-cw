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

st.set_page_config(page_title="View Results", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {color: #1f4788; font-weight: 600;}
    h2, h3 {color: #2c5aa0;}
    </style>
""", unsafe_allow_html=True)

# Configuration
AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
OUTPUTS_BASE = Path("outputs")

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# Header
st.title("📊 Strategic Alignment Results")
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
    st.info(f"📅 Viewing results for **Year {selected_year}**")

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
    st.error(f"⚠️ No analysis results found for year {selected_year}")
    st.info("👉 Go to **'⚙️ Run Analysis'** page to analyze this year")
    st.stop()

st.success(f"✅ Analysis completed for {selected_year}")

# Extract data
overall = report['overall_alignment']
objectives = report['objective_details']
gaps = report['gaps']
pillar_stats = report['pillar_stats']

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔍 Detailed Analysis",
    "💡 Gap Analysis",
    "📈 Pillar View"
])

# ============================================================
# TAB 1: Dashboard
# ============================================================
with tab1:
    st.subheader("📊 Strategic Synchronization Dashboard")
    
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
    
    st.markdown("---")
    
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
            annotations=[dict(
                text=f"{overall['coverage_rate']:.0f}%<br>Coverage", 
                x=0.5, y=0.5, font_size=20, showarrow=False
            )]
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)
    
    st.markdown("---")
    
    # Pillar Performance
    st.subheader("🎯 Pillar Performance Overview")
    
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
    fig_pillars.update_layout(height=400, xaxis_tickangle=-45)
    
    st.plotly_chart(fig_pillars, use_container_width=True)

# ============================================================
# TAB 2: Detailed Analysis
# ============================================================
with tab2:
    st.subheader("🔍 Strategy-wise Synchronization Analysis")
    
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
    st.subheader("🎯 Top Matching Actions")
    
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
    st.dataframe(df_matches, use_container_width=True, hide_index=True)

# ============================================================
# TAB 3: Gap Analysis
# ============================================================
with tab3:
    st.subheader("💡 Gap Analysis")
    
    # Weak Objectives
    st.markdown("### 🔴 Weak Objectives (< 50% Alignment)")
    
    if gaps['weak_objectives']:
        weak_data = []
        for gap in gaps['weak_objectives'][:20]:  # Show top 20
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
        st.success("✅ No weak objectives found!")
    
    st.markdown("---")
    
    # Orphan Actions
    st.markdown("### 🔵 Orphan Actions (Weakly Linked to Strategy)")
    
    if gaps['orphan_actions']:
        orphan_data = []
        for orphan in gaps['orphan_actions'][:20]:
            orphan_data.append({
                'Action ID': orphan['action_id'],
                'Title': orphan['title'][:60],
                'Pillar': orphan['pillar'],
                'Max Similarity': f"{orphan['max_similarity']:.1%}"
            })
        
        df_orphan = pd.DataFrame(orphan_data)
        st.dataframe(df_orphan, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No orphan actions found!")

# ============================================================
# TAB 4: Pillar View
# ============================================================
with tab4:
    st.subheader("📈 Pillar-Level Performance")
    
    for pillar, stats in pillar_stats.items():
        with st.expander(f"**{pillar}** - {stats['pillar_status']} ({stats['average_score']:.1f}%)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Objectives", stats['count'])
            with col2:
                st.metric("Average Score", f"{stats['average_score']:.1f}%")
            with col3:
                st.metric("Status", stats['pillar_status'])
            
            # Objective list
            st.markdown("**Objectives in this pillar:**")
            for i, (obj_id, obj_text) in enumerate(zip(stats['objective_ids'], stats['objectives']), 1):
                score = stats['scores'][i-1]
                st.markdown(f"{i}. `{obj_id}` - {obj_text} - **{score:.1f}%**")

# Download section
st.markdown("---")
st.subheader("📥 Download Reports")

col1, col2 = st.columns(2)

with col1:
    results_file = OUTPUTS_BASE / selected_year / "synchronization_report.json"
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            st.download_button(
                label="📄 Download Full Analysis (JSON)",
                data=f.read(),
                file_name=f"analysis_{selected_year}.json",
                mime="application/json",
                use_container_width=True
            )

with col2:
    if st.button("📋 View Executive Summary →", type="primary", use_container_width=True):
        st.switch_page("pages/04_Executive_Summary.py")

st.markdown("---")
st.caption("💡 **Tip:** Use the tabs above to explore different aspects of the analysis")