"""
Brandix ISPS - Multi-Year Strategic Planning Dashboard
Intelligent Strategic Planning Synchronization System
Home Page - Year Overview
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ISPS - Home",
    page_icon="🏠",
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
    h2 {
        color: #2c5aa0;
    }
    .year-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Configuration
# ============================================================

AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
UPLOAD_BASE = Path("data/uploaded")
OUTPUTS_BASE = Path("outputs")

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# ============================================================
# Functions
# ============================================================

def get_year_status(year):
    """Get comprehensive status for a given year"""
    year_path = UPLOAD_BASE / year
    outputs_path = OUTPUTS_BASE / year
    
    strategic_exists = (year_path / "strategic_plan.docx").exists()
    action_exists = (year_path / "action_plan.docx").exists()
    analysis_exists = (outputs_path / "synchronization_report.json").exists()
    
    # Load metadata
    metadata_path = year_path / "metadata.json"
    upload_date = None
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                upload_date = metadata.get('upload_date', 'Unknown')
        except:
            upload_date = "Unknown"
    
    # Load analysis results if available
    alignment_score = None
    if analysis_exists:
        try:
            with open(outputs_path / "synchronization_report.json", 'r') as f:
                report = json.load(f)
                alignment_score = report['overall_alignment']['overall_score']
        except:
            alignment_score = None
    
    return {
        'strategic_uploaded': strategic_exists,
        'action_uploaded': action_exists,
        'analysis_complete': analysis_exists,
        'upload_date': upload_date,
        'ready': strategic_exists and action_exists,
        'alignment_score': alignment_score
    }

def get_system_statistics():
    """Calculate system-wide statistics"""
    total_years = len(AVAILABLE_YEARS)
    years_with_data = 0
    years_analyzed = 0
    alignment_scores = []
    
    for year in AVAILABLE_YEARS:
        status = get_year_status(year)
        if status['ready']:
            years_with_data += 1
        if status['analysis_complete']:
            years_analyzed += 1
            if status['alignment_score']:
                alignment_scores.append(status['alignment_score'])
    
    avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
    completion_rate = (years_analyzed / total_years * 100) if total_years > 0 else 0
    
    return {
        'total_years': total_years,
        'years_with_data': years_with_data,
        'years_analyzed': years_analyzed,
        'completion_rate': completion_rate,
        'avg_alignment': avg_alignment,
        'alignment_scores': alignment_scores
    }

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🎯 Brandix ISPS")
st.sidebar.markdown("**Intelligent Strategic Planning Synchronization**")
st.sidebar.markdown("**Multi-Year Tracking System**")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📅 Strategic Period")
st.sidebar.info("**2025-2030**\n5-Year Strategic Plan")

st.sidebar.markdown("---")

# System statistics in sidebar
stats = get_system_statistics()
st.sidebar.markdown("### 📊 System Overview")
st.sidebar.metric("Years Analyzed", f"{stats['years_analyzed']}/{stats['total_years']}")
st.sidebar.metric("Completion Rate", f"{stats['completion_rate']:.0f}%")
if stats['avg_alignment'] > 0:
    st.sidebar.metric("Avg Alignment", f"{stats['avg_alignment']:.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Quick Links")
st.sidebar.page_link("pages/admin_upload.py", label="📤 Upload Documents")
st.sidebar.page_link("pages/run_analysis.py", label="⚙️ Run Analysis")
st.sidebar.page_link("pages/view_results.py", label="📊 View Results")

# ============================================================
# Main Content
# ============================================================

st.title("🏠 Brandix ISPS - Multi-Year Dashboard")
st.markdown("### Strategic Planning Synchronization System (2025-2030)")
st.markdown("Track alignment progress across multiple years of your strategic plan")

st.markdown("---")

# ============================================================
# System Statistics Overview
# ============================================================

st.subheader("📈 System Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Planning Period", 
        "2025-2030",
        help="5-year strategic planning period"
    )
    
with col2:
    st.metric(
        "Years with Data", 
        f"{stats['years_with_data']}/{stats['total_years']}",
        help="Years with both strategic and action plans uploaded"
    )
    
with col3:
    st.metric(
        "Years Analyzed", 
        f"{stats['years_analyzed']}/{stats['total_years']}",
        help="Years with completed synchronization analysis"
    )
    
with col4:
    st.metric(
        "Completion", 
        f"{stats['completion_rate']:.0f}%",
        help="Percentage of years with completed analysis"
    )

# Progress bar
if stats['years_analyzed'] > 0:
    progress = stats['years_analyzed'] / stats['total_years']
    st.progress(progress)
    st.caption(f"System Progress: {stats['years_analyzed']} of {stats['total_years']} years analyzed")

st.markdown("---")

# ============================================================
# Year Overview Cards
# ============================================================

st.subheader("📅 Year-by-Year Status")

# Create columns for year cards
cols = st.columns(len(AVAILABLE_YEARS))

for idx, year in enumerate(AVAILABLE_YEARS):
    with cols[idx]:
        status = get_year_status(year)
        
        # Determine card status
        if status['analysis_complete']:
            card_emoji = "✅"
            card_status = "Complete"
            card_color = "green"
        elif status['ready']:
            card_emoji = "📋"
            card_status = "Ready"
            card_color = "blue"
        elif status['strategic_uploaded'] or status['action_uploaded']:
            card_emoji = "⏳"
            card_status = "Partial"
            card_color = "orange"
        else:
            card_emoji = "⚪"
            card_status = "Not Started"
            card_color = "gray"
        
        # Year card
        st.markdown(f"### {card_emoji} {year}")
        st.markdown(f"**Status:** :{card_color}[{card_status}]")
        
        # Show alignment score if available
        if status['alignment_score']:
            st.metric("Alignment", f"{status['alignment_score']:.1f}%")
        
        # Progress indicators
        if status['strategic_uploaded']:
            st.success("📋 Strategic ✓", icon="✅")
        else:
            st.error("📋 Strategic ✗", icon="❌")
        
        if status['action_uploaded']:
            st.success("📅 Action ✓", icon="✅")
        else:
            st.error("📅 Action ✗", icon="❌")
        
        if status['analysis_complete']:
            st.success("📊 Analysis ✓", icon="✅")
        else:
            st.info("📊 Analysis ✗", icon="ℹ️")
        
        # Upload date
        if status['upload_date']:
            st.caption(f"📆 {status['upload_date']}")
        
        # Action button
        if status['analysis_complete']:
            if st.button("📊 View", key=f"view_{year}", use_container_width=True):
                st.session_state.selected_year = year
                st.switch_page("pages/view_results.py")
        elif status['ready']:
            if st.button("▶️ Analyze", key=f"analyze_{year}", use_container_width=True):
                st.session_state.selected_year = year
                st.switch_page("pages/run_analysis.py")
        else:
            if st.button("📤 Upload", key=f"upload_{year}", use_container_width=True):
                st.session_state.selected_year = year
                st.switch_page("pages/admin_upload.py")

st.markdown("---")

# ============================================================
# Alignment Trend Chart
# ============================================================

if stats['years_analyzed'] >= 2:
    st.subheader("📈 Strategic Alignment Trend")
    
    # Gather alignment data
    trend_data = []
    for year in AVAILABLE_YEARS:
        status = get_year_status(year)
        if status['alignment_score']:
            trend_data.append({
                'Year': year,
                'Alignment': status['alignment_score']
            })
    
    if len(trend_data) >= 2:
        df_trend = pd.DataFrame(trend_data)
        
        # Create line chart
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=df_trend['Year'],
            y=df_trend['Alignment'],
            mode='lines+markers+text',
            name='Overall Alignment',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=12),
            text=[f"{val:.1f}%" for val in df_trend['Alignment']],
            textposition='top center',
            textfont=dict(size=14, color='#1f77b4')
        ))
        
        # Add 2030 target line
        fig_trend.add_hline(
            y=75, 
            line_dash="dash", 
            line_color="green",
            annotation_text="2030 Target (75%)",
            annotation_position="right"
        )
        
        # Add current average
        fig_trend.add_hline(
            y=stats['avg_alignment'],
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Current Avg ({stats['avg_alignment']:.1f}%)",
            annotation_position="left"
        )
        
        fig_trend.update_layout(
            title="Strategic Alignment Progress Toward 2030 Goals",
            xaxis_title="Year",
            yaxis_title="Alignment Percentage (%)",
            yaxis_range=[0, 100],
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Improvement analysis
        if len(trend_data) >= 2:
            first_alignment = trend_data[0]['Alignment']
            last_alignment = trend_data[-1]['Alignment']
            improvement = last_alignment - first_alignment
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Starting ({trend_data[0]['Year']})",
                    f"{first_alignment:.1f}%"
                )
            
            with col2:
                st.metric(
                    f"Current ({trend_data[-1]['Year']})",
                    f"{last_alignment:.1f}%",
                    delta=f"{improvement:+.1f}%"
                )
            
            with col3:
                # Project to 2030
                if improvement > 0:
                    years_elapsed = len(trend_data) - 1
                    annual_rate = improvement / years_elapsed
                    years_to_2030 = 2030 - int(trend_data[-1]['Year'])
                    projected_2030 = last_alignment + (annual_rate * years_to_2030)
                    
                    st.metric(
                        "Projected 2030",
                        f"{projected_2030:.1f}%",
                        delta=f"{annual_rate:.1f}%/year",
                        help="Based on current improvement rate"
                    )

st.markdown("---")

# ============================================================
# Getting Started Guide
# ============================================================

st.subheader("🚀 Getting Started")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📝 For New Users:")
    st.markdown("""
    1. **Upload Documents** 📤
       - Go to Admin Upload page
       - Select a year (e.g., 2026)
       - Upload Strategic Plan and Action Plan
    
    2. **Run Analysis** ⚙️
       - Go to Run Analysis page
       - Execute step-by-step analysis
       - Wait for AI processing (2-3 minutes)
    
    3. **View Results** 📊
       - Go to View Results page
       - Explore detailed insights
       - Export reports
    """)

with col2:
    st.markdown("#### 📈 For Multi-Year Tracking:")
    st.markdown("""
    1. **Complete First Year**
       - Analyze Year 1 (2026) completely
       - Review results and gaps
    
    2. **Add Subsequent Years**
       - Upload Year 2 (2027) action plan
       - Run new analysis
       - Compare with previous years
    
    3. **Track Progress**
       - Use Multi-Year Comparison page
       - Monitor alignment trends
       - Measure strategic progress
    """)

st.markdown("---")

# ============================================================
# Recent Activity
# ============================================================

st.subheader("📌 Recent Activity")

# Gather all years with activity
recent_activity = []
for year in AVAILABLE_YEARS:
    status = get_year_status(year)
    if status['upload_date'] and status['upload_date'] != 'Unknown':
        recent_activity.append({
            'year': year,
            'date': status['upload_date'],
            'status': status
        })

if recent_activity:
    # Sort by date (most recent first)
    recent_activity.sort(key=lambda x: x['date'], reverse=True)
    
    # Show last 5 activities
    for item in recent_activity[:5]:
        year = item['year']
        date = item['date']
        status = item['status']
        
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            st.markdown(f"**{year}**")
        
        with col2:
            st.markdown(f"📅 {date}")
        
        with col3:
            if status['analysis_complete']:
                st.success(f"✅ Analysis Complete ({status['alignment_score']:.1f}%)")
            elif status['ready']:
                st.info("📋 Documents Uploaded - Ready to Analyze")
            else:
                st.warning("⏳ Partial Upload")
else:
    st.info("👋 No activity yet. Start by uploading documents for your first year!")

st.markdown("---")

# ============================================================
# Quick Actions
# ============================================================

st.subheader("⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📤 Upload New Year", use_container_width=True, type="primary"):
        st.switch_page("pages/admin_upload.py")

with col2:
    if stats['years_with_data'] > 0:
        if st.button("⚙️ Run Analysis", use_container_width=True):
            st.switch_page("pages/run_analysis.py")
    else:
        st.button("⚙️ Run Analysis", use_container_width=True, disabled=True)

with col3:
    if stats['years_analyzed'] > 0:
        if st.button("📊 View Results", use_container_width=True):
            st.switch_page("pages/view_results.py")
    else:
        st.button("📊 View Results", use_container_width=True, disabled=True)

with col4:
    if stats['years_analyzed'] >= 2:
        if st.button("📈 Compare Years", use_container_width=True):
            st.switch_page("pages/multi_year_comparison.py")
    else:
        st.button("📈 Compare Years", use_container_width=True, disabled=True)

# ============================================================
# Footer
# ============================================================

st.markdown("---")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.caption("🎯 **Brandix ISPS** - Intelligent Strategic Planning Synchronization System")
    st.caption("Strategic Period: 2025-2030 | Multi-Year Tracking Platform")

with col2:
    st.caption("📚 **Documentation**")
    st.caption("[GitHub](#) | [Help](#)")

with col3:
    st.caption("💡 **Support**")
    st.caption("[Report Issues](#)")

st.markdown("---")
st.caption("© 2025 Brandix ISPS. All rights reserved.")