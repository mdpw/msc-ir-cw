"""
Brandix ISPS - Multi-Year Comparison Page
Track strategic alignment progress across years
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Multi-Year Comparison", page_icon=None, layout="wide")

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
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
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

st.markdown('<h1><i class="fas fa-chart-line fa-icon-large"></i>Multi-Year Strategic Alignment Comparison</h1>', unsafe_allow_html=True)
st.markdown("### Track Progress Toward 2030 Strategic Goals")
st.markdown("---")

AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
OUTPUTS_BASE = Path(__file__).parent.parent / "outputs"

# Function to load year data
def load_year_data(year):
    results_file = OUTPUTS_BASE / year / "synchronization_report.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

# Check which years have data
years_with_data = []
for year in AVAILABLE_YEARS:
    data = load_year_data(year)
    if data:
        years_with_data.append(year)

if not years_with_data:
    st.warning("No analysis results available yet!")
    st.info("Upload documents and run analysis for at least one year first")
    st.stop()

if len(years_with_data) < 2:
    st.info(f"Currently showing data for {len(years_with_data)} year(s). Analyze more years to see trends!")

st.success(f"Data available for years: **{', '.join(years_with_data)}**")

# Load all data
all_data = {year: load_year_data(year) for year in years_with_data}

st.markdown("---")

# Overall Alignment Trend
st.markdown('<h3><i class="fas fa-chart-area fa-icon"></i>Overall Strategic Alignment Trend</h3>', unsafe_allow_html=True)

alignment_data = []
for year in years_with_data:
    data = all_data[year]
    overall = data['overall_alignment']
    alignment_data.append({
        'Year': year,
        'Alignment': overall['overall_score'],
        'Mean Max Similarity': overall['mean_max_similarity'],
        'Strong Alignments': overall['distribution']['strong'],
        'Moderate Alignments': overall['distribution']['moderate'],
        'Weak Alignments': overall['distribution']['weak'],
        'Critical Gaps': overall['distribution']['weak']
    })

df_alignment = pd.DataFrame(alignment_data)

# Line chart for overall alignment
fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=df_alignment['Year'],
    y=df_alignment['Alignment'],
    mode='lines+markers+text',
    name='Overall Alignment',
    line=dict(color='#4da6ff', width=3),
    marker=dict(size=12, color='#4da6ff'),
    text=[f"{val:.1f}%" for val in df_alignment['Alignment']],
    textposition='top center',
    textfont=dict(size=14, color='#ffffff')
))

# Add 2030 target line
fig_trend.add_hline(
    y=75, 
    line_dash="dash", 
    line_color="#2ecc71",
    annotation_text="2030 Target (75%)",
    annotation_position="right",
    annotation_font_color="#2ecc71"
)

fig_trend.update_layout(
    title="Strategic Alignment Progress (2026-2030)",
    xaxis_title="Year",
    yaxis_title="Alignment Percentage (%)",
    yaxis_range=[0, 100],
    height=400,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e0e0e0'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
)

st.plotly_chart(fig_trend, use_container_width=True)

# Show improvement rate
if len(years_with_data) >= 2:
    first_year_alignment = df_alignment.iloc[0]['Alignment']
    last_year_alignment = df_alignment.iloc[-1]['Alignment']
    improvement = last_year_alignment - first_year_alignment
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"Alignment in {years_with_data[0]}", 
            f"{first_year_alignment:.1f}%"
        )
    with col2:
        st.metric(
            f"Alignment in {years_with_data[-1]}", 
            f"{last_year_alignment:.1f}%",
            delta=f"{improvement:+.1f}%"
        )
    with col3:
        years_to_goal = 2030 - int(years_with_data[-1])
        if improvement > 0 and years_to_goal > 0:
            annual_rate = improvement / (len(years_with_data) - 1)
            projected_2030 = last_year_alignment + (annual_rate * years_to_goal)
            st.metric(
                "Projected 2030 Alignment",
                f"{projected_2030:.1f}%",
                help=f"Based on current improvement rate of {annual_rate:.1f}% per year"
            )

st.markdown("---")

# Alignment Distribution Comparison
st.markdown('<h3><i class="fas fa-chart-bar fa-icon"></i>Alignment Distribution by Year</h3>', unsafe_allow_html=True)

fig_dist = go.Figure()

for col, name, color in [
    ('Strong Alignments', 'Strong', '#2ecc71'),
    ('Moderate Alignments', 'Moderate', '#f39c12'),
    ('Weak Alignments', 'Weak', '#e74c3c'),
]:
    fig_dist.add_trace(go.Bar(
        name=name,
        x=df_alignment['Year'],
        y=df_alignment[col],
        marker_color=color
    ))

fig_dist.update_layout(
    title="Alignment Category Distribution",
    xaxis_title="Year",
    yaxis_title="Number of Alignments",
    barmode='stack',
    height=400,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e0e0e0'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
)

st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# Pillar-wise comparison
st.markdown('<h3><i class="fas fa-bullseye fa-icon"></i>Pillar-wise Alignment Comparison</h3>', unsafe_allow_html=True)

has_pillar_data = all('pillar_stats' in all_data[year] for year in years_with_data)

if has_pillar_data:
    pillar_data = []
    
    for year in years_with_data:
        for pillar_name, pillar_info in all_data[year]['pillar_stats'].items():
            pillar_data.append({
                'Year': year,
                'Pillar': pillar_name,
                'Alignment': pillar_info['average_score']
            })
    
    df_pillars = pd.DataFrame(pillar_data)
    
    # Create grouped bar chart
    fig_pillars = px.bar(
        df_pillars,
        x='Pillar',
        y='Alignment',
        color='Year',
        barmode='group',
        title="Pillar Alignment by Year",
        labels={'Alignment': 'Alignment (%)'},
        height=500,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig_pillars.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
    st.plotly_chart(fig_pillars, use_container_width=True)
else:
    st.info("Pillar-wise data not available. Ensure analysis includes pillar breakdown.")

st.markdown("---")

# Year-over-Year Comparison Table
st.markdown('<h3><i class="fas fa-table fa-icon"></i>Detailed Year-over-Year Metrics</h3>', unsafe_allow_html=True)

comparison_table = df_alignment.copy()
comparison_table['Alignment'] = comparison_table['Alignment'].apply(lambda x: f"{x:.1f}%")
comparison_table['Mean Max Similarity'] = comparison_table['Mean Max Similarity'].apply(lambda x: f"{x:.1f}%")

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True
)

# Download comparison report
st.markdown("---")
st.markdown('<h3><i class="fas fa-file-export fa-icon"></i>Export Comparison Report</h3>', unsafe_allow_html=True)

# Create comprehensive report
report_content = f"""# Multi-Year Strategic Alignment Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Years Analyzed: {', '.join(years_with_data)}

## Overall Trend

{df_alignment.to_markdown(index=False)}

## Key Insights

"""

if len(years_with_data) >= 2:
    report_content += f"""
- Starting Alignment ({years_with_data[0]}): {first_year_alignment:.1f}%
- Current Alignment ({years_with_data[-1]}): {last_year_alignment:.1f}%
- Total Improvement: {improvement:+.1f}%
- Average Annual Improvement: {improvement / (len(years_with_data) - 1):.1f}%
"""

st.download_button(
    label="Download Comparison Report (Markdown)",
    data=report_content,
    file_name=f"multi_year_comparison_{years_with_data[0]}_to_{years_with_data[-1]}.md",
    mime="text/markdown",
    use_container_width=True
)

st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-lightbulb fa-icon-small"></i><strong>Tip:</strong> Continue analyzing subsequent years to build comprehensive trend analysis
</div>
""", unsafe_allow_html=True)
