import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Multi-Year Comparison", page_icon="📈", layout="wide")

st.title("📈 Multi-Year Strategic Alignment Comparison")
st.markdown("### Track Progress Toward 2030 Strategic Goals")
st.markdown("---")

AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
OUTPUTS_BASE = Path("outputs")

# Function to load year data
def load_year_data(year):
    results_file = OUTPUTS_BASE / year / "analysis_results.json"
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
    st.warning("⚠️ No analysis results available yet!")
    st.info("👉 Upload documents and run analysis for at least one year first")
    st.stop()

if len(years_with_data) < 2:
    st.info(f"📊 Currently showing data for {len(years_with_data)} year(s). Analyze more years to see trends!")

st.success(f"✅ Data available for years: **{', '.join(years_with_data)}**")

# Load all data
all_data = {year: load_year_data(year) for year in years_with_data}

st.markdown("---")

# Overall Alignment Trend
st.subheader("📊 Overall Strategic Alignment Trend")

alignment_data = []
for year in years_with_data:
    data = all_data[year]
    alignment_data.append({
        'Year': year,
        'Alignment': data['overall_alignment'],
        'Strong Alignments': data['strong_alignments'],
        'Moderate Alignments': data['moderate_alignments'],
        'Weak Alignments': data['weak_alignments'],
        'Critical Gaps': data['critical_gaps']
    })

df_alignment = pd.DataFrame(alignment_data)

# Line chart for overall alignment
fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=df_alignment['Year'],
    y=df_alignment['Alignment'],
    mode='lines+markers+text',
    name='Overall Alignment',
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=12),
    text=[f"{val:.1f}%" for val in df_alignment['Alignment']],
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

fig_trend.update_layout(
    title="Strategic Alignment Progress (2026-2030)",
    xaxis_title="Year",
    yaxis_title="Alignment Percentage (%)",
    yaxis_range=[0, 100],
    height=400,
    hovermode='x unified'
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
st.subheader("📊 Alignment Distribution by Year")

fig_dist = go.Figure()

for col, name, color in [
    ('Strong Alignments', 'Strong', '#2ca02c'),
    ('Moderate Alignments', 'Moderate', '#ff7f0e'),
    ('Weak Alignments', 'Weak', '#d62728'),
    ('Critical Gaps', 'Critical Gap', '#9467bd')
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
    hovermode='x unified'
)

st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# Pillar-wise comparison (if data available)
st.subheader("🎯 Pillar-wise Alignment Comparison")

# Check if pillar data exists
has_pillar_data = all('pillar_analysis' in all_data[year] for year in years_with_data)

if has_pillar_data:
    pillar_data = []
    
    for year in years_with_data:
        for pillar_name, pillar_info in all_data[year]['pillar_analysis'].items():
            pillar_data.append({
                'Year': year,
                'Pillar': pillar_name,
                'Alignment': pillar_info['average_alignment']
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
        height=500
    )
    
    fig_pillars.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pillars, use_container_width=True)
else:
    st.info("Pillar-wise data not available. Ensure analysis includes pillar breakdown.")

st.markdown("---")

# Year-over-Year Comparison Table
st.subheader("📋 Detailed Year-over-Year Metrics")

comparison_table = df_alignment.copy()
comparison_table['Alignment'] = comparison_table['Alignment'].apply(lambda x: f"{x:.1f}%")

st.dataframe(
    comparison_table,
    use_container_width=True,
    hide_index=True
)

# Download comparison report
st.markdown("---")
st.subheader("📥 Export Comparison Report")

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
    label="📄 Download Comparison Report (Markdown)",
    data=report_content,
    file_name=f"multi_year_comparison_{years_with_data[0]}_to_{years_with_data[-1]}.md",
    mime="text/markdown",
    use_container_width=True
)

st.markdown("---")
st.caption("💡 Continue analyzing subsequent years to build comprehensive trend analysis")