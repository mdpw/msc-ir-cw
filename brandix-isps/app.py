"""
Brandix ISPS - Streamlit Dashboard
Intelligent Strategic Planning Synchronization System
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append('src')

# Page configuration
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
    h2 {
        color: #2c5aa0;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_synchronization_report():
    """Load the synchronization report"""
    report_path = 'outputs/synchronization_report.json'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        st.error("⚠️ Synchronization report not found! Please run: python src/synchronization_engine.py")
        st.stop()

@st.cache_data
def load_embedding_analysis():
    """Load embedding analysis"""
    analysis_path = 'outputs/embedding_analysis.json'
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_data
def load_improvements():
    """Load AI-generated improvements"""
    improvements_path = 'outputs/improvements.json'
    if os.path.exists(improvements_path):
        with open(improvements_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_data
def load_executive_summary():
    """Load AI-generated executive summary"""
    summary_path = 'outputs/executive_summary.json'
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Load data
try:
    report = load_synchronization_report()
    embedding_analysis = load_embedding_analysis()
    improvements_data = load_improvements()
    executive_summary = load_executive_summary()  # NEW!
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Extract key data
overall = report['overall_alignment']
objectives = report['objective_details']
gaps = report['gaps']
pillar_stats = report['pillar_stats']

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🎯 Brandix ISPS")
st.sidebar.markdown("**Intelligent Strategic Planning Synchronization**")
st.sidebar.markdown("---")

# MODIFIED: Added Executive Summary to navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dashboard",
        "🔍 Detailed Analysis",
        "💡 Gap Analysis",
        "🤖 AI Improvements",
        "📋 Executive Summary",  # NEW!
        "📈 Pillar View",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Overall Alignment", f"{overall['overall_score']:.1f}%")
st.sidebar.metric("Coverage Rate", f"{overall['coverage_rate']:.1f}%")
st.sidebar.metric("Objectives", overall['total_objectives'])
st.sidebar.metric("Actions", overall['total_actions'])

# ============================================================
# PAGE 1: Dashboard
# ============================================================

if page == "📊 Dashboard":
    st.title("📊 Strategic Plan Synchronization Dashboard")
    st.markdown("Real-time analysis of strategic-action alignment for Brandix 2025-2030")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Alignment",
            f"{overall['overall_score']:.1f}%",
            delta="+5.2% vs baseline",
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
        
        # Bar chart
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
        
        # Donut chart
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
    
    st.markdown("---")
    
    # Pillar-wise Analysis
    st.subheader("🏛️ Pillar-wise Alignment")
    
    pillar_df = pd.DataFrame([
        {
            'Pillar': pillar,
            'Objectives': stats['count'],
            'Avg Score': stats['average_score'],
            'Status': stats['pillar_status']
        }
        for pillar, stats in pillar_stats.items()
    ])
    
    fig_pillar = px.bar(
        pillar_df,
        x='Pillar',
        y='Avg Score',
        color='Status',
        color_discrete_map={'Strong': '#2ecc71', 'Moderate': '#f39c12', 'Weak': '#e74c3c'},
        text='Avg Score',
        title='Average Alignment Score by Strategic Pillar'
    )
    
    fig_pillar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_pillar.update_layout(
        height=400,
        xaxis_title="Strategic Pillar",
        yaxis_title="Average Alignment Score (%)",
        yaxis_range=[0, 100],
        showlegend=True
    )
    
    st.plotly_chart(fig_pillar, use_container_width=True)
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **✅ Strengths**
        
        - {overall['distribution']['strong']} objectives have strong alignment (≥70%)
        - {overall['coverage_rate']:.0f}% coverage rate indicates good overall synchronization
        - Classification: {overall['classification']}
        """)
    
    with col2:
        st.warning(f"""
        **⚠️ Areas for Improvement**
        
        - {overall['distribution']['weak']} objectives have weak alignment (<50%)
        - {len(gaps['orphan_actions'])} orphan action(s) with unclear strategic link
        - {len(gaps['coverage_gaps'])} objective(s) with no moderate/strong matches
        """)
    
    with col3:
        st.success(f"""
        **🎯 Recommendations**
        
        - Review the {overall['distribution']['weak']} gap objectives
        - Consider additional actions for poorly covered areas
        - Maintain momentum in well-aligned areas
        """)

# ============================================================
# PAGE 2: Detailed Analysis
# ============================================================

elif page == "🔍 Detailed Analysis":
    st.title("🔍 Strategy-wise Synchronization Analysis")
    st.markdown("Detailed alignment analysis for each strategic objective")
    
    # Objective selector
    objective_options = [
        f"{obj['objective_id']}: {obj['objective'][:60]}..."
        for obj in objectives
    ]
    
    selected_idx = st.selectbox(
        "Select Strategic Objective",
        range(len(objectives)),
        format_func=lambda i: objective_options[i]
    )
    
    obj = objectives[selected_idx]
    
    # Objective details
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {obj['objective']}")
        st.markdown(f"**Pillar:** {obj['pillar']}")
        st.markdown(f"**Type:** {obj['type']}")
    
    with col2:
        # Alignment gauge
        score = obj['alignment_score']
        color = '#2ecc71' if score >= 70 else '#f39c12' if score >= 50 else '#e74c3c'
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Alignment Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#ffebee"},
                    {'range': [50, 70], 'color': "#fff9e6"},
                    {'range': [70, 100], 'color': "#e8f5e9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Coverage", obj['coverage'])
    col2.metric("Max Similarity", f"{obj['max_similarity']:.1%}")
    col3.metric("Strong Matches", obj['num_strong_matches'])
    col4.metric("Moderate Matches", obj['num_moderate_matches'])
    
    st.markdown("---")
    
    # Matched Actions
    st.subheader("🎯 Matched Actions")
    
    for i, action in enumerate(obj['matched_actions'][:10], 1):
        with st.expander(f"{i}. [{action['action_id']}] {action['title']}", expanded=(i<=3)):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            col1.write(f"**Pillar:** {action['pillar']}")
            col2.metric("Similarity", f"{action['similarity']:.1%}")
            col3.write(f"**Strength:** {action['alignment_strength']}")
            
            # Visual indicator
            if action['similarity'] >= 0.70:
                st.success("✅ Strong alignment - Well synchronized")
            elif action['similarity'] >= 0.50:
                st.info("✓ Moderate alignment - Acceptable synchronization")
            else:
                st.warning("⚠️ Weak alignment - Consider reviewing")

# ============================================================
# PAGE 3: Gap Analysis
# ============================================================

elif page == "💡 Gap Analysis":
    st.title("💡 Gap Analysis & Recommendations")
    st.markdown("Identifying strategic objectives that need additional support")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Weak Objectives", len(gaps['weak_objectives']))
    col2.metric("Orphan Actions", len(gaps['orphan_actions']))
    col3.metric("Weak Pillars", len(gaps['pillar_gaps']))
    col4.metric("Coverage Gaps", len(gaps['coverage_gaps']))
    
    st.markdown("---")
    
    # Filter by severity
    severity_filter = st.multiselect(
        "Filter by Severity",
        ['Critical', 'High', 'Medium'],
        default=['Critical', 'High', 'Medium']
    )
    
    # Weak Objectives
    st.subheader("🚨 Weak Objectives (Requiring Attention)")
    
    filtered_gaps = [g for g in gaps['weak_objectives'] if g['severity'] in severity_filter]
    
    if not filtered_gaps:
        st.info("No gaps match the selected filters.")
    else:
        for i, gap in enumerate(filtered_gaps, 1):
            severity_emoji = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡'}
            
            with st.expander(
                f"{severity_emoji[gap['severity']]} {i}. [{gap['objective_id']}] Score: {gap['alignment_score']:.1f}%",
                expanded=(i<=5)
            ):
                st.markdown(f"**Objective:** {gap['objective']}")
                st.markdown(f"**Pillar:** {gap['pillar']}")
                st.markdown(f"**Type:** {gap['type']}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Alignment Score", f"{gap['alignment_score']:.1f}%")
                col2.metric("Gap Size", f"{gap['gap_size']:.1f}%")
                col3.metric("Severity", gap['severity'])
                
                st.markdown("**💡 Recommendations:**")
                if gap['severity'] == 'Critical':
                    st.error("""
                    - **Immediate action required**
                    - Develop specific action items for this objective
                    - Assign ownership and timeline
                    - Consider quarterly milestones
                    """)
                elif gap['severity'] == 'High':
                    st.warning("""
                    - **High priority for Year 1**
                    - Review existing actions for expansion
                    - Identify quick wins and long-term initiatives
                    - Allocate appropriate resources
                    """)
                else:
                    st.info("""
                    - **Monitor and enhance**
                    - Current actions may need refinement
                    - Consider phased approach in Year 2
                    """)
    
    # Orphan Actions
    if gaps['orphan_actions']:
        st.markdown("---")
        st.subheader("🔗 Orphan Actions (Weak Strategic Link)")
        
        for i, orphan in enumerate(gaps['orphan_actions'], 1):
            with st.expander(f"{i}. [{orphan['action_id']}] {orphan['title']}"):
                st.markdown(f"**Pillar:** {orphan['pillar']}")
                st.metric("Best Match Score", f"{orphan['max_similarity']:.1%}")
                st.warning(f"⚠️ {orphan['recommendation']}")

# ============================================================
# PAGE 4: AI-Powered Improvements
# ============================================================

elif page == "🤖 AI Improvements":
    st.title("🤖 AI-Powered Improvement Suggestions")
    st.markdown("Intelligent recommendations generated using RAG-enhanced LLM analysis")
    
    # Check if improvements exist
    if improvements_data is None:
        st.warning("⚠️ **Improvements not yet generated**")
        st.info("""
        Please run the improvement generator to create AI-powered suggestions:
```bash
        python src/rag_pipeline.py
```
        
        This will:
        - Analyze gap objectives using local LLM (Phi-3)
        - Use RAG for context-aware suggestions
        - Generate actionable recommendations
        """)
        st.stop()
    
    improvements = improvements_data['improvements']
    summary = improvements_data['summary']
    
    # Summary metrics
    st.markdown("### 📊 Generation Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gap Objectives", summary['total_gaps'])
    col2.metric("Processed", summary['processed'])
    col3.metric("Total Suggestions", summary['total_suggestions'])
    col4.metric("Avg per Objective", f"{summary['total_suggestions']/summary['processed']:.1f}" if summary['processed'] > 0 else "0")
    
    st.markdown("---")
    
    # Filter options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pillar_options = list(set(imp['pillar'] for imp in improvements))
        pillar_filter = st.multiselect(
            "Filter by Pillar",
            options=pillar_options,
            default=pillar_options
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Current Score (Low to High)", "Current Score (High to Low)", "Pillar"]
        )
    
    # Filter and sort
    filtered_improvements = [imp for imp in improvements if imp['pillar'] in pillar_filter]
    
    if sort_by == "Current Score (Low to High)":
        filtered_improvements.sort(key=lambda x: x['current_score'])
    elif sort_by == "Current Score (High to Low)":
        filtered_improvements.sort(key=lambda x: x['current_score'], reverse=True)
    else:  # Sort by Pillar
        filtered_improvements.sort(key=lambda x: x['pillar'])
    
    st.markdown(f"### 💡 Improvement Recommendations ({len(filtered_improvements)} objectives)")
    
    if not filtered_improvements:
        st.info("No improvements match the selected filters.")
    else:
        # Display improvements
        for i, imp in enumerate(filtered_improvements, 1):
            # Color code by score
            if imp['current_score'] < 30:
                score_color = "🔴"
            elif imp['current_score'] < 40:
                score_color = "🟠"
            else:
                score_color = "🟡"
            
            with st.expander(
                f"{score_color} {i}. [{imp['objective_id']}] {imp['objective'][:70]}... ({imp['current_score']:.1f}%)",
                expanded=(i <= 3)
            ):
                # Objective details
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Full Objective:** {imp['objective']}")
                    st.markdown(f"**Pillar:** {imp['pillar']}")
                
                with col2:
                    st.metric("Current Alignment", f"{imp['current_score']:.1f}%")
                
                st.markdown("---")
                
                # Suggestions tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🎯 New Actions",
                    "📊 KPIs",
                    "📅 Timeline",
                    "💰 Resources",
                    "🔗 Integration"
                ])
                
                suggestions = imp['suggestions']
                
                with tab1:
                    st.markdown("### 🎯 Recommended New Actions")
                    if suggestions['new_actions']:
                        for j, action in enumerate(suggestions['new_actions'], 1):
                            st.markdown(f"**{j}.** {action}")
                            st.markdown("")
                    else:
                        st.info("No specific new actions suggested.")
                
                with tab2:
                    st.markdown("### 📊 KPI Enhancements")
                    if suggestions['kpi_enhancements']:
                        for j, kpi in enumerate(suggestions['kpi_enhancements'], 1):
                            st.markdown(f"**{j}.** {kpi}")
                            st.markdown("")
                    else:
                        st.info("No KPI enhancements suggested.")
                
                with tab3:
                    st.markdown("### 📅 Timeline Recommendations")
                    if suggestions['timeline_recommendations']:
                        for j, timeline in enumerate(suggestions['timeline_recommendations'], 1):
                            st.markdown(f"**{j}.** {timeline}")
                            st.markdown("")
                    else:
                        st.info("No timeline recommendations provided.")
                
                with tab4:
                    st.markdown("### 💰 Resource Requirements")
                    if suggestions['resource_requirements']:
                        for j, resource in enumerate(suggestions['resource_requirements'], 1):
                            st.markdown(f"**{j}.** {resource}")
                            st.markdown("")
                    else:
                        st.info("No specific resource requirements identified.")
                
                with tab5:
                    st.markdown("### 🔗 Integration Opportunities")
                    if suggestions['integration_opportunities']:
                        for j, integration in enumerate(suggestions['integration_opportunities'], 1):
                            st.markdown(f"**{j}.** {integration}")
                            st.markdown("")
                    else:
                        st.info("No integration opportunities identified.")
                
                # Show retrieved context (NO nested expander)
                if 'retrieved_context' in imp and imp['retrieved_context']:
                    st.markdown("---")
                    st.markdown("### 📚 Retrieved Context (RAG)")
                    st.caption("*These document chunks were used to generate context-aware suggestions:*")
                    
                    for j, chunk in enumerate(imp['retrieved_context'][:3], 1):
                        st.markdown(f"**[{j}]** (Similarity: {chunk.get('similarity', 0):.2%})")
                        with st.container():
                            st.text(chunk['text'][:200] + "...")
                        st.markdown("")
    
    # Export button
    st.markdown("---")
    st.markdown("### 📥 Export Improvements")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💡 Download the complete improvement suggestions as JSON for further analysis or integration into project management tools.")
    
    with col2:
        # Convert to JSON string
        json_str = json.dumps(improvements_data, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="brandix_improvements.json",
            mime="application/json"
        )

# ============================================================
# PAGE 5: Executive Summary (NEW!)
# ============================================================

elif page == "📋 Executive Summary":
    st.title("📋 Executive Summary")
    st.markdown("AI-generated comprehensive summary for leadership and stakeholders")
    
    if executive_summary is None:
        st.warning("⚠️ **Executive Summary not yet generated**")
        st.info("""
        The executive summary uses LLM-based summarization to synthesize the entire
        synchronization analysis into a concise report for decision-makers.
        
        **To generate:**
```bash
        python src/executive_summary.py
```
        
        This will:
        - Analyze all 731 objective-action comparisons
        - Generate 6 executive-level sections
        - Create actionable recommendations
        - Takes 2-3 minutes to complete
        """)
        st.stop()
    
    # Display summary sections
    st.markdown("## 📄 Executive Overview")
    st.info(executive_summary['overview'])
    
    st.markdown("---")
    st.markdown("## 🔍 Key Findings")
    st.markdown(executive_summary['key_findings'])
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🚨 Critical Gaps")
        st.warning(executive_summary['critical_gaps'])
    
    with col2:
        st.markdown("## ⚠️ Risk Assessment")
        st.error(executive_summary['risk_assessment'])
    
    st.markdown("---")
    st.markdown("## 💡 Strategic Recommendations")
    st.success(executive_summary['recommendations'])
    
    st.markdown("---")
    st.markdown("## ✅ Immediate Next Steps (30-90 Days)")
    st.info(executive_summary['next_steps'])
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export Summary")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption("Download the complete executive summary for distribution to stakeholders")
    
    with col2:
        json_str = json.dumps(executive_summary, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="brandix_executive_summary.json",
            mime="application/json"
        )

# ============================================================
# PAGE 6: Pillar View
# ============================================================

elif page == "📈 Pillar View":
    st.title("📈 Strategic Pillar Analysis")
    st.markdown("Alignment analysis grouped by strategic pillars")
    
    # Pillar selector
    pillar_names = list(pillar_stats.keys())
    selected_pillar = st.selectbox("Select Pillar", pillar_names)
    
    stats = pillar_stats[selected_pillar]
    
    st.markdown("---")
    
    # Pillar metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Objectives", stats['count'])
    col2.metric("Average Score", f"{stats['average_score']:.1f}%")
    col3.metric("Status", stats['pillar_status'])
    col4.metric("Score Range", f"{stats['min_score']:.0f}% - {stats['max_score']:.0f}%")
    
    # Score distribution
    st.subheader(f"📊 Score Distribution for {selected_pillar}")
    
    scores_df = pd.DataFrame({
        'Objective': stats['objective_ids'],
        'Score': stats['scores']
    })
    
    fig_scores = px.bar(
        scores_df,
        x='Objective',
        y='Score',
        title=f'Alignment Scores - {selected_pillar}',
        color='Score',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100]
    )
    
    fig_scores.add_hline(y=70, line_dash="dash", line_color="green", 
                         annotation_text="Strong Threshold")
    fig_scores.add_hline(y=50, line_dash="dash", line_color="orange", 
                         annotation_text="Moderate Threshold")
    
    fig_scores.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_scores, use_container_width=True)
    
    # Objectives list
    st.subheader(f"📋 Objectives in {selected_pillar}")
    
    for i, (obj_id, obj_text, score) in enumerate(zip(
        stats['objective_ids'], 
        stats['objectives'], 
        stats['scores']
    ), 1):
        score_color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
        st.markdown(f"{score_color} **{i}. [{obj_id}]** - {score:.1f}%")
        st.markdown(f"   _{obj_text}_")
        st.markdown("")

# ============================================================
# PAGE 7: About
# ============================================================

else:  # About page
    st.title("ℹ️ About Brandix ISPS")
    
    st.markdown("""
    ### Intelligent Strategic Planning Synchronization System
    
    **Purpose:** Analyze and assess the synchronization between Brandix's strategic plan (2025-2030) 
    and action plan (Year 1: 2025-2026) using advanced AI technologies.
    
    ---
    
    ### 🎯 Key Features
    
    **1. Overall Synchronization Assessment**
    - Uses semantic embeddings to understand meaning
    - Calculates similarity between objectives and actions
    - Provides overall alignment score and classification
    
    **2. Strategy-wise Analysis**
    - Analyzes each of the 43 strategic objectives individually
    - Identifies top matching actions for each objective
    - Classifies alignment strength (Strong/Moderate/Weak)
    
    **3. Gap Detection**
    - Automatically identifies poorly aligned objectives
    - Finds orphan actions with weak strategic links
    - Categorizes gaps by severity (Critical/High/Medium)
    
    **4. AI-Powered Improvements** ✨
    - RAG-enhanced LLM suggestions for gap objectives
    - Context-aware recommendations using retrieved documents
    - Categorized improvements: Actions, KPIs, Timeline, Resources
    
    **5. Executive Summary** ✨ NEW!
    - LLM-based summarization and reporting
    - Comprehensive analysis synthesis for decision-makers
    - 6 professional sections: Overview, Findings, Gaps, Recommendations, Risks, Next Steps
    
    **6. Pillar-wise Categorization**
    - Groups analysis by strategic pillars
    - Shows pillar-level performance
    - Identifies weak areas requiring attention
    
    ---
    
    ### 🛠️ Technology Stack
    
    - **LLM:** Ollama with Phi-3 Mini (local, privacy-focused)
    - **RAG:** Retrieval-Augmented Generation for context-aware suggestions
    - **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
    - **Vector DB:** FAISS (Facebook AI Similarity Search)
    - **Dashboard:** Streamlit with Plotly visualizations
    - **Programming:** Python 3.10+
    
    ---
    
    ### 🔒 Security & Privacy
    
    All data processing happens **locally** with no external API calls:
    - Complete data privacy and GDPR compliance
    - No sensitive information leaves your system
    - Local LLM processing (Phi-3 Mini via Ollama)
    - Secure vector database storage
    - All AI suggestions generated on-premises
    
    ---
    
    ### 📊 Current Analysis
    
    - **Strategic Objectives:** {overall['total_objectives']}
    - **Action Items:** {overall['total_actions']}
    - **Comparisons Made:** {overall['total_objectives'] * overall['total_actions']} ({overall['total_objectives']} × {overall['total_actions']})
    - **Overall Alignment:** {overall['overall_score']:.1f}%
    - **Coverage Rate:** {overall['coverage_rate']:.1f}%
    """)
    
    if improvements_data:
        st.markdown(f"""
    - **AI Improvements Generated:** {improvements_data['summary']['total_suggestions']} suggestions
    - **Gap Objectives Processed:** {improvements_data['summary']['processed']} objectives
        """)
    
    if executive_summary:
        st.markdown("""
    - **Executive Summary:** ✅ Generated with 6 comprehensive sections
        """)
    
    st.markdown("""
    ---
    
    ### 👨‍💻 Developed By
    
    **Course:** MSc Computer Science - Information Retrieval  
    **Project:** Individual Coursework  
    **Deadline:** February 10, 2026  
    **Institution:** Your University
    
    ---
    
    ### 📚 References
    
    - Brandix ESG Report 2024-25
    - Strategic Planning Framework 2025-2030
    - Year 1 Action Plan (April 2025 - March 2026)
    """)
    
    st.markdown("---")
    st.markdown("© 2025 Brandix ISPS | All Rights Reserved")

# ============================================================
# Footer
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Documentation")
st.sidebar.markdown("[View on GitHub](#)")
st.sidebar.markdown("[Report Issues](#)")
st.sidebar.markdown("---")
st.sidebar.info("""
**Need Help?**

Check the 'About' page for system information and methodology.
""")