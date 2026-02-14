"""
Brandix ISPS - Knowledge Graph Visualization Page
Standalone page for interactive network visualization
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from knowledge_graph import KnowledgeGraphGenerator

# Page configuration
# Page configuration
st.set_page_config(
    page_title="Knowledge Graph",
    page_icon=None,
    layout="wide"
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

# ============================================================
# Configuration
# ============================================================

AVAILABLE_YEARS = ["2026", "2027", "2028", "2029", "2030"]
OUTPUTS_BASE = Path("outputs")

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = "2026"

# ============================================================
# Header
# ============================================================

st.markdown('<h1><i class="fas fa-network-wired fa-icon-large"></i>Strategic Alignment Knowledge Graph</h1>', unsafe_allow_html=True)
st.markdown("Interactive network visualization of strategic objectives and action items")
st.markdown("---")

# Year selector
col1, col2 = st.columns([1, 3])

with col1:
    selected_year = st.selectbox(
        "Select Year",
        AVAILABLE_YEARS,
        index=AVAILABLE_YEARS.index(st.session_state.selected_year),
        key='kg_year_selector'
    )
    
    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()

with col2:
    st.info(f"Visualizing strategic network for **Year {selected_year}**")

# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_sync_report(year):
    """Load synchronization report for year"""
    report_path = OUTPUTS_BASE / year / 'synchronization_report.json'
    
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

report = load_sync_report(selected_year)

if report is None:
    st.error(f"No analysis data found for year {selected_year}")
    st.info("Go to **'Run Analysis'** page to analyze this year first")
    st.stop()

st.success(f"Loaded analysis data for {selected_year}")
st.markdown("---")

# ============================================================
# Configuration Controls
# ============================================================

st.markdown('<h3><i class="fas fa-sliders-h fa-icon"></i>Graph Configuration</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Minimum similarity score to show connections. Set to 0.70 to see only Strong links."
    )

with col2:
    layout_type = st.selectbox(
        "Layout Algorithm",
        options=['radial', 'spring', 'hierarchical', 'circular', 'kamada_kawai'],
        index=0,
        help="Algorithm for positioning nodes. 'radial' is recommended for readability."
    )

with col3:
    graph_height = st.slider(
        "Graph Height",
        min_value=600,
        max_value=1200,
        value=900,
        step=50
    )

with col4:
    show_labels = st.checkbox(
        "Show Node Labels",
        value=False,
        help="Display labels directly on nodes (may be crowded)"
    )

st.markdown("---")

# ============================================================
# Generate and Display Graph
# ============================================================

st.markdown('<h3><i class="fas fa-project-diagram fa-icon"></i>Interactive Network Visualization</h3>', unsafe_allow_html=True)

st.markdown(f"""
<div class="info-box">
<strong>Legend:</strong><br>
- <i class="fas fa-circle" style="color: #4da6ff;"></i> <strong>Circles</strong> = Strategic Objectives (size = alignment strength)<br>
- <i class="fas fa-square" style="color: #1c83e1;"></i> <strong>Squares</strong> = Action Items<br>
- <strong>Edges:</strong> <span style="color: #2ecc71; font-weight: bold;">Strong (≥70%)</span>, <span style="color: #f39c12; font-weight: bold;">Moderate (50-70%)</span>, <span style="color: #e74c3c; font-weight: bold;">Weak (<50%)</span><br>
- <strong>Hover</strong> over nodes and edges for details
</div>
""", unsafe_allow_html=True)

with st.spinner("🔄 Generating knowledge graph..."):
    try:
        # Create knowledge graph
        kg = KnowledgeGraphGenerator()
        
        # Build graph from report
        graph = kg.create_graph_from_sync_report(
            report, 
            similarity_threshold=similarity_threshold
        )
        
        # Calculate layout
        kg.calculate_layout(layout_type)
        
        # Create visualization
        fig = kg.create_plotly_figure(
            title=f"Strategic Alignment Network - Year {selected_year} (Threshold: {similarity_threshold:.0%})",
            width=1400,
            height=graph_height
        )
        
        # Display the graph
        st.plotly_chart(fig, use_container_width=True)
        
        # ============================================================
        # Graph Statistics
        # ============================================================
        
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-chart-bar fa-icon"></i>Network Statistics</h3>', unsafe_allow_html=True)
        
        stats = kg.get_graph_statistics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Nodes",
                stats['total_nodes']
            )
        
        with col2:
            st.metric(
                "Total Connections",
                stats['total_edges']
            )
        
        with col3:
            st.metric(
                "Strong Links",
                stats['strong_edges'],
                delta="≥70%"
            )
        
        with col4:
            st.metric(
                "Moderate Links",
                stats['moderate_edges'],
                delta="50-70%"
            )
        
        with col5:
            st.metric(
                "Weak Links",
                stats['weak_edges'],
                delta="<50%",
                delta_color="inverse"
            )
        
        # Detailed statistics
        with st.expander("Detailed Network Metrics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Objective Nodes", stats['objective_nodes'])
                st.metric("Action Nodes", stats['action_nodes'])
            
            with col2:
                st.metric("Connected Components", stats['connected_components'])
                st.metric("Network Density", f"{stats['density']:.3f}")
            
            with col3:
                st.metric("Avg Connections/Node", f"{stats['average_degree']:.2f}")
        
        # ============================================================
        # Network Insights
        # ============================================================
        
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-lightbulb fa-icon"></i>Network Analysis Insights</h3>', unsafe_allow_html=True)
        
        if graph.number_of_nodes() > 0:
            # Calculate centrality
            degree_centrality = nx.degree_centrality(graph)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('**<i class="fas fa-link fa-icon-small"></i>Most Connected Objectives**', unsafe_allow_html=True)
                st.caption("Objectives with the highest number of action alignments")
                
                # Get objective nodes only
                obj_centrality = {
                    node: centrality 
                    for node, centrality in degree_centrality.items() 
                    if graph.nodes[node]['type'] == 'objective'
                }
                
                if obj_centrality:
                    top_objectives = sorted(
                        obj_centrality.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                    
                    for rank, (node, centrality) in enumerate(top_objectives, 1):
                        node_data = graph.nodes[node]
                        connections = graph.degree(node)
                        
                        st.markdown(f"""
                        **{rank}. {node}**
                        - {node_data['label']}
                        - {connections} connections | Score: <span style="color: #4da6ff; font-weight: bold;">{node_data['alignment_score']:.1f}%</span>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No objectives in current graph view")
            
            with col2:
                st.markdown('**<i class="fas fa-link-slash fa-icon-small" style="color: #f44336;"></i>Isolated or Weak Objectives**', unsafe_allow_html=True)
                st.caption("Objectives with few or no connections")
                
                # Find objectives with 0-2 connections
                weak_connected = [
                    node for node in graph.nodes() 
                    if graph.nodes[node]['type'] == 'objective' and graph.degree(node) <= 2
                ]
                
                if weak_connected:
                    for node in weak_connected[:5]:
                        node_data = graph.nodes[node]
                        connections = graph.degree(node)
                        
                        if connections == 0:
                            status = '<span style="color: #f44336; font-weight: bold;">CRITICAL - No connections</span>'
                        elif connections == 1:
                            status = '<span style="color: #ff9800; font-weight: bold;">HIGH - Only 1 connection</span>'
                        else:
                            status = '<span style="color: #ffeb3b; font-weight: bold;">MEDIUM - 2 connections</span>'
                        
                        st.markdown(f"""
                        **{node}** - {status}
                        - {node_data['label']}
                        - Score: <span style="color: #4da6ff; font-weight: bold;">{node_data['alignment_score']:.1f}%</span>
                        """, unsafe_allow_html=True)
                else:
                    st.success("All objectives are well-connected")
            
            # Pillar connectivity
            st.markdown("---")
            st.markdown('**<i class="fas fa-landmark fa-icon-small"></i>Connectivity by Strategic Pillar**', unsafe_allow_html=True)
            
            pillar_connections = {}
            for node in graph.nodes():
                if graph.nodes[node]['type'] == 'objective':
                    pillar = graph.nodes[node]['pillar']
                    connections = graph.degree(node)
                    
                    if pillar not in pillar_connections:
                        pillar_connections[pillar] = []
                    pillar_connections[pillar].append(connections)
            
            if pillar_connections:
                pillar_df = []
                for pillar, connections in pillar_connections.items():
                    pillar_df.append({
                        'Pillar': pillar,
                        'Objectives': len(connections),
                        'Avg Connections': sum(connections) / len(connections),
                        'Max Connections': max(connections),
                        'Min Connections': min(connections)
                    })
                
                import pandas as pd
                df = pd.DataFrame(pillar_df)
                df = df.sort_values('Avg Connections', ascending=False)
                
                st.dataframe(
                    df.style.format({
                        'Avg Connections': '{:.1f}',
                        'Max Connections': '{:.0f}',
                        'Min Connections': '{:.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        
        # ============================================================
        # Export Options
        # ============================================================
        
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-file-export fa-icon"></i>Export Graph Data</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as JSON
            graph_json = {
                'nodes': [
                    {'id': node, **data} 
                    for node, data in graph.nodes(data=True)
                ],
                'edges': [
                    {'source': s, 'target': t, **data} 
                    for s, t, data in graph.edges(data=True)
                ],
                'statistics': stats,
                'metadata': {
                    'year': selected_year,
                    'threshold': similarity_threshold,
                    'layout': layout_type
                }
            }
            
            st.download_button(
                label="Download JSON Data",
                data=json.dumps(graph_json, indent=2),
                file_name=f"knowledge_graph_{selected_year}_t{int(similarity_threshold*100)}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export as HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="Download Interactive HTML",
                data=html_str,
                file_name=f"knowledge_graph_{selected_year}_t{int(similarity_threshold*100)}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            # Export NetworkX graph
            import pickle
            graph_bytes = pickle.dumps(graph)
            st.download_button(
                label="Download NetworkX Graph (PKL)",
                data=graph_bytes,
                file_name=f"graph_{selected_year}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                help="For further analysis in Python"
            )
        
    except Exception as e:
        st.error(f"Error generating knowledge graph: {str(e)}")
        with st.expander("Error Details"):
            st.exception(e)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("""
<div class="info-box">
<i class="fas fa-lightbulb fa-icon-small"></i><strong>Tips:</strong><br>
- Lower threshold shows more connections but may be cluttered<br>
- Higher threshold shows only strong relationships<br>
- Try different layouts for different perspectives<br>
- Hover over nodes to see full details
</div>
""", unsafe_allow_html=True)
