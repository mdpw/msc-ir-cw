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
st.set_page_config(
    page_title="Knowledge Graph",
    page_icon="🕸️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #1f4788;
        font-weight: 600;
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

st.title("🕸️ Strategic Alignment Knowledge Graph")
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
    st.info(f"📅 Visualizing strategic network for **Year {selected_year}**")

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
    st.error(f"⚠️ No analysis data found for year {selected_year}")
    st.info("👉 Go to **'⚙️ Run Analysis'** page to analyze this year first")
    st.stop()

st.success(f"✅ Loaded analysis data for {selected_year}")
st.markdown("---")

# ============================================================
# Configuration Controls
# ============================================================

st.subheader("⚙️ Graph Configuration")

col1, col2, col3, col4 = st.columns(4)

with col1:
    similarity_threshold = st.slider(
        "🎯 Similarity Threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Minimum similarity score to show connections"
    )

with col2:
    layout_type = st.selectbox(
        "📐 Layout Algorithm",
        options=['spring', 'hierarchical', 'circular', 'kamada_kawai'],
        index=0,
        help="Algorithm for positioning nodes in the graph"
    )

with col3:
    graph_height = st.slider(
        "📏 Graph Height",
        min_value=600,
        max_value=1200,
        value=900,
        step=50
    )

with col4:
    show_labels = st.checkbox(
        "🏷️ Show Node Labels",
        value=False,
        help="Display labels directly on nodes (may be crowded)"
    )

st.markdown("---")

# ============================================================
# Generate and Display Graph
# ============================================================

st.subheader("🌐 Interactive Network Visualization")

st.markdown("""
**Legend:**
- 🔵 **Circles** = Strategic Objectives (size = alignment strength)
- 🟦 **Squares** = Action Items
- **Edges**: 🟢 Strong (≥70%), 🟠 Moderate (50-70%), 🔴 Weak (<50%)
- **Hover** over nodes and edges for details
""")

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
        st.subheader("📊 Network Statistics")
        
        stats = kg.get_graph_statistics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Nodes",
                stats['total_nodes'],
                help="Total number of objectives and actions"
            )
        
        with col2:
            st.metric(
                "Total Connections",
                stats['total_edges'],
                help="Number of alignment links above threshold"
            )
        
        with col3:
            st.metric(
                "Strong Links",
                stats['strong_edges'],
                delta="≥70% similarity",
                help="High-quality alignments"
            )
        
        with col4:
            st.metric(
                "Moderate Links",
                stats['moderate_edges'],
                delta="50-70%",
                help="Good alignments"
            )
        
        with col5:
            st.metric(
                "Weak Links",
                stats['weak_edges'],
                delta="<50%",
                delta_color="inverse",
                help="Low alignments (needs improvement)"
            )
        
        # Detailed statistics
        with st.expander("📈 Detailed Network Metrics"):
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
        st.subheader("🔍 Network Analysis Insights")
        
        if graph.number_of_nodes() > 0:
            # Calculate centrality
            degree_centrality = nx.degree_centrality(graph)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Most Connected Objectives**")
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
                        - {connections} connections | Score: {node_data['alignment_score']:.1f}%
                        """)
                else:
                    st.info("No objectives in current graph view")
            
            with col2:
                st.markdown("**⚠️ Isolated or Weak Objectives**")
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
                            status = "🔴 CRITICAL - No connections"
                        elif connections == 1:
                            status = "🟠 HIGH - Only 1 connection"
                        else:
                            status = "🟡 MEDIUM - 2 connections"
                        
                        st.markdown(f"""
                        **{node}** - {status}
                        - {node_data['label']}
                        - Score: {node_data['alignment_score']:.1f}%
                        """)
                else:
                    st.success("✅ All objectives are well-connected")
            
            # Pillar connectivity
            st.markdown("---")
            st.markdown("**🏛️ Connectivity by Strategic Pillar**")
            
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
        st.subheader("📥 Export Graph Data")
        
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
                label="📄 JSON Data",
                data=json.dumps(graph_json, indent=2),
                file_name=f"knowledge_graph_{selected_year}_t{int(similarity_threshold*100)}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export as HTML
            html_str = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="🌐 Interactive HTML",
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
                label="🔬 NetworkX Graph (PKL)",
                data=graph_bytes,
                file_name=f"graph_{selected_year}.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                help="For further analysis in Python"
            )
        
    except Exception as e:
        st.error(f"❌ Error generating knowledge graph: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.exception(e)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("""
💡 **Tips:**
- Lower threshold shows more connections but may be cluttered
- Higher threshold shows only strong relationships
- Try different layouts for different perspectives
- Hover over nodes to see full details
""")