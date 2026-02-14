"""
Knowledge Graph Generator
Creates interactive network visualizations of strategic alignments
Satisfies Requirement 3.5: Knowledge graph visualization
"""

import networkx as nx
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple
import json

class KnowledgeGraphGenerator:
    def __init__(self):
        """Initialize knowledge graph generator"""
        self.graph = nx.Graph()
        self.node_positions = {}
        self.node_colors = {}
        self.node_sizes = {}
        
    def create_graph_from_sync_report(self, sync_report: Dict, similarity_threshold=0.50):
        """
        Create knowledge graph from synchronization report
        
        Args:
            sync_report: Complete synchronization report
            similarity_threshold: Minimum similarity to create edge (default: 0.50)
            
        Returns:
            NetworkX graph object
        """
        print(f"\n{'='*80}")
        print("CREATING KNOWLEDGE GRAPH")
        print(f"{'='*80}")
        print(f"Similarity threshold: {similarity_threshold}")
        
        self.graph.clear()
        
        # Extract data
        objectives = sync_report['objective_details']
        pillar_stats = sync_report['pillar_stats']
        
        # Define pillar colors
        pillar_colors = {
            'Environmental Leadership': '#2ecc71',
            'Innovation & Digital Transformation': '#3498db',
            'People Excellence & Social Impact': '#e74c3c',
            'Operational Excellence': '#f39c12',
            'Governance & Risk Management': '#9b59b6',
            'Unknown': '#95a5a6'
        }
        
        print(f"\nAdding {len(objectives)} objectives as nodes...")
        
        # Add objective nodes
        for obj in objectives:
            obj_id = obj['objective_id']
            pillar = obj['pillar']
            
            self.graph.add_node(
                obj_id,
                type='objective',
                label=obj['objective'][:50] + '...',
                full_text=obj['objective'],
                pillar=pillar,
                alignment_score=obj['alignment_score'],
                coverage=obj['coverage'],
                node_size=20 + (obj['alignment_score'] / 5),  # Size based on alignment
                color=pillar_colors.get(pillar, '#95a5a6')
            )
        
        # Count actions to add
        actions_to_add = set()
        edges_to_add = []
        
        for obj in objectives:
            obj_id = obj['objective_id']
            
            # Add edges to matched actions above threshold
            for action in obj['matched_actions']:
                similarity = action['similarity']
                
                if similarity >= similarity_threshold:
                    action_id = action['action_id']
                    actions_to_add.add(action_id)
                    edges_to_add.append((obj_id, action_id, similarity, action))
        
        print(f"Adding {len(actions_to_add)} action nodes (filtered by threshold)...")
        
        # Add action nodes
        for obj in objectives:
            for action in obj['matched_actions']:
                action_id = action['action_id']
                
                if action_id in actions_to_add:
                    pillar = action['pillar']
                    
                    # Only add if not already added
                    if not self.graph.has_node(action_id):
                        self.graph.add_node(
                            action_id,
                            type='action',
                            label=action['title'][:40] + '...',
                            full_text=action['title'],
                            pillar=pillar,
                            node_size=15,
                            color=pillar_colors.get(pillar, '#95a5a6')
                        )
        
        print(f"Adding {len(edges_to_add)} edges...")
        
        # Add edges
        for obj_id, action_id, similarity, action in edges_to_add:
            self.graph.add_edge(
                obj_id,
                action_id,
                weight=similarity,
                strength=action['alignment_strength'],
                similarity=similarity
            )
        
        print(f"\nGraph created:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Connected components: {nx.number_connected_components(self.graph)}")
        
        return self.graph
    
    def calculate_layout(self, layout_type='spring'):
        """
        Calculate node positions using specified layout algorithm
        
        Args:
            layout_type: 'spring', 'circular', 'kamada_kawai', or 'hierarchical'
        """
        print(f"\nCalculating {layout_type} layout...")
        
        if layout_type == 'spring':
            # Spring layout - good for showing clusters
            self.node_positions = nx.spring_layout(
                self.graph,
                k=2,
                iterations=50,
                seed=42
            )
        
        elif layout_type == 'circular':
            # Circular layout
            self.node_positions = nx.circular_layout(self.graph)
        
        elif layout_type == 'kamada_kawai':
            # Kamada-Kawai layout - force-directed
            self.node_positions = nx.kamada_kawai_layout(self.graph)
        
        elif layout_type == 'hierarchical':
            # Hierarchical layout (objectives on left, actions on right)
            self.node_positions = self._hierarchical_layout()
        
        elif layout_type == 'radial':
            # Radial layout (actions in center, objectives in circle)
            self.node_positions = self._radial_layout()
        
        else:
            # Default to spring
            self.node_positions = nx.spring_layout(self.graph, k=2.0/np.sqrt(len(self.graph.nodes)), iterations=100, seed=42)
        
        print("Layout calculated")
        
        return self.node_positions
    
    def _hierarchical_layout(self) -> Dict:
        """Create hierarchical layout with objectives on left, actions on right"""
        positions = {}
        
        # Separate nodes by type
        objectives = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'objective']
        actions = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'action']
        
        # Position objectives on left (x=0)
        obj_y_spacing = 4.0 / max(len(objectives), 1)
        for i, obj_id in enumerate(objectives):
            positions[obj_id] = (0, i * obj_y_spacing - 2)
        
        # Position actions on right (x=2)
        action_y_spacing = 4.0 / max(len(actions), 1)
        for i, action_id in enumerate(actions):
            positions[action_id] = (2, i * action_y_spacing - 2)
        
        return positions

    def _radial_layout(self) -> Dict:
        """Create radial layout: Actions in center, Objectives in outer circle clustered by pillar"""
        positions = {}
        
        actions = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'action']
        objectives = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'objective']
        
        # Sort objectives by pillar for clustering
        objectives.sort(key=lambda x: self.graph.nodes[x].get('pillar', ''))
        
        # Position actions in a small inner circle
        num_actions = len(actions)
        for i, action_id in enumerate(actions):
            angle = (2 * np.pi * i) / max(num_actions, 1)
            radius = 0.5
            positions[action_id] = (radius * np.cos(angle), radius * np.sin(angle))
            
        # Position objectives in a large outer circle
        num_objectives = len(objectives)
        for i, obj_id in enumerate(objectives):
            angle = (2 * np.pi * i) / max(num_objectives, 1)
            radius = 2.5
            positions[obj_id] = (radius * np.cos(angle), radius * np.sin(angle))
            
        return positions
    
    def create_plotly_figure(self, title="Strategic Alignment Knowledge Graph", 
                            width=1200, height=800):
        """
        Create interactive Plotly figure
        
        Args:
            title: Graph title
            width: Figure width
            height: Figure height
            
        Returns:
            Plotly Figure object
        """
        if not self.node_positions:
            self.calculate_layout('spring')
        
        print("\nCreating Plotly visualization...")
        
        # Create edge traces
        edge_traces = self._create_edge_traces()
        
        # Create node traces (separate by type for legend)
        objective_trace, action_trace = self._create_node_traces()
        
        # Combine all traces
        fig = go.Figure(data=edge_traces + [objective_trace, action_trace])
        
        # Update layout for dark mode compatibility
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#4da6ff'}
            },
            showlegend=True,
            width=width,
            height=height,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                font=dict(color='#e0e0e0')
            )
        )
        
        print("Plotly figure created")
        
        return fig
    
    def _create_edge_traces(self) -> List[go.Scatter]:
        """Create edge traces with different colors for alignment strength"""
        
        # Group edges by strength
        strong_edges = []
        moderate_edges = []
        weak_edges = []
        
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            strength = data.get('strength', 'Weak')
            
            x0, y0 = self.node_positions[source]
            x1, y1 = self.node_positions[target]
            
            edge_data = {
                'x': [x0, x1, None],
                'y': [y0, y1, None],
                'similarity': data.get('similarity', 0)
            }
            
            if strength == 'Strong':
                strong_edges.append(edge_data)
            elif strength == 'Moderate':
                moderate_edges.append(edge_data)
            else:
                weak_edges.append(edge_data)
        
        traces = []
        
        # Strong edges (green)
        if strong_edges:
            x_coords = [coord for edge in strong_edges for coord in edge['x']]
            y_coords = [coord for edge in strong_edges for coord in edge['y']]
            
            traces.append(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(width=2, color='#2ecc71'),
                opacity=0.8,
                hoverinfo='skip',
                showlegend=True,
                name=f'Strong Alignment (≥70%) - {len(strong_edges)} links'
            ))
        
        # Moderate edges (orange)
        if moderate_edges:
            x_coords = [coord for edge in moderate_edges for coord in edge['x']]
            y_coords = [coord for edge in moderate_edges for coord in edge['y']]
            
            traces.append(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(width=1.5, color='#f39c12'),
                opacity=0.4,
                hoverinfo='skip',
                showlegend=True,
                name=f'Moderate Alignment (50-70%) - {len(moderate_edges)} links'
            ))
        
        # Weak edges (red, dashed)
        if weak_edges:
            x_coords = [coord for edge in weak_edges for coord in edge['x']]
            y_coords = [coord for edge in weak_edges for coord in edge['y']]
            
            traces.append(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(width=0.8, color='rgba(231, 76, 60, 0.15)', dash='dot'),
                opacity=0.2,
                hoverinfo='skip',
                showlegend=True,
                name=f'Weak Alignment (<50%) - {len(weak_edges)} links'
            ))
        
        return traces
    
    def _create_node_traces(self) -> Tuple[go.Scatter, go.Scatter]:
        """Create separate node traces for objectives and actions"""
        
        # Separate nodes by type
        obj_x, obj_y, obj_text, obj_color, obj_size = [], [], [], [], []
        action_x, action_y, action_text, action_color, action_size = [], [], [], [], []
        
        for node, data in self.graph.nodes(data=True):
            x, y = self.node_positions[node]
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"{data.get('full_text', data.get('label', ''))}<br>"
            hover_text += f"Pillar: {data.get('pillar', 'Unknown')}<br>"
            
            if data['type'] == 'objective':
                hover_text += f"Alignment: {data.get('alignment_score', 0):.1f}%<br>"
                hover_text += f"Coverage: {data.get('coverage', 'Unknown')}"
                
                obj_x.append(x)
                obj_y.append(y)
                obj_text.append(hover_text)
                obj_color.append(data.get('color', '#95a5a6'))
                obj_size.append(data.get('node_size', 20))
            
            else:  # action
                action_x.append(x)
                action_y.append(y)
                action_text.append(hover_text)
                action_color.append(data.get('color', '#95a5a6'))
                action_size.append(data.get('node_size', 15))
        
        # Create objective trace
        objective_trace = go.Scatter(
            x=obj_x,
            y=obj_y,
            mode='markers',
            marker=dict(
                size=obj_size,
                color=obj_color,
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            text=obj_text,
            hovertemplate='%{text}<extra></extra>',
            name=f'Strategic Objectives ({len(obj_x)})',
            showlegend=True
        )
        
        # Create action trace
        action_trace = go.Scatter(
            x=action_x,
            y=action_y,
            mode='markers',
            marker=dict(
                size=action_size,
                color=action_color,
                line=dict(width=2, color='white'),
                symbol='square'
            ),
            text=action_text,
            hovertemplate='%{text}<extra></extra>',
            name=f'Action Items ({len(action_x)})',
            showlegend=True
        )
        
        return objective_trace, action_trace
    
    def get_graph_statistics(self) -> Dict:
        """Calculate graph statistics"""
        
        if self.graph.number_of_nodes() == 0:
            return {}
        
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'objective_nodes': sum(1 for _, d in self.graph.nodes(data=True) if d['type'] == 'objective'),
            'action_nodes': sum(1 for _, d in self.graph.nodes(data=True) if d['type'] == 'action'),
            'connected_components': nx.number_connected_components(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph)
        }
        
        # Edge statistics by strength
        edge_strengths = [d.get('strength', 'Unknown') for _, _, d in self.graph.edges(data=True)]
        stats['strong_edges'] = edge_strengths.count('Strong')
        stats['moderate_edges'] = edge_strengths.count('Moderate')
        stats['weak_edges'] = edge_strengths.count('Weak')
        
        return stats
    
    def export_graph(self, filepath: str):
        """Export graph data to JSON"""
        
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    **data
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **data
                }
                for source, target, data in self.graph.edges(data=True)
            ],
            'statistics': self.get_graph_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph data exported to {filepath}")


# Test
if __name__ == "__main__":
    import json
    import os
    
    print("="*80)
    print("KNOWLEDGE GRAPH GENERATOR TEST")
    print("="*80)
    
    # Load synchronization report
    print("\nLoading synchronization report...")
    try:
        with open('outputs/synchronization_report.json', 'r', encoding='utf-8') as f:
            report = json.load(f)
        print(f"Loaded report")
    except Exception as e:
        print(f"ERROR loading report: {e}")
        print("\nPlease run: python src/synchronization_engine.py")
        exit(1)
    
    # Create knowledge graph
    print("\nCreating knowledge graph...")
    kg = KnowledgeGraphGenerator()
    
    # Test with different thresholds
    for threshold in [0.50, 0.60, 0.70]:
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*80}")
        
        graph = kg.create_graph_from_sync_report(report, similarity_threshold=threshold)
        
        # Calculate layout
        kg.calculate_layout('spring')
        
        # Get statistics
        stats = kg.get_graph_statistics()
        print(f"\nGraph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Create visualization
        fig = kg.create_plotly_figure(
            title=f"Strategic Alignment Network (Threshold: {threshold})"
        )
        
        # Save
        output_dir = 'outputs/knowledge_graphs'
        os.makedirs(output_dir, exist_ok=True)
        
        fig.write_html(f'{output_dir}/knowledge_graph_threshold_{int(threshold*100)}.html')
        print(f"\nSaved visualization to {output_dir}/knowledge_graph_threshold_{int(threshold*100)}.html")
        
        # Export graph data
        kg.export_graph(f'{output_dir}/graph_data_threshold_{int(threshold*100)}.json')
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH GENERATION COMPLETE!")
    print("="*80)
    print("\nCheck outputs/knowledge_graphs/ for:")
    print("  - Interactive HTML visualizations")
    print("  - Graph data exports")
    print("\nNext: Integrate into Streamlit dashboard")