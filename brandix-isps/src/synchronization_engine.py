"""
Synchronization Engine
Analyzes alignment between strategic objectives and action items
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import json
import os

class SynchronizationEngine:
    def __init__(self, doc_processor, embedding_engine, vector_store):
        """
        Initialize synchronization engine
        
        Args:
            doc_processor: BrandixDocumentProcessor instance
            embedding_engine: EmbeddingEngine instance
            vector_store: FAISSVectorStore instance
        """
        self.doc_processor = doc_processor
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.alignment_matrix = None
    
    # ============================================================
    # REQUIREMENT 1: Overall Synchronization Assessment
    # ============================================================
    
    def calculate_overall_alignment(self) -> Dict:
        """
        Calculate overall document-level alignment
        
        Returns:
            Dictionary with overall alignment metrics
        """
        if self.embedding_engine.similarity_matrix is None:
            self.embedding_engine.calculate_similarity_matrix()
        
        sim_matrix = self.embedding_engine.similarity_matrix
        
        # Calculate overall metrics
        overall_score = np.mean(sim_matrix) * 100
        max_per_objective = np.max(sim_matrix, axis=1)
        mean_max_similarity = np.mean(max_per_objective)
        coverage_rate = np.sum(max_per_objective >= 0.5) / len(max_per_objective) * 100
        
        # Classify overall alignment
        if mean_max_similarity >= 0.70:
            classification = "Strong Alignment"
        elif mean_max_similarity >= 0.50:
            classification = "Good Alignment"
        else:
            classification = "Needs Improvement"
        
        # Count alignments by strength
        strong = int(np.sum(max_per_objective >= 0.70))
        moderate = int(np.sum((max_per_objective >= 0.50) & (max_per_objective < 0.70)))
        weak = int(np.sum(max_per_objective < 0.50))
        
        return {
            'overall_score': float(overall_score),
            'mean_max_similarity': float(mean_max_similarity * 100),
            'classification': classification,
            'coverage_rate': float(coverage_rate),
            'distribution': {
                'strong': strong,
                'moderate': moderate,
                'weak': weak
            },
            'total_objectives': len(max_per_objective),
            'total_actions': sim_matrix.shape[1]
        }
    
    # ============================================================
    # REQUIREMENT 2: Strategy-wise Synchronization
    # ============================================================
    
    def analyze_objective_alignment(self, objective_idx: int) -> Dict:
        """
        Analyze alignment for a specific strategic objective
        
        Args:
            objective_idx: Index of the objective to analyze
            
        Returns:
            Dictionary with detailed alignment analysis
        """
        if self.embedding_engine.similarity_matrix is None:
            self.embedding_engine.calculate_similarity_matrix()
        
        sim_matrix = self.embedding_engine.similarity_matrix
        similarities = sim_matrix[objective_idx]
        
        # Get objective details
        objective = self.doc_processor.strategic_objectives[objective_idx]
        
        # Find top-K matches
        top_k = min(10, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build matched actions list
        matched_actions = []
        for idx in top_indices:
            action = self.doc_processor.action_items[idx]
            matched_actions.append({
                'action_idx': int(idx),
                'action_id': action['id'],
                'title': action['title'],
                'pillar': action.get('pillar', 'Unknown'),
                'similarity': float(similarities[idx]),
                'alignment_strength': self._classify_alignment(similarities[idx])
            })
        
        # Calculate objective-level metrics
        max_similarity = float(np.max(similarities))
        mean_similarity = float(np.mean(similarities))
        alignment_score = max_similarity * 100
        
        # Determine coverage level
        if alignment_score >= 70:
            coverage = 'High'
        elif alignment_score >= 50:
            coverage = 'Moderate'
        else:
            coverage = 'Low'
        
        return {
            'objective_idx': objective_idx,
            'objective_id': objective.get('id', f'OBJ-{objective_idx:03d}'),
            'objective': objective['text'],
            'pillar': objective.get('pillar', 'Unknown'),
            'type': objective.get('type', 'unknown'),
            'matched_actions': matched_actions,
            'alignment_score': float(alignment_score),
            'coverage': coverage,
            'max_similarity': max_similarity,
            'mean_similarity': mean_similarity,
            'num_strong_matches': int(np.sum(similarities >= 0.70)),
            'num_moderate_matches': int(np.sum((similarities >= 0.50) & (similarities < 0.70)))
        }
    
    def _classify_alignment(self, similarity: float) -> str:
        """Classify alignment strength"""
        if similarity >= 0.70:
            return "Strong"
        elif similarity >= 0.50:
            return "Moderate"
        else:
            return "Weak"
    
    def build_alignment_matrix(self) -> pd.DataFrame:
        """
        Build comprehensive alignment matrix as DataFrame
        
        Returns:
            DataFrame with objectives as rows, actions as columns
        """
        if self.embedding_engine.similarity_matrix is None:
            self.embedding_engine.calculate_similarity_matrix()
        
        sim_matrix = self.embedding_engine.similarity_matrix
        
        # Create labels
        objective_labels = [
            f"{obj.get('id', f'OBJ-{i:03d}')}: {obj['text'][:40]}..."
            for i, obj in enumerate(self.doc_processor.strategic_objectives)
        ]
        
        action_labels = [
            f"{act['id']}: {act['title'][:30]}..."
            for act in self.doc_processor.action_items
        ]
        
        # Create DataFrame
        df = pd.DataFrame(
            sim_matrix,
            index=objective_labels,
            columns=action_labels
        )
        
        return df
    
    def identify_gap_objectives(self, threshold=0.50) -> List[Dict]:
        """
        Identify objectives with weak alignment (gaps)
        
        Args:
            threshold: Minimum similarity score (default: 0.50)
            
        Returns:
            List of gap objectives with details
        """
        if self.embedding_engine.similarity_matrix is None:
            self.embedding_engine.calculate_similarity_matrix()
        
        sim_matrix = self.embedding_engine.similarity_matrix
        max_per_objective = np.max(sim_matrix, axis=1)
        
        gap_objectives = []
        for idx, max_sim in enumerate(max_per_objective):
            if max_sim < threshold:
                obj = self.doc_processor.strategic_objectives[idx]
                
                # Determine severity
                if max_sim < 0.30:
                    severity = 'Critical'
                elif max_sim < 0.40:
                    severity = 'High'
                else:
                    severity = 'Medium'
                
                gap_objectives.append({
                    'objective_idx': idx,
                    'objective_id': obj.get('id', f'OBJ-{idx:03d}'),
                    'objective': obj['text'],
                    'pillar': obj.get('pillar', 'Unknown'),
                    'type': obj.get('type', 'unknown'),
                    'max_similarity': float(max_sim),
                    'alignment_score': float(max_sim * 100),
                    'severity': severity,
                    'gap_size': float((threshold - max_sim) * 100)  # How far below threshold
                })
        
        # Sort by severity (lowest similarity first)
        gap_objectives.sort(key=lambda x: x['max_similarity'])
        
        return gap_objectives
    
    def categorize_by_pillar(self) -> Dict:
        """
        Analyze alignment grouped by strategic pillar
        
        Returns:
            Dictionary with pillar-level statistics
        """
        pillar_stats = {}
        
        for obj in self.doc_processor.strategic_objectives:
            pillar = obj.get('pillar', 'Unknown')
            
            if pillar not in pillar_stats:
                pillar_stats[pillar] = {
                    'objectives': [],
                    'objective_ids': [],
                    'scores': [],
                    'count': 0
                }
            
            # Get objective analysis
            obj_idx = self.doc_processor.strategic_objectives.index(obj)
            obj_analysis = self.analyze_objective_alignment(obj_idx)
            
            # Add to pillar stats
            pillar_stats[pillar]['objectives'].append(obj['text'][:60])
            pillar_stats[pillar]['objective_ids'].append(obj.get('id', f'OBJ-{obj_idx:03d}'))
            pillar_stats[pillar]['scores'].append(obj_analysis['alignment_score'])
            pillar_stats[pillar]['count'] += 1
        
        # Calculate aggregates for each pillar
        for pillar, stats in pillar_stats.items():
            scores = stats['scores']
            stats['average_score'] = float(np.mean(scores))
            stats['min_score'] = float(np.min(scores))
            stats['max_score'] = float(np.max(scores))
            stats['median_score'] = float(np.median(scores))
            
            # Classification
            avg = stats['average_score']
            if avg >= 70:
                stats['pillar_status'] = 'Strong'
            elif avg >= 50:
                stats['pillar_status'] = 'Moderate'
            else:
                stats['pillar_status'] = 'Weak'
        
        return pillar_stats
    
    # ============================================================
    # Gap Detection
    # ============================================================
    
    def detect_gaps(self) -> Dict:
        """
        Comprehensive gap detection across multiple dimensions
        
        Returns:
            Dictionary with different types of gaps identified
        """
        gaps = {
            'weak_objectives': [],
            'orphan_actions': [],
            'pillar_gaps': [],
            'coverage_gaps': []
        }
        
        # 1. Weak objectives (already covered by identify_gap_objectives)
        gaps['weak_objectives'] = self.identify_gap_objectives(threshold=0.50)
        
        # 2. Orphan actions (actions not strongly linked to any objective)
        sim_matrix = self.embedding_engine.similarity_matrix
        max_per_action = np.max(sim_matrix, axis=0)
        
        for idx, max_sim in enumerate(max_per_action):
            if max_sim < 0.50:
                action = self.doc_processor.action_items[idx]
                gaps['orphan_actions'].append({
                    'action_id': action['id'],
                    'title': action['title'],
                    'pillar': action.get('pillar', 'Unknown'),
                    'max_similarity': float(max_sim),
                    'recommendation': 'Review strategic alignment or consider removing'
                })
        
        # 3. Pillar-level gaps
        pillar_stats = self.categorize_by_pillar()
        for pillar, stats in pillar_stats.items():
            if stats['pillar_status'] == 'Weak':
                gaps['pillar_gaps'].append({
                    'pillar': pillar,
                    'average_score': stats['average_score'],
                    'objective_count': stats['count'],
                    'status': stats['pillar_status']
                })
        
        # 4. Coverage gaps (objectives with no moderate/strong matches)
        for idx, obj in enumerate(self.doc_processor.strategic_objectives):
            similarities = sim_matrix[idx]
            if not np.any(similarities >= 0.50):
                gaps['coverage_gaps'].append({
                    'objective_id': obj.get('id', f'OBJ-{idx:03d}'),
                    'objective': obj['text'],
                    'pillar': obj.get('pillar', 'Unknown'),
                    'best_match_score': float(np.max(similarities) * 100)
                })
        
        return gaps
    
    # ============================================================
    # Export & Summary
    # ============================================================
    
    def generate_summary_report(self) -> Dict:
        """
        Generate comprehensive summary report
        
        Returns:
            Complete analysis summary
        """
        print("\nGenerating comprehensive synchronization report...")
        
        # Overall alignment
        overall = self.calculate_overall_alignment()
        
        # Per-objective analysis
        objective_details = []
        for i in range(len(self.doc_processor.strategic_objectives)):
            obj_analysis = self.analyze_objective_alignment(i)
            objective_details.append(obj_analysis)
        
        # Gap analysis
        gaps = self.detect_gaps()
        
        # Pillar analysis
        pillar_stats = self.categorize_by_pillar()
        
        report = {
            'overall_alignment': overall,
            'objective_details': objective_details,
            'gaps': gaps,
            'pillar_stats': pillar_stats,
            'summary': {
                'total_objectives': overall['total_objectives'],
                'total_actions': overall['total_actions'],
                'well_covered_objectives': overall['distribution']['strong'] + overall['distribution']['moderate'],
                'gap_objectives_count': len(gaps['weak_objectives']),
                'orphan_actions_count': len(gaps['orphan_actions']),
                'weak_pillars_count': len(gaps['pillar_gaps'])
            }
        }
        
        return report
    
    def save_report(self, output_file: str):
        """Save comprehensive report to JSON"""
        report = self.generate_summary_report()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f" Report saved to {output_file}")
        return report


# ============================================================
# Test Script
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("SYNCHRONIZATION ENGINE TEST")
    print("="*80)
    
    # Import required modules
    import sys
    sys.path.append('src')
    from document_processor import BrandixDocumentProcessor
    from embedding_engine import EmbeddingEngine
    from vector_store import FAISSVectorStore
    
    # 1. Load documents
    print("\nStep 1: Loading documents...")
    processor = BrandixDocumentProcessor()
    objectives = processor.load_strategic_plan('data/BRANDIX_STRATEGIC_PLAN_2025.docx')
    actions = processor.load_action_plan('data/BRANDIX_ACTION_PLAN.docx')
    
    # 2. Create embeddings
    print("\nStep 2: Creating embeddings...")
    engine = EmbeddingEngine()
    engine.embed_objectives(objectives)
    engine.embed_actions(actions)
    engine.calculate_similarity_matrix()
    
    # 3. Create vector store
    print("\nStep 3: Creating vector store...")
    vs = FAISSVectorStore()
    vs.add_vectors(engine.action_embeddings, [{'id': a['id']} for a in actions])
    
    # 4. Initialize synchronization engine
    print("\nStep 4: Initializing Synchronization Engine...")
    sync_engine = SynchronizationEngine(processor, engine, vs)
    
    # 5. Test overall alignment
    print("\n" + "="*80)
    print("OVERALL ALIGNMENT ANALYSIS")
    print("="*80)
    overall = sync_engine.calculate_overall_alignment()
    print(f"Overall Score: {overall['overall_score']:.2f}%")
    print(f"Mean Max Similarity: {overall['mean_max_similarity']:.2f}%")
    print(f"Classification: {overall['classification']}")
    print(f"Coverage Rate: {overall['coverage_rate']:.1f}%")
    print(f"\nDistribution:")
    print(f"  Strong (≥70%): {overall['distribution']['strong']}")
    print(f"  Moderate (50-70%): {overall['distribution']['moderate']}")
    print(f"  Weak (<50%): {overall['distribution']['weak']}")
    
    # 6. Test strategy-wise analysis
    print("\n" + "="*80)
    print("STRATEGY-WISE ANALYSIS (Sample: First Objective)")
    print("="*80)
    obj_analysis = sync_engine.analyze_objective_alignment(0)
    print(f"Objective: {obj_analysis['objective'][:80]}...")
    print(f"Pillar: {obj_analysis['pillar']}")
    print(f"Alignment Score: {obj_analysis['alignment_score']:.2f}%")
    print(f"Coverage: {obj_analysis['coverage']}")
    print(f"\nTop 3 Matched Actions:")
    for i, action in enumerate(obj_analysis['matched_actions'][:3], 1):
        print(f"{i}. [{action['action_id']}] {action['title']}")
        print(f"   Similarity: {action['similarity']:.2%} ({action['alignment_strength']})")
    
    # 7. Test gap detection
    print("\n" + "="*80)
    print("GAP DETECTION")
    print("="*80)
    gaps = sync_engine.detect_gaps()
    print(f"Weak Objectives: {len(gaps['weak_objectives'])}")
    print(f"Orphan Actions: {len(gaps['orphan_actions'])}")
    print(f"Weak Pillars: {len(gaps['pillar_gaps'])}")
    print(f"Coverage Gaps: {len(gaps['coverage_gaps'])}")
    
    if gaps['weak_objectives']:
        print("\nTop 3 Gap Objectives:")
        for i, gap in enumerate(gaps['weak_objectives'][:3], 1):
            print(f"{i}. [{gap['objective_id']}] Score: {gap['alignment_score']:.1f}% ({gap['severity']})")
            print(f"   {gap['objective'][:70]}...")
    
    # 8. Test pillar analysis
    print("\n" + "="*80)
    print("PILLAR-WISE ANALYSIS")
    print("="*80)
    pillar_stats = sync_engine.categorize_by_pillar()
    for pillar, stats in pillar_stats.items():
        print(f"\n{pillar}:")
        print(f"  Objectives: {stats['count']}")
        print(f"  Average Score: {stats['average_score']:.1f}%")
        print(f"  Status: {stats['pillar_status']}")
    
    # 9. Generate and save full report
    print("\n" + "="*80)
    print("GENERATING FULL REPORT")
    print("="*80)
    report = sync_engine.save_report('outputs/synchronization_report.json')
    
    print("\n" + "="*80)    
    print("="*80)
    print("\nYou have successfully built:")
    print("   Overall synchronization assessment")
    print("   Strategy-wise alignment analysis")
    print("   Gap detection system")
    print("   Pillar-level categorization")
    print("\nCheck outputs/synchronization_report.json for full results!")