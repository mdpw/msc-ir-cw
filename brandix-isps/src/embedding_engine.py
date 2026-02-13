from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json
import sys
import os

# Add parent directory to path to import document_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.document_processor import DocumentProcessor

class EmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize embedding model"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.objective_embeddings = None
        self.action_embeddings = None
        self.similarity_matrix = None
    
    def _prepare_text_for_embedding(self, item: Dict) -> str:
        """Prepare comprehensive text for embedding"""
        # Combine multiple fields for richer representation
        parts = []
        
        # Add main text first (most important)
        if 'text' in item:
            parts.append(item['text'])
        
        # For actions, add title if different from text
        if 'title' in item and item['title'] != item.get('text', ''):
            parts.append(item['title'])
            
        # Add pillar only at the end to provide context without dominating
        if 'pillar' in item and item['pillar'] != 'Unknown':
            parts.append(f"({item['pillar']})")
        
        # Combine all parts
        full_text = ". ".join(parts)
        
        return full_text
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_objectives(self, objectives: List[Dict]):
        """Create embeddings for strategic objectives"""
        print(f"\nCreating embeddings for {len(objectives)} objectives...")
        
        # Prepare rich text representations
        texts = [self._prepare_text_for_embedding(obj) for obj in objectives]
        
        # Show sample
        if texts:
            print(f"Sample objective text (first one):")
            print(f"  {texts[0][:150]}...")
        
        self.objective_embeddings = self.create_embeddings(texts)
        return self.objective_embeddings
    
    def embed_actions(self, actions: List[Dict]):
        """Create embeddings for action items"""
        print(f"\nCreating embeddings for {len(actions)} actions...")
        
        # Prepare rich text representations
        texts = [self._prepare_text_for_embedding(action) for action in actions]
        
        # Show sample
        if texts:
            print(f"Sample action text (first one):")
            print(f"  {texts[0][:150]}...")
        
        self.action_embeddings = self.create_embeddings(texts)
        return self.action_embeddings
    
    def calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate cosine similarity between objectives and actions"""
        if self.objective_embeddings is None or self.action_embeddings is None:
            raise ValueError("Must embed objectives and actions first")
        
        print("\nCalculating similarity matrix...")
        raw_sim = cosine_similarity(
            self.objective_embeddings,
            self.action_embeddings
        )
        
        # Normalization/Sharpening: Apply power function to increase contrast
        # This helps push lower semantic matches further down while keeping strong matches high
        # Improves correlation with expert judgement
        self.similarity_matrix = np.power(np.maximum(raw_sim, 0), 1.5)
        
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def find_top_matches(self, objective_idx: int, k=5) -> List[Dict]:
        """Find top-K matching actions for an objective"""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        similarities = self.similarity_matrix[objective_idx]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        matches = []
        for idx in top_indices:
            matches.append({
                'action_idx': int(idx),
                'similarity': float(similarities[idx]),
                'alignment_strength': self._classify_alignment(similarities[idx])
            })
        return matches
    
    def _classify_alignment(self, similarity: float) -> str:
        """Classify alignment strength"""
        if similarity >= 0.70:
            return "Strong"
        elif similarity >= 0.50:
            return "Moderate"
        else:
            return "Weak"
    
    def analyze_overall_alignment(self) -> Dict:
        """Calculate overall alignment statistics"""
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        
        print("\nAnalyzing alignment...")
        
        # Calculate metrics
        mean_similarity = np.mean(self.similarity_matrix)
        max_per_objective = np.max(self.similarity_matrix, axis=1)
        mean_max_similarity = np.mean(max_per_objective)
        
        # Count alignment strengths (using max similarity per objective)
        strong_count = int(np.sum(max_per_objective >= 0.70))
        moderate_count = int(np.sum((max_per_objective >= 0.50) & (max_per_objective < 0.70)))
        weak_count = int(np.sum(max_per_objective < 0.50))
        
        results = {
            'overall_score': float(mean_similarity * 100),
            'mean_max_similarity': float(mean_max_similarity * 100),
            'strong_alignments': strong_count,
            'moderate_alignments': moderate_count,
            'weak_alignments': weak_count,
            'total_objectives': len(max_per_objective),
            'coverage_rate': float((strong_count + moderate_count) / len(max_per_objective) * 100)
        }
        
        return results
    
    def save_analysis(self, output_file: str):
        """Save analysis results to JSON"""
        results = self.analyze_overall_alignment()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis saved to {output_file}")
        return results

# Test
if __name__ == "__main__":
    print("="*80)
    print("BRANDIX EMBEDDING ENGINE TEST")
    print("="*80)
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load documents
    print("\nStep 1: Loading documents...")
    processor = DocumentProcessor()
    objectives = processor.load_strategic_plan('data/BRANDIX_STRATEGIC_PLAN_2025.docx')
    actions = processor.load_action_plan('data/BRANDIX_ACTION_PLAN_YEAR_1.docx')
    
    if not objectives or not actions:
        print("\nERROR: No objectives or actions loaded!")
        print("Make sure your data files are in the correct location.")
        sys.exit(1)
    
    # Create embeddings
    print("\nStep 2: Creating embeddings...")
    engine = EmbeddingEngine()
    engine.embed_objectives(objectives)
    engine.embed_actions(actions)
    
    # Calculate similarity
    print("\nStep 3: Calculating similarity...")
    engine.calculate_similarity_matrix()
    
    # Analyze alignment
    print("\nStep 4: Analyzing alignment...")
    results = engine.analyze_overall_alignment()
    
    # Display results
    print("\n" + "="*80)
    print("ALIGNMENT ANALYSIS RESULTS")
    print("="*80)
    print(f"Overall Alignment Score:     {results['overall_score']:.2f}%")
    print(f"Mean Max Similarity:         {results['mean_max_similarity']:.2f}%")
    print(f"Total Objectives Analyzed:   {results['total_objectives']}")
    print(f"\nAlignment Distribution:")
    print(f"  Strong (≥70%):   {results['strong_alignments']} objectives")
    print(f"  Moderate (50-70%): {results['moderate_alignments']} objectives")
    print(f"  Weak (<50%):     {results['weak_alignments']} objectives")
    print(f"\nCoverage Rate:              {results['coverage_rate']:.1f}%")
    print("="*80)
    
    # Save results
    engine.save_analysis('outputs/embedding_analysis.json')
    
    # Show sample matches
    print("\nSample Matches (First Objective):")
    print(f"Objective: {objectives[0]['text'][:80]}...")
    matches = engine.find_top_matches(0, k=3)
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. Action #{match['action_idx']} - {match['alignment_strength']}")
        print(f"   Similarity: {match['similarity']:.2%}")
        print(f"   Title: {actions[match['action_idx']]['title'][:60]}...")
    
    print("\nNext Steps:")
    print("  - Check outputs/embedding_analysis.json for detailed results")
    print("  - Proceed to Day 3: Local LLM Integration")