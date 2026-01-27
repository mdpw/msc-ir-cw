"""
RAG Pipeline - Retrieval-Augmented Generation
Uses document context to generate more relevant improvement suggestions
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, llm_engine, embedding_engine, doc_processor):
        """
        Initialize RAG pipeline
        
        Args:
            llm_engine: LLMEngine instance
            embedding_engine: EmbeddingEngine instance
            doc_processor: DocumentProcessor instance
        """
        self.llm = llm_engine
        self.embedding_engine = embedding_engine
        self.doc_processor = doc_processor
        self.chunk_embeddings = None
        self.chunks = []
    
    def create_document_chunks(self, max_chunk_size=500):
        """
        Create overlapping chunks from strategic and action plans
        """
        print("\n📄 Creating document chunks for RAG...")
        
        chunks = []
        
        # Chunk strategic objectives
        for obj in self.doc_processor.strategic_objectives:
            chunk_text = f"""Strategic Objective: {obj['text']}
Pillar: {obj.get('pillar', 'Unknown')}
Type: {obj.get('type', 'unknown')}
Context: This is a strategic objective from Brandix's 2025-2030 strategic plan."""
            
            chunks.append({
                'text': chunk_text,
                'source': 'strategic_plan',
                'id': obj.get('id', 'unknown'),
                'pillar': obj.get('pillar', 'Unknown')
            })
        
        # Chunk action items
        for action in self.doc_processor.action_items:
            chunk_text = f"""Action Item: {action['title']}
ID: {action['id']}
Pillar: {action.get('pillar', 'Unknown')}
Context: This is a Year 1 action item from Brandix's 2025-2026 action plan."""
            
            chunks.append({
                'text': chunk_text,
                'source': 'action_plan',
                'id': action['id'],
                'pillar': action.get('pillar', 'Unknown')
            })
        
        self.chunks = chunks
        print(f"✓ Created {len(chunks)} document chunks")
        
        # Create embeddings for chunks
        chunk_texts = [c['text'] for c in chunks]
        self.chunk_embeddings = self.embedding_engine.model.encode(
            chunk_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        print(f"✓ Generated embeddings for all chunks")
        
        return chunks
    
    def retrieve_relevant_context(self, query: str, k=5) -> List[Dict]:
        """
        Retrieve top-k most relevant document chunks for a query
        
        Args:
            query: Query text
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.chunk_embeddings is None:
            self.create_document_chunks()
        
        # Encode query
        query_embedding = self.embedding_engine.model.encode([query])[0]
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.chunk_embeddings
        )[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['similarity'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def generate_rag_enhanced_suggestions(self, gap_objective: Dict, top_actions: List[Dict]) -> Dict:
        """
        Generate improvement suggestions using RAG for better context
        
        Args:
            gap_objective: Objective with weak alignment
            top_actions: Top matching actions (even if weak)
            
        Returns:
            Enhanced suggestions with retrieved context
        """
        print(f"\n🔍 Using RAG for: {gap_objective['objective'][:60]}...")
        
        # Retrieve relevant context
        query = f"{gap_objective['pillar']} {gap_objective['objective']}"
        relevant_chunks = self.retrieve_relevant_context(query, k=5)
        
        # Build context string
        context_text = "\n\n".join([
            f"[{i+1}] {chunk['text']}" 
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Build enhanced prompt
        obj_text = gap_objective['objective']
        pillar = gap_objective.get('pillar', 'Unknown')
        score = gap_objective['alignment_score']
        
        actions_context = "\n".join([
            f"- {a.get('title', a.get('text', 'N/A'))} (Similarity: {a.get('similarity', 0):.1%})"
            for a in top_actions[:5]
        ])
        
        prompt = f"""You are a strategic planning expert for Brandix, an apparel manufacturing company.

OBJECTIVE REQUIRING IMPROVEMENT (Current Alignment: {score:.1f}%):
Pillar: {pillar}
Objective: {obj_text}

CURRENT BEST MATCHING ACTIONS:
{actions_context}

RELEVANT CONTEXT FROM DOCUMENTS:
{context_text}

TASK: Generate specific, actionable improvement suggestions to increase alignment for this objective.

Provide suggestions in these categories:

**NEW ACTIONS** (2-3 specific action items):
[Focus on concrete, implementable actions that directly address the objective]

**KPI ENHANCEMENTS** (2-3 measurable KPIs):
[Include specific targets and metrics with timelines]

**TIMELINE RECOMMENDATIONS**:
[Suggest quarterly milestones for Year 1]

**RESOURCE REQUIREMENTS**:
[Identify budget, team, or infrastructure needs]

**INTEGRATION OPPORTUNITIES**:
[How can this connect with existing actions or objectives?]

Be specific, realistic, and ensure suggestions align with Brandix's sustainability and innovation focus."""

        # Generate with LLM
        response = self.llm.generate(prompt, max_tokens=800)
        
        # Parse response
        suggestions = self._parse_suggestions(response)
        
        return {
            'objective_id': gap_objective.get('objective_id', 'Unknown'),
            'objective': obj_text,
            'pillar': pillar,
            'current_score': score,
            'suggestions': suggestions,
            'retrieved_context': relevant_chunks,
            'raw_response': response,
            'method': 'RAG-enhanced'
        }
    
    def _parse_suggestions(self, response: str) -> Dict:
        """Parse LLM response into structured categories"""
        categories = {
            'new_actions': [],
            'kpi_enhancements': [],
            'timeline_recommendations': [],
            'resource_requirements': [],
            'integration_opportunities': []
        }
        
        current_category = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Detect category headers
            line_upper = line.upper()
            if 'NEW ACTION' in line_upper:
                current_category = 'new_actions'
            elif 'KPI' in line_upper or 'METRIC' in line_upper:
                current_category = 'kpi_enhancements'
            elif 'TIMELINE' in line_upper or 'MILESTONE' in line_upper:
                current_category = 'timeline_recommendations'
            elif 'RESOURCE' in line_upper or 'BUDGET' in line_upper:
                current_category = 'resource_requirements'
            elif 'INTEGRATION' in line_upper or 'OPPORTUNIT' in line_upper:
                current_category = 'integration_opportunities'
            elif line.startswith(('-', '•', '*', '1.', '2.', '3.')):
                # Add bullet point to current category
                if current_category:
                    text = line.lstrip('-•*123456789.').strip()
                    if text and len(text) > 10:  # Minimum length
                        categories[current_category].append(text)
        
        return categories
    
    def batch_generate_improvements(self, gap_objectives: List[Dict], actions_per_objective: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Generate improvements for multiple gap objectives
        
        Args:
            gap_objectives: List of objectives needing improvement
            actions_per_objective: Dict mapping objective index to top actions
            
        Returns:
            List of improvement suggestions for each objective
        """
        print(f"\n🤖 Generating RAG-enhanced improvements for {len(gap_objectives)} objectives...")
        
        all_improvements = []
        
        for i, gap in enumerate(gap_objectives, 1):
            print(f"\n[{i}/{len(gap_objectives)}] Processing: {gap['objective'][:50]}...")
            
            obj_idx = gap['objective_idx']
            top_actions = actions_per_objective.get(obj_idx, [])
            
            try:
                improvement = self.generate_rag_enhanced_suggestions(gap, top_actions)
                all_improvements.append(improvement)
                print(f"  ✓ Generated {sum(len(v) for v in improvement['suggestions'].values())} suggestions")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        print(f"\n✓ Completed! Generated improvements for {len(all_improvements)}/{len(gap_objectives)} objectives")
        return all_improvements


# Integration with ImprovementGenerator
class ImprovementGenerator:
    """High-level improvement generator using RAG pipeline"""
    
    def __init__(self, synchronization_engine, llm_engine):
        """
        Initialize improvement generator
        
        Args:
            synchronization_engine: SynchronizationEngine instance
            llm_engine: LLMEngine instance
        """
        self.sync_engine = synchronization_engine
        self.llm = llm_engine
        
        # Create RAG pipeline
        self.rag_pipeline = RAGPipeline(
            llm_engine,
            synchronization_engine.embedding_engine,
            synchronization_engine.doc_processor
        )
        
        # Initialize chunks
        self.rag_pipeline.create_document_chunks()
    
    def generate_improvements_for_gaps(self, threshold=0.50, max_objectives=10) -> Dict:
        """
        Generate improvements for all gap objectives
        
        Args:
            threshold: Alignment threshold for gaps (default 0.50)
            max_objectives: Maximum number to process (default 10)
            
        Returns:
            Dictionary with improvements and summary
        """
        print("\n" + "="*80)
        print("IMPROVEMENT GENERATION SYSTEM")
        print("="*80)
        
        # Get gap objectives
        gap_objectives = self.sync_engine.identify_gap_objectives(threshold=threshold)
        
        if not gap_objectives:
            print("\n✓ No gap objectives found! All objectives are well-aligned.")
            return {
                'improvements': [],
                'summary': {
                    'total_gaps': 0,
                    'processed': 0,
                    'total_suggestions': 0
                }
            }
        
        # Limit number to process
        gaps_to_process = gap_objectives[:max_objectives]
        print(f"\nFound {len(gap_objectives)} gap objectives")
        print(f"Processing top {len(gaps_to_process)} (sorted by severity)\n")
        
        # Get top actions for each gap objective
        actions_map = {}
        for gap in gaps_to_process:
            obj_idx = gap['objective_idx']
            obj_analysis = self.sync_engine.analyze_objective_alignment(obj_idx)
            actions_map[obj_idx] = obj_analysis['matched_actions'][:5]
        
        # Generate improvements using RAG
        improvements = self.rag_pipeline.batch_generate_improvements(
            gaps_to_process,
            actions_map
        )
        
        # Calculate summary
        total_suggestions = sum(
            sum(len(v) for v in imp['suggestions'].values())
            for imp in improvements
        )
        
        summary = {
            'total_gaps': len(gap_objectives),
            'processed': len(improvements),
            'total_suggestions': total_suggestions,
            'by_severity': {
                'critical': len([g for g in gaps_to_process if g.get('severity') == 'Critical']),
                'high': len([g for g in gaps_to_process if g.get('severity') == 'High']),
                'medium': len([g for g in gaps_to_process if g.get('severity') == 'Medium'])
            }
        }
        
        return {
            'improvements': improvements,
            'summary': summary,
            'gap_objectives': gaps_to_process
        }
    
    def save_improvements(self, output_file: str):
        """Generate and save improvement suggestions"""
        import json
        import os
        
        results = self.generate_improvements_for_gaps()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Improvements saved to {output_file}")
        
        return results


# Test
if __name__ == "__main__":
    import sys
    sys.path.append('src')
    
    from document_processor import DocumentProcessor
    from embedding_engine import EmbeddingEngine
    from vector_store import FAISSVectorStore
    from synchronization_engine import SynchronizationEngine
    from llm_engine import LLMEngine
    
    print("="*80)
    print("RAG PIPELINE TEST")
    print("="*80)
    
    # Load everything
    print("\n1. Loading documents...")
    processor = DocumentProcessor()
    objectives = processor.load_strategic_plan('data/BRANDIX_STRATEGIC_PLAN_2025.docx')
    actions = processor.load_action_plan('data/BRANDIX_ACTION_PLAN.docx')
    
    print("\n2. Creating embeddings...")
    engine = EmbeddingEngine()
    engine.embed_objectives(objectives)
    engine.embed_actions(actions)
    engine.calculate_similarity_matrix()
    
    print("\n3. Creating vector store...")
    vs = FAISSVectorStore()
    vs.add_vectors(engine.action_embeddings, [{'id': a['id']} for a in actions])
    
    print("\n4. Initializing synchronization engine...")
    sync_engine = SynchronizationEngine(processor, engine, vs)
    
    print("\n5. Initializing LLM...")
    llm = LLMEngine(model_name="phi3:mini")
    if not llm.test_connection():
        print("❌ Ollama not running!")
        exit(1)
    
    print("\n6. Creating improvement generator...")
    improvement_gen = ImprovementGenerator(sync_engine, llm)
    
    # Generate improvements
    print("\n7. Generating improvements for gap objectives...")
    results = improvement_gen.generate_improvements_for_gaps(threshold=0.50, max_objectives=5)
    
    # Display results
    print("\n" + "="*80)
    print("IMPROVEMENT GENERATION SUMMARY")
    print("="*80)
    print(f"Total Gap Objectives: {results['summary']['total_gaps']}")
    print(f"Processed: {results['summary']['processed']}")
    print(f"Total Suggestions Generated: {results['summary']['total_suggestions']}")
    print(f"\nBy Severity:")
    print(f"  Critical: {results['summary']['by_severity']['critical']}")
    print(f"  High: {results['summary']['by_severity']['high']}")
    print(f"  Medium: {results['summary']['by_severity']['medium']}")
    
    # Show sample improvement
    if results['improvements']:
        print("\n" + "="*80)
        print("SAMPLE IMPROVEMENT (First Gap Objective)")
        print("="*80)
        imp = results['improvements'][0]
        print(f"\nObjective: {imp['objective']}")
        print(f"Pillar: {imp['pillar']}")
        print(f"Current Score: {imp['current_score']:.1f}%")
        print(f"\nNEW ACTIONS ({len(imp['suggestions']['new_actions'])}):")
        for i, action in enumerate(imp['suggestions']['new_actions'], 1):
            print(f"  {i}. {action}")
        
        print(f"\nKPI ENHANCEMENTS ({len(imp['suggestions']['kpi_enhancements'])}):")
        for i, kpi in enumerate(imp['suggestions']['kpi_enhancements'], 1):
            print(f"  {i}. {kpi}")
    
    # Save results
    improvement_gen.save_improvements('outputs/improvements.json')
    
    print("\n✓ RAG Pipeline Complete!")
    print("\nNext: Integrate into dashboard (app.py)")