import faiss
import numpy as np
import pickle
from typing import List, Dict

class FAISSVectorStore:
    def __init__(self, dimension=384):
        """Initialize FAISS index"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
    
    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add vectors and metadata to index"""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match {self.dimension}")
        
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k=5) -> List[Dict]:
        """Search for k nearest neighbors"""
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(distance)
                result['similarity'] = float(1 / (1 + distance))  # Convert distance to similarity
                results.append(result)
        
        return results
    
    def save(self, filepath: str):
        """Save index and metadata to disk"""
        # Save FAISS index
        faiss.write_index(self.index, filepath)
        
        # Save metadata
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Saved index to {filepath}")
    
    def load(self, filepath: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(filepath)
        
        # Load metadata
        with open(f"{filepath}.meta", 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index from {filepath}. Total vectors: {self.index.ntotal}")

# Test
if __name__ == "__main__":
    from document_processor import BrandixDocumentProcessor
    from embedding_engine import EmbeddingEngine
    
    # Load documents
    processor = BrandixDocumentProcessor()
    objectives = processor.load_strategic_plan('data/BRANDIX_STRATEGIC_PLAN_2025.docx')
    actions = processor.load_action_plan('data/BRANDIX_ACTION_PLAN.docx')
    
    # Create embeddings
    engine = EmbeddingEngine()
    action_embeddings = engine.embed_actions(actions)
    
    # Create vector store
    vector_store = FAISSVectorStore(dimension=384)
    
    # Add actions to vector store
    action_metadata = [
        {
            'id': action['id'],
            'title': action['title'],
            'text': action['text'],
            'pillar': action.get('pillar', 'Unknown')
        }
        for action in actions
    ]
    
    vector_store.add_vectors(action_embeddings, action_metadata)
    
    # Test search
    test_query = "renewable energy solar installation"
    query_embedding = engine.model.encode([test_query])[0]
    results = vector_store.search(query_embedding, k=3)
    
    print("\nSearch Results for: 'renewable energy solar installation'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Pillar: {result['pillar']}")
    
    # Save to disk
    vector_store.save('models/brandix_actions.faiss')