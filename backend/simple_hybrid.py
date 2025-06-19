import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from query_inferencer import QueryInferencer


class SimpleHybridRetriever:
    """Simple hybrid retrieval combining TF-IDF + dense embeddings."""
    
    def __init__(self, artifacts_path: str, alpha: float = 0.5):
        """
        Args:
            artifacts_path: Path to trained model artifacts
            alpha: Weight for dense vs TF-IDF (0.5 = equal weight)
        """
        self.dense_retriever = QueryInferencer(artifacts_path)
        self.alpha = alpha
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
        self.documents = []
        self.doc_embeddings = None
        
    def fit(self, documents: List[str]):
        """Fit on document corpus."""
        print(f"Fitting on {len(documents)} documents...")
        self.documents = documents
        
        # Fit TF-IDF
        self.tfidf_matrix = self.tfidf.fit_transform(documents)
        
        # Get dense embeddings
        self.doc_embeddings = []
        for doc in documents:
            emb = self.dense_retriever.get_query_embedding(doc)  # Use same encoder
            self.doc_embeddings.append(emb)
        self.doc_embeddings = np.array(self.doc_embeddings)
        
        print("✅ Fitting complete!")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search and return (document, score) pairs."""
        
        # Get TF-IDF scores
        query_tfidf = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        
        # Get dense scores
        query_emb = self.dense_retriever.get_query_embedding(query)
        dense_scores = cosine_similarity([query_emb], self.doc_embeddings)[0]
        
        # Combine scores
        combined_scores = self.alpha * dense_scores + (1 - self.alpha) * tfidf_scores
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], combined_scores[idx]))
        
        return results


# Usage example
if __name__ == "__main__":
    # Sample documents
    docs = [
        "Machine learning algorithms learn from data",
        "Deep neural networks have multiple layers", 
        "Natural language processing understands text",
        "Computer vision processes images and video"
    ]
    
    # Create retriever (update path to your artifacts)
    retriever = SimpleHybridRetriever("artifacts/run_20240101_120000", alpha=0.6)
    retriever.fit(docs)
    
    # Search
    results = retriever.search("machine learning", top_k=2)
    for doc, score in results:
        print(f"{score:.3f}: {doc}") 