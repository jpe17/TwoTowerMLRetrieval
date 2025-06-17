import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

from model import TwoTowerModel
from tokenizer import PretrainedTokenizer


class SimpleEvaluator:
    """Simple evaluator with query evaluation and text search capabilities."""
    
    def __init__(self, model: TwoTowerModel, tokenizer: PretrainedTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_query(self, query: str, documents: List[str], positive_docs: List[str] = None) -> List[Tuple[str, float, bool]]:
        """
        Evaluate a query against documents and return top 10 results with correctness.
        
        Args:
            query: Query string
            documents: List of documents to search
            positive_docs: List of known correct documents (optional)
            
        Returns:
            List of (document, score, is_correct) tuples
        """
        with torch.no_grad():
            # Encode query
            query_tokens = self.tokenizer.encode(query)
            query_tensor = pad_sequence([torch.tensor(query_tokens, dtype=torch.long)], batch_first=True).to(self.device)
            query_vec = self.model.encode_query(query_tensor)

            # Encode documents
            doc_tokens = [torch.tensor(self.tokenizer.encode(doc), dtype=torch.long) for doc in documents]
            doc_tensors = pad_sequence(doc_tokens, batch_first=True).to(self.device)
            doc_vecs = self.model.encode_document(doc_tensors)

            # Calculate cosine similarity
            scores = F.cosine_similarity(query_vec, doc_vecs, dim=1)
            
            # Get top 10 results
            top_indices = torch.argsort(scores, descending=True)[:10]
            
            results = []
            positive_set = set(positive_docs) if positive_docs else set()
            
            for idx in top_indices:
                doc = documents[idx.item()]
                score = scores[idx].item()
                is_correct = doc in positive_set if positive_docs else None
                results.append((doc, score, is_correct))
            
            return results
    
    def search_similar(self, text: str, documents: List[str], embeddings_path: str = None) -> List[Tuple[str, float]]:
        """
        Search for similar documents using saved embeddings or compute on-the-fly.
        
        Args:
            text: Input text to find similar documents for
            documents: List of documents to search through
            embeddings_path: Path to saved document embeddings (optional)
            
        Returns:
            List of (document, similarity_score) tuples
        """
        with torch.no_grad():
            # Encode input text
            text_tokens = self.tokenizer.encode(text)
            text_tensor = pad_sequence([torch.tensor(text_tokens, dtype=torch.long)], batch_first=True).to(self.device)
            text_vec = self.model.encode_query(text_tensor)
            
            # Load or compute document embeddings
            if embeddings_path and os.path.exists(embeddings_path):
                doc_embeddings = torch.tensor(np.load(embeddings_path), device=self.device)
            else:
                # Compute embeddings on-the-fly
                doc_tokens = [torch.tensor(self.tokenizer.encode(doc), dtype=torch.long) for doc in documents]
                doc_tensors = pad_sequence(doc_tokens, batch_first=True).to(self.device)
                doc_embeddings = self.model.encode_document(doc_tensors)
            
            # Calculate cosine similarity
            similarities = F.cosine_similarity(text_vec, doc_embeddings, dim=1)
            
            # Get top 10 results
            top_indices = torch.argsort(similarities, descending=True)[:10]
            
            results = []
            for idx in top_indices:
                doc = documents[idx.item()]
                score = similarities[idx].item()
                results.append((doc, score))
            
            return results
    
    def print_query_results(self, query: str, results: List[Tuple[str, float, bool]]):
        """Print formatted query evaluation results."""
        print(f"\n🔍 Query: {query}")
        print("=" * 80)
        
        for i, (doc, score, is_correct) in enumerate(results, 1):
            status = "✅" if is_correct else "❌" if is_correct is False else "❓"
            print(f"{i:2d}. {status} Score: {score:.4f}")
            print(f"    {doc[:100]}...")
            print()
    
    def print_search_results(self, text: str, results: List[Tuple[str, float]]):
        """Print formatted search results."""
        print(f"\n🔎 Similar to: {text}")
        print("=" * 80)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i:2d}. Score: {score:.4f}")
            print(f"    {doc[:100]}...")
            print() 