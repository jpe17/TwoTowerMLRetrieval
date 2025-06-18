import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

from model import TwoTowerModel
from tokenizer import PretrainedTokenizer


class SimpleEvaluator:
    """Task-aware evaluator with query evaluation and text search capabilities."""
    
    def __init__(self, model: TwoTowerModel, tokenizer: PretrainedTokenizer, device: torch.device, config: dict = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or {}
        self.model.eval()
        
        # Detect task mode
        self.task_mode = self.config.get('TASK_MODE', 'retrieval')
        self.is_ranking = self.task_mode == 'ranking'
    
    def score_query_doc_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair (for ranking tasks)."""
        if not self.is_ranking:
            raise ValueError("score_query_doc_pair only available for ranking tasks")
        
        # Tokenize and encode query and document together
        query_tokens = self.tokenizer.encode(query)
        doc_tokens = self.tokenizer.encode(document)
        
        # Combine tokens (you might need to adjust this based on your model)
        combined_tokens = query_tokens + doc_tokens  # Simple concatenation
        combined_tensor = torch.tensor(combined_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Assuming your combined model outputs a single relevance score
            score = self.model.encode_text(combined_tensor)
            return score.squeeze().item()
    
    def evaluate_query(self, query: str, documents: List[str], positive_docs: List[str] = None) -> List[Tuple[str, float, bool]]:
        """
        Evaluate a query against documents - task-aware.
        
        Args:
            query: Query string
            documents: List of documents to search
            positive_docs: List of known correct documents (optional)
            
        Returns:
            List of (document, score, is_correct) tuples
        """
        with torch.no_grad():
            if self.is_ranking:
                # For ranking: score each query-document pair individually
                scores = []
                for doc in documents:
                    score = self.score_query_doc_pair(query, doc)
                    scores.append(score)
                scores = torch.tensor(scores)
            else:
                # For retrieval: use separate encoders and cosine similarity
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
    
    def compute_ranking_metrics(self, query: str, documents: List[str], relevance_scores: List[int]) -> dict:
        """
        Compute ranking metrics (NDCG, MAP, MRR) for ranking tasks.
        
        Args:
            query: Query string
            documents: List of documents
            relevance_scores: List of relevance scores (0-3, where 3=highly relevant)
            
        Returns:
            Dictionary of ranking metrics
        """
        if not self.is_ranking:
            return {}
        
        # Get predicted scores
        predicted_scores = []
        for doc in documents:
            score = self.score_query_doc_pair(query, doc)
            predicted_scores.append(score)
        
        # Sort by predicted scores
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance = [relevance_scores[i] for i in sorted_indices]
        
        metrics = {}
        
        # NDCG@5 and NDCG@10
        for k in [5, 10]:
            if k > len(sorted_relevance):
                k = len(sorted_relevance)
            
            # DCG calculation
            dcg = 0
            for i in range(k):
                rel = sorted_relevance[i]
                dcg += (2**rel - 1) / np.log2(i + 2)
            
            # IDCG calculation
            ideal_relevance = sorted(relevance_scores, reverse=True)[:k]
            idcg = 0
            for i in range(k):
                rel = ideal_relevance[i]
                idcg += (2**rel - 1) / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg_at_{k}'] = ndcg
        
        # MAP calculation
        relevant_docs = sum(1 for score in relevance_scores if score > 0)
        if relevant_docs > 0:
            precision_at_k = 0.0
            relevant_retrieved = 0
            
            for i, rel in enumerate(sorted_relevance):
                if rel > 0:  # relevant document
                    relevant_retrieved += 1
                    precision_at_k += relevant_retrieved / (i + 1)
            
            metrics['map'] = precision_at_k / relevant_docs
        else:
            metrics['map'] = 0.0
        
        # MRR calculation
        for i, rel in enumerate(sorted_relevance):
            if rel > 0:
                metrics['mrr'] = 1.0 / (i + 1)
                break
        else:
            metrics['mrr'] = 0.0
        
        return metrics
    
    def search_similar(self, text: str, documents: List[str], embeddings_path: str = None) -> List[Tuple[str, float]]:
        """
        Search for similar documents - task-aware.
        
        Args:
            text: Input text to find similar documents for
            documents: List of documents to search through
            embeddings_path: Path to saved document embeddings (optional)
            
        Returns:
            List of (document, similarity_score) tuples
        """
        with torch.no_grad():
            if self.is_ranking:
                # For ranking: treat as query and score each document
                scores = []
                for doc in documents:
                    score = self.score_query_doc_pair(text, doc)
                    scores.append(score)
                scores = torch.tensor(scores)
            else:
                # For retrieval: use embeddings and cosine similarity
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
                scores = F.cosine_similarity(text_vec, doc_embeddings, dim=1)
            
            # Get top 10 results
            top_indices = torch.argsort(scores, descending=True)[:10]
            
            results = []
            for idx in top_indices:
                doc = documents[idx.item()]
                score = scores[idx].item()
                results.append((doc, score))
            
            return results
    
    def print_query_results(self, query: str, results: List[Tuple[str, float, bool]]):
        """Print formatted query evaluation results - task-aware."""
        task_emoji = "🎯" if self.is_ranking else "🔍"
        task_name = "Ranking" if self.is_ranking else "Retrieval"
        
        print(f"\n{task_emoji} {task_name} Query: {query}")
        print("=" * 80)
        
        for i, (doc, score, is_correct) in enumerate(results, 1):
            status = "✅" if is_correct else "❌" if is_correct is False else "❓"
            score_label = "Relevance" if self.is_ranking else "Similarity"
            print(f"{i:2d}. {status} {score_label}: {score:.4f}")
            print(f"    {doc[:100]}...")
            print()
    
    def print_search_results(self, text: str, results: List[Tuple[str, float]]):
        """Print formatted search results - task-aware."""
        task_emoji = "🎯" if self.is_ranking else "🔎"
        score_label = "Relevance" if self.is_ranking else "Similarity"
        
        print(f"\n{task_emoji} Similar to: {text}")
        print("=" * 80)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i:2d}. {score_label}: {score:.4f}")
            print(f"    {doc[:100]}...")
            print()
    
    def print_ranking_metrics(self, query: str, metrics: dict):
        """Print ranking metrics (only for ranking tasks)."""
        if not self.is_ranking or not metrics:
            return
        
        print(f"\n📊 Ranking Metrics for: {query}")
        print("=" * 60)
        print(f"NDCG@5:  {metrics.get('ndcg_at_5', 0):.4f}")
        print(f"NDCG@10: {metrics.get('ndcg_at_10', 0):.4f}")
        print(f"MAP:     {metrics.get('map', 0):.4f}")
        print(f"MRR:     {metrics.get('mrr', 0):.4f}")
        print() 