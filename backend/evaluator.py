import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

from model import TwoTowerModel
from tokenizer import PretrainedTokenizer

import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import ndcg_score
from collections import defaultdict


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


class AdvancedEvaluator:
    def __init__(self, model: TwoTowerModel, tokenizer: PretrainedTokenizer, device: torch.device, wandb_logger=None, top_k=10):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.wandb_logger = wandb_logger
        self.top_k = top_k
        self.model.eval()

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Batch encode a list of texts into embeddings."""
        tokens = [torch.tensor(self.tokenizer.encode(text), dtype=torch.long) for text in texts]
        padded = pad_sequence(tokens, batch_first=True).to(self.device)
        return self.model.encode_query(padded) if hasattr(self.model, 'encode_query') else self.model(padded)

    def evaluate_batch(self, queries: List[str], documents: List[str], positive_docs_dict: Dict[str, List[str]],
                      embeddings_path: Optional[str] = None) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate a batch of queries against documents and return metrics and top results.

        Args:
            queries: List of query strings
            documents: List of document strings
            positive_docs_dict: Dict mapping query to list of correct document strings
            embeddings_path: Optional path to precomputed document embeddings

        Returns:
            metrics: Dict of evaluation metrics
            top_results: List of dicts with top results for each query
        """
        # Encode all queries
        query_vecs = self.encode_batch(queries)

        # Load or compute document embeddings
        if embeddings_path and os.path.exists(embeddings_path):
            doc_embeddings = torch.tensor(np.load(embeddings_path), device=self.device)
        else:
            doc_embeddings = self.encode_batch(documents)

        # Compute cosine similarity: shape (num_queries, num_docs)
        scores = F.cosine_similarity(query_vecs.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2)

        # For each query, get top_k results and compute metrics
        metrics = defaultdict(list)
        top_results = []

        for i, (query, query_scores) in enumerate(zip(queries, scores)):
            # Get top_k indices and scores
            top_scores, top_indices = torch.topk(query_scores, k=self.top_k, dim=0)
            top_docs = [documents[idx.item()] for idx in top_indices]
            top_scores = top_scores.tolist()

            # Ground truth: 1 if doc is correct, 0 otherwise
            relevant_set = set(positive_docs_dict.get(query, []))
            ground_truth = [1 if doc in relevant_set else 0 for doc in documents]
            pred_scores = query_scores.detach().cpu().numpy()
            pred_rank = (-pred_scores).argsort()  # argsort in descending order

            # Compute metrics
            # Precision@k
            correct_in_top = sum(1 for doc in top_docs if doc in relevant_set)
            precision_at_k = correct_in_top / self.top_k
            metrics['precision@k'].append(precision_at_k)

            # Recall@k
            total_relevant = len(relevant_set)
            recall_at_k = correct_in_top / max(total_relevant, 1)
            metrics['recall@k'].append(recall_at_k)

            # MRR
            for rank, idx in enumerate(pred_rank, 1):
                if ground_truth[idx]:
                    mrr = 1.0 / rank
                    metrics['mrr'].append(mrr)
                    break

            # NDCG (sklearn expects relevance scores as floats)
            # Here, ground_truth is binary
            ndcg = ndcg_score([ground_truth], [pred_scores], k=self.top_k)
            metrics['ndcg@k'].append(ndcg)

            # Store top results for this query
            top_results.append({
                'query': query,
                'top_docs': top_docs,
                'top_scores': top_scores,
                'is_correct': [doc in relevant_set for doc in top_docs]
            })

        # Aggregate metrics
        agg_metrics = {
            'precision@k': np.mean(metrics['precision@k']),
            'recall@k': np.mean(metrics['recall@k']),
            'mrr': np.mean(metrics['mrr']) if metrics['mrr'] else 0.0,
            'ndcg@k': np.mean(metrics['ndcg@k'])
        }

        # Log to wandb if logger is provided
        if self.wandb_logger:
            self.wandb_logger.log(agg_metrics)

        return agg_metrics, top_results

    def analyze_errors(self, results: List[Dict], positive_docs_dict: Dict[str, List[str]]) -> Dict:
        """
        Analyze errors in evaluation results.

        Args:
            results: List of dicts from evaluate_batch
            positive_docs_dict: Dict mapping query to list of correct document strings

        Returns:
            error_stats: Dict with error analysis statistics
        """
        error_stats = {
            'queries_with_no_correct_in_top': 0,
            'queries_with_all_correct_missed': 0,
            'total_queries': len(results)
        }

        for res in results:
            query = res['query']
            relevant_set = set(positive_docs_dict.get(query, []))
            top_docs = res['top_docs']
            is_correct = res['is_correct']

            # Queries with no correct in top_k
            if not any(is_correct):
                error_stats['queries_with_no_correct_in_top'] += 1

            # Queries where all correct were missed (if any correct exists)
            if relevant_set and not any(doc in relevant_set for doc in top_docs):
                error_stats['queries_with_all_correct_missed'] += 1

        return error_stats

    def print_metrics(self, metrics: Dict):
        print("\n📊 Evaluation Metrics:")
        print("-" * 40)
        df = pd.DataFrame([metrics])
        print(df.to_string(index=False))


    def evaluate_triplets(self, triplets: List[Tuple[str, str, str]], embeddings_path: Optional[str] = None):
        """
        Evaluate a batch of triplets (query, positive_doc, negative_doc).
        This method automatically constructs the required query/document lists and positive_docs_dict.
        
        Args:
            triplets: List of (query, positive_doc, negative_doc) tuples
            embeddings_path: Optional path to precomputed document embeddings
        
        Returns:
            metrics: Dict of evaluation metrics
            top_results: List of dicts with top results for each query

        Usage:
        evaluator = AdvancedEvaluator(model, tokenizer, device)
        metrics, top_results = evaluator.evaluate_triplets(triplets)
        """
        from collections import defaultdict

        queries = set()
        documents = set()
        positive_docs_dict = defaultdict(list)

        for query, pos_doc, neg_doc in triplets:
            queries.add(query)
            documents.add(pos_doc)
            documents.add(neg_doc)
            positive_docs_dict[query].append(pos_doc)

        queries = list(queries)
        documents = list(documents)
        # Remove duplicate positives for each query
        positive_docs_dict = {q: list(set(docs)) for q, docs in positive_docs_dict.items()}

        return self.evaluate_batch(queries, documents, positive_docs_dict, embeddings_path)

    def evaluate_batch_per_query(self, queries: List[str], documents: List[str], positive_docs_dict: Dict[str, List[str]],
                            embeddings_path: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        # Encode all queries
        query_vecs = self.encode_batch(queries)

        # Load or compute document embeddings
        if embeddings_path and os.path.exists(embeddings_path):
            doc_embeddings = torch.tensor(np.load(embeddings_path), device=self.device)
        else:
            doc_embeddings = self.encode_batch(documents)

        # Compute cosine similarity: shape (num_queries, num_docs)
        scores = F.cosine_similarity(query_vecs.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2)

        query_metrics = []
        top_results = []

        for i, (query, query_scores) in enumerate(zip(queries, scores)):
            # Get top_k indices and scores
            top_scores, top_indices = torch.topk(query_scores, k=self.top_k, dim=0)
            top_docs = [documents[idx.item()] for idx in top_indices]
            top_scores = top_scores.tolist()

            # Ground truth: 1 if doc is correct, 0 otherwise
            relevant_set = set(positive_docs_dict.get(query, []))
            ground_truth = [1 if doc in relevant_set else 0 for doc in documents]
            pred_scores = query_scores.detach().cpu().numpy()
            pred_rank = (-pred_scores).argsort()  # argsort in descending order

            # Compute metrics
            correct_in_top = sum(1 for doc in top_docs if doc in relevant_set)
            precision_at_k = correct_in_top / self.top_k

            total_relevant = len(relevant_set)
            recall_at_k = correct_in_top / max(total_relevant, 1)

            mrr = 0.0
            for rank, idx in enumerate(pred_rank, 1):
                if ground_truth[idx]:
                    mrr = 1.0 / rank
                    break

            ndcg = ndcg_score([ground_truth], [pred_scores], k=self.top_k)

            query_metrics.append({
                'precision@k': precision_at_k,
                'recall@k': recall_at_k,
                'mrr': mrr,
                'ndcg@k': ndcg,
                'query': query
            })

            top_results.append({
                'query': query,
                'top_docs': top_docs,
                'top_scores': top_scores,
                'is_correct': [doc in relevant_set for doc in top_docs]
            })

        return query_metrics, top_results

    def evaluate_triplets_per_query(self, triplets: List[Tuple[str, str, str]], embeddings_path: Optional[str] = None):
        from collections import defaultdict

        queries = set()
        documents = set()
        positive_docs_dict = defaultdict(list)

        for query, pos_doc, neg_doc in triplets:
            queries.add(query)
            documents.add(pos_doc)
            documents.add(neg_doc)
            positive_docs_dict[query].append(pos_doc)

        #sample_query = list(queries)[0]
        #print(f"Sample query: {sample_query}")
        #print(f"Positive docs: {positive_docs_dict[sample_query]}")
        #print(f"Are positives in document pool? {all(p in documents for p in positive_docs_dict[sample_query])}")

        queries = list(queries)
        documents = list(documents)
        # Remove duplicate positives for each query
        positive_docs_dict = {q: list(set(docs)) for q, docs in positive_docs_dict.items()}

        return self.evaluate_batch_per_query(queries, documents, positive_docs_dict, embeddings_path)

