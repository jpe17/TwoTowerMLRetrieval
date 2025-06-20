import torch
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class BatchEvaluator:
    """
    Fast batch-wise evaluation for training monitoring.
    Assumes 1:1 query-document mapping within batches.
    """
    
    def __init__(self, top_k: List[int] = [1, 5, 10]):
        self.top_k = top_k
    
    def evaluate(self, model, val_loader, device: torch.device, config: Dict):
        """Evaluates the model on validation batches, returning metrics and validation loss."""
        from .model import triplet_loss_cosine  # Import here to avoid circular imports
        
        model.eval()
        all_query_embs, all_doc_embs = [], []
        total_val_loss = 0
        
        print("\n🔬 Batch-wise evaluation: Generating embeddings and calculating validation loss...")
        with torch.no_grad():
            for queries, pos_docs, neg_docs in val_loader:
                queries, pos_docs, neg_docs = queries.to(device), pos_docs.to(device), neg_docs.to(device)
                
                query_emb = model.encode_query(queries)
                doc_emb = model.encode_document(pos_docs)
                neg_emb = model.encode_document(neg_docs)
                
                # Calculate validation loss for the batch
                loss = triplet_loss_cosine((query_emb, doc_emb, neg_emb), margin=config.get('MARGIN', 0.2))
                total_val_loss += loss.item()
                
                all_query_embs.append(query_emb)
                all_doc_embs.append(doc_emb)

        if not all_query_embs:
            print("Evaluation set is empty.")
            return {}, 0

        query_embs = torch.cat(all_query_embs)
        doc_embs = torch.cat(all_doc_embs)
        
        # Calculate similarity scores (batch-wise dot product)
        sim_scores = torch.matmul(query_embs, doc_embs.t())
        
        # Calculate metrics
        mrr_scores = []
        recall_at_k = {k: 0 for k in self.top_k}
        num_queries = sim_scores.size(0)

        for i in range(num_queries):
            # The positive document for query 'i' is at index 'i'
            scores = sim_scores[i]
            
            # Sort scores to get rank of positive doc
            _, sorted_indices = torch.sort(scores, descending=True)
            
            # Find rank of the positive document
            pos_doc_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            
            # MRR
            mrr_scores.append(1.0 / pos_doc_rank)
            
            # Recall@k
            for k in self.top_k:
                if pos_doc_rank <= k:
                    recall_at_k[k] += 1
        
        final_metrics = {f"Recall@{k}": count / num_queries for k, count in recall_at_k.items()}
        final_metrics["MRR"] = np.mean(mrr_scores)
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        return final_metrics, avg_val_loss


class CorpusEvaluator:
    """
    Full corpus evaluation that handles multiple positives per query.
    More realistic evaluation against a larger candidate pool.
    """
    
    def __init__(self, top_k: List[int] = [1, 5, 10], max_candidates: int = 1000, max_queries: int = 50):
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.max_queries = max_queries
    
    def evaluate(self, model, val_data: List[Tuple[str, str, str]], tokenizer, device: torch.device):
        """
        Evaluate using full corpus approach with multiple positives per query.
        
        Args:
            model: Two-tower model
            val_data: List of (query, pos_doc, neg_doc) triplets
            tokenizer: Tokenizer for encoding text
            device: PyTorch device
        
        Returns:
            Dict of evaluation metrics
        """
        model.eval()
        
        # 1. Group validation data by unique query
        query_to_positives = {}
        all_docs = set()
        
        for query, pos_doc, neg_doc in val_data:
            if query not in query_to_positives:
                query_to_positives[query] = set()
            query_to_positives[query].add(pos_doc)
            all_docs.add(pos_doc)
            all_docs.add(neg_doc)
        
        unique_queries = list(query_to_positives.keys())
        unique_docs = list(all_docs)
        
        # Limit candidates to prevent OOM but ensure good evaluation
        if len(unique_docs) > self.max_candidates:
            unique_docs = random.sample(unique_docs, self.max_candidates)
            print(f"  📊 Using {self.max_candidates} candidate documents for evaluation")
        
        print(f"\n🔬 Corpus evaluation: {len(unique_queries)} unique queries against {len(unique_docs)} documents...")
        
        # 2. Pre-compute all document embeddings
        doc_embeddings = self._compute_document_embeddings(model, unique_docs, tokenizer, device)
        
        # 3. Evaluate each query
        metrics = {f"Recall@{k}": [] for k in self.top_k}
        metrics.update({f"Hit@{k}": [] for k in self.top_k})  # At least 1 positive in top-k
        
        # Sample queries for faster evaluation
        sample_queries = random.sample(unique_queries, min(self.max_queries, len(unique_queries)))
        
        with torch.no_grad():
            for query in sample_queries:
                query_metrics = self._evaluate_single_query(
                    query, query_to_positives[query], unique_docs, 
                    doc_embeddings, model, tokenizer, device
                )
                
                # Accumulate metrics
                for metric_name, value in query_metrics.items():
                    if metric_name in metrics:
                        metrics[metric_name].append(value)
        
        # 4. Average the metrics
        final_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                final_metrics[metric_name] = np.mean(values)
            else:
                final_metrics[metric_name] = 0.0
        
        return final_metrics
    
    def _compute_document_embeddings(self, model, documents: List[str], tokenizer, device: torch.device):
        """Pre-compute embeddings for all documents."""
        doc_embeddings = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_tokens = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in batch_docs]
                padded_batch = pad_sequence(batch_tokens, batch_first=True, padding_value=0).to(device)
                embeddings = model.encode_document(padded_batch)
                doc_embeddings.append(embeddings)
        
        return torch.cat(doc_embeddings)
    
    def _evaluate_single_query(self, query: str, known_positives: set, unique_docs: List[str], 
                              doc_embeddings: torch.Tensor, model, tokenizer, device: torch.device):
        """Evaluate a single query against the document corpus."""
        # Get query embedding
        query_tokens = torch.tensor(tokenizer.encode(query), dtype=torch.long).unsqueeze(0).to(device)
        query_emb = model.encode_query(query_tokens)
        
        # Calculate similarities
        sim_scores = torch.matmul(query_emb, doc_embeddings.t()).squeeze(0)
        _, top_indices = torch.topk(sim_scores, k=max(self.top_k))
        
        # Get known positives that are in our candidate set
        available_positives = [doc for doc in known_positives if doc in unique_docs]
        
        if not available_positives:
            return {}  # Skip if no positives in candidate set
        
        # Calculate metrics for this query
        query_metrics = {}
        
        for k in self.top_k:
            top_k_docs = [unique_docs[idx] for idx in top_indices[:k]]
            found_positives = len([doc for doc in top_k_docs if doc in known_positives])
            
            # Recall@k: fraction of positives found
            recall_at_k = found_positives / len(available_positives)
            query_metrics[f"Recall@{k}"] = recall_at_k
            
            # Hit@k: at least 1 positive found
            hit_at_k = 1 if any(doc in known_positives for doc in top_k_docs) else 0
            query_metrics[f"Hit@{k}"] = hit_at_k
        
        return query_metrics


class TestEvaluator:
    """
    Qualitative evaluation with detailed output for testing and debugging.
    """
    
    def __init__(self, num_examples: int = 10, top_k: int = 5):
        self.num_examples = num_examples
        self.top_k = top_k
    
    def evaluate(self, model, test_data: List[Tuple[str, str, str]], tokenizer, device: torch.device):
        """Run qualitative evaluation on test set with detailed output."""
        model.eval()

        # 1. Collect all unique queries, documents, and ground truth
        all_queries = {triplet[0] for triplet in test_data}
        all_docs = {triplet[1] for triplet in test_data}.union({triplet[2] for triplet in test_data})
        
        ground_truth = {}
        for query, pos_doc, _ in test_data:
            if query not in ground_truth:
                ground_truth[query] = set()
            ground_truth[query].add(pos_doc)

        unique_queries = list(all_queries)
        unique_docs = list(all_docs)
        
        print(f"\n🧪 Test Evaluation: {len(unique_queries)} queries, {len(unique_docs)} documents...")

        # 2. Generate embeddings for all unique docs
        print("  Generating document embeddings...")
        doc_embs = []
        with torch.no_grad():
            for i in range(0, len(unique_docs), 64):
                batch_docs = unique_docs[i:i+64]
                batch_tokens = [torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in batch_docs]
                padded_batch = pad_sequence(batch_tokens, batch_first=True, padding_value=0).to(device)
                embeddings = model.encode_document(padded_batch)
                doc_embs.append(embeddings)
        doc_embs = torch.cat(doc_embs)

        # 3. Select sample queries and evaluate
        sample_queries = random.sample(unique_queries, min(self.num_examples, len(unique_queries)))
        
        print("\n" + "="*80)
        print(f"🔍 QUALITATIVE EXAMPLES (Top {self.top_k})")
        print("="*80)

        with torch.no_grad():
            for i, query in enumerate(sample_queries):
                print(f"\n--- Example {i+1}/{len(sample_queries)} ---")
                print(f"❓ Query: {query}")
                
                # Get query embedding
                query_tokens = torch.tensor(tokenizer.encode(query), dtype=torch.long).unsqueeze(0).to(device)
                query_emb = model.encode_query(query_tokens)

                # Compute similarities
                sim_scores = torch.matmul(query_emb, doc_embs.t()).squeeze(0)
                
                # Get top K results
                top_scores, top_indices = torch.topk(sim_scores, k=self.top_k)
                
                print(f"\n🎯 Top {self.top_k} Retrieved Documents:")
                retrieved_pos_count = 0
                for rank, doc_idx in enumerate(top_indices):
                    retrieved_doc = unique_docs[doc_idx.item()]
                    is_positive = retrieved_doc in ground_truth.get(query, set())
                    marker = "✅" if is_positive else "❌"
                    if is_positive:
                        retrieved_pos_count += 1
                    print(f"  {rank+1}. {marker} {retrieved_doc[:100]}... (Score: {top_scores[rank]:.4f})")

                actual_pos_docs = ground_truth.get(query, set())
                print(f"\nℹ️  Summary: Found {retrieved_pos_count}/{len(actual_pos_docs)} ground truth positives in Top {self.top_k}.") 