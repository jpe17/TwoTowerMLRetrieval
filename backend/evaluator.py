import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

from .model import TwoTowerModel
from .tokenizer import PretrainedTokenizer
from .utils import clean_memory


class TwoTowerEvaluator:
    """Evaluator for the Two-Tower model with search and retrieval capabilities."""
    
    def __init__(
        self,
        model: TwoTowerModel,
        tokenizer: PretrainedTokenizer,
        device: torch.device
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained two-tower model
            tokenizer: Tokenizer for text processing
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # Set to evaluation mode
    
    def search(
        self, 
        query_text: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for relevant documents given a query.
        
        Args:
            query_text: Query string
            documents: List of document strings to search through
            top_k: Number of top results to return
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        with torch.no_grad():
            # Encode query
            query_tokens = self.tokenizer.encode(query_text)
            if not query_tokens:  # Handle empty tokenization
                return []
            
            query_tensor = pad_sequence(
                [torch.tensor(query_tokens, dtype=torch.long)], 
                batch_first=True
            ).to(self.device)
            query_vec = self.model.encode_query(query_tensor)

            # Encode documents
            doc_tokens_list = []
            valid_doc_indices = []
            
            for i, doc in enumerate(documents):
                doc_tokens = self.tokenizer.encode(doc)
                if doc_tokens:  # Only include documents with valid tokens
                    doc_tokens_list.append(torch.tensor(doc_tokens, dtype=torch.long))
                    valid_doc_indices.append(i)
            
            if not doc_tokens_list:  # Handle case where no documents have valid tokens
                return []
            
            doc_tensors = pad_sequence(doc_tokens_list, batch_first=True).to(self.device)
            doc_vecs = self.model.encode_document(doc_tensors)

            # Calculate similarity scores (cosine similarity)
            scores = F.cosine_similarity(query_vec, doc_vecs, dim=1)
            
            # Get top-k results
            top_k = min(top_k, len(scores))
            top_indices = torch.argsort(scores, descending=True)[:top_k]
            
            # Convert results back to original document indices and scores
            results = []
            for idx in top_indices:
                original_doc_idx = valid_doc_indices[idx.item()]
                score = scores[idx].item()
                results.append((documents[original_doc_idx], score))
            
            # Clear cache after inference
            if self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                
            return results
    
    def batch_search(
        self,
        queries: List[str],
        documents: List[str],
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            top_k: Number of top results per query
            batch_size: Batch size for processing
            
        Returns:
            List of search results for each query
        """
        results = []
        
        # Process queries in batches
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_results = []
            
            for query in batch_queries:
                query_results = self.search(query, documents, top_k)
                batch_results.append(query_results)
            
            results.extend(batch_results)
            
            # Periodic memory cleanup
            if i % (batch_size * 10) == 0:
                clean_memory()
        
        return results
    
    def evaluate_retrieval(
        self,
        test_data: List[Tuple[str, str, str]],
        num_samples: int = 100,
        num_distractors: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance using test data.
        
        Args:
            test_data: List of (query, positive_doc, negative_doc) triplets
            num_samples: Number of samples to evaluate
            num_distractors: Number of distractor documents per query
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"🔍 Evaluating retrieval performance on {num_samples} samples...")
        
        # Group test data by query to get all relevant docs per query
        query_to_docs = defaultdict(list)
        for query, pos_doc, neg_doc in test_data:
            query_to_docs[query].extend([pos_doc, neg_doc])
        
        # Sample queries for evaluation
        sample_queries = list(query_to_docs.keys())[:num_samples]
        
        metrics = {
            'precision_at_1': 0.0,
            'precision_at_3': 0.0,
            'precision_at_5': 0.0,
            'mrr': 0.0,  # Mean Reciprocal Rank
            'num_queries': len(sample_queries)
        }
        
        total_p1, total_p3, total_p5, total_mrr = 0, 0, 0, 0
        
        for i, query in enumerate(sample_queries):
            if i % 20 == 0:
                print(f"  Processing query {i+1}/{len(sample_queries)}")
            
            # Get relevant documents for this query
            relevant_docs = set(query_to_docs[query])
            
            # Add distractor documents from other queries
            all_other_docs = []
            for other_query in random.sample(
                [q for q in query_to_docs.keys() if q != query], 
                min(num_distractors, len(query_to_docs) - 1)
            ):
                all_other_docs.extend(query_to_docs[other_query])
            
            # Create candidate document set
            candidate_docs = list(relevant_docs) + random.sample(
                all_other_docs, 
                min(num_distractors * 2, len(all_other_docs))
            )
            random.shuffle(candidate_docs)
            
            # Search
            results = self.search(query, candidate_docs, top_k=5)
            
            # Calculate metrics
            retrieved_docs = [doc for doc, _ in results]
            
            # Precision at K
            p1 = 1.0 if len(retrieved_docs) > 0 and retrieved_docs[0] in relevant_docs else 0.0
            p3 = sum(1 for doc in retrieved_docs[:3] if doc in relevant_docs) / min(3, len(retrieved_docs))
            p5 = sum(1 for doc in retrieved_docs[:5] if doc in relevant_docs) / min(5, len(retrieved_docs))
            
            # Mean Reciprocal Rank
            mrr = 0.0
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc in relevant_docs:
                    mrr = 1.0 / rank
                    break
            
            total_p1 += p1
            total_p3 += p3
            total_p5 += p5
            total_mrr += mrr
        
        # Average metrics
        metrics['precision_at_1'] = total_p1 / len(sample_queries)
        metrics['precision_at_3'] = total_p3 / len(sample_queries)
        metrics['precision_at_5'] = total_p5 / len(sample_queries)
        metrics['mrr'] = total_mrr / len(sample_queries)
        
        print(f"📊 Retrieval Evaluation Results:")
        print(f"   Precision@1: {metrics['precision_at_1']:.3f}")
        print(f"   Precision@3: {metrics['precision_at_3']:.3f}")
        print(f"   Precision@5: {metrics['precision_at_5']:.3f}")
        print(f"   Mean Reciprocal Rank: {metrics['mrr']:.3f}")
        
        return metrics
    
    def interactive_search_demo(
        self,
        documents: List[str],
        num_demos: int = 5
    ):
        """
        Interactive search demonstration.
        
        Args:
            documents: List of documents to search through
            num_demos: Number of demo queries to run
        """
        print(f"🎯 Interactive Search Demo")
        print("=" * 50)
        
        # Sample some documents for demo queries
        sample_docs = random.sample(documents, min(100, len(documents)))
        
        demo_queries = [
            "What is machine learning?",
            "How to train neural networks?",
            "What is natural language processing?",
            "Explain deep learning concepts",
            "What are the applications of AI?"
        ]
        
        for i, query in enumerate(demo_queries[:num_demos]):
            print(f"\n🔎 Demo Query {i+1}: {query}")
            print("-" * 60)
            
            results = self.search(query, sample_docs, top_k=3)
            
            if results:
                print(f"🏆 Top {len(results)} Results:")
                for j, (doc, score) in enumerate(results):
                    print(f"{j+1}. Score: {score:.4f}")
                    print(f"   Document: {doc[:100]}...")
                    print()
            else:
                print("❌ No results found.")
    
    def similarity_analysis(
        self,
        text_pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, float]]:
        """
        Analyze similarity between text pairs.
        
        Args:
            text_pairs: List of (text1, text2) pairs
            
        Returns:
            List of (text1, text2, similarity_score) tuples
        """
        results = []
        
        with torch.no_grad():
            for text1, text2 in text_pairs:
                # Encode both texts
                tokens1 = self.tokenizer.encode(text1)
                tokens2 = self.tokenizer.encode(text2)
                
                if not tokens1 or not tokens2:
                    results.append((text1, text2, 0.0))
                    continue
                
                tensor1 = pad_sequence(
                    [torch.tensor(tokens1, dtype=torch.long)], 
                    batch_first=True
                ).to(self.device)
                
                tensor2 = pad_sequence(
                    [torch.tensor(tokens2, dtype=torch.long)], 
                    batch_first=True
                ).to(self.device)
                
                # Use query encoder for both (or you could use different encoders)
                vec1 = self.model.encode_query(tensor1)
                vec2 = self.model.encode_query(tensor2)
                
                # Calculate cosine similarity
                similarity = F.cosine_similarity(vec1, vec2, dim=1).item()
                results.append((text1, text2, similarity))
        
        return results
    
    def get_document_embeddings(
        self,
        documents: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Get embeddings for a list of documents.
        
        Args:
            documents: List of document strings
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of document embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Tokenize batch
                batch_tokens = []
                for doc in batch_docs:
                    tokens = self.tokenizer.encode(doc)
                    if tokens:
                        batch_tokens.append(torch.tensor(tokens, dtype=torch.long))
                    else:
                        # Handle empty tokenization with a dummy token
                        batch_tokens.append(torch.tensor([1], dtype=torch.long))
                
                # Pad and encode
                batch_tensors = pad_sequence(batch_tokens, batch_first=True).to(self.device)
                batch_embeddings = self.model.encode_document(batch_tensors)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                
                # Periodic cleanup
                if i % (batch_size * 10) == 0:
                    clean_memory()
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def save_embeddings(
        self,
        documents: List[str],
        save_path: str,
        batch_size: int = 32
    ):
        """
        Save document embeddings to file.
        
        Args:
            documents: List of documents
            save_path: Path to save embeddings
            batch_size: Batch size for processing
        """
        print(f"💾 Computing and saving embeddings for {len(documents)} documents...")
        
        embeddings = self.get_document_embeddings(documents, batch_size)
        np.save(save_path, embeddings)
        
        print(f"✅ Saved embeddings to {save_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB") 