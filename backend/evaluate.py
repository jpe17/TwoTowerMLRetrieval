#!/usr/bin/env python3
"""
Simplified Two-Tower ML Evaluator

This script evaluates a trained model's performance using pre-computed embeddings
saved during a training run. It is much faster than on-the-fly evaluation.
"""

import torch
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from data_loader import DataLoader
from utils import get_best_device


def load_embeddings_and_maps(artifacts_path, device):
    """Loads all necessary embeddings and mappings from an artifacts directory."""
    print(f"📂 Loading pre-computed embeddings from {artifacts_path}...")
    
    try:
        # Load documents
        doc_embeddings = torch.from_numpy(np.load(f"{artifacts_path}/document_embeddings.npy")).to(device)
        with open(f"{artifacts_path}/doc_to_idx.pkl", 'rb') as f:
            doc_to_idx = pickle.load(f)
        
        # Load queries
        query_embeddings = torch.from_numpy(np.load(f"{artifacts_path}/query_embeddings.npy")).to(device)
        with open(f"{artifacts_path}/query_to_idx.pkl", 'rb') as f:
            query_to_idx = pickle.load(f)
            
        # Create reverse mapping to get document text from its index
        idx_to_doc = {idx: doc for doc, idx in doc_to_idx.items()}

        print(f"  ✅ Loaded {len(doc_to_idx):,} document embeddings.")
        print(f"  ✅ Loaded {len(query_to_idx):,} query embeddings.")
        
        return doc_embeddings, doc_to_idx, idx_to_doc, query_embeddings, query_to_idx
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}.")
        print(f"   Ensure the path '{artifacts_path}' is correct and contains the required .npy and .pkl files.")
        print("   💡 Did you run training with the 'save_doc_embeddings=True' flag?")
        return None, None, None, None, None


def evaluate_with_saved_embeddings():
    """
    Evaluate model performance using pre-computed embeddings. This is much faster as
    it doesn't require model loading or on-the-fly embedding generation.
    """
    # --- CONFIGURATION ---
    # ❗ IMPORTANT: Change this path to point to your specific training run folder.
    ARTIFACTS_PATH = "artifacts/two_tower_run_20250619_140401"
    TEST_DATA_PATH = "data/ms_marco_test.parquet"
    TOP_K = 10
    NUM_EXAMPLES_TO_SHOW = 10
    # ---------------------
    
    device = get_best_device()
    
    # 1. Load pre-computed embeddings and mappings
    doc_embeddings, doc_to_idx, idx_to_doc, query_embeddings, query_to_idx = load_embeddings_and_maps(ARTIFACTS_PATH, device)
    if doc_embeddings is None:
        # The loading function already prints detailed errors
        return
        
    # 2. Load test data to get the ground truth for evaluation
    print(f"\n📚 Loading test data from {TEST_DATA_PATH} for ground truth...")
    data_loader = DataLoader({'TEST_DATASET_PATH': TEST_DATA_PATH, 'TASK_MODE': 'retrieval'})
    test_data = data_loader.load_and_process_parquet(TEST_DATA_PATH)
    if not test_data:
        print("❌ No test data found!")
        return
    print(f"  ✅ Loaded {len(test_data):,} test triplets.")

    # 3. Evaluate retrieval performance and show examples
    print(f"\n📊 EVALUATING RETRIEVAL (Top {TOP_K})")
    print("=" * 80)
    
    mrr_scores = []
    hit_at_k = 0
    total_queries_in_test_set = len(test_data)
    queries_evaluated = 0

    for i, (query, pos_doc, neg_doc) in enumerate(test_data):
        # We can only evaluate queries and documents that were present during training
        # and therefore have saved embeddings.
        if query not in query_to_idx or pos_doc not in doc_to_idx:
            continue
        
        queries_evaluated += 1
        query_idx = query_to_idx[query]
        query_emb = query_embeddings[query_idx]
        pos_doc_idx = doc_to_idx[pos_doc]
        
        # Compute similarities between the query and all documents
        similarities = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), doc_embeddings)
        
        # Get top K results
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(TOP_K, len(similarities)))
        top_k_indices = top_k_indices.cpu().numpy().tolist()
        
        # Check for hit and calculate its rank for MRR
        try:
            rank = top_k_indices.index(pos_doc_idx) + 1
            mrr_scores.append(1.0 / rank)
            hit_at_k += 1
        except ValueError:
            # The correct document was not in the top K
            mrr_scores.append(0.0)

        # For the first few queries, show the retrieval results
        if queries_evaluated <= NUM_EXAMPLES_TO_SHOW:
            print(f"\n🔍 EXAMPLE {queries_evaluated}/{NUM_EXAMPLES_TO_SHOW}")
            print("-" * 60)
            print(f"❓ Query: {query[:100]}...")
            print(f"✅ Expected Positive Doc: {pos_doc[:100]}...")
            print(f"\n🎯 Top {TOP_K} Retrieved Documents:")
            
            for j, (doc_idx, score) in enumerate(zip(top_k_indices, top_k_scores.cpu().numpy())):
                retrieved_doc = idx_to_doc[doc_idx]
                is_positive = (doc_idx == pos_doc_idx)
                marker = "✅" if is_positive else "  "
                print(f"  {marker} {j+1}. {retrieved_doc[:80]}... (sim: {score:.4f})")
    
    # 4. Calculate and print the final metrics
    if queries_evaluated > 0:
        mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        hit_rate = hit_at_k / queries_evaluated
        
        print("\n" + "="*80)
        print(f"📈 FINAL RETRIEVAL METRICS:")
        print(f"  Queries Evaluated: {queries_evaluated:,} (out of {total_queries_in_test_set:,} in test file)")
        print(f"   (Evaluation is only on queries/docs from the train/val sets)")
        print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"  Hit@{TOP_K}: {hit_rate:.4f} ({hit_at_k:,}/{queries_evaluated:,})")
    else:
        print("\n⚠️ No queries from the test set were found in the saved embeddings.")


if __name__ == "__main__":
    evaluate_with_saved_embeddings() 