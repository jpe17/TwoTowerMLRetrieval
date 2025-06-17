import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# --- Inference Function ---
def search(query_text, documents, tokenizer, query_encoder, doc_encoder):
    with torch.no_grad():
        query_tensor = pad_sequence([torch.tensor(tokenizer.encode(query_text), dtype=torch.long)], batch_first=True)
        query_vec = query_encoder(query_tensor)

        doc_tensors = pad_sequence([torch.tensor(tokenizer.encode(doc), dtype=torch.long) for doc in documents], batch_first=True)
        doc_vecs = doc_encoder(doc_tensors)

        scores = F.cosine_similarity(query_vec, doc_vecs)
        top_indices = torch.argsort(scores, descending=True)
        return [(documents[i], scores[i].item()) for i in top_indices]
    

# --- Comprehensive Testing with Real Data ---
import random
from collections import defaultdict

def evaluate_retrieval(test_data, query_encoder, doc_encoder, tokenizer, k=10):
    """
    Evaluate retrieval performance using real test data
    """
    print("🔍 COMPREHENSIVE RETRIEVAL EVALUATION")
    print("="*50)
    
    # Group test data by query to get all relevant docs per query
    query_to_docs = defaultdict(list)
    for query, pos_doc, neg_doc in test_data[:100]:  # Sample 100 for speed
        query_to_docs[query].extend([pos_doc, neg_doc])
    
    # Test multiple queries
    sample_queries = list(query_to_docs.keys())[:5]  # Test 5 queries
    
    for i, query in enumerate(sample_queries):
        print(f"\n🔎 TEST QUERY {i+1}: {query[:100]}...")
        print("-" * 60)
        
        # Get all documents for this query
        documents = query_to_docs[query]
        
        # Add some random documents from other queries for harder test
        other_docs = []
        for other_query in random.sample(list(query_to_docs.keys()), 3):
            if other_query != query:
                other_docs.extend(query_to_docs[other_query][:2])
        
        all_documents = documents + other_docs
        random.shuffle(all_documents)
        
        print(f"📚 Searching through {len(all_documents)} documents...")
        
        # Run search
        results = search(query, all_documents, tokenizer, query_encoder, doc_encoder)
        
        print(f"\n🏆 TOP {min(3, len(results))} RESULTS:")
        for j, (doc, score) in enumerate(results[:3]):
            relevance = "✅ RELEVANT" if doc in documents else "❌ NOT RELEVANT"
            print(f"{j+1}. Score: {score:.4f} {relevance}")
            print(f"   Doc: {doc[:80]}...")
            print()
