import torch
import numpy as np
from evaluator import AdvancedEvaluator
from tokenizer import PretrainedTokenizer
from data_loader import DataLoader
import json
from tqdm import tqdm
import os
import pickle
from tqdm import tqdm

def aggregate_metrics(metrics_list):
    """Aggregate metrics from multiple batches by averaging values."""
    avg_metrics = {}
    for key in ['precision@k', 'recall@k', 'mrr', 'ndcg@k']:
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    return avg_metrics

def save_results(metrics, top_results, metrics_path="results/metrics.json", top_results_path="results/top_results.json"):
    """Save metrics and top results to JSON files."""
    os.makedirs("results", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(top_results_path, "w") as f:
        json.dump(top_results, f, indent=2)

def main():
    # Load config
    with open('backend/config.json', 'r') as f:
        config = json.load(f)

    device = torch.device(config.get('DEVICE', 'cpu'))
    print("device:", device)

    # Load tokenizer
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])

    # Load test dataset
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets()
    test_triplets = datasets['test']
    print("Loaded dataset length: ", len(test_triplets))

    # Load precomputed document embeddings and mapping
    doc_embeddings = np.load('data/test_document_embeddings.npy')[:50]
    with open('data/test_documents.txt', 'r', encoding='utf-8') as f:
        idx_to_doc = [line.strip() for line in f]

    # Load model
    model = torch.load('data/full_model.pth', map_location=device)
    model.eval()

    # Create evaluator
    evaluator = AdvancedEvaluator(model, tokenizer, device, top_k=5)

    # Evaluate using precomputed doc embeddings
    metrics, top_results = evaluator.evaluate_with_precomputed_doc_embeddings(
        test_triplets, doc_embeddings, idx_to_doc, top_k=5
    )

    # Print and save results
    evaluator.print_metrics(metrics)

    # Save detailed results for N queries
    N_EXAMPLES_TO_SAVE = 20 
    detailed_results = []
    for i, res in enumerate(top_results[:N_EXAMPLES_TO_SAVE]):
        detailed_results.append({
            "query": res['query'],
            "top_docs": [
                {
                    "doc": doc,
                    "score": float(score),
                    "is_positive": correct
                }
                for doc, score, correct in zip(res['top_docs'], res['top_scores'], res['is_correct'])
            ],
            "actual_positive_doc": next(
                (doc for doc, correct in zip(res['top_docs'], res['is_correct']) if correct), None
            )
        })

    os.makedirs("results", exist_ok=True)
    with open(f"results/detailed_top_results_{N_EXAMPLES_TO_SAVE}.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"Saved detailed top-{evaluator.top_k} results for {N_EXAMPLES_TO_SAVE} queries to results/detailed_top_results_{N_EXAMPLES_TO_SAVE}.json")

if __name__ == "__main__":
    main()
