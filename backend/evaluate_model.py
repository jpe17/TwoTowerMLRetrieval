import torch
import numpy as np
from evaluator import AdvancedEvaluator
from tokenizer import PretrainedTokenizer
from data_loader import DataLoader
import json
from tqdm import tqdm
from model import ModelFactory
import os

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

    # Load pretrained embeddings
    pretrained_embeddings = np.load(config['EMBEDDINGS_PATH'])

    # Build model architecture
    model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
    model = torch.load('data/full_model.pth', map_location=device)
    model.to(device)
    model.eval()

    # Create evaluator
    evaluator = AdvancedEvaluator(model, tokenizer, device, top_k=5)

    # Batch evaluation with tqdm progress bar
    batch_size = 32 
    num_samples = len(test_triplets)
    all_query_metrics = []
    all_top_results = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating batches"):
        batch = test_triplets[i:i+batch_size]
        query_metrics, top_results = evaluator.evaluate_triplets_per_query(batch)
        all_query_metrics.extend(query_metrics)
        all_top_results.extend(top_results)

    # Aggregate metrics over all queries
    final_metrics = aggregate_metrics(all_query_metrics)
    
    # Print a few sample query metrics and top results
    print("\n📋 Example Query Metrics:")
    print("-" * 60)
    for i, m in enumerate(all_query_metrics[5:10], 1):
        print(f"Query {i}:")
        print(f"  precision@{evaluator.top_k}: {m['precision@k']:.4f}")
        print(f"  recall@{evaluator.top_k}:    {m['recall@k']:.4f}")
        print(f"  mrr:                        {m['mrr']:.4f}")
        print(f"  ndcg@{evaluator.top_k}:     {m['ndcg@k']:.4f}")
        print()

    print("\n Example Top Results (First 5 Queries):")
    print("-" * 60)
    for i, res in enumerate(all_top_results[5:10], 1):
        print(f"Query {i}: {res['query']}")
        print("Top docs:")
        for j, (doc, score, correct) in enumerate(zip(res['top_docs'], res['top_scores'], res['is_correct']), 1):
            print(f"  {j}. {'✅' if correct else '❌'} {doc[:50]}... (score: {score:.4f})")
        print()
    
    print("\nFinal aggregated metrics (over all queries):")
    evaluator.print_metrics(final_metrics)

    # Save results
    save_results(final_metrics, all_top_results)

if __name__ == "__main__":
    main()
