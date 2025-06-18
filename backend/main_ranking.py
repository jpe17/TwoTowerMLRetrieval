#!/usr/bin/env python3
"""
Two-Tower ML Retrieval - Ranking Training Script

Trains the merged model for ranking task using pretrained retrieval weights.
"""

import sys
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
import wandb

load_dotenv()  # Loads .env file

# Only login, don't init here (trainer will handle init)
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader_ranking import RankingDataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model_merged import MergedTripletModel, triplet_loss
from trainer import TwoTowerTrainer  # Reuse existing trainer
from evaluator import SimpleEvaluator
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, print_model_summary, load_model_artifacts
)


def load_pretrained_retrieval_weights(merged_model, retrieval_artifacts_dir: str):
    """Load pretrained weights from two-tower retrieval model into merged model."""
    print(f"🔄 Loading pretrained weights from {retrieval_artifacts_dir}")
    
    try:
        # Load the two-tower model artifacts
        two_tower_model, _, training_info = load_model_artifacts(retrieval_artifacts_dir)
        
        # Copy query encoder weights to shared encoder
        merged_model.shared_encoder.load_state_dict(two_tower_model.query_encoder.state_dict())
        
        print("✅ Successfully loaded pretrained query encoder weights into shared encoder")
        print(f"   Original training epochs: {training_info['training_results'].get('final_avg_loss', 'N/A')}")
        
        return merged_model
        
    except Exception as e:
        print(f"⚠️  Could not load pretrained weights: {str(e)}")
        print("   Proceeding with random initialization...")
        return merged_model


def main():
    """Main ranking training function."""
    print("🎯 Two-Tower ML Ranking Training Pipeline")
    print("=" * 60)

    # Load and validate configuration
    print("📋 Loading configuration...")
    config = load_config('backend/config.json')
    config = validate_config(config)

    print(f"✅ Configuration loaded and validated")
    if config.get('SUBSAMPLE_RATIO'):
        print(f"   📊 Subsampling ratio: {config['SUBSAMPLE_RATIO']}")

    # Setup device and memory optimization
    setup_memory_optimization()
    device = get_best_device()

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])

    # Load pretrained embeddings
    print("\n🔢 Loading pretrained embeddings...")
    pretrained_embeddings = load_pretrained_embeddings(config['EMBEDDINGS_PATH'])

    # Add dimensions to config
    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]

    # Load ranking datasets
    print("\n🎯 Loading ranking datasets...")
    ranking_loader = RankingDataLoader(config)
    datasets = ranking_loader.load_ranking_datasets(subsample_ratio=config.get('SUBSAMPLE_RATIO'))
    dataset_stats = ranking_loader.get_dataset_stats(datasets)

    print(f"\n📊 Ranking Dataset Statistics:")
    for split, count in dataset_stats.items():
        print(f"   {split.capitalize()}: {count:,} samples")

    # Create data loaders (reuse existing factory)
    print("\n🔄 Creating data loaders...")
    dataloader_factory = DataLoaderFactory(config)
    dataloaders = dataloader_factory.create_dataloaders(datasets, tokenizer)

    # Create merged model
    print("\n🏗️  Creating merged model...")
    model = MergedTripletModel(
        vocab_size=config['VOCAB_SIZE'],
        embed_dim=config['EMBED_DIM'],
        hidden_dim=config.get('HIDDEN_DIM', 128),
        pretrained_embeddings=pretrained_embeddings,
        rnn_type=config.get('RNN_TYPE', 'GRU'),
        num_layers=config.get('NUM_LAYERS', 1),
        dropout=config.get('DROPOUT', 0.0)
    )

    # Load pretrained retrieval weights if available
    retrieval_artifacts_dir = config.get('PRETRAINED_RETRIEVAL_PATH')
    if retrieval_artifacts_dir and os.path.exists(retrieval_artifacts_dir):
        model = load_pretrained_retrieval_weights(model, retrieval_artifacts_dir)

    model = model.to(device)
    print_model_summary(model, config)

    # Create trainer (reuse existing trainer with adapter)
    print("\n🎯 Setting up trainer...")
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('LR', 0.001))
    loss_function = lambda triplet: triplet_loss(triplet, margin=config.get('MARGIN', 1.0))
    
    # Adapter to make merged model compatible with existing trainer
    class MergedModelAdapter:
        def __init__(self, merged_model):
            self.merged_model = merged_model
            
        def encode_query(self, query):
            return self.merged_model.encode_text(query)
            
        def encode_document(self, document):
            return self.merged_model.encode_text(document)
            
        def __getattr__(self, name):
            return getattr(self.merged_model, name)
    
    adapted_model = MergedModelAdapter(model)
    
    trainer = TwoTowerTrainer(
        model=adapted_model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        config=config
    )

    # Train the model
    print("\n🚀 Starting ranking training...")
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=config.get('EPOCHS', 5)  # Fewer epochs for fine-tuning
    )

    print(f"\n📈 Ranking Training Summary:")
    print(f"   Final Training Loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"   Final Validation Loss: {history['val_losses'][-1]:.4f}")
        print(f"   Best Validation Loss: {history['best_val_loss']:.4f}")

    # Demo the evaluator with test data
    if 'test' in datasets and datasets['test']:
        print("\n🔍 Testing Ranking Evaluator...")
        print("=" * 60)
        
        # Create evaluator (adapter for the merged model)
        class EvaluatorAdapter:
            def __init__(self, merged_model, tokenizer, device):
                self.model = merged_model
                self.tokenizer = tokenizer
                self.device = device
                self.model.eval()
            
            def evaluate_query(self, query, documents, positive_docs=None):
                # Use the encode_text method for both query and documents
                from evaluator import SimpleEvaluator
                import torch.nn.functional as F
                from torch.nn.utils.rnn import pad_sequence
                
                with torch.no_grad():
                    # Encode query
                    query_tokens = self.tokenizer.encode(query)
                    query_tensor = pad_sequence([torch.tensor(query_tokens, dtype=torch.long)], batch_first=True).to(self.device)
                    query_vec = self.model.encode_text(query_tensor)

                    # Encode documents
                    doc_tokens = [torch.tensor(self.tokenizer.encode(doc), dtype=torch.long) for doc in documents]
                    doc_tensors = pad_sequence(doc_tokens, batch_first=True).to(self.device)
                    doc_vecs = self.model.encode_text(doc_tensors)

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
            
            def print_query_results(self, query, results):
                print(f"\n🎯 Ranking Query: {query}")
                print("=" * 80)
                
                for i, (doc, score, is_correct) in enumerate(results, 1):
                    status = "✅" if is_correct else "❌" if is_correct is False else "❓"
                    print(f"{i:2d}. {status} Score: {score:.4f}")
                    print(f"    {doc[:100]}...")
                    print()
        
        evaluator = EvaluatorAdapter(model, tokenizer, device)
        
        # Get sample data from test set
        test_sample = datasets['test'][:50]  # Use first 50 test samples
        all_docs = []
        test_queries = []
        positive_docs = {}
        
        for query, pos_doc, neg_doc in test_sample:
            all_docs.extend([pos_doc, neg_doc])
            if query not in test_queries[:3]:  # Limit to 3 demo queries
                test_queries.append(query)
                positive_docs[query] = [pos_doc]
        
        # Remove duplicates
        unique_docs = list(dict.fromkeys(all_docs))
        
        # Test 2-3 sample queries
        for i, demo_query in enumerate(test_queries[:3], 1):
            print(f"\n📝 Demo Ranking Query {i}:")
            results = evaluator.evaluate_query(
                query=demo_query,
                documents=unique_docs,
                positive_docs=positive_docs[demo_query]
            )
            evaluator.print_query_results(demo_query, results)
        
        print("\n" + "=" * 60)

    print("\n🎉 Ranking training completed successfully!")


if __name__ == "__main__":
    main() 