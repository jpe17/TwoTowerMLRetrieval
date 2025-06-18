#!/usr/bin/env python3
"""
Two-Tower ML Retrieval - Unified Training Script

Handles both retrieval and ranking tasks based on configuration.
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

from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import ModelFactory, TwoTowerModel
from model_combined import MergedTripletModel, triplet_loss
from trainer import TrainerFactory, TwoTowerTrainer
from evaluator import SimpleEvaluator
from utils import (
    load_config, validate_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings, print_model_summary, load_model_artifacts,
    save_model_artifacts
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


class MergedModelAdapter:
    """Adapter to make merged model compatible with existing trainer."""
    def __init__(self, merged_model):
        self.merged_model = merged_model
        
    def encode_query(self, query):
        return self.merged_model.encode_text(query)
        
    def encode_document(self, document):
        return self.merged_model.encode_text(document)
        
    def __getattr__(self, name):
        return getattr(self.merged_model, name)





def create_model_and_trainer(config, pretrained_embeddings, device):
    """Create model and trainer based on task mode."""
    task_mode = config.get('TASK_MODE', 'retrieval')
    
    if task_mode == 'ranking':
        print("\n🏗️  Creating merged model for ranking...")
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
        
        # Create trainer with adapter
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('LR', 0.001))
        loss_function = lambda triplet: triplet_loss(triplet, margin=config.get('MARGIN', 1.0))
        
        adapted_model = MergedModelAdapter(model)
        trainer = TwoTowerTrainer(
            model=adapted_model,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            config=config
        )
        
        return model, trainer
    
    else:  # retrieval mode
        print("\n🏗️  Creating two-tower model for retrieval...")
        model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
        model = model.to(device)
        
        # Create trainer using factory
        trainer = TrainerFactory.create_trainer(config, model, device)
        
        return model, trainer


def demo_evaluator(model, trainer, tokenizer, device, datasets, task_mode, config):
    """Demo the evaluator with test data."""
    if 'test' not in datasets or not datasets['test']:
        return
    
    task_emoji = "🎯" if task_mode == 'ranking' else "🔍"
    task_name = "Ranking" if task_mode == 'ranking' else "Retrieval"
    
    print(f"\n{task_emoji} Testing {task_name} Evaluator...")
    print("=" * 60)
    
    # Create task-aware evaluator (automatically detects task mode from config)
    if task_mode == 'ranking':
        evaluator = SimpleEvaluator(model, tokenizer, device, config)
    else:
        evaluator = SimpleEvaluator(trainer.model, tokenizer, device, config)
    
    # Get sample data from test set - create proper query-specific document sets
    test_sample = datasets['test'][:20]  # Use fewer samples for cleaner demo
    
    # Process each demo query individually to ensure correct positive docs are included
    demo_queries_data = []
    seen_queries = set()
    
    for query, pos_doc, neg_doc in test_sample:
        if query not in seen_queries and len(demo_queries_data) < 3:
            seen_queries.add(query)
            
            # Create a focused document set for this query
            query_docs = [pos_doc]  # Start with the correct positive document
            
            # Add some other documents from the test set as distractors
            for other_query, other_pos, other_neg in test_sample[:15]:
                if other_query != query:  # Don't add docs from same query
                    query_docs.extend([other_pos, other_neg])
            
            # Remove duplicates while preserving order
            unique_query_docs = []
            seen = set()
            for doc in query_docs:
                if doc not in seen:
                    unique_query_docs.append(doc)
                    seen.add(doc)
            
            demo_queries_data.append({
                'query': query,
                'documents': unique_query_docs[:20],  # Limit to 20 docs for manageable demo
                'positive_docs': [pos_doc]
            })
    
    # Test the demo queries
    for i, query_data in enumerate(demo_queries_data, 1):
        print(f"\n📝 Demo {task_name} Query {i}:")
        results = evaluator.evaluate_query(
            query=query_data['query'],
            documents=query_data['documents'],
            positive_docs=query_data['positive_docs']
        )
        evaluator.print_query_results(query_data['query'], results)
    
    print("\n" + "=" * 60)


def main():
    """Unified main training function."""
    # Load and validate configuration
    print("📋 Loading configuration...")
    config = load_config('backend/config.json')
    config = validate_config(config)
    
    task_mode = config.get('TASK_MODE', 'retrieval')
    task_emoji = "🎯" if task_mode == 'ranking' else "🚀"
    task_name = "Ranking" if task_mode == 'ranking' else "Retrieval"
    
    print(f"{task_emoji} Two-Tower ML {task_name} Training Pipeline")
    print("=" * 60)
    print(f"🎯 Task Mode: {task_mode.upper()}")

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

    # Load datasets using unified DataLoader
    print(f"\n📚 Loading {task_name.lower()} datasets...")
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets(subsample_ratio=config.get('SUBSAMPLE_RATIO'))
    dataset_stats = data_loader.get_dataset_stats(datasets)

    print(f"\n📊 {task_name} Dataset Statistics:")
    for split, count in dataset_stats.items():
        print(f"   {split.capitalize()}: {count:,} samples")

    # Create data loaders
    print("\n🔄 Creating data loaders...")
    dataloader_factory = DataLoaderFactory(config)
    dataloaders = dataloader_factory.create_dataloaders(datasets, tokenizer)

    # Create model and trainer
    model, trainer = create_model_and_trainer(config, pretrained_embeddings, device)
    print_model_summary(model, config)

    # Train the model
    epochs = config.get('EPOCHS', 5 if task_mode == 'ranking' else 10)
    print(f"\n🚀 Starting {task_name.lower()} training...")
    history = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('validation'),
        epochs=epochs
    )

    print(f"\n📈 {task_name} Training Summary:")
    print(f"   Final Training Loss: {history['train_losses'][-1]:.4f}")
    if history['val_losses']:
        print(f"   Final Validation Loss: {history['val_losses'][-1]:.4f}")
        print(f"   Best Validation Loss: {history['best_val_loss']:.4f}")

    # Save the trained model
    print(f"\n💾 Saving {task_name.lower()} model...")
    
    # Handle different model types for saving
    if task_mode == 'ranking':
        # For ranking mode, we need to save the underlying merged model
        actual_model = model  # The MergedTripletModel
        optimizer = trainer.optimizer
    else:
        # For retrieval mode, save the two-tower model
        actual_model = model  # The TwoTowerModel
        optimizer = trainer.optimizer
    
    artifacts_dir = save_model_artifacts(
        model=actual_model,
        optimizer=optimizer,
        config=config,
        epoch=epochs,
        final_loss=history['train_losses'][-1],
        datasets_stats=dataset_stats,
        artifacts_dir="artifacts",
        save_doc_embeddings=True,
        datasets=datasets,
        tokenizer=tokenizer
    )
    
    print(f"✅ Model saved to: {artifacts_dir}")

    # Demo the evaluator
    demo_evaluator(model, trainer, tokenizer, device, datasets, task_mode, config)

    print(f"\n🎉 {task_name} training completed successfully!")
    print(f"📁 Trained model saved in: {artifacts_dir}")


if __name__ == "__main__":
    main() 