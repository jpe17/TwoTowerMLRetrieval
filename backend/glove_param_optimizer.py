#!/usr/bin/env python3
"""
Quick Parameter Optimization for GloVe Embeddings in TwoTower Model
Optimizes key hyperparameters for 200-dimensional GloVe embeddings.
"""

import sys
import torch
import json
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import ModelFactory
from trainer import TwoTowerTrainer
from utils import (
    load_config, get_best_device, setup_memory_optimization,
    load_pretrained_embeddings
)


class GloVeOptimizer:
    """Quick parameter optimization for GloVe embeddings."""
    
    def __init__(self, base_config_path='backend/config_glove_optimized.json'):
        self.base_config = load_config(base_config_path)
        self.device = get_best_device()
        setup_memory_optimization()
        
        # Define parameter search space optimized for GloVe
        self.param_grid = {
            'HIDDEN_DIM': [256, 384, 512],
            'DROPOUT': [0.1, 0.15, 0.2],
            'EMBEDDING_DROPOUT': [0.05, 0.1, 0.15],
            'LR': [3e-3, 5e-3, 7e-3],
            'MARGIN': [0.3, 0.5, 0.7],
            'BATCH_SIZE': [512, 1024, 2048],
        }
        
        # Fast evaluation settings
        self.fast_config = {
            'SUBSAMPLE_RATIO': 0.001,  # Very small for quick testing
            'EPOCHS': 1,
            'MEMORY_CLEANUP_FREQUENCY': 10
        }
        
        self.results = []
        
    def load_data_once(self):
        """Load data and tokenizer once to reuse across experiments."""
        print("📚 Loading tokenizer and embeddings...")
        self.tokenizer = PretrainedTokenizer(self.base_config['WORD_TO_IDX_PATH'])
        self.pretrained_embeddings = load_pretrained_embeddings(self.base_config['EMBEDDINGS_PATH'])
        
        # Reduce vocab for faster optimization
        words_to_keep = min(50000, self.tokenizer.vocab_size())
        self.pretrained_embeddings = self.pretrained_embeddings[:words_to_keep]
        
        # Filter tokenizer
        filtered_word2idx = {word: idx for word, idx in self.tokenizer.word2idx.items() if idx < words_to_keep}
        self.tokenizer.word2idx = filtered_word2idx
        self.tokenizer.idx2word = {idx: word for word, idx in filtered_word2idx.items()}
        
        print(f"📊 Vocab size reduced to: {self.tokenizer.vocab_size():,}")
        
        # Load datasets
        data_loader = DataLoader(self.base_config)
        self.datasets = data_loader.load_datasets(subsample_ratio=self.fast_config['SUBSAMPLE_RATIO'])
        
        print(f"📈 Dataset sizes: train={len(self.datasets['train']):,}, val={len(self.datasets.get('validation', [])):,}")
        
    def create_config(self, params):
        """Create config with specific parameter combination."""
        config = self.base_config.copy()
        config.update(self.fast_config)
        config.update(params)
        
        # Add computed values
        config['VOCAB_SIZE'] = self.tokenizer.vocab_size()
        config['EMBED_DIM'] = self.pretrained_embeddings.shape[1]
        
        return config
        
    def train_and_evaluate(self, config, param_combo):
        """Train model with given config and return validation loss."""
        try:
            print(f"\n🧪 Testing: {param_combo}")
            
            # Create data loaders
            dataloader_factory = DataLoaderFactory(config)
            dataloaders = dataloader_factory.create_dataloaders(self.datasets, self.tokenizer)
            
            # Create model
            model = ModelFactory.create_two_tower_model(config, self.pretrained_embeddings)
            model = model.to(self.device)
            
            # Create trainer
            trainer = TwoTowerTrainer(model, config, self.device)
            
            # Quick training
            history = trainer.train(
                train_loader=dataloaders['train'],
                val_loader=dataloaders.get('validation'),
                epochs=config['EPOCHS']
            )
            
            # Get validation loss (use last training loss if no validation)
            val_loss = (history.get('val_losses', [])[-1] if history.get('val_losses') 
                       else history['train_losses'][-1])
            
            # Clean up
            del model, trainer, dataloaders
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return val_loss
            
        except Exception as e:
            print(f"❌ Error with {param_combo}: {e}")
            return float('inf')
    
    def optimize(self, max_combinations=15):
        """Run parameter optimization."""
        print("🚀 Starting GloVe Parameter Optimization")
        print("=" * 60)
        
        # Load data once
        self.load_data_once()
        
        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Create all combinations and sample for quick optimization
        all_combinations = list(itertools.product(*param_values))
        
        # Sample random combinations for diversity
        np.random.shuffle(all_combinations)
        selected_combinations = all_combinations[:max_combinations]
        
        print(f"🎯 Testing {len(selected_combinations)} parameter combinations...")
        
        best_loss = float('inf')
        best_params = None
        
        for i, param_values in enumerate(selected_combinations, 1):
            param_combo = dict(zip(param_names, param_values))
            config = self.create_config(param_combo)
            
            print(f"\n[{i}/{len(selected_combinations)}] ", end="")
            val_loss = self.train_and_evaluate(config, param_combo)
            
            # Record result
            result = {
                'combination': i,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat(),
                **param_combo
            }
            self.results.append(result)
            
            # Track best
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = param_combo
                print(f"🎉 New best! Loss: {val_loss:.4f}")
            else:
                print(f"📊 Loss: {val_loss:.4f}")
        
        return best_params, best_loss
    
    def save_results(self, best_params, best_loss):
        """Save optimization results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f'glove_optimization_results_{timestamp}.csv', index=False)
        
        # Create optimized config
        optimized_config = self.base_config.copy()
        optimized_config.update(best_params)
        
        # Reset to full training settings
        optimized_config.update({
            'SUBSAMPLE_RATIO': 0.01,  # Back to reasonable size
            'EPOCHS': 3,
            'MEMORY_CLEANUP_FREQUENCY': 50
        })
        
        with open(f'config_glove_optimized_{timestamp}.json', 'w') as f:
            json.dump(optimized_config, f, indent=4)
        
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print(f"🏆 Best validation loss: {best_loss:.4f}")
        print(f"📋 Best parameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"\n💾 Results saved to: glove_optimization_results_{timestamp}.csv")
        print(f"⚙️  Optimized config saved to: config_glove_optimized_{timestamp}.json")
        
        return optimized_config


def main():
    """Run quick GloVe parameter optimization."""
    optimizer = GloVeOptimizer()
    
    try:
        best_params, best_loss = optimizer.optimize(max_combinations=12)  # Quick test
        optimized_config = optimizer.save_results(best_params, best_loss)
        
        print(f"\n✨ To use optimized parameters, run:")
        print(f"   python backend/main.py --config config_glove_optimized_*.json")
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")


if __name__ == "__main__":
    main() 