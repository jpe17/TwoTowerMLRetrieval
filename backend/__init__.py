"""
Two-Tower ML Retrieval Backend Package

This package provides a complete implementation of a two-tower retrieval model
with modular components for data loading, tokenization, model training, and evaluation.
"""

from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import TripletDataset, DataLoaderFactory, collate_fn
from model import RNNEncoder, TwoTowerModel, ModelFactory, triplet_loss_function, cosine_similarity_loss
from trainer import TwoTowerTrainer, TrainerFactory
from evaluator import TwoTowerEvaluator
from utils import (
    get_best_device, setup_memory_optimization, clean_memory, get_memory_usage,
    load_config, save_config, load_pretrained_embeddings, save_model_artifacts,
    load_model_artifacts, print_model_summary, validate_config
)

__version__ = "1.0.0"
__author__ = "Two-Tower ML Retrieval"

# Main components
__all__ = [
    # Data handling
    'DataLoader',
    'PretrainedTokenizer',
    'TripletDataset',
    'DataLoaderFactory',
    'collate_fn',
    
    # Model components
    'RNNEncoder',
    'TwoTowerModel',
    'ModelFactory',
    'triplet_loss_function',
    'cosine_similarity_loss',
    
    # Training and evaluation
    'TwoTowerTrainer',
    'TrainerFactory',
    'TwoTowerEvaluator',
    
    # Utilities
    'get_best_device',
    'setup_memory_optimization',
    'clean_memory',
    'get_memory_usage',
    'load_config',
    'save_config',
    'load_pretrained_embeddings',
    'save_model_artifacts',
    'load_model_artifacts',
    'print_model_summary',
    'validate_config',
] 