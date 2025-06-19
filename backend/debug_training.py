#!/usr/bin/env python3
"""
Debug script to identify why training loss is stuck at 0.333
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from data_loader import DataLoader
from tokenizer import PretrainedTokenizer
from dataset import DataLoaderFactory
from model import ModelFactory, TwoTowerModel
from utils import load_config, get_best_device, load_pretrained_embeddings


def analyze_embeddings_and_similarities(model, dataloader, device, num_batches=3):
    """Analyze embeddings and similarities to debug learning issues."""
    model.eval()
    
    all_pos_sims = []
    all_neg_sims = []
    all_query_norms = []
    all_pos_norms = []
    all_neg_norms = []
    
    print("🔍 ANALYZING EMBEDDINGS AND SIMILARITIES")
    print("=" * 60)
    
    with torch.no_grad():
        for batch_idx, (queries, pos_docs, neg_docs) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            queries = queries.to(device)
            pos_docs = pos_docs.to(device)
            neg_docs = neg_docs.to(device)
            
            # Get embeddings
            query_emb = model.encode_query(queries)
            pos_emb = model.encode_document(pos_docs)
            neg_emb = model.encode_document(neg_docs)
            
            # Compute similarities
            pos_sim = F.cosine_similarity(query_emb, pos_emb, dim=1)
            neg_sim = F.cosine_similarity(query_emb, neg_emb, dim=1)
            
            # Compute norms
            query_norms = query_emb.norm(dim=1)
            pos_norms = pos_emb.norm(dim=1)
            neg_norms = neg_emb.norm(dim=1)
            
            all_pos_sims.extend(pos_sim.cpu().numpy())
            all_neg_sims.extend(neg_sim.cpu().numpy())
            all_query_norms.extend(query_norms.cpu().numpy())
            all_pos_norms.extend(pos_norms.cpu().numpy())
            all_neg_norms.extend(neg_norms.cpu().numpy())
            
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Query embeddings - Mean: {query_emb.mean().item():.4f}, Std: {query_emb.std().item():.4f}")
            print(f"  Pos embeddings   - Mean: {pos_emb.mean().item():.4f}, Std: {pos_emb.std().item():.4f}")
            print(f"  Neg embeddings   - Mean: {neg_emb.mean().item():.4f}, Std: {neg_emb.std().item():.4f}")
            print(f"  Pos similarities - Mean: {pos_sim.mean().item():.4f}, Std: {pos_sim.std().item():.4f}")
            print(f"  Neg similarities - Mean: {neg_sim.mean().item():.4f}, Std: {neg_sim.std().item():.4f}")
            print(f"  Similarity gap   - Mean: {(pos_sim - neg_sim).mean().item():.4f}")
    
    # Overall statistics
    all_pos_sims = np.array(all_pos_sims)
    all_neg_sims = np.array(all_neg_sims)
    all_query_norms = np.array(all_query_norms)
    all_pos_norms = np.array(all_pos_norms)
    all_neg_norms = np.array(all_neg_norms)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Positive similarities: μ={all_pos_sims.mean():.4f}, σ={all_pos_sims.std():.4f}")
    print(f"  Negative similarities: μ={all_neg_sims.mean():.4f}, σ={all_neg_sims.std():.4f}")
    print(f"  Similarity gap:        μ={np.mean(all_pos_sims - all_neg_sims):.4f}")
    print(f"  Query norm:            μ={all_query_norms.mean():.4f}, σ={all_query_norms.std():.4f}")
    print(f"  Pos doc norm:          μ={all_pos_norms.mean():.4f}, σ={all_pos_norms.std():.4f}")
    print(f"  Neg doc norm:          μ={all_neg_norms.mean():.4f}, σ={all_neg_norms.std():.4f}")
    
    return {
        'pos_sim_mean': all_pos_sims.mean(),
        'neg_sim_mean': all_neg_sims.mean(),
        'similarity_gap': np.mean(all_pos_sims - all_neg_sims),
        'query_norm_mean': all_query_norms.mean(),
        'pos_norm_mean': all_pos_norms.mean(),
        'neg_norm_mean': all_neg_norms.mean()
    }


def test_loss_computation(model, dataloader, device, margin=0.3):
    """Test loss computation to verify it's working correctly."""
    model.eval()
    
    print(f"\n🧮 TESTING LOSS COMPUTATION (margin={margin})")
    print("=" * 60)
    
    with torch.no_grad():
        for batch_idx, (queries, pos_docs, neg_docs) in enumerate(dataloader):
            if batch_idx >= 1:  # Just test one batch
                break
                
            queries = queries.to(device)
            pos_docs = pos_docs.to(device)
            neg_docs = neg_docs.to(device)
            
            # Get embeddings
            query_emb = model.encode_query(queries)
            pos_emb = model.encode_document(pos_docs)
            neg_emb = model.encode_document(neg_docs)
            
            # Compute similarities and distances
            pos_sim = F.cosine_similarity(query_emb, pos_emb, dim=1)
            neg_sim = F.cosine_similarity(query_emb, neg_emb, dim=1)
            
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim
            
            # Compute triplet loss manually
            loss_per_sample = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
            loss = loss_per_sample.mean()
            
            print(f"  Sample similarities (first 5):")
            for i in range(min(5, len(pos_sim))):
                print(f"    {i+1}: pos_sim={pos_sim[i]:.4f}, neg_sim={neg_sim[i]:.4f}, loss={loss_per_sample[i]:.4f}")
            
            print(f"  Batch loss: {loss.item():.4f}")
            print(f"  Samples with non-zero loss: {(loss_per_sample > 0).sum().item()}/{len(loss_per_sample)}")
            
            # Check if all samples have loss equal to margin (indicating no learning)
            margin_losses = (loss_per_sample == margin).sum().item()
            print(f"  Samples with loss = margin: {margin_losses}/{len(loss_per_sample)}")
            
            if margin_losses == len(loss_per_sample):
                print("  ⚠️  WARNING: All samples have loss equal to margin - model not learning!")
            
            return loss.item()


def check_gradient_flow(model, dataloader, device):
    """Check if gradients are flowing properly through the model."""
    model.train()
    
    print(f"\n⚡ CHECKING GRADIENT FLOW")
    print("=" * 60)
    
    # Get one batch
    batch = next(iter(dataloader))
    queries, pos_docs, neg_docs = batch
    queries = queries.to(device)
    pos_docs = pos_docs.to(device)
    neg_docs = neg_docs.to(device)
    
    # Forward pass
    query_emb = model.encode_query(queries)
    pos_emb = model.encode_document(pos_docs)
    neg_emb = model.encode_document(neg_docs)
    
    # Compute loss
    from model import ModelFactory
    loss_fn = ModelFactory.get_loss_function('adaptive_triplet', margin=0.3)
    loss = loss_fn((query_emb, pos_emb, neg_emb))
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    total_norm = 0
    param_count = 0
    zero_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if param_norm.item() < 1e-8:
                zero_grad_params += 1
                
            print(f"  {name}: grad_norm={param_norm.item():.6f}, param_shape={param.shape}")
        else:
            print(f"  {name}: NO GRADIENT")
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"\n  Total gradient norm: {total_norm:.6f}")
    print(f"  Parameters with gradients: {param_count}")
    print(f"  Parameters with near-zero gradients: {zero_grad_params}")
    
    if total_norm < 1e-6:
        print("  ⚠️  WARNING: Very small gradient norm - vanishing gradients!")
    elif total_norm > 10:
        print("  ⚠️  WARNING: Very large gradient norm - exploding gradients!")
    
    return total_norm


def main():
    print("🔍 DEBUGGING TRAINING ISSUES")
    print("=" * 80)
    
    # Load config
    config = load_config('backend/config.json')
    device = get_best_device()
    
    print(f"📋 Current configuration:")
    print(f"  Learning rate: {config.get('LR', 'N/A')}")
    print(f"  Margin: {config.get('MARGIN', 'N/A')}")
    print(f"  Hidden dim: {config.get('HIDDEN_DIM', 'N/A')}")
    print(f"  Dropout: {config.get('DROPOUT', 'N/A')}")
    print(f"  Normalize output: {config.get('NORMALIZE_OUTPUT', 'N/A')}")
    print(f"  Subsample ratio: {config.get('SUBSAMPLE_RATIO', 'N/A')}")
    
    # Load tokenizer and embeddings
    tokenizer = PretrainedTokenizer(config['WORD_TO_IDX_PATH'])
    pretrained_embeddings = load_pretrained_embeddings(config['EMBEDDINGS_PATH'])
    
    # Apply same vocab reduction as in main.py
    words_to_drop = 0
    new_vocab_size = tokenizer.vocab_size() - words_to_drop
    pretrained_embeddings = pretrained_embeddings[:new_vocab_size]
    
    config['VOCAB_SIZE'] = tokenizer.vocab_size()
    config['EMBED_DIM'] = pretrained_embeddings.shape[1]
    
    # Load small dataset
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets(subsample_ratio=0.0001)  # Very small for debugging
    
    # Create data loader
    dataloader_factory = DataLoaderFactory(config)
    dataloaders = dataloader_factory.create_dataloaders(datasets, tokenizer)
    train_loader = dataloaders['train']
    
    # Create model
    model = ModelFactory.create_two_tower_model(config, pretrained_embeddings)
    model.to(device)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run diagnostics
    stats = analyze_embeddings_and_similarities(model, train_loader, device)
    loss_value = test_loss_computation(model, train_loader, device, margin=config.get('MARGIN', 0.3))
    grad_norm = check_gradient_flow(model, train_loader, device)
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    if stats['similarity_gap'] < 0.01:
        print("  ❌ Issue: Very small similarity gap between positive and negative examples")
        print("     → The model can't distinguish between positive and negative documents")
        print("     → This is likely why loss is stuck around margin value")
    
    if abs(stats['query_norm_mean'] - 1.0) > 0.1:
        print("  ❌ Issue: Query embeddings not properly normalized")
        print("     → Enable normalize_output in config")
    
    if config.get('LR', 0.002) > 0.01:
        print("  ⚠️  Warning: Learning rate might be too high")
        print("     → Try reducing learning rate to 0.0001-0.001")
    
    if config.get('DROPOUT', 0.3) > 0.5:
        print("  ⚠️  Warning: Dropout might be too high")
        print("     → Try reducing dropout to 0.1-0.3")
    
    if config.get('MARGIN', 0.3) < 0.5:
        print("  ⚠️  Warning: Margin might be too small")
        print("     → Try increasing margin to 0.5-1.0")
    
    if not config.get('NORMALIZE_OUTPUT', True):
        print("  ❌ Issue: Output normalization disabled")
        print("     → Enable normalize_output for better cosine similarity computation")
    
    print("\n🔧 Suggested config changes:")
    print("  - Set NORMALIZE_OUTPUT: true")
    print("  - Set LR: 0.0005")
    print("  - Set MARGIN: 1.0") 
    print("  - Set DROPOUT: 0.2")
    print("  - Set HIDDEN_DIM: 512 (if computationally feasible)")


if __name__ == "__main__":
    main() 