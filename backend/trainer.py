import torch
import torch.nn.functional as F
import time
import numpy as np
import sys
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from model import TwoTowerModel, ModelFactory
from utils import clean_memory
import wandb


class TwoTowerTrainer:
    """Clean trainer for Two-Tower model with triplet learning and comprehensive metrics."""
    
    def __init__(self, model: TwoTowerModel, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # Create optimizer and loss function directly
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('LR', 0.001)
        )
        
        self.loss_function = ModelFactory.get_loss_function(
            loss_type=config.get('LOSS_TYPE', 'triplet'),
            margin=config.get('MARGIN', 1.0)
        )
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Initialize WandB
        if not wandb.run:
            wandb.init(project="two-tower-ml-retrieval")
    
    def create_progress_bar(self, current, total, width=40, loss=None, accuracy=None, prefix=""):
        """Create a visual progress bar with metrics."""
        percent = float(current) / total
        filled_width = int(width * percent)
        
        # Create the bar
        bar = '█' * filled_width + '░' * (width - filled_width)
        
        # Format the output
        percentage = percent * 100
        progress_info = f"{current:4d}/{total:4d}"
        
        # Add metrics if provided
        metrics_str = ""
        if loss is not None and accuracy is not None:
            metrics_str = f" │ Loss: {loss:6.4f} │ Acc: {accuracy:5.3f}"
        
        # Complete progress line
        line = f"\r{prefix}[{bar}] {percentage:5.1f}% ({progress_info}){metrics_str}"
        
        return line
    
    def compute_triplet_metrics(self, query_emb, pos_emb, neg_emb):
        """Compute basic triplet metrics."""
        with torch.no_grad():
            pos_sim = (query_emb * pos_emb).sum(dim=1)
            neg_sim = (query_emb * neg_emb).sum(dim=1)
            
            return {
                'accuracy': (pos_sim > neg_sim).float().mean().item(),
                'similarity_gap': (pos_sim - neg_sim).mean().item(),
                'pos_similarity': pos_sim.mean().item(),
                'neg_similarity': neg_sim.mean().item(),
                'embedding_magnitude': query_emb.norm(dim=1).mean().item()
            }
    
    def compute_retrieval_metrics(self, query_emb, doc_emb, k_values=[5, 10]):
        """Compute recall metrics for retrieval evaluation."""
        batch_size = query_emb.size(0)
        similarities = torch.mm(query_emb, doc_emb.t())
        _, top_k_indices = torch.topk(similarities, k=max(k_values), dim=1)
        
        metrics = {}
        for k in k_values:
            # Each query's positive doc is at index i
            recall_scores = []
            for i in range(batch_size):
                recall_scores.append(1.0 if i in top_k_indices[i, :k] else 0.0)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)
        
        return metrics
    
    def compute_ranking_metrics(self, query_emb, doc_emb, k_values=[5, 10]):
        """Compute ranking metrics: MRR, NDCG, MAP."""
        batch_size = query_emb.size(0)
        similarities = torch.mm(query_emb, doc_emb.t())
        
        # Get rankings (higher similarity = better rank)
        _, rankings = torch.sort(similarities, dim=1, descending=True)
        
        mrr_scores = []
        ndcg_scores = {k: [] for k in k_values}
        map_scores = {k: [] for k in k_values}
        
        for i in range(batch_size):
            # Find position of positive doc (index i) in rankings
            pos_rank = (rankings[i] == i).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
            
            # MRR: Mean Reciprocal Rank
            mrr_scores.append(1.0 / pos_rank)
            
            # NDCG and MAP at different k values
            for k in k_values:
                if pos_rank <= k:
                    # NDCG@k: For binary relevance, NDCG = 1/log2(rank+1)
                    ndcg_scores[k].append(1.0 / np.log2(pos_rank + 1))
                    
                    # MAP@k: For single relevant document, AP = 1/rank
                    map_scores[k].append(1.0 / pos_rank)
                else:
                    ndcg_scores[k].append(0.0)
                    map_scores[k].append(0.0)
        
        metrics = {'mrr': np.mean(mrr_scores)}
        for k in k_values:
            metrics[f'ndcg_at_{k}'] = np.mean(ndcg_scores[k])
            metrics[f'map_at_{k}'] = np.mean(map_scores[k])
        
        return metrics
    
    def evaluate_batch(self, val_loader, max_batches=3):
        """Quick evaluation with both retrieval and ranking metrics."""
        self.model.eval()
        
        all_retrieval_metrics = []
        all_ranking_metrics = []
        
        with torch.no_grad():
            for batch_idx, (queries, pos_docs, neg_docs) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                # Move to device
                queries = queries.to(self.device, non_blocking=True)
                pos_docs = pos_docs.to(self.device, non_blocking=True) 
                neg_docs = neg_docs.to(self.device, non_blocking=True)
                
                # Get embeddings
                query_emb = self.model.encode_query(queries)
                pos_emb = self.model.encode_document(pos_docs)
                neg_emb = self.model.encode_document(neg_docs)
                
                # Combine positive and negative docs for evaluation
                doc_emb = torch.cat([pos_emb, neg_emb], dim=0)
                
                # Compute metrics
                retrieval_metrics = self.compute_retrieval_metrics(query_emb, doc_emb)
                ranking_metrics = self.compute_ranking_metrics(query_emb, doc_emb)
                
                all_retrieval_metrics.append(retrieval_metrics)
                all_ranking_metrics.append(ranking_metrics)
        
        # Average across batches
        if not all_retrieval_metrics:
            return {}
        
        final_metrics = {}
        
        # Average retrieval metrics
        for key in all_retrieval_metrics[0].keys():
            final_metrics[key] = np.mean([m[key] for m in all_retrieval_metrics])
        
        # Average ranking metrics
        for key in all_ranking_metrics[0].keys():
            final_metrics[key] = np.mean([m[key] for m in all_ranking_metrics])
        
        return final_metrics
    
    def train_epoch(self, train_loader, val_loader, epoch):
        """Train one epoch with comprehensive logging."""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {'accuracy': 0, 'similarity_gap': 0, 'embedding_magnitude': 0}
        batch_count = 0
        
        total_batches = len(train_loader)
        print(f"\n🚀 Epoch {epoch+1:2d} Progress:")
        
        for batch_idx, (queries, pos_docs, neg_docs) in enumerate(train_loader):
            # Move to device
            queries = queries.to(self.device, non_blocking=True)
            pos_docs = pos_docs.to(self.device, non_blocking=True)
            neg_docs = neg_docs.to(self.device, non_blocking=True)
            
            # Forward pass
            query_emb = self.model.encode_query(queries)
            pos_emb = self.model.encode_document(pos_docs)
            neg_emb = self.model.encode_document(neg_docs)
            
            # Compute loss and metrics
            loss = self.loss_function((query_emb, pos_emb, neg_emb))
            batch_metrics = self.compute_triplet_metrics(query_emb, pos_emb, neg_emb)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track progress
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]
            batch_count += 1
            
            # Dynamic progress bar - update every batch
            progress_bar = self.create_progress_bar(
                current=batch_count,
                total=total_batches,
                loss=loss.item(),
                accuracy=batch_metrics['accuracy'],
                prefix="   "
            )
            print(progress_bar, end='', flush=True)
            
            # Detailed logging every 50 batches
            if batch_count % 50 == 0:
                print(f"\n     ┌─ Gap:  {batch_metrics['similarity_gap']:7.3f} │ Magnitude: {batch_metrics['embedding_magnitude']:6.3f}")
                print(f"     └─ Pos Sim: {batch_metrics['pos_similarity']:6.3f} │ Neg Sim: {batch_metrics['neg_similarity']:6.3f}")
                
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": batch_metrics['accuracy'],
                    "batch_similarity_gap": batch_metrics['similarity_gap'],
                    "batch_magnitude": batch_metrics['embedding_magnitude'],
                    "batch": batch_count,
                    "epoch": epoch + 1
                })
            
            # Evaluation metrics every 200 batches
            if batch_count % 200 == 0 and val_loader is not None:
                eval_metrics = self.evaluate_batch(val_loader)
                
                print(f"\n     🎯 EVALUATION METRICS at Batch {batch_count}")
                print(f"     ┌─ Retrieval  │ R@5: {eval_metrics.get('recall_at_5', 0):6.3f} │ R@10: {eval_metrics.get('recall_at_10', 0):6.3f}")
                print(f"     └─ Ranking   │ MRR: {eval_metrics.get('mrr', 0):6.3f} │ NDCG@5: {eval_metrics.get('ndcg_at_5', 0):6.3f}")
                
                # Log evaluation metrics to WandB
                wandb_metrics = {f"batch_{k}": v for k, v in eval_metrics.items()}
                wandb_metrics.update({"batch": batch_count, "epoch": epoch + 1})
                wandb.log(wandb_metrics)
                
                self.model.train()  # Back to training mode
                print(f"   ", end="")  # Reset progress bar indentation
            
            # Memory cleanup
            if batch_count % 500 == 0:
                clean_memory()
        
        # Complete the progress bar
        avg_loss = epoch_loss / batch_count
        progress_bar = self.create_progress_bar(
            current=total_batches,
            total=total_batches,
            loss=avg_loss,
            accuracy=epoch_metrics['accuracy'] / batch_count,
            prefix="   "
        )
        print(progress_bar)  # Final progress bar at 100%
        
        # Return epoch averages
        for key in epoch_metrics:
            epoch_metrics[key] /= batch_count
        
        return avg_loss, epoch_metrics
    
    def validate_epoch(self, val_loader):
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0
        total_metrics = {'accuracy': 0, 'similarity_gap': 0, 'embedding_magnitude': 0}
        batch_count = 0
        
        with torch.no_grad():
            for queries, pos_docs, neg_docs in val_loader:
                queries = queries.to(self.device, non_blocking=True)
                pos_docs = pos_docs.to(self.device, non_blocking=True)
                neg_docs = neg_docs.to(self.device, non_blocking=True)
                
                query_emb = self.model.encode_query(queries)
                pos_emb = self.model.encode_document(pos_docs)
                neg_emb = self.model.encode_document(neg_docs)
                
                loss = self.loss_function((query_emb, pos_emb, neg_emb))
                batch_metrics = self.compute_triplet_metrics(query_emb, pos_emb, neg_emb)
                
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]
                batch_count += 1
        
        # Return averages
        avg_loss = total_loss / batch_count
        for key in total_metrics:
            total_metrics[key] /= batch_count
        
        return avg_loss, total_metrics
    
    def train(self, train_loader, val_loader=None, epochs=None):
        """Main training loop."""
        if epochs is None:
            epochs = self.config.get('EPOCHS', 10)
        
        print(f"🚀 Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader, val_loader, epoch)
            self.train_losses.append(train_loss)
            
            print(f"\n{'='*60}")
            print(f"✅ EPOCH {epoch+1:2d} TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"📊 Training Metrics:")
            print(f"   ┌─ Loss:      {train_loss:8.4f} │ Accuracy:  {train_metrics['accuracy']:7.3f}")
            print(f"   └─ Gap:       {train_metrics['similarity_gap']:8.3f} │ Magnitude: {train_metrics['embedding_magnitude']:7.3f}")
            
            # Validation
            wandb_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_metrics['accuracy'],
                "train_similarity_gap": train_metrics['similarity_gap'],
                "train_magnitude": train_metrics['embedding_magnitude']
            }
            
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                print(f"📊 Epoch {epoch+1} Validation:")
                print(f"   Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.3f}")
                print(f"   Gap: {val_metrics['similarity_gap']:.3f}, Mag: {val_metrics['embedding_magnitude']:.3f}")
                
                # Full evaluation metrics
                eval_metrics = self.evaluate_batch(val_loader, max_batches=5)
                print(f"🎯 Final Metrics:")
                print(f"   Retrieval - R@5: {eval_metrics.get('recall_at_5', 0):.3f}, R@10: {eval_metrics.get('recall_at_10', 0):.3f}")
                print(f"   Ranking - MRR: {eval_metrics.get('mrr', 0):.3f}, NDCG@5: {eval_metrics.get('ndcg_at_5', 0):.3f}, MAP@5: {eval_metrics.get('map_at_5', 0):.3f}")
                
                # Track best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"🌟 New best validation loss: {val_loss:.4f}")
                
                # Add validation metrics to WandB log
                wandb_log.update({
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics['accuracy'],
                    "val_similarity_gap": val_metrics['similarity_gap'],
                    "val_magnitude": val_metrics['embedding_magnitude'],
                    **{f"epoch_{k}": v for k, v in eval_metrics.items()}
                })
            
            wandb.log(wandb_log)
            clean_memory()
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        } 