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
        
        # Create optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('LR', 0.001),
            weight_decay=config.get('WEIGHT_DECAY', 0.0)
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
    
    def compute_retrieval_metrics_fixed(self, query_emb, doc_emb, pos_labels, k_values=[5, 10]):
        """Compute recall metrics with explicit positive labels."""
        batch_size = query_emb.size(0)
        similarities = torch.mm(query_emb, doc_emb.t())
        _, top_k_indices = torch.topk(similarities, k=max(k_values), dim=1)
        
        metrics = {}
        for k in k_values:
            recall_scores = []
            for i in range(batch_size):
                # Check if the positive document for query i is in top-k
                pos_doc_idx = pos_labels[i].item()
                recall_scores.append(1.0 if pos_doc_idx in top_k_indices[i, :k] else 0.0)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)
        
        return metrics
    
    def compute_ranking_metrics_fixed(self, query_emb, doc_emb, pos_labels, k_values=[5, 10]):
        """Compute ranking metrics with explicit positive labels."""
        batch_size = query_emb.size(0)
        similarities = torch.mm(query_emb, doc_emb.t())
        
        # Get rankings (higher similarity = better rank)
        _, rankings = torch.sort(similarities, dim=1, descending=True)
        
        mrr_scores = []
        ndcg_scores = {k: [] for k in k_values}
        map_scores = {k: [] for k in k_values}
        
        for i in range(batch_size):
            # Find position of positive doc in rankings
            pos_doc_idx = pos_labels[i].item()
            pos_rank = (rankings[i] == pos_doc_idx).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
            
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
                # neg_docs are not used in this evaluation setup, but are part of the loader
                
                # Get embeddings
                query_emb = self.model.encode_query(queries)
                pos_emb = self.model.encode_document(pos_docs)
                
                # For a more realistic in-batch evaluation, we use only the positive 
                # documents from the batch as the search corpus. The task is to find the 
                # correct positive document for each query from this smaller, but more
                # relevant, pool.
                batch_size = query_emb.size(0)
                doc_emb = pos_emb  # Shape: [batch_size, hidden_dim]
                
                # The positive label for query `i` is at index `i` in the `doc_emb` pool.
                pos_labels = torch.arange(batch_size, device=query_emb.device)
                
                # Compute metrics with corrected labels
                retrieval_metrics = self.compute_retrieval_metrics_fixed(query_emb, doc_emb, pos_labels)
                ranking_metrics = self.compute_ranking_metrics_fixed(query_emb, doc_emb, pos_labels)
                
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
    
    def compute_val_loss_quick(self, val_loader, max_batches=3):
        """Quick validation loss computation on a few batches."""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (queries, pos_docs, neg_docs) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                # Move to device
                queries = queries.to(self.device, non_blocking=True)
                pos_docs = pos_docs.to(self.device, non_blocking=True)
                neg_docs = neg_docs.to(self.device, non_blocking=True)
                
                # Get embeddings and compute loss
                query_emb = self.model.encode_query(queries)
                pos_emb = self.model.encode_document(pos_docs)
                neg_emb = self.model.encode_document(neg_docs)
                
                loss = self.loss_function((query_emb, pos_emb, neg_emb))
                total_loss += loss.item()
                batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else 0.0
    
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
            
            # Gradient clipping for stability
            if self.config.get('GRAD_CLIP', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['GRAD_CLIP'])
            
            self.optimizer.step()
            
            # Track progress
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += batch_metrics[key]
            batch_count += 1
            
            # Step the scheduler 
            # OneCycleLR steps every batch, others step every 50 batches
            scheduler_type = self.config.get('SCHEDULER_TYPE', 'onecycle')
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if scheduler_type == 'onecycle':
                    self.scheduler.step()  # Step every batch
                elif batch_count % 1 == 0:
                    self.scheduler.step()  # Step every 50 batches
            
            # Dynamic progress bar - update every batch
            progress_bar = self.create_progress_bar(
                current=batch_count,
                total=total_batches,
                loss=loss.item(),
                accuracy=batch_metrics['accuracy'],
                prefix="   "
            )
            print(progress_bar, end='', flush=True)
            
            # Detailed logging and evaluation every 50 batches
            if batch_count % 1 == 0:
                print(f"\n     ┌─ Gap:  {batch_metrics['similarity_gap']:7.3f} │ Magnitude: {batch_metrics['embedding_magnitude']:6.3f}")
                print(f"     └─ Pos Sim: {batch_metrics['pos_similarity']:6.3f} │ Neg Sim: {batch_metrics['neg_similarity']:6.3f}")
                
                # Prepare WandB log with current training metrics
                current_step = batch_count + epoch * len(train_loader)
                wandb_log = {
                    "loss": loss.item(),
                    "accuracy": batch_metrics['accuracy'],
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "step": current_step
                }
                
                # Add validation metrics if validation loader provided
                if val_loader is not None:
                    # Compute validation loss on more batches for better estimates
                    val_loss = self.compute_val_loss_quick(val_loader, max_batches=10)
                    eval_metrics = self.evaluate_batch(val_loader, max_batches=10)
                    
                    print(f"\n     🎯 EVALUATION METRICS at Batch {batch_count}")
                    print(f"     ┌─ Train Loss: {loss.item():6.4f} │ Val Loss: {val_loss:6.4f}")
                    print(f"     ├─ Retrieval  │ R@5: {eval_metrics.get('recall_at_5', 0):6.3f} │ R@10: {eval_metrics.get('recall_at_10', 0):6.3f}")
                    print(f"     └─ Ranking    │ MRR: {eval_metrics.get('mrr', 0):6.3f} │ NDCG@5: {eval_metrics.get('ndcg_at_5', 0):6.3f}")
                    
                    # Add validation metrics to the same WandB log
                    wandb_log.update({
                        "val_loss": val_loss,
                        "recall_at_5": eval_metrics.get('recall_at_5', 0),
                        "recall_at_10": eval_metrics.get('recall_at_10', 0),
                        "mrr": eval_metrics.get('mrr', 0),
                        "ndcg_at_5": eval_metrics.get('ndcg_at_5', 0)
                    })
                    
                    # Track best validation loss
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        print(f"     🌟 New best validation loss: {val_loss:.4f}")
                    
                    self.model.train()  # Back to training mode
                
                # Single WandB log every 200 batches with all metrics
                wandb.log(wandb_log)
                print(f"   ", end="")  # Reset progress bar indentation
            
            # Memory cleanup
            if batch_count % 100 == 0:
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
        
        # Choose your learning rate scheduler
        scheduler_type = self.config.get('SCHEDULER_TYPE', 'onecycle')  # 'onecycle', 'step', 'cosine', 'exponential'
        
        if scheduler_type == 'onecycle':
            # Original OneCycleLR (increases then decreases, steps every batch)
            total_steps = len(train_loader) * epochs
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('MAX_LR', 0.01),
                total_steps=total_steps,
                pct_start=self.config.get('WARMUP_PCT', 0.1),
                anneal_strategy='cos',
                div_factor=self.config.get('DIV_FACTOR', 10),
                final_div_factor=self.config.get('FINAL_DIV_FACTOR', 100)
            )
            scheduler_info = f"OneCycleLR: {total_steps} total steps, max_lr={self.config.get('MAX_LR', 0.01)}"
            
        elif scheduler_type == 'step':
            # StepLR: Decreases LR by factor every 50 batches
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1,  # Step every call (we'll call it every 50 batches)
                gamma=self.config.get('GAMMA', 0.9)  # Multiply LR by 0.9
            )
            scheduler_info = f"StepLR: decay by {self.config.get('GAMMA', 0.9)} every 50 batches"
            
        elif scheduler_type == 'exponential':
            # ExponentialLR: Exponential decay every 50 batches
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.get('GAMMA', 0.95)  # Multiply by 0.95 every 50 batches
            )
            scheduler_info = f"ExponentialLR: decay by {self.config.get('GAMMA', 0.95)} every 50 batches"
            
        else:
            # No scheduler
            self.scheduler = None
            scheduler_info = "No scheduler - constant learning rate"
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📈 Learning Rate: {scheduler_info}")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_metrics = self.train_epoch(train_loader, val_loader, epoch)
            self.train_losses.append(train_loss)
            
            print(f"\n{'='*60}")
            print(f"✅ EPOCH {epoch+1:2d} TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"📊 Training Summary:")
            print(f"   ┌─ Train Loss: {train_loss:7.4f} │ Accuracy:  {train_metrics['accuracy']:7.3f}")
            print(f"   └─ Gap:       {train_metrics['similarity_gap']:8.3f} │ Magnitude: {train_metrics['embedding_magnitude']:7.3f}")
            
            # Simple epoch-level WandB logging - training metrics only
            wandb.log({
                "epoch": epoch + 1,
                "epoch_train_loss": train_loss,
                "epoch_train_accuracy": train_metrics['accuracy'],
                "epoch_learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            clean_memory()
        
        # Training summary
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        if self.best_val_loss < float('inf'):
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'best_val_loss': self.best_val_loss
        }