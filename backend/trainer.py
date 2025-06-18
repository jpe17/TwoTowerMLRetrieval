import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from model import TwoTowerModel, ModelFactory
from utils import clean_memory
import wandb


class TwoTowerTrainer:
    """Simple, clean trainer for Two-Tower model focusing on essential metrics."""
    
    def __init__(
        self, 
        model: TwoTowerModel, 
        optimizer: torch.optim.Optimizer,
        loss_function,
        device: torch.device,
        config: Dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.config = config
        
        # Detect task mode
        self.task_mode = config.get('TASK_MODE', 'retrieval')
        self.is_ranking = self.task_mode == 'ranking'
        
        # Simple tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Initialize WandB once
        if not wandb.run:
            wandb.init(project="two-tower-ml-retrieval")
    
    def compute_batch_metrics(self, q_vec, pos_vec, neg_vec):
        """Compute core training metrics - task-aware."""
        with torch.no_grad():
            if self.is_ranking:
                # For ranking: q_vec, pos_vec, neg_vec are relevance scores
                pos_scores = q_vec  # Actually the positive scores
                neg_scores = pos_vec  # Actually the negative scores
                
                # Ranking metrics
                acc = (pos_scores > neg_scores).float().mean().item()
                margin = (pos_scores - neg_scores).mean().item()
                margin_violations = (pos_scores - neg_scores < self.config.get('MARGIN', 1.0)).float().mean().item()
                
                return {
                    'accuracy': acc,
                    'margin': margin,
                    'margin_violations': margin_violations,
                    'pos_scores': pos_scores.mean().item(),
                    'neg_scores': neg_scores.mean().item()
                }
            else:
                # For retrieval: standard similarity metrics
                pos_sim = (q_vec * pos_vec).sum(dim=1)
                neg_sim = (q_vec * neg_vec).sum(dim=1)
                
                acc = (pos_sim > neg_sim).float().mean().item()
                gap = (pos_sim - neg_sim).mean().item()
                mag = q_vec.norm(dim=1).mean().item()
                
                return {
                    'accuracy': acc,
                    'similarity_gap': gap,
                    'magnitude': mag,
                    'pos_similarity': pos_sim.mean().item(),
                    'neg_similarity': neg_sim.mean().item()
                }
    
    def compute_recall_metrics(self, query_embeddings, doc_embeddings, k_values=[5, 10]):
        """Simple recall computation for retrieval tasks."""
        if self.is_ranking:
            return {}  # Skip for ranking tasks
            
        batch_size = query_embeddings.size(0)
        
        # Compute similarities
        similarities = torch.mm(query_embeddings, doc_embeddings.t())
        
        # Get top-k indices
        _, top_k_indices = torch.topk(similarities, k=max(k_values), dim=1)
        
        metrics = {}
        for k in k_values:
            # For each query, check if positive doc (index i) is in top-k
            recall_scores = []
            for i in range(batch_size):
                if i in top_k_indices[i, :k]:
                    recall_scores.append(1.0)
                else:
                    recall_scores.append(0.0)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)
        
        return metrics
    
    def compute_ranking_metrics(self, pos_scores, neg_scores, k_values=[5, 10]):
        """Compute ranking metrics for ranking tasks."""
        if not self.is_ranking:
            return {}  # Skip for retrieval tasks
            
        with torch.no_grad():
            # Convert to numpy for easier computation
            pos_scores_np = pos_scores.cpu().numpy()
            neg_scores_np = neg_scores.cpu().numpy()
            
            metrics = {}
            
            # NDCG-like metric: how well we rank positive vs negative
            all_scores = np.concatenate([pos_scores_np, neg_scores_np])
            labels = np.concatenate([np.ones(len(pos_scores_np)), np.zeros(len(neg_scores_np))])
            
            # Sort by scores (descending)
            sorted_indices = np.argsort(all_scores)[::-1]
            sorted_labels = labels[sorted_indices]
            
            # Simple NDCG approximation for k values
            for k in k_values:
                if k > len(sorted_labels):
                    k = len(sorted_labels)
                
                # DCG calculation
                dcg = 0
                for i in range(k):
                    rel = sorted_labels[i]
                    dcg += rel / np.log2(i + 2)  # +2 because log2(1) = 0
                
                # IDCG (perfect ranking)
                ideal_labels = np.sort(labels)[::-1][:k]
                idcg = 0
                for i in range(k):
                    rel = ideal_labels[i]
                    idcg += rel / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics[f'ndcg_at_{k}'] = ndcg
            
            # Mean Reciprocal Rank approximation
            first_relevant_pos = np.where(sorted_labels == 1)[0]
            if len(first_relevant_pos) > 0:
                mrr = 1.0 / (first_relevant_pos[0] + 1)
            else:
                mrr = 0.0
            metrics['mrr'] = mrr
            
            return metrics
    
    def train_epoch(self, train_loader, val_loader, epoch):
        """Train one epoch with task-aware metrics."""
        self.model.train()
        total_loss = 0
        
        # Initialize metrics based on task
        if self.is_ranking:
            total_metrics = {'accuracy': 0, 'margin': 0, 'margin_violations': 0}
        else:
            total_metrics = {'accuracy': 0, 'similarity_gap': 0, 'magnitude': 0}
        
        num_batches = 0
        
        for batch_idx, (query_batch, pos_batch, neg_batch) in enumerate(train_loader):
            # Move to device
            query_batch = query_batch.to(self.device, non_blocking=True)
            pos_batch = pos_batch.to(self.device, non_blocking=True)
            neg_batch = neg_batch.to(self.device, non_blocking=True)
            
            # Forward pass - task-aware
            if self.is_ranking:
                # For ranking: model outputs scores directly
                triplet = (query_batch, pos_batch, neg_batch)
                loss = self.loss_function(triplet)
                
                # Get scores for metrics from the combined model
                with torch.no_grad():
                    # Score positive pairs (query + positive doc)
                    pos_combined = torch.cat([query_batch, pos_batch], dim=1)  # Simple concatenation
                    pos_scores = self.model.encode_text(pos_combined).squeeze()
                    
                    # Score negative pairs (query + negative doc)
                    neg_combined = torch.cat([query_batch, neg_batch], dim=1)  # Simple concatenation
                    neg_scores = self.model.encode_text(neg_combined).squeeze()
                    
                batch_metrics = self.compute_batch_metrics(pos_scores, neg_scores, None)
            else:
                # For retrieval: standard two-tower approach
                q_vec = self.model.encode_query(query_batch)
                pos_vec = self.model.encode_document(pos_batch)
                neg_vec = self.model.encode_document(neg_batch)
                
                loss = self.loss_function((q_vec, pos_vec, neg_vec))
                batch_metrics = self.compute_batch_metrics(q_vec, pos_vec, neg_vec)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            num_batches += 1
            
            # Progress every 50 batches
            if num_batches % 50 == 0:
                current_loss = loss.item()
                current_acc = batch_metrics['accuracy']
                
                if self.is_ranking:
                    current_margin = batch_metrics['margin']
                    current_violations = batch_metrics['margin_violations']
                    print(f"  Epoch {epoch+1}, Batch {num_batches}/{len(train_loader)}, "
                          f"Loss: {current_loss:.4f}, Acc: {current_acc:.3f}, "
                          f"Margin: {current_margin:.3f}, Violations: {current_violations:.3f}")
                    
                    wandb.log({
                        "batch_loss": current_loss,
                        "batch_accuracy": current_acc,
                        "batch_margin": current_margin,
                        "batch_margin_violations": current_violations,
                        "batch": num_batches,
                        "epoch": epoch + 1
                    })
                else:
                    current_gap = batch_metrics['similarity_gap']
                    current_mag = batch_metrics['magnitude']
                    print(f"  Epoch {epoch+1}, Batch {num_batches}/{len(train_loader)}, "
                          f"Loss: {current_loss:.4f}, Acc: {current_acc:.3f}, "
                          f"Gap: {current_gap:.3f}, Mag: {current_mag:.3f}")
                    
                    wandb.log({
                        "batch_loss": current_loss,
                        "batch_accuracy": current_acc,
                        "batch_similarity_gap": current_gap,
                        "batch_magnitude": current_mag,
                        "batch": num_batches,
                        "epoch": epoch + 1
                    })
            
            # Task-specific metrics every 200 batches
            if num_batches % 200 == 0 and val_loader is not None:
                self.model.eval()
                if self.is_ranking:
                    ranking_metrics = self.quick_ranking_check(val_loader)
                    print(f"    NDCG@5: {ranking_metrics.get('ndcg_at_5', 0):.3f}, "
                          f"NDCG@10: {ranking_metrics.get('ndcg_at_10', 0):.3f}, "
                          f"MRR: {ranking_metrics.get('mrr', 0):.3f}")
                    wandb_metrics = {f"batch_{k}": v for k, v in ranking_metrics.items()}
                else:
                    recall_metrics = self.quick_recall_check(val_loader)
                    print(f"    R@5: {recall_metrics.get('recall_at_5', 0):.3f}, "
                          f"R@10: {recall_metrics.get('recall_at_10', 0):.3f}")
                    wandb_metrics = {f"batch_{k}": v for k, v in recall_metrics.items()}
                
                wandb_metrics.update({"batch": num_batches, "epoch": epoch + 1})
                wandb.log(wandb_metrics)
                self.model.train()
            
            # Memory cleanup
            if num_batches % 500 == 0:
                clean_memory()
            
            # Clear tensors
            del query_batch, pos_batch, neg_batch, loss
        
        # Average metrics for epoch
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return avg_loss, total_metrics
    
    def validate_epoch(self, val_loader, epoch):
        """Simple validation with task-aware metrics."""
        self.model.eval()
        total_loss = 0
        
        # Initialize metrics based on task
        if self.is_ranking:
            total_metrics = {'accuracy': 0, 'margin': 0, 'margin_violations': 0}
        else:
            total_metrics = {'accuracy': 0, 'similarity_gap': 0, 'magnitude': 0}
        
        num_batches = 0
        
        with torch.no_grad():
            for query_batch, pos_batch, neg_batch in val_loader:
                query_batch = query_batch.to(self.device, non_blocking=True)
                pos_batch = pos_batch.to(self.device, non_blocking=True)
                neg_batch = neg_batch.to(self.device, non_blocking=True)
                
                if self.is_ranking:
                    triplet = (query_batch, pos_batch, neg_batch)
                    loss = self.loss_function(triplet)
                    
                    # Get scores for metrics from the combined model
                    pos_combined = torch.cat([query_batch, pos_batch], dim=1)
                    pos_scores = self.model.encode_text(pos_combined).squeeze()
                    
                    neg_combined = torch.cat([query_batch, neg_batch], dim=1)
                    neg_scores = self.model.encode_text(neg_combined).squeeze()
                    
                    batch_metrics = self.compute_batch_metrics(pos_scores, neg_scores, None)
                else:
                    q_vec = self.model.encode_query(query_batch)
                    pos_vec = self.model.encode_document(pos_batch)
                    neg_vec = self.model.encode_document(neg_batch)
                    
                    loss = self.loss_function((q_vec, pos_vec, neg_vec))
                    batch_metrics = self.compute_batch_metrics(q_vec, pos_vec, neg_vec)
                
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key]
                num_batches += 1
                
                del query_batch, pos_batch, neg_batch, loss
        
        avg_loss = total_loss / num_batches
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return avg_loss, total_metrics
    
    def quick_recall_check(self, val_loader, max_batches=3):
        """Quick recall check for retrieval tasks."""
        if self.is_ranking:
            return {}
            
        all_recall_metrics = []
        
        with torch.no_grad():
            for batch_idx, (query_batch, pos_batch, neg_batch) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                query_batch = query_batch.to(self.device, non_blocking=True)
                pos_batch = pos_batch.to(self.device, non_blocking=True)
                neg_batch = neg_batch.to(self.device, non_blocking=True)
                
                q_vec = self.model.encode_query(query_batch)
                pos_vec = self.model.encode_document(pos_batch)
                neg_vec = self.model.encode_document(neg_batch)
                
                doc_vec = torch.cat([pos_vec, neg_vec], dim=0)
                recall_metrics = self.compute_recall_metrics(q_vec, doc_vec)
                all_recall_metrics.append(recall_metrics)
                
                del query_batch, pos_batch, neg_batch, q_vec, pos_vec, neg_vec, doc_vec
        
        if not all_recall_metrics:
            return {}
        
        final_metrics = {}
        for key in all_recall_metrics[0].keys():
            final_metrics[key] = np.mean([m[key] for m in all_recall_metrics])
        
        return final_metrics
    
    def quick_ranking_check(self, val_loader, max_batches=3):
        """Quick ranking check for ranking tasks."""
        if not self.is_ranking:
            return {}
            
        all_ranking_metrics = []
        
        with torch.no_grad():
            for batch_idx, (query_batch, pos_batch, neg_batch) in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                
                query_batch = query_batch.to(self.device, non_blocking=True)
                pos_batch = pos_batch.to(self.device, non_blocking=True)
                neg_batch = neg_batch.to(self.device, non_blocking=True)
                
                # Get actual scores from the combined model
                pos_combined = torch.cat([query_batch, pos_batch], dim=1)
                pos_scores = self.model.encode_text(pos_combined).squeeze()
                
                neg_combined = torch.cat([query_batch, neg_batch], dim=1)
                neg_scores = self.model.encode_text(neg_combined).squeeze()
                
                ranking_metrics = self.compute_ranking_metrics(pos_scores, neg_scores)
                all_ranking_metrics.append(ranking_metrics)
                
                del query_batch, pos_batch, neg_batch, pos_scores, neg_scores
        
        if not all_ranking_metrics:
            return {}
        
        final_metrics = {}
        for key in all_ranking_metrics[0].keys():
            final_metrics[key] = np.mean([m[key] for m in all_ranking_metrics])
        
        return final_metrics
    
    def train(self, train_loader, val_loader=None, epochs=None):
        """Main training loop - clean and simple with task-aware metrics."""
        if epochs is None:
            epochs = self.config.get('EPOCHS', 10)
        
        task_name = "Ranking" if self.is_ranking else "Retrieval"
        print(f"🚀 Starting {task_name.lower()} training for {epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, val_loader, epoch)
            self.train_losses.append(train_loss)
            
            print(f"\n✅ Epoch {epoch+1} Training:")
            if self.is_ranking:
                print(f"   Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.3f}, "
                      f"Margin: {train_metrics['margin']:.3f}, Violations: {train_metrics['margin_violations']:.3f}")
            else:
                print(f"   Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.3f}, "
                      f"Gap: {train_metrics['similarity_gap']:.3f}, Mag: {train_metrics['magnitude']:.3f}")
            
            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
                self.val_losses.append(val_loss)
                
                print(f"📊 Epoch {epoch+1} Validation:")
                if self.is_ranking:
                    print(f"   Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.3f}, "
                          f"Margin: {val_metrics['margin']:.3f}, Violations: {val_metrics['margin_violations']:.3f}")
                else:
                    print(f"   Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.3f}, "
                          f"Gap: {val_metrics['similarity_gap']:.3f}, Mag: {val_metrics['magnitude']:.3f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"🌟 New best validation loss: {val_loss:.4f}")
                
                # Task-specific final metrics
                if self.is_ranking:
                    ranking_metrics = self.quick_ranking_check(val_loader, max_batches=5)
                    print(f"🎯 Ranking - NDCG@5: {ranking_metrics.get('ndcg_at_5', 0):.3f}, "
                          f"NDCG@10: {ranking_metrics.get('ndcg_at_10', 0):.3f}, "
                          f"MRR: {ranking_metrics.get('mrr', 0):.3f}")
                    
                    # Log to WandB
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_metrics['accuracy'],
                        "train_margin": train_metrics['margin'],
                        "train_margin_violations": train_metrics['margin_violations'],
                        "val_loss": val_loss,
                        "val_accuracy": val_metrics['accuracy'],
                        "val_margin": val_metrics['margin'],
                        "val_margin_violations": val_metrics['margin_violations'],
                        **{f"epoch_{k}": v for k, v in ranking_metrics.items()}
                    })
                else:
                    recall_metrics = self.quick_recall_check(val_loader, max_batches=5)
                    print(f"🎯 Recall - R@5: {recall_metrics.get('recall_at_5', 0):.3f}, "
                          f"R@10: {recall_metrics.get('recall_at_10', 0):.3f}")
                    
                    # Log to WandB
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_metrics['accuracy'],
                        "train_similarity_gap": train_metrics['similarity_gap'],
                        "train_magnitude": train_metrics['magnitude'],
                        "val_loss": val_loss,
                        "val_accuracy": val_metrics['accuracy'],
                        "val_similarity_gap": val_metrics['similarity_gap'],
                        "val_magnitude": val_metrics['magnitude'],
                        **{f"epoch_{k}": v for k, v in recall_metrics.items()}
                    })
            else:
                # Training only logging
                if self.is_ranking:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_metrics['accuracy'],
                        "train_margin": train_metrics['margin'],
                        "train_margin_violations": train_metrics['margin_violations']
                    })
                else:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_accuracy": train_metrics['accuracy'],
                        "train_similarity_gap": train_metrics['similarity_gap'],
                        "train_magnitude": train_metrics['magnitude']
                    })
            
            clean_memory()
        
        total_time = time.time() - start_time
        print(f"\n🎉 {task_name} training completed in {total_time/60:.1f} minutes!")
        print(f"Final train loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


class TrainerFactory:
    """Simple factory for creating trainers."""
    
    @staticmethod
    def create_trainer(config: Dict, model: TwoTowerModel, device: torch.device):
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('LR', 0.001))
        
        # Get loss function
        loss_function = ModelFactory.get_loss_function(
            loss_type=config.get('LOSS_TYPE', 'triplet'),
            margin=config.get('MARGIN', 1.0)
        )
        
        return TwoTowerTrainer(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            config=config
        ) 