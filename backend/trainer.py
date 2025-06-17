import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from model import TwoTowerModel, ModelFactory
from utils import clean_memory, get_memory_usage
import wandb

class TwoTowerTrainer:
    """Trainer class for the Two-Tower model with comprehensive training, validation, and testing."""
    
    def __init__(
        self, 
        model: TwoTowerModel, 
        optimizer: torch.optim.Optimizer,
        loss_function,
        device: torch.device,
        config: Dict
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The two-tower model to train
            optimizer: Optimizer for training
            loss_function: Loss function to use
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.config = config
        
        # Training configuration
        self.gradient_accumulation_steps = config.get('GRADIENT_ACCUMULATION_STEPS', 1)
        self.memory_cleanup_frequency = config.get('MEMORY_CLEANUP_FREQUENCY', 100)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        accumulation_loss = 0
        
        epoch_start = time.time()
        
        for batch_idx, (query_batch, pos_batch, neg_batch) in enumerate(train_loader):
            try:
                # Move tensors to device
                query_batch = query_batch.to(self.device, non_blocking=True)
                pos_batch = pos_batch.to(self.device, non_blocking=True)
                neg_batch = neg_batch.to(self.device, non_blocking=True)
                
                # Forward pass
                q_vec = self.model.encode_query(query_batch)
                pos_vec = self.model.encode_document(pos_batch)
                neg_vec = self.model.encode_document(neg_batch)

                loss = self.loss_function((q_vec, pos_vec, neg_vec))
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                accumulation_loss += loss.item()
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += accumulation_loss * self.gradient_accumulation_steps
                    accumulation_loss = 0
                
                num_batches += 1
                
                # Progress indicator and memory management
                if num_batches % 50 == 0:
                    current_loss = loss.item() * self.gradient_accumulation_steps
                    print(f"  Epoch {epoch+1}, Batch {num_batches}/{len(train_loader)}, Loss: {current_loss:.4f}")
                
                # Periodic memory cleanup
                if num_batches % self.memory_cleanup_frequency == 0:
                    clean_memory()
                    
                # Clear intermediate tensors
                del query_batch, pos_batch, neg_batch, q_vec, pos_vec, neg_vec, loss
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Memory error at batch {num_batches}: {str(e)}")
                    print("🧹 Attempting memory cleanup and continuing...")
                    clean_memory()
                    self.optimizer.zero_grad()
                    continue
                else:
                    raise e
        
        # Handle any remaining accumulated gradients
        if accumulation_loss > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulation_loss * self.gradient_accumulation_steps
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)
        
        print(f"✅ Epoch {epoch+1} completed: Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})  # <--- Add this line
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (query_batch, pos_batch, neg_batch) in enumerate(val_loader):
                try:
                    # Move tensors to device
                    query_batch = query_batch.to(self.device, non_blocking=True)
                    pos_batch = pos_batch.to(self.device, non_blocking=True)
                    neg_batch = neg_batch.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    q_vec = self.model.encode_query(query_batch)
                    pos_vec = self.model.encode_document(pos_batch)
                    neg_vec = self.model.encode_document(neg_batch)

                    loss = self.loss_function((q_vec, pos_vec, neg_vec))
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clear intermediate tensors
                    del query_batch, pos_batch, neg_batch, q_vec, pos_vec, neg_vec, loss
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"❌ Validation memory error at batch {num_batches}")
                        clean_memory()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"📊 Validation - Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        wandb.log({"val_loss": avg_loss, "epoch": epoch + 1}) 
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_state = {
                'query_encoder_state_dict': self.model.query_encoder.state_dict(),
                'doc_encoder_state_dict': self.model.doc_encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_loss
            }
            print(f"🌟 New best validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Complete training loop with validation.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs (uses config if not provided)
            
        Returns:
            Dictionary with training history
        """
        if epochs is None:
            epochs = self.config.get('EPOCHS', 10)
        
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"   📊 Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"   🧹 Memory cleanup every {self.memory_cleanup_frequency} batches")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader, epoch)
                self.val_losses.append(val_loss)
            
            # Memory cleanup after each epoch
            clean_memory()
            print(f"   Memory after epoch: {get_memory_usage()}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed! Total time: {total_time/60:.1f} minutes")
        print(f"Final training loss: {self.train_losses[-1]:.4f}")
        if self.val_losses:
            print(f"Final validation loss: {self.val_losses[-1]:.4f}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test metrics dictionary
        """
        print("🧪 Testing model...")
        
        # Use best model if available
        if self.best_model_state is not None:
            print("📥 Loading best model from validation...")
            self.model.query_encoder.load_state_dict(self.best_model_state['query_encoder_state_dict'])
            self.model.doc_encoder.load_state_dict(self.best_model_state['doc_encoder_state_dict'])
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (query_batch, pos_batch, neg_batch) in enumerate(test_loader):
                try:
                    # Move tensors to device
                    query_batch = query_batch.to(self.device, non_blocking=True)
                    pos_batch = pos_batch.to(self.device, non_blocking=True)
                    neg_batch = neg_batch.to(self.device, non_blocking=True)
                    
                    # Forward pass
                    q_vec = self.model.encode_query(query_batch)
                    pos_vec = self.model.encode_document(pos_batch)
                    neg_vec = self.model.encode_document(neg_batch)

                    loss = self.loss_function((q_vec, pos_vec, neg_vec))
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clear intermediate tensors
                    del query_batch, pos_batch, neg_batch, q_vec, pos_vec, neg_vec, loss
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"❌ Test memory error at batch {num_batches}")
                        clean_memory()
                        continue
                    else:
                        raise e
        
        avg_test_loss = total_loss / max(num_batches, 1)
        
        print(f"🏁 Test Results:")
        print(f"   Average Test Loss: {avg_test_loss:.4f}")
        
        return {
            'test_loss': avg_test_loss,
            'num_test_batches': num_batches
        }
    
    def get_model_for_inference(self):
        """Get the best model for inference."""
        if self.best_model_state is not None:
            # Load best model state
            model_copy = TwoTowerModel(
                vocab_size=self.config['VOCAB_SIZE'],
                embed_dim=self.config['EMBED_DIM'],
                hidden_dim=self.config['HIDDEN_DIM']
            ).to(self.device)
            
            model_copy.query_encoder.load_state_dict(self.best_model_state['query_encoder_state_dict'])
            model_copy.doc_encoder.load_state_dict(self.best_model_state['doc_encoder_state_dict'])
            return model_copy
        else:
            return self.model


class TrainerFactory:
    """Factory class for creating trainers with different configurations."""
    
    @staticmethod
    def create_trainer(
        config: Dict,
        model: TwoTowerModel,
        device: torch.device
    ) -> TwoTowerTrainer:
        """
        Create a trainer based on configuration.
        
        Args:
            config: Configuration dictionary
            model: The model to train
            device: Device to train on
            
        Returns:
            Configured trainer instance
        """
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get('LR', 0.001)
        )
        
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