import torch
import numpy as np
import json
from typing import Optional
from tokenizer import PretrainedTokenizer
from model import ModelFactory


class QueryInferencer:
    """Minimal query inferencer for getting embeddings."""
    
    def __init__(self, 
                 config_path: str = "backend/config.json",
                 model_path: Optional[str] = None):
        """Initialize with model, tokenizer, and embeddings."""
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Define max sequence length for padding/truncation
        self.max_seq_len = 32
        
        # Set device
        self.device = self.config.get('DEVICE', 'cpu')
        
        # Load tokenizer
        self.tokenizer = PretrainedTokenizer(self.config['WORD_TO_IDX_PATH'])
        
        # Load pretrained embeddings
        pretrained_embeddings = None
        if self.config.get('EMBEDDINGS_PATH'):
            pretrained_embeddings = np.load(self.config['EMBEDDINGS_PATH'])
        
        # Create model
        model_config = {
            'VOCAB_SIZE': self.tokenizer.vocab_size(),
            'EMBED_DIM': pretrained_embeddings.shape[1] if pretrained_embeddings is not None else 300,
            'HIDDEN_DIM': self.config.get('HIDDEN_DIM', 32),
            'RNN_TYPE': self.config.get('RNN_TYPE', 'GRU'),
            'NUM_LAYERS': self.config.get('NUM_LAYERS', 1),
            'DROPOUT': self.config.get('DROPOUT', 0.0)
        }
        
        self.model = ModelFactory.create_two_tower_model(model_config, pretrained_embeddings)
        self.model.to(self.device)
        
        # Load trained model
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Case 1: The checkpoint is the model object itself
            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
            # Case 2: The checkpoint is a state dictionary
            else:
                if 'query_encoder_state_dict' in checkpoint and 'doc_encoder_state_dict' in checkpoint:
                    self.model.query_encoder.load_state_dict(checkpoint['query_encoder_state_dict'])
                    self.model.doc_encoder.load_state_dict(checkpoint['doc_encoder_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model_state' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state'])
                else:
                    self.model.load_state_dict(checkpoint)
            
            # Ensure backward compatibility for loaded models
            if hasattr(self.model, 'query_encoder') and hasattr(self.model.query_encoder, '_ensure_backward_compatibility'):
                self.model.query_encoder._ensure_backward_compatibility()
            if hasattr(self.model, 'doc_encoder') and hasattr(self.model.doc_encoder, '_ensure_backward_compatibility'):
                self.model.doc_encoder._ensure_backward_compatibility()
        
        self.model.eval()
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query."""
        with torch.no_grad():
            # Tokenize and truncate
            token_ids = self.tokenizer.encode(query)[:self.max_seq_len]
            
            # Left pad if necessary
            if len(token_ids) < self.max_seq_len:
                padding = [0] * (self.max_seq_len - len(token_ids))
                token_ids = padding + token_ids
            
            tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Get embedding
            embedding = self.model.encode_query(tokens)
            return embedding.cpu().numpy().squeeze(0)


if __name__ == "__main__":
    # To test this, you need a trained model artifact
    # Example: artifacts/two_tower_run_20250619_140401/model_epoch_10.pt
    model_path = "artifacts/two_tower_run_20250619_163538/full_model.pth"
    inferencer = QueryInferencer(model_path=model_path)
    
    # Test
    query = "machine learning"
    embedding = inferencer.get_query_embedding(query)
    print(f"Query: '{query}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"Full embedding:")
    print(embedding) 