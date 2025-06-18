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
            'SHARED_ENCODER': self.config.get('SHARED_ENCODER', False),
            'RNN_TYPE': self.config.get('RNN_TYPE', 'GRU'),
            'NUM_LAYERS': self.config.get('NUM_LAYERS', 1),
            'DROPOUT': self.config.get('DROPOUT', 0.0)
        }
        
        self.model = ModelFactory.create_two_tower_model(model_config, pretrained_embeddings)
        self.model.to(self.device)
        
        # Load trained model
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query."""
        with torch.no_grad():
            # Tokenize
            token_ids = self.tokenizer.encode(query)
            tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Get embedding
            embedding = self.model.encode_query(tokens)
            return embedding.cpu().numpy().squeeze(0)


if __name__ == "__main__":
    inferencer = QueryInferencer()
    
    # Test
    query = "machine learning"
    embedding = inferencer.get_query_embedding(query)
    print(f"Query: '{query}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}") 