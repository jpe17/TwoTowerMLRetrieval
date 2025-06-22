import torch
import numpy as np
import json
from pathlib import Path
import sys
import json

def load_config(path: str):
    """Loads a JSON config file."""
    with open(path, 'r') as f:
        return json.load(f)

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent))
config = load_config('frontend/config.json')

from tokenizer import PretrainedTokenizer
from model import TwoTowerModel

class QueryInferencer:
    """Handles loading a trained model and tokenizer from an artifacts directory to perform inference."""
    
    def __init__(self, artifacts_path: str, device: torch.device = None):
        """
        Initializes the inferencer by loading all required artifacts from a specified run directory.
        
        Args:
            artifacts_path: Path to the directory containing training artifacts 
                            (model.pth, config.json, word_to_idx.pkl).
            device: The torch device to run the model on.
        """
        print(f"🔎 Initializing QueryInferencer from artifacts: {artifacts_path}")
        self.artifacts_path = Path(artifacts_path)
        self.device = device or self._get_best_device()
        
        # Load configuration
        with open(self.artifacts_path / 'config.json', 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = PretrainedTokenizer(str(self.artifacts_path / 'word_to_idx.pkl'))
        
        # Add vocab size and embed_dim to config for model creation
        self.config['VOCAB_SIZE'] = self.tokenizer.vocab_size()
        # This info is not in the config, but required for model init.
        # We can determine it from the saved embeddings if they exist or set a default.
        if not 'EMBED_DIM' in self.config:
            self.config['EMBED_DIM'] = 200 # A common default for GloVe

        # Create model architecture
        self.model = TwoTowerModel(self.config, pretrained_embeddings=None).to(self.device)
        
        # Load the trained model weights
        model_path = self.artifacts_path / 'model.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✅ Model, config, and tokenizer loaded successfully.")

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generates an embedding for a given text query."""
        with torch.no_grad():
            # Tokenize and convert to tensor
            token_ids = self.tokenizer.encode(query)

            # If the query has no words in our vocabulary, return a zero vector.
            if not token_ids:
                print(f"⚠️ Query '{query}' contains no known tokens. Returning a zero vector.")
                hidden_dim = self.config.get('HIDDEN_DIM', 128)
                return np.zeros(hidden_dim, dtype=np.float32)
            
            tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Generate embedding
            embedding = self.model.encode_query(tokens)
            return embedding.cpu().numpy().squeeze(0)

    def _get_best_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

if __name__ == '__main__':
    # --- Example Usage ---
    ARTIFACTS_PATH = config['ARTIFACTS_PATH']
    
    if not Path(ARTIFACTS_PATH).exists():
        print(f"Error: Artifacts directory not found at '{ARTIFACTS_PATH}'")
        print("Please run a training first (python backend/main.py) and update the path.")
    else:
        inferencer = QueryInferencer(ARTIFACTS_PATH)
        
        test_query = "what is machine learning"
        embedding = inferencer.get_query_embedding(test_query)
        
        print(f"\nQuery: '{test_query}'")
        print(f"Embedding Shape: {embedding.shape}")
        print(f"Embedding L2 Norm: {np.linalg.norm(embedding):.4f}")
        print("Embedding (first 5 values):", embedding[:5]) 