import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class RNNEncoder(nn.Module):
    """RNN Encoder for text encoding."""
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        pretrained_embeddings: Optional[np.ndarray] = None,
        rnn_type: str = 'GRU',
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        freeze_embeddings: bool = False,
        fine_tune_embeddings: bool = True
    ):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Load pretrained embeddings
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # For GloVe embeddings, normalize them for better training stability
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)
        
        # Control embedding training
        if freeze_embeddings or not fine_tune_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Embedding dropout for regularization
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # RNN layer with optimized dimensions for GloVe
        if rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, 
                num_layers=num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False  # Keep unidirectional for efficiency
            )
        elif rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, 
                num_layers=num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        else:
            self.rnn = nn.RNN(
                embed_dim, hidden_dim, 
                num_layers=num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        
        self.rnn_type = rnn_type.upper()
        self.hidden_dim = hidden_dim
        
        # Additional normalization layer for better stability with GloVe
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        # RNN
        if self.rnn_type == 'LSTM':
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        
        # Use last layer hidden state
        hidden = h_n[-1]
        
        # Apply layer normalization for stability
        hidden = self.layer_norm(hidden)
        
        # L2 normalize
        return F.normalize(hidden, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """Two-tower model with separate encoders for queries and documents."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pretrained_embeddings: Optional[np.ndarray] = None,
        **encoder_kwargs
    ):
        super().__init__()
        
        # Print once when loading embeddings
        if pretrained_embeddings is not None:
            print(f"Loading pretrained embeddings: {pretrained_embeddings.shape}")
            print(f"GloVe embedding dimension: {embed_dim}")
        
        self.query_encoder = RNNEncoder(
            vocab_size, embed_dim, hidden_dim, 
            pretrained_embeddings, **encoder_kwargs
        )
        
        self.doc_encoder = RNNEncoder(
            vocab_size, embed_dim, hidden_dim, 
            pretrained_embeddings, **encoder_kwargs
        )
        
        # Initialize RNN weights properly for GloVe
        self._init_rnn_weights()
        
    def _init_rnn_weights(self):
        """Initialize RNN weights using Xavier initialization for better training with GloVe."""
        for encoder in [self.query_encoder, self.doc_encoder]:
            for name, param in encoder.rnn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(query)
    
    def encode_document(self, document: torch.Tensor) -> torch.Tensor:
        return self.doc_encoder(document)
    
    def forward(self, query: torch.Tensor, document: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_query(query), self.encode_document(document)


def triplet_loss(triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], margin: float = 1.0) -> torch.Tensor:
    """Optimized triplet loss for GloVe embeddings with cosine similarity."""
    query, pos_doc, neg_doc = triplet
    
    # Use cosine similarity instead of Euclidean distance for normalized embeddings
    pos_sim = F.cosine_similarity(query, pos_doc, dim=1)
    neg_sim = F.cosine_similarity(query, neg_doc, dim=1)
    
    # Convert to distances (1 - similarity)
    pos_dist = 1 - pos_sim
    neg_dist = 1 - neg_sim
    
    return torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()


class ModelFactory:
    """Simple factory for creating models."""
    
    @staticmethod
    def create_two_tower_model(config: dict, pretrained_embeddings: Optional[np.ndarray] = None) -> TwoTowerModel:
        return TwoTowerModel(
            vocab_size=config.get('VOCAB_SIZE'),
            embed_dim=config.get('EMBED_DIM'),
            hidden_dim=config.get('HIDDEN_DIM'),
            pretrained_embeddings=pretrained_embeddings,
            rnn_type=config.get('RNN_TYPE'),
            num_layers=config.get('NUM_LAYERS'),
            dropout=config.get('DROPOUT'),
            embedding_dropout=config.get('EMBEDDING_DROPOUT', 0.0),
            freeze_embeddings=config.get('FREEZE_EMBEDDINGS', False),
            fine_tune_embeddings=config.get('FINE_TUNE_EMBEDDINGS', True)
        )
    
    @staticmethod
    def get_loss_function(loss_type: str = 'triplet', margin: float = 1.0):
        return lambda triplet: triplet_loss(triplet, margin) 