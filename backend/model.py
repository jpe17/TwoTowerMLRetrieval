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
        dropout: float = 0.0
    ):
        super().__init__()
        
        print(f"----> vocab_size: {vocab_size}, embed_dim: {embed_dim}")
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Load pretrained embeddings
        if pretrained_embeddings is not None:
            print(f"pretrained_embeddings shape: {pretrained_embeddings.shape}")
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # RNN layer
        if rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, 
                num_layers=num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, 
                num_layers=num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.RNN(
                embed_dim, hidden_dim, 
                num_layers=num_layers,
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )
        
        self.rnn_type = rnn_type.upper()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.embedding(x)
        
        # RNN
        if self.rnn_type == 'LSTM':
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        
        # Use last layer hidden state and L2 normalize
        hidden = h_n[-1]
        return F.normalize(hidden, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """Two-tower model with separate encoders for queries and documents."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pretrained_embeddings: Optional[np.ndarray] = None,
        shared_encoder: bool = False,
        **encoder_kwargs
    ):
        super().__init__()
        
        # Print once when loading embeddings
        if pretrained_embeddings is not None:
            print(f"Loading pretrained embeddings: {pretrained_embeddings.shape}")
        
        self.query_encoder = RNNEncoder(
            vocab_size, embed_dim, hidden_dim, 
            pretrained_embeddings, **encoder_kwargs
        )
        
        self.doc_encoder = RNNEncoder(
            vocab_size, embed_dim, hidden_dim, 
            pretrained_embeddings, **encoder_kwargs
            )
        
        self.shared_encoder = shared_encoder
    
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(query)
    
    def encode_document(self, document: torch.Tensor) -> torch.Tensor:
        return self.doc_encoder(document)
    
    def forward(self, query: torch.Tensor, document: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_query(query), self.encode_document(document)


def triplet_loss(triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], margin: float = 1.0) -> torch.Tensor:
    """Simple triplet loss using pairwise distance."""
    query, pos_doc, neg_doc = triplet
    d_pos = F.pairwise_distance(query, pos_doc)
    d_neg = F.pairwise_distance(query, neg_doc)
    return torch.clamp(d_pos - d_neg + margin, min=0.0).mean()


class ModelFactory:
    """Simple factory for creating models."""
    
    @staticmethod
    def create_two_tower_model(config: dict, pretrained_embeddings: Optional[np.ndarray] = None) -> TwoTowerModel:
        return TwoTowerModel(
            vocab_size=config.get('VOCAB_SIZE'),
            embed_dim=config.get('EMBED_DIM'),
            hidden_dim=config.get('HIDDEN_DIM'),
            pretrained_embeddings=pretrained_embeddings,
            shared_encoder=config.get('SHARED_ENCODER', False),
            rnn_type=config.get('RNN_TYPE', 'GRU'),
            num_layers=config.get('NUM_LAYERS', 1),
            dropout=config.get('DROPOUT', 0.0)
        )
    
    @staticmethod
    def get_loss_function(loss_type: str = 'triplet', margin: float = 1.0):
        return lambda triplet: triplet_loss(triplet, margin) 