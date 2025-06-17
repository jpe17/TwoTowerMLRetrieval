import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable


class RNNEncoder(nn.Module):
    """RNN Encoder for text encoding in the two-tower architecture."""
    
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
        """
        Initialize the RNN encoder.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension of RNN
            pretrained_embeddings: Optional pretrained embeddings matrix
            rnn_type: Type of RNN ('GRU', 'LSTM', 'RNN')
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Use padding_idx=0 since we pad with 0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            print(f"Loading pretrained embeddings: {pretrained_embeddings.shape}")
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Keep embeddings trainable (they are by default)
        
        # Choose RNN type
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
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Encoded representation of shape (batch_size, hidden_dim)
        """
        # Embedding layer
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # RNN layer
        if self.rnn_type == 'LSTM':
            _, (h_n, _) = self.rnn(x)  # h_n: (num_layers, batch_size, hidden_dim)
        else:
            _, h_n = self.rnn(x)  # h_n: (num_layers, batch_size, hidden_dim)
        
        # Use the last layer's hidden state
        return h_n[-1]  # (batch_size, hidden_dim)


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
        """
        Initialize the two-tower model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension of encoders
            pretrained_embeddings: Optional pretrained embeddings
            shared_encoder: Whether to share weights between query and doc encoders
            **encoder_kwargs: Additional arguments for RNNEncoder
        """
        super().__init__()
        
        self.query_encoder = RNNEncoder(
            vocab_size, embed_dim, hidden_dim, 
            pretrained_embeddings, **encoder_kwargs
        )
        
        if shared_encoder:
            self.doc_encoder = self.query_encoder
        else:
            self.doc_encoder = RNNEncoder(
                vocab_size, embed_dim, hidden_dim, 
                pretrained_embeddings, **encoder_kwargs
            )
        
        self.shared_encoder = shared_encoder
    
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        """Encode query text."""
        return self.query_encoder(query)
    
    def encode_document(self, document: torch.Tensor) -> torch.Tensor:
        """Encode document text."""
        return self.doc_encoder(document)
    
    def forward(self, query: torch.Tensor, document: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both query and document.
        
        Args:
            query: Query tensor
            document: Document tensor
            
        Returns:
            Tuple of (query_embedding, document_embedding)
        """
        return self.encode_query(query), self.encode_document(document)


def triplet_loss_function(
    triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
    distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    margin: float
) -> torch.Tensor:
    """
    Triplet loss function for training.
    
    Args:
        triplet: Tuple of (query, positive_doc, negative_doc) embeddings
        distance_function: Function to compute distance between embeddings
        margin: Margin for triplet loss
        
    Returns:
        Triplet loss value
    """
    query, pos_doc, neg_doc = triplet
    d_pos = distance_function(query, pos_doc)
    d_neg = distance_function(query, neg_doc)
    return torch.clamp(d_pos - d_neg + margin, min=0.0).mean()


def cosine_similarity_loss(
    triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
    margin: float = 0.1
) -> torch.Tensor:
    """
    Cosine similarity-based triplet loss.
    
    Args:
        triplet: Tuple of (query, positive_doc, negative_doc) embeddings
        margin: Margin for the loss
        
    Returns:
        Loss value
    """
    query, pos_doc, neg_doc = triplet
    
    # Cosine similarity (higher is better)
    sim_pos = F.cosine_similarity(query, pos_doc, dim=1)
    sim_neg = F.cosine_similarity(query, neg_doc, dim=1)
    
    # We want sim_pos > sim_neg + margin
    return torch.clamp(margin - sim_pos + sim_neg, min=0.0).mean()


class ModelFactory:
    """Factory class for creating models with different configurations."""
    
    @staticmethod
    def create_two_tower_model(config: dict, pretrained_embeddings: Optional[np.ndarray] = None) -> TwoTowerModel:
        """
        Create a two-tower model based on configuration.
        
        Args:
            config: Configuration dictionary
            pretrained_embeddings: Optional pretrained embeddings
            
        Returns:
            Initialized TwoTowerModel
        """
        return TwoTowerModel(
            vocab_size=config.get('VOCAB_SIZE'),
            embed_dim=config.get('EMBED_DIM'),
            hidden_dim=config.get('HIDDEN_DIM', 128),
            pretrained_embeddings=pretrained_embeddings,
            shared_encoder=config.get('SHARED_ENCODER', False),
            rnn_type=config.get('RNN_TYPE', 'GRU'),
            num_layers=config.get('NUM_LAYERS', 1),
            dropout=config.get('DROPOUT', 0.0)
        )
    
    @staticmethod
    def get_loss_function(loss_type: str = 'triplet', margin: float = 1.0):
        """
        Get loss function based on type.
        
        Args:
            loss_type: Type of loss ('triplet', 'cosine')
            margin: Margin for the loss
            
        Returns:
            Loss function
        """
        if loss_type.lower() == 'triplet':
            return lambda triplet: triplet_loss_function(triplet, F.pairwise_distance, margin)
        elif loss_type.lower() == 'cosine':
            return lambda triplet: cosine_similarity_loss(triplet, margin)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}") 