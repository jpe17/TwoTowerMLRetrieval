import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MergedTripletModel(nn.Module):
    """Single unified model that processes query, positive, and negative documents together."""
    
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
        
        # Single shared encoder for all inputs
        self.shared_encoder = RNNEncoder(
            vocab_size, embed_dim, hidden_dim, 
            pretrained_embeddings, rnn_type, num_layers, dropout
        )
        
        print(f"🔄 Created merged model with shared encoder")
    
    def forward(self, query: torch.Tensor, pos_doc: torch.Tensor, neg_doc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process all three inputs through the same encoder.
        
        Args:
            query: Query tensor [batch_size, seq_len]
            pos_doc: Positive document tensor [batch_size, seq_len]  
            neg_doc: Negative document tensor [batch_size, seq_len]
            
        Returns:
            Tuple of (query_embedding, pos_embedding, neg_embedding)
        """
        # Process all through the same encoder
        query_emb = self.shared_encoder(query)
        pos_emb = self.shared_encoder(pos_doc)
        neg_emb = self.shared_encoder(neg_doc)
        
        return query_emb, pos_emb, neg_emb
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode any text through the shared encoder."""
        return self.shared_encoder(text)


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
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Load pretrained embeddings
        if pretrained_embeddings is not None:
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


def triplet_loss(triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], margin: float = 1.0) -> torch.Tensor:
    """Same triplet loss as before."""
    query, pos_doc, neg_doc = triplet
    d_pos = F.pairwise_distance(query, pos_doc)
    d_neg = F.pairwise_distance(query, neg_doc)
    return torch.clamp(d_pos - d_neg + margin, min=0.0).mean() 