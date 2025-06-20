import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


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
        bidirectional: bool = False,
        normalize_output: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = False  # Prevent updates to the embeddings
        
        self.bidirectional = bidirectional
        rnn_class = getattr(nn, rnn_type.upper())
        self.rnn = rnn_class(
            embed_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.rnn_type = rnn_type.upper()
        self.normalize_output = normalize_output
        
        # If bidirectional, we need to project the concatenated hidden states
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        
        # Get actual sequence lengths (non-zero tokens)
        lengths = (x != 0).sum(dim=1).cpu()
        
        # Pack sequences for efficient RNN processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        if self.rnn_type == 'LSTM':
            packed_output, (h_n, _) = self.rnn(packed)
        else: # GRU or RNN
            packed_output, h_n = self.rnn(packed)
        
        # Use the hidden state from the last layer
        if self.bidirectional:
            # Concatenate forward and backward hidden states from the last layer
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            # Project back to original hidden dimension
            hidden = self.projection(hidden)
        else:
            hidden = h_n[-1]
        
        if self.normalize_output:
            return F.normalize(hidden, p=2, dim=1)
        return hidden


class TwoTowerModel(nn.Module):
    """A two-tower model using RNNEncoder."""
    
    def __init__(self, config: Dict, pretrained_embeddings: Optional[np.ndarray] = None):
        super().__init__()
        
        encoder_args = {
            'vocab_size': config['VOCAB_SIZE'],
            'embed_dim': config['EMBED_DIM'],
            'hidden_dim': config['HIDDEN_DIM'],
            'pretrained_embeddings': pretrained_embeddings,
            'rnn_type': config.get('RNN_TYPE', 'GRU'),
            'num_layers': config.get('NUM_LAYERS', 1),
            'dropout': config.get('DROPOUT', 0.0),
            'bidirectional': config.get('BIDIRECTIONAL', False),
            'normalize_output': config.get('NORMALIZE_OUTPUT', True)
        }
        
        self.query_encoder = RNNEncoder(**encoder_args)
        self.doc_encoder = RNNEncoder(**encoder_args)
        
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(query)
    
    def encode_document(self, document: torch.Tensor) -> torch.Tensor:
        return self.doc_encoder(document)
    
    def forward(self, query: torch.Tensor, document: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_query(query), self.encode_document(document)


def triplet_loss_cosine(triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], margin: float = 0.2) -> torch.Tensor:
    """Triplet loss using cosine similarity. Assumes embeddings are normalized."""
    query, pos_doc, neg_doc = triplet
    pos_sim = F.cosine_similarity(query, pos_doc)
    neg_sim = F.cosine_similarity(query, neg_doc)
    return torch.clamp(neg_sim - pos_sim + margin, min=0.0).mean() 