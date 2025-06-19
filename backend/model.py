import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class RNNEncoder(nn.Module):
    """A much simpler RNN encoder that keeps only the essentials."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pretrained_embeddings: Optional[np.ndarray] = None,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.0,               # kept for API-compat but unused
        embedding_dropout: float = 0.0,     # kept for API-compat but unused
        freeze_embeddings: bool = False,
        fine_tune_embeddings: bool = True,
        normalize_pretrained_embeddings: bool = True,
        normalize_output: bool = True,
        **unused_kwargs,                    # absorb any extra args passed by old configs
    ):
        super().__init__()

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Load optional pretrained weights
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if normalize_pretrained_embeddings:
                self.embedding.weight.data = F.normalize(
                    self.embedding.weight.data, p=2, dim=1
                )

        # Optionally freeze embeddings
        if freeze_embeddings or not fine_tune_embeddings:
            self.embedding.weight.requires_grad_(False)

        # Very small RNN stack – default 1-layer, unidirectional
        rnn_type = rnn_type.upper()
        if rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.rnn_type = rnn_type
        self.normalize_output = normalize_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # RNN returns hidden states; we only care about the last layer's final hidden state
        if self.rnn_type == "LSTM":
            _, (h_n, _) = self.rnn(x)  # h_n shape: (num_layers, batch, hidden_dim)
        else:
            _, h_n = self.rnn(x)

        # Take last layer hidden state and normalize if requested
        output = h_n[-1]
        if self.normalize_output:
            output = F.normalize(output, p=2, dim=1)
        return output


class TwoTowerModel(nn.Module):
    """A lightweight two-tower retrieval model with independent encoders."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pretrained_embeddings: Optional[np.ndarray] = None,
        **encoder_kwargs,
    ) -> None:
        super().__init__()

        # Build the twin encoders
        self.query_encoder = RNNEncoder(
            vocab_size,
            embed_dim,
            hidden_dim,
            pretrained_embeddings,
            **encoder_kwargs,
        )
        self.doc_encoder = RNNEncoder(
            vocab_size,
            embed_dim,
            hidden_dim,
            pretrained_embeddings,
            **encoder_kwargs,
        )

    # Convenience wrappers
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(query)

    def encode_document(self, document: torch.Tensor) -> torch.Tensor:
        return self.doc_encoder(document)

    def forward(
        self, query: torch.Tensor, document: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


def triplet_loss_euclidean(triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], margin: float = 1.0) -> torch.Tensor:
    """Triplet loss using Euclidean distance for normalized embeddings."""
    query, pos_doc, neg_doc = triplet
    
    # Euclidean distances
    pos_dist = torch.norm(query - pos_doc, p=2, dim=1)
    neg_dist = torch.norm(query - neg_doc, p=2, dim=1)
    
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
            fine_tune_embeddings=config.get('FINE_TUNE_EMBEDDINGS', True),
            normalize_pretrained_embeddings=config.get('NORMALIZE_PRETRAINED_EMBEDDINGS', True),
            normalize_output=config.get('NORMALIZE_OUTPUT', True)
        )
    
    @staticmethod
    def get_loss_function(loss_type: str = 'triplet', margin: float = 1.0):
        if loss_type == 'triplet_euclidean':
            return lambda triplet: triplet_loss_euclidean(triplet, margin)
        else:  # default to cosine-based triplet loss
            return lambda triplet: triplet_loss(triplet, margin) 