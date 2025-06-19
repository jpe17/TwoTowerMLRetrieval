import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    """RNN encoder that properly handles variable-length sequences."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pretrained_embeddings: Optional[np.ndarray] = None,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        freeze_embeddings: bool = False,
        fine_tune_embeddings: bool = True,
        normalize_pretrained_embeddings: bool = True,
        normalize_output: bool = True,
        pooling_strategy: str = "last",  # "last", "mean", "max"
        **unused_kwargs,
    ):
        super().__init__()

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(embedding_dropout) if embedding_dropout > 0 else None

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

        # RNN with dropout
        rnn_type = rnn_type.upper()
        if rnn_type == "GRU":
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, num_layers=num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, num_layers=num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.RNN(
                embed_dim, hidden_dim, num_layers=num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )

        self.rnn_type = rnn_type
        self.normalize_output = normalize_output
        self.pooling_strategy = pooling_strategy
        
        # Output dropout
        self.output_dropout = nn.Dropout(dropout) if dropout > 0 else None

    def get_sequence_lengths(self, x: torch.Tensor) -> torch.Tensor:
        """Get actual sequence lengths (excluding padding tokens)."""
        # For right padding, count non-zero tokens
        mask = (x != 0).long()
        lengths = mask.sum(dim=1)
        return lengths.clamp(min=1)  # Ensure at least length 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        # Get actual sequence lengths
        lengths = self.get_sequence_lengths(x)
        
        # Embed tokens: (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(x)
        
        # Apply embedding dropout
        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)
        
        # Process through RNN
        if self.rnn_type == "LSTM":
            output, (h_n, c_n) = self.rnn(x)  # output: (batch, seq_len, hidden_dim)
        else:
            output, h_n = self.rnn(x)  # output: (batch, seq_len, hidden_dim)
        
        # Apply pooling strategy
        if self.pooling_strategy == "mean":
            # Mean pooling over valid tokens only
            mask = (torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            final_output = (output * mask).sum(dim=1) / lengths.unsqueeze(1).float()
        elif self.pooling_strategy == "max":
            # Max pooling over valid tokens only
            mask = (torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
            output_masked = output.clone()
            output_masked[~mask] = float('-inf')
            final_output = output_masked.max(dim=1)[0]
        else:  # "last" (default)
            # Get the last valid output for each sequence
            batch_indices = torch.arange(batch_size, device=x.device)
            last_indices = (lengths - 1).clamp(min=0)
            final_output = output[batch_indices, last_indices]
        
        # Apply output dropout
        if self.output_dropout is not None:
            final_output = self.output_dropout(final_output)
        
        # Normalize if requested
        if self.normalize_output:
            final_output = F.normalize(final_output, p=2, dim=1)
        
        return final_output


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
    """Robust triplet loss with better numerical stability."""
    query, pos_doc, neg_doc = triplet
    
    # Use cosine similarity instead of Euclidean distance for normalized embeddings
    pos_sim = F.cosine_similarity(query, pos_doc, dim=1)
    neg_sim = F.cosine_similarity(query, neg_doc, dim=1)
    
    # Convert to distances (1 - similarity)
    pos_dist = 1 - pos_sim
    neg_dist = 1 - neg_sim
    
    # Triplet loss with better numerical stability
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    
    # Add some statistics for monitoring
    with torch.no_grad():
        hard_negatives = (loss > 0).float().mean()
        avg_pos_sim = pos_sim.mean()
        avg_neg_sim = neg_sim.mean()
    
    return loss.mean()


def adaptive_triplet_loss(triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], margin: float = 1.0, alpha: float = 0.1) -> torch.Tensor:
    """Adaptive triplet loss that focuses on hard examples and prevents collapse."""
    query, pos_doc, neg_doc = triplet
    
    # Cosine similarities
    pos_sim = F.cosine_similarity(query, pos_doc, dim=1)
    neg_sim = F.cosine_similarity(query, neg_doc, dim=1)
    
    # Convert to distances
    pos_dist = 1 - pos_sim
    neg_dist = 1 - neg_sim
    
    # Basic triplet loss
    basic_loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    
    # Add regularization to prevent collapse
    # Encourage diversity in embeddings
    query_norm_penalty = torch.abs(query.norm(dim=1) - 1.0).mean()
    pos_norm_penalty = torch.abs(pos_doc.norm(dim=1) - 1.0).mean()
    neg_norm_penalty = torch.abs(neg_doc.norm(dim=1) - 1.0).mean()
    
    # Total loss
    total_loss = basic_loss.mean() + alpha * (query_norm_penalty + pos_norm_penalty + neg_norm_penalty)
    
    return total_loss


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
            normalize_output=config.get('NORMALIZE_OUTPUT', True),
            pooling_strategy=config.get('POOLING_STRATEGY', "last")
        )
    
    @staticmethod
    def get_loss_function(loss_type: str = 'triplet', margin: float = 1.0, alpha: float = 0.1):
        if loss_type == 'adaptive_triplet':
            return lambda triplet: adaptive_triplet_loss(triplet, margin, alpha)
        else:  # default to cosine-based triplet loss
            return lambda triplet: triplet_loss(triplet, margin) 