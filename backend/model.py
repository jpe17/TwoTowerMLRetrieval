# --- Dataset Class ---
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch

class TripletDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, pos_doc, neg_doc = self.data[idx]
        return (torch.tensor(self.tokenizer.encode(query), dtype=torch.long),
                torch.tensor(self.tokenizer.encode(pos_doc), dtype=torch.long),
                torch.tensor(self.tokenizer.encode(neg_doc), dtype=torch.long))
    

# --- Collate Function ---
def collate_fn(batch):
    queries, pos_docs, neg_docs = zip(*batch)
    return (
        pad_sequence(queries, batch_first=True),
        pad_sequence(pos_docs, batch_first=True),
        pad_sequence(neg_docs, batch_first=True)
    )

# --- Dual RNN Encoder Model ---
class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Keep embeddings trainable (they are by default)
            
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        return h_n.squeeze(0)  # shape: (batch, hidden_dim)
    
# --- Triplet Loss Function ---
def triplet_loss_function(triplet, distance_function, margin):
    query, pos_doc, neg_doc = triplet
    d_pos = distance_function(query, pos_doc)
    d_neg = distance_function(query, neg_doc)
    return torch.clamp(d_pos - d_neg + margin, min=0.0).mean()
