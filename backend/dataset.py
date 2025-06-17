import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict
from tokenizer import PretrainedTokenizer


class TripletDataset(Dataset):
    """Dataset class for triplet data (query, positive_doc, negative_doc)."""
    
    def __init__(self, data: List[Tuple[str, str, str]], tokenizer: PretrainedTokenizer):
        """
        Initialize the dataset.
        
        Args:
            data: List of triplets (query, positive_doc, negative_doc)
            tokenizer: Tokenizer instance for encoding text
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get tokenized tensors for query, positive doc, and negative doc."""
        query, pos_doc, neg_doc = self.data[idx]
        return (
            torch.tensor(self.tokenizer.encode(query), dtype=torch.long),
            torch.tensor(self.tokenizer.encode(pos_doc), dtype=torch.long),
            torch.tensor(self.tokenizer.encode(neg_doc), dtype=torch.long)
        )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for batching variable-length sequences with LEFT padding."""
    queries, pos_docs, neg_docs = zip(*batch)
    
    def left_pad_sequences(sequences):
        """Pad sequences on the left (beginning) instead of right."""
        # Find the maximum length
        max_len = max(len(seq) for seq in sequences)
        
        # Create padded tensors
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                # Create padding of zeros at the beginning
                padding = torch.zeros(max_len - len(seq), dtype=seq.dtype)
                padded_seq = torch.cat([padding, seq])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        
        # Stack into a batch tensor
        return torch.stack(padded)
    
    return (
        left_pad_sequences(queries),
        left_pad_sequences(pos_docs), 
        left_pad_sequences(neg_docs)
    )


class DataLoaderFactory:
    """Factory class for creating DataLoaders with consistent settings."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def create_dataloaders(self, datasets: Dict[str, List[Tuple[str, str, str]]], 
                          tokenizer: PretrainedTokenizer) -> Dict[str, DataLoader]:
        """Create DataLoaders for all dataset splits."""
        dataloaders = {}
        
        for split, data in datasets.items():
            if not data:
                print(f"⚠️  Skipping {split} dataset (empty)")
                continue
                
            dataset = TripletDataset(data, tokenizer)
            batch_size = self.config.get('BATCH_SIZE', 64)
            
            # Use smaller batch size for validation/test
            if split in ['validation', 'test']:
                batch_size = min(batch_size, 32)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=2 if split == 'train' else 1,  # More workers for training
                collate_fn=collate_fn,
                drop_last=(split == 'train')
            )
            
            dataloaders[split] = dataloader
            print(f"✅ {split} dataloader ready! ({len(dataset):,} samples, {len(dataloader)} batches)")
        
        return dataloaders
    
    def get_dataloader_stats(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Dict[str, int]]:
        """Get dataloader statistics."""
        return {
            split: {
                'num_samples': len(dataloader.dataset),
                'num_batches': len(dataloader),
                'batch_size': dataloader.batch_size
            }
            for split, dataloader in dataloaders.items()
        } 