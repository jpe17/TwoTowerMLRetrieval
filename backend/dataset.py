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
    """Collate function for batching variable-length sequences."""
    queries, pos_docs, neg_docs = zip(*batch)
    return (
        pad_sequence(queries, batch_first=True, padding_value=0),
        pad_sequence(pos_docs, batch_first=True, padding_value=0),
        pad_sequence(neg_docs, batch_first=True, padding_value=0)
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
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=False,
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