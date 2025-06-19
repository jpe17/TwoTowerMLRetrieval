import pickle
import re
from typing import List, Dict


class PretrainedTokenizer:
    """Tokenizer that uses a pretrained word-to-index mapping and handles unknown words."""
    
    def __init__(self, word_to_idx_path: str):
        """
        Initialize tokenizer with pretrained vocabulary.
        
        Args:
            word_to_idx_path: Path to the pickled word-to-index dictionary
        """
        with open(word_to_idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        
        # Define and handle the unknown token
        self.unk_token = '<UNK>'
        if self.unk_token not in self.word2idx:
            unk_index = len(self.word2idx)
            self.word2idx[self.unk_token] = unk_index
            print(f"'{self.unk_token}' token not found in vocabulary. Added it at index {unk_index}.")
        
        self.unk_token_id = self.word2idx[self.unk_token]
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Loaded vocabulary with {len(self.word2idx):,} tokens (including '{self.unk_token}').")

    def encode(self, sentence: str) -> List[int]:
        """
        Encode a sentence into token indices, mapping unknown words to <UNK>.
        
        Args:
            sentence: Input text to tokenize
            
        Returns:
            List of token indices
        """
        # Standard practice: lowercase and tokenize
        tokens = re.findall(r"\w+|[.,!?;]", str(sentence).lower())
        # Map words to indices, using the UNK token for words not in the vocabulary
        return [self.word2idx.get(word, self.unk_token_id) for word in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token indices back to text.
        
        Args:
            token_ids: List of token indices
            
        Returns:
            Decoded text string
        """
        tokens = [self.idx2word.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join(tokens)

    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.word2idx)
    
    def get_word_index(self, word: str) -> int:
        """Get the index of a specific word."""
        return self.word2idx.get(word, -1)
    
    def get_index_word(self, index: int) -> str:
        """Get the word at a specific index."""
        return self.idx2word.get(index, '<UNK>')
    
    def contains_word(self, word: str) -> bool:
        """Check if a word exists in the vocabulary."""
        return word in self.word2idx 