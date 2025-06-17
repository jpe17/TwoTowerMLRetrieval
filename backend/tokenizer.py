import pickle
import re
from typing import List, Dict


class PretrainedTokenizer:
    """Tokenizer that uses a pretrained word-to-index mapping."""
    
    def __init__(self, word_to_idx_path: str):
        """
        Initialize tokenizer with pretrained vocabulary.
        
        Args:
            word_to_idx_path: Path to the pickled word-to-index dictionary
        """
        # Load pretrained word_to_idx mapping
        with open(word_to_idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Loaded vocabulary with {len(self.word2idx):,} tokens")

    def encode(self, sentence: str) -> List[int]:
        """
        Encode a sentence into token indices.
        
        Args:
            sentence: Input text to tokenize
            
        Returns:
            List of token indices
        """
        # Tokenize sentence, preserving punctuation
        tokens = re.findall(r"\w+|[.,!?;]", str(sentence))
        # Only include words that exist in vocabulary, skip unknown words
        return [self.word2idx[word] for word in tokens if word in self.word2idx]

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