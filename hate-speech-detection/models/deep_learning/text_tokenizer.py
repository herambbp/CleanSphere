"""
Text Tokenizer for Deep Learning Models
Converts text to integer sequences for neural networks

Features:
- Build vocabulary from training data
- Text to sequence conversion
- Padding/truncating to fixed length
- OOV (out-of-vocabulary) handling
- Save/load functionality
"""

import numpy as np
import pickle
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

from config import (
    DL_VOCAB_SIZE, DL_MAX_LENGTH, MODEL_FILES
)

# ==================== TEXT TOKENIZER ====================

class TextTokenizer:
    """
    Tokenizer for converting text to integer sequences.
    
    Wraps Keras Tokenizer with additional functionality:
    - Vocabulary building from training data
    - Text to sequence conversion
    - Padding/truncating sequences
    - Save/load functionality
    """
    
    def __init__(
        self, 
        vocab_size: int = DL_VOCAB_SIZE,
        max_length: int = DL_MAX_LENGTH,
        oov_token: str = '<OOV>',
        padding: str = 'post',
        truncating: str = 'post'
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            oov_token: Token for out-of-vocabulary words
            padding: Padding strategy ('pre' or 'post')
            truncating: Truncating strategy ('pre' or 'post')
        """
        if not HAS_KERAS:
            raise ImportError(
                "TensorFlow/Keras is required for TextTokenizer.\n"
                "Install with: pip install tensorflow==2.13.0"
            )
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.padding = padding
        self.truncating = truncating
        
        # Initialize Keras tokenizer
        self.tokenizer = KerasTokenizer(
            num_words=vocab_size,
            oov_token=oov_token,
            lower=True,
            char_level=False
        )
        
        self.is_fitted = False
        self.word_index = None
        self.index_word = None
    
    def fit_on_texts(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of training texts
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")
        
        # Fit tokenizer
        self.tokenizer.fit_on_texts(texts)
        
        # Store word mappings
        self.word_index = self.tokenizer.word_index
        self.index_word = self.tokenizer.index_word
        
        self.is_fitted = True
        
        # Log statistics
        total_words = len(self.word_index)
        vocab_used = min(total_words, self.vocab_size - 1)  # -1 for OOV
        
        return {
            'total_unique_words': total_words,
            'vocab_size_used': vocab_used,
            'coverage': vocab_used / total_words if total_words > 0 else 0
        }
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of integers.
        
        Args:
            texts: List of texts
        
        Returns:
            List of integer sequences
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before transforming")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences
    
    def pad_sequences(
        self, 
        sequences: List[List[int]],
        max_length: int = None,
        padding: str = None,
        truncating: str = None
    ) -> np.ndarray:
        """
        Pad sequences to fixed length.
        
        Args:
            sequences: List of integer sequences
            max_length: Maximum length (uses self.max_length if None)
            padding: Padding strategy (uses self.padding if None)
            truncating: Truncating strategy (uses self.truncating if None)
        
        Returns:
            Padded numpy array of shape (n_samples, max_length)
        """
        max_length = max_length or self.max_length
        padding = padding or self.padding
        truncating = truncating or self.truncating
        
        return pad_sequences(
            sequences,
            maxlen=max_length,
            padding=padding,
            truncating=truncating,
            value=0
        )
    
    def texts_to_padded_sequences(
        self, 
        texts: List[str],
        max_length: int = None
    ) -> np.ndarray:
        """
        Convert texts directly to padded sequences (convenience method).
        
        Args:
            texts: List of texts
            max_length: Maximum length
        
        Returns:
            Padded sequences as numpy array
        """
        sequences = self.texts_to_sequences(texts)
        return self.pad_sequences(sequences, max_length=max_length)
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert sequence back to text (for debugging).
        
        Args:
            sequence: Integer sequence
        
        Returns:
            Reconstructed text
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted first")
        
        words = []
        for idx in sequence:
            if idx == 0:  # Padding
                continue
            word = self.index_word.get(idx, self.oov_token)
            words.append(word)
        
        return ' '.join(words)
    
    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """
        Convert multiple sequences back to texts.
        
        Args:
            sequences: List of integer sequences
        
        Returns:
            List of reconstructed texts
        """
        return [self.sequence_to_text(seq) for seq in sequences]
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size."""
        if not self.is_fitted:
            return 0
        return min(len(self.word_index) + 1, self.vocab_size)  # +1 for padding
    
    def get_word_index(self, word: str) -> int:
        """
        Get index for a word.
        
        Args:
            word: Word to look up
        
        Returns:
            Word index (1 if OOV)
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted first")
        
        return self.word_index.get(word.lower(), 1)  # 1 is OOV token index
    
    def get_index_word(self, index: int) -> str:
        """
        Get word for an index.
        
        Args:
            index: Index to look up
        
        Returns:
            Word (OOV token if not found)
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted first")
        
        return self.index_word.get(index, self.oov_token)
    
    def get_top_words(self, n: int = 20) -> List[Tuple[str, int]]:
        """
        Get top N most frequent words.
        
        Args:
            n: Number of words to return
        
        Returns:
            List of (word, index) tuples
        """
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted first")
        
        # Word index is already sorted by frequency (lower index = more frequent)
        top_words = []
        for word, idx in sorted(self.word_index.items(), key=lambda x: x[1])[:n]:
            top_words.append((word, idx))
        
        return top_words
    
    def get_statistics(self) -> Dict:
        """
        Get tokenizer statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.is_fitted:
            return {
                'is_fitted': False,
                'vocab_size': 0,
                'max_length': self.max_length
            }
        
        return {
            'is_fitted': True,
            'vocab_size': self.get_vocab_size(),
            'total_unique_words': len(self.word_index),
            'max_length': self.max_length,
            'oov_token': self.oov_token,
            'padding': self.padding,
            'truncating': self.truncating
        }
    
    def save(self, filepath: str = None):
        """
        Save tokenizer to disk.
        
        Args:
            filepath: Path to save (uses config default if None)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted tokenizer")
        
        filepath = filepath or MODEL_FILES['tokenizer']
        
        # Prepare data to save
        save_data = {
            'config': {
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'oov_token': self.oov_token,
                'padding': self.padding,
                'truncating': self.truncating
            },
            'tokenizer_config': self.tokenizer.get_config(),
            'word_index': self.word_index,
            'index_word': self.index_word
        }
        
        # Save using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    @staticmethod
    def load(filepath: str = None) -> 'TextTokenizer':
        """
        Load tokenizer from disk.
        
        Args:
            filepath: Path to load from (uses config default if None)
        
        Returns:
            Loaded TextTokenizer instance
        """
        filepath = filepath or MODEL_FILES['tokenizer']
        
        # Load data
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Reconstruct tokenizer
        config = save_data['config']
        tokenizer = TextTokenizer(
            vocab_size=config['vocab_size'],
            max_length=config['max_length'],
            oov_token=config['oov_token'],
            padding=config['padding'],
            truncating=config['truncating']
        )
        
        # Restore internal state
        tokenizer.word_index = save_data['word_index']
        tokenizer.index_word = save_data['index_word']
        
        # Restore Keras tokenizer state
        tokenizer.tokenizer.word_index = save_data['word_index']
        tokenizer.tokenizer.index_word = save_data['index_word']
        
        tokenizer.is_fitted = True
        
        return tokenizer
    
    def __repr__(self):
        """String representation."""
        if self.is_fitted:
            return (
                f"TextTokenizer(vocab_size={self.get_vocab_size()}, "
                f"max_length={self.max_length}, fitted=True)"
            )
        else:
            return (
                f"TextTokenizer(vocab_size={self.vocab_size}, "
                f"max_length={self.max_length}, fitted=False)"
            )

# ==================== CONVENIENCE FUNCTIONS ====================

def create_tokenizer(
    texts: List[str],
    vocab_size: int = DL_VOCAB_SIZE,
    max_length: int = DL_MAX_LENGTH
) -> TextTokenizer:
    """
    Create and fit tokenizer on texts.
    
    Args:
        texts: Training texts
        vocab_size: Vocabulary size
        max_length: Max sequence length
    
    Returns:
        Fitted TextTokenizer
    """
    tokenizer = TextTokenizer(
        vocab_size=vocab_size,
        max_length=max_length
    )
    
    stats = tokenizer.fit_on_texts(texts)
    
    return tokenizer, stats

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("TEXT TOKENIZER TEST")
    print("=" * 70)
    
    # Sample texts
    train_texts = [
        "I hate when people do that",
        "You are stupid and annoying",
        "Good morning everyone!",
        "This is offensive language",
        "All groups should be treated equally",
        "Can't believe how dumb some people are",
        "Looking forward to the weekend",
        "These idiots don't know anything",
        "Congratulations on your achievement",
        "I disagree with your political views"
    ]
    
    print(f"\nTraining on {len(train_texts)} texts...")
    
    # Create and fit tokenizer
    tokenizer = TextTokenizer(vocab_size=100, max_length=20)
    stats = tokenizer.fit_on_texts(train_texts)
    
    print(f"\nTokenizer Statistics:")
    print(f"  Total unique words: {stats['total_unique_words']}")
    print(f"  Vocabulary size used: {stats['vocab_size_used']}")
    print(f"  Coverage: {stats['coverage']:.2%}")
    
    # Show top words
    print(f"\nTop 10 most frequent words:")
    for word, idx in tokenizer.get_top_words(10):
        print(f"  {idx:3d}. {word}")
    
    # Test text to sequence conversion
    test_texts = [
        "I hate you",
        "Good morning",
        "You are an idiot"
    ]
    
    print(f"\n{'Text':<30s} | {'Sequence':<30s} | Padded Shape")
    print("-" * 70)
    
    for text in test_texts:
        # Convert to sequence
        seq = tokenizer.texts_to_sequences([text])[0]
        
        # Pad sequence
        padded = tokenizer.pad_sequences([seq])
        
        print(f"{text:<30s} | {str(seq):<30s} | {padded.shape}")
    
    # Test sequence to text conversion
    print(f"\nSequence to Text Conversion:")
    for text in test_texts[:2]:
        seq = tokenizer.texts_to_sequences([text])[0]
        reconstructed = tokenizer.sequence_to_text(seq)
        print(f"  Original:      {text}")
        print(f"  Sequence:      {seq}")
        print(f"  Reconstructed: {reconstructed}")
        print()
    
    # Test save/load
    print("Testing save/load...")
    tokenizer.save()
    print(" Tokenizer saved")
    
    loaded_tokenizer = TextTokenizer.load()
    print(" Tokenizer loaded")
    
    # Verify loaded tokenizer works
    test_seq = loaded_tokenizer.texts_to_padded_sequences(["I hate you"])
    print(f" Loaded tokenizer works: {test_seq.shape}")
    
    print("\n" + "=" * 70)
    print("TEXT TOKENIZER TEST COMPLETE")
    print("=" * 70)