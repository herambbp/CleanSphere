"""
Embedding Adapter - Load Self-Trained Word2Vec/FastText for Deep Learning Models
Bridges Phase 3 embeddings with Phase 5 deep learning models

This adapter loads your existing Word2Vec/FastText models (trained in Phase 3)
and converts them to embedding matrices usable by PyTorch CNN/LSTM models.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import FEATURE_FILES

# ==================== EMBEDDING MATRIX BUILDER ====================

def load_self_trained_embeddings_for_dl(
    tokenizer,
    embedding_type: str = 'word2vec',
    embedding_dim: int = 100
) -> Optional[np.ndarray]:
    """
    Load self-trained Word2Vec or FastText embeddings and convert to embedding matrix.
    
    This is the bridge between Phase 3 (traditional ML) and Phase 5 (deep learning).
    
    Args:
        tokenizer: TextTokenizer instance with word_index
        embedding_type: 'word2vec' or 'fasttext'
        embedding_dim: Dimension of embeddings (should be 100 for your Phase 3 models)
    
    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim) or None if not found
    
    Example:
        from models.deep_learning.text_tokenizer import TextTokenizer
        from embedding_adapter import load_self_trained_embeddings_for_dl
        
        # Prepare tokenizer
        tokenizer = TextTokenizer(vocab_size=20000, max_length=100)
        tokenizer.fit_on_texts(X_train)
        
        # Load your Phase 3 embeddings
        embeddings = load_self_trained_embeddings_for_dl(tokenizer, 'word2vec', 100)
        
        # Use in CNN
        model = ImprovedCNNModel(config=config, pretrained_embeddings=embeddings)
    """
    
    print(f"\n{'='*80}")
    print(f"LOADING SELF-TRAINED {embedding_type.upper()} EMBEDDINGS FOR DEEP LEARNING")
    print(f"{'='*80}")
    
    # Check if embedding type is valid
    if embedding_type.lower() not in ['word2vec', 'fasttext']:
        print(f"[ERROR] Invalid embedding type: {embedding_type}")
        print(f"[ERROR] Must be 'word2vec' or 'fasttext'")
        return None
    
    # Get embedding file path
    embedding_file = FEATURE_FILES.get(embedding_type.lower())
    
    if embedding_file is None or not Path(embedding_file).exists():
        print(f"[ERROR] {embedding_type} embeddings not found at {embedding_file}")
        print(f"[INFO] Embeddings are trained in Phase 3")
        print(f"[INFO] Make sure you've run: python main_train.py")
        return None
    
    try:
        # Load the gensim model
        print(f"[LOADING] {embedding_file}")
        model = joblib.load(embedding_file)
        
        # Get word_index from tokenizer
        word_index = tokenizer.word_index
        vocab_size = tokenizer.vocab_size
        
        print(f"[INFO] Loaded {embedding_type} model")
        print(f"[INFO] Embedding vocabulary: {len(model.wv):,} words")
        print(f"[INFO] Tokenizer vocabulary: {vocab_size:,} words")
        print(f"[INFO] Embedding dimension: {embedding_dim}")
        
        # Initialize embedding matrix with zeros
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        # Fill embedding matrix
        found = 0
        missing = 0
        
        print(f"\n[BUILDING] Creating embedding matrix...")
        
        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue
            
            try:
                # Try to get word vector
                if word in model.wv:
                    embedding_matrix[idx] = model.wv[word]
                    found += 1
                else:
                    # For missing words, initialize with small random values
                    embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
                    missing += 1
            except Exception as e:
                # Fallback to random
                embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)
                missing += 1
        
        # Calculate coverage
        total = found + missing
        coverage = (found / total * 100) if total > 0 else 0
        
        print(f"\n[STATISTICS]")
        print(f"  Words found:     {found:,} ({coverage:.2f}%)")
        print(f"  Words missing:   {missing:,} ({100-coverage:.2f}%)")
        print(f"  Total:           {total:,}")
        print(f"  Matrix shape:    {embedding_matrix.shape}")
        
        # Test: Show similar words for common hate speech terms
        print(f"\n[SANITY CHECK] Testing embedding quality...")
        test_words = ['hate', 'fuck', 'bitch', 'nigger', 'love']
        
        for word in test_words:
            if word in model.wv and word in word_index:
                try:
                    similar = model.wv.most_similar(word, topn=3)
                    similar_words = [w for w, _ in similar]
                    print(f"  '{word}' → {similar_words}")
                except:
                    pass
        
        print(f"\n[SUCCESS] Embedding matrix ready for deep learning!")
        print(f"{'='*80}\n")
        
        return embedding_matrix
        
    except Exception as e:
        print(f"[ERROR] Failed to load embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_both_embeddings_for_dl(
    tokenizer,
    embedding_dim: int = 100
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load both Word2Vec and FastText embeddings.
    
    Returns:
        Tuple of (word2vec_matrix, fasttext_matrix)
    """
    
    print(f"\n{'='*80}")
    print(f"LOADING BOTH WORD2VEC AND FASTTEXT EMBEDDINGS")
    print(f"{'='*80}")
    
    # Load Word2Vec
    word2vec_matrix = load_self_trained_embeddings_for_dl(
        tokenizer,
        embedding_type='word2vec',
        embedding_dim=embedding_dim
    )
    
    # Load FastText
    fasttext_matrix = load_self_trained_embeddings_for_dl(
        tokenizer,
        embedding_type='fasttext',
        embedding_dim=embedding_dim
    )
    
    return word2vec_matrix, fasttext_matrix


def get_averaged_embeddings(
    tokenizer,
    embedding_dim: int = 100
) -> Optional[np.ndarray]:
    """
    Get averaged Word2Vec + FastText embeddings.
    
    This combines both embedding types for potentially better results.
    
    Returns:
        Averaged embedding matrix or None
    """
    
    print(f"\n{'='*80}")
    print(f"CREATING AVERAGED WORD2VEC + FASTTEXT EMBEDDINGS")
    print(f"{'='*80}")
    
    w2v_matrix, ft_matrix = load_both_embeddings_for_dl(tokenizer, embedding_dim)
    
    if w2v_matrix is None and ft_matrix is None:
        print(f"[ERROR] Neither Word2Vec nor FastText embeddings available")
        return None
    
    if w2v_matrix is None:
        print(f"[INFO] Using FastText only (Word2Vec not available)")
        return ft_matrix
    
    if ft_matrix is None:
        print(f"[INFO] Using Word2Vec only (FastText not available)")
        return w2v_matrix
    
    # Average both
    print(f"[AVERAGING] Combining Word2Vec and FastText...")
    averaged = (w2v_matrix + ft_matrix) / 2.0
    
    print(f"[SUCCESS] Averaged embedding matrix: {averaged.shape}")
    print(f"{'='*80}\n")
    
    return averaged


# ==================== USAGE EXAMPLES ====================

def example_usage():
    """Show how to use the embedding adapter."""
    
    print("\n" + "="*80)
    print("EMBEDDING ADAPTER - USAGE EXAMPLES")
    print("="*80)
    
    print("""
# Example 1: Load Word2Vec for CNN
from models.deep_learning.text_tokenizer import TextTokenizer
from models.deep_learning.cnn_model_improved import ImprovedCNNModel
from config_improved_cnn import IMPROVED_CNN_CONFIG
from embedding_adapter import load_self_trained_embeddings_for_dl

# Prepare tokenizer
tokenizer = TextTokenizer(vocab_size=20000, max_length=100)
tokenizer.fit_on_texts(X_train)

# Load YOUR Phase 3 Word2Vec embeddings
embeddings = load_self_trained_embeddings_for_dl(
    tokenizer=tokenizer,
    embedding_type='word2vec',
    embedding_dim=100  # Your Phase 3 dimension
)

# Train CNN with YOUR embeddings
model = ImprovedCNNModel(
    config=IMPROVED_CNN_CONFIG,
    pretrained_embeddings=embeddings
)
model.train(X_train_seq, y_train, X_val_seq, y_val)

# Expected: +5-8% improvement from using your self-trained embeddings!
""")
    
    print("""
# Example 2: Load FastText for CNN
embeddings = load_self_trained_embeddings_for_dl(
    tokenizer=tokenizer,
    embedding_type='fasttext',  # Use FastText instead
    embedding_dim=100
)

model = ImprovedCNNModel(
    config=IMPROVED_CNN_CONFIG,
    pretrained_embeddings=embeddings
)
""")
    
    print("""
# Example 3: Average both Word2Vec + FastText (BEST)
from embedding_adapter import get_averaged_embeddings

embeddings = get_averaged_embeddings(
    tokenizer=tokenizer,
    embedding_dim=100
)

model = ImprovedCNNModel(
    config=IMPROVED_CNN_CONFIG,
    pretrained_embeddings=embeddings
)

# This combines the strengths of both!
""")
    
    print("""
# Example 4: Compare with vs without embeddings
# Without embeddings (baseline)
model_baseline = ImprovedCNNModel(config=IMPROVED_CNN_CONFIG)
model_baseline.train(X_train_seq, y_train, X_val_seq, y_val)

# With YOUR embeddings
embeddings = load_self_trained_embeddings_for_dl(tokenizer, 'word2vec', 100)
model_improved = ImprovedCNNModel(
    config=IMPROVED_CNN_CONFIG,
    pretrained_embeddings=embeddings
)
model_improved.train(X_train_seq, y_train, X_val_seq, y_val)

# Compare results!
""")


if __name__ == "__main__":
    example_usage()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
This adapter loads your EXISTING Word2Vec/FastText models (from Phase 3)
and converts them to embedding matrices for PyTorch CNN/LSTM models.

Key Points:
1. Uses YOUR self-trained embeddings (trained on 24K tweets)
2. No need to download external embeddings
3. Converts gensim models → numpy matrices → PyTorch tensors
4. Expected improvement: +5-8% accuracy

Files needed:
- data/features/word2vec_model.pkl (from Phase 3)
- data/features/fasttext_model.pkl (from Phase 3)

These were already trained when you ran Phase 3!
""")