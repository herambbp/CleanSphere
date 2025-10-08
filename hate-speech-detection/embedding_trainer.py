"""
Train Word2Vec and FastText embedding models on tweet corpus
"""

import numpy as np
import joblib
from typing import List
import warnings
warnings.filterwarnings('ignore')

from config import WORD2VEC_CONFIG, FASTTEXT_CONFIG, FEATURE_FILES
from text_preprocessor import tokenize_for_word2vec
from utils import logger, print_section_header, ProgressTracker

# Try importing gensim
try:
    from gensim.models import Word2Vec, FastText
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    logger.warning("Gensim not installed. Word2Vec and FastText will not be available.")
    logger.warning("Install with: pip install gensim")

# ==================== EMBEDDING TRAINER ====================

class EmbeddingTrainer:
    """
    Train Word2Vec and FastText models on tweet corpus.
    """
    
    def __init__(self):
        """Initialize embedding trainer."""
        if not HAS_GENSIM:
            raise ImportError(
                "Gensim is required for embedding training. "
                "Install with: pip install gensim"
            )
        
        self.word2vec_model = None
        self.fasttext_model = None
        
        logger.info("Embedding Trainer initialized")
    
    def prepare_corpus(self, texts: List[str]) -> List[List[str]]:
        """
        Prepare corpus by tokenizing texts.
        
        Args:
            texts: List of raw texts
        
        Returns:
            List of tokenized sentences
        """
        print_section_header("PREPARING CORPUS FOR EMBEDDING TRAINING")
        
        logger.info(f"Tokenizing {len(texts)} texts...")
        
        tokenized_corpus = []
        progress = ProgressTracker(len(texts), "Tokenizing")
        
        for i, text in enumerate(texts):
            tokens = tokenize_for_word2vec(text)
            if tokens:  # Only add non-empty
                tokenized_corpus.append(tokens)
            
            if (i + 1) % 1000 == 0:
                progress.update(1000)
        
        progress.close()
        
        logger.info(f"Corpus prepared: {len(tokenized_corpus)} tokenized documents")
        
        # Log vocabulary statistics
        all_tokens = [token for doc in tokenized_corpus for token in doc]
        vocab_size = len(set(all_tokens))
        avg_len = np.mean([len(doc) for doc in tokenized_corpus])
        
        logger.info(f"Vocabulary size: {vocab_size:,} unique tokens")
        logger.info(f"Average document length: {avg_len:.1f} tokens")
        
        return tokenized_corpus
    
    def train_word2vec(self, corpus: List[List[str]]) -> Word2Vec:
        """
        Train Word2Vec model.
        
        Args:
            corpus: Tokenized corpus
        
        Returns:
            Trained Word2Vec model
        """
        print_section_header("TRAINING WORD2VEC")
        
        logger.info("Training Word2Vec model...")
        logger.info(f"Configuration: {WORD2VEC_CONFIG}")
        
        # Train Word2Vec
        model = Word2Vec(
            sentences=corpus,
            **WORD2VEC_CONFIG
        )
        
        # Log model statistics
        vocab_size = len(model.wv)
        logger.info(f"Word2Vec training complete!")
        logger.info(f"Vocabulary size: {vocab_size:,} words")
        
        # Test similarity
        if 'hate' in model.wv:
            similar = model.wv.most_similar('hate', topn=5)
            logger.info(f"Words similar to 'hate': {[w for w, _ in similar]}")
        
        self.word2vec_model = model
        return model
    
    def train_fasttext(self, corpus: List[List[str]]) -> FastText:
        """
        Train FastText model.
        
        Args:
            corpus: Tokenized corpus
        
        Returns:
            Trained FastText model
        """
        print_section_header("TRAINING FASTTEXT")
        
        logger.info("Training FastText model...")
        logger.info(f"Configuration: {FASTTEXT_CONFIG}")
        
        # Train FastText
        model = FastText(
            sentences=corpus,
            **FASTTEXT_CONFIG
        )
        
        # Log model statistics
        vocab_size = len(model.wv)
        logger.info(f"FastText training complete!")
        logger.info(f"Vocabulary size: {vocab_size:,} words")
        
        # Test similarity
        if 'hate' in model.wv:
            similar = model.wv.most_similar('hate', topn=5)
            logger.info(f"Words similar to 'hate': {[w for w, _ in similar]}")
        
        # Test out-of-vocabulary word (FastText advantage)
        try:
            # Test with a misspelled word
            vec = model.wv['hateeee']  # Misspelling
            logger.info("FastText successfully handles out-of-vocabulary words!")
        except:
            pass
        
        self.fasttext_model = model
        return model
    
    def train_all(self, texts: List[str]):
        """
        Train both Word2Vec and FastText.
        
        Args:
            texts: List of raw texts
        """
        print_section_header("EMBEDDING TRAINING PIPELINE")
        
        # Prepare corpus
        corpus = self.prepare_corpus(texts)
        
        # Train Word2Vec
        self.train_word2vec(corpus)
        
        # Train FastText
        self.train_fasttext(corpus)
        
        logger.info("All embedding models trained successfully!")
    
    def save_models(self):
        """Save trained embedding models."""
        print_section_header("SAVING EMBEDDING MODELS")
        
        if self.word2vec_model:
            joblib.dump(self.word2vec_model, FEATURE_FILES['word2vec'])
            logger.info(f"Word2Vec saved to {FEATURE_FILES['word2vec'].name}")
        
        if self.fasttext_model:
            joblib.dump(self.fasttext_model, FEATURE_FILES['fasttext'])
            logger.info(f"FastText saved to {FEATURE_FILES['fasttext'].name}")
        
        logger.info("All embedding models saved successfully!")
    
    @staticmethod
    def load_models():
        """
        Load saved embedding models.
        
        Returns:
            Tuple of (word2vec_model, fasttext_model)
        """
        logger.info("Loading embedding models...")
        
        word2vec_model = None
        fasttext_model = None
        
        try:
            word2vec_model = joblib.load(FEATURE_FILES['word2vec'])
            logger.info("Word2Vec model loaded")
        except Exception as e:
            logger.warning(f"Could not load Word2Vec: {e}")
        
        try:
            fasttext_model = joblib.load(FEATURE_FILES['fasttext'])
            logger.info("FastText model loaded")
        except Exception as e:
            logger.warning(f"Could not load FastText: {e}")
        
        return word2vec_model, fasttext_model

# ==================== CONVENIENCE FUNCTIONS ====================

def train_embeddings(texts: List[str], save: bool = True):
    """
    Convenience function to train and save embeddings.
    
    Args:
        texts: List of texts
        save: Whether to save models
    
    Returns:
        EmbeddingTrainer instance
    """
    if not HAS_GENSIM:
        logger.error("Cannot train embeddings without gensim")
        logger.error("Install with: pip install gensim")
        return None
    
    trainer = EmbeddingTrainer()
    trainer.train_all(texts)
    
    if save:
        trainer.save_models()
    
    return trainer

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("EMBEDDING TRAINER TEST")
    print("=" * 70)
    
    if not HAS_GENSIM:
        print("\nError: Gensim not installed!")
        print("Install with: pip install gensim")
        exit(1)
    
    # Test with sample texts
    test_texts = [
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
    ] * 100  # Repeat to have enough data
    
    print(f"\nTraining embeddings on {len(test_texts)} sample texts...")
    
    # Train
    trainer = EmbeddingTrainer()
    trainer.train_all(test_texts)
    
    # Test Word2Vec
    print("\n" + "=" * 70)
    print("WORD2VEC TEST")
    print("=" * 70)
    
    if trainer.word2vec_model:
        print(f"\nVocabulary size: {len(trainer.word2vec_model.wv)}")
        
        test_words = ['hate', 'stupid', 'good', 'people']
        for word in test_words:
            if word in trainer.word2vec_model.wv:
                similar = trainer.word2vec_model.wv.most_similar(word, topn=3)
                print(f"\nWords similar to '{word}':")
                for sim_word, score in similar:
                    print(f"  {sim_word}: {score:.3f}")
    
    # Test FastText
    print("\n" + "=" * 70)
    print("FASTTEXT TEST")
    print("=" * 70)
    
    if trainer.fasttext_model:
        print(f"\nVocabulary size: {len(trainer.fasttext_model.wv)}")
        
        # Test OOV handling
        print("\nTesting out-of-vocabulary words:")
        oov_words = ['haaaate', 'stuuupid', 'goooood']
        for word in oov_words:
            try:
                vec = trainer.fasttext_model.wv[word]
                print(f"  '{word}': ✓ Can generate vector")
            except:
                print(f"  '{word}': ✗ Cannot generate vector")
    
    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    trainer.save_models()
    
    # Test loading
    print("\n" + "=" * 70)
    print("TESTING LOAD")
    print("=" * 70)
    w2v, ft = EmbeddingTrainer.load_models()
    
    if w2v:
        print(f"✓ Word2Vec loaded: {len(w2v.wv)} words")
    if ft:
        print(f"✓ FastText loaded: {len(ft.wv)} words")
    
    print("\n✓ Embedding trainer test complete!")