"""
Feature extraction for hate speech detection
Phase 3: Includes TF-IDF, Word2Vec, FastText, and linguistic features
"""

import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import (
    TFIDF_CHAR_CONFIG, TFIDF_WORD_CONFIG,
    URL_PATTERN, HASHTAG_PATTERN, MENTION_PATTERN,
    FEATURE_FILES, VIOLENCE_KEYWORDS, FEATURE_COMBINATION
)
from text_preprocessor import get_standard_preprocessor, tokenize_for_word2vec
from utils import logger, ProgressTracker

# Try importing optional dependencies
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    logger.warning("vaderSentiment not installed. Sentiment features will be basic.")

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    logger.warning("textstat not installed. Readability features will be estimated.")

# ==================== FEATURE EXTRACTOR ====================

class FeatureExtractor:
    """
    Extract multiple types of features from text for hate speech detection.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        logger.info("Initializing feature extractor...")
        
        # TF-IDF vectorizers
        self.tfidf_char = TfidfVectorizer(**TFIDF_CHAR_CONFIG)
        self.tfidf_word = TfidfVectorizer(**TFIDF_WORD_CONFIG)
        
        # Embedding models
        self.word2vec_model = None
        self.fasttext_model = None
        
        # Scaler for numerical features
        self.scaler = StandardScaler()
        
        # Text preprocessor
        self.preprocessor = get_standard_preprocessor()
        
        # Sentiment analyzer
        if HAS_VADER:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
        
        # Feature combination strategy
        self.feature_combination = FEATURE_COMBINATION
        
        # Fitted flag
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit the feature extractors on training data.
        
        Args:
            texts: List of training texts
        """
        logger.info("Fitting feature extractors...")
        
        # Fit TF-IDF vectorizers (if using them)
        if self.feature_combination in ['all', 'tfidf_only']:
            logger.info("Fitting character-level TF-IDF...")
            self.tfidf_char.fit(texts)
            
            logger.info("Fitting word-level TF-IDF...")
            self.tfidf_word.fit(texts)
        
        # Load embedding models (if using them)
        if self.feature_combination in ['all', 'embeddings_only']:
            logger.info("Loading embedding models...")
            self.load_embedding_models()
        
        # Fit scaler on linguistic features
        logger.info("Fitting scaler on linguistic features...")
        ling_features = self._extract_linguistic_features(texts)
        self.scaler.fit(ling_features)
        
        self.is_fitted = True
        logger.info("Feature extractors fitted successfully")
    
    def load_embedding_models(self):
        """Load Word2Vec and FastText models directly from files."""
        try:
            # Load Word2Vec
            if FEATURE_FILES['word2vec'].exists():
                self.word2vec_model = joblib.load(FEATURE_FILES['word2vec'])
                logger.info(f"Word2Vec loaded: {len(self.word2vec_model.wv)} words")
            else:
                logger.warning(f"Word2Vec model not found at {FEATURE_FILES['word2vec']}")
                self.word2vec_model = None
            
            # Load FastText
            if FEATURE_FILES['fasttext'].exists():
                self.fasttext_model = joblib.load(FEATURE_FILES['fasttext'])
                logger.info(f"FastText loaded: {len(self.fasttext_model.wv)} words")
            else:
                logger.warning(f"FastText model not found at {FEATURE_FILES['fasttext']}")
                self.fasttext_model = None
                
        except Exception as e:
            logger.warning(f"Could not load embedding models: {e}")
            logger.warning("Continuing without embeddings...")
            self.word2vec_model = None
            self.fasttext_model = None
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into feature vectors.
        
        Args:
            texts: List of texts
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Feature extractor must be fitted before transform")
        
        logger.info(f"Extracting features from {len(texts)} texts...")
        
        all_features = []
        
        # Extract TF-IDF features (if using them)
        if self.feature_combination in ['all', 'tfidf_only']:
            char_features = self.tfidf_char.transform(texts).toarray()
            word_features = self.tfidf_word.transform(texts).toarray()
            all_features.extend([char_features, word_features])
        
        # Extract embedding features (if using them)
        if self.feature_combination in ['all', 'embeddings_only']:
            if self.word2vec_model:
                w2v_features = self._get_word2vec_features(texts)
                all_features.append(w2v_features)
            
            if self.fasttext_model:
                ft_features = self._get_fasttext_features(texts)
                all_features.append(ft_features)
        
        # Extract linguistic features (always included)
        ling_features = self._extract_linguistic_features(texts)
        ling_features_scaled = self.scaler.transform(ling_features)
        all_features.append(ling_features_scaled)
        
        # Combine all features
        combined_features = np.concatenate(all_features, axis=1)
        
        logger.info(f"Feature extraction complete. Shape: {combined_features.shape}")
        return combined_features
    
    def _get_word2vec_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract Word2Vec features (average word vectors).
        
        Args:
            texts: List of texts
        
        Returns:
            Feature matrix (n_samples, vector_size)
        """
        features = []
        
        for text in texts:
            tokens = tokenize_for_word2vec(text)
            
            # Get vectors for tokens in vocabulary
            vectors = []
            for token in tokens:
                try:
                    if token in self.word2vec_model.wv:
                        vectors.append(self.word2vec_model.wv[token])
                except:
                    pass
            
            # Average vectors or return zeros
            if vectors:
                features.append(np.mean(vectors, axis=0))
            else:
                # Return zero vector if no words found
                vector_size = self.word2vec_model.wv.vector_size
                features.append(np.zeros(vector_size))
        
        return np.array(features)
    
    def _get_fasttext_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract FastText features (average word vectors).
        FastText can handle out-of-vocabulary words via character n-grams.
        
        Args:
            texts: List of texts
        
        Returns:
            Feature matrix (n_samples, vector_size)
        """
        features = []
        
        for text in texts:
            tokens = tokenize_for_word2vec(text)
            
            # Get vectors for all tokens (FastText handles OOV)
            vectors = []
            for token in tokens:
                try:
                    # FastText can generate vector even if not in vocab
                    vectors.append(self.fasttext_model.wv[token])
                except:
                    pass
            
            # Average vectors or return zeros
            if vectors:
                features.append(np.mean(vectors, axis=0))
            else:
                vector_size = self.fasttext_model.wv.vector_size
                features.append(np.zeros(vector_size))
        
        return np.array(features)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of texts
        
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def _extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract linguistic and statistical features from texts.
        
        Args:
            texts: List of texts
        
        Returns:
            Feature matrix (n_samples, n_linguistic_features)
        """
        features = []
        
        for text in texts:
            feature_vector = self._get_text_features(text)
            features.append(feature_vector)
        
        return np.array(features)
    
    def _get_text_features(self, text: str) -> List[float]:
        """
        Extract features from a single text.
        
        Returns:
            List of feature values
        """
        # Sentiment features
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            sent_compound = sentiment['compound']
            sent_pos = sentiment['pos']
            sent_neg = sentiment['neg']
            sent_neu = sentiment['neu']
        else:
            sent_compound = sent_pos = sent_neg = sent_neu = 0.0
        
        # Preprocess for word-level features
        processed_text = self.preprocessor.preprocess(text)
        
        # Basic text statistics
        num_chars = len(text)
        num_chars_no_space = len(text.replace(' ', ''))
        num_words = len(text.split())
        num_unique_words = len(set(text.lower().split()))
        
        # Readability features
        if HAS_TEXTSTAT and num_words > 0:
            try:
                fk_grade = textstat.flesch_kincaid_grade(text)
                f_reading = textstat.flesch_reading_ease(text)
                syllables = textstat.syllable_count(text)
            except:
                fk_grade = 10.0
                f_reading = 60.0
                syllables = num_words * 2.5
        else:
            # Simple estimates
            syllables = num_words * 2.5
            fk_grade = 10.0
            f_reading = 60.0
        
        # Social media features
        num_hashtags = len(re.findall(HASHTAG_PATTERN, text))
        num_mentions = len(re.findall(MENTION_PATTERN, text))
        num_urls = len(re.findall(URL_PATTERN, text))
        
        # Caps and punctuation
        num_caps = sum(1 for c in text if c.isupper())
        caps_ratio = num_caps / max(1, len(text))
        num_exclamation = text.count('!')
        num_question = text.count('?')
        num_dots = text.count('...')
        
        # Toxicity indicators
        has_violence = any(word in text.lower() for word in VIOLENCE_KEYWORDS)
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        
        # All caps words
        words = text.split()
        all_caps_words = sum(1 for w in words if len(w) > 1 and w.isupper())
        
        # Combine all features
        feature_vector = [
            # Readability (3)
            fk_grade, f_reading, syllables,
            
            # Text stats (4)
            num_chars, num_chars_no_space, num_words, num_unique_words,
            
            # Sentiment (4)
            sent_compound, sent_pos, sent_neg, sent_neu,
            
            # Social media (3)
            num_hashtags, num_mentions, num_urls,
            
            # Caps/Punctuation (4)
            caps_ratio, num_exclamation, num_question, num_dots,
            
            # Toxicity indicators (3)
            float(has_violence), repeated_chars, all_caps_words
        ]
        
        return feature_vector
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        if not self.is_fitted:
            raise RuntimeError("Feature extractor must be fitted first")
        
        feature_names = []
        
        # TF-IDF feature names (if using them)
        if self.feature_combination in ['all', 'tfidf_only']:
            char_names = [f"char_tfidf_{i}" for i in range(len(self.tfidf_char.get_feature_names_out()))]
            word_names = [f"word_tfidf_{i}" for i in range(len(self.tfidf_word.get_feature_names_out()))]
            feature_names.extend(char_names)
            feature_names.extend(word_names)
        
        # Embedding feature names (if using them)
        if self.feature_combination in ['all', 'embeddings_only']:
            if self.word2vec_model:
                w2v_size = self.word2vec_model.wv.vector_size
                w2v_names = [f"word2vec_{i}" for i in range(w2v_size)]
                feature_names.extend(w2v_names)
            
            if self.fasttext_model:
                ft_size = self.fasttext_model.wv.vector_size
                ft_names = [f"fasttext_{i}" for i in range(ft_size)]
                feature_names.extend(ft_names)
        
        # Linguistic feature names (always included)
        ling_names = [
            'fk_grade', 'f_reading', 'syllables',
            'num_chars', 'num_chars_no_space', 'num_words', 'num_unique_words',
            'sent_compound', 'sent_pos', 'sent_neg', 'sent_neu',
            'num_hashtags', 'num_mentions', 'num_urls',
            'caps_ratio', 'num_exclamation', 'num_question', 'num_dots',
            'has_violence', 'repeated_chars', 'all_caps_words'
        ]
        feature_names.extend(ling_names)
        
        return feature_names
    
    def save(self):
        """Save feature extractor to disk."""
        logger.info("Saving feature extractor...")
        
        # Save TF-IDF vectorizers
        if self.feature_combination in ['all', 'tfidf_only']:
            joblib.dump(self.tfidf_char, FEATURE_FILES['tfidf_char'])
            joblib.dump(self.tfidf_word, FEATURE_FILES['tfidf_word'])
        
        # Save scaler
        joblib.dump(self.scaler, FEATURE_FILES['scaler'])
        
        # Save main feature extractor
        joblib.dump(self, FEATURE_FILES['feature_extractor'])
        
        # Note: Embedding models are saved separately by EmbeddingTrainer
        
        logger.info("Feature extractor saved successfully")
    
    @staticmethod
    def load():
        """Load feature extractor from disk."""
        logger.info("Loading feature extractor...")
        
        feature_extractor = joblib.load(FEATURE_FILES['feature_extractor'])
        
        logger.info("Feature extractor loaded successfully")
        return feature_extractor

# ==================== CONVENIENCE FUNCTIONS ====================

def extract_features(texts: List[str], fit: bool = True) -> Tuple[np.ndarray, FeatureExtractor]:
    """
    Convenience function to extract features.
    
    Args:
        texts: List of texts
        fit: Whether to fit (True for training, False for inference)
    
    Returns:
        Tuple of (features, feature_extractor)
    """
    extractor = FeatureExtractor()
    
    if fit:
        features = extractor.fit_transform(texts)
    else:
        # Load pre-fitted extractor
        extractor = FeatureExtractor.load()
        features = extractor.transform(texts)
    
    return features, extractor

# ==================== TESTING ====================

if __name__ == "__main__":
    # Test feature extraction
    print("=" * 70)
    print("FEATURE EXTRACTION TEST")
    print("=" * 70)
    
    # Sample texts
    test_texts = [
        "Good morning everyone! Have a wonderful day ahead!",
        "You are a fucking idiot and should die",
        "I disagree with your political views but respect your opinion",
        "THIS IS SHOUTING!!! STOP IT NOW!!!",
        "Check out this link: http://example.com #awesome @user123"
    ]
    
    print(f"\nExtracting features from {len(test_texts)} texts...")
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.fit_transform(test_texts)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Number of features: {features.shape[1]}")
    
    # Show feature breakdown
    print(f"\nFeature strategy: {extractor.feature_combination}")
    
    # Test save/load
    print("\nTesting save/load...")
    extractor.save()
    
    loaded_extractor = FeatureExtractor.load()
    print("Successfully loaded feature extractor")
    
    # Transform with loaded extractor
    new_features = loaded_extractor.transform(test_texts)
    print(f"Transformed features shape: {new_features.shape}")
    
    # Verify they match
    if np.allclose(features, new_features):
        print("Save/load test passed!")
    else:
        print("Save/load test failed!")