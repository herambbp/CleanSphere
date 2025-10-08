"""
Text preprocessing for hate speech detection
Handles cleaning, tokenization, and normalization
"""

import re
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from typing import List
import warnings
warnings.filterwarnings('ignore')

from config import (
    URL_PATTERN, MENTION_PATTERN, HASHTAG_PATTERN, EMAIL_PATTERN, 
    PHONE_PATTERN, SPACE_PATTERN, REPEATED_CHAR_PATTERN,
    URL_TOKEN, MENTION_TOKEN, EMAIL_TOKEN, PHONE_TOKEN,
    CUSTOM_STOPWORDS, CONTRACTIONS
)
from utils import logger

# ==================== NLTK SETUP ====================

def ensure_nltk_downloads():
    """Download required NLTK resources."""
    resources = ['stopwords', 'punkt', 'punkt_tab', 'vader_lexicon']
    
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    logger.info(f"Downloaded NLTK resource: {resource}")
                except Exception as e:
                    logger.warning(f"Could not download {resource}: {e}")

# Initialize NLTK
ensure_nltk_downloads()

# Initialize stemmer
stemmer = PorterStemmer()

# Initialize stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.update(CUSTOM_STOPWORDS)
except Exception as e:
    logger.warning(f"Could not load stopwords: {e}")
    STOPWORDS = set(CUSTOM_STOPWORDS)

# ==================== TEXT PREPROCESSOR ====================

class TextPreprocessor:
    """
    Comprehensive text preprocessing for hate speech detection.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_emails: bool = True,
        remove_phones: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = False,
        expand_contractions: bool = True,
        remove_stopwords: bool = False,
        apply_stemming: bool = False,
        normalize_whitespace: bool = True,
        keep_hashtags: bool = True
    ):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Convert to lowercase
            remove_urls: Replace URLs with token
            remove_mentions: Replace @mentions with token
            remove_emails: Replace emails with token
            remove_phones: Replace phone numbers with token
            remove_numbers: Remove all numbers
            remove_punctuation: Remove punctuation
            expand_contractions: Expand contractions (don't -> do not)
            remove_stopwords: Remove stopwords
            apply_stemming: Apply Porter stemming
            normalize_whitespace: Normalize spaces
            keep_hashtags: Keep hashtags (important for hate speech context)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_emails = remove_emails
        self.remove_phones = remove_phones
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.normalize_whitespace = normalize_whitespace
        self.keep_hashtags = keep_hashtags
        
        self.stemmer = stemmer
        self.stopwords = STOPWORDS
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 1. Lowercase
        if self.lowercase:
            text = text.lower()
        
        # 2. Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # 3. Replace URLs
        if self.remove_urls:
            text = re.sub(URL_PATTERN, URL_TOKEN, text)
        
        # 4. Replace emails
        if self.remove_emails:
            text = re.sub(EMAIL_PATTERN, EMAIL_TOKEN, text)
        
        # 5. Replace phone numbers
        if self.remove_phones:
            text = re.sub(PHONE_PATTERN, PHONE_TOKEN, text)
        
        # 6. Replace mentions
        if self.remove_mentions:
            text = re.sub(MENTION_PATTERN, MENTION_TOKEN, text)
        
        # 7. Remove or keep hashtags
        if not self.keep_hashtags:
            text = re.sub(HASHTAG_PATTERN, '', text)
        
        # 8. Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # 9. Remove punctuation (be careful - can lose important info)
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 10. Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(SPACE_PATTERN, ' ', text)
        
        # 11. Strip
        text = text.strip()
        
        return text
    
    def preprocess_and_tokenize(self, text: str) -> List[str]:
        """
        Preprocess and tokenize text into words.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Preprocess
        text = self.preprocess(text)
        
        if not text:
            return []
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords]
        
        # Apply stemming
        if self.apply_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions using the contractions map."""
        for contraction, expansion in CONTRACTIONS.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts
        
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

# ==================== PRESET PREPROCESSORS ====================

def get_minimal_preprocessor() -> TextPreprocessor:
    """
    Minimal preprocessing - keeps most original text.
    Good for models that need raw text (like BERT).
    """
    return TextPreprocessor(
        lowercase=False,
        remove_urls=True,
        remove_mentions=False,
        remove_emails=True,
        remove_phones=True,
        remove_numbers=False,
        remove_punctuation=False,
        expand_contractions=False,
        remove_stopwords=False,
        apply_stemming=False,
        normalize_whitespace=True,
        keep_hashtags=True
    )


def get_standard_preprocessor() -> TextPreprocessor:
    """
    Standard preprocessing - recommended for most ML models.
    Balanced cleaning while keeping important features.
    """
    return TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_emails=True,
        remove_phones=True,
        remove_numbers=False,
        remove_punctuation=False,
        expand_contractions=True,
        remove_stopwords=False,
        apply_stemming=False,
        normalize_whitespace=True,
        keep_hashtags=True  # Hashtags can be informative for hate speech
    )


def get_aggressive_preprocessor() -> TextPreprocessor:
    """
    Aggressive preprocessing - maximum cleaning.
    For traditional ML models that need clean tokens.
    """
    return TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_emails=True,
        remove_phones=True,
        remove_numbers=True,
        remove_punctuation=True,
        expand_contractions=True,
        remove_stopwords=True,
        apply_stemming=True,
        normalize_whitespace=True,
        keep_hashtags=False
    )

# ==================== UTILITY FUNCTIONS ====================

def tokenize_for_word2vec(text: str) -> List[str]:
    """
    Tokenize text for Word2Vec training.
    Uses standard preprocessing with stopword removal and stemming.
    
    Args:
        text: Input text
    
    Returns:
        List of tokens
    """
    preprocessor = get_standard_preprocessor()
    preprocessor.remove_stopwords = True
    preprocessor.apply_stemming = True
    return preprocessor.preprocess_and_tokenize(text)


def tokenize_for_bert(text: str) -> str:
    """
    Minimal preprocessing for BERT (BERT has its own tokenizer).
    
    Args:
        text: Input text
    
    Returns:
        Minimally preprocessed text
    """
    preprocessor = get_minimal_preprocessor()
    return preprocessor.preprocess(text)


def clean_text_simple(text: str) -> str:
    """
    Simple cleaning function for quick preprocessing.
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    preprocessor = get_standard_preprocessor()
    return preprocessor.preprocess(text)

# ==================== ADDITIONAL UTILITIES ====================

def remove_repeated_chars(text: str, max_repeats: int = 2) -> str:
    """
    Reduce repeated characters (e.g., 'yessss' -> 'yess').
    
    Args:
        text: Input text
        max_repeats: Maximum repetitions to keep
    
    Returns:
        Text with reduced repetitions
    """
    pattern = r'(.)\1{' + str(max_repeats) + ',}'
    replacement = r'\1' * max_repeats
    return re.sub(pattern, replacement, text)


def extract_hashtags(text: str) -> List[str]:
    """Extract all hashtags from text."""
    return re.findall(HASHTAG_PATTERN, text)


def extract_mentions(text: str) -> List[str]:
    """Extract all mentions from text."""
    return re.findall(MENTION_PATTERN, text)


def has_urls(text: str) -> bool:
    """Check if text contains URLs."""
    return bool(re.search(URL_PATTERN, text))


def count_caps(text: str) -> int:
    """Count uppercase characters."""
    return sum(1 for c in text if c.isupper())


def get_caps_ratio(text: str) -> float:
    """Get ratio of uppercase to total characters."""
    if len(text) == 0:
        return 0.0
    return count_caps(text) / len(text)

# ==================== TESTING ====================

if __name__ == "__main__":
    # Test examples
    test_texts = [
        "Check out http://example.com #awesome @user123",
        "I can't believe this!!! It's soooo amazing!!!",
        "Call me at 555-123-4567 or email test@example.com",
        "THIS IS SHOUTING!!! Stop doing that...",
        "Don't you think we're going too fast?",
        "You are a fucking idiot and should die"
    ]
    
    print("=" * 70)
    print("TEXT PREPROCESSING EXAMPLES")
    print("=" * 70)
    
    # Test different preprocessors
    preprocessors = {
        'Minimal': get_minimal_preprocessor(),
        'Standard': get_standard_preprocessor(),
        'Aggressive': get_aggressive_preprocessor()
    }
    
    for text in test_texts[:3]:  # First 3 examples
        print(f"\nOriginal: {text}")
        
        for name, preprocessor in preprocessors.items():
            processed = preprocessor.preprocess(text)
            print(f"{name:12s}: {processed}")
    
    # Test tokenization
    print("\n" + "=" * 70)
    print("TOKENIZATION EXAMPLES")
    print("=" * 70)
    
    text = "I can't believe these idiots don't know what they're doing!"
    print(f"Original: {text}")
    
    tokens = tokenize_for_word2vec(text)
    print(f"Tokens for Word2Vec: {tokens}")
    
    clean = tokenize_for_bert(text)
    print(f"For BERT: {clean}")