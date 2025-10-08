"""
Central configuration file for Hate Speech Detection Project
Contains all paths, hyperparameters, and settings

PHASE 1-3: Traditional ML Models + Embeddings
PHASE 4: Severity Classification System
PHASE 5: Deep Learning Models (LSTM, BiLSTM, CNN, BERT) (NEW)
"""

import re
from pathlib import Path

# ==================== PROJECT PATHS ====================

# Project root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "saved_models"
TRADITIONAL_ML_DIR = MODELS_DIR / "traditional_ml"
DEEP_LEARNING_DIR = MODELS_DIR / "deep_learning"
SEVERITY_MODELS_DIR = MODELS_DIR / "severity"

# Feature directories
FEATURES_DIR = PROJECT_ROOT / "saved_features"

# Results and logs
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
USER_REPORTS_DIR = RESULTS_DIR / "user_reports"

# Create directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, 
                  TRADITIONAL_ML_DIR, DEEP_LEARNING_DIR, SEVERITY_MODELS_DIR,
                  FEATURES_DIR, RESULTS_DIR, LOGS_DIR, USER_REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATA FILES ====================

LABELED_DATA_CSV = RAW_DATA_DIR / "labeled_data.csv"
COMBINED_DATA_CSV = RAW_DATA_DIR / "combined_dataset.csv"

# ==================== CLASS LABELS ====================

CLASS_LABELS = {
    0: "Hate speech",
    1: "Offensive language",
    2: "Neither"
}

NUM_CLASSES = 3

# ==================== DATA SPLIT CONFIGURATION ====================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# ==================== TEXT PREPROCESSING ====================

# Regex patterns
URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
MENTION_PATTERN = r'@[\w\-]+'
HASHTAG_PATTERN = r'#\w+'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
SPACE_PATTERN = r'\s+'
REPEATED_CHAR_PATTERN = r'(.)\1{2,}'

# Replacement tokens
URL_TOKEN = 'URLHERE'
MENTION_TOKEN = 'MENTIONHERE'
EMAIL_TOKEN = 'EMAILHERE'
PHONE_TOKEN = 'PHONEHERE'

# Custom stopwords (social media specific)
CUSTOM_STOPWORDS = [
    'rt', 'retweet', 'ff', 'followfriday', '#ff',
    'lol', 'lmao', 'omg', 'wtf', 'tbh', 'imo', 'imho'
]

# Contractions map
CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it's": "it is", "let's": "let us", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what's": "what is", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

# ==================== FEATURE EXTRACTION ====================

# TF-IDF Configuration
TFIDF_CHAR_CONFIG = {
    'analyzer': 'char',
    'ngram_range': (2, 4),
    'max_features': 200,
    'min_df': 2,
    'max_df': 0.95
}

TFIDF_WORD_CONFIG = {
    'analyzer': 'word',
    'ngram_range': (1, 3),
    'max_features': 300,
    'min_df': 2,
    'max_df': 0.95
}

# Word2Vec Configuration
WORD2VEC_CONFIG = {
    'vector_size': 100,
    'window': 5,
    'min_count': 2,
    'workers': 4,
    'epochs': 10,
    'sg': 0  # 0=CBOW, 1=Skip-gram
}

# FastText Configuration
FASTTEXT_CONFIG = {
    'vector_size': 100,
    'window': 5,
    'min_count': 2,
    'workers': 4,
    'epochs': 10,
    'sg': 1,  # 1=Skip-gram (better for FastText)
    'min_n': 3,  # Min character n-gram
    'max_n': 6   # Max character n-gram
}

# Sentence-BERT model
SENTENCE_BERT_MODEL = 'all-MiniLM-L6-v2'

# Feature Combination Strategy
FEATURE_COMBINATION = 'all'  # Options: 'all', 'embeddings_only', 'tfidf_only'

# ==================== TRADITIONAL ML MODELS ====================

RANDOM_FOREST_CONFIG = {
    'n_estimators': 200,
    'max_depth': 20,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'verbose': 1
}

XGBOOST_CONFIG = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.1,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss',
    'random_state': RANDOM_STATE,
    'verbosity': 1
}

SVM_CONFIG = {
    'kernel': 'rbf',
    'probability': True,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'verbose': True,
    'max_iter': 10000
}

GRADIENT_BOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 8,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'verbose': 1
}

MLP_CONFIG = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'max_iter': 300,
    'early_stopping': True,
    'random_state': RANDOM_STATE,
    'verbose': True
}

# SVM training limit (for efficiency)
SVM_MAX_SAMPLES = 20000

# ==================== PHASE 5: DEEP LEARNING MODELS ====================

# General Deep Learning Settings
DL_VOCAB_SIZE = 20000              # Maximum vocabulary size
DL_MAX_LENGTH = 100                # Maximum sequence length (tokens)
DL_EMBEDDING_DIM = 128             # Embedding dimensions
DL_VALIDATION_SPLIT = 0.2          # Validation split during training

# LSTM Configuration
LSTM_CONFIG = {
    'vocab_size': DL_VOCAB_SIZE,
    'embedding_dim': DL_EMBEDDING_DIM,
    'lstm_units': 128,              # LSTM layer units
    'dropout': 0.5,                 # Dropout rate
    'recurrent_dropout': 0.2,       # Recurrent dropout
    'max_length': DL_MAX_LENGTH,
    'batch_size': 32,
    'epochs': 10,
    'validation_split': DL_VALIDATION_SPLIT,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# BiLSTM Configuration
BILSTM_CONFIG = {
    'vocab_size': DL_VOCAB_SIZE,
    'embedding_dim': DL_EMBEDDING_DIM,
    'lstm_units': 128,              # Units per direction (total: 256)
    'dropout': 0.5,
    'recurrent_dropout': 0.2,
    'max_length': DL_MAX_LENGTH,
    'batch_size': 32,
    'epochs': 10,
    'validation_split': DL_VALIDATION_SPLIT,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# CNN Configuration
CNN_CONFIG = {
    'vocab_size': DL_VOCAB_SIZE,
    'embedding_dim': DL_EMBEDDING_DIM,
    'filter_sizes': [3, 4, 5],      # Multiple kernel sizes for n-grams
    'num_filters': 128,              # Filters per size
    'dropout': 0.5,
    'max_length': DL_MAX_LENGTH,
    'batch_size': 32,
    'epochs': 10,
    'validation_split': DL_VALIDATION_SPLIT,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# BERT Configuration
BERT_CONFIG = {
    'model_name': 'bert-base-uncased',  # Pretrained BERT model
    'max_length': 128,                   # Max sequence length for BERT
    'batch_size': 16,                    # Smaller batch for memory
    'epochs': 5,                         # Fewer epochs (pretrained)
    'learning_rate': 2e-5,               # Small LR for fine-tuning
    'warmup_steps': 100,                 # Warmup for learning rate
    'weight_decay': 0.01,                # L2 regularization
    'max_grad_norm': 1.0,                # Gradient clipping
    'optimizer': 'adamw',                # Adam with weight decay
    'epsilon': 1e-8                      # Optimizer epsilon
}

# Early Stopping Configuration (for all DL models)
EARLY_STOPPING_CONFIG = {
    'monitor': 'val_accuracy',
    'patience': 3,                       # Stop if no improvement for 3 epochs
    'restore_best_weights': True,
    'verbose': 1
}

# Model Checkpoint Configuration
MODEL_CHECKPOINT_CONFIG = {
    'monitor': 'val_accuracy',
    'save_best_only': True,
    'mode': 'max',
    'verbose': 1
}

# Training Options
USE_EARLY_STOPPING = True           # Enable early stopping
USE_MODEL_CHECKPOINT = True         # Save best model during training
USE_GPU = True                      # Try to use GPU if available
MIXED_PRECISION = False             # Mixed precision training (for newer GPUs)

# ==================== SEVERITY CLASSIFICATION ====================

# Severity Levels
SEVERITY_LEVELS = {
    1: "LOW",
    2: "MODERATE",
    3: "HIGH",
    4: "SEVERE",
    5: "EXTREME"
}

# -------------------- KEYWORD CATEGORIES --------------------

# Violence Keywords
VIOLENCE_KEYWORDS = [
    'kill', 'murder', 'die', 'death', 'dead', 'shoot', 'bomb', 'attack',
    'destroy', 'hurt', 'harm', 'beat', 'stab', 'rape', 'torture',
    'execute', 'assassinate', 'massacre', 'genocide', 'exterminate',
    'slaughter', 'butcher', 'annihilate', 'eliminate', 'eradicate',
    'lynch', 'hang', 'strangle', 'drown', 'suffocate', 'gas',
    'behead', 'decapitate', 'mutilate', 'dismember', 'crucify'
]

# Explicit Threat Patterns
THREAT_PATTERNS = [
    'will kill', 'gonna kill', 'going to kill', 'should die', 'must die',
    'deserve death', 'hope you die', 'wish you were dead', 'want you dead',
    'i will find you', 'watch your back', 'you\'re dead', 'will hurt you',
    'coming for you', 'better watch out', 'you\'ll pay', 'will get you',
    'gonna hurt', 'will make you', 'you will regret', 'gonna make you suffer',
    'count your days', 'your days are numbered', 'say goodbye'
]

# Dehumanization Terms
DEHUMANIZATION_KEYWORDS = [
    'vermin', 'pest', 'parasite', 'disease', 'infection', 'cancer',
    'trash', 'garbage', 'scum', 'filth', 'waste', 'animal', 'beast',
    'subhuman', 'inferior', 'savage', 'primitive', 'barbaric', 'uncivilized',
    'cockroach', 'rat', 'insect', 'creature', 'thing', 'infestation',
    'plague', 'contamination', 'pollution', 'stain', 'blight'
]

# Racial Slurs
RACIAL_SLURS = [
    'n*gga', 'n*gger', 'n1gga', 'n1gger', 'niqqa', 'niglet',
    'ch*nk', 'chink', 'ch1nk', 'chinaman',
    'sp*c', 'spic', 'sp1c', 'beaner', 'wetback', 'w3tback',
    'k*ke', 'kike', 'k1ke', 'hymie',
    'sand n*gger', 'sandnigger', 'raghead', 'rag head', 'towelhead',
    'p*ki', 'paki', 'curry muncher', 'dot head',
    'gook', 'slant', 'zipperhead', 'nip', 'jap',
    'redskin', 'injun', 'prairie n*gger',
    'cracker', 'honkey', 'whitey', 'redneck', 'hillbilly',
    'wigger', 'w1gger', 'race traitor'
]

# LGBTQ+ Slurs
LGBTQ_SLURS = [
    'f*ggot', 'f*g', 'faggot', 'fag', 'f4g', 'f4ggot', 'fagget',
    'd*ke', 'dyke', 'dike', 'lesbo', 'carpet muncher',
    'tr*nny', 'tranny', 'tr4nny', 'tr*ns', 'transtrender',
    'shemale', 'he-she', 'it', 'trap', 'ladyboy',
    'abomination', 'sodomite', 'homo', 'queer'
]

# Sexist/Misogynistic Terms
SEXIST_SLURS = [
    'bitch', 'b*tch', 'b1tch', 'bitches', 'biatch',
    'whore', 'wh*re', 'slut', 'sl*t', 's1ut', 'hoe', 'ho', 'thot',
    'cunt', 'c*nt', 'cnt', 'pussy', 'twat',
    'skank', 'tramp', 'prostitute', 'hooker',
    'dishwasher', 'sandwich maker', 'kitchen appliance',
    'cum dumpster', 'cock sleeve', 'cumdump',
    'feminazi', 'femoid', 'roastie', 'thot'
]

# Religious Hate Terms
RELIGIOUS_SLURS = [
    'infidel', 'kaffir', 'kafir', 'heathen',
    'jihadi', 'jihadist', 'terrorist', 'islamist', 'muzzie',
    'christ killer', 'jew banker', 'shekel',
    'godless', 'heretic', 'blasphemer', 'devil worshipper',
    'pagan', 'cultist', 'zealot', 'fanatic',
    'holy roller', 'bible thumper', 'jesus freak'
]

# Ableist Slurs
ABLEIST_SLURS = [
    'retard', 'r*tard', 'retarded', 'r3tard', 'tard',
    'mongoloid', 'mongol', 'down syndrome',
    'cripple', 'crip', 'gimp', 'vegetable',
    'spaz', 'spastic', 'autist', 'autistic',
    'psycho', 'crazy', 'insane', 'mental', 'lunatic',
    'schizo', 'bipolar', 'OCD'
]

# -------------------- SEVERITY SCORING --------------------

# Severity Score Weights
SEVERITY_WEIGHTS = {
    'violence_keywords': 15,
    'explicit_threats': 25,
    'dehumanization': 20,
    'racial_slurs': 30,
    'lgbtq_slurs': 30,
    'sexist_slurs': 25,
    'religious_slurs': 25,
    'ableist_slurs': 20,
    'all_caps_ratio': 10,
    'repeated_punctuation': 5,
    'targeted_at_person': 15,
    'multiple_slurs': 10
}

# Severity Score Thresholds
SEVERITY_THRESHOLDS = {
    1: (0, 20),      # LOW
    2: (21, 40),     # MODERATE
    3: (41, 65),     # HIGH
    4: (66, 85),     # SEVERE
    5: (86, 100)     # EXTREME
}

# Maximum severity score
MAX_SEVERITY_SCORE = 100

# -------------------- CONTEXT ANALYSIS --------------------

# Context Modifiers
CONTEXT_MODIFIERS = {
    'has_quote_marks': -10,
    'has_question_mark': -5,
    'starts_with_rt': -10,
    'contains_news_url': -5,
    'has_sarcasm': -10,
    'is_educational': -15
}

# Sarcasm Indicators
SARCASM_INDICATORS = [
    'yeah right', 'sure buddy', 'oh really', 'how nice',
    '/s', 'not', 'as if', 'like that will happen',
    'totally', 'obviously', 'clearly', 'sure thing'
]

# Educational Context Indicators
EDUCATIONAL_INDICATORS = [
    'according to', 'research shows', 'study found',
    'history of', 'definition of', 'awareness',
    'education', 'learning about', 'understanding'
]

# Quote Indicators
QUOTE_INDICATORS = [
    'he said', 'she said', 'they said', 'quote',
    'according to', 'claimed that', 'stated that'
]

# -------------------- ACTION RECOMMENDATIONS --------------------

# Action Rules
ACTION_RULES = {
    # Hate Speech
    (0, 5): "IMMEDIATE_BAN + REPORT_AUTHORITIES + REMOVE_CONTENT",
    (0, 4): "IMMEDIATE_BAN + REPORT_AUTHORITIES + REMOVE_CONTENT",
    (0, 3): "PERMANENT_BAN + REMOVE_CONTENT + NOTIFY_USER",
    (0, 2): "TEMPORARY_BAN_7_DAYS + REMOVE_CONTENT + WARNING",
    (0, 1): "WARNING + REMOVE_CONTENT + NOTIFY_USER",
    
    # Offensive Language
    (1, 5): "TEMPORARY_BAN_30_DAYS + REMOVE_CONTENT + FINAL_WARNING",
    (1, 4): "TEMPORARY_BAN_14_DAYS + REMOVE_CONTENT + FINAL_WARNING",
    (1, 3): "TEMPORARY_BAN_7_DAYS + REMOVE_CONTENT + WARNING",
    (1, 2): "WARNING + REMOVE_CONTENT",
    (1, 1): "WARNING + REDUCE_VISIBILITY",
    
    # Neither
    (2, 5): "REVIEW_MANUALLY + NOTIFY_USER",
    (2, 4): "REVIEW_MANUALLY + NOTIFY_USER",
    (2, 3): "REVIEW_MANUALLY",
    (2, 2): "NO_ACTION",
    (2, 1): "NO_ACTION"
}

# Action Descriptions
ACTION_DESCRIPTIONS = {
    "IMMEDIATE_BAN": "Permanently ban user account immediately without warning",
    "PERMANENT_BAN": "Permanently ban user account from platform",
    "TEMPORARY_BAN_30_DAYS": "Suspend user account for 30 days",
    "TEMPORARY_BAN_14_DAYS": "Suspend user account for 14 days",
    "TEMPORARY_BAN_7_DAYS": "Suspend user account for 7 days",
    "REMOVE_CONTENT": "Delete the offending content immediately",
    "WARNING": "Issue formal warning to user about policy violation",
    "FINAL_WARNING": "Issue final warning before permanent ban",
    "REPORT_AUTHORITIES": "Report content to law enforcement authorities",
    "NOTIFY_USER": "Send notification to user about content policy violation",
    "REDUCE_VISIBILITY": "Reduce content visibility and reach",
    "REVIEW_MANUALLY": "Flag content for human moderator review",
    "NO_ACTION": "No moderation action required"
}

# Action Urgency Levels
ACTION_URGENCY = {
    "IMMEDIATE_BAN": "CRITICAL",
    "REPORT_AUTHORITIES": "CRITICAL",
    "PERMANENT_BAN": "HIGH",
    "TEMPORARY_BAN_30_DAYS": "HIGH",
    "TEMPORARY_BAN_14_DAYS": "HIGH",
    "TEMPORARY_BAN_7_DAYS": "MEDIUM",
    "REMOVE_CONTENT": "HIGH",
    "WARNING": "MEDIUM",
    "FINAL_WARNING": "HIGH",
    "NOTIFY_USER": "LOW",
    "REDUCE_VISIBILITY": "LOW",
    "REVIEW_MANUALLY": "MEDIUM",
    "NO_ACTION": "NONE"
}

# ==================== EXPLAINABILITY ====================

LIME_CONFIG = {
    'num_samples': 1000,
    'num_features': 10
}

SHAP_CONFIG = {
    'max_display': 10
}

# ==================== ANALYTICS ====================

# User footprint scoring weights
FOOTPRINT_WEIGHTS = {
    'hate_percentage': 0.4,
    'severity_avg': 0.3,
    'offensive_percentage': 0.2,
    'frequency': 0.1
}

# Theme detection
MIN_THEME_FREQUENCY = 0.05

# ==================== LOGGING ====================

LOG_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ==================== MODEL SAVING ====================

# Save all models
SAVE_ALL_MODELS = True

# Model file names
MODEL_FILES = {
    # Traditional ML Models
    'random_forest': TRADITIONAL_ML_DIR / 'random_forest.pkl',
    'xgboost': TRADITIONAL_ML_DIR / 'xgboost.pkl',
    'svm': TRADITIONAL_ML_DIR / 'svm.pkl',
    'gradient_boost': TRADITIONAL_ML_DIR / 'gradient_boost.pkl',
    'mlp': TRADITIONAL_ML_DIR / 'neural_network.pkl',
    
    # Deep Learning Models (Phase 5)
    'lstm': DEEP_LEARNING_DIR / 'lstm_model.keras',
    'bilstm': DEEP_LEARNING_DIR / 'bilstm_model.keras',
    'cnn': DEEP_LEARNING_DIR / 'cnn_model.keras',
    'bert': DEEP_LEARNING_DIR / 'bert_model',
    'tokenizer': DEEP_LEARNING_DIR / 'tokenizer.pkl',
}

# Feature extractor files
FEATURE_FILES = {
    'tfidf_char': FEATURES_DIR / 'tfidf_char.pkl',
    'tfidf_word': FEATURES_DIR / 'tfidf_word.pkl',
    'word2vec': FEATURES_DIR / 'word2vec.pkl',
    'fasttext': FEATURES_DIR / 'fasttext_model.pkl',
    'scaler': FEATURES_DIR / 'scaler.pkl',
    'feature_extractor': FEATURES_DIR / 'feature_extractor.pkl'
}

# Severity model files
SEVERITY_FILES = {
    'severity_scorer': SEVERITY_MODELS_DIR / 'severity_scorer.pkl',
    'action_recommender': SEVERITY_MODELS_DIR / 'action_recommender.pkl'
}

# Metadata
METADATA_FILE = RESULTS_DIR / 'model_metadata.json'
COMPARISON_FILE = RESULTS_DIR / 'model_comparison.csv'
DL_COMPARISON_FILE = RESULTS_DIR / 'dl_model_comparison.csv'  # Phase 5 specific


# ==================== CLASS IMBALANCE HANDLING ====================

# Class weights to handle imbalanced dataset
# Calculated as: total_samples / (num_classes * class_count)
# Normalized so Offensive (majority class) = 1.0

CLASS_WEIGHTS = {
    0: 13.4,  # Hate speech (5.8% of data → highest weight)
    1: 1.0,   # Offensive language (77.4% of data → baseline weight)
    2: 4.6    # Neither (16.8% of data → medium weight)
}

# For sklearn models, convert to list format
CLASS_WEIGHTS_LIST = [13.4, 1.0, 4.6]

# For detailed info
CLASS_DISTRIBUTION = {
    'hate_speech': {'percentage': 5.8, 'weight': 13.4},
    'offensive': {'percentage': 77.4, 'weight': 1.0},
    'neither': {'percentage': 16.8, 'weight': 4.6}
}