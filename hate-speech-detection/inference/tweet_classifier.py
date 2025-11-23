"""
Unified Tweet Classifier - Supports Traditional ML, PyTorch DL, and BERT Models
Automatically selects the best model from ALL trained models

PHASE 1-3: Base classification (Hate/Offensive/Neither)
PHASE 4: Severity analysis + Action recommendations
PHASE 5: Deep learning models (LSTM, BiLSTM, CNN, BERT) - PyTorch Implementation
"""

import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_FILES, CLASS_LABELS, RESULTS_DIR, MODELS_DIR
from utils import logger, print_section_header, load_results

# Check for PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available. Deep learning models will not work.")

# ==================== UNIFIED TWEET CLASSIFIER ====================

class TweetClassifier:
    """
    Unified classifier supporting Traditional ML, PyTorch DL, and BERT models.
    
    Features:
    - Automatically selects best model (Traditional ML, PyTorch DL, or BERT)
    - Base classification (Hate/Offensive/Neither)
    - Confidence scores
    - Severity analysis (LOW to EXTREME) [Phase 4]
    - Action recommendations [Phase 4]
    - Supports all model types: RF, XGBoost, SVM, MLP, LSTM, BiLSTM, CNN, BERT
    """
    
    def __init__(self, model_name: str = 'best', model_type: str = 'auto'):
        """
        Initialize classifier.
        
        Args:
            model_name: Name of model ('best', 'cnn', 'bert', 'xgboost', etc.)
            model_type: Type of model ('auto', 'traditional', 'deep_learning', 'bert')
        """
        logger.info("Initializing Unified Tweet Classifier...")
        
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.metadata = {}
        
        # Phase 4 components (lazy loaded)
        self.severity_scorer = None
        self.action_recommender = None
        
        # Load metadata first
        self._load_metadata()
        
        # Resolve 'best' to actual model name BEFORE determining type
        if model_name.lower() == 'best':
            actual_model = self._get_best_model_name()
            self.model_name = actual_model
            model_name = actual_model
            logger.info(f"'best' resolved to: {actual_model}")
        
        # Determine type and load components
        self._determine_model_type(model_name)
        self._load_model_components()
        self._load_model(model_name)
        
        logger.info(f"Classifier ready with model: {self.model_name} ({self.model_type})")
        if self.model_name in self.metadata:
            logger.info(f"Model accuracy: {self.metadata[self.model_name]['accuracy']:.4f}")
            logger.info(f"Model F1-Macro: {self.metadata[self.model_name]['f1_macro']:.4f}")
    
    def _load_metadata(self):
        """Load metadata from both Traditional ML, Deep Learning, and BERT."""
        # Load Traditional ML metadata
        try:
            trad_metadata = load_results('model_metadata.json')
            if trad_metadata and 'all_results' in trad_metadata:
                for model_name, metrics in trad_metadata['all_results'].items():
                    self.metadata[model_name] = {
                        'accuracy': metrics.get('accuracy', 0),
                        'f1_macro': metrics.get('f1_macro', 0),
                        'f1_weighted': metrics.get('f1_weighted', 0),
                        'type': 'Traditional ML'
                    }
            logger.info("Traditional ML metadata loaded")
        except Exception as e:
            logger.warning(f"Could not load Traditional ML metadata: {e}")
        
        # Load Deep Learning metadata (PyTorch + BERT)
        try:
            dl_metadata = load_results('dl_model_metadata.json')
            if dl_metadata and 'all_results' in dl_metadata:
                for model_name, metrics in dl_metadata['all_results'].items():
                    # Determine if it's BERT or regular DL
                    model_type = 'BERT' if 'bert' in model_name.lower() else 'Deep Learning'
                    
                    self.metadata[model_name] = {
                        'accuracy': metrics.get('accuracy', 0),
                        'f1_macro': metrics.get('f1_macro', 0),
                        'f1_weighted': metrics.get('f1_weighted', 0),
                        'type': model_type
                    }
            logger.info("Deep Learning metadata loaded")
        except Exception as e:
            logger.warning(f"Could not load Deep Learning metadata: {e}")
    
    def _get_best_model_name(self) -> str:
        """Get the name of the best model from all metadata."""
        if not self.metadata:
            logger.warning("No metadata found, defaulting to CNN")
            return 'CNN'
        
        # Find model with highest F1-Macro
        best_model = max(self.metadata.items(), key=lambda x: x[1].get('f1_macro', 0))
        
        logger.info(f"Best model selected: {best_model[0]} (F1-Macro: {best_model[1]['f1_macro']:.4f})")
        return best_model[0]
    
    def _determine_model_type(self, model_name: str):
        """Determine if model is Traditional ML, PyTorch DL, or BERT."""
        if self.model_type != 'auto':
            return
        
        model_lower = model_name.lower().replace(' ', '').replace('_', '').replace('-', '')
        
        # Check if it's BERT
        if 'bert' in model_lower:
            self.model_type = 'bert'
        # Check if it's PyTorch DL
        elif any(dl in model_lower for dl in ['lstm', 'bilstm', 'cnn']):
            self.model_type = 'deep_learning'
        # Check if it's Traditional ML
        elif any(trad in model_lower for trad in ['randomforest', 'xgboost', 'svm', 'gradientboosting', 'mlp']):
            self.model_type = 'traditional'
        else:
            # Default based on metadata
            if model_name in self.metadata:
                meta_type = self.metadata[model_name]['type']
                if meta_type == 'BERT':
                    self.model_type = 'bert'
                elif meta_type == 'Deep Learning':
                    self.model_type = 'deep_learning'
                else:
                    self.model_type = 'traditional'
            else:
                logger.warning(f"Unknown model type for {model_name}, defaulting to traditional")
                self.model_type = 'traditional'
    
    def _load_model_components(self):
        """Load required components based on model type."""
        if self.model_type == 'traditional':
            self._load_feature_extractor()
        elif self.model_type in ['deep_learning', 'bert']:
            self._load_tokenizer()
    
    def _load_feature_extractor(self):
        """Load the feature extractor (for Traditional ML)."""
        try:
            from feature_extractor import FeatureExtractor
            self.feature_extractor = FeatureExtractor.load()
            logger.info("Feature extractor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading feature extractor: {e}")
            raise RuntimeError(
                "Could not load feature extractor. "
                "Make sure you have trained the models first."
            )
    
    def _load_tokenizer(self):
        """Load the tokenizer (for Deep Learning and BERT)."""
        try:
            # Import TextTokenizer class
            from models.deep_learning.text_tokenizer import TextTokenizer
            
            tokenizer_path = MODEL_FILES.get('tokenizer')
            logger.info(f"Looking for tokenizer at: {tokenizer_path}")
            
            if not tokenizer_path:
                error_msg = "Tokenizer path not found in MODEL_FILES config"
                logger.error(error_msg)
                if self.model_type == 'deep_learning':
                    raise RuntimeError(error_msg)
                self.tokenizer = None
                return
                
            if not tokenizer_path.exists():
                error_msg = f"Tokenizer file not found at: {tokenizer_path}"
                logger.error(error_msg)
                if self.model_type == 'deep_learning':
                    raise FileNotFoundError(error_msg)
                self.tokenizer = None
                return
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            loaded_data = joblib.load(tokenizer_path)
            logger.info(f"Tokenizer loaded, type: {type(loaded_data)}")
            
            # Handle different tokenizer formats
            if isinstance(loaded_data, TextTokenizer):
                # Already a TextTokenizer object
                self.tokenizer = loaded_data
                logger.info("Tokenizer loaded successfully (TextTokenizer object)")
                
            elif isinstance(loaded_data, dict):
                logger.info(f"Tokenizer is dict with keys: {loaded_data.keys()}")
                
                # Format 1: Has 'tokenizer' key (older format)
                if 'tokenizer' in loaded_data:
                    self.tokenizer = TextTokenizer(
                        vocab_size=loaded_data.get('vocab_size', 20000),
                        max_length=loaded_data.get('max_length', 100)
                    )
                    self.tokenizer.tokenizer = loaded_data['tokenizer']
                    self.tokenizer.is_fitted = loaded_data.get('is_fitted', True)
                    logger.info("TextTokenizer reconstructed from dict (format 1)")
                
                # Format 2: Has 'word_index' and 'config' keys (Keras tokenizer state)
                elif 'word_index' in loaded_data and 'config' in loaded_data:
                    logger.info("Detected Keras tokenizer format, reconstructing...")
                    
                    # Import Keras Tokenizer
                    from tensorflow.keras.preprocessing.text import Tokenizer
                    from tensorflow.keras.preprocessing.sequence import pad_sequences
                    
                    # Get config - handle both dict and string formats
                    config_data = loaded_data['config']
                    logger.info(f"Config type: {type(config_data)}")
                    logger.info(f"Config value: {config_data}")
                    
                    # If config is a string, try to parse it
                    if isinstance(config_data, str):
                        import json
                        try:
                            config = json.loads(config_data)
                            logger.info("Parsed config from JSON string")
                        except:
                            # Default values if parsing fails
                            logger.warning("Could not parse config, using defaults")
                            config = {'vocab_size': 20000, 'max_length': 100}
                    else:
                        config = config_data
                    
                    # Get vocab_size and max_length with fallback
                    if isinstance(config, dict):
                        vocab_size = config.get('vocab_size', 20000)
                        max_length = config.get('max_length', 100)
                    else:
                        logger.warning(f"Config is not a dict, using defaults. Config type: {type(config)}")
                        vocab_size = 20000
                        max_length = 100
                    
                    logger.info(f"Using: vocab_size={vocab_size}, max_length={max_length}")
                    
                    # Reconstruct Keras tokenizer from saved state
                    keras_tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

                    # Get word_index and validate it
                    word_index_data = loaded_data['word_index']
                    logger.info(f"word_index type: {type(word_index_data)}")

                    # CRITICAL FIX: Ensure word_index is a proper dict with string keys
                    if isinstance(word_index_data, dict):
                        # Ensure all keys are strings (Keras requirement)
                        cleaned_word_index = {}
                        for key, value in word_index_data.items():
                            if isinstance(key, str):
                                cleaned_word_index[key] = int(value)
                            else:
                                # Convert non-string keys to strings
                                cleaned_word_index[str(key)] = int(value)
                        
                        keras_tokenizer.word_index = cleaned_word_index
                        logger.info(f"✓ word_index properly formatted with {len(cleaned_word_index)} entries")
                    elif isinstance(word_index_data, str):
                        logger.error("word_index is a STRING - tokenizer file is corrupted!")
                        logger.error(f"word_index value: {word_index_data[:200]}")
                        raise ValueError("Tokenizer file corrupted: word_index is a string instead of dict")
                    else:
                        logger.error(f"Unexpected word_index type: {type(word_index_data)}")
                        raise ValueError(f"Cannot load tokenizer: word_index has type {type(word_index_data)}")

                    # Restore index_word if available
                    if 'index_word' in loaded_data:
                        index_word_data = loaded_data['index_word']
                        if isinstance(index_word_data, dict):
                            # Ensure keys are integers and values are strings
                            cleaned_index_word = {}
                            for key, value in index_word_data.items():
                                try:
                                    cleaned_index_word[int(key)] = str(value)
                                except (ValueError, TypeError):
                                    logger.warning(f"Skipping invalid index_word entry: {key} → {value}")
                            keras_tokenizer.index_word = cleaned_index_word
                        else:
                            logger.warning(f"index_word has unexpected type: {type(index_word_data)}")

                    # Restore additional tokenizer config if available
                    if 'tokenizer_config' in loaded_data:
                        tokenizer_config = loaded_data['tokenizer_config']
                        logger.info(f"Tokenizer config type: {type(tokenizer_config)}")
                        if isinstance(tokenizer_config, dict):
                            for key, value in tokenizer_config.items():
                                if hasattr(keras_tokenizer, key) and key not in ['word_index', 'index_word']:
                                    try:
                                        setattr(keras_tokenizer, key, value)
                                    except Exception as e:
                                        logger.warning(f"Could not set {key}: {e}")

                    # Create wrapper class with proper error handling
                    class KerasTokenizerWrapper:
                        def __init__(self, tokenizer, max_length):
                            self.tokenizer = tokenizer
                            self.max_length = max_length
                            self.is_fitted = True
                        
                        def texts_to_padded_sequences(self, texts):
                            """Convert texts to padded sequences."""
                            # DEFENSIVE: Ensure texts is a list of strings
                            if isinstance(texts, str):
                                logger.warning("texts_to_padded_sequences received string, converting to list")
                                texts = [texts]
                            elif not isinstance(texts, (list, tuple)):
                                logger.warning(f"texts_to_padded_sequences received {type(texts)}, converting")
                                texts = list(texts)
                            
                            # Ensure all elements are Python strings (not numpy strings)
                            texts = [str(t) for t in texts]
                            
                            logger.info(f"Converting {len(texts)} texts to sequences")
                            
                            try:
                                sequences = self.tokenizer.texts_to_sequences(texts)
                            except AttributeError as e:
                                logger.error(f"AttributeError in texts_to_sequences: {e}")
                                logger.error(f"tokenizer.word_index type: {type(self.tokenizer.word_index)}")
                                if hasattr(self.tokenizer, 'word_index'):
                                    logger.error(f"word_index keys sample: {list(self.tokenizer.word_index.keys())[:5]}")
                                raise
                            except Exception as e:
                                logger.error(f"Error in texts_to_sequences: {e}")
                                raise
                            
                            logger.info(f"Sequences created, padding to length {self.max_length}")
                            padded = pad_sequences(sequences, maxlen=self.max_length, 
                                                padding='post', truncating='post')
                            logger.info(f"Padded sequences shape: {padded.shape}")
                            return padded

                    # Use the wrapper
                    self.tokenizer = KerasTokenizerWrapper(keras_tokenizer, max_length)

                    logger.info("✓ Keras tokenizer successfully reconstructed")
                    logger.info(f"  Vocabulary size: {len(keras_tokenizer.word_index)}")
                    logger.info(f"  Sample words: {list(keras_tokenizer.word_index.keys())[:10]}")

                
                else:
                    error_msg = f"Dict format tokenizer has unexpected keys: {list(loaded_data.keys())}"
                    logger.error(error_msg)
                    if self.model_type == 'deep_learning':
                        raise RuntimeError(error_msg)
                    self.tokenizer = None
            else:
                error_msg = f"Unexpected tokenizer type: {type(loaded_data)}"
                logger.error(error_msg)
                if self.model_type == 'deep_learning':
                    raise RuntimeError(error_msg)
                self.tokenizer = None
            
            # Verify tokenizer has required method
            if self.tokenizer and not hasattr(self.tokenizer, 'texts_to_padded_sequences'):
                error_msg = f"Tokenizer missing 'texts_to_padded_sequences' method"
                logger.error(error_msg)
                if self.model_type == 'deep_learning':
                    raise RuntimeError(error_msg)
                self.tokenizer = None
            
            if self.tokenizer:
                logger.info("✓ Tokenizer loaded and verified successfully")
        
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            import traceback
            traceback.print_exc()
            if self.model_type == 'deep_learning':
                raise RuntimeError(f"Failed to load tokenizer required for deep learning models: {e}")
            self.tokenizer = None
    
    def _load_model(self, model_name: str):
        """Load the specified model (Traditional ML, PyTorch DL, or BERT)."""
        model_key = model_name.lower().replace(' ', '_').replace('-', '_')
        
        try:
            if self.model_type == 'bert':
                self._load_bert_model(model_name, model_key)
            elif self.model_type == 'deep_learning':
                self._load_pytorch_dl_model(model_name, model_key)
            else:
                self._load_traditional_model(model_key)
            
            logger.info(f"Loaded model: {model_name} ({self.model_type})")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def _load_traditional_model(self, model_key: str):
        """Load a traditional ML model."""
        if model_key not in MODEL_FILES:
            available = [k for k in MODEL_FILES.keys() if k not in ['tokenizer', 'feature_extractor']]
            raise ValueError(f"Model '{model_key}' not found. Available: {available}")
        
        model_path = MODEL_FILES[model_key]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Loaded traditional ML model from {model_path}")
    
    def _load_pytorch_dl_model(self, model_name: str, model_key: str):
        """Load a PyTorch deep learning model (.pt file)."""
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Cannot load deep learning models.")
        
        # Import PyTorch model classes
        try:
            if 'cnn' in model_key:
                from models.deep_learning.cnn_model import CNNModel
                from config import CNN_CONFIG
                model_class = CNNModel
                config = CNN_CONFIG
            elif 'bilstm' in model_key:
                from models.deep_learning.bilstm_model import BiLSTMModel
                from config import BILSTM_CONFIG
                model_class = BiLSTMModel
                config = BILSTM_CONFIG
            elif 'lstm' in model_key:
                from models.deep_learning.lstm_model import LSTMModel
                from config import LSTM_CONFIG
                model_class = LSTMModel
                config = LSTM_CONFIG
            else:
                raise ValueError(f"Unknown PyTorch DL model type: {model_key}")
            
            # Create model instance
            self.model = model_class(config=config)
            
            # Load PyTorch state dict
            model_path = MODEL_FILES.get(model_key)
            if not model_path:
                # Fallback: try to find .pt file in deep_learning directory
                dl_dir = MODELS_DIR / 'deep_learning'
                model_path = dl_dir / f"{model_key}.pt"
            
            if not model_path.exists():
                raise FileNotFoundError(f"PyTorch model file not found: {model_path}")
            
            # Build model architecture first
            self.model.build_model()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            logger.info(f"Loaded PyTorch checkpoint from {model_path}")
            
            # Extract state_dict from checkpoint (FIX for the error)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("Extracted 'model_state_dict' from checkpoint")
                
                # Log additional checkpoint info if available
                if 'config' in checkpoint:
                    logger.info(f"Checkpoint config available")
                if 'history' in checkpoint:
                    logger.info(f"Training history available in checkpoint")
            else:
                # Checkpoint is already a state_dict
                state_dict = checkpoint
                logger.info("Using checkpoint directly as state_dict")
            
            # Load state dict into model
            self.model.model.load_state_dict(state_dict)
            self.model.model.eval()
            
            logger.info(f"PyTorch DL model loaded successfully and set to eval mode")
            
        except Exception as e:
            logger.error(f"Error loading PyTorch DL model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_bert_model(self, model_name: str, model_key: str):
        """Load a BERT model from saved directory."""
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Cannot load BERT models.")
        
        try:
            # Import BERT model class
            from bert_model_heavy_gpu import HeavyGPUBERTModel
            
            # BERT models are saved in directories like saved_models/bert_bert_base
            bert_dir = MODELS_DIR / f"bert_{model_key}"
            
            # Alternative: try without 'bert_' prefix
            if not bert_dir.exists():
                bert_dir = MODELS_DIR / model_key
            
            if not bert_dir.exists():
                raise FileNotFoundError(f"BERT model directory not found: {bert_dir}")
            
            logger.info(f"Loading BERT model from {bert_dir}")
            
            # Load the BERT model using the static load method
            self.model = HeavyGPUBERTModel.load(str(bert_dir))
            
            logger.info(f"BERT model loaded successfully from {bert_dir}")
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _prepare_input(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Prepare input based on model type."""
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type == 'traditional':
            # Use feature extractor
            if self.feature_extractor is None:
                raise RuntimeError("Feature extractor not loaded")
            return self.feature_extractor.transform(texts)
        
        elif self.model_type == 'bert':
            # BERT uses raw text - return as-is
            return texts
        
        else:  # deep_learning
            # Use tokenizer for PyTorch DL
            if self.tokenizer is None:
                error_msg = (
                    f"Tokenizer not loaded. Cannot use deep learning model '{self.model_name}'.\n"
                    f"Possible causes:\n"
                    f"1. Tokenizer file doesn't exist at the configured path\n"
                    f"2. Tokenizer failed to load (check logs above for errors)\n"
                    f"3. You need to train the tokenizer first\n"
                    f"\nSolutions:\n"
                    f"- Run: python diagnose_tokenizer.py (to check tokenizer status)\n"
                    f"- Or use a traditional ML model instead: model_name='xgboost' or 'mlp'\n"
                    f"- Or train/retrain the deep learning models to generate the tokenizer"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if not hasattr(self.tokenizer, 'texts_to_padded_sequences'):
                raise AttributeError(
                    f"Tokenizer (type: {type(self.tokenizer)}) doesn't have "
                    "'texts_to_padded_sequences' method."
                )
            
            try:
                # Additional validation before calling tokenizer
                if not isinstance(texts, (list, tuple)):
                    texts = [texts]
                
                # Ensure all elements are Python strings
                texts = [str(t) if not isinstance(t, str) else t for t in texts]
                
                logger.debug(f"Prepared texts type: {type(texts)}, sample: {texts[:50] if texts else 'empty'}")
                return self.tokenizer.texts_to_padded_sequences(texts)
            except AttributeError as e:
                logger.error(f"Attribute error in tokenizer: {e}")
                logger.error(f"texts: {texts}")
                logger.error(f"tokenizer type: {type(self.tokenizer)}")
                raise
            except Exception as e:
                logger.error(f"Error tokenizing texts: {e}")
                logger.error(f"texts type: {type(texts)}, length: {len(texts) if hasattr(texts, '__len__') else 'N/A'}")
                raise
 
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict class labels for texts.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            Array of predicted class labels (0, 1, or 2)
        """
        single_input = isinstance(texts, str)
        
        # Prepare input
        input_data = self._prepare_input(texts)
        
        # Predict
        predictions = self.model.predict(input_data)
        
        # Return single value if single input
        if single_input:
            return predictions[0] if hasattr(predictions, '__len__') else predictions
        
        return predictions
    
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Predict class probabilities for texts.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            Array of probabilities for each class
        """
        single_input = isinstance(texts, str)
        
        # Prepare input
        input_data = self._prepare_input(texts)
        
        # Predict probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_data)
        else:
            # If model doesn't support probabilities, create one-hot
            predictions = self.model.predict(input_data)
            probabilities = np.zeros((len(predictions) if hasattr(predictions, '__len__') else 1, len(CLASS_LABELS)))
            if hasattr(predictions, '__len__'):
                for i, pred in enumerate(predictions):
                    probabilities[i, int(pred)] = 1.0
            else:
                probabilities[0, int(predictions)] = 1.0
        
        # Return single value if single input
        if single_input:
            return probabilities[0] if len(probabilities.shape) > 1 else probabilities
        
        return probabilities
    
    def predict_with_details(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Predict with detailed information.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            List of dictionaries with prediction details
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Get predictions and probabilities
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Build results
        results = []
        for i, text in enumerate(texts):
            if hasattr(predictions, '__len__'):
                pred_class = int(predictions[i])
                probs = probabilities[i] if len(probabilities.shape) > 1 else probabilities
            else:
                pred_class = int(predictions)
                probs = probabilities
            
            result = {
                'text': text,
                'prediction': CLASS_LABELS[pred_class],
                'class': pred_class,
                'confidence': float(probs[pred_class]),
                'probabilities': {
                    CLASS_LABELS[0]: float(probs[0]),
                    CLASS_LABELS[1]: float(probs[1]),
                    CLASS_LABELS[2]: float(probs[2])
                },
                'model_name': self.model_name,
                'model_type': self.model_type
            }
            
            results.append(result)
        
        return results
    
    def _load_severity_components(self):
        """Lazy load Phase 4 severity components."""
        if self.severity_scorer is None or self.action_recommender is None:
            try:
                from severity.severity_classifier import SeverityScorer
                from severity.action_recommender import ActionRecommender
                
                self.severity_scorer = SeverityScorer()
                self.action_recommender = ActionRecommender()
                
                logger.info("Phase 4 severity components loaded")
            except Exception as e:
                logger.error(f"Error loading Phase 4 components: {e}")
                raise RuntimeError(
                    "Could not load Phase 4 severity components. "
                    "Make sure severity/ module is available."
                )
    
    def classify_with_severity(self, text: str, verbose: bool = False) -> Dict:
        """
        Complete classification with severity analysis and action recommendations (Phase 4).
        
        This is the main method for Phase 4 that combines:
        1. Base classification (Hate/Offensive/Neither)
        2. Severity analysis (LOW to EXTREME)
        3. Action recommendations (WARNING to BAN)
        
        Args:
            text: Tweet text to classify
            verbose: If True, print detailed output
        
        Returns:
            Dictionary with complete analysis
        """
        # Load Phase 4 components if not already loaded
        self._load_severity_components()
        
        # Step 1: Get base classification
        base_result = self.predict_with_details(text)[0]
        
        # Step 2: Get severity analysis
        severity_result = self.severity_scorer.analyze_severity(text, verbose=False)
        
        # Step 3: Get action recommendation
        action_result = self.action_recommender.format_recommendation(
            predicted_class=base_result['class'],
            severity_level=severity_result['severity_level'],
            class_name=base_result['prediction'],
            severity_name=severity_result['severity_label']
        )
        
        # Step 4: Combine all results
        complete_result = {
            # Base classification (Phase 1-3)
            'text': base_result['text'],
            'prediction': base_result['prediction'],
            'class': base_result['class'],
            'confidence': base_result['confidence'],
            'probabilities': base_result['probabilities'],
            'model_info': {
                'name': base_result['model_name'],
                'type': base_result['model_type']
            },
            
            # Severity analysis (Phase 4)
            'severity': {
                'severity_score': severity_result['severity_score'],
                'base_score': severity_result['base_score'],
                'severity_level': severity_result['severity_level'],
                'severity_label': severity_result['severity_label'],
                'factors': severity_result['factors'],
                'context_adjustments': severity_result['context_adjustments'],
                'explanation': severity_result['explanation']
            },
            
            # Action recommendation (Phase 4)
            'action': {
                'primary_action': action_result['primary_action'],
                'additional_actions': action_result['additional_actions'],
                'all_actions': action_result['all_actions'],
                'action_string': action_result['action_string'],
                'descriptions': action_result['descriptions'],
                'urgency': action_result['urgency'],
                'reasoning': action_result['reasoning']
            }
        }
        
        if verbose:
            self._print_complete_classification(complete_result)
        
        return complete_result
    
    def classify_tweet(self, tweet: str, verbose: bool = True) -> Dict:
        """
        Classify a single tweet with optional verbose output (Phase 1-3 only).
        
        For complete Phase 4 output, use classify_with_severity() instead.
        
        Args:
            tweet: Tweet text
            verbose: Whether to print detailed output
        
        Returns:
            Dictionary with classification results
        """
        result = self.predict_with_details(tweet)[0]
        
        if verbose:
            self._print_classification(result)
        
        return result
    
    def _print_classification(self, result: Dict):
        """Print classification result (Phase 1-3)."""
        print("\n" + "=" * 70)
        print("CLASSIFICATION RESULT (PHASE 1-3)")
        print("=" * 70)
        print(f"\nText: {result['text'][:100]}...")
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Model: {result['model_name']} ({result['model_type']})")
        print(f"\nProbability Breakdown:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name:20s}: {prob:.2%}")
        print("=" * 70)
    
    def _print_complete_classification(self, result: Dict):
        """Print complete classification with severity and actions (Phase 4)."""
        print("\n" + "=" * 80)
        print("COMPLETE CLASSIFICATION RESULT (PHASE 4)")
        print("=" * 80)
        
        # Text
        print(f"\nText: {result['text'][:150]}...")
        
        # Base Classification
        print("\n" + "-" * 80)
        print("BASE CLASSIFICATION")
        print("-" * 80)
        print(f"Prediction: {result['prediction']} (Class {result['class']})")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Model: {result['model_info']['name']} ({result['model_info']['type']})")
        print(f"\nProbability Breakdown:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name:20s}: {prob:.2%}")
        
        # Severity Analysis
        print("\n" + "-" * 80)
        print("SEVERITY ANALYSIS")
        print("-" * 80)
        severity = result['severity']
        print(f"Severity Level: {severity['severity_label']} (Level {severity['severity_level']})")
        print(f"Severity Score: {severity['severity_score']}/100 (Base: {severity['base_score']})")
        print(f"\nKey Factors:")
        factors = severity['factors']
        if factors['violence_keywords'] > 0:
            print(f"  - Violence keywords: {factors['violence_keywords']}")
        if factors['threat_patterns'] > 0:
            print(f"  - Explicit threats: {factors['threat_patterns']}")
        if factors['racial_slurs'] > 0:
            print(f"  - Racial slurs: {factors['racial_slurs']}")
        if factors['lgbtq_slurs'] > 0:
            print(f"  - LGBTQ slurs: {factors['lgbtq_slurs']}")
        if factors['sexist_slurs'] > 0:
            print(f"  - Sexist slurs: {factors['sexist_slurs']}")
        if factors['dehumanization_terms'] > 0:
            print(f"  - Dehumanization terms: {factors['dehumanization_terms']}")
        if factors['caps_ratio'] > 0.5:
            print(f"  - Excessive caps: {factors['caps_ratio']:.0%}")
        if factors['targeted_at_person']:
            print(f"  - Targeted at person: Yes")
        
        if severity['context_adjustments']['reasons']:
            print(f"\nContext Adjustments:")
            for reason in severity['context_adjustments']['reasons']:
                print(f"  - {reason}")
        
        print(f"\nExplanation: {severity['explanation']}")
        
        # Action Recommendation
        print("\n" + "-" * 80)
        print("ACTION RECOMMENDATION")
        print("-" * 80)
        action = result['action']
        print(f"Urgency: {action['urgency']}")
        print(f"Primary Action: {action['primary_action']}")
        if action['additional_actions']:
            print(f"Additional Actions: {', '.join(action['additional_actions'])}")
        print(f"\nComplete Action: {action['action_string']}")
        print(f"\nAction Details:")
        for act, desc in zip(action['all_actions'], action['descriptions']):
            print(f"  - {act}")
            print(f"    -> {desc}")
        print(f"\nReasoning: {action['reasoning']}")
        
        print("=" * 80)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_class': type(self.model).__name__,
        }
        
        if self.model_name in self.metadata:
            info['metrics'] = self.metadata[self.model_name]
        
        return info


# ==================== CONVENIENCE FUNCTIONS ====================

def classify_tweet(tweet: str, model_name: str = 'best') -> Dict:
    """
    Convenience function to classify a single tweet (Phase 1-3).
    
    Args:
        tweet: Tweet text
        model_name: Model to use
    
    Returns:
        Classification result dictionary
    """
    classifier = TweetClassifier(model_name=model_name)
    return classifier.classify_tweet(tweet, verbose=False)


def classify_tweets(tweets: List[str], model_name: str = 'best') -> List[Dict]:
    """
    Convenience function to classify multiple tweets (Phase 1-3).
    
    Args:
        tweets: List of tweet texts
        model_name: Model to use
    
    Returns:
        List of classification results
    """
    classifier = TweetClassifier(model_name=model_name)
    return classifier.predict_with_details(tweets)


def classify_with_severity(tweet: str, model_name: str = 'best') -> Dict:
    """
    Convenience function for complete classification with severity (Phase 4).
    
    Args:
        tweet: Tweet text
        model_name: Model to use
    
    Returns:
        Complete classification with severity and actions
    """
    classifier = TweetClassifier(model_name=model_name)
    return classifier.classify_with_severity(tweet, verbose=False)


# ==================== DEMO FUNCTIONS ====================

def demo_basic():
    """Demo 1: Basic classification (Phase 1-3 only)."""
    print("\n" + "=" * 80)
    print("DEMO 1: BASIC CLASSIFICATION (PHASE 1-3)")
    print("=" * 80)
    
    tweets = [
        "I will kill you fucking bitch",
        "You're such an idiot",
        "Have a great day everyone!",
        "This movie was terrible"
    ]
    
    try:
        classifier = TweetClassifier()
        
        print(f"\nUsing model: {classifier.model_name} ({classifier.model_type})")
        
        for i, tweet in enumerate(tweets, 1):
            print(f"\n--- Tweet {i} ---")
            result = classifier.classify_tweet(tweet, verbose=False)
            print(f"Text: {tweet}")
            print(f"Prediction: {result['prediction']} ({result['confidence']:.1%} confidence)")
            print(f"Model: {result['model_name']}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_with_severity():
    """Demo 2: Complete classification with severity (Phase 4)."""
    print("\n" + "=" * 80)
    print("DEMO 2: COMPLETE CLASSIFICATION WITH SEVERITY (PHASE 4)")
    print("=" * 80)
    
    tweets = [
        {
            'text': "I will kill you and your family you fucking bitch",
            'expected': "EXTREME severity"
        },
        {
            'text': "You're a stupid idiot",
            'expected': "MODERATE severity"
        },
        {
            'text': "Good morning everyone!",
            'expected': "LOW/None severity"
        }
    ]
    
    try:
        classifier = TweetClassifier()
        
        for i, sample in enumerate(tweets, 1):
            print(f"\n{'=' * 80}")
            print(f"SAMPLE {i}: {sample['expected']}")
            print('=' * 80)
            
            result = classifier.classify_with_severity(sample['text'], verbose=False)
            
            print(f"\nText: {sample['text']}")
            print(f"\nClassification: {result['prediction']} ({result['confidence']:.1%})")
            print(f"Severity: {result['severity']['severity_label']} (Score: {result['severity']['severity_score']}/100)")
            print(f"Action: {result['action']['action_string']}")
            print(f"Urgency: {result['action']['urgency']}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("HATE SPEECH CLASSIFIER - DEMOS")
    print("=" * 80)
    
    demos = [
        ("Basic Classification", demo_basic),
        ("With Severity Analysis", demo_with_severity)
    ]
    
    for name, demo_func in demos:
        print(f"\n{'#' * 80}")
        print(f"# {name}")
        print(f"{'#' * 80}")
        try:
            demo_func()
        except Exception as e:
            print(f"\nDemo failed: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n\nPress Enter to continue to next demo...")


# ==================== TESTING ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        if demo_name == 'basic':
            demo_basic()
        elif demo_name == 'severity':
            demo_with_severity()
        else:
            print(f"Unknown demo: {demo_name}")
            print("Available demos: basic, severity")
    else:
        demo()