"""
Unified Tweet Classifier - Supports Both Traditional ML and Deep Learning
Automatically selects the best model from ALL trained models

PHASE 1-3: Base classification (Hate/Offensive/Neither)
PHASE 4: Severity analysis + Action recommendations
PHASE 5: Deep learning models (LSTM, BiLSTM, CNN, BERT)
"""

import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_FILES, CLASS_LABELS, RESULTS_DIR
from utils import logger, print_section_header, load_results

# ==================== UNIFIED TWEET CLASSIFIER ====================

class TweetClassifier:
    """
    Unified classifier supporting both Traditional ML and Deep Learning models.
    
    Features:
    - Automatically selects best model (Traditional ML or Deep Learning)
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
            model_type: Type of model ('auto', 'traditional', 'deep_learning')
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
        
        # FIXED: Resolve 'best' to actual model name BEFORE determining type
        if model_name.lower() == 'best':
            actual_model = self._get_best_model_name()
            self.model_name = actual_model
            model_name = actual_model  # Update local variable
            logger.info(f"'best' resolved to: {actual_model}")
        
        # Now determine type and load
        self._determine_model_type(model_name)
        self._load_model_components()
        self._load_model(model_name)
        
        logger.info(f"Classifier ready with model: {self.model_name} ({self.model_type})")
        if self.model_name in self.metadata:
            logger.info(f"Model accuracy: {self.metadata[self.model_name]['accuracy']:.4f}")
            logger.info(f"Model F1-Macro: {self.metadata[self.model_name]['f1_macro']:.4f}")
    
    def _load_metadata(self):
        """Load metadata from both Traditional ML and Deep Learning."""
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
        
        # Load Deep Learning metadata
        try:
            dl_metadata = load_results('dl_model_metadata.json')
            if dl_metadata and 'all_results' in dl_metadata:
                for model_name, metrics in dl_metadata['all_results'].items():
                    self.metadata[model_name] = {
                        'accuracy': metrics.get('accuracy', 0),
                        'f1_macro': metrics.get('f1_macro', 0),
                        'f1_weighted': metrics.get('f1_weighted', 0),
                        'type': 'Deep Learning'
                    }
            logger.info("Deep Learning metadata loaded")
        except Exception as e:
            logger.warning(f"Could not load Deep Learning metadata: {e}")
    
    def _get_best_model_name(self) -> str:
        """Get the name of the best model from all metadata."""
        if not self.metadata:
            logger.warning("No metadata found, defaulting to CNN")
            return 'CNN'
        
        # Find model with highest F1-Macro (better metric for imbalanced data)
        best_model = max(self.metadata.items(), key=lambda x: x[1].get('f1_macro', 0))
        
        logger.info(f"Best model selected: {best_model[0]} (F1-Macro: {best_model[1]['f1_macro']:.4f})")
        return best_model[0]
    
    def _determine_model_type(self, model_name: str):
        """Determine if model is Traditional ML or Deep Learning."""
        if self.model_type != 'auto':
            return
        
        # Check model type (model_name should already be resolved)
        dl_models = ['lstm', 'bilstm', 'cnn', 'bert']
        trad_models = ['randomforest', 'xgboost', 'svm', 'gradientboosting', 'mlp']
        
        model_lower = model_name.lower().replace(' ', '').replace('_', '')
        
        if any(dl in model_lower for dl in dl_models):
            self.model_type = 'deep_learning'
        elif any(trad in model_lower for trad in trad_models):
            self.model_type = 'traditional'
        else:
            # Default based on metadata
            if model_name in self.metadata:
                self.model_type = 'traditional' if self.metadata[model_name]['type'] == 'Traditional ML' else 'deep_learning'
            else:
                logger.warning(f"Unknown model type for {model_name}, defaulting to traditional")
                self.model_type = 'traditional'
    
    def _load_model_components(self):
        """Load required components based on model type."""
        if self.model_type == 'traditional':
            self._load_feature_extractor()
        else:  # deep_learning
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
        """Load the tokenizer (for Deep Learning)."""
        try:
            # Import TextTokenizer class
            from models.deep_learning.text_tokenizer import TextTokenizer
            
            tokenizer_path = MODEL_FILES.get('tokenizer')
            if not tokenizer_path or not tokenizer_path.exists():
                logger.warning(f"Tokenizer file not found at: {tokenizer_path}")
                self.tokenizer = None
                return
            
            # Load tokenizer
            loaded_data = joblib.load(tokenizer_path)
            
            # Handle different tokenizer formats
            if isinstance(loaded_data, TextTokenizer):
                # Already a TextTokenizer object
                self.tokenizer = loaded_data
                logger.info("Tokenizer loaded successfully (TextTokenizer object)")
            
            elif isinstance(loaded_data, dict):
                # Dictionary format - need to reconstruct
                logger.info("Tokenizer loaded as dict, reconstructing TextTokenizer...")
                
                # Check if it has the keras tokenizer
                if 'tokenizer' in loaded_data:
                    # Create new TextTokenizer and restore state
                    self.tokenizer = TextTokenizer(
                        vocab_size=loaded_data.get('vocab_size', 20000),
                        max_length=loaded_data.get('max_length', 100)
                    )
                    self.tokenizer.tokenizer = loaded_data['tokenizer']
                    self.tokenizer.is_fitted = loaded_data.get('is_fitted', True)
                    logger.info("TextTokenizer reconstructed from dict")
                else:
                    logger.error("Dict format tokenizer missing 'tokenizer' key")
                    self.tokenizer = None
            
            else:
                logger.error(f"Unexpected tokenizer type: {type(loaded_data)}")
                self.tokenizer = None
        
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            import traceback
            traceback.print_exc()
            self.tokenizer = None

    def _load_model(self, model_name: str):
        """Load the specified model (model_name should already be resolved)."""
        # Get model file path
        model_key = model_name.lower().replace(' ', '_')
        
        if model_key not in MODEL_FILES:
            available = [k for k in MODEL_FILES.keys() if k not in ['tokenizer', 'feature_extractor']]
            raise ValueError(
                f"Model '{model_name}' not found. Available: {available}"
            )
        
        model_path = MODEL_FILES[model_key]
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Please train the models first."
            )
        
        try:
            if self.model_type == 'deep_learning':
                # Load deep learning model
                self._load_dl_model(model_path, model_key)
            else:
                # Load traditional ML model
                self.model = joblib.load(model_path)
            
            logger.info(f"Loaded model: {model_name} ({self.model_type})")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def _load_dl_model(self, model_path: Path, model_key: str):
        """Load a deep learning model."""
        try:
            if 'bert' in model_key:
                # Load BERT model
                from models.deep_learning.bert_model import BERTModel
                from config import BERT_CONFIG
                self.model = BERTModel(config=BERT_CONFIG)
                self.model.load(str(model_path))
                
            elif 'cnn' in model_key:
                # Load CNN model
                from models.deep_learning.cnn_model import CNNModel
                from config import CNN_CONFIG
                
                # Create model instance
                self.model = CNNModel(config=CNN_CONFIG)
                
                # Load the saved Keras model directly
                import tensorflow as tf
                keras_model = tf.keras.models.load_model(str(model_path))
                self.model.model = keras_model
                
                logger.info(f"CNN model loaded successfully from {model_path}")
                
            elif 'bilstm' in model_key:
                # Load BiLSTM model
                from models.deep_learning.bilstm_model import BiLSTMModel
                from config import BILSTM_CONFIG
                
                self.model = BiLSTMModel(config=BILSTM_CONFIG)
                
                # Load the saved Keras model directly
                import tensorflow as tf
                keras_model = tf.keras.models.load_model(str(model_path))
                self.model.model = keras_model
                
                logger.info(f"BiLSTM model loaded successfully from {model_path}")
                
            elif 'lstm' in model_key:
                # Load LSTM model
                from models.deep_learning.lstm_model import LSTMModel
                from config import LSTM_CONFIG
                
                self.model = LSTMModel(config=LSTM_CONFIG)
                
                # Load the saved Keras model directly
                import tensorflow as tf
                keras_model = tf.keras.models.load_model(str(model_path))
                self.model.model = keras_model
                
                logger.info(f"LSTM model loaded successfully from {model_path}")
                
            else:
                raise ValueError(f"Unknown deep learning model type: {model_key}")
                
        except Exception as e:
            logger.error(f"Error loading deep learning model: {e}")
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
        else:
            # Use tokenizer for deep learning
            if self.tokenizer is None:
                raise RuntimeError(
                    f"Tokenizer not loaded. Cannot use deep learning model '{self.model_name}'. "
                    "Please use a traditional ML model instead (e.g., model_name='mlp')"
                )
            
            # Check if tokenizer has the method
            if not hasattr(self.tokenizer, 'texts_to_padded_sequences'):
                raise AttributeError(
                    f"Tokenizer (type: {type(self.tokenizer)}) doesn't have "
                    "'texts_to_padded_sequences' method."
                )
            
            # Tokenize texts
            try:
                return self.tokenizer.texts_to_padded_sequences(texts)
            except Exception as e:
                logger.error(f"Error tokenizing texts: {e}")
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
        if self.model_type == 'deep_learning' and 'bert' in self.model_name.lower():
            # BERT uses raw text
            input_data = [texts] if single_input else texts
        else:
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
        if self.model_type == 'deep_learning' and 'bert' in self.model_name.lower():
            input_data = [texts] if single_input else texts
        else:
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
    
    # Sample tweets
    tweets = [
        "I will kill you fucking bitch",
        "You're such an idiot",
        "Have a great day everyone!",
        "This movie was terrible"
    ]
    
    try:
        # Initialize classifier (uses best model)
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
    
    # Sample tweets with different severity levels
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


def demo_compare_models():
    """Demo 3: Compare different models."""
    print("\n" + "=" * 80)
    print("DEMO 3: COMPARE DIFFERENT MODELS")
    print("=" * 80)
    
    tweet = "I will fucking kill you bitch"
    models_to_test = ['best', 'xgboost', 'cnn', 'bilstm']
    
    print(f"\nTest tweet: {tweet}\n")
    
    for model_name in models_to_test:
        try:
            print(f"\n--- Testing {model_name.upper()} ---")
            classifier = TweetClassifier(model_name=model_name)
            result = classifier.classify_tweet(tweet, verbose=False)
            
            print(f"Model: {result['model_name']} ({result['model_type']})")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.1%}")
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")


def demo_batch_classification():
    """Demo 4: Batch classification of multiple tweets."""
    print("\n" + "=" * 80)
    print("DEMO 4: BATCH CLASSIFICATION")
    print("=" * 80)
    
    tweets = [
        "I hate you so much",
        "You're wonderful!",
        "Go kill yourself",
        "That's interesting",
        "Fuck off bitch"
    ]
    
    try:
        classifier = TweetClassifier()
        
        print(f"\nClassifying {len(tweets)} tweets with {classifier.model_name}...\n")
        
        results = classifier.predict_with_details(tweets)
        
        for i, (tweet, result) in enumerate(zip(tweets, results), 1):
            print(f"{i}. \"{tweet}\"")
            print(f"   -> {result['prediction']} ({result['confidence']:.1%})")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo_detailed_analysis():
    """Demo 5: Detailed analysis with all information."""
    print("\n" + "=" * 80)
    print("DEMO 5: DETAILED ANALYSIS (ALL FEATURES)")
    print("=" * 80)
    
    tweet = "I will fucking kill you, you fucking bitch you whore"
    
    try:
        classifier = TweetClassifier()
        
        print(f"\nAnalyzing tweet: \"{tweet}\"\n")
        
        # Get complete analysis with verbose output
        result = classifier.classify_with_severity(tweet, verbose=True)
        
        # Also show model info
        print("\n" + "=" * 80)
        print("MODEL INFORMATION")
        print("=" * 80)
        info = classifier.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Type: {info['model_type']}")
        print(f"Class: {info['model_class']}")
        if 'metrics' in info:
            print(f"\nModel Performance:")
            print(f"  Accuracy: {info['metrics']['accuracy']:.4f}")
            print(f"  F1-Macro: {info['metrics']['f1_macro']:.4f}")
            print(f"  F1-Weighted: {info['metrics']['f1_weighted']:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def demo():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("HATE SPEECH CLASSIFIER - ALL DEMOS")
    print("=" * 80)
    
    demos = [
        ("Basic Classification", demo_basic),
        ("With Severity Analysis", demo_with_severity),
        ("Compare Models", demo_compare_models),
        ("Batch Classification", demo_batch_classification),
        ("Detailed Analysis", demo_detailed_analysis)
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
        # Run specific demo
        demo_name = sys.argv[1].lower()
        if demo_name == 'basic':
            demo_basic()
        elif demo_name == 'severity':
            demo_with_severity()
        elif demo_name == 'compare':
            demo_compare_models()
        elif demo_name == 'batch':
            demo_batch_classification()
        elif demo_name == 'detailed':
            demo_detailed_analysis()
        else:
            print(f"Unknown demo: {demo_name}")
            print("Available demos: basic, severity, compare, batch, detailed")
    else:
        # Run all demos
        demo()