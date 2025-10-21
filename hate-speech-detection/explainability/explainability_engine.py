"""
Explainability Engine for Hate Speech Detection
Provides interpretable explanations for model predictions using multiple XAI techniques

Techniques:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. SHAP (SHapley Additive exPlanations) - if available
3. Feature Importance Analysis
4. Keyword Highlighting
5. Attention-based Explanations (for deep learning models)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from config import (
    CLASS_LABELS, LIME_CONFIG, SHAP_CONFIG,
    VIOLENCE_KEYWORDS, THREAT_PATTERNS, DEHUMANIZATION_KEYWORDS,
    RACIAL_SLURS, LGBTQ_SLURS, SEXIST_SLURS, RELIGIOUS_SLURS, ABLEIST_SLURS
)
from utils import logger

# Try importing LIME
try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    logger.warning("LIME not installed. Install with: pip install lime")

# Try importing SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed. Install with: pip install shap")

# ==================== BASE EXPLAINER ====================

class BaseExplainer:
    """Base class for all explainers."""
    
    def __init__(self, model, feature_extractor=None, tokenizer=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model
            feature_extractor: Feature extractor (for traditional ML)
            tokenizer: Tokenizer (for deep learning)
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.model_type = self._detect_model_type()
    
    def _detect_model_type(self) -> str:
        """Detect if model is traditional ML or deep learning."""
        model_class = type(self.model).__name__.lower()
        
        if any(x in model_class for x in ['lstm', 'bilstm', 'cnn', 'bert']):
            return 'deep_learning'
        else:
            return 'traditional'
    
    def explain(self, text: str, **kwargs) -> Dict:
        """
        Generate explanation for a prediction.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

# ==================== KEYWORD-BASED EXPLAINER ====================

class KeywordExplainer(BaseExplainer):
    """
    Simple keyword-based explanations.
    Highlights harmful keywords that contributed to the prediction.
    """
    
    def __init__(self, model=None, feature_extractor=None, tokenizer=None):
        super().__init__(model, feature_extractor, tokenizer)
        
        # Compile all keyword categories
        self.keyword_categories = {
            'violence': set(k.lower() for k in VIOLENCE_KEYWORDS),
            'threats': set(p.lower() for p in THREAT_PATTERNS),
            'dehumanization': set(k.lower() for k in DEHUMANIZATION_KEYWORDS),
            'racial_slurs': set(k.lower() for k in RACIAL_SLURS),
            'lgbtq_slurs': set(k.lower() for k in LGBTQ_SLURS),
            'sexist_slurs': set(k.lower() for k in SEXIST_SLURS),
            'religious_slurs': set(k.lower() for k in RELIGIOUS_SLURS),
            'ableist_slurs': set(k.lower() for k in ABLEIST_SLURS)
        }
    
    def _find_keywords_in_text(self, text: str) -> Dict[str, List[str]]:
        """Find all harmful keywords in text."""
        text_lower = text.lower()
        found = {}
        
        for category, keywords in self.keyword_categories.items():
            matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(keyword)
            
            if matches:
                found[category] = matches
        
        return found
    
    def explain(self, text: str, predicted_class: int, **kwargs) -> Dict:
        """
        Generate keyword-based explanation.
        
        Args:
            text: Input text
            predicted_class: Predicted class (0=Hate, 1=Offensive, 2=Neither)
        
        Returns:
            Dictionary with explanation
        """
        found_keywords = self._find_keywords_in_text(text)
        
        # Build explanation text
        if not found_keywords:
            explanation = "No explicit harmful keywords detected."
            if predicted_class in [0, 1]:
                explanation += " Classification based on context and patterns."
        else:
            parts = []
            for category, keywords in found_keywords.items():
                category_name = category.replace('_', ' ').title()
                parts.append(f"{category_name}: {', '.join(keywords)}")
            
            explanation = "Detected harmful content: " + "; ".join(parts)
        
        return {
            'method': 'keyword_analysis',
            'explanation': explanation,
            'found_keywords': found_keywords,
            'total_categories': len(found_keywords),
            'predicted_class': CLASS_LABELS[predicted_class]
        }

# ==================== LIME EXPLAINER ====================

class LIMEExplainer(BaseExplainer):
    """
    LIME-based explanations.
    Shows which words contributed most to the prediction.
    """
    
    def __init__(self, model, feature_extractor=None, tokenizer=None):
        super().__init__(model, feature_extractor, tokenizer)
        
        if not HAS_LIME:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        # Initialize LIME explainer
        class_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
        self.explainer = LimeTextExplainer(class_names=class_names)
    
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Wrapper for model prediction that LIME can use."""
        if self.model_type == 'traditional':
            # Traditional ML: need feature extraction
            if self.feature_extractor is None:
                raise ValueError("Feature extractor required for traditional ML models")
            
            features = self.feature_extractor.transform(texts)
            return self.model.predict_proba(features)
        
        else:
            # Deep learning: handle different input formats
            model_class = type(self.model).__name__.lower()
            
            if 'bert' in model_class:
                # BERT uses raw text
                return self.model.predict_proba(texts)
            else:
                # LSTM/BiLSTM/CNN use sequences
                if self.tokenizer is None:
                    raise ValueError("Tokenizer required for deep learning models")
                
                sequences = self.tokenizer.texts_to_padded_sequences(texts)
                return self.model.predict_proba(sequences)
    
    def explain(
        self, 
        text: str, 
        predicted_class: int,
        num_features: int = None,
        num_samples: int = None
    ) -> Dict:
        """
        Generate LIME explanation.
        
        Args:
            text: Input text
            predicted_class: Predicted class
            num_features: Number of features to show (default from config)
            num_samples: Number of samples for LIME (default from config)
        
        Returns:
            Dictionary with LIME explanation
        """
        num_features = num_features or LIME_CONFIG['num_features']
        num_samples = num_samples or LIME_CONFIG['num_samples']
        
        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=[predicted_class]
        )
        
        # Extract feature weights
        feature_weights = exp.as_list(label=predicted_class)
        
        # Separate positive and negative contributions
        positive_words = [(word, weight) for word, weight in feature_weights if weight > 0]
        negative_words = [(word, weight) for word, weight in feature_weights if weight < 0]
        
        # Sort by absolute weight
        positive_words.sort(key=lambda x: abs(x[1]), reverse=True)
        negative_words.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Build explanation text
        explanation_parts = []
        
        if positive_words:
            top_positive = positive_words[:3]
            words = [f"'{w}' ({wt:+.3f})" for w, wt in top_positive]
            explanation_parts.append(f"Words supporting this classification: {', '.join(words)}")
        
        if negative_words:
            top_negative = negative_words[:3]
            words = [f"'{w}' ({wt:+.3f})" for w, wt in top_negative]
            explanation_parts.append(f"Words opposing this classification: {', '.join(words)}")
        
        explanation = ". ".join(explanation_parts) if explanation_parts else "No significant word contributions found."
        
        # Get prediction probabilities for all classes
        proba = exp.predict_proba[predicted_class]
        
        return {
            'method': 'lime',
            'explanation': explanation,
            'feature_weights': feature_weights,
            'positive_contributions': positive_words,
            'negative_contributions': negative_words,
            'prediction_probability': float(proba),
            'predicted_class': CLASS_LABELS[predicted_class],
            'lime_object': exp  # For visualization
        }

# ==================== FEATURE IMPORTANCE EXPLAINER ====================

class FeatureImportanceExplainer(BaseExplainer):
    """
    Feature importance-based explanations (for tree-based models).
    Shows which features are most important globally.
    """
    
    def __init__(self, model, feature_extractor=None, tokenizer=None):
        super().__init__(model, feature_extractor, tokenizer)
        
        if self.model_type != 'traditional':
            logger.warning("Feature importance is most relevant for traditional ML models")
    
    def explain(self, text: str, predicted_class: int, top_n: int = 10) -> Dict:
        """
        Generate feature importance explanation.
        
        Args:
            text: Input text
            predicted_class: Predicted class
            top_n: Number of top features to show
        
        Returns:
            Dictionary with feature importance explanation
        """
        # Check if model has feature_importances_
        if not hasattr(self.model, 'feature_importances_'):
            return {
                'method': 'feature_importance',
                'explanation': f"Model type '{type(self.model).__name__}' does not provide feature importances.",
                'available': False
            }
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Get feature names
        if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_names'):
            feature_names = self.feature_extractor.get_feature_names()
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Get top N features
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in top_indices]
        
        # Build explanation
        feature_list = [f"{name}: {imp:.4f}" for name, imp in top_features[:5]]
        explanation = f"Top contributing features globally: {', '.join(feature_list)}"
        
        return {
            'method': 'feature_importance',
            'explanation': explanation,
            'top_features': top_features,
            'predicted_class': CLASS_LABELS[predicted_class],
            'available': True
        }

# ==================== COMPREHENSIVE EXPLAINER ====================

class ComprehensiveExplainer:
    """
    Combines multiple explanation methods for comprehensive insights.
    """
    
    def __init__(self, model, feature_extractor=None, tokenizer=None):
        """
        Initialize comprehensive explainer.
        
        Args:
            model: Trained model
            feature_extractor: Feature extractor (for traditional ML)
            tokenizer: Tokenizer (for deep learning)
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        
        # Initialize all available explainers
        self.explainers = {}
        
        # Keyword explainer (always available)
        self.explainers['keywords'] = KeywordExplainer(model, feature_extractor, tokenizer)
        
        # LIME explainer
        if HAS_LIME:
            try:
                self.explainers['lime'] = LIMEExplainer(model, feature_extractor, tokenizer)
                logger.info("LIME explainer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize LIME: {e}")
        
        # Feature importance explainer
        if hasattr(model, 'feature_importances_'):
            self.explainers['feature_importance'] = FeatureImportanceExplainer(
                model, feature_extractor, tokenizer
            )
            logger.info("Feature importance explainer initialized")
        
        logger.info(f"Initialized {len(self.explainers)} explainers: {list(self.explainers.keys())}")
    
    def explain(
        self, 
        text: str, 
        predicted_class: int,
        methods: List[str] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Generate comprehensive explanation using all available methods.
        
        Args:
            text: Input text
            predicted_class: Predicted class (0=Hate, 1=Offensive, 2=Neither)
            methods: List of methods to use (default: all available)
            verbose: If True, print detailed explanations
        
        Returns:
            Dictionary with all explanations
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        # Generate explanations from each method
        explanations = {}
        
        for method in methods:
            if method not in self.explainers:
                logger.warning(f"Method '{method}' not available")
                continue
            
            try:
                explanation = self.explainers[method].explain(text, predicted_class)
                explanations[method] = explanation
            except Exception as e:
                logger.error(f"Error generating {method} explanation: {e}")
                explanations[method] = {
                    'method': method,
                    'error': str(e),
                    'available': False
                }
        
        # Compile comprehensive result
        result = {
            'text': text,
            'predicted_class': predicted_class,
            'predicted_class_name': CLASS_LABELS[predicted_class],
            'explanations': explanations,
            'methods_used': list(explanations.keys())
        }
        
        if verbose:
            self._print_explanation(result)
        
        return result
    
    def _print_explanation(self, result: Dict):
        """Print formatted explanation."""
        print("\n" + "=" * 80)
        print("EXPLAINABLE AI - PREDICTION EXPLANATION")
        print("=" * 80)
        
        print(f"\nText: {result['text'][:200]}...")
        print(f"Prediction: {result['predicted_class_name']} (Class {result['predicted_class']})")
        
        for method, explanation in result['explanations'].items():
            print(f"\n{'-' * 80}")
            print(f"Method: {method.upper()}")
            print(f"{'-' * 80}")
            
            if 'error' in explanation:
                print(f"Error: {explanation['error']}")
                continue
            
            print(f"Explanation: {explanation['explanation']}")
            
            # Method-specific details
            if method == 'keywords' and 'found_keywords' in explanation:
                if explanation['found_keywords']:
                    print("\nDetected Keywords by Category:")
                    for category, keywords in explanation['found_keywords'].items():
                        print(f"  {category}: {', '.join(keywords)}")
            
            elif method == 'lime' and 'positive_contributions' in explanation:
                print("\nTop Contributing Words:")
                for word, weight in explanation['positive_contributions'][:5]:
                    print(f"  {word}: {weight:+.3f}")
                
                if explanation['negative_contributions']:
                    print("\nTop Opposing Words:")
                    for word, weight in explanation['negative_contributions'][:3]:
                        print(f"  {word}: {weight:+.3f}")
            
            elif method == 'feature_importance' and 'top_features' in explanation:
                print("\nTop Global Features:")
                for feature, importance in explanation['top_features'][:5]:
                    print(f"  {feature}: {importance:.4f}")
        
        print("=" * 80)
    
    def explain_multiple(
        self, 
        texts: List[str], 
        predicted_classes: List[int],
        methods: List[str] = None
    ) -> List[Dict]:
        """
        Generate explanations for multiple texts.
        
        Args:
            texts: List of texts
            predicted_classes: List of predicted classes
            methods: List of methods to use
        
        Returns:
            List of explanation dictionaries
        """
        results = []
        
        for text, pred_class in zip(texts, predicted_classes):
            result = self.explain(text, pred_class, methods=methods, verbose=False)
            results.append(result)
        
        return results

# ==================== CONVENIENCE FUNCTIONS ====================

def create_explainer(
    model,
    feature_extractor=None,
    tokenizer=None
) -> ComprehensiveExplainer:
    """
    Create comprehensive explainer for a model.
    
    Args:
        model: Trained model
        feature_extractor: Feature extractor (for traditional ML)
        tokenizer: Tokenizer (for deep learning)
    
    Returns:
        ComprehensiveExplainer instance
    """
    return ComprehensiveExplainer(model, feature_extractor, tokenizer)


def explain_prediction(
    text: str,
    predicted_class: int,
    model,
    feature_extractor=None,
    tokenizer=None,
    verbose: bool = True
) -> Dict:
    """
    Quick function to explain a single prediction.
    
    Args:
        text: Input text
        predicted_class: Predicted class
        model: Trained model
        feature_extractor: Feature extractor (optional)
        tokenizer: Tokenizer (optional)
        verbose: If True, print explanation
    
    Returns:
        Explanation dictionary
    """
    explainer = create_explainer(model, feature_extractor, tokenizer)
    return explainer.explain(text, predicted_class, verbose=verbose)

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("EXPLAINABILITY ENGINE TEST")
    print("=" * 80)
    
    # Test keyword explainer (works without actual model)
    print("\n1. Testing Keyword Explainer...")
    keyword_explainer = KeywordExplainer()
    
    test_cases = [
        ("I will kill you fucking bitch", 0),  # Hate speech
        ("You're an idiot", 1),  # Offensive
        ("Good morning everyone", 2)  # Neither
    ]
    
    for text, pred_class in test_cases:
        result = keyword_explainer.explain(text, pred_class)
        print(f"\nText: {text}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Explanation: {result['explanation']}")
        if result['found_keywords']:
            print(f"Keywords: {result['found_keywords']}")
    
    print("\n" + "=" * 80)
    print("Basic tests complete!")
    print("\nFor full testing with LIME and model-specific explanations:")
    print("  1. Train a model first (run main_train.py)")
    print("  2. Load the model and use ComprehensiveExplainer")
    print("\nExample:")
    print("  from inference.tweet_classifier import TweetClassifier")
    print("  from explainability.explainability_engine import create_explainer")
    print("  ")
    print("  classifier = TweetClassifier()")
    print("  explainer = create_explainer(")
    print("      classifier.model,")
    print("      classifier.feature_extractor,")
    print("      classifier.tokenizer")
    print("  )")
    print("  ")
    print("  result = explainer.explain('Your text here', predicted_class=0, verbose=True)")