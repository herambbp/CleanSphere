"""
Explainable Tweet Classifier
Extends TweetClassifier with explainability features

Usage:
    from inference.explainable_classifier import ExplainableTweetClassifier
    
    classifier = ExplainableTweetClassifier()
    result = classifier.classify_with_explanation("Your tweet here")
"""

import sys
sys.path.append('..')

from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from inference.tweet_classifier import TweetClassifier
from explainability.explainability_engine import (
    ComprehensiveExplainer,
    KeywordExplainer,
    LIMEExplainer,
    FeatureImportanceExplainer,
    HAS_LIME
)
from config import CLASS_LABELS
from utils import logger

# ==================== EXPLAINABLE TWEET CLASSIFIER ====================

class ExplainableTweetClassifier(TweetClassifier):
    """
    Extended TweetClassifier with explainability features.
    
    Provides:
    - All original classification features
    - Keyword-based explanations
    - LIME explanations (if available)
    - Feature importance (for tree models)
    - Visualization options
    """
    
    def __init__(self, model_name: str = 'best', model_type: str = 'auto'):
        """
        Initialize explainable classifier.
        
        Args:
            model_name: Name of model to use
            model_type: Type of model ('auto', 'traditional', 'deep_learning')
        """
        # Initialize base classifier
        super().__init__(model_name=model_name, model_type=model_type)
        
        # Initialize explainer
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize the comprehensive explainer."""
        logger.info("Initializing explainability engine...")
        
        try:
            self.explainer = ComprehensiveExplainer(
                model=self.model,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer
            )
            logger.info("Explainability engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing explainer: {e}")
            self.explainer = None
    
    def classify_with_explanation(
        self, 
        text: str,
        methods: List[str] = None,
        include_severity: bool = True,
        verbose: bool = False
    ) -> Dict:
        """
        Classify text and provide explanation.
        
        Args:
            text: Tweet text
            methods: List of explanation methods to use (default: all available)
            include_severity: Include Phase 4 severity analysis
            verbose: If True, print detailed output
        
        Returns:
            Dictionary with classification, explanation, and severity
        """
        # Get base classification
        if include_severity:
            result = self.classify_with_severity(text, verbose=False)
        else:
            result = self.classify_tweet(text, verbose=False)
        
        # Add explainability
        if self.explainer is not None:
            explanation = self.explainer.explain(
                text=text,
                predicted_class=result['class'],
                methods=methods,
                verbose=False
            )
            result['explanation'] = explanation
        else:
            # Fallback to keyword-only explanation
            keyword_explainer = KeywordExplainer()
            explanation = keyword_explainer.explain(text, result['class'])
            result['explanation'] = {'keywords': explanation}
        
        if verbose:
            self._print_explainable_result(result)
        
        return result
    
    def explain_batch(
        self, 
        texts: List[str],
        methods: List[str] = None,
        include_severity: bool = False
    ) -> List[Dict]:
        """
        Classify and explain multiple texts.
        
        Args:
            texts: List of texts
            methods: Explanation methods to use
            include_severity: Include severity analysis
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for text in texts:
            result = self.classify_with_explanation(
                text=text,
                methods=methods,
                include_severity=include_severity,
                verbose=False
            )
            results.append(result)
        
        return results
    
    def _print_explainable_result(self, result: Dict):
        """Print formatted explainable classification result."""
        print("\n" + "=" * 100)
        print("EXPLAINABLE CLASSIFICATION RESULT")
        print("=" * 100)
        
        # Text
        print(f"\nText: {result['text'][:150]}...")
        
        # Classification
        print(f"\n{'-' * 100}")
        print("CLASSIFICATION")
        print(f"{'-' * 100}")
        print(f"Prediction: {result['prediction']} (Class {result['class']})")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Model: {result['model_info']['name']} ({result['model_info']['type']})")
        
        print(f"\nProbability Breakdown:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name:20s}: {prob:.2%}")
        
        # Severity (if included)
        if 'severity' in result:
            print(f"\n{'-' * 100}")
            print("SEVERITY ANALYSIS")
            print(f"{'-' * 100}")
            severity = result['severity']
            print(f"Severity: {severity['severity_label']} (Level {severity['severity_level']}, Score: {severity['severity_score']}/100)")
            print(f"Explanation: {severity['explanation']}")
        
        # Explanations
        print(f"\n{'-' * 100}")
        print("EXPLAINABILITY ANALYSIS")
        print(f"{'-' * 100}")
        
        if 'explanation' in result:
            explanations = result['explanation'].get('explanations', {})
            
            for method, exp_data in explanations.items():
                print(f"\n>>> {method.upper()} <<<")
                
                if 'error' in exp_data:
                    print(f"Error: {exp_data['error']}")
                    continue
                
                print(f"Explanation: {exp_data['explanation']}")
                
                # Method-specific details
                if method == 'keywords':
                    if exp_data.get('found_keywords'):
                        print("\nDetected Keywords:")
                        for category, keywords in exp_data['found_keywords'].items():
                            print(f"  • {category.replace('_', ' ').title()}: {', '.join(keywords)}")
                
                elif method == 'lime':
                    if exp_data.get('positive_contributions'):
                        print("\nWords Supporting This Classification:")
                        for word, weight in exp_data['positive_contributions'][:5]:
                            bar = '█' * int(abs(weight) * 20)
                            print(f"  {word:15s} {weight:+.3f} {bar}")
                    
                    if exp_data.get('negative_contributions'):
                        print("\nWords Opposing This Classification:")
                        for word, weight in exp_data['negative_contributions'][:3]:
                            bar = '░' * int(abs(weight) * 20)
                            print(f"  {word:15s} {weight:+.3f} {bar}")
                
                elif method == 'feature_importance':
                    if exp_data.get('top_features'):
                        print("\nTop Contributing Features:")
                        for feature, importance in exp_data['top_features'][:5]:
                            bar = '▓' * int(importance * 50)
                            print(f"  {feature:30s} {importance:.4f} {bar}")
        
        # Action (if included)
        if 'action' in result:
            print(f"\n{'-' * 100}")
            print("RECOMMENDED ACTION")
            print(f"{'-' * 100}")
            action = result['action']
            print(f"Urgency: {action['urgency']}")
            print(f"Action: {action['action_string']}")
        
        print("=" * 100)
    
    def compare_explanations(
        self,
        texts: List[str],
        show_agreement: bool = True
    ):
        """
        Compare explanations for multiple texts side-by-side.
        
        Args:
            texts: List of texts to compare
            show_agreement: Show method agreement analysis
        """
        print("\n" + "=" * 100)
        print("COMPARATIVE EXPLANATION ANALYSIS")
        print("=" * 100)
        
        results = self.explain_batch(texts, include_severity=False)
        
        for i, result in enumerate(results, 1):
            print(f"\n{'-' * 100}")
            print(f"Text {i}: {result['text'][:80]}...")
            print(f"{'-' * 100}")
            print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")
            
            if 'explanation' in result:
                explanations = result['explanation'].get('explanations', {})
                
                for method, exp_data in explanations.items():
                    if 'explanation' in exp_data:
                        print(f"  [{method}] {exp_data['explanation'][:120]}...")
        
        print("=" * 100)
    
    def get_explanation_summary(self, text: str) -> str:
        """
        Get a concise one-line explanation summary.
        
        Args:
            text: Input text
        
        Returns:
            One-line explanation string
        """
        try:
            result = self.classify_with_explanation(
                text=text,
                methods=None,
                include_severity=False,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to classify text for summary: {e}")
            return "Error: could not generate explanation summary."

        pred = result.get('prediction') or CLASS_LABELS.get(result.get('class'), str(result.get('class')))
        confidence = result.get('confidence', 0.0)
        conf_str = f"{confidence:.1%}" if isinstance(confidence, (float, int)) else str(confidence)

        # Try to extract best available explanation info
        summary_parts: List[str] = [f"{pred} ({conf_str})"]

        explanation = result.get('explanation', {}) or {}
        explanations = explanation.get('explanations') if isinstance(explanation, dict) else None

        # CASE 1: Comprehensive explainer structure -> explanations dict
        if explanations:
            # Keywords
            kw_data = explanations.get('keywords', {})
            found_keywords = kw_data.get('found_keywords') if isinstance(kw_data, dict) else None
            if found_keywords:
                # flatten categories -> words
                flat = []
                for cat, kws in found_keywords.items():
                    if isinstance(kws, (list, tuple)) and kws:
                        flat.extend([str(w) for w in kws])
                if flat:
                    summary_parts.append(f"keywords: {', '.join(flat[:6])}")

            # Feature importance
            fi = explanations.get('feature_importance', {})
            top_feats = fi.get('top_features') if isinstance(fi, dict) else None
            if top_feats and len(top_feats) > 0:
                feats = [str(f[0]) for f in top_feats[:3]]
                summary_parts.append(f"top features: {', '.join(feats)}")

            # LIME (positive contributions)
            lime = explanations.get('lime', {})
            pos = lime.get('positive_contributions') if isinstance(lime, dict) else None
            if pos and len(pos) > 0:
                words = [str(w[0]) for w in pos[:3]]
                summary_parts.append(f"supporting words: {', '.join(words)}")

        else:
            # CASE 2: Fallback keyword-only explainer (older shape)
            # e.g., result['explanation'] could be {'keywords': ['word1','word2',...]}
            if isinstance(explanation, dict):
                # direct keywords list
                kw_list = explanation.get('keywords')
                if isinstance(kw_list, (list, tuple)) and kw_list:
                    summary_parts.append(f"keywords: {', '.join(str(k) for k in kw_list[:6])}")

            # Another possible fallback: explanation may be a plain string
            if not summary_parts or len(summary_parts) == 1:
                # try to get any short free-text explanation
                free = explanation.get('explanation') if isinstance(explanation, dict) else None
                if isinstance(free, str) and free.strip():
                    # take first 12 words
                    w = " ".join(free.strip().split()[:12])
                    summary_parts.append(f"explain: {w}...")

        # Build final one-line summary, keep it short
        summary = " | ".join(summary_parts)

        # Truncate to a reasonable length (e.g., 300 chars) to keep it one-line
        max_len = 300
        if len(summary) > max_len:
            summary = summary[: max_len - 3].rstrip() + "..."

        return summary


# ---------------------------------------------------------------------
# Module-level convenience wrappers (make functions importable directly)
# ---------------------------------------------------------------------

__all__ = [
    "ExplainableTweetClassifier",
    "classify_with_explanation",
    "get_explanation_summary",
]

def classify_with_explanation(
    text: str,
    methods: List[str] = None,
    include_severity: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Convenience wrapper so callers can do:
        from inference.explainable_classifier import classify_with_explanation

    It constructs an ExplainableTweetClassifier (with defaults) and returns the result.
    """
    clf = ExplainableTweetClassifier()
    return clf.classify_with_explanation(
        text=text,
        methods=methods,
        include_severity=include_severity,
        verbose=verbose
    )


def get_explanation_summary(text: str) -> str:
    """
    Convenience wrapper exposing the instance method as a top-level function.
    """
    clf = ExplainableTweetClassifier()
    return clf.get_explanation_summary(text)
