"""
Ensemble Manager for Hate Speech Detection
===========================================

Two ensemble strategies for improved accuracy:
1. Multi-Model Ensemble: Different architectures (HateBERT, DeBERTa, RoBERTa, BERT)
2. Cross-Validation Ensemble: Same architecture, different data splits

Expected gain: +2-3% over single best model

Ensemble methods:
- Soft voting (average probabilities)
- Weighted voting (learned weights)
- Stacking (meta-classifier)
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
import warnings
import joblib
warnings.filterwarnings('ignore')

try:
    from config import PROJECT_ROOT, MODELS_DIR, NUM_CLASSES
    from utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / 'saved_models' / 'advanced' / 'ensemble'
    NUM_CLASSES = 3

# ==================== ENSEMBLE MANAGER ====================

class EnsembleManager:
    """
    Multi-model ensemble for hate speech detection.
    
    Combines predictions from multiple diverse models:
    - HateBERT (hate speech specialist)
    - DeBERTa-v3-large (state-of-the-art)
    - RoBERTa-large (strong baseline)
    - BERT-large (reference)
    
    Ensemble methods:
    1. Soft voting: Average probability outputs
    2. Weighted voting: Learn optimal weights per model
    3. Stacking: Train meta-classifier on predictions
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        """
        Initialize ensemble manager.
        
        Args:
            num_classes: Number of output classes
        """
        self.num_classes = num_classes
        self.models = {}
        self.model_weights = None
        self.meta_classifier = None
        
        logger.info("Initialized Ensemble Manager")
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name/identifier
            model: Model instance (must have predict_proba method)
            weight: Initial weight for weighted voting
        """
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {name} must have predict_proba method")
        
        self.models[name] = {
            'model': model,
            'weight': weight
        }
        
        logger.info(f"Added {name} to ensemble (weight: {weight})")
    
    def predict_soft_voting(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using soft voting (average probabilities).
        
        Args:
            texts: List of texts
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        logger.info(f"Soft voting with {len(self.models)} models")
        
        # Collect predictions from all models
        all_probas = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            probas = model.predict_proba(texts)
            all_probas.append(probas)
        
        # Average probabilities
        ensemble_probas = np.mean(all_probas, axis=0)
        predictions = np.argmax(ensemble_probas, axis=1)
        
        return predictions, ensemble_probas
    
    def predict_weighted_voting(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using weighted voting.
        
        Args:
            texts: List of texts
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        logger.info(f"Weighted voting with {len(self.models)} models")
        
        # Collect weighted predictions
        weighted_probas = None
        total_weight = 0.0
        
        for name, model_info in self.models.items():
            model = model_info['model']
            weight = model_info['weight']
            
            probas = model.predict_proba(texts)
            
            if weighted_probas is None:
                weighted_probas = probas * weight
            else:
                weighted_probas += probas * weight
            
            total_weight += weight
        
        # Normalize by total weight
        ensemble_probas = weighted_probas / total_weight
        predictions = np.argmax(ensemble_probas, axis=1)
        
        return predictions, ensemble_probas
    
    def learn_optimal_weights(
        self,
        X_val: List[str],
        y_val: np.ndarray
    ):
        """
        Learn optimal weights for weighted voting using validation set.
        
        Args:
            X_val: Validation texts
            y_val: Validation labels
        """
        from scipy.optimize import minimize
        from sklearn.metrics import f1_score
        
        logger.info("Learning optimal ensemble weights...")
        
        # Get predictions from all models
        model_names = list(self.models.keys())
        all_probas = []
        
        for name in model_names:
            model = self.models[name]['model']
            probas = model.predict_proba(X_val)
            all_probas.append(probas)
        
        all_probas = np.array(all_probas)  # Shape: (n_models, n_samples, n_classes)
        
        # Define objective function (negative F1 score)
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Weighted average of probabilities
            weighted_probas = np.tensordot(weights, all_probas, axes=([0], [0]))
            predictions = np.argmax(weighted_probas, axis=1)
            
            # Return negative F1 score (we minimize)
            return -f1_score(y_val, predictions, average='macro', zero_division=0)
        
        # Initial weights (equal)
        initial_weights = np.ones(len(model_names))
        
        # Constraints: weights sum to 1 and are positive
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(len(model_names))]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Update weights
        optimal_weights = result.x / result.x.sum()
        
        for name, weight in zip(model_names, optimal_weights):
            self.models[name]['weight'] = float(weight)
            logger.info(f"  {name}: {weight:.4f}")
        
        # Evaluate with optimized weights
        predictions, _ = self.predict_weighted_voting(X_val)
        f1 = f1_score(y_val, predictions, average='macro', zero_division=0)
        
        logger.info(f"Optimized ensemble F1 (validation): {f1:.4f}")
    
    def train_stacking_classifier(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        meta_classifier_type: str = 'logistic'
    ):
        """
        Train meta-classifier for stacking ensemble.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            meta_classifier_type: Type of meta-classifier ('logistic', 'rf', 'xgboost')
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info(f"Training stacking meta-classifier ({meta_classifier_type})...")
        
        # Get predictions from all models on training set
        logger.info("Generating meta-features from training set...")
        train_meta_features = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            probas = model.predict_proba(X_train)
            train_meta_features.append(probas)
        
        train_meta_features = np.hstack(train_meta_features)
        
        # Get predictions on validation set
        logger.info("Generating meta-features from validation set...")
        val_meta_features = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            probas = model.predict_proba(X_val)
            val_meta_features.append(probas)
        
        val_meta_features = np.hstack(val_meta_features)
        
        # Train meta-classifier
        if meta_classifier_type == 'logistic':
            self.meta_classifier = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        elif meta_classifier_type == 'rf':
            self.meta_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif meta_classifier_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.meta_classifier = XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            except ImportError:
                logger.warning("XGBoost not available, using Logistic Regression")
                self.meta_classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown meta-classifier type: {meta_classifier_type}")
        
        logger.info(f"Training {type(self.meta_classifier).__name__}...")
        self.meta_classifier.fit(train_meta_features, y_train)
        
        # Evaluate
        val_predictions = self.meta_classifier.predict(val_meta_features)
        
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_val, val_predictions)
        f1 = f1_score(y_val, val_predictions, average='macro', zero_division=0)
        
        logger.info(f"Stacking ensemble performance (validation):")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 (macro): {f1:.4f}")
    
    def predict_stacking(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using stacking ensemble.
        
        Args:
            texts: List of texts
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.meta_classifier is None:
            raise ValueError("Meta-classifier not trained. Call train_stacking_classifier() first.")
        
        logger.info(f"Stacking prediction with {len(self.models)} models")
        
        # Get predictions from all models
        meta_features = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            probas = model.predict_proba(texts)
            meta_features.append(probas)
        
        meta_features = np.hstack(meta_features)
        
        # Predict with meta-classifier
        predictions = self.meta_classifier.predict(meta_features)
        
        # Get probabilities if available
        if hasattr(self.meta_classifier, 'predict_proba'):
            probabilities = self.meta_classifier.predict_proba(meta_features)
        else:
            # Create one-hot probabilities
            probabilities = np.zeros((len(predictions), self.num_classes))
            probabilities[np.arange(len(predictions)), predictions] = 1.0
        
        return predictions, probabilities
    
    def predict(
        self,
        texts: List[str],
        method: str = 'soft_voting'
    ) -> np.ndarray:
        """
        Predict using specified ensemble method.
        
        Args:
            texts: List of texts
            method: Ensemble method ('soft_voting', 'weighted_voting', 'stacking')
        
        Returns:
            Predictions
        """
        if method == 'soft_voting':
            predictions, _ = self.predict_soft_voting(texts)
        elif method == 'weighted_voting':
            predictions, _ = self.predict_weighted_voting(texts)
        elif method == 'stacking':
            predictions, _ = self.predict_stacking(texts)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return predictions
    
    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray,
        method: str = 'soft_voting'
    ) -> Dict:
        """
        Evaluate ensemble on test set.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            method: Ensemble method
        
        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score
        )
        
        logger.info(f"Evaluating ensemble ({method}) on test set...")
        
        predictions = self.predict(X_test, method=method)
        
        metrics = {
            'method': method,
            'accuracy': accuracy_score(y_test, predictions),
            'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, predictions, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_test, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, predictions, average='macro', zero_division=0)
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def save(self, output_dir: Path):
        """
        Save ensemble configuration.
        
        Args:
            output_dir: Directory to save ensemble
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save weights and configuration
        config = {
            'num_classes': self.num_classes,
            'models': {
                name: {'weight': info['weight']}
                for name, info in self.models.items()
            }
        }
        
        config_path = output_dir / 'ensemble_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save meta-classifier if exists
        if self.meta_classifier is not None:
            meta_path = output_dir / 'meta_classifier.pkl'
            joblib.dump(self.meta_classifier, meta_path)
        
        logger.info(f"Saved ensemble to {output_dir}")

# ==================== CROSS-VALIDATION ENSEMBLE ====================

class CrossValidationEnsemble:
    """
    Cross-validation ensemble using same architecture with different data splits.
    
    Trains K models (typically 5) with different random seeds or K-fold CV.
    Each model sees slightly different data perspectives.
    
    Benefits:
    - Reduces variance and overfitting
    - More robust predictions
    - Expected gain: +1-2% over single model
    """
    
    def __init__(
        self,
        model_class,
        model_kwargs: Dict = None,
        n_folds: int = 5,
        num_classes: int = NUM_CLASSES
    ):
        """
        Initialize cross-validation ensemble.
        
        Args:
            model_class: Model class to use (e.g., SpecializedTransformerModel)
            model_kwargs: Kwargs to pass to model constructor
            n_folds: Number of folds
            num_classes: Number of classes
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.n_folds = n_folds
        self.num_classes = num_classes
        
        self.models = []
        
        logger.info(f"Initialized Cross-Validation Ensemble ({n_folds} folds)")
    
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        output_dir: Path = None,
        **train_kwargs
    ):
        """
        Train ensemble using K-fold cross-validation.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            output_dir: Directory to save models
            **train_kwargs: Additional training arguments
        """
        from sklearn.model_selection import KFold
        
        logger.info(f"\nTraining {self.n_folds}-fold cross-validation ensemble")
        
        # Setup output directory
        if output_dir is None:
            output_dir = MODELS_DIR / 'cv_ensemble'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # K-fold split
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        X_train_array = np.array(X_train)
        y_train_array = np.array(y_train)
        
        # Train each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_array), 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Training Fold {fold}/{self.n_folds}")
            logger.info(f"{'='*80}")
            
            # Split data
            X_fold_train = X_train_array[train_idx].tolist()
            y_fold_train = y_train_array[train_idx]
            X_fold_val = X_train_array[val_idx].tolist()
            y_fold_val = y_train_array[val_idx]
            
            # Create model
            model = self.model_class(**self.model_kwargs)
            
            # Train
            fold_output_dir = output_dir / f'fold_{fold}'
            model.train(
                X_train=X_fold_train,
                y_train=y_fold_train,
                X_val=X_fold_val,
                y_val=y_fold_val,
                output_dir=fold_output_dir,
                **train_kwargs
            )
            
            self.models.append(model)
            
            logger.info(f"Fold {fold} complete")
        
        logger.info(f"\nAll {self.n_folds} folds trained successfully")
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict using ensemble of all folds (soft voting).
        
        Args:
            texts: List of texts
        
        Returns:
            Predictions
        """
        probas = self.predict_proba(texts)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities using ensemble.
        
        Args:
            texts: List of texts
        
        Returns:
            Average probabilities
        """
        if not self.models:
            raise ValueError("No models trained")
        
        # Collect predictions from all folds
        all_probas = []
        
        for model in self.models:
            probas = model.predict_proba(texts)
            all_probas.append(probas)
        
        # Average probabilities
        ensemble_probas = np.mean(all_probas, axis=0)
        
        return ensemble_probas
    
    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate ensemble on test set.
        
        Args:
            X_test: Test texts
            y_test: Test labels
        
        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score
        )
        
        logger.info(f"Evaluating {self.n_folds}-fold ensemble on test set...")
        
        predictions = self.predict(X_test)
        
        metrics = {
            'n_folds': self.n_folds,
            'accuracy': accuracy_score(y_test, predictions),
            'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, predictions, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_test, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, predictions, average='macro', zero_division=0)
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        
        return metrics

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("ENSEMBLE MANAGER TEST")
    print("=" * 80)
    
    print("\nEnsemble Strategies:")
    print("-" * 80)
    print("1. Multi-Model Ensemble")
    print("   - Combine HateBERT, DeBERTa, RoBERTa, BERT")
    print("   - Expected gain: +2-3%")
    print("   ")
    print("2. Cross-Validation Ensemble")
    print("   - Train 5 models with different data splits")
    print("   - Expected gain: +1-2%")
    print("   ")
    print("Total potential gain: +3-5%")
    
    print("\n" + "=" * 80)
    print("USAGE: Multi-Model Ensemble")
    print("=" * 80)
    print("""
from models.advanced.specialized_transformers import SpecializedTransformerModel
from models.advanced.ensemble_manager import EnsembleManager

# Train diverse models
hatebert = SpecializedTransformerModel('hatebert')
hatebert.train(X_train, y_train, X_val, y_val)

deberta = SpecializedTransformerModel('deberta-v3-large')
deberta.train(X_train, y_train, X_val, y_val)

roberta = SpecializedTransformerModel('roberta-large')
roberta.train(X_train, y_train, X_val, y_val)

# Create ensemble
ensemble = EnsembleManager()
ensemble.add_model('hatebert', hatebert)
ensemble.add_model('deberta', deberta)
ensemble.add_model('roberta', roberta)

# Method 1: Soft voting (simple average)
predictions = ensemble.predict(X_test, method='soft_voting')

# Method 2: Weighted voting (learn optimal weights)
ensemble.learn_optimal_weights(X_val, y_val)
predictions = ensemble.predict(X_test, method='weighted_voting')

# Method 3: Stacking (meta-classifier)
ensemble.train_stacking_classifier(X_train, y_train, X_val, y_val)
predictions = ensemble.predict(X_test, method='stacking')
""")
    
    print("\n" + "=" * 80)
    print("USAGE: Cross-Validation Ensemble")
    print("=" * 80)
    print("""
from models.advanced.specialized_transformers import SpecializedTransformerModel
from models.advanced.ensemble_manager import CrossValidationEnsemble

# Create 5-fold ensemble
cv_ensemble = CrossValidationEnsemble(
    model_class=SpecializedTransformerModel,
    model_kwargs={'model_type': 'hatebert'},
    n_folds=5
)

# Train all folds
cv_ensemble.train(X_train, y_train, X_val, y_val)

# Predict (automatically averages all folds)
predictions = cv_ensemble.predict(X_test)

# Evaluate
metrics = cv_ensemble.evaluate(X_test, y_test)
""")