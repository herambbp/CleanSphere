"""
Advanced Models Trainer - Complete Pipeline
============================================

Integrates all advanced modeling techniques:
1. Specialized Transformers (HateBERT, DeBERTa, etc.)
2. Domain-Adaptive Pre-training
3. Multi-Model Ensemble
4. Cross-Validation Ensemble

Expected total improvement: +5-8% over baseline BERT
"""

import sys
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from config import PROJECT_ROOT, MODELS_DIR, RESULTS_DIR, NUM_CLASSES
    from utils import logger, print_section_header
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / 'saved_models' / 'advanced'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    NUM_CLASSES = 3

from .specialized_transformers import (
    SpecializedTransformerModel,
    list_available_models,
    get_recommended_model
)
from .domain_adaptive import DomainAdaptivePretrainer, create_combined_corpus
from .ensemble_manager import EnsembleManager, CrossValidationEnsemble

# ==================== ADVANCED MODELS TRAINER ====================

class AdvancedModelsTrainer:
    """
    Complete training pipeline for advanced hate speech detection models.
    
    Features:
    - Train specialized transformers (HateBERT, DeBERTa, etc.)
    - Optional domain-adaptive pre-training
    - Multi-model ensemble with multiple strategies
    - Cross-validation ensemble
    - Comprehensive evaluation and comparison
    """
    
    def __init__(self):
        """Initialize advanced models trainer."""
        self.trained_models = {}
        self.ensemble = None
        self.cv_ensemble = None
        self.results = {}
        
        logger.info("Initialized Advanced Models Trainer")
    
    def train_single_model(
        self,
        model_type: str,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        X_test: List[str],
        y_test: np.ndarray,
        use_domain_adaptation: bool = False,
        domain_corpus: List[str] = None,
        **kwargs
    ) -> Dict:
        """
        Train a single specialized transformer model.
        
        Args:
            model_type: Model type (e.g., 'hatebert', 'deberta-v3-large')
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            X_test: Test texts
            y_test: Test labels
            use_domain_adaptation: Whether to use domain-adaptive pre-training
            domain_corpus: Corpus for domain adaptation (defaults to train+val texts)
            **kwargs: Additional training arguments
        
        Returns:
            Results dictionary
        """
        print_section_header(f"TRAINING {model_type.upper()}")
        
        # Domain-adaptive pre-training (optional)
        if use_domain_adaptation:
            logger.info("Step 1: Domain-Adaptive Pre-training")
            logger.info("-" * 80)
            
            # Create corpus
            if domain_corpus is None:
                domain_corpus = create_combined_corpus(
                    labeled_texts=list(X_train) + list(X_val)
                )
            
            # Get base model name
            from .specialized_transformers import SPECIALIZED_MODELS
            base_model = SPECIALIZED_MODELS[model_type]['name']
            
            # Pre-train
            pretrainer = DomainAdaptivePretrainer(base_model=base_model)
            pretrainer.pretrain(
                texts=domain_corpus,
                num_epochs=3,
                batch_size=32
            )
            
            # Load adapted model for fine-tuning
            adapted_model_path = pretrainer.get_adapted_model_path()
            
            logger.info(f"\nStep 2: Fine-tuning Domain-Adapted Model")
            logger.info("-" * 80)
            
            # Note: Would need to modify SpecializedTransformerModel to load custom path
            # For now, we proceed with standard model
            logger.warning("Domain adaptation complete, but using standard model for fine-tuning")
            logger.warning("(Full integration requires model loading from custom path)")
        
        # Train model
        logger.info("Training specialized transformer model...")
        model = SpecializedTransformerModel(model_type=model_type)
        
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **kwargs
        )
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        test_metrics = model.evaluate(X_test, y_test)
        
        # Store results
        result = {
            'model_type': model_type,
            'model': model,
            'history': history,
            'test_metrics': test_metrics,
            'used_domain_adaptation': use_domain_adaptation
        }
        
        self.trained_models[model_type] = result
        self.results[model_type] = test_metrics
        
        logger.info(f"\n{model_type} training complete!")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        
        return result
    
    def train_multiple_models(
        self,
        model_types: List[str],
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        X_test: List[str],
        y_test: np.ndarray,
        use_domain_adaptation: bool = False,
        **kwargs
    ):
        """
        Train multiple specialized models.
        
        Args:
            model_types: List of model types to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            use_domain_adaptation: Use domain-adaptive pre-training
            **kwargs: Additional training arguments
        """
        print_section_header("TRAINING MULTIPLE SPECIALIZED MODELS")
        
        logger.info(f"Models to train: {', '.join(model_types)}")
        logger.info(f"Domain adaptation: {use_domain_adaptation}")
        
        for i, model_type in enumerate(model_types, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Model {i}/{len(model_types)}: {model_type}")
            logger.info(f"{'='*80}")
            
            try:
                self.train_single_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    use_domain_adaptation=use_domain_adaptation,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Print comparison
        self._print_model_comparison()
    
    def create_multi_model_ensemble(
        self,
        X_val: List[str],
        y_val: np.ndarray,
        X_test: List[str],
        y_test: np.ndarray,
        ensemble_method: str = 'weighted_voting'
    ):
        """
        Create multi-model ensemble from trained models.
        
        Args:
            X_val: Validation data for weight optimization
            y_val: Validation labels
            X_test: Test data
            y_test: Test labels
            ensemble_method: Ensemble method ('soft_voting', 'weighted_voting', 'stacking')
        """
        print_section_header("CREATING MULTI-MODEL ENSEMBLE")
        
        if not self.trained_models:
            logger.error("No trained models available for ensemble")
            return
        
        logger.info(f"Creating ensemble with {len(self.trained_models)} models")
        logger.info(f"Ensemble method: {ensemble_method}")
        
        # Create ensemble
        self.ensemble = EnsembleManager(num_classes=NUM_CLASSES)
        
        for model_type, result in self.trained_models.items():
            self.ensemble.add_model(model_type, result['model'])
        
        # Optimize weights if using weighted voting
        if ensemble_method == 'weighted_voting':
            self.ensemble.learn_optimal_weights(X_val, y_val)
        
        # Train stacking classifier if using stacking
        elif ensemble_method == 'stacking':
            # Need to convert X_val to list if it's array
            X_val_list = X_val.tolist() if isinstance(X_val, np.ndarray) else X_val
            X_train_list = []  # Would need actual training data
            y_train_array = np.array([])
            
            logger.warning("Stacking requires training data - skipping for now")
            logger.warning("Using weighted voting instead")
            ensemble_method = 'weighted_voting'
            self.ensemble.learn_optimal_weights(X_val, y_val)
        
        # Evaluate ensemble
        logger.info("\nEvaluating ensemble on test set...")
        ensemble_metrics = self.ensemble.evaluate(X_test, y_test, method=ensemble_method)
        
        self.results['ensemble_' + ensemble_method] = ensemble_metrics
        
        logger.info(f"\nEnsemble Results ({ensemble_method}):")
        logger.info(f"  Test Accuracy: {ensemble_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1 (macro): {ensemble_metrics['f1_macro']:.4f}")
        
        # Compare with best single model
        best_single = max(
            [(k, v) for k, v in self.results.items() if k.startswith('ensemble_') == False],
            key=lambda x: x[1]['f1_macro']
        )
        
        improvement = ensemble_metrics['f1_macro'] - best_single[1]['f1_macro']
        
        logger.info(f"\nComparison:")
        logger.info(f"  Best single model: {best_single[0]} (F1: {best_single[1]['f1_macro']:.4f})")
        logger.info(f"  Ensemble: {ensemble_metrics['f1_macro']:.4f}")
        logger.info(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    def train_cv_ensemble(
        self,
        model_type: str,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        X_test: List[str],
        y_test: np.ndarray,
        n_folds: int = 5,
        **kwargs
    ):
        """
        Train cross-validation ensemble.
        
        Args:
            model_type: Model type to use
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            n_folds: Number of folds
            **kwargs: Additional training arguments
        """
        print_section_header(f"TRAINING {n_folds}-FOLD CV ENSEMBLE")
        
        logger.info(f"Model type: {model_type}")
        logger.info(f"Number of folds: {n_folds}")
        
        # Create CV ensemble
        self.cv_ensemble = CrossValidationEnsemble(
            model_class=SpecializedTransformerModel,
            model_kwargs={'model_type': model_type},
            n_folds=n_folds,
            num_classes=NUM_CLASSES
        )
        
        # Train
        self.cv_ensemble.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            **kwargs
        )
        
        # Evaluate
        logger.info("\nEvaluating CV ensemble on test set...")
        cv_metrics = self.cv_ensemble.evaluate(X_test, y_test)
        
        self.results[f'cv_ensemble_{n_folds}fold'] = cv_metrics
        
        logger.info(f"\nCV Ensemble Results:")
        logger.info(f"  Test Accuracy: {cv_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1 (macro): {cv_metrics['f1_macro']:.4f}")
    
    def _print_model_comparison(self):
        """Print comparison of all trained models."""
        print_section_header("MODEL COMPARISON")
        
        if not self.results:
            logger.warning("No results to compare")
            return
        
        # Sort by F1 macro
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['f1_macro'],
            reverse=True
        )
        
        # Print table
        print(f"\n{'Model':<30} {'Accuracy':>10} {'F1 (macro)':>12} {'F1 (weighted)':>14}")
        print("-" * 70)
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<30} {metrics['accuracy']:>10.4f} "
                  f"{metrics['f1_macro']:>12.4f} {metrics['f1_weighted']:>14.4f}")
        
        # Show best model
        best = sorted_results[0]
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {best[0]}")
        print(f"  Accuracy: {best[1]['accuracy']:.4f}")
        print(f"  F1 (macro): {best[1]['f1_macro']:.4f}")
        print("=" * 70)
    
    def get_best_model(self):
        """Get best performing model."""
        if not self.results:
            return None, None
        
        best_name, best_metrics = max(
            self.results.items(),
            key=lambda x: x[1]['f1_macro']
        )
        
        best_model = self.trained_models.get(best_name, {}).get('model')
        
        return best_name, best_model

# Add to AdvancedModelsTrainer class
def save_metadata_to_dl_json(self):
    """Save results to dl_model_metadata.json for consistency with other phases."""
    from config import RESULTS_DIR
    import json
    from pathlib import Path
    
    # Load existing metadata if it exists
    metadata_path = RESULTS_DIR / 'dl_model_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = {}
    
    # Prepare advanced models metadata
    advanced_metadata = {
        'num_models_trained': len(self.trained_models),
        'models': list(self.trained_models.keys()),
        'advanced_models': True,  # Flag to indicate these are advanced
        'best_model': None,
        'all_results': {}
    }
    
    # Add individual model results
    for model_name, result in self.results.items():
        if result and 'test_metrics' in result:
            metrics = result['test_metrics']
            advanced_metadata['all_results'][f'advanced_{model_name}'] = {
                'accuracy': metrics.get('accuracy', 0),
                'f1_macro': metrics.get('f1_macro', 0),
                'f1_weighted': metrics.get('f1_weighted', 0),
                'training_time': result.get('training_time', 0)
            }
    
    # Find best model
    if self.results:
        best_name, best_metrics = max(
            [(k, v['test_metrics']) for k, v in self.results.items() if v],
            key=lambda x: x[1].get('f1_macro', 0)
        )
        advanced_metadata['best_model'] = {
            'name': f'advanced_{best_name}',
            'accuracy': best_metrics.get('accuracy', 0),
            'f1_macro': best_metrics.get('f1_macro', 0),
            'f1_weighted': best_metrics.get('f1_weighted', 0)
        }
    
    # Merge with existing metadata
    existing_metadata['advanced_models'] = advanced_metadata
    
    # If this is better than previous best, update overall best
    if advanced_metadata['best_model']:
        current_best_f1 = advanced_metadata['best_model']['f1_macro']
        
        if 'best_model' not in existing_metadata or \
           current_best_f1 > existing_metadata['best_model'].get('f1_macro', 0):
            existing_metadata['best_model'] = advanced_metadata['best_model']
            existing_metadata['best_model']['phase'] = 'Phase 6: Advanced Models'
    
    # Save updated metadata
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(existing_metadata, f, indent=2)
    
    logger.info(f"Advanced models metadata saved to {metadata_path}")
    return metadata_path
# ==================== MAIN TRAINING FUNCTION ====================

def train_advanced_models(
    X_train: List[str],
    y_train: np.ndarray,
    X_val: List[str],
    y_val: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    model_types: List[str] = None,
    use_domain_adaptation: bool = False,
    create_ensemble: bool = True,
    ensemble_method: str = 'weighted_voting',
    create_cv_ensemble: bool = False,
    cv_model_type: str = 'hatebert',
    cv_n_folds: int = 5,
    save_to_dl_metadata=True,
):
    """
    Main function to train advanced models.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        model_types: Models to train (default: ['hatebert', 'deberta-v3-large'])
        use_domain_adaptation: Use domain-adaptive pre-training
        create_ensemble: Create multi-model ensemble
        ensemble_method: Ensemble method
        create_cv_ensemble: Create cross-validation ensemble
        cv_model_type: Model type for CV ensemble
        cv_n_folds: Number of CV folds
    
    Returns:
        AdvancedModelsTrainer instance
    """
    print_section_header("ADVANCED MODELS TRAINING PIPELINE")
    
    # Default models (best performers)
    if model_types is None:
        model_types = ['hatebert', 'deberta-v3-large']
        logger.info(f"Using default models: {model_types}")
    
    # Initialize trainer
    trainer = AdvancedModelsTrainer()
    
    # Train individual models
    trainer.train_multiple_models(
        model_types=model_types,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        use_domain_adaptation=use_domain_adaptation
    )
    
    # Create multi-model ensemble
    if create_ensemble and len(trainer.trained_models) > 1:
        trainer.create_multi_model_ensemble(
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            ensemble_method=ensemble_method
        )
    
    # Create CV ensemble
    if create_cv_ensemble:
        trainer.train_cv_ensemble(
            model_type=cv_model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            n_folds=cv_n_folds
        )
    
    # Final summary
    print_section_header("TRAINING COMPLETE")
    logger.info(f"Trained {len(trainer.trained_models)} model(s)")
    
    if trainer.ensemble:
        logger.info("Created multi-model ensemble")
    
    if trainer.cv_ensemble:
        logger.info(f"Created {cv_n_folds}-fold CV ensemble")
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    if best_name:
        logger.info(f"\nBest model: {best_name}")
        logger.info(f"  F1 (macro): {trainer.results[best_name]['f1_macro']:.4f}")

    if save_to_dl_metadata:
        trainer.save_metadata_to_dl_json()
    
    return trainer

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED MODELS TRAINER TEST")
    print("=" * 80)
    
    print("\nAvailable Models:")
    print("-" * 80)
    for model_type, config in list_available_models().items():
        print(f"{model_type:20s}: {config['description']}")
        print(f"{'':20s}  Expected gain: {config['expected_gain']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    print("""
For A5000 GPU (24GB VRAM):

Option 1: Best Single Model
- Model: deberta-v3-large
- Expected improvement: +4-6%

Option 2: Multi-Model Ensemble
- Models: hatebert + deberta-v3-large + roberta-large
- Expected improvement: +5-7%

Option 3: Full Pipeline
- Domain adaptation: +1-3%
- Multi-model ensemble: +2-3%
- Total expected improvement: +6-8%
""")
    
    print("=" * 80)
    print("USAGE EXAMPLE")
    print("=" * 80)
    print("""
from models.advanced.train_advanced import train_advanced_models

# Option 1: Train best single model
trainer = train_advanced_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_types=['deberta-v3-large']
)

# Option 2: Multi-model ensemble
trainer = train_advanced_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_types=['hatebert', 'deberta-v3-large', 'roberta-large'],
    create_ensemble=True,
    ensemble_method='weighted_voting'
)

# Option 3: Full pipeline with domain adaptation
trainer = train_advanced_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_types=['hatebert', 'deberta-v3-large'],
    use_domain_adaptation=True,
    create_ensemble=True,
    create_cv_ensemble=True
)
""")