"""
Integration Module - Heavy GPU BERT with Main Training Pipeline
=================================================================

This module integrates the enhanced Heavy GPU BERT model with the
main_train_enhanced.py training pipeline.

Features:
- Train multiple BERT variants (BERT-Large, RoBERTa-Large, etc.)
- Model comparison and ensembling
- Automatic best model selection
- Integration with existing evaluation framework
- Seamless integration with Phase 5 training
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    from utils import logger
    from config import (
        PROJECT_ROOT, MODELS_DIR, RESULTS_DIR,
        NUM_CLASSES, CLASS_LABELS, MODEL_FILES
    )
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    PROJECT_ROOT = Path(__file__).parent
    MODELS_DIR = PROJECT_ROOT / 'saved_models'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    NUM_CLASSES = 3
    CLASS_LABELS = {0: 'Hate', 1: 'Offensive', 2: 'Neither'}

try:
    from bert_model_heavy_gpu import (
        HeavyGPUBERTModel,
        HeavyGPUBERTConfig,
        create_bert_model,
        HAS_TORCH,
        HAS_TRANSFORMERS
    )
    HAS_HEAVY_BERT = True
except ImportError:
    HAS_HEAVY_BERT = False
    logger.warning("Heavy GPU BERT module not found")


# ==================== BERT TRAINER WITH MULTIPLE MODELS ====================

class HeavyGPUBERTTrainer:
    """
    Trainer class for Heavy GPU BERT models.
    Supports training multiple model variants and comparison.
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize BERT trainer.
        
        Args:
            models_dir: Directory to save models
        """
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.trained_models = {}
        self.evaluation_results = {}
        self.best_model_name = None
        self.best_model = None
        self.best_accuracy = 0.0
    
    def train_single_model(
        self,
        model_name: str,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        X_test: List[str] = None,
        y_test: np.ndarray = None,
        config: Dict = None,
        save_model: bool = True
    ) -> Dict:
        """
        Train a single BERT model.
        
        Args:
            model_name: Model identifier (bert-base, bert-large, roberta-large, etc.)
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            X_test: Test texts (optional)
            y_test: Test labels (optional)
            config: Model configuration overrides
            save_model: Whether to save the trained model
        
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING {model_name.upper()}")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Create model configuration
        if config is None:
            config = {}
        
        config['model_name'] = model_name
        
        # Create model
        logger.info(f"Creating {model_name} model...")
        model = HeavyGPUBERTModel(config=config, num_classes=NUM_CLASSES)
        model.build_model()
        
        # Train model
        logger.info(f"Training {model_name}...")
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set if provided
        test_metrics = None
        if X_test is not None and y_test is not None:
            logger.info(f"\nEvaluating {model_name} on test set...")
            test_metrics = model.evaluate(X_test, y_test, verbose=1)
        
        # Save model
        if save_model:
            model_save_path = self.models_dir / f"bert_{model_name.replace('-', '_')}"
            logger.info(f"\nSaving {model_name} model to {model_save_path}")
            model.save(model_save_path)
        
        # Store results
        model_info = model.get_model_info()
        
        results = {
            'model_name': model_name,
            'model': model,
            'history': history,
            'training_time': training_time,
            'model_info': model_info,
            'test_metrics': test_metrics,
            'val_accuracy': model.best_val_acc,
            'test_accuracy': test_metrics['accuracy'] if test_metrics else None
        }
        
        self.trained_models[model_name] = results
        
        # Track best model
        if test_metrics and test_metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = test_metrics['accuracy']
            self.best_model_name = model_name
            self.best_model = model
            logger.info(f"\n NEW BEST MODEL: {model_name} (Test Acc: {self.best_accuracy:.4f})")
        
        logger.info("\n" + "="*80)
        logger.info(f"{model_name.upper()} TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Training time: {training_time/60:.2f} minutes")
        if test_metrics:
            logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info("="*80 + "\n")
        
        return results
    
    def train_multiple_models(
        self,
        model_names: List[str],
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        X_test: List[str] = None,
        y_test: np.ndarray = None,
        configs: Dict[str, Dict] = None
    ) -> Dict:
        """
        Train multiple BERT models and compare.
        
        Args:
            model_names: List of model identifiers
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            X_test: Test texts
            y_test: Test labels
            configs: Dictionary of model-specific configurations
        
        Returns:
            Dictionary with all results
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING MULTIPLE BERT MODELS")
        logger.info("="*80)
        logger.info(f"Models to train: {', '.join(model_names)}")
        logger.info("="*80 + "\n")
        
        all_results = {}
        
        for i, model_name in enumerate(model_names, 1):
            logger.info(f"\n[{i}/{len(model_names)}] Training {model_name}...")
            
            config = configs.get(model_name, {}) if configs else {}
            
            try:
                results = self.train_single_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    config=config,
                    save_model=True
                )
                all_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def generate_comparison_report(self, results: Dict = None):
        """Generate comparison report for all trained models."""
        if results is None:
            results = self.trained_models
        
        if not results:
            logger.warning("No models to compare")
            return
        
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON REPORT")
        logger.info("="*80)
        
        # Prepare comparison data
        comparison_data = []
        for model_name, result in results.items():
            test_metrics = result.get('test_metrics')
            if test_metrics:
                comparison_data.append({
                    'Model': model_name,
                    'Params': result['model_info']['total_params'],
                    'Train Time (min)': result['training_time'] / 60,
                    'Val Acc': result['val_accuracy'],
                    'Test Acc': test_metrics['accuracy'],
                    'Test F1': test_metrics['f1_macro'],
                    'Test Precision': test_metrics['precision_macro'],
                    'Test Recall': test_metrics['recall_macro']
                })
        
        # Sort by test accuracy
        comparison_data.sort(key=lambda x: x['Test Acc'], reverse=True)
        
        # Print table
        logger.info("\n{:<20} {:<12} {:<15} {:<10} {:<10} {:<10} {:<12} {:<10}".format(
            "Model", "Params (M)", "Time (min)", "Val Acc", "Test Acc", "Test F1", "Precision", "Recall"
        ))
        logger.info("-" * 115)
        
        for data in comparison_data:
            logger.info("{:<20} {:<12.1f} {:<15.2f} {:<10.4f} {:<10.4f} {:<10.4f} {:<12.4f} {:<10.4f}".format(
                data['Model'],
                data['Params'] / 1e6,
                data['Train Time (min)'],
                data['Val Acc'],
                data['Test Acc'],
                data['Test F1'],
                data['Test Precision'],
                data['Test Recall']
            ))
        
        logger.info("\n" + "="*80)
        logger.info(f"BEST MODEL: {comparison_data[0]['Model']}")
        logger.info(f"  Test Accuracy: {comparison_data[0]['Test Acc']:.4f}")
        logger.info(f"  Test F1 Score: {comparison_data[0]['Test F1']:.4f}")
        logger.info("="*80 + "\n")
    
    def get_best_model(self) -> Tuple[str, HeavyGPUBERTModel]:
        """Get the best performing model."""
        if self.best_model is None:
            raise ValueError("No models have been trained yet")
        
        return self.best_model_name, self.best_model
    
    def ensemble_predict(
        self,
        texts: List[str],
        models: List[HeavyGPUBERTModel] = None,
        method: str = 'average'
    ) -> np.ndarray:
        """
        Make predictions using ensemble of models.
        
        Args:
            texts: Texts to predict
            models: List of models (uses all trained if None)
            method: Ensemble method ('average', 'voting')
        
        Returns:
            Ensemble predictions
        """
        if models is None:
            models = [result['model'] for result in self.trained_models.values()]
        
        if not models:
            raise ValueError("No models available for ensemble")
        
        logger.info(f"Ensemble prediction using {len(models)} models (method: {method})")
        
        # Get predictions from all models
        all_probas = []
        for model in models:
            probas = model.predict_proba(texts)
            all_probas.append(probas)
        
        all_probas = np.array(all_probas)
        
        if method == 'average':
            # Average probabilities
            ensemble_probas = np.mean(all_probas, axis=0)
            predictions = np.argmax(ensemble_probas, axis=1)
        elif method == 'voting':
            # Majority voting
            all_predictions = [model.predict(texts) for model in models]
            all_predictions = np.array(all_predictions)
            predictions = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(),
                axis=0,
                arr=all_predictions
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return predictions
    
    def save_metadata(self):
    """Save Heavy GPU BERT training metadata to dl_model_metadata.json."""
    import json
    from config import RESULTS_DIR
    
    logger.info("Saving Heavy GPU BERT metadata to dl_model_metadata.json...")
    
    if not self.trained_models:
        logger.warning("No trained models to save metadata for")
        return
    
    metadata_path = RESULTS_DIR / 'dl_model_metadata.json'
    
    # Load existing metadata if it exists
    existing_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            logger.info("Loaded existing dl_model_metadata.json")
        except Exception as e:
            logger.warning(f"Could not load existing metadata: {e}")
            existing_metadata = {}
    
    # Prepare BERT results
    bert_results = {}
    for model_name, result in self.trained_models.items():
        test_metrics = result.get('test_metrics')
        if test_metrics:
            bert_results[model_name] = {
                'accuracy': float(test_metrics['accuracy']),
                'f1_macro': float(test_metrics['f1_macro']),
                'f1_weighted': float(test_metrics['f1_weighted']),
                'precision_macro': float(test_metrics['precision_macro']),
                'recall_macro': float(test_metrics['recall_macro']),
                'training_time': float(result['training_time']),
                'training_time_minutes': float(result['training_time'] / 60),
                'val_accuracy': float(result['val_accuracy']),
                'total_params': int(result['model_info']['total_params']),
                'model_type': 'Heavy GPU BERT',
                'bert_variant': result['model_name']
            }
    
    # Update existing metadata with BERT results
    if 'all_results' not in existing_metadata:
        existing_metadata['all_results'] = {}
    
    # Add BERT models to all_results
    existing_metadata['all_results'].update(bert_results)
    
    # Update models list
    if 'models' not in existing_metadata:
        existing_metadata['models'] = []
    
    for model_name in bert_results.keys():
        if model_name not in existing_metadata['models']:
            existing_metadata['models'].append(model_name)
    
    # Update num_models_trained
    existing_metadata['num_models_trained'] = len(existing_metadata.get('models', []))
    
    # Find overall best model (comparing all DL models including BERT)
    all_results = existing_metadata.get('all_results', {})
    if all_results:
        best_model_name = max(
            all_results.items(),
            key=lambda x: x[1].get('f1_macro', 0)
        )[0]
        
        best_model_metrics = all_results[best_model_name]
        
        existing_metadata['best_model'] = {
            'name': best_model_name,
            'accuracy': float(best_model_metrics['accuracy']),
            'f1_macro': float(best_model_metrics['f1_macro']),
            'f1_weighted': float(best_model_metrics['f1_weighted'])
        }
        
        logger.info(f"Overall best model: {best_model_name} (F1 Macro: {best_model_metrics['f1_macro']:.4f})")
    
    # Add framework info
    if 'frameworks' not in existing_metadata:
        existing_metadata['frameworks'] = []
    
    if 'PyTorch (Heavy GPU BERT)' not in existing_metadata['frameworks']:
        existing_metadata['frameworks'].append('PyTorch (Heavy GPU BERT)')
    
    # Save updated metadata
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)
        
        logger.info(f"[SUCCESS] Metadata saved to {metadata_path}")
        logger.info(f"  Total models: {existing_metadata['num_models_trained']}")
        logger.info(f"  BERT models added: {len(bert_results)}")
        
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        import traceback
        traceback.print_exc()


# ==================== INTEGRATION WITH MAIN TRAINING PIPELINE ====================

def train_heavy_gpu_bert(
    X_train: List[str],
    y_train: np.ndarray,
    X_val: List[str],
    y_val: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    model_names: List[str] = None,
    use_ensemble: bool = False
) -> HeavyGPUBERTTrainer:
    """
    Train Heavy GPU BERT models - for Phase 5 integration.
    
    Args:
        X_train: Training texts
        y_train: Training labels
        X_val: Validation texts
        y_val: Validation labels
        X_test: Test texts
        y_test: Test labels
        model_names: List of models to train (default: bert-large, roberta-base)
        use_ensemble: Whether to use ensemble prediction
    
    Returns:
        HeavyGPUBERTTrainer instance with results
    """
    if not HAS_HEAVY_BERT:
        logger.error("Heavy GPU BERT module not available!")
        return None
    
    if not HAS_TORCH:
        logger.error("PyTorch not installed!")
        return None
    
    if not HAS_TRANSFORMERS:
        logger.error("Transformers not installed!")
        return None
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: HEAVY GPU BERT TRAINING")
    logger.info("="*80)
    
    # Default models - using larger models since we have heavy GPU
    if model_names is None:
        model_names = ['bert-base']  # Changed default to bert-base (best performing)
        logger.info("Using default model: bert-base (best performing)")
    
    # Convert numpy arrays to lists if needed
    if isinstance(X_train, np.ndarray):
        X_train = X_train.tolist()
    if isinstance(X_val, np.ndarray):
        X_val = X_val.tolist()
    if isinstance(X_test, np.ndarray):
        X_test = X_test.tolist()
    
    # Create trainer
    trainer = HeavyGPUBERTTrainer()
    
    # Train models
    results = trainer.train_multiple_models(
        model_names=model_names,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )
    
    # Save metadata to dl_model_metadata.json
    trainer.save_metadata()
    
    # Test ensemble if requested
    if use_ensemble and len(results) > 1:
        logger.info("\n" + "="*80)
        logger.info("TESTING ENSEMBLE PREDICTION")
        logger.info("="*80)
        
        ensemble_pred = trainer.ensemble_predict(X_test, method='average')
        
        from sklearn.metrics import accuracy_score, f1_score
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average='macro')
        
        logger.info(f"\nEnsemble Results:")
        logger.info(f"  Accuracy: {ensemble_acc:.4f}")
        logger.info(f"  F1 Score: {ensemble_f1:.4f}")
        
        # Compare with best single model
        best_name, best_model = trainer.get_best_model()
        logger.info(f"\nBest Single Model ({best_name}):")
        logger.info(f"  Accuracy: {trainer.best_accuracy:.4f}")
        
        if ensemble_acc > trainer.best_accuracy:
            logger.info(f"\n✓ Ensemble is better by {ensemble_acc - trainer.best_accuracy:.4f}!")
        
        logger.info("="*80 + "\n")
    
    return trainer

# ==================== QUICK TRAINING PRESETS ====================

def quick_train_bert_large(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int = 64,
    epochs: int = 10
):
    """Quick training with BERT-Large only."""
    config = HeavyGPUBERTConfig(
        model_name='bert-large',
        batch_size=batch_size,
        epochs=epochs
    )
    
    trainer = HeavyGPUBERTTrainer()
    results = trainer.train_single_model(
        model_name='bert-large',
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config=config.to_dict()
    )
    
    return trainer, results


def quick_train_roberta_large(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int = 64,
    epochs: int = 10
):
    """Quick training with RoBERTa-Large only."""
    config = HeavyGPUBERTConfig(
        model_name='roberta-large',
        batch_size=batch_size,
        epochs=epochs
    )
    
    trainer = HeavyGPUBERTTrainer()
    results = trainer.train_single_model(
        model_name='roberta-large',
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config=config.to_dict()
    )
    
    return trainer, results


def train_all_large_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int = 48,
    epochs: int = 8
):
    """Train all large models and compare."""
    model_names = ['bert-large', 'roberta-large']
    
    return train_heavy_gpu_bert(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        model_names=model_names,
        use_ensemble=True
    )


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*80)
    print("HEAVY GPU BERT INTEGRATION TEST")
    print("="*80)
    
    if not HAS_HEAVY_BERT:
        print("\n Heavy GPU BERT module not available")
        exit(1)
    
    if not HAS_TORCH:
        print("\n PyTorch not installed")
        exit(1)
    
    if not HAS_TRANSFORMERS:
        print("\n Transformers not installed")
        exit(1)
    
    print("\n All dependencies available")
    print(" Ready for heavy GPU BERT training")
    
    # Create dummy data for testing
    print("\n" + "="*80)
    print("TESTING WITH DUMMY DATA")
    print("="*80)
    
    X_train = ["I hate you"] * 100 + ["Good morning"] * 100 + ["You idiot"] * 100
    y_train = np.array([0] * 100 + [2] * 100 + [1] * 100)
    
    X_val = ["I hate you"] * 20 + ["Good morning"] * 20 + ["You idiot"] * 20
    y_val = np.array([0] * 20 + [2] * 20 + [1] * 20)
    
    X_test = ["I hate you"] * 10 + ["Good morning"] * 10 + ["You idiot"] * 10
    y_test = np.array([0] * 10 + [2] * 10 + [1] * 10)
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    print("\n Integration module test complete")
    print("="*80)


