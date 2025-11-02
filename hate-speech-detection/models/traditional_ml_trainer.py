"""
Training script for traditional ML models
Trains Random Forest, XGBoost, SVM, Gradient Boosting, and MLP
"""

import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

from config import (
    RANDOM_FOREST_CONFIG, XGBOOST_CONFIG, SVM_CONFIG,
    HIST_GRADIENT_BOOST_CONFIG, MLP_CONFIG, SVM_MAX_SAMPLES,
    MODEL_FILES, METADATA_FILE, COMPARISON_FILE, CLASS_WEIGHTS_LIST, CLASS_WEIGHTS
)
from utils import (
    logger, print_section_header, ModelEvaluator,
    get_classification_report, save_results
)

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Skipping XGBoost model.")

# ==================== MODEL TRAINER ====================

class TraditionalMLTrainer:
    """
    Train and evaluate traditional machine learning models.
    """
    
    def __init__(self):
        """Initialize trainer with all models."""
        logger.info("Initializing Traditional ML Trainer...")
        
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.evaluator = ModelEvaluator()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all ML models with configurations."""
        # Random Forest
        self.models['RandomForest'] = RandomForestClassifier(**RANDOM_FOREST_CONFIG)
        
        # Gradient Boosting
        self.models['GradientBoosting'] = HistGradientBoostingClassifier(**HIST_GRADIENT_BOOST_CONFIG)
        
        # SVM
        self.models['SVM'] = SVC(**SVM_CONFIG)
        
        # Neural Network (MLP)
        self.models['MLP'] = MLPClassifier(**MLP_CONFIG)
        
        # XGBoost (if available)
        if HAS_XGBOOST:
            self.models['XGBoost'] = XGBClassifier(**XGBOOST_CONFIG)
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights based on class weights.
        
        Args:
            y: Labels
        
        Returns:
            Sample weights array
        """
        from config import CLASS_WEIGHTS
        
        # Map each sample to its class weight
        sample_weights = np.array([CLASS_WEIGHTS[int(label)] for label in y])
        
        return sample_weights
    
    def train_single_model(
    self, 
    model_name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> dict:
        """
        Train a single model and evaluate (WITH CLASS WEIGHTS).
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Dictionary with results
        """
        print_section_header(f"TRAINING {model_name}")
        
        # Handle SVM data size limitation
        if 'SVM' in model_name and len(X_train) > SVM_MAX_SAMPLES:
            logger.info(f"Limiting SVM training to {SVM_MAX_SAMPLES} samples for efficiency")
            indices = np.random.choice(len(X_train), SVM_MAX_SAMPLES, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Train
        start_time = time.time()
        logger.info(f"Training {model_name} on {len(X_train_subset)} samples...")
        logger.info(f"Using class weights to handle imbalance (Hate: 13.4x, Offensive: 1.0x, Neither: 4.6x)")
        
        try:
            # Fit model WITH CLASS WEIGHTS
            if hasattr(model, 'fit'):
                # Most sklearn models support class_weight parameter
                try:
                    if 'SVM' in model_name or 'RandomForest' in model_name or 'MLP' in model_name:
                        # These models accept dict format
                        model.fit(X_train_subset, y_train_subset, sample_weight=self._compute_sample_weights(y_train_subset))
                    else:
                        # XGBoost and GradientBoosting might handle it differently
                        model.fit(X_train_subset, y_train_subset, sample_weight=self._compute_sample_weights(y_train_subset))
                except TypeError:
                    # If class_weight not supported, use sample_weight
                    logger.warning(f"{model_name} doesn't support class_weight directly, using sample_weight")
                    model.fit(X_train_subset, y_train_subset, sample_weight=self._compute_sample_weights(y_train_subset))
            
            training_time = time.time() - start_time
            
            logger.info(f"{model_name} training completed in {training_time:.2f}s")
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)
            else:
                y_pred_proba = None
            
            # Evaluate
            result = self.evaluator.evaluate_model(
                model_name=model_name,
                y_true=y_val,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                training_time=training_time
            )
            
            # Print results
            self.evaluator.print_evaluation(result)
            
            # Store trained model
            self.trained_models[model_name] = model
            
            return result
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print_section_header("TRAINING ALL MODELS")
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Validation set: {X_val.shape}")
        
        # Train each model
        for model_name, model in self.models.items():
            result = self.train_single_model(
                model_name, model, X_train, y_train, X_val, y_val
            )
            
            if result:
                self.results[model_name] = result
        
        # Print comparison
        print_section_header("MODEL COMPARISON")
        self.evaluator.print_comparison()
    
    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print_section_header("TEST SET EVALUATION")
        
        logger.info(f"Evaluating on test set: {X_test.shape}")
        
        test_evaluator = ModelEvaluator()
        
        for model_name, model in self.trained_models.items():
            logger.info(f"\nEvaluating {model_name} on test set...")
            
            # Predict
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
            
            # Evaluate
            result = test_evaluator.evaluate_model(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba
            )
            
            test_evaluator.print_evaluation(result)
        
        # Print comparison
        print_section_header("TEST SET MODEL COMPARISON")
        test_evaluator.print_comparison()
        
        return test_evaluator
    
    def save_all_models(self):
        """Save all trained models to disk."""
        print_section_header("SAVING MODELS")
        
        logger.info(f"Saving {len(self.trained_models)} models...")
        
        saved_count = 0
        for model_name, model in self.trained_models.items():
            # Get file path
            model_key = model_name.lower().replace(' ', '_')
            
            if model_key in MODEL_FILES:
                file_path = MODEL_FILES[model_key]
            else:
                # Fallback path
                from config import TRADITIONAL_ML_DIR
                file_path = TRADITIONAL_ML_DIR / f"{model_key}.pkl"
            
            # Save model
            try:
                joblib.dump(model, file_path)
                logger.info(f"Saved {model_name} to {file_path.name}")
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
        
        logger.info(f"Successfully saved {saved_count}/{len(self.trained_models)} models")
    
    def save_metadata(self):
        """Save training metadata and results."""
        logger.info("Saving metadata...")
        
        # Get best model
        comparison_df = self.evaluator.get_comparison_df()
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            
            metadata = {
                'num_models_trained': len(self.trained_models),
                'models': list(self.trained_models.keys()),
                'best_model': {
                    'name': best_model['Model'],
                    'accuracy': float(best_model['Accuracy']),
                    'f1_macro': float(best_model['F1 (Macro)']),
                    'f1_weighted': float(best_model['F1 (Weighted)'])
                },
                'all_results': {}
            }
            
            # Add all model results
            for model_name, result in self.results.items():
                metadata['all_results'][model_name] = {
                    'accuracy': result['metrics']['accuracy'],
                    'f1_macro': result['metrics']['f1_macro'],
                    'f1_weighted': result['metrics']['f1_weighted'],
                    'training_time': result.get('training_time', 0)
                }
            
            # Save metadata
            save_results(metadata, 'model_metadata.json')
            
            # Save comparison table
            comparison_df.to_csv(COMPARISON_FILE, index=False)
            logger.info(f"Saved comparison table to {COMPARISON_FILE}")
        
        logger.info("Metadata saved successfully")
    
    def get_best_model(self):
        """Get the best performing model."""
        comparison_df = self.evaluator.get_comparison_df()
        
        if comparison_df.empty:
            return None, None
        
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.trained_models.get(best_model_name)
        
        return best_model_name, best_model

# ==================== MAIN TRAINING FUNCTION ====================

def train_traditional_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    save_models: bool = True
):
    """
    Main function to train all traditional ML models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        save_models: Whether to save models
    
    Returns:
        Trained TraditionalMLTrainer instance
    """
    print_section_header("TRADITIONAL ML TRAINING PIPELINE")
    
    # Initialize trainer
    trainer = TraditionalMLTrainer()
    
    # Train all models
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        trainer.evaluate_on_test(X_test, y_test)
    
    # Save models and metadata
    if save_models:
        trainer.save_all_models()
        trainer.save_metadata()
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    
    print_section_header("TRAINING COMPLETE")
    logger.info(f"Best Model: {best_name}")
    logger.info(f"All models saved to {MODEL_FILES['random_forest'].parent}")
    
    return trainer

# ==================== TESTING ====================

if __name__ == "__main__":
    # This would normally be called from a main training script
    # For testing purposes, we'll create dummy data
    
    print("Testing traditional_ml_trainer.py...")
    print("\nNote: This requires actual data from data_handler and feature_extractor")
    print("Run the full training pipeline from a main script instead.")
    
    # Example usage (commented out):
    """
    from data_handler import load_and_split_data
    from feature_extractor import FeatureExtractor
    
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # Extract features
    extractor = FeatureExtractor()
    X_train_features = extractor.fit_transform(X_train)
    X_val_features = extractor.transform(X_val)
    X_test_features = extractor.transform(X_test)
    
    # Save feature extractor
    extractor.save()
    
    # Train models
    trainer = train_traditional_models(
        X_train_features, y_train,
        X_val_features, y_val,
        X_test_features, y_test,
        save_models=True
    )
    """