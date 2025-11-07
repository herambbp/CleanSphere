"""
PyTorch Deep Learning Trainer Integration
Replaces TensorFlow/Keras trainer with PyTorch version
Maintains same interface for compatibility with main training script
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pytorch_dl_models import (
    PyTorchDLTrainer,
    PyTorchPredictor,
    DEVICE
)
from utils import logger

# Check if PyTorch is available
try:
    import torch
    HAS_PYTORCH = True
    logger.info(f"[OK] PyTorch version: {torch.__version__}")
    logger.info(f"[OK] CUDA available: {torch.cuda.is_available()}")
except ImportError:
    HAS_PYTORCH = False
    logger.warning("[WARN] PyTorch not available")


def train_pytorch_deep_learning_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vocab_size: int = 20000,
    max_length: int = 100,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    save_dir: Path = Path('saved_models/pytorch_dl')
) -> Optional[PyTorchDLTrainer]:
    """
    Train PyTorch deep learning models (LSTM, BiLSTM, CNN)
    
    This function replaces the TensorFlow/Keras training function
    and maintains the same interface for compatibility.
    
    Args:
        X_train, y_train: Training data (sequences and labels)
        X_val, y_val: Validation data
        X_test, y_test: Test data
        vocab_size: Vocabulary size
        max_length: Maximum sequence length
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        save_dir: Directory to save models
    
    Returns:
        PyTorchDLTrainer instance with trained models
    """
    if not HAS_PYTORCH:
        logger.error("[ERROR] PyTorch not available!")
        logger.error("Install with: pip install torch")
        return None
    
    logger.info("\n" + "="*80)
    logger.info("PYTORCH DEEP LEARNING TRAINING (CUDA ACCELERATED)")
    logger.info("="*80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Training samples: {len(y_train):,}")
    logger.info(f"Validation samples: {len(y_val):,}")
    logger.info(f"Test samples: {len(y_test):,}")
    logger.info(f"Vocabulary size: {vocab_size:,}")
    logger.info(f"Max sequence length: {max_length}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("="*80 + "\n")
    
    # Validate input shapes
    assert X_train.shape[0] == len(y_train), "X_train and y_train size mismatch"
    assert X_val.shape[0] == len(y_val), "X_val and y_val size mismatch"
    assert X_test.shape[0] == len(y_test), "X_test and y_test size mismatch"
    assert X_train.shape[1] == max_length, f"X_train shape mismatch: expected {max_length}, got {X_train.shape[1]}"
    
    # Initialize trainer
    trainer = PyTorchDLTrainer(
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=3  # Hate, Offensive, Neither
    )
    
    # Train all models
    try:
        models, histories = trainer.train_all_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            save_dir=save_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        logger.info("\n[SUCCESS] PyTorch training completed!")
        logger.info(f"[BEST] {trainer.best_model_name.upper()} - Accuracy: {trainer.best_accuracy:.4f}")
        logger.info(f"[SAVED] Models saved to: {save_dir}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_pytorch_model(
    model_name: str,
    load_dir: Path = Path('saved_models/pytorch_dl')
) -> Optional[PyTorchPredictor]:
    """
    Load a trained PyTorch model for inference
    
    Args:
        model_name: Model name ('lstm', 'bilstm', 'cnn')
        load_dir: Directory containing saved models
    
    Returns:
        PyTorchPredictor instance
    """
    if not HAS_PYTORCH:
        logger.error("[ERROR] PyTorch not available!")
        return None
    
    try:
        trainer = PyTorchDLTrainer()
        model = trainer.load_model(model_name, load_dir)
        predictor = PyTorchPredictor(model)
        
        logger.info(f"[LOADED] {model_name.upper()} model ready for inference")
        return predictor
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
        return None


def get_pytorch_model_info(load_dir: Path = Path('saved_models/pytorch_dl')) -> Dict:
    """
    Get information about available PyTorch models
    
    Returns:
        Dictionary with model information
    """
    info = {
        'pytorch_available': HAS_PYTORCH,
        'device': str(DEVICE) if HAS_PYTORCH else 'N/A',
        'cuda_available': torch.cuda.is_available() if HAS_PYTORCH else False,
        'models': []
    }
    
    if HAS_PYTORCH and load_dir.exists():
        for model_file in load_dir.glob('*.pt'):
            model_name = model_file.stem
            info['models'].append(model_name)
    
    return info


# ==================== COMPATIBILITY WRAPPER ====================

class PyTorchDLCompatibilityWrapper:
    """
    Wrapper to maintain compatibility with TensorFlow trainer interface
    """
    
    def __init__(self, pytorch_trainer: PyTorchDLTrainer):
        self.pytorch_trainer = pytorch_trainer
        self.models = pytorch_trainer.models
        self.histories = pytorch_trainer.histories
        self.best_model_name = pytorch_trainer.best_model_name
        self.best_accuracy = pytorch_trainer.best_accuracy
    
    def get_best_model_name(self) -> str:
        """Get best performing model name"""
        return self.best_model_name
    
    def get_best_metrics(self) -> Dict:
        """Get best model metrics"""
        return {
            'test_accuracy': self.best_accuracy,
            'model_name': self.best_model_name
        }
    
    def get_model(self, model_name: str):
        """Get a specific model"""
        return self.models.get(model_name)
    
    def get_history(self, model_name: str) -> Dict:
        """Get training history for a model"""
        return self.histories.get(model_name)
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictor = PyTorchPredictor(model)
        return predictor.predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictor = PyTorchPredictor(model)
        return predictor.predict_proba(X)


# ==================== MAIN INTERFACE ====================

def train_deep_learning_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_bert: bool = False,
    **kwargs
) -> Optional[PyTorchDLCompatibilityWrapper]:
    """
    Main interface function that replaces TensorFlow trainer
    
    This function signature matches the original TensorFlow trainer
    for drop-in replacement in main_train_enhanced.py
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        use_bert: Whether to include BERT (not implemented in PyTorch version)
        **kwargs: Additional arguments
    
    Returns:
        Compatibility wrapper with trained models
    """
    if not HAS_PYTORCH:
        logger.error("[ERROR] PyTorch not available!")
        logger.error("Install with: pip install torch")
        return None
    
    if use_bert:
        logger.warning("[WARN] BERT training not implemented in PyTorch version")
        logger.warning("[WARN] Training LSTM, BiLSTM, CNN only")
    
    # Extract parameters
    vocab_size = kwargs.get('vocab_size', 20000)
    max_length = kwargs.get('max_length', 100)
    batch_size = kwargs.get('batch_size', 32)
    epochs = kwargs.get('epochs', 10)
    learning_rate = kwargs.get('learning_rate', 0.001)
    save_dir = kwargs.get('save_dir', Path('saved_models/pytorch_dl'))
    
    # Train models
    pytorch_trainer = train_pytorch_deep_learning_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        vocab_size=vocab_size,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    if pytorch_trainer is None:
        return None
    
    # Wrap for compatibility
    wrapper = PyTorchDLCompatibilityWrapper(pytorch_trainer)
    return wrapper


# ==================== EXPORT ====================

__all__ = [
    'train_deep_learning_models',
    'train_pytorch_deep_learning_models',
    'load_pytorch_model',
    'get_pytorch_model_info',
    'PyTorchDLTrainer',
    'PyTorchPredictor',
    'HAS_PYTORCH'
]


if __name__ == "__main__":
    # Test module
    print("\n" + "="*80)
    print("PYTORCH DEEP LEARNING TRAINER MODULE")
    print("="*80)
    print(f"PyTorch available: {HAS_PYTORCH}")
    
    if HAS_PYTORCH:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device: {DEVICE}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\nInstall PyTorch with:")
        print("  pip install torch")
        print("\nFor CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*80)