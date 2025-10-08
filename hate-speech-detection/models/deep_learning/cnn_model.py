"""
CNN Model for Hate Speech Classification
Convolutional Neural Network for text classification

Uses multiple filter sizes to detect n-gram patterns.
Often outperforms LSTM/BiLSTM for text classification!

Architecture:
- Embedding Layer (learns word representations)
- Multiple Conv1D Layers (3, 4, 5 word windows in parallel)
- GlobalMaxPooling (extract most important features)
- Concatenate (combine all filter outputs)
- Dense Layers (classification)
"""

import numpy as np
import time
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from config import (
    CNN_CONFIG, NUM_CLASSES, MODEL_FILES,
    EARLY_STOPPING_CONFIG, MODEL_CHECKPOINT_CONFIG,
    USE_EARLY_STOPPING, USE_MODEL_CHECKPOINT, CLASS_WEIGHTS
)

# ==================== CNN MODEL ====================

class CNNModel:
    """
    CNN-based text classifier for hate speech detection.
    
    Architecture (Kim CNN):
        Input (sequences)
            ↓
        Embedding Layer (trainable word embeddings)
            ↓
        ┌─────────────┬─────────────┬─────────────┐
        │ Conv1D (3)  │ Conv1D (4)  │ Conv1D (5)  │  Parallel convolutions
        │     ↓       │     ↓       │     ↓       │  (different n-gram sizes)
        │ MaxPool1D   │ MaxPool1D   │ MaxPool1D   │
        └─────────────┴─────────────┴─────────────┘
                            ↓
                      Concatenate
                            ↓
                    Dense (128, ReLU)
                            ↓
                       Dropout (0.5)
                            ↓
                Output (Softmax, 3 classes)
    
    Key Advantages:
    - Fast training (parallel convolutions)
    - Detects local patterns (n-grams)
    - No sequential processing needed
    - Often best performer for text classification
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize CNN model.
        
        Args:
            config: Model configuration (uses CNN_CONFIG if None)
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for CNN model.\n"
                "Install with: pip install tensorflow==2.13.0"
            )
        
        self.config = config or CNN_CONFIG
        self.model = None
        self.history = None
        self.training_time = None
        
        # Extract config
        self.vocab_size = self.config['vocab_size']
        self.embedding_dim = self.config['embedding_dim']
        self.filter_sizes = self.config['filter_sizes']  # [3, 4, 5]
        self.num_filters = self.config['num_filters']    # 128 per size
        self.dropout = self.config['dropout']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.learning_rate = self.config['learning_rate']
    
    def build_model(self):
        """
        Build CNN model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.max_length,), name='input')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # Multiple Conv1D layers with different filter sizes (parallel)
        conv_blocks = []
        for i, filter_size in enumerate(self.filter_sizes):
            # Conv1D layer
            conv = layers.Conv1D(
                filters=self.num_filters,
                kernel_size=filter_size,
                activation='relu',
                padding='valid',
                name=f'conv1d_{filter_size}'
            )(embedding)
            
            # Global max pooling (get most important feature)
            pool = layers.GlobalMaxPooling1D(
                name=f'maxpool_{filter_size}'
            )(conv)
            
            conv_blocks.append(pool)
        
        # Concatenate all pooled features
        if len(conv_blocks) > 1:
            x = layers.Concatenate(name='concatenate')(conv_blocks)
        else:
            x = conv_blocks[0]
        
        # Dense layer for feature transformation
        x = layers.Dense(
            128,
            activation='relu',
            name='dense_1'
        )(x)
        
        # Dropout for regularization
        x = layers.Dropout(self.dropout, name='dropout')(x)
        
        # Output layer (3 classes: hate, offensive, neither)
        outputs = layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='CNN_Classifier')
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, checkpoint_path: str = None) -> List:
        """
        Get training callbacks.
        
        Args:
            checkpoint_path: Path to save best model
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping
        if USE_EARLY_STOPPING:
            early_stop = EarlyStopping(**EARLY_STOPPING_CONFIG)
            callbacks.append(early_stop)
        
        # Model checkpoint
        if USE_MODEL_CHECKPOINT and checkpoint_path:
            checkpoint = ModelCheckpoint(
                checkpoint_path,
                **MODEL_CHECKPOINT_CONFIG
            )
            callbacks.append(checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the CNN model.
        
        Args:
            X_train: Training sequences (n_samples, max_length)
            y_train: Training labels (n_samples,)
            X_val: Validation sequences
            y_val: Validation labels
            verbose: Verbosity level (0=silent, 1=progress, 2=one line per epoch)
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_split = self.config.get('validation_split', 0.2)
        
        # Get callbacks
        checkpoint_path = str(MODEL_FILES['cnn'])
        callbacks = self.get_callbacks(checkpoint_path)
        
        # Train model
        start_time = time.time()
        
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            validation_split=validation_split,
            class_weight=CLASS_WEIGHTS,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        self.history = history.history
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input sequences (n_samples, max_length)
        
        Returns:
            Predicted class labels (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before prediction")
        
        probabilities = self.model.predict(X, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input sequences (n_samples, max_length)
        
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before prediction")
        
        probabilities = self.model.predict(X, verbose=0)
        return probabilities
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 0
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            verbose: Verbosity level
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before evaluation")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        if verbose > 0:
            from config import CLASS_LABELS
            target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
            report = classification_report(
                y_test, y_pred,
                target_names=target_names,
                digits=4,
                zero_division=0
            )
            print("\nClassification Report:")
            print(report)
        
        return metrics
    
    def save(self, filepath: str = None):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (uses config default if None)
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        filepath = filepath or MODEL_FILES['cnn']
        self.model.save(filepath)
    
    @staticmethod
    def load(filepath: str = None) -> 'CNNModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load from (uses config default if None)
        
        Returns:
            Loaded CNNModel instance
        """
        filepath = filepath or MODEL_FILES['cnn']
        
        # Load Keras model
        keras_model = load_model(filepath)
        
        # Create CNNModel wrapper
        cnn_model = CNNModel()
        cnn_model.model = keras_model
        
        return cnn_model
    
    def summary(self):
        """Print model architecture summary."""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
        else:
            self.model.summary()
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        if self.model is None:
            return {
                'model_type': 'CNN',
                'built': False
            }
        
        info = {
            'model_type': 'CNN',
            'built': True,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'filter_sizes': self.filter_sizes,
            'num_filters_per_size': self.num_filters,
            'total_filters': len(self.filter_sizes) * self.num_filters,
            'max_length': self.max_length,
            'num_classes': NUM_CLASSES,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        }
        
        if self.training_time is not None:
            info['training_time'] = self.training_time
        
        if self.history is not None:
            info['final_train_accuracy'] = self.history['accuracy'][-1]
            if 'val_accuracy' in self.history:
                info['final_val_accuracy'] = self.history['val_accuracy'][-1]
        
        return info
    
    def plot_training_history(self):
        """
        Plot training history (requires matplotlib).
        """
        if self.history is None:
            print("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot accuracy
            axes[0].plot(self.history['accuracy'], label='Train')
            if 'val_accuracy' in self.history:
                axes[0].plot(self.history['val_accuracy'], label='Validation')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True)
            
            # Plot loss
            axes[1].plot(self.history['loss'], label='Train')
            if 'val_loss' in self.history:
                axes[1].plot(self.history['val_loss'], label='Validation')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Cannot plot training history.")
    
    def __repr__(self):
        """String representation."""
        if self.model is None:
            return "CNNModel(not built)"
        else:
            params = self.model.count_params()
            filters = f"{self.filter_sizes}"
            return f"CNNModel(params={params:,}, filters={filters})"

# ==================== CONVENIENCE FUNCTIONS ====================

def create_cnn_model(config: Dict = None) -> CNNModel:
    """
    Create and build CNN model.
    
    Args:
        config: Model configuration
    
    Returns:
        Built CNNModel instance
    """
    model = CNNModel(config=config)
    model.build_model()
    return model

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("CNN MODEL TEST")
    print("=" * 70)
    
    # Check TensorFlow
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create model
    print("\nBuilding CNN model...")
    model = CNNModel()
    model.build_model()
    
    # Print summary
    print("\nModel Architecture:")
    model.summary()
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    print(f"  Filter sizes: {info['filter_sizes']}")
    print(f"  Filters per size: {info['num_filters_per_size']}")
    print(f"  Total filters: {info['total_filters']}")
    print(f"  Embedding dim: {info['embedding_dim']}")
    print(f"  Max length: {info['max_length']}")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    n_samples = 100
    X_dummy = np.random.randint(0, 1000, size=(n_samples, model.max_length))
    y_dummy = np.random.randint(0, NUM_CLASSES, size=(n_samples,))
    
    # Test prediction (before training)
    predictions = model.predict(X_dummy[:10])
    print(f" Prediction works: {predictions.shape}")
    
    probabilities = model.predict_proba(X_dummy[:10])
    print(f" Predict proba works: {probabilities.shape}")
    
    # Test training (1 epoch for speed)
    print("\nTraining for 1 epoch (test)...")
    test_config = CNN_CONFIG.copy()
    test_config['epochs'] = 1
    test_config['batch_size'] = 16
    
    test_model = CNNModel(config=test_config)
    test_model.build_model()
    
    history = test_model.train(
        X_dummy[:80],
        y_dummy[:80],
        X_dummy[80:],
        y_dummy[80:],
        verbose=0
    )
    
    print(f" Training works")
    print(f"  Training time: {test_model.training_time:.2f}s")
    print(f"  Final accuracy: {history['accuracy'][-1]:.4f}")
    
    # Test evaluation
    metrics = test_model.evaluate(X_dummy[:20], y_dummy[:20], verbose=0)
    print(f" Evaluation works")
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    
    # Test save/load
    print("\nTesting save/load...")
    test_model.save()
    print(" Model saved")
    
    loaded_model = CNNModel.load()
    print(" Model loaded")
    
    # Verify loaded model works
    test_pred = loaded_model.predict(X_dummy[:5])
    print(f" Loaded model works: {test_pred.shape}")
    
    print("\n" + "=" * 70)
    print("CNN MODEL TEST COMPLETE ")
    print("=" * 70)