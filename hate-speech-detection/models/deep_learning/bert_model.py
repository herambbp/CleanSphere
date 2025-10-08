"""
BERT Model for Hate Speech Classification
Bidirectional Encoder Representations from Transformers

State-of-the-art transformer-based model using pretrained BERT.
Best accuracy but slowest training (~30 min on CPU).

Architecture:
- BERT Tokenizer (WordPiece tokenization)
- Pretrained BERT Base (12 transformer layers)
- Dropout
- Classification Head (3 classes)
"""

import numpy as np
import time
from typing import Tuple, Dict, List
import warnings
from utils import logger
warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Transformers imports
try:
    from transformers import (
        BertTokenizer,
        TFBertForSequenceClassification,
        create_optimizer
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from config import (
    BERT_CONFIG, NUM_CLASSES, MODEL_FILES,
    EARLY_STOPPING_CONFIG, USE_EARLY_STOPPING, CLASS_WEIGHTS
)

# ==================== BERT MODEL ====================

class BERTModel:
    """
    BERT-based text classifier for hate speech detection.
    
    Architecture:
        Input (text)
            ↓
        BERT Tokenizer (WordPiece)
            ↓
        BERT Base Encoder (pretrained, 12 layers)
            ├─ Self-Attention Layers
            ├─ Feed-Forward Layers
            └─ Layer Normalization
            ↓
        [CLS] Token Output (768-dim)
            ↓
        Dropout (0.3)
            ↓
        Dense Classification Head (3 classes)
    
    Key Advantages:
    - Pretrained on massive text corpus
    - Bidirectional context understanding
    - Transfer learning (fine-tuning)
    - State-of-the-art accuracy (93-95%)
    
    Requirements:
    - transformers library
    - torch (PyTorch)
    - More memory (~4GB)
    - Longer training time (~30 min CPU)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize BERT model.
        
        Args:
            config: Model configuration (uses BERT_CONFIG if None)
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for BERT model.\n"
                "Install with: pip install tensorflow==2.13.0"
            )
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers is required for BERT model.\n"
                "Install with: pip install transformers==4.30.0 torch==2.0.1"
            )
        
        self.config = config or BERT_CONFIG
        self.model = None
        self.tokenizer = None
        self.history = None
        self.training_time = None
        
        # Extract config
        self.model_name = self.config['model_name']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.learning_rate = self.config['learning_rate']
        self.warmup_steps = self.config['warmup_steps']
        self.weight_decay = self.config['weight_decay']
        
        # Initialize tokenizer
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize BERT tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
    
    def build_model(self):
        """
        Build BERT model architecture.
        
        Returns:
            Compiled BERT model
        """
        # Load pretrained BERT for sequence classification
        self.model = TFBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_CLASSES
        )
        
        return self.model
    
    def tokenize_texts(
        self,
        texts: List[str],
        max_length: int = None
    ) -> Dict:
        """
        Tokenize texts using BERT tokenizer.
        
        Args:
            texts: List of texts
            max_length: Maximum length (uses config if None)
        
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        max_length = max_length or self.max_length
        
        # Tokenize with BERT tokenizer
        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='tf',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded['token_type_ids']
        }
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: np.ndarray = None,
        batch_size: int = None
    ) -> tf.data.Dataset:
        """
        Prepare TensorFlow dataset for training.
        
        Args:
            texts: List of texts
            labels: Labels array (optional for inference)
            batch_size: Batch size (uses config if None)
        
        Returns:
            TensorFlow dataset
        """
        batch_size = batch_size or self.batch_size
        
        # Tokenize texts
        encoded = self.tokenize_texts(texts)
        
        # Create dataset
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': encoded['input_ids'],
                    'attention_mask': encoded['attention_mask'],
                    'token_type_ids': encoded['token_type_ids']
                },
                labels
            ))
        else:
            dataset = tf.data.Dataset.from_tensor_slices({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'token_type_ids': encoded['token_type_ids']
            })
        
        # Shuffle and batch
        if labels is not None:
            dataset = dataset.shuffle(1000).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)
        
        return dataset
    
    def train(
        self,
        X_train: List[str],  # Note: Expects raw texts, not sequences!
        y_train: np.ndarray,
        X_val: List[str] = None,
        y_val: np.ndarray = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the BERT model.
        
        Args:
            X_train: Training texts (raw strings, NOT tokenized)
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            verbose: Verbosity level
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(X_train, y_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = self.prepare_dataset(X_val, y_val)
        
        # Calculate training steps
        num_train_steps = len(X_train) // self.batch_size * self.epochs
        
        # Create optimizer with warmup
        optimizer, lr_schedule = create_optimizer(
            init_lr=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=self.warmup_steps,
            weight_decay_rate=self.weight_decay
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Prepare callbacks
        callbacks = []
        # Skip EarlyStopping for BERT due to compatibility issues with Transformers
        # The model will train for all epochs but still uses learning rate scheduling
        logger.info("Note: Early stopping disabled for BERT (compatibility)")
        
        # Train model
        start_time = time.time()
        
        history = self.model.fit(
            train_dataset,
            validation_data=validation_data,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=CLASS_WEIGHTS,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        self.history = history.history
        
        return self.history
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            texts: List of texts (raw strings)
        
        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before prediction")
        
        # Prepare dataset
        dataset = self.prepare_dataset(texts, labels=None)
        
        # Predict
        predictions = self.model.predict(dataset, verbose=0)
        logits = predictions.logits
        
        # Get predicted classes
        predicted_classes = np.argmax(logits, axis=1)
        
        return predicted_classes
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of texts (raw strings)
        
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before prediction")
        
        # Prepare dataset
        dataset = self.prepare_dataset(texts, labels=None)
        
        # Predict
        predictions = self.model.predict(dataset, verbose=0)
        logits = predictions.logits
        
        # Convert logits to probabilities
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        
        return probabilities
    
    def evaluate(
        self,
        texts: List[str],
        labels: np.ndarray,
        verbose: int = 0
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            texts: Test texts
            labels: Test labels
            verbose: Verbosity level
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before evaluation")
        
        # Get predictions
        y_pred = self.predict(texts)
        y_pred_proba = self.predict_proba(texts)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(labels, y_pred),
            'precision_macro': precision_score(labels, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(labels, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(labels, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(labels, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, y_pred, average='weighted', zero_division=0),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        if verbose > 0:
            from config import CLASS_LABELS
            target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
            report = classification_report(
                labels, y_pred,
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
        
        filepath = filepath or MODEL_FILES['bert']
        
        # Save BERT model
        self.model.save_pretrained(filepath)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(filepath)
    
    @staticmethod
    def load(filepath: str = None) -> 'BERTModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load from (uses config default if None)
        
        Returns:
            Loaded BERTModel instance
        """
        filepath = filepath or MODEL_FILES['bert']
        
        # Create BERTModel instance
        bert_model = BERTModel()
        
        # Load BERT model
        bert_model.model = TFBertForSequenceClassification.from_pretrained(filepath)
        
        # Load tokenizer
        bert_model.tokenizer = BertTokenizer.from_pretrained(filepath)
        
        return bert_model
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        if self.model is None:
            return {
                'model_type': 'BERT',
                'built': False
            }
        
        info = {
            'model_type': 'BERT',
            'built': True,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_classes': NUM_CLASSES,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else None,
            'total_params': sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        }
        
        if self.training_time is not None:
            info['training_time'] = self.training_time
        
        if self.history is not None:
            info['final_train_accuracy'] = self.history['accuracy'][-1]
            if 'val_accuracy' in self.history:
                info['final_val_accuracy'] = self.history['val_accuracy'][-1]
        
        return info
    
    def __repr__(self):
        """String representation."""
        if self.model is None:
            return f"BERTModel(model_name='{self.model_name}', not built)"
        else:
            params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
            return f"BERTModel(model_name='{self.model_name}', params={params:,})"

# ==================== CONVENIENCE FUNCTIONS ====================

def create_bert_model(config: Dict = None) -> BERTModel:
    """
    Create and build BERT model.
    
    Args:
        config: Model configuration
    
    Returns:
        Built BERTModel instance
    """
    model = BERTModel(config=config)
    model.build_model()
    return model

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("BERT MODEL TEST")
    print("=" * 70)
    
    # Check dependencies
    if not HAS_TENSORFLOW:
        print("\n❌ TensorFlow not installed")
        print("Install: pip install tensorflow==2.13.0")
        exit(1)
    
    if not HAS_TRANSFORMERS:
        print("\n❌ Transformers not installed")
        print("Install: pip install transformers==4.30.0 torch==2.0.1")
        exit(1)
    
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create model
    print("\nInitializing BERT model...")
    print("(This will download pretrained weights if not cached)")
    model = BERTModel()
    model.build_model()
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Model name: {info['model_name']}")
    print(f"  Max length: {info['max_length']}")
    print(f"  Vocab size: {info['vocab_size']:,}")
    print(f"  Total parameters: {info['total_params']:,}")
    
    # Test tokenization
    print("\nTesting tokenization...")
    test_texts = [
        "I hate you",
        "Good morning everyone",
        "You are an idiot"
    ]
    
    encoded = model.tokenize_texts(test_texts)
    print(f" Tokenization works")
    print(f"  Input IDs shape: {encoded['input_ids'].shape}")
    print(f"  Attention mask shape: {encoded['attention_mask'].shape}")
    
    # Show tokenization example
    print(f"\nTokenization example:")
    print(f"  Text: '{test_texts[0]}'")
    tokens = model.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    print(f"  Tokens: {tokens[:10]}...")
    
    # Test prediction (before training)
    print("\nTesting prediction (before training)...")
    predictions = model.predict(test_texts)
    print(f" Prediction works: {predictions.shape}")
    
    probabilities = model.predict_proba(test_texts)
    print(f" Predict proba works: {probabilities.shape}")
    print(f"  Sample probabilities: {probabilities[0]}")
    
    # Test training (1 epoch with small data)
    print("\nTraining for 1 epoch (small test)...")
    print("(This will take 2-3 minutes even with small data)")
    
    # Create small dummy dataset
    dummy_texts = [
        "I hate you", "You're stupid", "Good morning",
        "Nice day", "You're an idiot", "Have a great day",
        "I love this", "This is bad", "Amazing work"
    ] * 5  # 45 samples
    
    dummy_labels = np.array([0, 1, 2, 2, 1, 2, 2, 1, 2] * 5)
    
    test_config = BERT_CONFIG.copy()
    test_config['epochs'] = 1
    test_config['batch_size'] = 8
    
    test_model = BERTModel(config=test_config)
    test_model.build_model()
    
    history = test_model.train(
        dummy_texts[:36],
        dummy_labels[:36],
        dummy_texts[36:],
        dummy_labels[36:],
        verbose=1
    )
    
    print(f"\n Training works")
    print(f"  Training time: {test_model.training_time:.2f}s")
    print(f"  Final accuracy: {history['accuracy'][-1]:.4f}")
    
    # Test evaluation
    metrics = test_model.evaluate(dummy_texts[:9], dummy_labels[:9], verbose=0)
    print(f" Evaluation works")
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    
    # Test save/load
    print("\nTesting save/load...")
    test_model.save()
    print(" Model saved")
    
    loaded_model = BERTModel.load()
    print(" Model loaded")
    
    # Verify loaded model works
    test_pred = loaded_model.predict(test_texts)
    print(f" Loaded model works: {test_pred.shape}")
    
    print("\n" + "=" * 70)
    print("BERT MODEL TEST COMPLETE ")
    print("=" * 70)
    print("\nNote: Full training on 24k samples will take ~30 min on CPU")
    print("      Consider using GPU for faster training (5-10x speedup)")