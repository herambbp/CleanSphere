"""
Deep Learning Trainer for Hate Speech Detection
Trains LSTM, BiLSTM, CNN, and optionally BERT models

PHASE 5: Deep Learning Models

FIXES:
- GPU detection and reporting
- Windows Unicode compatibility (no emojis)
- BERT compatibility issues
- Better error handling
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Check TensorFlow availability
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from config import (
    LSTM_CONFIG, BILSTM_CONFIG, CNN_CONFIG, BERT_CONFIG,
    MODEL_FILES, DL_COMPARISON_FILE, RESULTS_DIR,
    USE_GPU
)
from utils import (
    logger, print_section_header, print_subsection_header,
    ModelEvaluator, format_time, save_results
)

# Import deep learning models
if HAS_TENSORFLOW:
    from models.deep_learning.text_tokenizer import TextTokenizer
    from models.deep_learning.lstm_model import LSTMModel
    from models.deep_learning.bilstm_model import BiLSTMModel
    from models.deep_learning.cnn_model import CNNModel
    
    # Try importing BERT (optional)
    try:
        from models.deep_learning.bert_model import BERTModel
        HAS_BERT = True
    except ImportError:
        HAS_BERT = False
        logger.warning("BERT model not available. Install transformers and torch to enable BERT.")
else:
    logger.error("TensorFlow not installed! Cannot use deep learning models.")
    logger.error("Install with: pip install tensorflow==2.13.0")

# ==================== GPU CONFIGURATION ====================

def configure_gpu():
    """
    Configure GPU settings for TensorFlow with detailed reporting.
    
    FIXED: Better GPU detection and ASCII-safe output
    """
    if not HAS_TENSORFLOW:
        return False
    
    print_section_header("GPU CONFIGURATION")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to prevent TF from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logger.info(f"[GPU DETECTED] {len(gpus)} device(s) available")
            
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.name}")
                
                # Try to get more GPU details
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if 'device_name' in gpu_details:
                        logger.info(f"    Device Name: {gpu_details['device_name']}")
                    if 'compute_capability' in gpu_details:
                        logger.info(f"    Compute Capability: {gpu_details['compute_capability']}")
                except Exception:
                    pass
            
            # Test GPU computation
            logger.info("\nTesting GPU computation...")
            try:
                with tf.device('/GPU:0'):
                    # Simple matrix multiplication test
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                    _ = c.numpy()  # Force execution
                
                logger.info("[SUCCESS] GPU computation test passed")
                logger.info("Training will use GPU (5-10x faster than CPU)")
                return True
                
            except Exception as e:
                logger.warning(f"[WARNING] GPU test failed: {e}")
                logger.warning("Falling back to CPU")
                return False
            
        except RuntimeError as e:
            logger.warning(f"[WARNING] GPU configuration failed: {e}")
            logger.warning("Training will use CPU")
            return False
    else:
        logger.info("[NO GPU] Training will use CPU")
        logger.info("Expected training times on CPU:")
        logger.info("  - LSTM: ~2-3 minutes")
        logger.info("  - BiLSTM: ~6-8 minutes")
        logger.info("  - CNN: ~1 minute")
        logger.info("  - BERT: ~30-40 minutes")
        logger.info("\nTIP: Use Google Colab or Kaggle for free GPU access")
        return False

# ==================== DEEP LEARNING TRAINER ====================

class DeepLearningTrainer:
    """
    Train and evaluate deep learning models for hate speech detection.
    
    Models:
    - LSTM: Long Short-Term Memory
    - BiLSTM: Bidirectional LSTM
    - CNN: Convolutional Neural Network
    - BERT: Transformer-based (optional)
    """
    
    def __init__(self, use_bert: bool = False):
        """
        Initialize Deep Learning Trainer.
        
        Args:
            use_bert: Whether to train BERT model (requires more resources)
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for deep learning models.\n"
                "Install with: pip install tensorflow==2.13.0"
            )
        
        logger.info("Initializing Deep Learning Trainer...")
        
        self.use_bert = use_bert and HAS_BERT
        self.tokenizer = None
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.evaluator = ModelEvaluator()
        
        # Configure GPU
        self.has_gpu = configure_gpu() if USE_GPU else False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all deep learning models."""
        logger.info("Initializing deep learning models...")
        
        # LSTM
        self.models['LSTM'] = LSTMModel(config=LSTM_CONFIG)
        
        # BiLSTM
        self.models['BiLSTM'] = BiLSTMModel(config=BILSTM_CONFIG)
        
        # CNN
        self.models['CNN'] = CNNModel(config=CNN_CONFIG)
        
        # BERT (optional)
        if self.use_bert:
            self.models['BERT'] = BERTModel(config=BERT_CONFIG)
            logger.info("BERT model included (this will significantly increase training time)")
        else:
            logger.info("BERT model excluded (set use_bert=True to include)")
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def prepare_tokenizer(
        self,
        X_train: np.ndarray,
        save_tokenizer: bool = True
    ) -> TextTokenizer:
        """
        Prepare and fit tokenizer on training data.
        
        Args:
            X_train: Training texts (raw strings)
            save_tokenizer: Whether to save tokenizer
        
        Returns:
            Fitted TextTokenizer
        """
        print_section_header("TOKENIZER PREPARATION")
        
        logger.info("Initializing TextTokenizer...")
        logger.info(f"Vocabulary size: {LSTM_CONFIG['vocab_size']:,}")
        logger.info(f"Max sequence length: {LSTM_CONFIG['max_length']}")
        
        # Create tokenizer
        self.tokenizer = TextTokenizer(
            vocab_size=LSTM_CONFIG['vocab_size'],
            max_length=LSTM_CONFIG['max_length']
        )
        
        # Fit on training data
        logger.info(f"Fitting tokenizer on {len(X_train):,} training samples...")
        start_time = time.time()
        
        stats = self.tokenizer.fit_on_texts(X_train.tolist())
        
        fit_time = time.time() - start_time
        
        # Log statistics
        logger.info(f"Tokenizer fitted in {fit_time:.2f}s")
        logger.info(f"Total unique words: {stats['total_unique_words']:,}")
        logger.info(f"Vocabulary coverage: {stats['coverage']:.2%}")
        logger.info(f"Actual vocab size: {self.tokenizer.get_vocab_size():,}")
        
        # Show top words
        logger.info("\nTop 15 most frequent words:")
        for word, idx in self.tokenizer.get_top_words(15):
            logger.info(f"  {idx:4d}. {word}")
        
        # Save tokenizer
        if save_tokenizer:
            logger.info("\nSaving tokenizer...")
            self.tokenizer.save()
            logger.info(f"Tokenizer saved to {MODEL_FILES['tokenizer']}")
        
        return self.tokenizer
    
    def tokenize_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokenize all datasets for LSTM/BiLSTM/CNN models.
        
        Args:
            X_train: Training texts
            X_val: Validation texts
            X_test: Test texts
        
        Returns:
            Tuple of tokenized (X_train, X_val, X_test)
        """
        print_section_header("DATA TOKENIZATION")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be prepared before tokenizing data")
        
        logger.info("Tokenizing datasets for LSTM/BiLSTM/CNN models...")
        
        # Tokenize training data
        logger.info(f"Tokenizing {len(X_train):,} training samples...")
        start_time = time.time()
        X_train_seq = self.tokenizer.texts_to_padded_sequences(X_train.tolist())
        train_time = time.time() - start_time
        logger.info(f"  Training data: {X_train_seq.shape} in {train_time:.2f}s")
        
        # Tokenize validation data
        logger.info(f"Tokenizing {len(X_val):,} validation samples...")
        start_time = time.time()
        X_val_seq = self.tokenizer.texts_to_padded_sequences(X_val.tolist())
        val_time = time.time() - start_time
        logger.info(f"  Validation data: {X_val_seq.shape} in {val_time:.2f}s")
        
        # Tokenize test data
        logger.info(f"Tokenizing {len(X_test):,} test samples...")
        start_time = time.time()
        X_test_seq = self.tokenizer.texts_to_padded_sequences(X_test.tolist())
        test_time = time.time() - start_time
        logger.info(f"  Test data: {X_test_seq.shape} in {test_time:.2f}s")
        
        total_time = train_time + val_time + test_time
        logger.info(f"\nTotal tokenization time: {format_time(total_time)}")
        
        return X_train_seq, X_val_seq, X_test_seq
    
    def train_sequence_model(
        self,
        model_name: str,
        model,
        X_train_seq: np.ndarray,
        y_train: np.ndarray,
        X_val_seq: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        Train a sequence model (LSTM/BiLSTM/CNN).
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train_seq: Training sequences (tokenized)
            y_train: Training labels
            X_val_seq: Validation sequences (tokenized)
            y_val: Validation labels
        
        Returns:
            Dictionary with results
        """
        print_section_header(f"TRAINING {model_name}")
        
        # Build model
        logger.info(f"Building {model_name} architecture...")
        model.build_model()
        
        # Print model summary
        print_subsection_header(f"{model_name} Architecture")
        model.summary()
        
        # Get model info
        info = model.get_model_info()
        logger.info(f"\nModel Configuration:")
        logger.info(f"  Total parameters: {info['total_params']:,}")
        logger.info(f"  Trainable parameters: {info['trainable_params']:,}")
        logger.info(f"  Embedding dim: {info['embedding_dim']}")
        logger.info(f"  Max length: {info['max_length']}")
        
        # Train model
        logger.info(f"\nTraining {model_name}...")
        logger.info(f"  Training samples: {len(X_train_seq):,}")
        logger.info(f"  Validation samples: {len(X_val_seq):,}")
        logger.info(f"  Batch size: {model.batch_size}")
        logger.info(f"  Epochs: {model.epochs}")
        logger.info(f"  Device: {'GPU' if self.has_gpu else 'CPU'}")
        
        try:
            # Train
            history = model.train(
                X_train_seq,
                y_train,
                X_val_seq,
                y_val,
                verbose=1
            )
            
            training_time = model.training_time
            logger.info(f"\n{model_name} training completed in {format_time(training_time)}")
            
            # Show training history
            logger.info(f"\nTraining History:")
            logger.info(f"  Final train accuracy: {history['accuracy'][-1]:.4f}")
            logger.info(f"  Final train loss: {history['loss'][-1]:.4f}")
            if 'val_accuracy' in history:
                logger.info(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")
                logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
                logger.info(f"  Best val accuracy: {max(history['val_accuracy']):.4f}")
            
            # Predict on validation set
            logger.info(f"\nEvaluating {model_name} on validation set...")
            y_pred = model.predict(X_val_seq)
            y_pred_proba = model.predict_proba(X_val_seq)
            
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
            
            # Check for potential issues
            self._check_model_issues(model_name, result)
            
            # Store trained model
            self.trained_models[model_name] = model
            
            return result
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_bert_model(
        self,
        model_name: str,
        model,
        X_train: np.ndarray,  # Raw text!
        y_train: np.ndarray,
        X_val: np.ndarray,    # Raw text!
        y_val: np.ndarray
    ) -> Dict:
        """
        Train BERT model (uses raw text, not sequences).
        
        Args:
            model_name: Name of the model
            model: BERT model instance
            X_train: Training texts (raw strings)
            y_train: Training labels
            X_val: Validation texts (raw strings)
            y_val: Validation labels
        
        Returns:
            Dictionary with results
        """
        print_section_header(f"TRAINING {model_name}")
        
        logger.info("[WARNING] BERT training is SLOW (~30 min on CPU, ~5 min on GPU)")
        logger.info("Consider reducing epochs or batch size if needed")
        
        # Build model
        logger.info(f"Building {model_name} architecture...")
        logger.info("(This will download pretrained weights if not cached)")
        
        try:
            model.build_model()
        except Exception as e:
            logger.error(f"Failed to build BERT model: {e}")
            logger.error("Make sure transformers and torch are installed:")
            logger.error("  pip install transformers==4.30.0 torch==2.0.1")
            return None
        
        # Get model info
        info = model.get_model_info()
        logger.info(f"\nModel Configuration:")
        logger.info(f"  Model: {info['model_name']}")
        logger.info(f"  Total parameters: {info['total_params']:,}")
        logger.info(f"  Vocab size: {info['vocab_size']:,}")
        logger.info(f"  Max length: {info['max_length']}")
        
        # Train model
        logger.info(f"\nTraining {model_name}...")
        logger.info(f"  Training samples: {len(X_train):,}")
        logger.info(f"  Validation samples: {len(X_val):,}")
        logger.info(f"  Batch size: {model.batch_size}")
        logger.info(f"  Epochs: {model.epochs}")
        logger.info(f"  Device: {'GPU' if self.has_gpu else 'CPU'}")
        
        try:
            # Convert numpy arrays to lists for BERT
            X_train_list = X_train.tolist()
            X_val_list = X_val.tolist()
            
            # Train
            history = model.train(
                X_train_list,
                y_train,
                X_val_list,
                y_val,
                verbose=1
            )
            
            training_time = model.training_time
            logger.info(f"\n{model_name} training completed in {format_time(training_time)}")
            
            # Show training history
            logger.info(f"\nTraining History:")
            logger.info(f"  Final train accuracy: {history['accuracy'][-1]:.4f}")
            logger.info(f"  Final train loss: {history['loss'][-1]:.4f}")
            if 'val_accuracy' in history:
                logger.info(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")
                logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
                logger.info(f"  Best val accuracy: {max(history['val_accuracy']):.4f}")
            
            # Predict on validation set
            logger.info(f"\nEvaluating {model_name} on validation set...")
            y_pred = model.predict(X_val_list)
            y_pred_proba = model.predict_proba(X_val_list)
            
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
            logger.error("\nPossible causes:")
            logger.error("  1. Version incompatibility (try: pip install transformers==4.30.0)")
            logger.error("  2. Insufficient memory (reduce batch_size in config.py)")
            logger.error("  3. Missing dependencies (ensure torch is installed)")
            import traceback
            traceback.print_exc()
            return None
    
    def _check_model_issues(self, model_name: str, result: Dict):
        """Check for common training issues and print warnings."""
        per_class = result.get('per_class', {})
        
        issues = []
        
        # Check for classes with no predictions
        for class_name, metrics in per_class.items():
            if metrics['precision'] == 0 and metrics['recall'] == 0:
                issues.append(f"  [WARNING] {class_name}: Model makes no predictions (0% precision/recall)")
        
        # Check for severe class imbalance effects
        accuracies = [metrics['f1'] for metrics in per_class.values()]
        if accuracies:
            min_f1 = min(accuracies)
            max_f1 = max(accuracies)
            if max_f1 - min_f1 > 0.5:  # Large gap between classes
                issues.append(f"  [WARNING] Large performance gap between classes (F1: {min_f1:.2f} - {max_f1:.2f})")
                issues.append(f"  Suggestion: Try adding class weights or data augmentation")
        
        if issues:
            logger.warning(f"\n[POTENTIAL ISSUES DETECTED in {model_name}]")
            for issue in issues:
                logger.warning(issue)
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """
        Train all deep learning models.
        
        Args:
            X_train: Training texts (raw strings)
            y_train: Training labels
            X_val: Validation texts (raw strings)
            y_val: Validation labels
        """
        print_section_header("TRAINING ALL DEEP LEARNING MODELS")
        
        logger.info(f"Training {len(self.models)} models:")
        for model_name in self.models.keys():
            logger.info(f"  - {model_name}")
        
        # Step 1: Prepare tokenizer
        self.prepare_tokenizer(X_train, save_tokenizer=True)
        
        # Step 2: Tokenize data for sequence models
        X_train_seq, X_val_seq, X_test_seq = self.tokenize_data(X_train, X_val, X_val)  # Note: Using val as test for now
        
        # Step 3: Train sequence models (LSTM, BiLSTM, CNN)
        sequence_models = ['LSTM', 'BiLSTM', 'CNN']
        
        for model_name in sequence_models:
            if model_name in self.models:
                result = self.train_sequence_model(
                    model_name=model_name,
                    model=self.models[model_name],
                    X_train_seq=X_train_seq,
                    y_train=y_train,
                    X_val_seq=X_val_seq,
                    y_val=y_val
                )
                
                if result:
                    self.results[model_name] = result
        
        # Step 4: Train BERT (if enabled)
        if 'BERT' in self.models:
            logger.info("\n" + "="*70)
            logger.info("Starting BERT training (this may take a while)...")
            logger.info("="*70 + "\n")
            
            result = self.train_bert_model(
                model_name='BERT',
                model=self.models['BERT'],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val
            )
            
            if result:
                self.results['BERT'] = result
            else:
                logger.warning("[SKIPPED] BERT training failed - continuing with other models")
        
        # Step 5: Print comparison
        print_section_header("DEEP LEARNING MODEL COMPARISON")
        self.evaluator.print_comparison()
    
    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test: Test texts (raw strings)
            y_test: Test labels
        """
        print_section_header("TEST SET EVALUATION")
        
        logger.info(f"Evaluating on test set: {len(X_test):,} samples")
        
        # Tokenize test data for sequence models
        if self.tokenizer is None:
            logger.error("Tokenizer not available. Cannot evaluate sequence models.")
            return
        
        logger.info("Tokenizing test data...")
        X_test_seq = self.tokenizer.texts_to_padded_sequences(X_test.tolist())
        
        test_evaluator = ModelEvaluator()
        
        # Evaluate sequence models
        sequence_models = ['LSTM', 'BiLSTM', 'CNN']
        
        for model_name in sequence_models:
            if model_name in self.trained_models:
                logger.info(f"\nEvaluating {model_name} on test set...")
                
                model = self.trained_models[model_name]
                
                # Predict
                y_pred = model.predict(X_test_seq)
                y_pred_proba = model.predict_proba(X_test_seq)
                
                # Evaluate
                result = test_evaluator.evaluate_model(
                    model_name=model_name,
                    y_true=y_test,
                    y_pred=y_pred,
                    y_pred_proba=y_pred_proba
                )
                
                test_evaluator.print_evaluation(result)
        
        # Evaluate BERT
        if 'BERT' in self.trained_models:
            logger.info(f"\nEvaluating BERT on test set...")
            
            model = self.trained_models['BERT']
            X_test_list = X_test.tolist()
            
            # Predict
            y_pred = model.predict(X_test_list)
            y_pred_proba = model.predict_proba(X_test_list)
            
            # Evaluate
            result = test_evaluator.evaluate_model(
                model_name='BERT',
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
            try:
                logger.info(f"Saving {model_name}...")
                model.save()
                logger.info(f"  [OK] {model_name} saved to {MODEL_FILES[model_name.lower()]}")
                saved_count += 1
            except Exception as e:
                logger.error(f"  [ERROR] Error saving {model_name}: {e}")
        
        logger.info(f"\nSuccessfully saved {saved_count}/{len(self.trained_models)} models")
    
    def save_metadata(self):
        """Save training metadata and results."""
        logger.info("Saving deep learning metadata...")
        
        # Get best model
        comparison_df = self.evaluator.get_comparison_df()
        
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            
            metadata = {
                'num_models_trained': len(self.trained_models),
                'models': list(self.trained_models.keys()),
                'tokenizer_info': self.tokenizer.get_statistics() if self.tokenizer else None,
                'gpu_used': self.has_gpu,
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
            save_results(metadata, 'dl_model_metadata.json')
            
            # Save comparison table
            comparison_df.to_csv(DL_COMPARISON_FILE, index=False)
            logger.info(f"Saved comparison table to {DL_COMPARISON_FILE}")
        
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

def train_deep_learning_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    use_bert: bool = False,
    save_models: bool = True
):
    """
    Main function to train all deep learning models.
    
    Args:
        X_train: Training texts (raw strings)
        y_train: Training labels
        X_val: Validation texts (raw strings)
        y_val: Validation labels
        X_test: Test texts (raw strings, optional)
        y_test: Test labels (optional)
        use_bert: Whether to train BERT model
        save_models: Whether to save models
    
    Returns:
        Trained DeepLearningTrainer instance
    """
    print_section_header("DEEP LEARNING TRAINING PIPELINE")
    
    # Check TensorFlow
    if not HAS_TENSORFLOW:
        logger.error("TensorFlow not installed!")
        logger.error("Install with: pip install tensorflow==2.13.0")
        sys.exit(1)
    
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check BERT availability
    if use_bert and not HAS_BERT:
        logger.warning("BERT requested but not available!")
        logger.warning("Install with: pip install transformers==4.30.0 torch==2.0.1")
        use_bert = False
    
    # Initialize trainer
    trainer = DeepLearningTrainer(use_bert=use_bert)
    
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
    logger.info(f"Best Deep Learning Model: {best_name}")
    
    # Compare with traditional ML
    try:
        import json
        metadata_path = RESULTS_DIR / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                trad_metadata = json.load(f)
            
            best_trad_acc = trad_metadata['best_model']['accuracy']
            
            comparison_df = trainer.evaluator.get_comparison_df()
            if not comparison_df.empty:
                best_dl_acc = comparison_df.iloc[0]['Accuracy']
                
                logger.info(f"\nPerformance Comparison:")
                logger.info(f"  Best Traditional ML: {best_trad_acc:.4f} ({trad_metadata['best_model']['name']})")
                logger.info(f"  Best Deep Learning:   {best_dl_acc:.4f} ({best_name})")
                logger.info(f"  Improvement:          {(best_dl_acc - best_trad_acc):.4f} ({(best_dl_acc - best_trad_acc)*100:+.2f}%)")
    except Exception as e:
        logger.warning(f"Could not compare with traditional ML: {e}")
    
    logger.info(f"\nAll models saved to: {MODEL_FILES['lstm'].parent}")
    logger.info(f"Tokenizer saved to: {MODEL_FILES['tokenizer']}")
    
    return trainer

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("DEEP LEARNING TRAINER TEST")
    print("=" * 80)
    
    print("\nThis module requires actual data to train.")
    print("Use from main_train.py or run:")
    print()
    print("  from data_handler import load_and_split_data")
    print("  from models.deep_learning_trainer import train_deep_learning_models")
    print()
    print("  # Load data")
    print("  X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()")
    print()
    print("  # Train models (without BERT)")
    print("  trainer = train_deep_learning_models(")
    print("      X_train, y_train,")
    print("      X_val, y_val,")
    print("      X_test, y_test,")
    print("      use_bert=False,")
    print("      save_models=True")
    print("  )")
    print()
    print("  # Or include BERT (slower)")
    print("  trainer = train_deep_learning_models(")
    print("      X_train, y_train,")
    print("      X_val, y_val,")
    print("      X_test, y_test,")
    print("      use_bert=True,  # Takes ~30 min on CPU")
    print("      save_models=True")
    print("  )")
    
    print("\n" + "=" * 80)