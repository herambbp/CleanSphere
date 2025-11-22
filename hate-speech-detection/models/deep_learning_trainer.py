"""
Deep Learning Trainer for Hate Speech Detection - PyTorch Implementation
Trains LSTM, BiLSTM, CNN models with GPU acceleration

PHASE 5: Deep Learning Models with PyTorch + CUDA

Features:
- GPU detection and utilization
- Automatic CUDA optimization
- Progress tracking and early stopping
- Model comparison and evaluation
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Check PyTorch availability
try:
    import torch
    HAS_PYTORCH = True
    print(f"[OK] PyTorch {torch.__version__} detected")
    if torch.cuda.is_available():
        print(f"[OK] CUDA {torch.version.cuda} available")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA not available, will use CPU")
except ImportError:
    HAS_PYTORCH = False
    print("[ERROR] PyTorch not installed!")

from config import (
    LSTM_CONFIG, BILSTM_CONFIG, CNN_CONFIG,
    MODEL_FILES, DL_COMPARISON_FILE, RESULTS_DIR,
    USE_GPU
)
from utils import (
    logger, print_section_header, print_subsection_header,
    ModelEvaluator, format_time, save_results
)

# Import PyTorch deep learning models
if HAS_PYTORCH:
    from models.deep_learning.text_tokenizer import TextTokenizer
    from models.deep_learning.lstm_model import LSTMModel
    from models.deep_learning.bilstm_model import BiLSTMModel
    from models.deep_learning.cnn_model import CNNModel
else:
    logger.error("PyTorch not installed! Cannot use deep learning models.")
    logger.error("Install with: pip install torch")

# ==================== GPU CONFIGURATION ====================

def configure_gpu():
    """
    Configure GPU settings for PyTorch with detailed reporting.
    """
    if not HAS_PYTORCH:
        return False
    
    print_section_header("GPU CONFIGURATION")
    
    if torch.cuda.is_available():
        try:
            # Get GPU info
            gpu_count = torch.cuda.device_count()
            logger.info(f"[GPU DETECTED] {gpu_count} device(s) available")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name}")
                logger.info(f"    Total Memory: {gpu_memory:.2f} GB")
            
            # Test GPU computation
            logger.info("\nTesting GPU computation...")
            device = torch.device('cuda:0')
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            logger.info("[SUCCESS] GPU computation test passed")
            logger.info("Training will use GPU (5-10x faster than CPU)")
            
            # Print CUDA optimizations
            logger.info("\nCUDA Optimizations:")
            logger.info(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
            logger.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            
            # Enable cuDNN benchmark for better performance
            torch.backends.cudnn.benchmark = True
            
            return True
            
        except Exception as e:
            logger.warning(f"[WARNING] GPU test failed: {e}")
            logger.warning("Falling back to CPU")
            return False
    else:
        logger.info("[NO GPU] Training will use CPU")
        logger.info("Expected training times on CPU:")
        logger.info("  - LSTM: ~2-3 minutes")
        logger.info("  - BiLSTM: ~4-6 minutes")
        logger.info("  - CNN: ~1-2 minutes")
        logger.info("\nTIP: Install CUDA-enabled PyTorch for GPU acceleration:")
        logger.info("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return False

# ==================== DEEP LEARNING TRAINER ====================

class DeepLearningTrainer:
    """
    Train and evaluate PyTorch deep learning models for hate speech detection.
    
    Models:
    - LSTM: Long Short-Term Memory
    - BiLSTM: Bidirectional LSTM
    - CNN: Convolutional Neural Network
    """
    
    def __init__(self):
        """Initialize Deep Learning Trainer."""
        if not HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for deep learning models.\n"
                "Install with: pip install torch"
            )
        
        logger.info("Initializing PyTorch Deep Learning Trainer...")
        
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
        logger.info("Initializing PyTorch deep learning models...")
        
        # LSTM
        self.models['LSTM'] = LSTMModel(config=LSTM_CONFIG)
        
        # BiLSTM
        self.models['BiLSTM'] = BiLSTMModel(config=BILSTM_CONFIG)
        
        # CNN
        self.models['CNN'] = CNNModel(config=CNN_CONFIG)
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        logger.info(f"Framework: PyTorch {torch.__version__}")
        logger.info(f"Device: {'CUDA' if self.has_gpu else 'CPU'}")
    
    def prepare_tokenizer(
        self,
        X_train: np.ndarray,
        save_tokenizer: bool = True
    ) -> TextTokenizer:
        """Prepare and fit tokenizer on training data."""
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
        """Tokenize all datasets."""
        print_section_header("DATA TOKENIZATION")
        
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be prepared before tokenizing data")
        
        logger.info("Tokenizing datasets...")
        
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
    
    def load_phase3_embeddings(self) -> Optional[np.ndarray]:
        """Load Phase 3 embeddings for CNN."""
        if self.tokenizer is None:
            return None
        
        try:
            # Import adapter
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models.embedding_adaptor import get_averaged_embeddings
            
            print("[INFO] Loading Phase 3 Word2Vec + FastText embeddings...")
            embeddings = get_averaged_embeddings(self.tokenizer, embedding_dim=100)
            
            if embeddings is not None:
                print(f"[SUCCESS] Loaded embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"[WARNING] Could not load Phase 3 embeddings: {e}")
            return None
    
    def train_model(
    self,
    model_name: str,
    model,
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray
    ) -> Dict:
        """Train a PyTorch model."""
        print_section_header(f"TRAINING {model_name}")
        
        # Load embeddings for CNN ONLY
        if model_name == 'CNN':
            embeddings = self.load_phase3_embeddings()
            
            if embeddings is not None:
                # Rebuild CNN model with embeddings
                logger.info("[CNN] Rebuilding with Phase 3 embeddings...")
                from models.deep_learning.cnn_model import CNNModel
                model = CNNModel(config=CNN_CONFIG, pretrained_embeddings=embeddings)
                logger.info("[CNN] Model will use your trained Word2Vec + FastText embeddings")
        
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
        logger.info(f"  Device: {info['device']}")
        
        # Train model
        logger.info(f"\nTraining {model_name}...")
        logger.info(f"  Training samples: {len(X_train_seq):,}")
        logger.info(f"  Validation samples: {len(X_val_seq):,}")
        logger.info(f"  Batch size: {model.batch_size}")
        logger.info(f"  Epochs: {model.epochs}")
        logger.info(f"  Device: {'GPU (CUDA)' if self.has_gpu else 'CPU'}")
        
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
            if 'val_accuracy' in history and history['val_accuracy']:
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
        """Train all deep learning models."""
        print_section_header("TRAINING ALL DEEP LEARNING MODELS (PyTorch)")
        
        logger.info(f"Training {len(self.models)} models:")
        for model_name in self.models.keys():
            logger.info(f"  - {model_name}")
        
        # Step 1: Prepare tokenizer
        self.prepare_tokenizer(X_train, save_tokenizer=True)
        
        # Step 2: Tokenize data
        X_train_seq, X_val_seq, X_test_seq = self.tokenize_data(X_train, X_val, X_val)
        
        # Step 3: Train all models
        for model_name in ['CNN']:
            if model_name in self.models:
                result = self.train_model(
                    model_name=model_name,
                    model=self.models[model_name],
                    X_train_seq=X_train_seq,
                    y_train=y_train,
                    X_val_seq=X_val_seq,
                    y_val=y_val
                )
                
                if result:
                    self.results[model_name] = result
        
        # Step 4: Print comparison
        print_section_header("DEEP LEARNING MODEL COMPARISON")
        self.evaluator.print_comparison()
    
    def evaluate_on_test(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """Evaluate all trained models on test set."""
        print_section_header("TEST SET EVALUATION")
        
        logger.info(f"Evaluating on test set: {len(X_test):,} samples")
        
        # Tokenize test data
        if self.tokenizer is None:
            logger.error("Tokenizer not available. Cannot evaluate.")
            return
        
        logger.info("Tokenizing test data...")
        X_test_seq = self.tokenizer.texts_to_padded_sequences(X_test.tolist())
        
        test_evaluator = ModelEvaluator()
        
        # Evaluate all models
        for model_name in ['LSTM', 'BiLSTM', 'CNN']:
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
        
        # Print comparison
        print_section_header("TEST SET MODEL COMPARISON")
        test_evaluator.print_comparison()
        
        return test_evaluator
    
    def save_all_models(self):
        """Save all trained models to disk."""
        print_section_header("SAVING MODELS")
        
        logger.info(f"Saving {len(self.trained_models)} PyTorch models...")
        
        saved_count = 0
        for model_name, model in self.trained_models.items():
            try:
                logger.info(f"Saving {model_name}...")
                model.save()
                filepath = str(MODEL_FILES[model_name.lower()]).replace('.keras', '.pt')
                logger.info(f"  [OK] {model_name} saved to {filepath}")
                saved_count += 1
            except Exception as e:
                logger.error(f"  [ERROR] Error saving {model_name}: {e}")
        
        logger.info(f"\nSuccessfully saved {saved_count}/{len(self.trained_models)} models")
    
    def save_metadata(self):
        """Save training metadata and results."""
        logger.info("Saving deep learning metadata...")
        
        comparison_df = self.evaluator.get_comparison_df()
        
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            
            metadata = {
                'framework': 'PyTorch',
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
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
    
    def get_best_model_name(self):
        """Get the name of the best performing model."""
        best_name, _ = self.get_best_model()
        return best_name
    
    def get_best_metrics(self):
        """Get metrics for the best model."""
        best_name, best_model = self.get_best_model()
        if best_name and best_name in self.results:
            return self.results[best_name]['metrics']
        return {}

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
    Main function to train all PyTorch deep learning models.
    
    Args:
        X_train: Training texts (raw strings)
        y_train: Training labels
        X_val: Validation texts (raw strings)
        y_val: Validation labels
        X_test: Test texts (raw strings, optional)
        y_test: Test labels (optional)
        use_bert: Ignored (not implemented in PyTorch version)
        save_models: Whether to save models
    
    Returns:
        Trained DeepLearningTrainer instance
    """
    print_section_header("PyTorch DEEP LEARNING TRAINING PIPELINE")
    
    # Check PyTorch
    if not HAS_PYTORCH:
        logger.error("PyTorch not installed!")
        logger.error("Install with: pip install torch")
        logger.error("For CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize trainer
    trainer = DeepLearningTrainer()
    
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
    
    model_dir = MODEL_FILES['lstm'].parent
    logger.info(f"\nAll models saved to: {model_dir}")
    logger.info(f"Tokenizer saved to: {MODEL_FILES['tokenizer']}")
    logger.info(f"Framework: PyTorch {torch.__version__}")
    logger.info(f"Device: {'CUDA' if trainer.has_gpu else 'CPU'}")
    
    return trainer

# Export
HAS_TENSORFLOW = HAS_PYTORCH  # For compatibility

if __name__ == "__main__":
    print("=" * 80)
    print("PyTorch DEEP LEARNING TRAINER")
    print("=" * 80)
    
    print("\nThis module requires actual data to train.")
    print("Use from main_train_enhanced.py with --phase5 flag")
    print()
    print("Example:")
    print("  python main_train_enhanced.py --phase5")
    print()
    print("=" * 80)