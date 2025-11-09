"""
Specialized Transformer Models for Hate Speech Detection
=========================================================

Priority 1: Specialized transformers that outperform standard BERT
- HateBERT: Pre-trained on hate speech data (+2-4% improvement)
- DeBERTa-v3-large: State-of-the-art with disentangled attention
- RoBERTa-large: Strong baseline
- BERT-large: Standard reference

All models optimized for A5000 GPU (24GB VRAM)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer,
        EarlyStoppingCallback,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup
    )
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from config import NUM_CLASSES, PROJECT_ROOT, MODELS_DIR
    from utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    NUM_CLASSES = 3
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / 'saved_models' / 'advanced'

# ==================== MODEL CONFIGURATIONS ====================

SPECIALIZED_MODELS = {
    'hatebert': {
        'name': 'GroNLP/hateBERT',
        'description': 'BERT pre-trained on Reddit hate speech corpus',
        'params': '110M',
        'max_length': 256,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'epochs': 5,
        'expected_gain': '+2-4%'
    },
    'deberta-v3-base': {
        'name': 'microsoft/deberta-v3-base',
        'description': 'DeBERTa v3 with disentangled attention',
        'params': '184M',
        'max_length': 256,
        'batch_size': 24,
        'learning_rate': 1e-5,
        'epochs': 4,
        'expected_gain': '+3-5%'
    },
    'deberta-v3-large': {
        'name': 'microsoft/deberta-v3-large',
        'description': 'Large DeBERTa v3 (best single model)',
        'params': '434M',
        'max_length': 256,
        'batch_size': 16,
        'learning_rate': 8e-6,
        'epochs': 4,
        'expected_gain': '+4-6%'
    },
    'roberta-large': {
        'name': 'roberta-large',
        'description': 'RoBERTa Large - strong baseline',
        'params': '355M',
        'max_length': 256,
        'batch_size': 20,
        'learning_rate': 1e-5,
        'epochs': 4,
        'expected_gain': '+2-3%'
    },
    'bert-large': {
        'name': 'bert-large-uncased',
        'description': 'BERT Large - reference model',
        'params': '340M',
        'max_length': 256,
        'batch_size': 24,
        'learning_rate': 2e-5,
        'epochs': 4,
        'expected_gain': '+1-2%'
    }
}

# ==================== SPECIALIZED TRANSFORMER MODEL ====================

class SpecializedTransformerModel:
    """
    Wrapper for specialized transformer models optimized for hate speech detection.
    
    Features:
    - Easy selection of specialized models (HateBERT, DeBERTa, etc.)
    - Optimized hyperparameters for each model
    - A5000 GPU optimization (24GB VRAM)
    - Mixed precision training
    - Gradient accumulation for larger effective batch sizes
    """
    
    def __init__(
        self,
        model_type: str = 'hatebert',
        num_classes: int = NUM_CLASSES,
        device: str = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize specialized transformer model.
        
        Args:
            model_type: One of 'hatebert', 'deberta-v3-base', 'deberta-v3-large', 
                       'roberta-large', 'bert-large'
            num_classes: Number of output classes
            device: Device to use (auto-detect if None)
            use_mixed_precision: Use FP16 training
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers library required. Install with:\n"
                "pip install transformers torch"
            )
        
        # Validate model type
        if model_type not in SPECIALIZED_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(SPECIALIZED_MODELS.keys())}"
            )
        
        self.model_type = model_type
        self.config = SPECIALIZED_MODELS[model_type]
        self.num_classes = num_classes
        self.use_mixed_precision = use_mixed_precision
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Training state
        self.history = None
        self.training_time = None
        self.best_metrics = None
        
        logger.info(f"Initialized {self.config['description']}")
        logger.info(f"  Model: {self.config['name']}")
        logger.info(f"  Parameters: {self.config['params']}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed Precision: {use_mixed_precision}")
    
    def load_model(self):
        """Load pre-trained model and tokenizer."""
        model_name = self.config['name']
        
        logger.info(f"Loading {model_name}...")
        logger.info("This may take a few minutes for first-time download...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"Model loaded successfully")
            logger.info(f"  Tokenizer vocab size: {len(self.tokenizer):,}")
            logger.info(f"  Model parameters: {self.count_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: np.ndarray = None
    ):
        """
        Prepare dataset for training/inference.
        
        Args:
            texts: List of text strings
            labels: Optional labels
        
        Returns:
            HuggingFace Dataset
        """
        from torch.utils.data import Dataset
        
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                item = {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten()
                }
                
                if self.labels is not None:
                    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                
                return item
        
        return TextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.config['max_length']
        )
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0)
        }
    
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str],
        y_val: np.ndarray,
        output_dir: Path = None,
        num_epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        save_best_only: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            output_dir: Directory to save model
            num_epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
            learning_rate: Learning rate (uses config default if None)
            warmup_ratio: Warmup ratio for learning rate scheduler
            weight_decay: Weight decay for regularization
            gradient_accumulation_steps: Steps to accumulate gradients
            save_best_only: Only save best checkpoint
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.load_model()
        
        # Use config defaults if not specified
        num_epochs = num_epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']
        learning_rate = learning_rate or self.config['learning_rate']
        
        # Setup output directory
        if output_dir is None:
            output_dir = MODELS_DIR / self.model_type
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nTraining {self.model_type}")
        logger.info(f"  Training samples: {len(X_train):,}")
        logger.info(f"  Validation samples: {len(X_val):,}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Expected improvement: {self.config['expected_gain']}")
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        train_dataset = self.prepare_dataset(X_train, y_train)
        val_dataset = self.prepare_dataset(X_val, y_val)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            logging_dir=str(output_dir / 'logs'),
            logging_steps=50,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
            greater_is_better=True,
            save_total_limit=2 if save_best_only else None,
            fp16=self.use_mixed_precision and self.device.type == 'cuda',
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to='none'
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        logger.info("Starting training...")
        start_time = time.time()
        
        train_result = self.trainer.train()
        
        self.training_time = time.time() - start_time
        
        # Get metrics
        metrics = train_result.metrics
        self.best_metrics = self.trainer.evaluate()
        
        logger.info(f"\nTraining completed in {self.training_time/60:.2f} minutes")
        logger.info(f"  Best validation accuracy: {self.best_metrics['eval_accuracy']:.4f}")
        logger.info(f"  Best validation F1 (macro): {self.best_metrics['eval_f1_macro']:.4f}")
        
        # Save training history
        self.history = {
            'train_metrics': metrics,
            'eval_metrics': self.best_metrics,
            'training_time': self.training_time
        }
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save config
        config_path = output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'model_name': self.config['name'],
                'num_classes': self.num_classes,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'training_time': self.training_time,
                'best_metrics': {k: float(v) for k, v in self.best_metrics.items()}
            }, f, indent=2)
        
        return self.history
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            texts: List of texts
        
        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(texts)
        return np.argmax(probas, axis=1)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of texts
        
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() or train() first.")
        
        self.model.eval()
        
        # Prepare dataset
        dataset = self.prepare_dataset(texts)
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False
        )
        
        # Predict
        all_probas = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probas = torch.softmax(logits, dim=1)
                all_probas.append(probas.cpu().numpy())
        
        return np.vstack(all_probas)
    
    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate on test set.
        
        Args:
            X_test: Test texts
            y_test: Test labels
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {self.model_type} on test set...")
        
        # Predict
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            classification_report, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    @classmethod
    def load_trained(cls, model_path: Path, device: str = None):
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model directory
            device: Device to load model on
        
        Returns:
            Loaded SpecializedTransformerModel instance
        """
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / 'training_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_type = config['model_type']
        else:
            # Try to infer from path
            model_type = model_path.name
        
        # Create instance
        instance = cls(model_type=model_type, device=device)
        
        # Load model and tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        instance.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        instance.model.to(instance.device)
        instance.model.eval()
        
        logger.info(f"Loaded trained {model_type} from {model_path}")
        
        return instance

# ==================== CONVENIENCE FUNCTIONS ====================

def list_available_models() -> Dict:
    """List all available specialized models."""
    return SPECIALIZED_MODELS.copy()

def get_recommended_model(vram_gb: float = 24.0) -> str:
    """
    Get recommended model based on available VRAM.
    
    Args:
        vram_gb: Available VRAM in GB
    
    Returns:
        Recommended model type
    """
    if vram_gb >= 24:
        return 'deberta-v3-large'  # Best single model
    elif vram_gb >= 16:
        return 'hatebert'  # Best for hate speech
    elif vram_gb >= 12:
        return 'deberta-v3-base'
    else:
        return 'bert-large'

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("SPECIALIZED TRANSFORMER MODELS TEST")
    print("=" * 80)
    
    # List available models
    print("\nAvailable Models:")
    print("-" * 80)
    for model_type, config in SPECIALIZED_MODELS.items():
        print(f"\n{model_type}:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Parameters: {config['params']}")
        print(f"  Expected Gain: {config['expected_gain']}")
    
    # Get recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    vram_options = [8, 12, 16, 24, 40]
    for vram in vram_options:
        rec = get_recommended_model(vram)
        print(f"  {vram}GB VRAM: {rec} ({SPECIALIZED_MODELS[rec]['description']})")
    
    print("\n" + "=" * 80)
    print("Test complete. To train a model:")
    print("=" * 80)
    print("""
from models.advanced.specialized_transformers import SpecializedTransformerModel

# Initialize model
model = SpecializedTransformerModel(model_type='hatebert')

# Train
model.train(
    X_train=train_texts,
    y_train=train_labels,
    X_val=val_texts,
    y_val=val_labels
)

# Evaluate
metrics = model.evaluate(test_texts, test_labels)
""")