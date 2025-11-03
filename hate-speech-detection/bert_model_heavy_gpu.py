"""
Enhanced BERT Model for Hate Speech Classification - Heavy GPU Version
========================================================================

NOW WITH HEAVY GPU - WE CAN DO MORE:
- Use BERT-Large instead of BERT-Base (3x bigger, better accuracy)
- Much larger batch sizes (64-128 instead of 16)
- More training epochs (10+ instead of 3-4)
- Train multiple models (RoBERTa, DistilBERT, BERT variants)
- Ensemble methods for best accuracy
- Advanced fine-tuning strategies
- Hyperparameter optimization
- Cross-validation for robust evaluation

Architecture Options:
- BERT-Base-Uncased (110M params) - Fast baseline
- BERT-Large-Uncased (340M params) - Better accuracy
- RoBERTa-Base (125M params) - Improved BERT
- RoBERTa-Large (355M params) - Best performance
- DistilBERT (66M params) - Fast inference
- ALBERT (12M-235M params) - Parameter efficient
"""

import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import warnings
import json

warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Transformers imports
try:
    from transformers import (
        BertTokenizer, BertForSequenceClassification,
        RobertaTokenizer, RobertaForSequenceClassification,
        DistilBertTokenizer, DistilBertForSequenceClassification,
        AlbertTokenizer, AlbertForSequenceClassification,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

try:
    from config import NUM_CLASSES, MODEL_FILES, CLASS_LABELS
except ImportError:
    NUM_CLASSES = 3
    MODEL_FILES = {'bert': 'saved_models/bert_model'}
    CLASS_LABELS = {0: 'Hate', 1: 'Offensive', 2: 'Neither'}


# ==================== HEAVY GPU CONFIGURATION ====================

class HeavyGPUBERTConfig:
    """
    Configuration for BERT training with heavy GPU capabilities.
    NO MORE LIMITATIONS - use full GPU power!
    """

    # Model Selection - NOW WE CAN USE LARGER MODELS
    MODEL_OPTIONS = {
        'bert-base': 'bert-base-uncased',           # 110M params - Fast
        'bert-large': 'bert-large-uncased',         # 340M params - Better
        'roberta-base': 'roberta-base',             # 125M params - Improved BERT
        'roberta-large': 'roberta-large',           # 355M params - Best
        'distilbert': 'distilbert-base-uncased',    # 66M params - Fast inference
        'albert-base': 'albert-base-v2',            # 12M params - Efficient
        'albert-large': 'albert-large-v2',          # 18M params - Efficient
        'albert-xlarge': 'albert-xlarge-v2',        # 60M params - Efficient
        'albert-xxlarge': 'albert-xxlarge-v2',      # 235M params - Very efficient
    }

    def __init__(
        self,
        model_name: str = 'bert-large',  # Now using LARGE by default!
        max_length: int = 256,  # Increased from 128
        batch_size: int = 64,  # Increased from 16-32
        epochs: int = 10,  # Increased from 3-4
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,  # No need with large batch
        scheduler_type: str = 'cosine',  # Better than linear
        early_stopping: bool = True,
        patience: int = 3,
        dropout_rate: float = 0.1,
        use_class_weights: bool = True,
        save_best_only: bool = True,
        evaluate_during_training: bool = True,
        eval_steps: int = 100,  # More frequent evaluation
        logging_steps: int = 50,
        save_steps: int = 500,
        **kwargs
    ):
        """
        Initialize heavy GPU BERT configuration.

        Args:
            model_name: Model identifier (see MODEL_OPTIONS)
            max_length: Maximum sequence length (increased for GPU)
            batch_size: Batch size (large for GPU)
            epochs: Number of training epochs (more with GPU)
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio for scheduler
            weight_decay: Weight decay for regularization
            max_grad_norm: Max gradient norm for clipping
            use_mixed_precision: Use FP16 training
            gradient_accumulation_steps: Steps to accumulate gradients
            scheduler_type: Learning rate scheduler type
            early_stopping: Enable early stopping
            patience: Early stopping patience
            dropout_rate: Dropout rate
            use_class_weights: Use class weights for imbalanced data
            save_best_only: Only save best model
            evaluate_during_training: Evaluate during training
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            save_steps: Steps between saving checkpoints
        """
        # Get full model name
        if model_name in self.MODEL_OPTIONS:
            self.model_key = model_name
            self.model_name = self.MODEL_OPTIONS[model_name]
        else:
            self.model_key = 'custom'
            self.model_name = model_name

        # Determine model type from name
        if 'roberta' in self.model_name.lower():
            self.model_type = 'roberta'
        elif 'distilbert' in self.model_name.lower():
            self.model_type = 'distilbert'
        elif 'albert' in self.model_name.lower():
            self.model_type = 'albert'
        else:
            self.model_type = 'bert'

        # Store configuration
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scheduler_type = scheduler_type
        self.early_stopping = early_stopping
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.use_class_weights = use_class_weights
        self.save_best_only = save_best_only
        self.evaluate_during_training = evaluate_during_training
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Log configuration
        self._log_config()

    def _log_config(self):
        """Log configuration details."""
        logger.info("\n" + "="*80)
        logger.info("HEAVY GPU BERT CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Model: {self.model_name} ({self.model_key})")
        logger.info(f"Model Type: {self.model_type}")
        logger.info(f"Max Length: {self.max_length}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Learning Rate: {self.learning_rate}")
        logger.info(f"Scheduler: {self.scheduler_type}")
        logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        logger.info(
            f"Early Stopping: {self.early_stopping} (patience={self.patience})")
        logger.info("="*80 + "\n")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model_key': self.model_key,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'warmup_ratio': self.warmup_ratio,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'use_mixed_precision': self.use_mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'scheduler_type': self.scheduler_type,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'dropout_rate': self.dropout_rate,
            'use_class_weights': self.use_class_weights,
        }


# ==================== HEAVY GPU BERT MODEL ====================

class HeavyGPUBERTModel:
    """
    Enhanced BERT model that takes full advantage of heavy GPU capabilities.

    Features:
    - Larger models (BERT-Large, RoBERTa-Large)
    - Bigger batch sizes
    - More training epochs
    - Advanced training strategies
    - Better monitoring and logging
    """

    def __init__(
        self,
        config: Union[HeavyGPUBERTConfig, Dict, str] = None,
        num_classes: int = NUM_CLASSES
    ):
        """
        Initialize heavy GPU BERT model.

        Args:
            config: Configuration (HeavyGPUBERTConfig, dict, or model_name string)
            num_classes: Number of output classes
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch required. Install with:\n"
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            )

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers required. Install with:\n"
                "pip install transformers"
            )

        # Parse configuration
        if isinstance(config, str):
            # If string, treat as model name
            self.config = HeavyGPUBERTConfig(model_name=config)
        elif isinstance(config, dict):
            # If dict, convert to config object
            self.config = HeavyGPUBERTConfig(**config)
        elif config is None:
            # Use default
            self.config = HeavyGPUBERTConfig()
        else:
            self.config = config

        self.num_classes = num_classes
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        # Training state
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.training_time = None
        self.global_step = 0

        # Setup device
        self.setup_device()

        # Initialize tokenizer
        self._initialize_tokenizer()

    def setup_device(self):
        """Setup device - check GPU capabilities."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_count = torch.cuda.device_count()

            logger.info("\n" + "="*80)
            logger.info("GPU SETUP")
            logger.info("="*80)
            logger.info(f" CUDA Available: YES")
            logger.info(f" GPU Count: {gpu_count}")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory_gb = torch.cuda.get_device_properties(
                    i).total_memory / 1e9
                logger.info(f" GPU {i}: {gpu_name}")
                logger.info(f"  Memory: {gpu_memory_gb:.1f} GB")

            logger.info(f" CUDA Version: {torch.version.cuda}")
            logger.info(f" cuDNN Version: {torch.backends.cudnn.version()}")

            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True

            # Check if we can use larger models
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 16:
                logger.info(
                    f" GPU Memory >= 16GB - Can train BERT-Large efficiently!")
            if gpu_memory >= 24:
                logger.info(
                    f" GPU Memory >= 24GB - Can train RoBERTa-Large efficiently!")
            if gpu_memory >= 40:
                logger.info(
                    f" GPU Memory >= 40GB - Can train largest models with big batches!")

            logger.info("="*80 + "\n")
        else:
            self.device = torch.device('cpu')
            logger.warning("\n NO GPU DETECTED - Training will be very slow!")
            logger.warning(
                "Install CUDA-enabled PyTorch for GPU acceleration\n")

    def _initialize_tokenizer(self):
        """Initialize tokenizer based on model type."""
        model_type = self.config.model_type
        model_name = self.config.model_name

        logger.info(f"Loading {model_type} tokenizer: {model_name}")

        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        elif model_type == 'distilbert':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        elif model_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        else:  # bert
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

        logger.info(f" Tokenizer loaded: {model_type}")
        logger.info(f"  Vocab size: {self.tokenizer.vocab_size:,}")

    def build_model(self):
        """Build BERT model architecture."""
        model_type = self.config.model_type
        model_name = self.config.model_name

        logger.info(f"\nBuilding {model_type} model: {model_name}")
        logger.info("This may take a few minutes for large models...")

        start_time = time.time()

        # Load pretrained model
        if model_type == 'roberta':
            self.model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                hidden_dropout_prob=self.config.dropout_rate,
                attention_probs_dropout_prob=self.config.dropout_rate
            )
        elif model_type == 'distilbert':
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                dropout=self.config.dropout_rate
            )
        elif model_type == 'albert':
            self.model = AlbertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                hidden_dropout_prob=self.config.dropout_rate,
                attention_probs_dropout_prob=self.config.dropout_rate
            )
        else:  # bert
            self.model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                hidden_dropout_prob=self.config.dropout_rate,
                attention_probs_dropout_prob=self.config.dropout_rate
            )

        # Move to device
        self.model = self.model.to(self.device)

        load_time = time.time() - start_time

        # Calculate model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        model_size_gb = total_params * 4 / 1e9  # FP32 size

        logger.info(f" Model built in {load_time:.1f} seconds")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size (FP32): ~{model_size_gb:.2f} GB")
        logger.info(f"  Model size (FP16): ~{model_size_gb/2:.2f} GB")

        return self.model

    def tokenize_texts(
        self,
        texts: List[str],
        max_length: int = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts."""
        max_length = max_length or self.config.max_length

        encoded = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_attention_mask=True
        )

        return encoded

    def prepare_dataloader(
        self,
        texts: List[str],
        labels: np.ndarray = None,
        batch_size: int = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Prepare PyTorch DataLoader."""
        batch_size = batch_size or self.config.batch_size

        # Tokenize
        encoded = self.tokenize_texts(texts)

        # Create dataset
        if labels is not None:
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(
                encoded['input_ids'],
                encoded['attention_mask'],
                labels_tensor
            )
        else:
            dataset = TensorDataset(
                encoded['input_ids'],
                encoded['attention_mask']
            )

        # Create dataloader with GPU optimizations
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Parallel data loading
            pin_memory=True,  # Faster GPU transfer
            drop_last=False
        )

        return dataloader

    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Prepare optimizer parameters with weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        # Create optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            eps=1e-8
        )

        # Calculate warmup steps
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        # Create scheduler
        if self.config.scheduler_type == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(
                f" Scheduler: Cosine with warmup ({num_warmup_steps} steps)")
        else:  # linear
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(
                f" Scheduler: Linear with warmup ({num_warmup_steps} steps)")

        # Setup mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
            logger.info(" Mixed Precision: Enabled (FP16)")

        logger.info(f" Optimizer: AdamW (lr={self.config.learning_rate})")

    def compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        weights_tensor = torch.tensor(
            weights, dtype=torch.float32).to(self.device)

        logger.info(f" Class weights computed: {weights}")
        return weights_tensor

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        epoch: int,
        class_weights: torch.Tensor = None
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        num_batches = len(train_dataloader)
        log_interval = max(1, num_batches // 20)  # Log 20 times per epoch

        for step, batch in enumerate(train_dataloader):
            self.global_step += 1

            # Move batch to device
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            # Forward pass with mixed precision
            if self.config.use_mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                    # Apply class weights if provided
                    if class_weights is not None:
                        logits = outputs.logits
                        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                        loss = loss_fct(
                            logits.view(-1, self.num_classes), labels.view(-1))

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard precision
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                if class_weights is not None:
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                    loss = loss_fct(
                        logits.view(-1, self.num_classes), labels.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            # Scheduler step
            self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

            # Calculate metrics
            with torch.no_grad():
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            total_loss += loss.item()

            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / (step + 1)
                accuracy = correct / total
                lr = self.scheduler.get_last_lr()[0]

                progress = (step + 1) / num_batches * 100
                logger.info(
                    f"Epoch {epoch+1} [{step+1}/{num_batches}] ({progress:.1f}%) | "
                    f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | LR: {lr:.2e}"
                )

        avg_loss = total_loss / num_batches
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate_epoch(
        self,
        val_dataloader: DataLoader,
        class_weights: torch.Tensor = None
    ) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                # Forward pass
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        
                        if class_weights is not None:
                            logits = outputs.logits
                            class_weights_fp16 = class_weights.to(logits.dtype)
                            loss_fct = nn.CrossEntropyLoss(weight=class_weights_fp16)
                            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    if class_weights is not None:
                        logits = outputs.logits
                        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                # Calculate metrics
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: List[str] = None,
        y_val: np.ndarray = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the BERT model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING HEAVY GPU BERT TRAINING")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Build model
        if self.model is None:
            self.build_model()
        
        # Prepare dataloaders
        logger.info("\nPreparing dataloaders...")
        train_dataloader = self.prepare_dataloader(X_train, y_train, shuffle=True)
        
        val_dataloader = None
        if X_val is not None and y_val is not None:
            val_dataloader = self.prepare_dataloader(X_val, y_val, shuffle=False)
            logger.info(f" Training samples: {len(X_train):,}")
            logger.info(f" Validation samples: {len(X_val):,}")
        else:
            logger.info(f" Training samples: {len(X_train):,}")
        
        logger.info(f" Batch size: {self.config.batch_size}")
        logger.info(f" Batches per epoch: {len(train_dataloader)}")
        
        # Calculate training steps
        num_training_steps = len(train_dataloader) * self.config.epochs
        logger.info(f" Total training steps: {num_training_steps:,}")
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(num_training_steps)
        
        # Compute class weights if enabled
        class_weights = None
        if self.config.use_class_weights:
            class_weights = self.compute_class_weights(y_train)
        
        # Training loop
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING FOR {self.config.epochs} EPOCHS")
        logger.info(f"{'='*80}\n")
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"EPOCH {epoch + 1}/{self.config.epochs}")
            logger.info(f"{'='*80}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_dataloader,
                epoch=epoch,
                class_weights=class_weights
            )
            
            # Store metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"\n[EPOCH {epoch+1} SUMMARY]")
            logger.info(f"  Training Loss: {train_loss:.4f}")
            logger.info(f"  Training Accuracy: {train_acc:.4f}")
            logger.info(f"  Epoch Time: {epoch_time/60:.2f} minutes")
            
            # Validation
            if val_dataloader is not None:
                val_loss, val_acc = self.evaluate_epoch(val_dataloader, class_weights)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                logger.info(f"  Validation Loss: {val_loss:.4f}")
                logger.info(f"  Validation Accuracy: {val_acc:.4f}")
                
                # Track improvement
                improved = False
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    improved = True
                    logger.info(f"   NEW BEST MODEL! (Val Acc: {val_acc:.4f})")
                else:
                    self.epochs_no_improve += 1
                    logger.info(f"  No improvement for {self.epochs_no_improve} epoch(s)")
                
                # Early stopping
                if self.config.early_stopping and self.epochs_no_improve >= self.config.patience:
                    logger.info(f"\n EARLY STOPPING at epoch {epoch+1}")
                    logger.info(f"  Best validation accuracy: {self.best_val_acc:.4f}")
                    break
            
            logger.info(f"{'='*80}")
        
        # Training complete
        self.training_time = time.time() - start_time
        
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"Total training time: {self.training_time/60:.2f} minutes")
        logger.info(f"Average time per epoch: {self.training_time/len(self.history['train_loss'])/60:.2f} minutes")
        if val_dataloader is not None:
            logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"{'='*80}\n")
        
        return self.history
    
    def predict(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(texts, batch_size)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model must be built or loaded before prediction")
        
        self.model.eval()
        
        batch_size = batch_size or self.config.batch_size
        dataloader = self.prepare_dataloader(texts, labels=None, shuffle=False, batch_size=batch_size)
        
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                probabilities = F.softmax(outputs.logits, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())
        
        return np.vstack(all_probabilities)
    
    def evaluate(
        self,
        texts: List[str],
        labels: np.ndarray,
        verbose: int = 1
    ) -> Dict:
        """Evaluate model on test data."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING MODEL ON TEST SET")
        logger.info("="*80)
        
        # Get predictions
        y_pred = self.predict(texts)
        y_pred_proba = self.predict_proba(texts)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report, confusion_matrix
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
            'probabilities': y_pred_proba,
            'confusion_matrix': confusion_matrix(labels, y_pred)
        }
        
        if verbose > 0:
            logger.info(f"\n{'='*80}")
            logger.info("TEST SET PERFORMANCE")
            logger.info(f"{'='*80}")
            logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score (macro):  {metrics['f1_macro']:.4f}")
            logger.info(f"F1 Score (weight): {metrics['f1_weighted']:.4f}")
            logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
            logger.info(f"Recall (macro):    {metrics['recall_macro']:.4f}")
            
            # Classification report
            target_names = [CLASS_LABELS.get(i, f'Class {i}') for i in range(self.num_classes)]
            report = classification_report(
                labels, y_pred,
                target_names=target_names,
                digits=4,
                zero_division=0
            )
            logger.info("\n" + report)
            
            logger.info(f"\n{'='*80}\n")
        
        return metrics
    
    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving model to {filepath}")
        
        # Save model and tokenizer
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        
        # Save configuration
        config_path = filepath / 'heavy_gpu_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save training history
        if self.history:
            history_path = filepath / 'training_history.json'
            history_serializable = {
                k: [float(x) for x in v] if isinstance(v, list) else v
                for k, v in self.history.items()
            }
            with open(history_path, 'w') as f:
                json.dump(history_serializable, f, indent=2)
        
        logger.info(f" Model saved successfully")
    
    @staticmethod
    def load(filepath: str) -> 'HeavyGPUBERTModel':
        """Load model from disk."""
        filepath = Path(filepath)
        
        logger.info(f"\nLoading model from {filepath}")
        
        # Load configuration
        config_path = filepath / 'heavy_gpu_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = HeavyGPUBERTConfig(**config_dict)
        else:
            logger.warning("No config found, using defaults")
            config = HeavyGPUBERTConfig()
        
        # Create model instance
        bert_model = HeavyGPUBERTModel(config=config)
        
        # Load model
        model_type = config.model_type
        if model_type == 'roberta':
            bert_model.model = RobertaForSequenceClassification.from_pretrained(filepath)
        elif model_type == 'distilbert':
            bert_model.model = DistilBertForSequenceClassification.from_pretrained(filepath)
        elif model_type == 'albert':
            bert_model.model = AlbertForSequenceClassification.from_pretrained(filepath)
        else:
            bert_model.model = BertForSequenceClassification.from_pretrained(filepath)
        
        bert_model.model.to(bert_model.device)
        
        # Load history
        history_path = filepath / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                bert_model.history = json.load(f)
        
        logger.info(f" Model loaded successfully")
        
        return bert_model
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            return {
                'model_type': f'Heavy GPU BERT ({self.config.model_type})',
                'model_name': self.config.model_name,
                'built': False
            }
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        info = {
            'model_type': f'Heavy GPU BERT ({self.config.model_type})',
            'model_name': self.config.model_name,
            'model_key': self.config.model_key,
            'built': True,
            'max_length': self.config.max_length,
            'batch_size': self.config.batch_size,
            'num_classes': self.num_classes,
            'vocab_size': self.tokenizer.vocab_size,
            'total_params': total_params,
            'device': str(self.device),
            'mixed_precision': self.config.use_mixed_precision,
        }
        
        if self.training_time:
            info['training_time'] = self.training_time
            info['training_time_minutes'] = self.training_time / 60
        
        if self.history and self.history['train_acc']:
            info['final_train_acc'] = self.history['train_acc'][-1]
            if self.history['val_acc']:
                info['final_val_acc'] = self.history['val_acc'][-1]
                info['best_val_acc'] = self.best_val_acc
        
        return info


# ==================== CONVENIENCE FUNCTIONS ====================

def create_bert_model(
    model_name: str = 'bert-large',
    batch_size: int = 64,
    epochs: int = 10,
    **kwargs
) -> HeavyGPUBERTModel:
    """
    Create and build BERT model with heavy GPU configuration.
    
    Args:
        model_name: Model name (bert-base, bert-large, roberta-base, roberta-large, etc.)
        batch_size: Batch size
        epochs: Number of epochs
        **kwargs: Additional configuration parameters
    
    Returns:
        Built HeavyGPUBERTModel instance
    """
    config = HeavyGPUBERTConfig(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        **kwargs
    )
    
    model = HeavyGPUBERTModel(config=config)
    model.build_model()
    
    return model


# ==================== COMPATIBILITY ALIASES ====================

# Make it compatible with existing code
BERTModel = HeavyGPUBERTModel
BERTModelPyTorch = HeavyGPUBERTModel


# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*80)
    print("HEAVY GPU BERT MODEL TEST")
    print("="*80)
    
    # Check dependencies
    if not HAS_TORCH:
        print("\n PyTorch not installed")
        print("Install: pip install torch torchvision torchaudio")
        exit(1)
    
    if not HAS_TRANSFORMERS:
        print("\n Transformers not installed")
        print("Install: pip install transformers")
        exit(1)
    
    print(f"\n PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f" CUDA version: {torch.version.cuda}")
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" GPU Memory: {gpu_memory:.1f} GB")
    
    # Test model creation
    print("\n" + "="*80)
    print("TESTING MODEL CREATION")
    print("="*80)
    
    print("\n1. Testing BERT-Large...")
    model = create_bert_model(model_name='bert-large', batch_size=32, epochs=2)
    
    print("\n2. Testing tokenization...")
    test_texts = [
        "I hate you",
        "Good morning everyone",
        "You are an idiot"
    ]
    
    encoded = model.tokenize_texts(test_texts)
    print(f" Tokenization works")
    print(f"  Input shape: {encoded['input_ids'].shape}")
    
    print("\n3. Testing prediction...")
    predictions = model.predict(test_texts)
    print(f" Prediction works: {predictions}")
    
    probabilities = model.predict_proba(test_texts)
    print(f" Predict proba works: {probabilities.shape}")
    
    # Show model info
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    info = model.get_model_info()
    for key, value in info.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"{key:25s}: {value:.4f}" if value < 1 else f"{key:25s}: {value:,.0f}")
            else:
                print(f"{key:25s}: {value:,}")
        else:
            print(f"{key:25s}: {value}")
    
    print("\n" + "="*80)
    print("HEAVY GPU BERT MODEL TEST COMPLETE ")
    print("="*80)
    print("\nReady for training with heavy GPU!")
    if torch.cuda.is_available():
        print(f"Your {torch.cuda.get_device_name(0)} is ready to train large models!")