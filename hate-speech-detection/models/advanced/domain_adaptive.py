"""
Domain-Adaptive Pre-training for Hate Speech Detection
=======================================================

Continue pre-training transformer models on hate speech corpus before fine-tuning.
This teaches models hate speech-specific vocabulary and patterns.

Expected gain: +1-3% over direct fine-tuning

Approach:
1. Masked Language Modeling (MLM) on unlabeled hate speech data
2. Learn domain-specific representations
3. Fine-tune on labeled data

Works with any transformer model (BERT, RoBERTa, DeBERTa, etc.)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import (
        AutoTokenizer, AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        TrainingArguments, Trainer
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from config import PROJECT_ROOT, MODELS_DIR
    from utils import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / 'saved_models' / 'advanced'

# ==================== DOMAIN ADAPTIVE PRETRAINER ====================

class DomainAdaptivePretrainer:
    """
    Domain-adaptive pre-training for hate speech detection.
    
    Process:
    1. Load pre-trained transformer (BERT, RoBERTa, etc.)
    2. Continue pre-training with MLM on hate speech corpus
    3. Save adapted model for fine-tuning
    
    This teaches the model:
    - Hate speech vocabulary and slang
    - Context-specific word meanings
    - Domain-specific patterns
    """
    
    def __init__(
        self,
        base_model: str = 'roberta-base',
        mlm_probability: float = 0.15,
        device: str = None
    ):
        """
        Initialize domain-adaptive pre-trainer.
        
        Args:
            base_model: Base transformer model name
            mlm_probability: Probability of masking tokens (default 0.15)
            device: Device to use
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers library required. Install with:\n"
                "pip install transformers torch"
            )
        
        self.base_model = base_model
        self.mlm_probability = mlm_probability
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        
        logger.info(f"Initialized Domain-Adaptive Pre-trainer")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  MLM probability: {mlm_probability}")
        logger.info(f"  Device: {self.device}")
    
    def load_model(self):
        """Load base model and tokenizer for MLM."""
        logger.info(f"Loading {self.base_model} for masked language modeling...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                use_fast=True
            )
            
            # Load model for MLM
            self.model = AutoModelForMaskedLM.from_pretrained(self.base_model)
            self.model.to(self.device)
            
            # Create data collator for MLM
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability
            )
            
            logger.info(f"Model loaded successfully")
            logger.info(f"  Parameters: {self.count_parameters():,}")
            
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
        max_length: int = 256
    ):
        """
        Prepare dataset for MLM pre-training.
        
        Args:
            texts: List of unlabeled texts
            max_length: Maximum sequence length
        
        Returns:
            Tokenized dataset
        """
        from torch.utils.data import Dataset
        
        class MLMDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
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
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten()
                }
        
        return MLMDataset(texts, self.tokenizer, max_length)
    
    def pretrain(
        self,
        texts: List[str],
        output_dir: Path = None,
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
        max_length: int = 256,
        save_steps: int = 500,
        logging_steps: int = 100
    ) -> Dict:
        """
        Pre-train model on domain-specific corpus.
        
        Args:
            texts: List of unlabeled hate speech texts
            output_dir: Directory to save adapted model
            num_epochs: Number of pre-training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            max_length: Maximum sequence length
            save_steps: Steps between checkpoints
            logging_steps: Steps between logging
        
        Returns:
            Pre-training results
        """
        if self.model is None:
            self.load_model()
        
        # Setup output directory
        if output_dir is None:
            output_dir = MODELS_DIR / f"domain_adapted_{self.base_model.replace('/', '_')}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nDomain-Adaptive Pre-training")
        logger.info(f"  Corpus size: {len(texts):,} texts")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  MLM probability: {self.mlm_probability}")
        
        # Prepare dataset
        logger.info("Tokenizing corpus...")
        dataset = self.prepare_dataset(texts, max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_dir=str(output_dir / 'logs'),
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            fp16=self.device.type == 'cuda',
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to='none'
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator
        )
        
        # Pre-train
        logger.info("Starting domain-adaptive pre-training...")
        start_time = time.time()
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        logger.info(f"\nPre-training completed in {training_time/60:.2f} minutes")
        logger.info(f"  Final loss: {train_result.training_loss:.4f}")
        
        # Save adapted model
        logger.info(f"Saving domain-adapted model to {output_dir}")
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        # Save metadata
        metadata = {
            'base_model': self.base_model,
            'corpus_size': len(texts),
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'mlm_probability': self.mlm_probability,
            'training_time': training_time,
            'final_loss': float(train_result.training_loss)
        }
        
        metadata_path = output_dir / 'pretraining_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Domain-adapted model ready for fine-tuning!")
        logger.info(f"  Use model: {output_dir}")
        
        return metadata
    
    def get_adapted_model_path(self) -> Path:
        """Get path to adapted model."""
        return MODELS_DIR / f"domain_adapted_{self.base_model.replace('/', '_')}"

# ==================== CONVENIENCE FUNCTIONS ====================

def pretrain_on_corpus(
    texts: List[str],
    base_model: str = 'roberta-base',
    output_dir: Path = None,
    num_epochs: int = 3,
    batch_size: int = 32
) -> Path:
    """
    Convenience function for domain-adaptive pre-training.
    
    Args:
        texts: Unlabeled hate speech texts
        base_model: Base transformer model
        output_dir: Output directory
        num_epochs: Number of epochs
        batch_size: Batch size
    
    Returns:
        Path to adapted model
    """
    pretrainer = DomainAdaptivePretrainer(base_model=base_model)
    pretrainer.pretrain(
        texts=texts,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    return pretrainer.get_adapted_model_path()

def create_combined_corpus(
    labeled_texts: List[str],
    additional_texts: List[str] = None
) -> List[str]:
    """
    Create combined corpus from labeled and additional unlabeled data.
    
    Args:
        labeled_texts: Texts from labeled dataset
        additional_texts: Additional unlabeled hate speech texts
    
    Returns:
        Combined corpus
    """
    corpus = list(labeled_texts)
    
    if additional_texts:
        corpus.extend(additional_texts)
    
    # Remove duplicates
    corpus = list(set(corpus))
    
    logger.info(f"Created corpus with {len(corpus):,} unique texts")
    
    return corpus

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("DOMAIN-ADAPTIVE PRE-TRAINING TEST")
    print("=" * 80)
    
    print("\nDomain-Adaptive Pre-training Process:")
    print("-" * 80)
    print("1. Load base transformer (BERT, RoBERTa, DeBERTa, etc.)")
    print("2. Continue pre-training with Masked Language Modeling (MLM)")
    print("3. Model learns hate speech vocabulary and patterns")
    print("4. Save adapted model for fine-tuning")
    print("5. Fine-tune on labeled data (expect +1-3% improvement)")
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLE")
    print("=" * 80)
    print("""
from models.advanced.domain_adaptive import DomainAdaptivePretrainer

# Step 1: Domain-adaptive pre-training
pretrainer = DomainAdaptivePretrainer(base_model='roberta-base')

# Combine labeled texts (without labels) and any additional unlabeled data
corpus = list(X_train) + list(X_val) + additional_unlabeled_texts

# Pre-train on hate speech corpus
pretrainer.pretrain(
    texts=corpus,
    num_epochs=3,
    batch_size=32
)

# Step 2: Fine-tune adapted model on labeled data
from models.advanced.specialized_transformers import SpecializedTransformerModel

# Load domain-adapted model
adapted_model_path = pretrainer.get_adapted_model_path()

# Now train for classification (with labels)
model = SpecializedTransformerModel.load_trained(adapted_model_path)
model.train(X_train, y_train, X_val, y_val)
""")
    
    print("\n" + "=" * 80)
    print("EXPECTED IMPROVEMENTS")
    print("=" * 80)
    print("  Direct fine-tuning:          Baseline")
    print("  With domain adaptation:      +1-3%")
    print("  ")
    print("  Why it works:")
    print("  - Learns hate speech vocabulary (slang, slurs, etc.)")
    print("  - Understands context-specific meanings")
    print("  - Better initial weights for fine-tuning")