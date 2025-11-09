# Advanced Models for Hate Speech Detection

Complete implementation of state-of-the-art techniques for hate speech detection, optimized for A5000 GPU (24GB VRAM).

**Expected Total Improvement: +5-8% over baseline BERT**

## Overview

This module implements three key strategies for improved accuracy:

1. **Specialized Transformers** (+3-5%)
2. **Domain-Adaptive Pre-training** (+1-3%)
3. **Ensemble Methods** (+2-3%)

## Quick Start

```python
from models.advanced.train_advanced import train_advanced_models

# Train best model (simplest)
trainer = train_advanced_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_types=['deberta-v3-large']
)

# Or create multi-model ensemble (best accuracy)
trainer = train_advanced_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_types=['hatebert', 'deberta-v3-large', 'roberta-large'],
    create_ensemble=True
)
```

## Components

### 1. Specialized Transformers

Pre-trained models optimized for hate speech:

| Model | Parameters | Expected Gain | Best For |
|-------|-----------|---------------|----------|
| **HateBERT** | 110M | +2-4% | Hate speech specialist |
| **DeBERTa-v3-large** | 434M | +4-6% | Best single model |
| **DeBERTa-v3-base** | 184M | +3-5% | Balanced performance |
| **RoBERTa-large** | 355M | +2-3% | Strong baseline |
| **BERT-large** | 340M | +1-2% | Reference model |

**Why these models?**

- **HateBERT**: Pre-trained on Reddit hate speech corpus - understands hate speech vocabulary
- **DeBERTa**: State-of-the-art disentangled attention mechanism
- **RoBERTa**: Improved BERT with better pre-training

**Usage:**

```python
from models.advanced.specialized_transformers import SpecializedTransformerModel

# Train HateBERT
model = SpecializedTransformerModel(model_type='hatebert')
model.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"F1 Score: {metrics['f1_macro']:.4f}")
```

### 2. Domain-Adaptive Pre-training

Continue pre-training on hate speech corpus before fine-tuning.

**Process:**
1. Load base transformer (BERT, RoBERTa, etc.)
2. Continue pre-training with Masked Language Modeling (MLM)
3. Learn hate speech vocabulary and patterns
4. Fine-tune on labeled data

**Expected gain: +1-3%**

**Usage:**

```python
from models.advanced.domain_adaptive import DomainAdaptivePretrainer

# Step 1: Pre-train on hate speech corpus
pretrainer = DomainAdaptivePretrainer(base_model='roberta-base')

# Combine all available hate speech text (unlabeled)
corpus = list(X_train) + list(X_val) + additional_texts

pretrainer.pretrain(
    texts=corpus,
    num_epochs=3,
    batch_size=32
)

# Step 2: Fine-tune on labeled data
adapted_model_path = pretrainer.get_adapted_model_path()
# ... load and fine-tune
```

### 3. Ensemble Methods

Combine multiple models for improved accuracy.

#### Multi-Model Ensemble

Combine different architectures (HateBERT, DeBERTa, RoBERTa).

**Expected gain: +2-3%**

```python
from models.advanced.ensemble_manager import EnsembleManager

# Train multiple models
hatebert = SpecializedTransformerModel('hatebert')
hatebert.train(X_train, y_train, X_val, y_val)

deberta = SpecializedTransformerModel('deberta-v3-large')
deberta.train(X_train, y_train, X_val, y_val)

# Create ensemble
ensemble = EnsembleManager()
ensemble.add_model('hatebert', hatebert)
ensemble.add_model('deberta', deberta)

# Method 1: Soft voting (simple average)
predictions = ensemble.predict(X_test, method='soft_voting')

# Method 2: Weighted voting (learned weights)
ensemble.learn_optimal_weights(X_val, y_val)
predictions = ensemble.predict(X_test, method='weighted_voting')

# Method 3: Stacking (meta-classifier)
ensemble.train_stacking_classifier(X_train, y_train, X_val, y_val)
predictions = ensemble.predict(X_test, method='stacking')
```

#### Cross-Validation Ensemble

Train same architecture with different data splits.

**Expected gain: +1-2%**

```python
from models.advanced.ensemble_manager import CrossValidationEnsemble

# Create 5-fold ensemble
cv_ensemble = CrossValidationEnsemble(
    model_class=SpecializedTransformerModel,
    model_kwargs={'model_type': 'hatebert'},
    n_folds=5
)

# Train all folds
cv_ensemble.train(X_train, y_train, X_val, y_val)

# Predict (automatically averages all folds)
predictions = cv_ensemble.predict(X_test)
```

## Complete Pipeline

Full pipeline with all techniques:

```python
from models.advanced.train_advanced import train_advanced_models

trainer = train_advanced_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    
    # Train multiple specialized models
    model_types=['hatebert', 'deberta-v3-large', 'roberta-large'],
    
    # Use domain-adaptive pre-training
    use_domain_adaptation=True,
    
    # Create multi-model ensemble
    create_ensemble=True,
    ensemble_method='weighted_voting',
    
    # Optional: CV ensemble for single model
    create_cv_ensemble=True,
    cv_model_type='hatebert',
    cv_n_folds=5
)

# Get best result
best_name, best_model = trainer.get_best_model()
print(f"Best: {best_name}")
```

## GPU Requirements

Optimized for NVIDIA A5000 (24GB VRAM):

| Model | Batch Size | VRAM Usage | Training Time |
|-------|-----------|------------|---------------|
| HateBERT | 32 | ~8GB | ~10 min |
| DeBERTa-v3-base | 24 | ~12GB | ~15 min |
| DeBERTa-v3-large | 16 | ~16GB | ~25 min |
| RoBERTa-large | 20 | ~14GB | ~20 min |

**Memory optimization:**
- Mixed precision (FP16) enabled by default
- Gradient checkpointing for large models
- Configurable batch sizes

## Expected Results

Based on typical hate speech datasets:

| Configuration | Expected F1 (macro) | Improvement |
|--------------|---------------------|-------------|
| BERT-base (baseline) | 0.75 | - |
| HateBERT | 0.77-0.79 | +2-4% |
| DeBERTa-v3-large | 0.79-0.81 | +4-6% |
| Multi-model ensemble | 0.80-0.83 | +5-8% |

**Actual results depend on:**
- Dataset size and quality
- Class balance
- Domain specificity

## Installation

```bash
# Install required packages
pip install torch transformers accelerate
pip install scikit-learn scipy numpy

# Or use requirements file
pip install -r models/advanced/requirements.txt
```

## Tips for Best Results

1. **Start with HateBERT**
   - Best single model for hate speech
   - Faster than DeBERTa
   - Good baseline

2. **Use DeBERTa-v3-large for maximum accuracy**
   - Best single model performance
   - Requires more VRAM and time
   - Worth it for production systems

3. **Ensemble for critical applications**
   - 2-3% improvement worth the extra computation
   - Use weighted voting (best balance)
   - Stacking if you have time for meta-training

4. **Domain adaptation helps with specialized vocabulary**
   - Important if your domain has unique slang/terms
   - Requires unlabeled data from target domain
   - +1-3% improvement for domain-specific data

5. **CV ensemble for single best model**
   - Good alternative to multi-model ensemble
   - More robust to data variations
   - Easier to deploy (same architecture)

## Integration with Existing Pipeline

To integrate with your existing training pipeline:

```python
# In main_train_enhanced.py, add:

def phase_6_advanced_models(
    X_train, y_train, X_val, y_val, X_test, y_test
):
    """Phase 6: Advanced models with specialized transformers."""
    from models.advanced.train_advanced import train_advanced_models
    
    trainer = train_advanced_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_types=['hatebert', 'deberta-v3-large'],
        create_ensemble=True
    )
    
    return trainer

# Then in main():
if args.advanced_models:
    trainer = phase_6_advanced_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
```

## File Structure

```
models/advanced/
├── __init__.py                      # Module initialization
├── specialized_transformers.py      # HateBERT, DeBERTa, etc.
├── domain_adaptive.py               # Domain-adaptive pre-training
├── ensemble_manager.py              # Multi-model & CV ensembles
├── train_advanced.py                # Complete training pipeline
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

## Troubleshooting

**Out of memory error:**
```python
# Reduce batch size
model = SpecializedTransformerModel('deberta-v3-large')
model.train(..., batch_size=8)  # Lower from 16

# Or use gradient accumulation
model.train(..., batch_size=8, gradient_accumulation_steps=2)
```

**Slow training:**
```python
# Use smaller model
model = SpecializedTransformerModel('hatebert')  # Instead of deberta-v3-large

# Or reduce epochs
model.train(..., num_epochs=3)  # Instead of 5
```

**Models not improving:**
- Check data quality and preprocessing
- Verify class balance
- Try domain-adaptive pre-training
- Use ensemble methods

## References

1. HateBERT: https://arxiv.org/abs/2010.12472
2. DeBERTa: https://arxiv.org/abs/2006.03654
3. Domain-Adaptive Pre-training: https://arxiv.org/abs/2004.10964
4. Ensemble Methods: https://arxiv.org/abs/1906.00051

## Support

For issues or questions:
1. Check this README
2. Review example code in `__main__` sections
3. Check training logs for error messages
4. Verify GPU availability: `torch.cuda.is_available()`

## License

Same as main project.