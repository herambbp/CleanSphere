# Quick Start Guide - Advanced Models

## Installation

```bash
# 1. Install requirements
cd hate-speech-detection
pip install -r models/advanced/requirements.txt

# 2. Verify installation
python -m models.advanced.test_advanced
```

## 5-Minute Quick Start

Train HateBERT (fastest, good results):

```python
from models.advanced.specialized_transformers import SpecializedTransformerModel

# Load your data
X_train, y_train = ...  # Your training data
X_val, y_val = ...      # Your validation data
X_test, y_test = ...    # Your test data

# Train HateBERT
model = SpecializedTransformerModel(model_type='hatebert')
model.train(X_train, y_train, X_val, y_val)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_macro']:.4f}")

# Predict
predictions = model.predict(["I hate you", "Good morning"])
print(predictions)  # [0, 2]
```

**Expected:** +2-4% improvement over BERT-base

## 25-Minute Best Single Model

Train DeBERTa-v3-large (best accuracy):

```python
from models.advanced.specialized_transformers import SpecializedTransformerModel

model = SpecializedTransformerModel(model_type='deberta-v3-large')
model.train(X_train, y_train, X_val, y_val, num_epochs=4)
metrics = model.evaluate(X_test, y_test)
```

**Expected:** +4-6% improvement

## 45-Minute Multi-Model Ensemble

Best accuracy with ensemble:

```python
from models.advanced.train_advanced import train_advanced_models

trainer = train_advanced_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    model_types=['hatebert', 'deberta-v3-large'],
    create_ensemble=True,
    ensemble_method='weighted_voting'
)

# Results
best_name, best_model = trainer.get_best_model()
print(f"Best: {best_name}")
print(f"F1 Score: {trainer.results[best_name]['f1_macro']:.4f}")
```

**Expected:** +5-7% improvement

## Integration with Existing System

Add to your `main_train_enhanced.py`:

```python
# Add import at top
from models.advanced.train_advanced import train_advanced_models

# Add phase 6 function
def phase6_advanced_models(
    X_train, y_train, X_val, y_val, X_test, y_test
):
    """Phase 6: Advanced models."""
    print_section_header("PHASE 6: ADVANCED MODELS")
    
    trainer = train_advanced_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_types=['hatebert', 'deberta-v3-large'],
        create_ensemble=True
    )
    
    return trainer

# Add to main() function
if args.advanced:
    phase6_trainer = phase6_advanced_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

# Add argument parser
parser.add_argument('--advanced', action='store_true',
                   help='Run Phase 6: Advanced models')
```

Usage:
```bash
python main_train_enhanced.py --advanced
```

## Model Selection Guide

Choose based on your needs:

### Priority: Speed
```python
model = SpecializedTransformerModel('hatebert')
```
- Training: ~10 min
- Improvement: +2-4%
- VRAM: ~8GB

### Priority: Accuracy
```python
model = SpecializedTransformerModel('deberta-v3-large')
```
- Training: ~25 min
- Improvement: +4-6%
- VRAM: ~16GB

### Priority: Maximum Accuracy
```python
trainer = train_advanced_models(
    ...,
    model_types=['hatebert', 'deberta-v3-large', 'roberta-large'],
    create_ensemble=True
)
```
- Training: ~45 min
- Improvement: +5-7%
- VRAM: ~16GB (sequential training)

## Common Tasks

### Save Model
```python
model = SpecializedTransformerModel('hatebert')
model.train(...)
model.trainer.save_model('path/to/save')
```

### Load Model
```python
model = SpecializedTransformerModel.load_trained('path/to/model')
predictions = model.predict(texts)
```

### Predict with Confidence
```python
probas = model.predict_proba(texts)
predictions = np.argmax(probas, axis=1)
confidence = np.max(probas, axis=1)

for text, pred, conf in zip(texts, predictions, confidence):
    print(f"{text}: {CLASS_LABELS[pred]} ({conf:.2%})")
```

### Batch Prediction
```python
# Automatically handles batching
predictions = model.predict(large_text_list)
```

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
model.train(..., batch_size=8)

# Or use gradient accumulation
model.train(..., batch_size=8, gradient_accumulation_steps=4)
```

### Slow Training
```python
# Use smaller model
model = SpecializedTransformerModel('hatebert')  # Instead of deberta

# Or reduce epochs
model.train(..., num_epochs=3)
```

### Model Not Improving
```python
# Try domain-adaptive pre-training
from models.advanced.domain_adaptive import DomainAdaptivePretrainer

pretrainer = DomainAdaptivePretrainer(base_model='roberta-base')
corpus = list(X_train) + list(X_val)
pretrainer.pretrain(texts=corpus, num_epochs=3)
```

## Advanced Usage

### Cross-Validation Ensemble
```python
from models.advanced.ensemble_manager import CrossValidationEnsemble

cv_ensemble = CrossValidationEnsemble(
    model_class=SpecializedTransformerModel,
    model_kwargs={'model_type': 'hatebert'},
    n_folds=5
)

cv_ensemble.train(X_train, y_train, X_val, y_val)
predictions = cv_ensemble.predict(X_test)
```

### Custom Ensemble Weights
```python
from models.advanced.ensemble_manager import EnsembleManager

ensemble = EnsembleManager()
ensemble.add_model('hatebert', model1, weight=0.4)
ensemble.add_model('deberta', model2, weight=0.6)

predictions = ensemble.predict(X_test, method='weighted_voting')
```

### Stacking Ensemble
```python
ensemble = EnsembleManager()
ensemble.add_model('hatebert', model1)
ensemble.add_model('deberta', model2)

# Train meta-classifier
ensemble.train_stacking_classifier(
    X_train, y_train, X_val, y_val,
    meta_classifier_type='logistic'
)

predictions = ensemble.predict(X_test, method='stacking')
```

## Getting Help

1. **Check documentation**: `models/advanced/README.md`
2. **Run test script**: `python -m models.advanced.test_advanced`
3. **View examples**: Each module has `__main__` section with examples
4. **Check logs**: Training produces detailed logs

## Next Steps

1. **Start simple**: Train HateBERT first
2. **Evaluate results**: Compare with your baseline
3. **Try ensemble**: If +2-4% is good, try ensemble for +5-7%
4. **Optimize**: Adjust hyperparameters based on results
5. **Deploy**: Use best model in production

## Expected Results

Typical improvements on hate speech datasets:

| Your Baseline F1 | Expected After HateBERT | Expected After Ensemble |
|------------------|-------------------------|-------------------------|
| 0.70 | 0.72-0.74 (+2-4%) | 0.74-0.76 (+4-7%) |
| 0.75 | 0.77-0.79 (+2-4%) | 0.79-0.82 (+4-7%) |
| 0.80 | 0.82-0.84 (+2-4%) | 0.84-0.87 (+4-7%) |

Actual results depend on dataset quality and size.

---

**Ready to start?**

```bash
python -m models.advanced.test_advanced
```

This will verify your installation and show available options.