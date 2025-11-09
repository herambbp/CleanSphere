"""
Advanced Models Module for Hate Speech Detection
=================================================

Specialized transformer models and ensemble strategies for improved accuracy.

Components:
1. Specialized Transformers (HateBERT, DeBERTa, etc.)
2. Domain-Adaptive Pre-training
3. Multi-Model Ensembles
4. Cross-Validation Ensembles

Expected improvements:
- Specialized Transformers: +3-5%
- Ensemble Strategy: +2-3%
- Total potential gain: +5-8%
"""

__version__ = '1.0.0'

# Check dependencies
try:
    import torch
    import transformers
    HAS_REQUIREMENTS = True
except ImportError:
    HAS_REQUIREMENTS = False

__all__ = [
    'HAS_REQUIREMENTS',
    'SpecializedTransformerModel',
    'DomainAdaptivePretrainer',
    'EnsembleManager',
    'CrossValidationEnsemble'
]

if HAS_REQUIREMENTS:
    from .specialized_transformers import SpecializedTransformerModel
    from .domain_adaptive import DomainAdaptivePretrainer
    from .ensemble_manager import EnsembleManager, CrossValidationEnsemble