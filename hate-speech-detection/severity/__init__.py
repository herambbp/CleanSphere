"""
Severity Classification System for Hate Speech Detection

This module provides severity analysis and action recommendations for detected hate speech.

Components:
- severity_classifier: Multi-level severity detection (LOW to EXTREME)
- action_recommender: Platform-agnostic action recommendations
"""

from .severity_classifier import (
    KeywordDetector,
    TextFeaturesAnalyzer,
    ContextAnalyzer,
    SeverityScorer
)

__all__ = [
    'KeywordDetector',
    'TextFeaturesAnalyzer',
    'ContextAnalyzer',
    'SeverityScorer'
]

__version__ = '1.0.0'