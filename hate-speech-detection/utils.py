"""
Utility functions for logging, metrics, and evaluation
"""

import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from config import LOGS_DIR, LOG_FORMAT, LOG_DATE_FORMAT, CLASS_LABELS

# ==================== LOGGING SETUP ====================

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO):
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOGS_DIR / log_file, mode='a')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger('hate_speech_detection', 'training.log')

# ==================== METRICS CALCULATION ====================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate per-class metrics.
    
    Returns:
        Dictionary with class-wise precision, recall, f1
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class = {}
    for i, class_name in CLASS_LABELS.items():
        if i < len(precision):
            per_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(np.sum(y_true == i))
            }
    
    return per_class


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Get sklearn classification report as string."""
    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    return classification_report(y_true, y_pred, target_names=target_names, digits=3, zero_division=0)

# ==================== MODEL EVALUATION ====================

class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None, training_time: float = None):
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            training_time: Training time in seconds (optional)
        
        Returns:
            Dictionary of evaluation results
        """
        result = {
            'model_name': model_name,
            'metrics': calculate_metrics(y_true, y_pred),
            'per_class': calculate_per_class_metrics(y_true, y_pred),
            'confusion_matrix': get_confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': get_classification_report(y_true, y_pred)
        }
        
        if training_time is not None:
            result['training_time'] = training_time
        
        self.results.append(result)
        return result
    
    def print_evaluation(self, result: Dict):
        """Print formatted evaluation results."""
        print("\n" + "=" * 70)
        print(f"MODEL EVALUATION: {result['model_name']}")
        print("=" * 70)
        
        # Overall metrics
        print("\nOVERALL METRICS:")
        print("-" * 70)
        metrics = result['metrics']
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1 Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted):{metrics['f1_weighted']:.4f}")
        
        if 'training_time' in result:
            print(f"\nTraining Time:      {result['training_time']:.2f}s")
        
        # Classification report
        print("\nCLASSIFICATION REPORT:")
        print("-" * 70)
        print(result['classification_report'])
        
        print("=" * 70)
    
    def get_comparison_df(self) -> pd.DataFrame:
        """Get comparison DataFrame for all evaluated models."""
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for result in self.results:
            row = {
                'Model': result['model_name'],
                'Accuracy': result['metrics']['accuracy'],
                'F1 (Macro)': result['metrics']['f1_macro'],
                'F1 (Weighted)': result['metrics']['f1_weighted'],
                'Precision': result['metrics']['precision_macro'],
                'Recall': result['metrics']['recall_macro']
            }
            
            if 'training_time' in result:
                row['Training Time (s)'] = result['training_time']
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        return df.sort_values('Accuracy', ascending=False)
    
    def print_comparison(self):
        """Print model comparison table."""
        df = self.get_comparison_df()
        
        if df.empty:
            print("No models to compare")
            return
        
        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        # Highlight best model
        best = df.iloc[0]
        print(f"\nBEST MODEL: {best['Model']}")
        print(f"  Accuracy: {best['Accuracy']:.4f}")
        print(f"  F1 Score (Macro): {best['F1 (Macro)']:.4f}")

# ==================== PROGRESS TRACKING ====================

class ProgressTracker:
    """Simple progress tracker for long operations."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        
        # Print progress every 10% or at completion
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            percentage = (self.current / self.total) * 100
            
            if elapsed > 0:
                rate = self.current / elapsed
                eta = (self.total - self.current) / rate if rate > 0 else 0
                logger.info(f"{self.desc}: {self.current}/{self.total} ({percentage:.1f}%) | "
                          f"Rate: {rate:.1f} items/s | ETA: {eta:.1f}s")
            else:
                logger.info(f"{self.desc}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def close(self):
        """Complete progress tracking."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.desc}: Complete! {self.total} items in {elapsed:.2f}s")

# ==================== PRETTY PRINTING ====================

def print_section_header(title: str, width: int = 70):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection_header(title: str, width: int = 70):
    """Print formatted subsection header."""
    print("\n" + title)
    print("-" * width)


def print_dict(d: dict, indent: int = 2):
    """Pretty print dictionary."""
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 2)
        elif isinstance(value, float):
            print(" " * indent + f"{key}: {value:.4f}")
        else:
            print(" " * indent + f"{key}: {value}")


def print_class_distribution(y: np.ndarray):
    """Print class distribution."""
    print("\nClass Distribution:")
    total = len(y)
    
    for class_id in sorted(np.unique(y)):
        count = np.sum(y == class_id)
        percentage = count / total * 100
        class_name = CLASS_LABELS.get(class_id, f"Class {class_id}")
        print(f"  {class_name:20s}: {count:6d} ({percentage:5.2f}%)")


def format_time(seconds: float) -> str:
    """Format seconds into readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

# ==================== DATA VALIDATION ====================

def validate_arrays(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Validate input arrays for consistency.
    
    Args:
        X: Features array
        y: Labels array
    
    Returns:
        True if valid, raises exception otherwise
    """
    if len(X) != len(y):
        raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
    
    if len(X) == 0:
        raise ValueError("Empty arrays provided")
    
    # Check for valid class labels
    unique_classes = np.unique(y)
    expected_classes = set(CLASS_LABELS.keys())
    invalid_classes = set(unique_classes) - expected_classes
    
    if invalid_classes:
        raise ValueError(f"Invalid class labels found: {invalid_classes}. "
                        f"Expected: {expected_classes}")
    
    return True


def check_class_balance(y: np.ndarray, threshold: float = 0.05):
    """
    Check if classes are severely imbalanced.
    
    Args:
        y: Labels array
        threshold: Minimum acceptable class percentage
    
    Returns:
        True if balanced enough, False otherwise
    """
    total = len(y)
    class_counts = {class_id: np.sum(y == class_id) for class_id in np.unique(y)}
    
    imbalanced = False
    for class_id, count in class_counts.items():
        percentage = count / total
        if percentage < threshold:
            class_name = CLASS_LABELS.get(class_id, f"Class {class_id}")
            logger.warning(f"Class '{class_name}' is severely underrepresented: "
                         f"{percentage*100:.2f}% (threshold: {threshold*100:.2f}%)")
            imbalanced = True
    
    return not imbalanced

# ==================== FILE OPERATIONS ====================

def save_results(results: Dict, filename: str):
    """Save results to JSON file."""
    import json
    from config import RESULTS_DIR
    
    filepath = RESULTS_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filepath}")


def load_results(filename: str) -> Dict:
    """Load results from JSON file."""
    import json
    from config import RESULTS_DIR
    
    filepath = RESULTS_DIR / filename
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Results loaded from {filepath}")
    return results