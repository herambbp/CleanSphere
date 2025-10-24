"""
Metrics Collection System
Collects training metrics in real-time during training pipeline
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class MetricsCollector:
    """
    Collect metrics during training run
    
    This class captures all important metrics during a training session
    and organizes them into a structured format suitable for TrainingHistory.
    
    Usage:
        collector = MetricsCollector()
        collector.start_run()
        
        # During training:
        collector.set_datasets(['labeled_data.csv'])
        collector.set_splits(22946, 4917, 4917, class_dist)
        collector.add_model_result('xgboost', metrics)
        collector.set_feature_info(5234)
        
        # After training:
        run_data = collector.finalize()
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.run_id = None
        self.start_time = None
        self.end_time = None
        
        # Core data storage
        self.datasets = []
        self.new_datasets = []
        self.total_samples = 0
        self.train_samples = 0
        self.val_samples = 0
        self.test_samples = 0
        self.class_distribution = {}
        
        # Model results
        self.model_results = {}
        self.best_model = None
        self.best_accuracy = 0.0
        
        # Feature information
        self.feature_dimensions = 0
        
        # Additional metadata
        self.metadata = {}
    
    def start_run(self, run_id: Optional[str] = None):
        """
        Start a new metrics collection session
        
        Args:
            run_id: Optional custom run ID. If None, generates timestamp-based ID
        """
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.run_id = run_id
        self.start_time = time.time()
        
        # Reset all data
        self.datasets = []
        self.new_datasets = []
        self.total_samples = 0
        self.train_samples = 0
        self.val_samples = 0
        self.test_samples = 0
        self.class_distribution = {}
        self.model_results = {}
        self.best_model = None
        self.best_accuracy = 0.0
        self.feature_dimensions = 0
        self.metadata = {}
    
    def set_datasets(
        self, 
        datasets: List[str], 
        total_samples: int,
        new_datasets: Optional[List[str]] = None
    ):
        """
        Record which datasets are being used
        
        Args:
            datasets: List of dataset filenames (e.g., ['labeled_data.csv'])
            total_samples: Total number of samples across all datasets
            new_datasets: List of datasets that are new in this run (optional)
        """
        self.datasets = datasets.copy() if isinstance(datasets, list) else [datasets]
        self.total_samples = total_samples
        self.new_datasets = new_datasets.copy() if new_datasets else []
    
    def set_splits(
        self,
        train_size: int,
        val_size: int,
        test_size: int,
        class_distribution: Dict[int, int]
    ):
        """
        Record train/validation/test split information
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
            class_distribution: Dict mapping class ID to count
                               e.g., {0: 9834, 1: 14123, 2: 8823}
        """
        self.train_samples = train_size
        self.val_samples = val_size
        self.test_samples = test_size
        
        # Convert class IDs to strings for JSON compatibility
        self.class_distribution = {
            str(k): int(v) for k, v in class_distribution.items()
        }
    
    def add_model_result(
        self,
        model_name: str,
        metrics: Dict[str, float]
    ):
        """
        Add results for a single model
        
        Args:
            model_name: Name of the model (e.g., 'xgboost', 'svm')
            metrics: Dictionary of metrics, must include 'accuracy'
                    e.g., {
                        'accuracy': 0.8712,
                        'f1_macro': 0.8445,
                        'precision_macro': 0.8556,
                        'recall_macro': 0.8389
                    }
        """
        if 'accuracy' not in metrics:
            raise ValueError(f"Model {model_name} metrics must include 'accuracy'")
        
        # Store model results
        self.model_results[model_name] = metrics.copy()
        
        # Update best model if this is better
        accuracy = metrics['accuracy']
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model = model_name
    
    def set_feature_info(self, feature_dimensions: int):
        """
        Record feature extraction information
        
        Args:
            feature_dimensions: Number of features extracted
        """
        self.feature_dimensions = feature_dimensions
    
    def add_metadata(self, key: str, value: Any):
        """
        Add custom metadata to the run
        
        Args:
            key: Metadata key
            value: Metadata value (must be JSON-serializable)
        """
        self.metadata[key] = value
    
    def finalize(self) -> Dict:
        """
        Finalize metrics collection and return complete run data
        
        Returns:
            Dictionary with all collected metrics, ready for TrainingHistory
            
        Structure:
            {
                'run_id': '20250124_143000',
                'timestamp': '2025-01-24 14:30:00',
                'datasets': ['labeled_data.csv'],
                'new_datasets': [],
                'total_samples': 32780,
                'train_samples': 22946,
                'val_samples': 4917,
                'test_samples': 4917,
                'class_distribution': {'0': 9834, '1': 14123, '2': 8823},
                'models': {
                    'svm': {'accuracy': 0.8534, ...},
                    'xgboost': {'accuracy': 0.8612, ...}
                },
                'best_model': 'xgboost',
                'best_accuracy': 0.8612,
                'feature_dimensions': 5234,
                'duration_seconds': 342.5,
                'metadata': {...}
            }
        """
        if self.run_id is None:
            raise RuntimeError("Must call start_run() before finalize()")
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Build run data dictionary
        run_data = {
            'run_id': self.run_id,
            'timestamp': timestamp,
            'datasets': self.datasets,
            'new_datasets': self.new_datasets,
            'total_samples': self.total_samples,
            'train_samples': self.train_samples,
            'val_samples': self.val_samples,
            'test_samples': self.test_samples,
            'class_distribution': self.class_distribution,
            'models': self.model_results,
            'best_model': self.best_model,
            'best_accuracy': self.best_accuracy,
            'feature_dimensions': self.feature_dimensions,
            'duration_seconds': round(duration, 2)
        }
        
        # Add metadata if any
        if self.metadata:
            run_data['metadata'] = self.metadata
        
        return run_data
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of collected metrics
        
        Returns:
            Formatted string with summary
        """
        if self.run_id is None:
            return "No metrics collected yet. Call start_run() first."
        
        lines = []
        lines.append(f"Run ID: {self.run_id}")
        lines.append(f"Datasets: {', '.join(self.datasets)}")
        
        if self.new_datasets:
            lines.append(f"New datasets: {', '.join(self.new_datasets)}")
        
        lines.append(f"Total samples: {self.total_samples:,}")
        lines.append(f"  Train: {self.train_samples:,}")
        lines.append(f"  Val:   {self.val_samples:,}")
        lines.append(f"  Test:  {self.test_samples:,}")
        
        if self.class_distribution:
            lines.append("Class distribution:")
            for class_id, count in sorted(self.class_distribution.items()):
                percentage = (int(count) / self.total_samples * 100) if self.total_samples > 0 else 0
                lines.append(f"  Class {class_id}: {count:,} ({percentage:.1f}%)")
        
        if self.model_results:
            lines.append(f"Models trained: {len(self.model_results)}")
            if self.best_model:
                lines.append(f"Best model: {self.best_model} ({self.best_accuracy:.4f})")
        
        if self.feature_dimensions > 0:
            lines.append(f"Feature dimensions: {self.feature_dimensions:,}")
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            lines.append(f"Duration: {duration:.1f} seconds")
        
        return '\n'.join(lines)
    
    def validate(self) -> bool:
        """
        Validate that all required data has been collected
        
        Returns:
            True if all required fields are present, False otherwise
        """
        if self.run_id is None:
            return False
        
        if not self.datasets:
            return False
        
        if self.total_samples <= 0:
            return False
        
        if not self.model_results:
            return False
        
        if self.best_model is None:
            return False
        
        return True
    
    def reset(self):
        """Reset collector to initial state"""
        self.__init__()