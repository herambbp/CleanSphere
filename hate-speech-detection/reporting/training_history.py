"""
Training History Management System
Stores and queries historical training run data in JSON format
CORRECTED VERSION with get_previous_run method
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import shutil


class TrainingHistory:
    """
    Manage training history storage and retrieval
    
    Stores all training runs in a JSON file with the following structure:
    {
        "runs": [
            {
                "run_id": "20250124_143000",
                "timestamp": "2025-01-24 14:30:00",
                "datasets": ["labeled_data.csv", "dataset2.csv"],
                "new_datasets": ["dataset2.csv"],
                "total_samples": 45230,
                "train_samples": 31661,
                "val_samples": 6784,
                "test_samples": 6785,
                "class_distribution": {"0": 9834, "1": 14123, "2": 8823},
                "models": {
                    "svm": {"accuracy": 0.8534, "f1_macro": 0.8201, ...},
                    ...
                },
                "best_model": "xgboost",
                "best_accuracy": 0.8612,
                "feature_dimensions": 5234,
                "duration_seconds": 342.5
            }
        ]
    }
    """
    
    def __init__(self, history_file: str = 'results/training_history.json'):
        """
        Initialize training history manager
        
        Args:
            history_file: Path to JSON history file
        """
        self.history_file = Path(history_file)
        self.runs = []
        self._load_history()
    
    def _load_history(self):
        """Load existing history from JSON file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.runs = data.get('runs', [])
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse history file: {e}")
                print("Starting with empty history")
                self.runs = []
            except Exception as e:
                print(f"Warning: Error loading history: {e}")
                self.runs = []
        else:
            self.runs = []
    
    def _save_history(self):
        """Save current history to JSON file with backup"""
        # Create directory if it doesn't exist
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if it exists
        if self.history_file.exists():
            backup_file = self.history_file.with_suffix('.json.backup')
            shutil.copy2(self.history_file, backup_file)
        
        # Write new history
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({'runs': self.runs}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")
            # Restore from backup if save failed
            backup_file = self.history_file.with_suffix('.json.backup')
            if backup_file.exists():
                shutil.copy2(backup_file, self.history_file)
                print("Restored from backup")
    
    def add_run(self, run_data: Dict):
        """
        Add a new training run to history
        
        Args:
            run_data: Dictionary containing run information
                Required keys: run_id, timestamp, datasets, total_samples,
                              models, best_model, best_accuracy
        """
        # Validate required fields
        required_fields = [
            'run_id', 'timestamp', 'datasets', 'total_samples',
            'models', 'best_model', 'best_accuracy'
        ]
        
        missing_fields = [f for f in required_fields if f not in run_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Add to runs list
        self.runs.append(run_data)
        
        # Save to disk
        self._save_history()
    
    def get_last_run(self) -> Optional[Dict]:
        """
        Get the most recent training run
        
        Returns:
            Dictionary with run data, or None if no runs exist
        """
        if not self.runs:
            return None
        return self.runs[-1]
    
    def get_previous_run(self) -> Optional[Dict]:
        """
        Get the second most recent training run (for comparison with current)
        
        Returns:
            Dictionary with run data, or None if fewer than 2 runs exist
        """
        if len(self.runs) < 2:
            return None
        return self.runs[-2]
    
    def get_run_by_id(self, run_id: str) -> Optional[Dict]:
        """
        Get a specific run by its ID
        
        Args:
            run_id: The run identifier (e.g., "20250124_143000")
        
        Returns:
            Run data dictionary, or None if not found
        """
        for run in self.runs:
            if run.get('run_id') == run_id:
                return run
        return None
    
    def get_all_runs(self) -> List[Dict]:
        """
        Get all training runs
        
        Returns:
            List of all run dictionaries
        """
        return self.runs.copy()
    
    def get_last_n_runs(self, n: int) -> List[Dict]:
        """
        Get the last N training runs
        
        Args:
            n: Number of runs to retrieve
        
        Returns:
            List of last N run dictionaries
        """
        if n <= 0:
            return []
        return self.runs[-n:]
    
    def get_run_count(self) -> int:
        """
        Get total number of training runs
        
        Returns:
            Number of runs in history
        """
        return len(self.runs)
    
    def compare_last_two(self) -> Optional[Dict]:
        """
        Compare the last two training runs
        
        Returns:
            Dictionary with comparison data, or None if fewer than 2 runs
            
            Structure:
            {
                'previous_run': {...},
                'current_run': {...},
                'datasets_added': [...],
                'datasets_removed': [...],
                'samples_delta': 12450,
                'samples_delta_percent': 38.0,
                'accuracy_delta': 0.0178,
                'accuracy_delta_percent': 2.08,
                'f1_delta': 0.0244,
                'improved': True,
                'summary': "Added dataset2.csv (+12,450 samples). Accuracy improved by 2.08%"
            }
        """
        if len(self.runs) < 2:
            return None
        
        previous = self.runs[-2]
        current = self.runs[-1]
        
        # Calculate dataset changes
        prev_datasets = set(previous.get('datasets', []))
        curr_datasets = set(current.get('datasets', []))
        
        datasets_added = list(curr_datasets - prev_datasets)
        datasets_removed = list(prev_datasets - curr_datasets)
        
        # Calculate sample changes
        prev_samples = previous.get('total_samples', 0)
        curr_samples = current.get('total_samples', 0)
        samples_delta = curr_samples - prev_samples
        samples_delta_percent = (samples_delta / prev_samples * 100) if prev_samples > 0 else 0
        
        # Calculate accuracy changes
        prev_accuracy = previous.get('best_accuracy', 0)
        curr_accuracy = current.get('best_accuracy', 0)
        accuracy_delta = curr_accuracy - prev_accuracy
        accuracy_delta_percent = (accuracy_delta / prev_accuracy * 100) if prev_accuracy > 0 else 0
        
        # Calculate F1 changes (get from best model)
        prev_best_model = previous.get('best_model', '')
        curr_best_model = current.get('best_model', '')
        
        prev_f1 = previous.get('models', {}).get(prev_best_model, {}).get('f1_macro', 0)
        curr_f1 = current.get('models', {}).get(curr_best_model, {}).get('f1_macro', 0)
        f1_delta = curr_f1 - prev_f1
        
        # Determine if improved
        improved = accuracy_delta > 0
        
        # Generate summary
        summary_parts = []
        if datasets_added:
            summary_parts.append(f"Added {', '.join(datasets_added)} (+{samples_delta:,} samples)")
        if datasets_removed:
            summary_parts.append(f"Removed {', '.join(datasets_removed)}")
        
        if improved:
            summary_parts.append(f"Accuracy improved by {accuracy_delta_percent:.2f}%")
        else:
            summary_parts.append(f"Accuracy decreased by {abs(accuracy_delta_percent):.2f}%")
        
        summary = ". ".join(summary_parts)
        
        return {
            'previous_run': previous,
            'current_run': current,
            'datasets_added': datasets_added,
            'datasets_removed': datasets_removed,
            'samples_delta': samples_delta,
            'samples_delta_percent': samples_delta_percent,
            'accuracy_delta': accuracy_delta,
            'accuracy_delta_percent': accuracy_delta_percent,
            'f1_delta': f1_delta,
            'improved': improved,
            'summary': summary
        }
    
    def get_metric_history(
        self, 
        metric_name: str, 
        model_name: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get historical values of a specific metric
        
        Args:
            metric_name: Name of metric (e.g., 'accuracy', 'f1_macro')
            model_name: Specific model name, or None for best model
        
        Returns:
            List of (timestamp, value) tuples
        """
        history = []
        
        for run in self.runs:
            timestamp = run.get('timestamp', '')
            
            if model_name:
                # Get metric for specific model
                value = run.get('models', {}).get(model_name, {}).get(metric_name)
            else:
                # Get metric for best model
                if metric_name == 'accuracy':
                    value = run.get('best_accuracy')
                else:
                    best_model = run.get('best_model', '')
                    value = run.get('models', {}).get(best_model, {}).get(metric_name)
            
            if value is not None:
                history.append((timestamp, value))
        
        return history
    
    def get_dataset_first_use(self, dataset_name: str) -> Optional[Dict]:
        """
        Find when a dataset was first used
        
        Args:
            dataset_name: Name of the dataset file
        
        Returns:
            Run data when dataset was first used, or None if never used
        """
        for run in self.runs:
            if dataset_name in run.get('datasets', []):
                return run
        return None
    
    def get_statistics(self) -> Dict:
        """
        Calculate overall statistics from history
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.runs:
            return {
                'total_runs': 0,
                'unique_datasets': [],
                'best_accuracy_ever': 0,
                'best_run_id': None,
                'average_accuracy': 0,
                'total_samples_trained': 0
            }
        
        # Collect all unique datasets
        all_datasets = set()
        for run in self.runs:
            all_datasets.update(run.get('datasets', []))
        
        # Find best accuracy
        best_accuracy = 0
        best_run_id = None
        accuracies = []
        
        for run in self.runs:
            acc = run.get('best_accuracy', 0)
            accuracies.append(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_run_id = run.get('run_id')
        
        # Calculate average
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Total unique samples (use max from any run)
        max_samples = max((run.get('total_samples', 0) for run in self.runs), default=0)
        
        return {
            'total_runs': len(self.runs),
            'unique_datasets': sorted(list(all_datasets)),
            'best_accuracy_ever': best_accuracy,
            'best_run_id': best_run_id,
            'average_accuracy': avg_accuracy,
            'total_samples_trained': max_samples
        }