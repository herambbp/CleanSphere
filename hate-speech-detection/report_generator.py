"""
Comprehensive Model Evaluation and Reporting System
Generates detailed metrics, visualizations, and comparison reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MetricsCalculator:
    """Calculates comprehensive evaluation metrics for classification models."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names or ["Hate speech", "Offensive language", "Neither"]
    
    def calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics
    
    def calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary mapping class names to metric dictionaries
        """
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(np.sum(y_true == i))
            }
        
        return per_class
    
    def calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def calculate_roc_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate ROC curve and AUC for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with ROC data for each class
        """
        n_classes = len(self.class_names)
        
        # Binarize the labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        roc_data = {}
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_data[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc)
            }
        
        # Calculate micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        roc_data['micro_average'] = {
            'fpr': fpr_micro.tolist(),
            'tpr': tpr_micro.tolist(),
            'auc': float(roc_auc_micro)
        }
        
        return roc_data
    
    def calculate_precision_recall_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate precision-recall curves for each class.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with precision-recall data
        """
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        pr_data = {}
        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], 
                y_pred_proba[:, i]
            )
            pr_auc = auc(recall, precision)
            
            pr_data[class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'auc': float(pr_auc)
            }
        
        return pr_data


class VisualizationGenerator:
    """Generates visualization plots for model evaluation."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualization generator.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        model_name: str,
        normalize: bool = False
    ) -> Path:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            model_name: Name of the model
            normalize: Whether to normalize values
            
        Returns:
            Path to saved plot
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix plot: {filepath}")
        
        return filepath
    
    def plot_roc_curves(
        self,
        roc_data: Dict[str, Dict],
        model_name: str
    ) -> Path:
        """
        Plot ROC curves for all classes.
        
        Args:
            roc_data: ROC data dictionary
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for class_name, data in roc_data.items():
            if class_name != 'micro_average':
                ax.plot(
                    data['fpr'],
                    data['tpr'],
                    label=f"{class_name} (AUC = {data['auc']:.3f})",
                    linewidth=2
                )
        
        # Plot micro-average
        if 'micro_average' in roc_data:
            ax.plot(
                roc_data['micro_average']['fpr'],
                roc_data['micro_average']['tpr'],
                label=f"Micro-average (AUC = {roc_data['micro_average']['auc']:.3f})",
                linewidth=3,
                linestyle='--',
                color='black'
            )
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {model_name}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'roc_curves_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curves plot: {filepath}")
        
        return filepath
    
    def plot_precision_recall_curves(
        self,
        pr_data: Dict[str, Dict],
        model_name: str
    ) -> Path:
        """
        Plot precision-recall curves for all classes.
        
        Args:
            pr_data: Precision-recall data dictionary
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for class_name, data in pr_data.items():
            ax.plot(
                data['recall'],
                data['precision'],
                label=f"{class_name} (AUC = {data['auc']:.3f})",
                linewidth=2
            )
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves - {model_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'pr_curves_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved PR curves plot: {filepath}")
        
        return filepath
    
    def plot_class_performance(
        self,
        per_class_metrics: Dict[str, Dict],
        model_name: str
    ) -> Path:
        """
        Plot per-class performance metrics.
        
        Args:
            per_class_metrics: Per-class metrics dictionary
            model_name: Name of the model
            
        Returns:
            Path to saved plot
        """
        classes = list(per_class_metrics.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        data = {metric: [per_class_metrics[cls][metric] for cls in classes]
                for metric in metrics}
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            offset = (i - 1) * width
            ax.bar(x + offset, data[metric], width, 
                  label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Performance - {model_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = f'class_performance_{model_name.lower().replace(" ", "_")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved class performance plot: {filepath}")
        
        return filepath
    
    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'accuracy'
    ) -> Path:
        """
        Plot comparison of multiple models.
        
        Args:
            comparison_df: DataFrame with model comparison data
            metric: Metric to compare
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = comparison_df['Model'].values
        values = comparison_df[metric.replace('_', ' ').title()].values
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, values, color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 0.01, i, f'{value:.4f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        filename = f'model_comparison_{metric}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot: {filepath}")
        
        return filepath


class ModelEvaluator:
    """Evaluates a single model and generates comprehensive metrics."""
    
    def __init__(
        self,
        model,
        model_name: str,
        class_names: List[str] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model instance
            model_name: Name of the model
            class_names: List of class names
        """
        self.model = model
        self.model_name = model_name
        self.class_names = class_names or ["Hate speech", "Offensive language", "Neither"]
        
        self.metrics_calc = MetricsCalculator(class_names=self.class_names)
        self.results = {}
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        training_time: float = None
    ) -> Dict[str, Any]:
        """
        Perform complete evaluation of the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            training_time: Training time in seconds (optional)
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info(f"Evaluating model: {self.model_name}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Get probabilities if available
        try:
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test)
            else:
                y_pred_proba = None
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
            y_pred_proba = None
        
        # Calculate metrics
        basic_metrics = self.metrics_calc.calculate_basic_metrics(y_test, y_pred)
        per_class_metrics = self.metrics_calc.calculate_per_class_metrics(y_test, y_pred)
        cm = self.metrics_calc.calculate_confusion_matrix(y_test, y_pred)
        
        # Calculate ROC and PR metrics if probabilities available
        roc_data = None
        pr_data = None
        if y_pred_proba is not None:
            try:
                roc_data = self.metrics_calc.calculate_roc_metrics(y_test, y_pred_proba)
                pr_data = self.metrics_calc.calculate_precision_recall_metrics(y_test, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR metrics: {e}")
        
        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
        
        # Compile results
        self.results = {
            'model_name': self.model_name,
            'basic_metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm.tolist(),
            'roc_data': roc_data,
            'pr_data': pr_data,
            'classification_report': report,
            'training_time': training_time,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(y_test)
        }
        
        logger.info(f"Evaluation complete for {self.model_name}")
        logger.info(f"  Accuracy: {basic_metrics['accuracy']:.4f}")
        logger.info(f"  F1 (Macro): {basic_metrics['f1_macro']:.4f}")
        
        return self.results
    
    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        if not self.results:
            logger.warning("No evaluation results available")
            return
        
        print("\n" + "=" * 80)
        print(f"MODEL EVALUATION SUMMARY: {self.model_name}")
        print("=" * 80)
        
        # Basic metrics
        print("\nOVERALL METRICS:")
        print("-" * 80)
        metrics = self.results['basic_metrics']
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1 Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted):{metrics['f1_weighted']:.4f}")
        print(f"MCC:                {metrics['mcc']:.4f}")
        print(f"Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
        
        if self.results['training_time']:
            print(f"\nTraining Time:      {self.results['training_time']:.2f}s")
        
        # Per-class metrics
        print("\nPER-CLASS METRICS:")
        print("-" * 80)
        for class_name, class_metrics in self.results['per_class_metrics'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1 Score:  {class_metrics['f1_score']:.4f}")
            print(f"  Support:   {class_metrics['support']}")
        
        # ROC AUC scores
        if self.results['roc_data']:
            print("\nROC AUC SCORES:")
            print("-" * 80)
            for class_name, data in self.results['roc_data'].items():
                print(f"{class_name}: {data['auc']:.4f}")
        
        # Classification report
        print("\nCLASSIFICATION REPORT:")
        print("-" * 80)
        print(self.results['classification_report'])
        
        print("=" * 80)


class ComprehensiveReportGenerator:
    """Generates comprehensive evaluation reports for all models."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        self.viz_gen = VisualizationGenerator(self.plots_dir)
        self.all_results = []
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        training_time: float = None,
        class_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model and generate visualizations.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            training_time: Training time in seconds
            class_names: List of class names
            
        Returns:
            Evaluation results dictionary
        """
        evaluator = ModelEvaluator(model, model_name, class_names)
        results = evaluator.evaluate(X_test, y_test, training_time)
        
        # Generate visualizations
        cm = np.array(results['confusion_matrix'])
        class_names = class_names or evaluator.class_names
        
        # Confusion matrices (regular and normalized)
        self.viz_gen.plot_confusion_matrix(cm, class_names, model_name, normalize=False)
        self.viz_gen.plot_confusion_matrix(cm, class_names, model_name, normalize=True)
        
        # ROC curves
        if results['roc_data']:
            self.viz_gen.plot_roc_curves(results['roc_data'], model_name)
        
        # Precision-recall curves
        if results['pr_data']:
            self.viz_gen.plot_precision_recall_curves(results['pr_data'], model_name)
        
        # Per-class performance
        self.viz_gen.plot_class_performance(results['per_class_metrics'], model_name)
        
        # Print summary
        evaluator.print_summary()
        
        # Store results
        self.all_results.append(results)
        
        return results
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Generate comparison report for all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.all_results:
            logger.warning("No models have been evaluated")
            return pd.DataFrame()
        
        logger.info("Generating model comparison report")
        
        comparison_data = []
        
        for result in self.all_results:
            metrics = result['basic_metrics']
            row = {
                'Model': result['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1 Score (Macro)': metrics['f1_macro'],
                'F1 Score (Weighted)': metrics['f1_weighted'],
                'MCC': metrics['mcc'],
                'Cohen Kappa': metrics['cohen_kappa']
            }
            
            if result['training_time']:
                row['Training Time (s)'] = result['training_time']
            
            # Add ROC AUC if available
            if result['roc_data'] and 'micro_average' in result['roc_data']:
                row['ROC AUC (Micro)'] = result['roc_data']['micro_average']['auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Save comparison table
        comparison_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Saved comparison table: {comparison_path}")
        
        # Generate comparison plots
        for metric in ['accuracy', 'f1_score_(macro)', 'precision_(macro)', 'recall_(macro)']:
            try:
                self.viz_gen.plot_model_comparison(comparison_df, metric)
            except Exception as e:
                logger.warning(f"Could not plot comparison for {metric}: {e}")
        
        return comparison_df
    
    def save_detailed_report(self) -> Path:
        """
        Save detailed JSON report with all results.
        
        Returns:
            Path to saved report
        """
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(self.all_results),
            'models': self.all_results
        }
        
        report_path = self.output_dir / 'detailed_report.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved detailed report: {report_path}")
        
        return report_path
    
    def generate_html_report(self) -> Path:
        """
        Generate HTML report with all visualizations.
        
        Returns:
            Path to HTML report
        """
        html_content = self._create_html_content()
        
        html_path = self.output_dir / 'evaluation_report.html'
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML report: {html_path}")
        
        return html_path
    
    def _create_html_content(self) -> str:
        """Create HTML content for the report."""
        if not self.all_results:
            return "<html><body><h1>No evaluation results available</h1></body></html>"
        
        # Get comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Accuracy': r['basic_metrics']['accuracy'],
                'F1 (Macro)': r['basic_metrics']['f1_macro'],
                'Precision': r['basic_metrics']['precision_macro'],
                'Recall': r['basic_metrics']['recall_macro']
            }
            for r in self.all_results
        ]).sort_values('Accuracy', ascending=False)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metric-value {{
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .best-model {{
                    background-color: #2ecc71;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 5px;
                }}
                .plot-section {{
                    margin: 30px 0;
                }}
                .plot-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .plot-item {{
                    text-align: center;
                }}
                .plot-item img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Evaluation Report</h1>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Executive Summary</h2>
                <p>Total Models Evaluated: <span class="metric-value">{len(self.all_results)}</span></p>
                <p>Best Model: <span class="best-model">{comparison_df.iloc[0]['Model']}</span></p>
                <p>Best Accuracy: <span class="metric-value">{comparison_df.iloc[0]['Accuracy']:.4f}</span></p>
                
                <h2>Model Comparison</h2>
                {comparison_df.to_html(index=False, classes='comparison-table')}
                
        """
        
        # Add visualizations for each model
        for result in self.all_results:
            model_name = result['model_name']
            model_safe_name = model_name.lower().replace(' ', '_')
            
            html += f"""
                <h2>Detailed Results: {model_name}</h2>
                
                <h3>Classification Report</h3>
                <pre>{result['classification_report']}</pre>
                
                <div class="plot-section">
                    <h3>Visualizations</h3>
                    <div class="plot-grid">
                        <div class="plot-item">
                            <h4>Confusion Matrix</h4>
                            <img src="plots/confusion_matrix_{model_safe_name}.png" 
                                 alt="Confusion Matrix">
                        </div>
                        <div class="plot-item">
                            <h4>Normalized Confusion Matrix</h4>
                            <img src="plots/confusion_matrix_{model_safe_name}.png" 
                                 alt="Normalized Confusion Matrix">
                        </div>
            """
            
            if result['roc_data']:
                html += f"""
                        <div class="plot-item">
                            <h4>ROC Curves</h4>
                            <img src="plots/roc_curves_{model_safe_name}.png" 
                                 alt="ROC Curves">
                        </div>
                """
            
            if result['pr_data']:
                html += f"""
                        <div class="plot-item">
                            <h4>Precision-Recall Curves</h4>
                            <img src="plots/pr_curves_{model_safe_name}.png" 
                                 alt="Precision-Recall Curves">
                        </div>
                """
            
            html += f"""
                        <div class="plot-item">
                            <h4>Per-Class Performance</h4>
                            <img src="plots/class_performance_{model_safe_name}.png" 
                                 alt="Class Performance">
                        </div>
                    </div>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html


def generate_complete_report(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    training_times: Dict[str, float] = None,
    output_dir: Path = None,
    class_names: List[str] = None
) -> Tuple[pd.DataFrame, Path]:
    """
    Generate complete evaluation report for all models.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
        X_test: Test features
        y_test: Test labels
        training_times: Dictionary mapping model names to training times
        output_dir: Output directory for reports
        class_names: List of class names
        
    Returns:
        Tuple of (comparison_df, html_report_path)
    """
    if output_dir is None:
        output_dir = Path('evaluation_reports') / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir = Path(output_dir)
    
    logger.info(f"Generating comprehensive evaluation report in {output_dir}")
    
    training_times = training_times or {}
    
    report_gen = ComprehensiveReportGenerator(output_dir)
    
    # Evaluate each model
    for model_name, model in models_dict.items():
        logger.info(f"\n{'='*80}\nEvaluating: {model_name}\n{'='*80}")
        
        training_time = training_times.get(model_name)
        
        try:
            report_gen.evaluate_model(
                model=model,
                model_name=model_name,
                X_test=X_test,
                y_test=y_test,
                training_time=training_time,
                class_names=class_names
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    # Generate comparison report
    comparison_df = report_gen.generate_comparison_report()
    
    # Save detailed JSON report
    report_gen.save_detailed_report()
    
    # Generate HTML report
    html_path = report_gen.generate_html_report()
    
    logger.info(f"\n{'='*80}")
    logger.info("REPORT GENERATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"HTML report: {html_path}")
    logger.info(f"Comparison CSV: {output_dir / 'model_comparison.csv'}")
    logger.info(f"Detailed JSON: {output_dir / 'detailed_report.json'}")
    logger.info(f"Plots directory: {output_dir / 'plots'}")
    
    return comparison_df, html_path


# Example usage script
if __name__ == "__main__":
    """
    Example script showing how to use the report generator.
    """
    
    print("Model Evaluation Report Generator")
    print("=" * 80)
    
    # This is an example - replace with your actual model loading code
    try:
        # Load your models
        from pathlib import Path
        import joblib
        
        models_dir = Path('saved_models/traditional_ml')
        
        # Load models (example)
        models = {}
        model_files = {
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl',
            'SVM': 'svm.pkl',
            'MLP': 'neural_network.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = models_dir / filename
            if filepath.exists():
                models[name] = joblib.load(filepath)
                print(f"Loaded: {name}")
        
        # Load test data (example)
        processed_dir = Path('data/processed')
        test_data = joblib.load(processed_dir / 'test_data.pkl')
        X_test = test_data['X']
        y_test = test_data['y']
        
        # Load feature extractor and transform test data
        from pathlib import Path
        feature_extractor = joblib.load(Path('saved_features/feature_extractor.pkl'))
        X_test_features = feature_extractor.transform(X_test)
        
        print(f"\nTest set size: {len(y_test)} samples")
        
        # Generate comprehensive report
        comparison_df, html_path = generate_complete_report(
            models_dict=models,
            X_test=X_test_features,
            y_test=y_test,
            class_names=["Hate speech", "Offensive language", "Neither"]
        )
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        print(f"\nOpen the HTML report: {html_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Trained models in saved_models/")
        print("2. Test data in data/processed/")
        print("3. Feature extractor in saved_features/")


# After training your models
# from report_generator import generate_complete_report

# comparison_df, html_path = generate_complete_report(
#     models_dict={
#         'Random Forest': rf_model,
#         'XGBoost': xgb_model,
#         'SVM': svm_model
#     },
#     X_test=X_test_features,
#     y_test=y_test,
#     output_dir='evaluation_reports/2024_01_01'
# )