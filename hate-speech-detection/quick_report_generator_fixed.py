"""
Quick Report Generator - Training Report Creation (FIXED VERSION)
Generates HTML reports with embedded visualizations for training runs.
Handles missing/None values gracefully.
"""

import io
import base64
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import json

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class QuickReportGenerator:
    """
    Generates professional HTML reports with embedded visualizations.
    
    Features:
    - Executive summary with key metrics
    - Dataset analysis and class distribution
    - Model performance comparison
    - Historical comparison (if previous runs exist)
    - Actionable recommendations
    - Embedded base64 charts (no separate files)
    - Handles missing/None values gracefully
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports (default: results/training_reports)
        """
        self.output_dir = output_dir or Path('results/training_reports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup seaborn style
        if HAS_PLOTTING:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
    
    def generate_report(
        self,
        metrics: Dict,
        previous_metrics: Optional[Dict] = None,
        report_name: Optional[str] = None
    ) -> Path:
        """
        Generate complete HTML report.
        
        Args:
            metrics: Current training metrics from MetricsCollector
            previous_metrics: Previous run metrics for comparison (optional)
            report_name: Custom report filename (default: auto-generated)
        
        Returns:
            Path to generated report
        """
        # Generate report filename
        if report_name is None:
            run_id = metrics.get('run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
            report_name = f"training_report_{run_id}.html"
        
        report_path = self.output_dir / report_name
        
        # Build HTML sections
        html_parts = [
            self._html_header(),
            self._section_executive_summary(metrics, previous_metrics),
            self._section_dataset_analysis(metrics),
            self._section_data_splits(metrics),
            self._section_model_performance(metrics),
        ]
        
        # Add historical comparison if previous metrics available
        if previous_metrics:
            html_parts.append(self._section_historical_comparison(metrics, previous_metrics))
        
        html_parts.extend([
            self._section_recommendations(metrics, previous_metrics),
            self._html_footer()
        ])
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))
        
        return report_path
    
    def _html_header(self) -> str:
        """Generate HTML header with CSS."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - Hate Speech Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        .section {
            padding: 30px;
            border-bottom: 1px solid #eee;
        }
        .section:last-child { border-bottom: none; }
        .section h2 {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #667eea;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }
        .metric-card .label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-card .value.na {
            font-size: 1.5em;
            color: #999;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .comparison-table th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }
        .comparison-table tr:hover {
            background: #f8f9fa;
        }
        .improvement {
            color: #28a745;
            font-weight: bold;
        }
        .decline {
            color: #dc3545;
            font-weight: bold;
        }
        .recommendations {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .recommendations ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        .recommendations li {
            margin: 8px 0;
        }
        .warning-box {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            color: #721c24;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
"""
    
    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return """    </div>
    <div class="footer">
        <p>Generated by Hate Speech Detection Training System</p>
        <p>Report generated on {timestamp}</p>
    </div>
</body>
</html>""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    def _section_executive_summary(
        self,
        metrics: Dict,
        previous_metrics: Optional[Dict]
    ) -> str:
        """Generate executive summary section."""
        run_id = metrics.get('run_id', 'N/A')
        timestamp = metrics.get('timestamp', 'N/A')
        best_model = metrics.get('best_model')
        best_accuracy = metrics.get('best_accuracy', 0)
        total_samples = metrics.get('total_samples', 0)
        duration = metrics.get('duration_seconds', 0)
        
        # Handle None or missing best_model
        if best_model is None or best_model == '':
            best_model_display = 'N/A'
            best_model_class = 'na'
            warning_message = """
            <div class="warning-box">
                <strong>Warning:</strong> No best model found. This usually means model training 
                didn't complete properly or metrics weren't collected. Check your training logs.
            </div>
"""
        else:
            best_model_display = best_model.upper()
            best_model_class = ''
            warning_message = ''
        
        # Format accuracy
        if best_accuracy > 0:
            accuracy_display = f"{best_accuracy * 100:.2f}%"
        else:
            accuracy_display = 'N/A'
            best_model_class = 'na'
        
        html = f"""
        <div class="header">
            <h1>Training Report</h1>
            <p>Run ID: {run_id} | {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            {warning_message}
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">Best Model</div>
                    <div class="value {best_model_class}">{best_model_display}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Best Accuracy</div>
                    <div class="value {best_model_class}">{accuracy_display}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Total Samples</div>
                    <div class="value">{total_samples:,}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Training Time</div>
                    <div class="value">{duration:.1f}s</div>
                </div>
            </div>
"""
        
        if previous_metrics and best_accuracy > 0:
            prev_accuracy = previous_metrics.get('best_accuracy', 0) * 100
            curr_accuracy = best_accuracy * 100
            diff = curr_accuracy - prev_accuracy
            change_class = 'improvement' if diff > 0 else 'decline'
            change_text = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
            
            html += f"""
            <p style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 6px;">
                <strong>Comparison to Previous Run:</strong>
                <span class="{change_class}"> {change_text}</span> change in accuracy
            </p>
"""
        
        html += "        </div>\n"
        return html
    
    def _section_dataset_analysis(self, metrics: Dict) -> str:
        """Generate dataset analysis section with class distribution chart."""
        datasets = metrics.get('datasets', [])
        new_datasets = metrics.get('new_datasets', [])
        class_dist = metrics.get('class_distribution', {})
        
        html = f"""
        <div class="section">
            <h2>Dataset Analysis</h2>
            <p><strong>Datasets Used ({len(datasets)}):</strong></p>
            <ul style="margin-left: 30px; margin-top: 10px;">
"""
        
        for ds in datasets:
            marker = " (NEW)" if ds in new_datasets else ""
            html += f"                <li>{ds}{marker}</li>\n"
        
        html += "            </ul>\n"
        
        # Add class distribution chart
        if HAS_PLOTTING and class_dist:
            chart = self._create_class_distribution_chart(class_dist)
            if chart:
                html += f"""
            <div class="chart-container">
                <h3>Class Distribution</h3>
                <img src="data:image/png;base64,{chart}" alt="Class Distribution">
            </div>
"""
        
        html += "        </div>\n"
        return html
    
    def _section_data_splits(self, metrics: Dict) -> str:
        """Generate data splits section with visualization."""
        train_samples = metrics.get('train_samples', 0)
        val_samples = metrics.get('val_samples', 0)
        test_samples = metrics.get('test_samples', 0)
        total = train_samples + val_samples + test_samples
        
        train_pct = (train_samples / total * 100) if total > 0 else 0
        val_pct = (val_samples / total * 100) if total > 0 else 0
        test_pct = (test_samples / total * 100) if total > 0 else 0
        
        html = f"""
        <div class="section">
            <h2>Train/Validation/Test Split</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="label">Training Set</div>
                    <div class="value">{train_samples:,}</div>
                    <div class="label">{train_pct:.1f}% of total</div>
                </div>
                <div class="metric-card">
                    <div class="label">Validation Set</div>
                    <div class="value">{val_samples:,}</div>
                    <div class="label">{val_pct:.1f}% of total</div>
                </div>
                <div class="metric-card">
                    <div class="label">Test Set</div>
                    <div class="value">{test_samples:,}</div>
                    <div class="label">{test_pct:.1f}% of total</div>
                </div>
            </div>
"""
        
        # Add split visualization
        if HAS_PLOTTING and total > 0:
            chart = self._create_split_chart(train_samples, val_samples, test_samples)
            if chart:
                html += f"""
            <div class="chart-container">
                <img src="data:image/png;base64,{chart}" alt="Data Split">
            </div>
"""
        
        html += "        </div>\n"
        return html
    
    def _section_model_performance(self, metrics: Dict) -> str:
        """Generate model performance comparison section."""
        models = metrics.get('models', {})
        
        html = """
        <div class="section">
            <h2>Model Performance Comparison</h2>
"""
        
        if not models:
            html += """
            <div class="warning-box">
                <strong>Warning:</strong> No model performance data available. 
                This means model metrics weren't collected during training.
                Check your training script's model metrics collection code.
            </div>
        </div>
"""
            return html
        
        html += """
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>F1 (Macro)</th>
                        <th>Precision (Macro)</th>
                        <th>Recall (Macro)</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Sort models by accuracy
        sorted_models = sorted(
            models.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        for model_name, model_metrics in sorted_models:
            accuracy = model_metrics.get('accuracy', 0) * 100
            f1 = model_metrics.get('f1_macro', 0) * 100
            precision = model_metrics.get('precision_macro', 0) * 100
            recall = model_metrics.get('recall_macro', 0) * 100
            
            html += f"""
                    <tr>
                        <td><strong>{model_name.upper()}</strong></td>
                        <td>{accuracy:.2f}%</td>
                        <td>{f1:.2f}%</td>
                        <td>{precision:.2f}%</td>
                        <td>{recall:.2f}%</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
"""
        
        # Add performance comparison chart
        if HAS_PLOTTING and models:
            chart = self._create_model_comparison_chart(models)
            if chart:
                html += f"""
            <div class="chart-container">
                <h3>Model Performance Comparison</h3>
                <img src="data:image/png;base64,{chart}" alt="Model Comparison">
            </div>
"""
        
        html += "        </div>\n"
        return html
    
    def _section_historical_comparison(
        self,
        current: Dict,
        previous: Dict
    ) -> str:
        """Generate historical comparison section."""
        html = """
        <div class="section">
            <h2>Historical Comparison</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Previous Run</th>
                        <th>Current Run</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Compare key metrics
        comparisons = [
            ('Best Accuracy', 'best_accuracy', True),
            ('Total Samples', 'total_samples', False),
            ('Training Time (s)', 'duration_seconds', False),
        ]
        
        for label, key, is_percentage in comparisons:
            prev_val = previous.get(key, 0)
            curr_val = current.get(key, 0)
            
            if is_percentage:
                prev_display = f"{prev_val * 100:.2f}%" if prev_val else 'N/A'
                curr_display = f"{curr_val * 100:.2f}%" if curr_val else 'N/A'
                if prev_val and curr_val:
                    diff = (curr_val - prev_val) * 100
                    diff_display = f"{diff:+.2f}%"
                else:
                    diff = 0
                    diff_display = 'N/A'
            else:
                prev_display = f"{prev_val:,.1f}" if prev_val else 'N/A'
                curr_display = f"{curr_val:,.1f}" if curr_val else 'N/A'
                if prev_val and curr_val:
                    diff = curr_val - prev_val
                    diff_display = f"{diff:+,.1f}"
                else:
                    diff = 0
                    diff_display = 'N/A'
            
            change_class = 'improvement' if diff > 0 else 'decline' if diff < 0 else ''
            
            html += f"""
                    <tr>
                        <td><strong>{label}</strong></td>
                        <td>{prev_display}</td>
                        <td>{curr_display}</td>
                        <td class="{change_class}">{diff_display}</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
"""
        return html
    
    def _section_recommendations(
        self,
        metrics: Dict,
        previous_metrics: Optional[Dict]
    ) -> str:
        """Generate recommendations section."""
        recommendations = self._generate_recommendations(metrics, previous_metrics)
        
        html = """
        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <strong>Actionable Next Steps:</strong>
                <ul>
"""
        
        for rec in recommendations:
            html += f"                    <li>{rec}</li>\n"
        
        html += """
                </ul>
            </div>
        </div>
"""
        return html
    
    def _generate_recommendations(
        self,
        metrics: Dict,
        previous_metrics: Optional[Dict]
    ) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        best_model = metrics.get('best_model')
        best_accuracy = metrics.get('best_accuracy', 0)
        total_samples = metrics.get('total_samples', 0)
        models = metrics.get('models', {})
        
        # Check if training completed properly
        if not best_model or not models:
            recommendations.append(
                "CRITICAL: Model training did not complete properly. "
                "Check your training script to ensure model metrics are being collected correctly."
            )
            recommendations.append(
                "Verify that 'collector.add_model_result()' is being called for each trained model."
            )
            return recommendations
        
        # Accuracy-based recommendations
        if best_accuracy < 0.80:
            recommendations.append(
                "Consider collecting more diverse training data to improve model accuracy."
            )
            recommendations.append(
                "Experiment with feature engineering or different embeddings."
            )
        elif best_accuracy >= 0.90:
            recommendations.append(
                "Excellent accuracy achieved! Consider deploying this model to production."
            )
        
        # Sample size recommendations
        if total_samples < 10000:
            recommendations.append(
                "Dataset is relatively small. Consider augmenting with additional data sources."
            )
        
        # Historical comparison recommendations
        if previous_metrics:
            prev_accuracy = previous_metrics.get('best_accuracy', 0)
            if prev_accuracy and best_accuracy < prev_accuracy - 0.02:
                recommendations.append(
                    "Performance declined from previous run. Verify data quality and feature consistency."
                )
            elif prev_accuracy and best_accuracy > prev_accuracy + 0.02:
                recommendations.append(
                    "Significant improvement detected! Document changes that led to this improvement."
                )
        
        # Model-specific recommendations
        if best_model == 'xgboost':
            recommendations.append(
                "XGBoost performed best. Consider hyperparameter tuning for further gains."
            )
        
        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Continue monitoring model performance and collecting diverse training examples."
            )
        
        return recommendations
    
    def _create_class_distribution_chart(self, class_dist: Dict) -> str:
        """Create class distribution bar chart as base64 string."""
        if not HAS_PLOTTING:
            return ""
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
            labels = [class_labels.get(int(k), f"Class {k}") for k in class_dist.keys()]
            values = list(class_dist.values())
            
            colors = sns.color_palette("husl", len(labels))
            bars = ax.bar(labels, values, color=colors, alpha=0.8)
            
            ax.set_ylabel('Number of Samples', fontsize=12)
            ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{int(height):,}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"Warning: Could not create class distribution chart: {e}")
            return ""
    
    def _create_split_chart(
        self,
        train: int,
        val: int,
        test: int
    ) -> str:
        """Create data split pie chart as base64 string."""
        if not HAS_PLOTTING:
            return ""
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            sizes = [train, val, test]
            labels = ['Training', 'Validation', 'Test']
            colors = sns.color_palette("pastel")[0:3]
            explode = (0.05, 0.05, 0.05)
            
            ax.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90
            )
            ax.axis('equal')
            ax.set_title('Data Split Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"Warning: Could not create split chart: {e}")
            return ""
    
    def _create_model_comparison_chart(self, models: Dict) -> str:
        """Create model performance comparison chart as base64 string."""
        if not HAS_PLOTTING:
            return ""
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            model_names = list(models.keys())
            accuracies = [m.get('accuracy', 0) * 100 for m in models.values()]
            f1_scores = [m.get('f1_macro', 0) * 100 for m in models.values()]
            
            x = range(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(
                [i - width/2 for i in x],
                accuracies,
                width,
                label='Accuracy',
                alpha=0.8
            )
            bars2 = ax.bar(
                [i + width/2 for i in x],
                f1_scores,
                width,
                label='F1 Score (Macro)',
                alpha=0.8
            )
            
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel('Score (%)', fontsize=12)
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in model_names])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 100])
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return image_base64
        except Exception as e:
            print(f"Warning: Could not create model comparison chart: {e}")
            return ""