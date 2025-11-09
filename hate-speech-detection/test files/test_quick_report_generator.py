"""
Test Suite for QuickReportGenerator
Tests HTML report generation with various scenarios.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reporting.quick_report_generator import QuickReportGenerator


class TestQuickReportGenerator(unittest.TestCase):
    """Test cases for QuickReportGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = Path('test_reports')
        self.test_output_dir.mkdir(exist_ok=True)
        
        self.generator = QuickReportGenerator(output_dir=self.test_output_dir)
        
        # Sample metrics for testing
        self.sample_metrics = {
            'run_id': '20250124_143000',
            'timestamp': '2025-01-24 14:30:00',
            'datasets': ['labeled_data.csv', 'dataset2.csv'],
            'new_datasets': ['dataset2.csv'],
            'total_samples': 45230,
            'train_samples': 31661,
            'val_samples': 6784,
            'test_samples': 6785,
            'class_distribution': {'0': 11734, '1': 18523, '2': 14973},
            'models': {
                'svm': {
                    'accuracy': 0.8712,
                    'f1_macro': 0.8445,
                    'precision_macro': 0.8534,
                    'recall_macro': 0.8398
                },
                'xgboost': {
                    'accuracy': 0.8789,
                    'f1_macro': 0.8556,
                    'precision_macro': 0.8623,
                    'recall_macro': 0.8512
                },
                'random_forest': {
                    'accuracy': 0.8634,
                    'f1_macro': 0.8401,
                    'precision_macro': 0.8478,
                    'recall_macro': 0.8356
                }
            },
            'best_model': 'xgboost',
            'best_accuracy': 0.8789,
            'feature_dimensions': 5234,
            'duration_seconds': 389.2
        }
        
        self.previous_metrics = {
            'run_id': '20250123_120000',
            'timestamp': '2025-01-23 12:00:00',
            'best_accuracy': 0.8512,
            'total_samples': 38450,
            'duration_seconds': 345.8
        }
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertTrue(self.test_output_dir.exists())
    
    def test_generate_basic_report(self):
        """Test generating a basic report without previous metrics."""
        report_path = self.generator.generate_report(self.sample_metrics)
        
        self.assertTrue(report_path.exists())
        self.assertTrue(report_path.name.startswith('training_report_'))
        self.assertTrue(report_path.name.endswith('.html'))
        
        # Read and verify content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key sections
        self.assertIn('<!DOCTYPE html>', content)
        self.assertIn('Executive Summary', content)
        self.assertIn('Dataset Analysis', content)
        self.assertIn('Train/Validation/Test Split', content)
        self.assertIn('Model Performance Comparison', content)
        self.assertIn('Recommendations', content)
        
        # Check for metrics
        self.assertIn('XGBOOST', content)
        self.assertIn('87.89%', content)  # Best accuracy
        self.assertIn('45,230', content)  # Total samples
    
    def test_generate_report_with_comparison(self):
        """Test generating report with historical comparison."""
        report_path = self.generator.generate_report(
            self.sample_metrics,
            previous_metrics=self.previous_metrics
        )
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for comparison section
        self.assertIn('Historical Comparison', content)
        self.assertIn('Previous Run', content)
        self.assertIn('Current Run', content)
        self.assertIn('Change', content)
    
    def test_custom_report_name(self):
        """Test generating report with custom name."""
        custom_name = 'custom_test_report.html'
        report_path = self.generator.generate_report(
            self.sample_metrics,
            report_name=custom_name
        )
        
        self.assertEqual(report_path.name, custom_name)
        self.assertTrue(report_path.exists())
    
    def test_html_header(self):
        """Test HTML header generation."""
        header = self.generator._html_header()
        
        self.assertIn('<!DOCTYPE html>', header)
        self.assertIn('<html lang="en">', header)
        self.assertIn('<style>', header)
        self.assertIn('body {', header)
        self.assertIn('.container {', header)
    
    def test_html_footer(self):
        """Test HTML footer generation."""
        footer = self.generator._html_footer()
        
        self.assertIn('</body>', footer)
        self.assertIn('</html>', footer)
        self.assertIn('Generated by', footer)
        self.assertIn('Report generated on', footer)
    
    def test_executive_summary_section(self):
        """Test executive summary section generation."""
        section = self.generator._section_executive_summary(
            self.sample_metrics,
            None
        )
        
        self.assertIn('Executive Summary', section)
        self.assertIn('XGBOOST', section)
        self.assertIn('87.89%', section)
        self.assertIn('45,230', section)
        self.assertIn('389.2s', section)
    
    def test_executive_summary_with_comparison(self):
        """Test executive summary with previous metrics."""
        section = self.generator._section_executive_summary(
            self.sample_metrics,
            self.previous_metrics
        )
        
        self.assertIn('Comparison to Previous Run', section)
        # Check for improvement indicator
        self.assertIn('+2.77%', section)  # 87.89 - 85.12
        self.assertIn('improvement', section)
    
    def test_dataset_analysis_section(self):
        """Test dataset analysis section generation."""
        section = self.generator._section_dataset_analysis(self.sample_metrics)
        
        self.assertIn('Dataset Analysis', section)
        self.assertIn('labeled_data.csv', section)
        self.assertIn('dataset2.csv', section)
        self.assertIn('(NEW)', section)
    
    def test_data_splits_section(self):
        """Test data splits section generation."""
        section = self.generator._section_data_splits(self.sample_metrics)
        
        self.assertIn('Train/Validation/Test Split', section)
        self.assertIn('31,661', section)  # Train
        self.assertIn('6,784', section)   # Val
        self.assertIn('6,785', section)   # Test
        self.assertIn('70.0%', section)   # Train percentage
    
    def test_model_performance_section(self):
        """Test model performance section generation."""
        section = self.generator._section_model_performance(self.sample_metrics)
        
        self.assertIn('Model Performance Comparison', section)
        self.assertIn('SVM', section)
        self.assertIn('XGBOOST', section)
        self.assertIn('RANDOM_FOREST', section)
        self.assertIn('87.12%', section)  # SVM accuracy
        self.assertIn('87.89%', section)  # XGBoost accuracy
    
    def test_historical_comparison_section(self):
        """Test historical comparison section generation."""
        section = self.generator._section_historical_comparison(
            self.sample_metrics,
            self.previous_metrics
        )
        
        self.assertIn('Historical Comparison', section)
        self.assertIn('Previous Run', section)
        self.assertIn('Current Run', section)
        self.assertIn('85.12%', section)  # Previous accuracy
        self.assertIn('87.89%', section)  # Current accuracy
        self.assertIn('+2.77%', section)  # Change
    
    def test_recommendations_section(self):
        """Test recommendations section generation."""
        section = self.generator._section_recommendations(
            self.sample_metrics,
            None
        )
        
        self.assertIn('Recommendations', section)
        self.assertIn('Actionable Next Steps', section)
        self.assertIn('<li>', section)
    
    def test_generate_recommendations_high_accuracy(self):
        """Test recommendations for high accuracy scenario."""
        high_acc_metrics = self.sample_metrics.copy()
        high_acc_metrics['best_accuracy'] = 0.92
        
        recommendations = self.generator._generate_recommendations(
            high_acc_metrics,
            None
        )
        
        self.assertTrue(any('production' in r.lower() for r in recommendations))
    
    def test_generate_recommendations_low_accuracy(self):
        """Test recommendations for low accuracy scenario."""
        low_acc_metrics = self.sample_metrics.copy()
        low_acc_metrics['best_accuracy'] = 0.75
        
        recommendations = self.generator._generate_recommendations(
            low_acc_metrics,
            None
        )
        
        self.assertTrue(any('data' in r.lower() for r in recommendations))
    
    def test_generate_recommendations_small_dataset(self):
        """Test recommendations for small dataset."""
        small_data_metrics = self.sample_metrics.copy()
        small_data_metrics['total_samples'] = 5000
        
        recommendations = self.generator._generate_recommendations(
            small_data_metrics,
            None
        )
        
        self.assertTrue(any('dataset' in r.lower() for r in recommendations))
    
    def test_generate_recommendations_performance_decline(self):
        """Test recommendations for performance decline."""
        decline_metrics = self.sample_metrics.copy()
        decline_metrics['best_accuracy'] = 0.82
        
        high_prev_metrics = self.previous_metrics.copy()
        high_prev_metrics['best_accuracy'] = 0.88
        
        recommendations = self.generator._generate_recommendations(
            decline_metrics,
            high_prev_metrics
        )
        
        self.assertTrue(any('declined' in r.lower() for r in recommendations))
    
    def test_minimal_metrics(self):
        """Test report generation with minimal metrics."""
        minimal_metrics = {
            'run_id': 'test_run',
            'timestamp': datetime.now().isoformat(),
            'datasets': ['test.csv'],
            'total_samples': 1000,
            'train_samples': 700,
            'val_samples': 150,
            'test_samples': 150,
            'class_distribution': {'0': 400, '1': 400, '2': 200},
            'models': {
                'test_model': {
                    'accuracy': 0.80,
                    'f1_macro': 0.78,
                    'precision_macro': 0.79,
                    'recall_macro': 0.77
                }
            },
            'best_model': 'test_model',
            'best_accuracy': 0.80,
            'duration_seconds': 100.0
        }
        
        report_path = self.generator.generate_report(minimal_metrics)
        self.assertTrue(report_path.exists())
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn('test_run', content)
        self.assertIn('1,000', content)
        self.assertIn('TEST_MODEL', content)
    
    def test_chart_generation_class_distribution(self):
        """Test class distribution chart generation."""
        try:
            chart_base64 = self.generator._create_class_distribution_chart(
                self.sample_metrics['class_distribution']
            )
            # If plotting available, should return non-empty string
            # If not available, returns empty string
            self.assertIsInstance(chart_base64, str)
        except Exception as e:
            self.fail(f"Chart generation raised exception: {e}")
    
    def test_chart_generation_split(self):
        """Test split chart generation."""
        try:
            chart_base64 = self.generator._create_split_chart(
                self.sample_metrics['train_samples'],
                self.sample_metrics['val_samples'],
                self.sample_metrics['test_samples']
            )
            self.assertIsInstance(chart_base64, str)
        except Exception as e:
            self.fail(f"Chart generation raised exception: {e}")
    
    def test_chart_generation_model_comparison(self):
        """Test model comparison chart generation."""
        try:
            chart_base64 = self.generator._create_model_comparison_chart(
                self.sample_metrics['models']
            )
            self.assertIsInstance(chart_base64, str)
        except Exception as e:
            self.fail(f"Chart generation raised exception: {e}")
    
    def test_multiple_reports_generation(self):
        """Test generating multiple reports."""
        report1 = self.generator.generate_report(
            self.sample_metrics,
            report_name='report1.html'
        )
        
        report2 = self.generator.generate_report(
            self.sample_metrics,
            report_name='report2.html'
        )
        
        self.assertTrue(report1.exists())
        self.assertTrue(report2.exists())
        self.assertNotEqual(report1, report2)


def run_tests():
    """Run all tests and print results."""
    print("=" * 70)
    print("QUICK REPORT GENERATOR TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQuickReportGenerator)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)