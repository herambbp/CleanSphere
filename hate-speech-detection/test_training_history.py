"""
Test Script for TrainingHistory Module
Tests all functionality of the training history system
"""

import sys
import os
import json
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reporting.training_history import TrainingHistory


def create_sample_run_data(run_number: int):
    """Create sample run data for testing"""
    return {
        'run_id': f'2025012{run_number}_143000',
        'timestamp': f'2025-01-2{run_number} 14:30:00',
        'datasets': ['labeled_data.csv'] if run_number == 1 else ['labeled_data.csv', 'dataset2.csv'],
        'new_datasets': [] if run_number == 1 else ['dataset2.csv'],
        'total_samples': 32780 if run_number == 1 else 45230,
        'train_samples': 22946 if run_number == 1 else 31661,
        'val_samples': 4917 if run_number == 1 else 6784,
        'test_samples': 4917 if run_number == 1 else 6785,
        'class_distribution': {
            '0': 9834 if run_number == 1 else 11734,
            '1': 14123 if run_number == 1 else 18523,
            '2': 8823 if run_number == 1 else 14973
        },
        'models': {
            'svm': {
                'accuracy': 0.8534 if run_number == 1 else 0.8712,
                'f1_macro': 0.8201 if run_number == 1 else 0.8445,
                'precision_macro': 0.8312 if run_number == 1 else 0.8556,
                'recall_macro': 0.8156 if run_number == 1 else 0.8389
            },
            'xgboost': {
                'accuracy': 0.8612 if run_number == 1 else 0.8789,
                'f1_macro': 0.8334 if run_number == 1 else 0.8556,
                'precision_macro': 0.8421 if run_number == 1 else 0.8667,
                'recall_macro': 0.8267 if run_number == 1 else 0.8478
            }
        },
        'best_model': 'xgboost',
        'best_accuracy': 0.8612 if run_number == 1 else 0.8789,
        'feature_dimensions': 5234,
        'duration_seconds': 342.5 if run_number == 1 else 389.2
    }


def test_initialization():
    """Test TrainingHistory initialization"""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        # Test with non-existent file
        history = TrainingHistory(temp_file)
        assert history.get_run_count() == 0, "Should start with 0 runs"
        print("[PASS] Initialization with non-existent file")
        
        # Test file creation
        assert Path(temp_file).parent.exists(), "Parent directory should be created"
        print("[PASS] Parent directory created")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 1 completed successfully")


def test_add_and_retrieve_runs():
    """Test adding and retrieving runs"""
    print("\n" + "="*60)
    print("TEST 2: Add and Retrieve Runs")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        history = TrainingHistory(temp_file)
        
        # Add first run
        run1 = create_sample_run_data(1)
        history.add_run(run1)
        
        assert history.get_run_count() == 1, "Should have 1 run"
        print("[PASS] Added first run")
        
        # Retrieve last run
        last_run = history.get_last_run()
        assert last_run is not None, "Should retrieve last run"
        assert last_run['run_id'] == run1['run_id'], "Should match run ID"
        print("[PASS] Retrieved last run correctly")
        
        # Add second run
        run2 = create_sample_run_data(2)
        history.add_run(run2)
        
        assert history.get_run_count() == 2, "Should have 2 runs"
        print("[PASS] Added second run")
        
        # Test get_all_runs
        all_runs = history.get_all_runs()
        assert len(all_runs) == 2, "Should return all runs"
        print("[PASS] Retrieved all runs")
        
        # Test get_last_n_runs
        last_2 = history.get_last_n_runs(2)
        assert len(last_2) == 2, "Should return last 2 runs"
        print("[PASS] Retrieved last N runs")
        
        # Test get_run_by_id
        retrieved = history.get_run_by_id(run1['run_id'])
        assert retrieved is not None, "Should find run by ID"
        assert retrieved['run_id'] == run1['run_id'], "Should match run ID"
        print("[PASS] Retrieved run by ID")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 2 completed successfully")


def test_comparison():
    """Test run comparison functionality"""
    print("\n" + "="*60)
    print("TEST 3: Run Comparison")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        history = TrainingHistory(temp_file)
        
        # Test comparison with no runs
        comparison = history.compare_last_two()
        assert comparison is None, "Should return None with no runs"
        print("[PASS] Comparison with no runs returns None")
        
        # Add first run
        run1 = create_sample_run_data(1)
        history.add_run(run1)
        
        # Test comparison with only one run
        comparison = history.compare_last_two()
        assert comparison is None, "Should return None with only 1 run"
        print("[PASS] Comparison with 1 run returns None")
        
        # Add second run
        run2 = create_sample_run_data(2)
        history.add_run(run2)
        
        # Test comparison with two runs
        comparison = history.compare_last_two()
        assert comparison is not None, "Should return comparison"
        print("[PASS] Comparison with 2 runs returns data")
        
        # Check comparison structure
        assert 'previous_run' in comparison, "Should have previous_run"
        assert 'current_run' in comparison, "Should have current_run"
        assert 'datasets_added' in comparison, "Should have datasets_added"
        assert 'samples_delta' in comparison, "Should have samples_delta"
        assert 'accuracy_delta' in comparison, "Should have accuracy_delta"
        assert 'improved' in comparison, "Should have improved flag"
        print("[PASS] Comparison has all required fields")
        
        # Check specific values
        assert 'dataset2.csv' in comparison['datasets_added'], "Should detect added dataset"
        print(f"[INFO] Datasets added: {comparison['datasets_added']}")
        
        assert comparison['samples_delta'] > 0, "Should show sample increase"
        print(f"[INFO] Sample delta: {comparison['samples_delta']:,}")
        
        assert comparison['accuracy_delta'] > 0, "Should show accuracy improvement"
        print(f"[INFO] Accuracy delta: {comparison['accuracy_delta']:.4f}")
        
        assert comparison['improved'] is True, "Should mark as improved"
        print("[PASS] Correctly identified improvement")
        
        print(f"\n[INFO] Summary: {comparison['summary']}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 3 completed successfully")


def test_metric_history():
    """Test metric history tracking"""
    print("\n" + "="*60)
    print("TEST 4: Metric History")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        history = TrainingHistory(temp_file)
        
        # Add multiple runs
        for i in range(1, 4):
            run = create_sample_run_data(i if i <= 2 else 2)
            run['run_id'] = f'20250124_14300{i}'
            run['timestamp'] = f'2025-01-24 14:30:0{i}'
            history.add_run(run)
        
        # Test accuracy history
        acc_history = history.get_metric_history('accuracy')
        assert len(acc_history) == 3, "Should have 3 accuracy values"
        print(f"[PASS] Retrieved accuracy history: {len(acc_history)} values")
        
        # Test F1 history for specific model
        f1_history = history.get_metric_history('f1_macro', 'xgboost')
        assert len(f1_history) == 3, "Should have 3 F1 values"
        print(f"[PASS] Retrieved F1 history for XGBoost: {len(f1_history)} values")
        
        # Print history
        print("\n[INFO] Accuracy trend:")
        for timestamp, value in acc_history:
            print(f"  {timestamp}: {value:.4f}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 4 completed successfully")


def test_statistics():
    """Test statistics calculation"""
    print("\n" + "="*60)
    print("TEST 5: Statistics")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        history = TrainingHistory(temp_file)
        
        # Test with no runs
        stats = history.get_statistics()
        assert stats['total_runs'] == 0, "Should have 0 runs"
        print("[PASS] Statistics with no runs")
        
        # Add multiple runs
        history.add_run(create_sample_run_data(1))
        history.add_run(create_sample_run_data(2))
        
        # Get statistics
        stats = history.get_statistics()
        
        assert stats['total_runs'] == 2, "Should have 2 runs"
        print(f"[PASS] Total runs: {stats['total_runs']}")
        
        assert len(stats['unique_datasets']) == 2, "Should have 2 unique datasets"
        print(f"[INFO] Unique datasets: {stats['unique_datasets']}")
        
        assert stats['best_accuracy_ever'] > 0, "Should have best accuracy"
        print(f"[INFO] Best accuracy ever: {stats['best_accuracy_ever']:.4f}")
        
        assert stats['best_run_id'] is not None, "Should have best run ID"
        print(f"[INFO] Best run: {stats['best_run_id']}")
        
        assert stats['average_accuracy'] > 0, "Should have average accuracy"
        print(f"[INFO] Average accuracy: {stats['average_accuracy']:.4f}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 5 completed successfully")


def test_persistence():
    """Test data persistence across instances"""
    print("\n" + "="*60)
    print("TEST 6: Persistence")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        # Create first instance and add data
        history1 = TrainingHistory(temp_file)
        run1 = create_sample_run_data(1)
        history1.add_run(run1)
        
        assert history1.get_run_count() == 1, "First instance should have 1 run"
        print("[PASS] First instance created and saved")
        
        # Create second instance (should load from file)
        history2 = TrainingHistory(temp_file)
        
        assert history2.get_run_count() == 1, "Second instance should load 1 run"
        print("[PASS] Second instance loaded data from file")
        
        # Verify data matches
        run_from_file = history2.get_last_run()
        assert run_from_file['run_id'] == run1['run_id'], "Data should match"
        print("[PASS] Data persisted correctly")
        
        # Verify JSON structure
        with open(temp_file, 'r') as f:
            data = json.load(f)
            assert 'runs' in data, "JSON should have 'runs' key"
            assert len(data['runs']) == 1, "JSON should have 1 run"
            print("[PASS] JSON file structure correct")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 6 completed successfully")


def test_dataset_tracking():
    """Test dataset usage tracking"""
    print("\n" + "="*60)
    print("TEST 7: Dataset Tracking")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        history = TrainingHistory(temp_file)
        
        # Add runs
        history.add_run(create_sample_run_data(1))
        history.add_run(create_sample_run_data(2))
        
        # Test finding when dataset was first used
        first_use = history.get_dataset_first_use('labeled_data.csv')
        assert first_use is not None, "Should find first use"
        assert first_use['run_id'] == '20250121_143000', "Should be first run"
        print("[PASS] Found when labeled_data.csv was first used")
        
        first_use_2 = history.get_dataset_first_use('dataset2.csv')
        assert first_use_2 is not None, "Should find first use"
        assert first_use_2['run_id'] == '20250122_143000', "Should be second run"
        print("[PASS] Found when dataset2.csv was first used")
        
        # Test dataset that was never used
        never_used = history.get_dataset_first_use('nonexistent.csv')
        assert never_used is None, "Should return None for unused dataset"
        print("[PASS] Returns None for unused dataset")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 7 completed successfully")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*60)
    print("TRAINING HISTORY MODULE - TEST SUITE")
    print("="*60)
    
    tests = [
        test_initialization,
        test_add_and_retrieve_runs,
        test_comparison,
        test_metric_history,
        test_statistics,
        test_persistence,
        test_dataset_tracking
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {test_func.__name__}: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\n{failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)