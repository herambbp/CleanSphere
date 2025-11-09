"""
Test Script for MetricsCollector Module
Tests all functionality of the metrics collection system
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reporting.metrics_collector import MetricsCollector


def test_initialization():
    """Test MetricsCollector initialization"""
    print("\n" + "="*60)
    print("TEST 1: Initialization")
    print("="*60)
    
    collector = MetricsCollector()
    
    assert collector.run_id is None, "Should start with no run ID"
    assert collector.datasets == [], "Should start with empty datasets"
    assert collector.total_samples == 0, "Should start with 0 samples"
    assert collector.model_results == {}, "Should start with no results"
    print("[PASS] Initialization successful")
    
    print("\nTest 1 completed successfully")


def test_start_run():
    """Test starting a metrics collection run"""
    print("\n" + "="*60)
    print("TEST 2: Start Run")
    print("="*60)
    
    collector = MetricsCollector()
    
    # Test auto-generated run ID
    collector.start_run()
    assert collector.run_id is not None, "Should generate run ID"
    assert collector.start_time is not None, "Should set start time"
    print(f"[PASS] Auto-generated run ID: {collector.run_id}")
    
    # Test custom run ID
    custom_id = "test_run_123"
    collector.start_run(custom_id)
    assert collector.run_id == custom_id, "Should use custom run ID"
    print(f"[PASS] Custom run ID: {collector.run_id}")
    
    print("\nTest 2 completed successfully")


def test_set_datasets():
    """Test recording dataset information"""
    print("\n" + "="*60)
    print("TEST 3: Set Datasets")
    print("="*60)
    
    collector = MetricsCollector()
    collector.start_run()
    
    # Test single dataset
    datasets = ['labeled_data.csv']
    total_samples = 32780
    collector.set_datasets(datasets, total_samples)
    
    assert collector.datasets == datasets, "Should store datasets"
    assert collector.total_samples == total_samples, "Should store total samples"
    print(f"[PASS] Single dataset: {collector.datasets}")
    print(f"[INFO] Total samples: {collector.total_samples:,}")
    
    # Test multiple datasets with new datasets
    datasets = ['labeled_data.csv', 'dataset2.csv']
    new_datasets = ['dataset2.csv']
    total_samples = 45230
    collector.set_datasets(datasets, total_samples, new_datasets)
    
    assert collector.datasets == datasets, "Should store multiple datasets"
    assert collector.new_datasets == new_datasets, "Should store new datasets"
    assert collector.total_samples == total_samples, "Should update total samples"
    print(f"[PASS] Multiple datasets: {collector.datasets}")
    print(f"[INFO] New datasets: {collector.new_datasets}")
    
    print("\nTest 3 completed successfully")


def test_set_splits():
    """Test recording split information"""
    print("\n" + "="*60)
    print("TEST 4: Set Splits")
    print("="*60)
    
    collector = MetricsCollector()
    collector.start_run()
    
    train_size = 22946
    val_size = 4917
    test_size = 4917
    class_dist = {0: 9834, 1: 14123, 2: 8823}
    
    collector.set_splits(train_size, val_size, test_size, class_dist)
    
    assert collector.train_samples == train_size, "Should store train size"
    assert collector.val_samples == val_size, "Should store val size"
    assert collector.test_samples == test_size, "Should store test size"
    print(f"[PASS] Split sizes recorded")
    print(f"[INFO] Train: {collector.train_samples:,}")
    print(f"[INFO] Val:   {collector.val_samples:,}")
    print(f"[INFO] Test:  {collector.test_samples:,}")
    
    # Check class distribution (should be converted to strings)
    assert '0' in collector.class_distribution, "Should convert keys to strings"
    assert collector.class_distribution['0'] == 9834, "Should store correct counts"
    print(f"[PASS] Class distribution: {collector.class_distribution}")
    
    print("\nTest 4 completed successfully")


def test_add_model_results():
    """Test adding model results"""
    print("\n" + "="*60)
    print("TEST 5: Add Model Results")
    print("="*60)
    
    collector = MetricsCollector()
    collector.start_run()
    
    # Add first model
    svm_metrics = {
        'accuracy': 0.8534,
        'f1_macro': 0.8201,
        'precision_macro': 0.8312,
        'recall_macro': 0.8156
    }
    collector.add_model_result('svm', svm_metrics)
    
    assert 'svm' in collector.model_results, "Should store model results"
    assert collector.best_model == 'svm', "Should set as best model"
    assert collector.best_accuracy == 0.8534, "Should store best accuracy"
    print(f"[PASS] Added SVM results")
    print(f"[INFO] Best model: {collector.best_model} ({collector.best_accuracy:.4f})")
    
    # Add better model
    xgb_metrics = {
        'accuracy': 0.8612,
        'f1_macro': 0.8334,
        'precision_macro': 0.8421,
        'recall_macro': 0.8267
    }
    collector.add_model_result('xgboost', xgb_metrics)
    
    assert 'xgboost' in collector.model_results, "Should store XGBoost results"
    assert collector.best_model == 'xgboost', "Should update best model"
    assert collector.best_accuracy == 0.8612, "Should update best accuracy"
    print(f"[PASS] Added XGBoost results")
    print(f"[INFO] Best model: {collector.best_model} ({collector.best_accuracy:.4f})")
    
    # Test that missing accuracy raises error
    try:
        collector.add_model_result('bad_model', {'f1': 0.85})
        assert False, "Should raise error for missing accuracy"
    except ValueError as e:
        print(f"[PASS] Correctly raised error for missing accuracy")
    
    print("\nTest 5 completed successfully")


def test_set_feature_info():
    """Test recording feature information"""
    print("\n" + "="*60)
    print("TEST 6: Set Feature Info")
    print("="*60)
    
    collector = MetricsCollector()
    collector.start_run()
    
    feature_dim = 5234
    collector.set_feature_info(feature_dim)
    
    assert collector.feature_dimensions == feature_dim, "Should store feature dimensions"
    print(f"[PASS] Feature dimensions: {collector.feature_dimensions:,}")
    
    print("\nTest 6 completed successfully")


def test_finalize():
    """Test finalization and data export"""
    print("\n" + "="*60)
    print("TEST 7: Finalize")
    print("="*60)
    
    collector = MetricsCollector()
    collector.start_run('test_run_final')
    
    # Add all data
    collector.set_datasets(['labeled_data.csv'], 32780)
    collector.set_splits(22946, 4917, 4917, {0: 9834, 1: 14123, 2: 8823})
    collector.add_model_result('svm', {
        'accuracy': 0.8534,
        'f1_macro': 0.8201
    })
    collector.add_model_result('xgboost', {
        'accuracy': 0.8612,
        'f1_macro': 0.8334
    })
    collector.set_feature_info(5234)
    
    # Wait a bit to have measurable duration
    time.sleep(0.1)
    
    # Finalize
    run_data = collector.finalize()
    
    # Check all required fields
    assert 'run_id' in run_data, "Should have run_id"
    assert 'timestamp' in run_data, "Should have timestamp"
    assert 'datasets' in run_data, "Should have datasets"
    assert 'total_samples' in run_data, "Should have total_samples"
    assert 'models' in run_data, "Should have models"
    assert 'best_model' in run_data, "Should have best_model"
    assert 'best_accuracy' in run_data, "Should have best_accuracy"
    assert 'duration_seconds' in run_data, "Should have duration"
    print("[PASS] All required fields present")
    
    # Check values
    assert run_data['run_id'] == 'test_run_final', "Should match run ID"
    assert run_data['datasets'] == ['labeled_data.csv'], "Should match datasets"
    assert run_data['total_samples'] == 32780, "Should match total samples"
    assert run_data['best_model'] == 'xgboost', "Should identify best model"
    assert run_data['best_accuracy'] == 0.8612, "Should have best accuracy"
    assert run_data['duration_seconds'] > 0, "Should have positive duration"
    print("[PASS] All values correct")
    
    print(f"\n[INFO] Run data structure:")
    for key, value in run_data.items():
        if isinstance(value, dict) and len(str(value)) > 50:
            print(f"  {key}: <dict with {len(value)} items>")
        else:
            print(f"  {key}: {value}")
    
    print("\nTest 7 completed successfully")


def test_get_summary():
    """Test summary generation"""
    print("\n" + "="*60)
    print("TEST 8: Get Summary")
    print("="*60)
    
    collector = MetricsCollector()
    
    # Test before start
    summary = collector.get_summary()
    assert "No metrics collected" in summary, "Should indicate no data"
    print("[PASS] Summary before start correct")
    
    # Add data and test summary
    collector.start_run()
    collector.set_datasets(['labeled_data.csv', 'dataset2.csv'], 45230, ['dataset2.csv'])
    collector.set_splits(31661, 6784, 6785, {0: 11734, 1: 18523, 2: 14973})
    collector.add_model_result('xgboost', {'accuracy': 0.8789, 'f1_macro': 0.8556})
    collector.set_feature_info(5234)
    
    summary = collector.get_summary()
    
    assert 'Run ID:' in summary, "Should include run ID"
    assert 'Datasets:' in summary, "Should include datasets"
    assert 'Total samples:' in summary, "Should include total samples"
    assert 'Best model:' in summary, "Should include best model"
    print("[PASS] Summary contains all expected information")
    
    print(f"\n[INFO] Generated summary:")
    print(summary)
    
    print("\nTest 8 completed successfully")


def test_validate():
    """Test validation"""
    print("\n" + "="*60)
    print("TEST 9: Validate")
    print("="*60)
    
    collector = MetricsCollector()
    
    # Should fail validation initially
    assert not collector.validate(), "Should fail validation without data"
    print("[PASS] Validation fails when no data")
    
    # Start run
    collector.start_run()
    assert not collector.validate(), "Should still fail without complete data"
    print("[PASS] Validation fails without complete data")
    
    # Add minimum required data
    collector.set_datasets(['test.csv'], 1000)
    collector.add_model_result('test_model', {'accuracy': 0.85})
    
    assert collector.validate(), "Should pass validation with minimum data"
    print("[PASS] Validation passes with complete data")
    
    print("\nTest 9 completed successfully")


def test_metadata():
    """Test custom metadata"""
    print("\n" + "="*60)
    print("TEST 10: Custom Metadata")
    print("="*60)
    
    collector = MetricsCollector()
    collector.start_run()
    
    # Add custom metadata
    collector.add_metadata('experiment_name', 'test_experiment')
    collector.add_metadata('learning_rate', 0.001)
    collector.add_metadata('batch_size', 32)
    
    # Setup minimal required data
    collector.set_datasets(['test.csv'], 1000)
    collector.add_model_result('test', {'accuracy': 0.85})
    
    # Finalize and check metadata
    run_data = collector.finalize()
    
    assert 'metadata' in run_data, "Should include metadata"
    assert run_data['metadata']['experiment_name'] == 'test_experiment', "Should store string metadata"
    assert run_data['metadata']['learning_rate'] == 0.001, "Should store float metadata"
    assert run_data['metadata']['batch_size'] == 32, "Should store int metadata"
    print("[PASS] Custom metadata stored correctly")
    
    print(f"\n[INFO] Metadata: {run_data['metadata']}")
    
    print("\nTest 10 completed successfully")


def test_reset():
    """Test reset functionality"""
    print("\n" + "="*60)
    print("TEST 11: Reset")
    print("="*60)
    
    collector = MetricsCollector()
    collector.start_run()
    collector.set_datasets(['test.csv'], 1000)
    collector.add_model_result('test', {'accuracy': 0.85})
    
    # Verify data exists
    assert collector.run_id is not None, "Should have run ID"
    assert collector.datasets, "Should have datasets"
    print("[PASS] Data exists before reset")
    
    # Reset
    collector.reset()
    
    # Verify data cleared
    assert collector.run_id is None, "Should clear run ID"
    assert collector.datasets == [], "Should clear datasets"
    assert collector.model_results == {}, "Should clear results"
    print("[PASS] Data cleared after reset")
    
    print("\nTest 11 completed successfully")


def test_integration_with_training_history():
    """Test integration with TrainingHistory"""
    print("\n" + "="*60)
    print("TEST 12: Integration with TrainingHistory")
    print("="*60)
    
    from reporting.training_history import TrainingHistory
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'test_history.json')
    
    try:
        # Collect metrics
        collector = MetricsCollector()
        collector.start_run()
        collector.set_datasets(['labeled_data.csv'], 32780)
        collector.set_splits(22946, 4917, 4917, {0: 9834, 1: 14123, 2: 8823})
        collector.add_model_result('xgboost', {
            'accuracy': 0.8612,
            'f1_macro': 0.8334,
            'precision_macro': 0.8421,
            'recall_macro': 0.8267
        })
        collector.set_feature_info(5234)
        
        # Finalize
        run_data = collector.finalize()
        print("[PASS] Metrics collected and finalized")
        
        # Add to TrainingHistory
        history = TrainingHistory(temp_file)
        history.add_run(run_data)
        print("[PASS] Run data added to TrainingHistory")
        
        # Retrieve and verify
        last_run = history.get_last_run()
        assert last_run is not None, "Should retrieve run"
        assert last_run['run_id'] == run_data['run_id'], "Should match run ID"
        assert last_run['best_accuracy'] == 0.8612, "Should match accuracy"
        print("[PASS] Data retrieved correctly from TrainingHistory")
        
        print(f"\n[INFO] Successfully integrated with TrainingHistory")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\nTest 12 completed successfully")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*60)
    print("METRICS COLLECTOR MODULE - TEST SUITE")
    print("="*60)
    
    tests = [
        test_initialization,
        test_start_run,
        test_set_datasets,
        test_set_splits,
        test_add_model_results,
        test_set_feature_info,
        test_finalize,
        test_get_summary,
        test_validate,
        test_metadata,
        test_reset,
        test_integration_with_training_history
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
            import traceback
            traceback.print_exc()
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