"""
Extract Model Metrics from Saved Models
This checks your saved_models/ directory and evaluates them on your test data
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def extract_model_metrics():
    print("=" * 80)
    print("EXTRACTING METRICS FROM SAVED MODELS")
    print("=" * 80)
    
    # Check if models exist
    models_dir = Path('saved_models')
    if not models_dir.exists():
        print(f"\n[ERROR] Models directory not found: {models_dir}")
        return None
    
    model_files = list(models_dir.glob('*.pkl'))
    if not model_files:
        print(f"\n[ERROR] No model files found in {models_dir}")
        return None
    
    print(f"\n[OK] Found {len(model_files)} saved model(s)")
    for mf in model_files:
        print(f"  - {mf.name}")
    
    # Check if test data exists
    data_dir = Path('data/processed')
    test_files = ['X_test.npy', 'y_test.npy']
    
    # Try multiple locations
    possible_locations = [
        Path('data/processed'),
        Path('data/splits'),
        Path('data'),
    ]
    
    X_test, y_test = None, None
    
    for loc in possible_locations:
        x_path = loc / 'X_test.npy'
        y_path = loc / 'y_test.npy'
        
        if x_path.exists() and y_path.exists():
            print(f"\n[OK] Found test data in: {loc}")
            try:
                X_test = np.load(x_path, allow_pickle=True)
                y_test = np.load(y_path, allow_pickle=True)
                print(f"[OK] Loaded test data: {X_test.shape}, {y_test.shape}")
                break
            except Exception as e:
                print(f"[WARNING] Could not load from {loc}: {e}")
                continue
    
    if X_test is None:
        print("\n[ERROR] Could not find test data (X_test.npy, y_test.npy)")
        print("Checked locations:")
        for loc in possible_locations:
            print(f"  - {loc}")
        print("\nWithout test data, cannot extract metrics.")
        print("\nAlternative: Re-run inference on a validation dataset")
        return None
    
    # Evaluate each model
    results = {}
    
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    for model_file in model_files:
        model_name = model_file.stem  # filename without .pkl
        
        try:
            print(f"\n[INFO] Loading {model_name}...")
            model = joblib.load(model_file)
            
            print(f"[INFO] Making predictions...")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            precision_macro = precision_score(y_test, y_pred, average='macro')
            recall_macro = recall_score(y_test, y_pred, average='macro')
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro
            }
            
            print(f"[OK] {model_name}:")
            print(f"     Accuracy: {accuracy:.4f}")
            print(f"     F1 (macro): {f1_macro:.4f}")
            print(f"     Precision (macro): {precision_macro:.4f}")
            print(f"     Recall (macro): {recall_macro:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name}: {e}")
            continue
    
    if not results:
        print("\n[ERROR] No models could be evaluated!")
        return None
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_name = best_model[0]
    best_accuracy = best_model[1]['accuracy']
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 (Macro)': f"{metrics['f1_macro']:.4f}",
            'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
            'Recall (Macro)': f"{metrics['recall_macro']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    print(f"\n[BEST MODEL] {best_name} with {best_accuracy:.4f} accuracy")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save to CSV
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / 'extracted_model_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved to: {csv_path}")
    
    # Update training history
    try:
        from reporting import TrainingHistory, MetricsCollector
        
        history = TrainingHistory()
        runs = history.get_all_runs()
        
        if runs:
            # Update the last run with extracted metrics
            last_run = runs[-1]
            
            print(f"\n[INFO] Updating training history with extracted metrics...")
            
            # Convert results to format needed by MetricsCollector
            last_run['models'] = {
                name: {
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro']
                }
                for name, metrics in results.items()
            }
            last_run['best_model'] = best_name
            last_run['best_accuracy'] = best_accuracy
            
            # Save updated history
            import json
            history_file = Path('results/training_history.json')
            with open(history_file, 'w') as f:
                json.dump({'runs': runs}, f, indent=2)
            
            print(f"[OK] Updated training history: {history_file}")
            print(f"[OK] Best model: {best_name}")
            print(f"[OK] Best accuracy: {best_accuracy:.4f}")
            
    except Exception as e:
        print(f"[WARNING] Could not update training history: {e}")
        print("But metrics are saved in CSV file")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] METRICS EXTRACTED!")
    print("=" * 80)
    
    return results, best_name, best_accuracy

def generate_updated_report():
    """Generate report with updated metrics"""
    print("\n" + "=" * 80)
    print("GENERATING UPDATED REPORT")
    print("=" * 80)
    
    try:
        from reporting import TrainingHistory, QuickReportGenerator
        
        history = TrainingHistory()
        runs = history.get_all_runs()
        
        if not runs:
            print("[ERROR] No runs in history")
            return False
        
        current = runs[-1]
        previous = runs[-2] if len(runs) >= 2 else None
        
        # Check if metrics are now available
        if current.get('best_model') and current.get('best_model') != 'None':
            print(f"[OK] Metrics found in history!")
            print(f"     Best model: {current['best_model']}")
            print(f"     Best accuracy: {current['best_accuracy']:.4f}")
            
            generator = QuickReportGenerator()
            report_path = generator.generate_report(
                metrics=current,
                previous_metrics=previous
            )
            
            print("\n[SUCCESS] Report generated with ACTUAL metrics!")
            print(f"Report: {report_path}")
            print(f"Open: file:///{report_path.absolute()}")
            
            return True
        else:
            print("[WARNING] Metrics still not in history")
            print("But they are in: results/extracted_model_metrics.csv")
            return False
            
    except Exception as e:
        print(f"[ERROR] Could not generate report: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    print("*" * 80)
    print("MODEL METRICS EXTRACTOR")
    print("*" * 80)
    print("\nThis script will:")
    print("  1. Load your saved models from saved_models/")
    print("  2. Evaluate them on test data")
    print("  3. Extract actual performance metrics")
    print("  4. Update training_history.json")
    print("  5. Generate report with REAL metrics")
    print("\n")
    
    result = extract_model_metrics()
    
    if result:
        results, best_name, best_accuracy = result
        
        print("\n" + "=" * 80)
        print("NEXT STEP: GENERATE REPORT")
        print("=" * 80)
        
        try:
            choice = input("\nGenerate report with extracted metrics? (y/n): ").strip().lower()
            if choice == 'y':
                generate_updated_report()
            else:
                print("\nMetrics saved. Run this to generate report:")
                print("  python generate_report_manual.py")
        except:
            print("\nMetrics saved. Run this to generate report:")
            print("  python generate_report_manual.py")
    else:
        print("\n[ERROR] Could not extract metrics")
        print("\nPossible reasons:")
        print("  1. Test data not found (X_test.npy, y_test.npy)")
        print("  2. Models not compatible with test data")
        print("  3. Model files corrupted")
        print("\nSolution: Re-train with fixed script")