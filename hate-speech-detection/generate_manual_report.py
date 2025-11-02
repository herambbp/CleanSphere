"""
Standalone Report Generator
Generate training reports from existing training history
Use this after training if reports weren't automatically generated
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def generate_report_from_history():
    """Generate report from training history"""
    
    print("=" * 80)
    print("STANDALONE REPORT GENERATOR")
    print("=" * 80)
    
    # Check if training history exists
    history_file = PROJECT_ROOT / 'results' / 'training_history.json'
    
    if not history_file.exists():
        print(f"\n[ERROR] Training history not found: {history_file}")
        print("\nThis means training either:")
        print("  1. Never completed successfully")
        print("  2. Failed to save metrics")
        print("  3. Used a different output directory")
        print("\nPlease run training first with the corrected main_train_enhanced.py")
        return False
    
    print(f"\n[OK] Found training history: {history_file}")
    
    # Load training history
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
            runs = data.get('runs', [])
    except Exception as e:
        print(f"\n[ERROR] Could not read training history: {e}")
        return False
    
    if not runs:
        print("\n[ERROR] No training runs found in history!")
        return False
    
    print(f"[OK] Found {len(runs)} training run(s)")
    
    # Display available runs
    print("\nAvailable training runs:")
    for i, run in enumerate(runs, 1):
        run_id = run.get('run_id', 'N/A')
        timestamp = run.get('timestamp', 'N/A')
        best_model = run.get('best_model', 'N/A')
        accuracy = run.get('best_accuracy', 0) * 100
        print(f"  {i}. Run ID: {run_id} | {timestamp} | {best_model.upper()} | {accuracy:.2f}%")
    
    # Ask which run to generate report for
    if len(runs) == 1:
        selected_idx = 0
        print(f"\n[INFO] Using the only available run")
    else:
        try:
            choice = input(f"\nSelect run to generate report for (1-{len(runs)}, or press Enter for latest): ").strip()
            if choice == "":
                selected_idx = len(runs) - 1
                print(f"[INFO] Using latest run (#{len(runs)})")
            else:
                selected_idx = int(choice) - 1
                if selected_idx < 0 or selected_idx >= len(runs):
                    print(f"[ERROR] Invalid selection. Using latest run.")
                    selected_idx = len(runs) - 1
        except:
            print(f"[ERROR] Invalid input. Using latest run.")
            selected_idx = len(runs) - 1
    
    # Get selected run and previous run
    current_metrics = runs[selected_idx]
    previous_metrics = runs[selected_idx - 1] if selected_idx > 0 else None
    
    print(f"\n[INFO] Generating report for run: {current_metrics.get('run_id', 'N/A')}")
    if previous_metrics:
        print(f"[INFO] Comparing with previous run: {previous_metrics.get('run_id', 'N/A')}")
    
    # Try to import reporting modules
    try:
        from reporting import QuickReportGenerator
    except ImportError as e:
        print(f"\n[ERROR] Could not import QuickReportGenerator: {e}")
        print("\nMake sure the reporting package is in your project:")
        print("  - reporting/__init__.py")
        print("  - reporting/quick_report_generator.py")
        return False
    
    # Check for plotting libraries
    try:
        import matplotlib
        import seaborn
        has_plotting = True
    except ImportError:
        print("\n[WARNING] matplotlib/seaborn not installed - charts will be skipped")
        print("Install with: pip install matplotlib seaborn")
        has_plotting = False
    
    # Generate report
    try:
        print("\n[INFO] Generating HTML report...")
        
        generator = QuickReportGenerator()
        report_path = generator.generate_report(
            metrics=current_metrics,
            previous_metrics=previous_metrics
        )
        
        print("\n" + "=" * 80)
        print("[SUCCESS] REPORT GENERATED!")
        print("=" * 80)
        print(f"\nReport location: {report_path}")
        print(f"File size: {report_path.stat().st_size / 1024:.1f} KB")
        print(f"\nOpen in browser:")
        print(f"  file://{report_path.absolute()}")
        print("\n" + "=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate report: {e}")
        print("\nDetailed error:")
        import traceback
        traceback.print_exc()
        
        print("\nPossible solutions:")
        print("  1. Check that all reporting modules are installed")
        print("  2. Install plotting libraries: pip install matplotlib seaborn")
        print("  3. Check training_history.json for corrupt data")
        print("  4. Check file permissions in results/training_reports/")
        
        return False


def generate_all_reports():
    """Generate reports for all training runs"""
    
    print("=" * 80)
    print("BATCH REPORT GENERATOR")
    print("=" * 80)
    
    history_file = PROJECT_ROOT / 'results' / 'training_history.json'
    
    if not history_file.exists():
        print(f"\n[ERROR] Training history not found: {history_file}")
        return False
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
            runs = data.get('runs', [])
    except Exception as e:
        print(f"\n[ERROR] Could not read training history: {e}")
        return False
    
    if not runs:
        print("\n[ERROR] No training runs found!")
        return False
    
    print(f"\n[INFO] Found {len(runs)} training run(s)")
    print("[INFO] Generating reports for all runs...")
    
    try:
        from reporting import QuickReportGenerator
        generator = QuickReportGenerator()
        
        success_count = 0
        
        for i, current_metrics in enumerate(runs):
            run_id = current_metrics.get('run_id', f'run_{i+1}')
            previous_metrics = runs[i-1] if i > 0 else None
            
            try:
                print(f"\n[{i+1}/{len(runs)}] Generating report for {run_id}...")
                
                report_path = generator.generate_report(
                    metrics=current_metrics,
                    previous_metrics=previous_metrics,
                    report_name=f"training_report_{run_id}.html"
                )
                
                print(f"  [OK] Generated: {report_path.name}")
                success_count += 1
                
            except Exception as e:
                print(f"  [ERROR] Failed: {e}")
        
        print("\n" + "=" * 80)
        print(f"[SUMMARY] Generated {success_count}/{len(runs)} reports")
        print("=" * 80)
        print(f"\nReports location: {PROJECT_ROOT / 'results' / 'training_reports'}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"\n[ERROR] Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_training_summary():
    """Show summary of all training runs"""
    
    print("=" * 80)
    print("TRAINING HISTORY SUMMARY")
    print("=" * 80)
    
    history_file = PROJECT_ROOT / 'results' / 'training_history.json'
    
    if not history_file.exists():
        print(f"\n[ERROR] Training history not found: {history_file}")
        return
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
            runs = data.get('runs', [])
    except Exception as e:
        print(f"\n[ERROR] Could not read training history: {e}")
        return
    
    if not runs:
        print("\n[INFO] No training runs found")
        return
    
    print(f"\n[INFO] Total training runs: {len(runs)}")
    print("\nRun Details:")
    print("-" * 80)
    
    for i, run in enumerate(runs, 1):
        run_id = run.get('run_id', 'N/A')
        timestamp = run.get('timestamp', 'N/A')
        datasets = run.get('datasets', [])
        new_datasets = run.get('new_datasets', [])
        total_samples = run.get('total_samples', 0)
        best_model = run.get('best_model', 'N/A')
        best_accuracy = run.get('best_accuracy', 0) * 100
        duration = run.get('duration_seconds', 0)
        
        print(f"\nRun #{i}:")
        print(f"  Run ID: {run_id}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Datasets: {', '.join(datasets)}")
        if new_datasets:
            print(f"  New Datasets: {', '.join(new_datasets)}")
        print(f"  Total Samples: {total_samples:,}")
        print(f"  Best Model: {best_model.upper()}")
        print(f"  Best Accuracy: {best_accuracy:.2f}%")
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Show all models if available
        models = run.get('models', {})
        if models:
            print(f"  All Models:")
            for model_name, metrics in sorted(models.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
                acc = metrics.get('accuracy', 0) * 100
                f1 = metrics.get('f1_macro', 0) * 100
                print(f"    - {model_name.upper():12s}: {acc:.2f}% accuracy, {f1:.2f}% F1")
    
    print("\n" + "=" * 80)


def main():
    """Main menu"""
    
    print("\n")
    print("*" * 80)
    print("TRAINING REPORT GENERATOR")
    print("*" * 80)
    print("\nOptions:")
    print("  1. Generate report for specific run (interactive)")
    print("  2. Generate report for latest run")
    print("  3. Generate reports for all runs (batch)")
    print("  4. Show training history summary")
    print("  5. Exit")
    print("\n")
    
    try:
        choice = input("Select option (1-5): ").strip()
    except:
        choice = "2"
    
    if choice == "1":
        generate_report_from_history()
    elif choice == "2":
        # Generate for latest run automatically
        history_file = PROJECT_ROOT / 'results' / 'training_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    runs = data.get('runs', [])
                if runs:
                    from reporting import QuickReportGenerator
                    current_metrics = runs[-1]
                    previous_metrics = runs[-2] if len(runs) >= 2 else None
                    
                    print(f"\n[INFO] Generating report for latest run: {current_metrics.get('run_id', 'N/A')}")
                    
                    generator = QuickReportGenerator()
                    report_path = generator.generate_report(
                        metrics=current_metrics,
                        previous_metrics=previous_metrics
                    )
                    
                    print("\n" + "=" * 80)
                    print("[SUCCESS] REPORT GENERATED!")
                    print("=" * 80)
                    print(f"\nReport location: {report_path}")
                    print(f"Open in browser: file://{report_path.absolute()}")
                    print("\n" + "=" * 80)
                else:
                    print("\n[ERROR] No training runs found!")
            except Exception as e:
                print(f"\n[ERROR] Failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n[ERROR] Training history not found!")
    elif choice == "3":
        generate_all_reports()
    elif choice == "4":
        show_training_summary()
    elif choice == "5":
        print("\nExiting...")
    else:
        print("\n[ERROR] Invalid option. Running option 2 (latest run)...")
        # Default to latest run
        history_file = PROJECT_ROOT / 'results' / 'training_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    runs = data.get('runs', [])
                if runs:
                    from reporting import QuickReportGenerator
                    current_metrics = runs[-1]
                    previous_metrics = runs[-2] if len(runs) >= 2 else None
                    
                    generator = QuickReportGenerator()
                    report_path = generator.generate_report(
                        metrics=current_metrics,
                        previous_metrics=previous_metrics
                    )
                    
                    print("\n" + "=" * 80)
                    print("[SUCCESS] REPORT GENERATED!")
                    print("=" * 80)
                    print(f"\nReport location: {report_path}")
                    print(f"Open in browser: file://{report_path.absolute()}")
                    print("\n" + "=" * 80)
            except Exception as e:
                print(f"\n[ERROR] Failed: {e}")


if __name__ == "__main__":
    main()