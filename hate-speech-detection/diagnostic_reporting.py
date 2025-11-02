"""
Diagnostic Script - Check Reporting System and Generate Missing Report
Run this after training to diagnose why reports weren't generated
"""

import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_training_history():
    """Check if training history file exists and has data"""
    print("=" * 80)
    print("STEP 1: Checking Training History")
    print("=" * 80)
    
    history_file = PROJECT_ROOT / 'results' / 'training_history.json'
    
    if not history_file.exists():
        print(f"[ERROR] Training history file not found: {history_file}")
        print("This means metrics were never saved during training.")
        return None
    
    print(f"[OK] Training history file exists: {history_file}")
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
            runs = data.get('runs', [])
            
        print(f"[OK] Found {len(runs)} training run(s) in history")
        
        if not runs:
            print("[ERROR] No runs found in history file!")
            return None
        
        # Show latest run
        latest_run = runs[-1]
        print(f"\nLatest Run Info:")
        print(f"  Run ID: {latest_run.get('run_id', 'N/A')}")
        print(f"  Timestamp: {latest_run.get('timestamp', 'N/A')}")
        print(f"  Datasets: {', '.join(latest_run.get('datasets', []))}")
        print(f"  Best Model: {latest_run.get('best_model', 'N/A')}")
        print(f"  Best Accuracy: {latest_run.get('best_accuracy', 0) * 100:.2f}%")
        print(f"  Total Samples: {latest_run.get('total_samples', 0):,}")
        
        return runs
        
    except Exception as e:
        print(f"[ERROR] Could not read training history: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def check_report_directory():
    """Check if report directory exists"""
    print("\n" + "=" * 80)
    print("STEP 2: Checking Report Directory")
    print("=" * 80)
    
    report_dir = PROJECT_ROOT / 'results' / 'training_reports'
    
    if not report_dir.exists():
        print(f"[WARNING] Report directory doesn't exist: {report_dir}")
        print(f"[INFO] Creating directory...")
        report_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created: {report_dir}")
        return report_dir
    
    print(f"[OK] Report directory exists: {report_dir}")
    
    # List any existing reports
    reports = list(report_dir.glob('*.html'))
    if reports:
        print(f"[INFO] Found {len(reports)} existing report(s):")
        for report in reports:
            size = report.stat().st_size / 1024  # KB
            print(f"  - {report.name} ({size:.1f} KB)")
    else:
        print(f"[WARNING] No HTML reports found in directory")
    
    return report_dir


def check_reporting_modules():
    """Check if reporting modules can be imported"""
    print("\n" + "=" * 80)
    print("STEP 3: Checking Reporting Modules")
    print("=" * 80)
    
    try:
        from reporting import TrainingHistory, MetricsCollector, QuickReportGenerator
        print("[OK] All reporting modules imported successfully")
        return True
    except ImportError as e:
        print(f"[ERROR] Could not import reporting modules: {e}")
        print("\nChecking individual modules:")
        
        try:
            from reporting import TrainingHistory
            print("  [OK] TrainingHistory")
        except ImportError as e:
            print(f"  [ERROR] TrainingHistory: {e}")
        
        try:
            from reporting import MetricsCollector
            print("  [OK] MetricsCollector")
        except ImportError as e:
            print(f"  [ERROR] MetricsCollector: {e}")
        
        try:
            from reporting import QuickReportGenerator
            print("  [OK] QuickReportGenerator")
        except ImportError as e:
            print(f"  [ERROR] QuickReportGenerator: {e}")
        
        return False


def check_plotting_libraries():
    """Check if matplotlib and seaborn are available"""
    print("\n" + "=" * 80)
    print("STEP 4: Checking Plotting Libraries")
    print("=" * 80)
    
    has_matplotlib = False
    has_seaborn = False
    
    try:
        import matplotlib
        print(f"[OK] matplotlib installed (version {matplotlib.__version__})")
        has_matplotlib = True
    except ImportError:
        print("[ERROR] matplotlib not installed")
        print("Install with: pip install matplotlib")
    
    try:
        import seaborn
        print(f"[OK] seaborn installed (version {seaborn.__version__})")
        has_seaborn = True
    except ImportError:
        print("[ERROR] seaborn not installed")
        print("Install with: pip install seaborn")
    
    return has_matplotlib and has_seaborn


def generate_missing_report():
    """Generate report from existing training history"""
    print("\n" + "=" * 80)
    print("STEP 5: Generating Missing Report")
    print("=" * 80)
    
    try:
        from reporting import TrainingHistory, QuickReportGenerator
        
        # Load training history
        history = TrainingHistory()
        all_runs = history.get_all_runs()
        
        if not all_runs:
            print("[ERROR] No training runs found in history!")
            return False
        
        # Get latest run
        latest_metrics = all_runs[-1]
        print(f"[INFO] Using latest run: {latest_metrics.get('run_id', 'N/A')}")
        
        # Get previous run if exists
        previous_metrics = all_runs[-2] if len(all_runs) >= 2 else None
        if previous_metrics:
            print(f"[INFO] Comparing with previous run: {previous_metrics.get('run_id', 'N/A')}")
        else:
            print(f"[INFO] No previous run for comparison")
        
        # Generate report
        print("\n[INFO] Generating HTML report...")
        generator = QuickReportGenerator()
        report_path = generator.generate_report(
            metrics=latest_metrics,
            previous_metrics=previous_metrics
        )
        
        print("\n" + "=" * 80)
        print("[SUCCESS] REPORT GENERATED!")
        print("=" * 80)
        print(f"Report location: {report_path}")
        print(f"Open in browser: file://{report_path.absolute()}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Run all diagnostic checks"""
    print("\n")
    print("*" * 80)
    print("TRAINING REPORT DIAGNOSTIC TOOL")
    print("*" * 80)
    print("\nThis tool will:")
    print("  1. Check if training history was saved")
    print("  2. Check if report directory exists")
    print("  3. Check if reporting modules are available")
    print("  4. Check if plotting libraries are installed")
    print("  5. Generate missing report if possible")
    print("\n")
    
    # Run checks
    runs = check_training_history()
    report_dir = check_report_directory()
    modules_ok = check_reporting_modules()
    plotting_ok = check_plotting_libraries()
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    issues = []
    
    if runs is None:
        issues.append("Training history not found or empty")
        print("[ERROR] Training history: NOT FOUND")
    else:
        print(f"[OK] Training history: {len(runs)} run(s) found")
    
    if not modules_ok:
        issues.append("Reporting modules cannot be imported")
        print("[ERROR] Reporting modules: IMPORT FAILED")
    else:
        print("[OK] Reporting modules: Available")
    
    if not plotting_ok:
        issues.append("Plotting libraries not installed")
        print("[WARNING] Plotting libraries: MISSING (charts won't be generated)")
    else:
        print("[OK] Plotting libraries: Available")
    
    # Decision
    print("\n" + "=" * 80)
    
    if runs and modules_ok:
        print("ATTEMPTING TO GENERATE REPORT...")
        print("=" * 80)
        success = generate_missing_report()
        
        if success:
            print("\n[SUCCESS] Report generation completed!")
        else:
            print("\n[ERROR] Report generation failed. See errors above.")
    else:
        print("CANNOT GENERATE REPORT")
        print("=" * 80)
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nPossible causes:")
        print("  1. Training script crashed before saving metrics")
        print("  2. Reporting system wasn't initialized during training")
        print("  3. Exception occurred during report generation")
        print("  4. Missing dependencies (matplotlib, seaborn)")
        
        print("\nRecommended actions:")
        print("  1. Check training logs for errors")
        print("  2. Verify reporting imports in main_train_enhanced.py")
        print("  3. Install missing dependencies:")
        print("     pip install matplotlib seaborn")
        print("  4. Re-run training with fixed main_train_enhanced.py")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()