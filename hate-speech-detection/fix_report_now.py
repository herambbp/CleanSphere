"""
IMMEDIATE FIX - Generate Report Despite Missing best_model

This script will generate your training report even though best_model is None.
Run this right now to get your report!
"""

import sys
from pathlib import Path

# Fix the issue immediately
def quick_fix():
    print("=" * 80)
    print("IMMEDIATE FIX - Generating Report")
    print("=" * 80)
    
    # Step 1: Replace the broken quick_report_generator.py
    print("\nStep 1: Replacing broken report generator...")
    
    try:
        import shutil
        
        # Backup original
        original = Path('reporting/quick_report_generator.py')
        backup = Path('reporting/quick_report_generator.py.backup')
        
        if original.exists():
            shutil.copy2(original, backup)
            print(f"[OK] Backed up original to: {backup}")
        
        # Copy fixed version
        fixed = Path('quick_report_generator_fixed.py')
        if not fixed.exists():
            print(f"[ERROR] Fixed file not found: {fixed}")
            print("Please ensure quick_report_generator_fixed.py is in the current directory")
            return False
        
        shutil.copy2(fixed, original)
        print(f"[OK] Installed fixed version")
        
    except Exception as e:
        print(f"[ERROR] Could not replace file: {e}")
        print("\nManual fix:")
        print("  cp quick_report_generator_fixed.py reporting/quick_report_generator.py")
        return False
    
    # Step 2: Generate the report
    print("\nStep 2: Generating report...")
    
    try:
        from reporting import TrainingHistory, QuickReportGenerator
        
        # Reload the module to use fixed version
        import importlib
        import reporting.quick_report_generator
        importlib.reload(reporting.quick_report_generator)
        from reporting.quick_report_generator import QuickReportGenerator
        
        history = TrainingHistory()
        runs = history.get_all_runs()
        
        if not runs:
            print("[ERROR] No training runs found!")
            return False
        
        current = runs[-1]
        previous = runs[-2] if len(runs) >= 2 else None
        
        print(f"[INFO] Generating report for run: {current.get('run_id', 'N/A')}")
        
        generator = QuickReportGenerator()
        report_path = generator.generate_report(
            metrics=current,
            previous_metrics=previous
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
        print(f"[ERROR] Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        return False

def explain_problem():
    """Explain what went wrong"""
    print("\n" + "=" * 80)
    print("WHAT WENT WRONG")
    print("=" * 80)
    
    print("""
Your training completed but the model metrics weren't collected properly.
Specifically:
  - best_model is None
  - best_accuracy is 0.00%
  - No model performance data in training_history.json

This means in your training script, the code that collects model results
wasn't executed or had errors.

The problem is in main_train_enhanced.py around this code:

    for _, row in comparison_df.iterrows():
        collector.add_model_result(
            model_name=model_name,
            metrics=metrics_dict
        )

This code either:
  1. Never ran
  2. comparison_df was empty
  3. An exception occurred and was silently caught

Your trained models ARE STILL SAVED in saved_models/ folder!
You can still use them for predictions.

The report generator crashed because it tried to call .upper() on None:
    best_model.upper()  # Error when best_model is None

The fixed version handles this gracefully and shows a warning in the report.
""")

def next_steps():
    """Show next steps"""
    print("\n" + "=" * 80)
    print("NEXT STEPS TO PREVENT THIS")
    print("=" * 80)
    
    print("""
1. Use the corrected main_train_enhanced.py for future training
   - It has better error handling
   - It properly collects all metrics
   - It logs each step clearly

2. Check your training logs for errors around "Collecting model metrics"

3. Verify your models directory:
   ls -la saved_models/

4. If models exist, you can still use them:
   from inference.tweet_classifier import TweetClassifier
   classifier = TweetClassifier()
   result = classifier.classify("test tweet")

5. For next training run, watch for these log messages:
   [OK] Collected metrics for X models

   If you don't see this, stop and fix the issue immediately.
""")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix and generate training report')
    parser.add_argument('--explain', action='store_true', help='Explain what went wrong')
    parser.add_argument('--help-next', action='store_true', help='Show next steps')
    
    args = parser.parse_args()
    
    if args.explain:
        explain_problem()
    elif args.help_next:
        next_steps()
    else:
        success = quick_fix()
        
        if success:
            print("\n")
            print("Your report is ready! Despite the missing model metrics,")
            print("the report shows your dataset info and training duration.")
            print("\nFor your next training run, use the corrected main_train_enhanced.py")
            print("to ensure all metrics are collected properly.")
        else:
            print("\nAutomatic fix failed. Try manual steps:")
            print("1. cp quick_report_generator_fixed.py reporting/quick_report_generator.py")
            print("2. python generate_report_manual.py")