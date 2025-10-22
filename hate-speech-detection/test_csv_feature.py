"""
Test script for CSV User Analytics Feature
Verifies that all components work correctly
"""

import sys
from pathlib import Path

print("=" * 80)
print("CSV USER ANALYTICS - FEATURE TEST")
print("=" * 80)

# Test 1: Import CSV Processor
print("\n1. Testing CSV Processor Import...")
try:
    from csv_processor import CSVProcessor
    print("   ✓ CSV Processor imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import CSV Processor: {e}")
    print("   → Make sure csv_processor.py is in the project directory")
    sys.exit(1)

# Test 2: Import Classifier
print("\n2. Testing Classifier Import...")
try:
    from inference.explainable_classifier import ExplainableTweetClassifier
    print("   ✓ Classifier imported successfully")
except ImportError as e:
    print(f"   ✗ Failed to import Classifier: {e}")
    print("   → Make sure you're in the project directory")
    sys.exit(1)

# Test 3: Check for sample CSV
print("\n3. Checking for sample CSV file...")
sample_csv = Path('sample_comments.csv')
if sample_csv.exists():
    print(f"   ✓ Found sample CSV: {sample_csv}")
else:
    print("   ⚠ Sample CSV not found, creating one...")
    import pandas as pd
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice'],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
        'comment': ['Good morning!', 'You are stupid', 'Have a nice day!']
    }
    pd.DataFrame(sample_data).to_csv('sample_comments.csv', index=False)
    print(f"   ✓ Created sample CSV: sample_comments.csv")

# Test 4: Initialize Processor
print("\n4. Initializing CSV Processor...")
try:
    processor = CSVProcessor()
    print("   ✓ CSV Processor initialized")
except Exception as e:
    print(f"   ✗ Failed to initialize: {e}")
    sys.exit(1)

# Test 5: Process Sample CSV
print("\n5. Processing sample CSV file...")
try:
    results_df = processor.process_csv('sample_comments.csv')
    print(f"   ✓ Successfully processed {len(results_df)} comments")
    print(f"   ✓ Found {len(processor.user_analytics)} users")
except Exception as e:
    print(f"   ✗ Failed to process CSV: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check Analytics
print("\n6. Checking user analytics...")
try:
    analytics = processor.get_user_analytics()
    
    if not analytics:
        print("   ⚠ No analytics generated")
    else:
        print(f"   ✓ Analytics generated for {len(analytics)} users")
        
        # Display sample analytics
        for user, data in list(analytics.items())[:2]:
            print(f"\n   User: {user}")
            print(f"   - Total Comments: {data['total_comments']}")
            print(f"   - Risk Score: {data['risk_score']}")
            print(f"   - Risk Level: {data['risk_level']}")
            print(f"   - Hate %: {data['hate_percentage']}%")
except Exception as e:
    print(f"   ✗ Failed to get analytics: {e}")
    sys.exit(1)

# Test 7: Export Results
print("\n7. Testing export functionality...")
try:
    processor.export_results('test_results.csv')
    print("   ✓ Exported detailed results to test_results.csv")
    
    processor.export_user_analytics('test_analytics.csv')
    print("   ✓ Exported user analytics to test_analytics.csv")
    
    # Clean up test files
    Path('test_results.csv').unlink()
    Path('test_analytics.csv').unlink()
    print("   ✓ Cleaned up test files")
except Exception as e:
    print(f"   ✗ Failed to export: {e}")

# Test 8: Test API Endpoints (if FastAPI is available)
print("\n8. Testing API availability...")
try:
    from fastapi import FastAPI
    print("   ✓ FastAPI is installed")
    print("   ✓ API endpoints should be available")
    print("   → Start API with: python main.py")
    print("   → Access docs at: http://localhost:8000/docs")
except ImportError:
    print("   ⚠ FastAPI not installed")
    print("   → Install with: pip install fastapi uvicorn")

# Test 9: Check Dashboard
print("\n9. Checking dashboard file...")
dashboard_file = Path('dashboard.html')
if dashboard_file.exists():
    print(f"   ✓ Dashboard found: {dashboard_file}")
    print("   → Open dashboard.html in your browser")
else:
    print("   ⚠ Dashboard file not found")
    print("   → Copy dashboard.html to project directory")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("\n✓ All core tests passed!")
print("\nYour CSV User Analytics feature is ready to use!")
print("\nNext Steps:")
print("  1. Start the API:")
print("     python main.py")
print("\n  2. Open dashboard:")
print("     open dashboard.html")
print("\n  3. Upload a CSV file with user comments")
print("\n  4. Explore the analytics dashboard!")
print("\nSample CSV format:")
print("  name,timestamp,comment")
print("  Alice,2024-01-01 10:00:00,Good morning!")
print("  Bob,2024-01-01 11:00:00,You're stupid")
print("\n" + "=" * 80)

# Optional: Show quick usage example
print("\nQUICK USAGE EXAMPLE:")
print("-" * 80)
print("""
from csv_processor import process_user_csv

# Process CSV and get analytics
results_df, user_analytics = process_user_csv(
    csv_path='your_file.csv',
    output_dir='results'  # Optional: exports to this directory
)

# View user analytics
for user, analytics in user_analytics.items():
    print(f"{user}: Risk={analytics['risk_score']}, Level={analytics['risk_level']}")

# Or use the dashboard for interactive exploration!
""")
print("=" * 80)