#!/usr/bin/env python3
"""
Quick test of the annotation system using sample data
"""

import os
import sys

# Check API key
if not os.getenv('OPENROUTER_API_KEY'):
    print("=" * 70)
    print("OPENROUTER_API_KEY NOT SET")
    print("=" * 70)
    print("\nBefore running this test, please set your API key:\n")
    print("  export OPENROUTER_API_KEY='sk-or-v1-your-key-here'\n")
    print("Get a free API key from: https://openrouter.ai\n")
    print("=" * 70)
    sys.exit(1)

print("=" * 70)
print("RUNNING TEST WITH SAMPLE DATA (10 rows)")
print("=" * 70)
print("\nThis will demonstrate:")
print("  ✓ Loading CSV with all columns")
print("  ✓ Passing context to model")
print("  ✓ Batch annotation")
print("  ✓ Live progress tracking")
print("  ✓ Output format: count,hate_speech,offensive_language,neither,class,tweet")
print("  ✓ Final summary statistics")
print("\n" + "=" * 70 + "\n")

input("Press ENTER to start test...")

# Import the annotator
from hate_speech_annotator import HateSpeechAnnotator

# Run test
annotator = HateSpeechAnnotator(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    batch_size=5,  # Small batch for demo
    checkpoint_interval=5
)

try:
    df = annotator.process_dataset(
        input_csv='sample_dataset.csv',
        output_csv='test_output.csv',
        sample_size=None  # Process all 10 rows
    )
    
    annotator.print_final_summary(df)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    print("\nCheck these files:")
    print("     test_output.csv - Annotated data")
    print("     annotation_summary.txt - Statistics\n")
    print("Now you can run the full dataset with:")
    print("  python hate_speech_annotator.py")
    print("\n(Remember to update INPUT_CSV to your actual file)")
    print("=" * 70 + "\n")
    
except Exception as e:
    print(f"\nTest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)