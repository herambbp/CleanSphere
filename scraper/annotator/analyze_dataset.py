#!/usr/bin/env python3
"""
Dataset Analyzer - Inspect your dataset before annotation
"""

import pandas as pd
import sys
import os

def analyze_dataset(csv_path: str):
    """Analyze the structure and content of the dataset"""
    
    print("=" * 70)
    print("📊 DATASET ANALYZER")
    print("=" * 70)
    
    try:
        # Load dataset
        print(f"\n📂 Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"✓ Loaded successfully\n")
        
        # Basic info
        print("=" * 70)
        print("📋 BASIC INFORMATION")
        print("=" * 70)
        print(f"Total Rows:        {len(df):,}")
        print(f"Total Columns:     {len(df.columns)}")
        print(f"Memory Usage:      {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column info
        print("\n" + "=" * 70)
        print("📊 COLUMN INFORMATION")
        print("=" * 70)
        print(f"{'Column Name':<30} {'Type':<15} {'Non-Null':<12} {'Sample'}")
        print("-" * 70)
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            sample = str(df[col].iloc[0])[:30] if len(df) > 0 else 'N/A'
            print(f"{col:<30} {dtype:<15} {non_null:<12,} {sample}")
        
        # Identify text column
        text_candidates = ['text', 'tweet', 'content', 'message', 'post']
        text_col = None
        for col in text_candidates:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            # Try to find a column with string data
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.len().mean() > 10:
                    text_col = col
                    break
        
        if text_col:
            print(f"\n✓ Text column identified: '{text_col}'")
        else:
            print(f"\n⚠️  Could not identify text column automatically")
        
        # Text statistics
        if text_col:
            print("\n" + "=" * 70)
            print("📝 TEXT STATISTICS")
            print("=" * 70)
            text_lengths = df[text_col].astype(str).str.len()
            print(f"Average Length:    {text_lengths.mean():.1f} characters")
            print(f"Median Length:     {text_lengths.median():.1f} characters")
            print(f"Min Length:        {text_lengths.min()} characters")
            print(f"Max Length:        {text_lengths.max()} characters")
            print(f"Empty Texts:       {df[text_col].isna().sum():,}")
        
        # Existing labels (if any)
        label_cols = ['label', 'class', 'category', 'sentiment']
        existing_label = None
        for col in label_cols:
            if col in df.columns:
                existing_label = col
                break
        
        if existing_label:
            print("\n" + "=" * 70)
            print(f"🏷️  EXISTING LABELS ('{existing_label}' column)")
            print("=" * 70)
            label_counts = df[existing_label].value_counts()
            for label, count in label_counts.items():
                pct = count / len(df) * 100
                print(f"{str(label):<20} {count:>8,} ({pct:>5.2f}%)")
        
        # Sample rows
        print("\n" + "=" * 70)
        print("👀 SAMPLE ROWS (first 3)")
        print("=" * 70)
        print(df.head(3).to_string())
        
        # Context columns that will be passed to model
        exclude_cols = ['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet', text_col] if text_col else []
        context_cols = [col for col in df.columns if col not in exclude_cols]
        
        print("\n" + "=" * 70)
        print("🔍 CONTEXT COLUMNS (will be passed to model)")
        print("=" * 70)
        for col in context_cols:
            print(f"  - {col}")
        
        # Cost estimation
        print("\n" + "=" * 70)
        print("💰 ANNOTATION COST ESTIMATE")
        print("=" * 70)
        batch_size = 15
        num_batches = (len(df) // batch_size) + 1
        
        print(f"Total Rows:        {len(df):,}")
        print(f"Batch Size:        {batch_size}")
        print(f"API Calls Needed:  ~{num_batches:,}")
        print(f"Model:             Llama 3.3 70B Instruct")
        print(f"Cost:              FREE ✅")
        print(f"Est. Time:         ~{num_batches * 2 / 60:.1f} minutes (with rate limiting)")
        
        # Recommendations
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATIONS")
        print("=" * 70)
        
        if len(df) > 1000:
            print("  ✓ Large dataset detected")
            print("    → Recommended: Start with sample (100-1000 rows)")
            print("    → Then run full dataset overnight")
        
        if text_col and text_lengths.max() > 500:
            print("  ⚠️  Some texts are very long (>500 chars)")
            print("    → May need longer processing time")
        
        if df.isna().sum().sum() > len(df) * 0.1:
            print("  ⚠️  Dataset has many missing values")
            print("    → Check data quality before annotation")
        
        print("\n" + "=" * 70)
        print("✅ Analysis complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review the information above")
        print("  2. Run test: python test_run.py (on 10 sample rows)")
        print("  3. Run full: python hate_speech_annotator.py")
        print("=" * 70 + "\n")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{csv_path}' not found")
        print("\nMake sure:")
        print("  1. File is in the current directory, OR")
        print("  2. Provide full path to the file")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try common names
        for name in ['input_dataset.csv', 'dataset.csv', 'data.csv']:
            if os.path.exists(name):
                csv_path = name
                break
        else:
            print("Usage: python analyze_dataset.py <your_dataset.csv>")
            print("\nOr place your CSV as 'input_dataset.csv' in current directory")
            sys.exit(1)
    
    analyze_dataset(csv_path)