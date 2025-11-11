#!/usr/bin/env python3
"""
Calculate Distribution - Count hate speech, offensive, and neither categories
"""

import pandas as pd
import sys
import os


def calculate_distribution(csv_path: str):
    """Calculate and display the distribution of categories in the dataset"""
    
    print("=" * 80)
    print("📊 HATE SPEECH DATASET DISTRIBUTION CALCULATOR")
    print("=" * 80)
    
    try:
        # Load dataset
        print(f"\n📂 Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded successfully\n")
        
        # Check if required columns exist
        required_cols = ['hate_speech', 'offensive_language', 'neither']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Error: Missing required columns: {missing_cols}")
            print(f"\nAvailable columns: {list(df.columns)}")
            sys.exit(1)
        
        # Calculate totals
        total_rows = len(df)
        hate_speech_total = df['hate_speech'].sum()
        offensive_total = df['offensive_language'].sum()
        neither_total = df['neither'].sum()
        
        # Total votes across all categories
        total_votes = hate_speech_total + offensive_total + neither_total
        
        # Calculate percentages
        hate_speech_pct = (hate_speech_total / total_votes * 100) if total_votes > 0 else 0
        offensive_pct = (offensive_total / total_votes * 100) if total_votes > 0 else 0
        neither_pct = (neither_total / total_votes * 100) if total_votes > 0 else 0
        
        # Display results
        print("=" * 80)
        print("📈 DISTRIBUTION BY VOTE COUNTS")
        print("=" * 80)
        print(f"\n{'Category':<25} {'Count':<15} {'Percentage':<15} {'Bar'}")
        print("-" * 80)
        
        # Hate Speech
        bar_hate = "█" * int(hate_speech_pct / 2)
        print(f"{'Hate Speech':<25} {hate_speech_total:>10,}     {hate_speech_pct:>6.2f}%     {bar_hate}")
        
        # Offensive Language
        bar_offensive = "█" * int(offensive_pct / 2)
        print(f"{'Offensive Language':<25} {offensive_total:>10,}     {offensive_pct:>6.2f}%     {bar_offensive}")
        
        # Neither
        bar_neither = "█" * int(neither_pct / 2)
        print(f"{'Neither':<25} {neither_total:>10,}     {neither_pct:>6.2f}%     {bar_neither}")
        
        print("-" * 80)
        print(f"{'TOTAL VOTES':<25} {total_votes:>10,}     {100.00:>6.2f}%")
        
        # If 'class' column exists, show distribution by final classification
        if 'class' in df.columns:
            print("\n" + "=" * 80)
            print("🏷️  DISTRIBUTION BY FINAL CLASSIFICATION (class column)")
            print("=" * 80)
            
            class_mapping = {
                0: 'Hate Speech',
                1: 'Offensive Language',
                2: 'Neither'
            }
            
            class_counts = df['class'].value_counts().sort_index()
            
            print(f"\n{'Category':<25} {'Row Count':<15} {'Percentage':<15} {'Bar'}")
            print("-" * 80)
            
            for class_num, count in class_counts.items():
                class_name = class_mapping.get(class_num, f'Unknown ({class_num})')
                pct = (count / total_rows * 100)
                bar = "█" * int(pct / 2)
                print(f"{class_name:<25} {count:>10,}     {pct:>6.2f}%     {bar}")
            
            print("-" * 80)
            print(f"{'TOTAL ROWS':<25} {total_rows:>10,}     {100.00:>6.2f}%")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("📋 SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total Rows in Dataset:           {total_rows:>10,}")
        print(f"Total Votes Cast (all rows):     {total_votes:>10,}")
        print(f"Average Votes per Row:           {total_votes/total_rows:>10.2f}")
        
        if 'count' in df.columns:
            total_count = df['count'].sum()
            avg_count = df['count'].mean()
            print(f"Total Count (from 'count' col):  {total_count:>10,}")
            print(f"Average Count per Row:           {avg_count:>10.2f}")
        
        # Additional insights
        print("\n" + "=" * 80)
        print("💡 INSIGHTS")
        print("=" * 80)
        
        # Determine which category is most common
        categories = {
            'Hate Speech': hate_speech_total,
            'Offensive Language': offensive_total,
            'Neither': neither_total
        }
        most_common = max(categories, key=categories.get)
        least_common = min(categories, key=categories.get)
        
        print(f"  • Most common category:  {most_common} ({categories[most_common]:,} votes)")
        print(f"  • Least common category: {least_common} ({categories[least_common]:,} votes)")
        
        # Check for class imbalance
        if max(categories.values()) > 2 * min(categories.values()):
            print(f"  • ⚠️  Significant class imbalance detected")
            print(f"     Consider this when training models or analyzing results")
        else:
            print(f"  • ✓ Dataset is relatively balanced")
        
        print("\n" + "=" * 80)
        print("✅ Calculation complete!")
        print("=" * 80 + "\n")
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{csv_path}' not found")
        print("\nMake sure:")
        print("  1. File is in the current directory, OR")
        print("  2. Provide full path to the file")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error calculating distribution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try common names
        for name in ['input_dataset.csv', 'dataset.csv', 'data.csv', 'labeled_data.csv']:
            if os.path.exists(name):
                csv_path = name
                break
        else:
            print("Usage: python calculate_distribution.py <your_dataset.csv>")
            print("\nOr place your CSV as 'input_dataset.csv' in current directory")
            sys.exit(1)
    
    calculate_distribution(csv_path)