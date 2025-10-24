#!/usr/bin/env python3
"""
CSV Encoding Fixer
Converts CSV files with encoding issues to UTF-8
"""

import pandas as pd
import sys
import os

def fix_csv_encoding(input_file, output_file=None):
    """Fix CSV encoding by converting to UTF-8"""
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return False
    
    if output_file is None:
        output_file = input_file.replace('.csv', '_utf8.csv')
    
    print(f"Reading: {input_file}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252', 'utf-16']
    df = None
    detected_encoding = None
    
    for encoding in encodings:
        try:
            print(f"   Trying {encoding}...", end='')
            df = pd.read_csv(input_file, encoding=encoding)
            detected_encoding = encoding
            print(f" ✓ Success!")
            break
        except Exception as e:
            print(f" ✗")
            continue
    
    if df is None:
        print("\nCould not read CSV with any standard encoding")
        print("\nTry these manual steps:")
        print("1. Open the CSV in Excel or LibreOffice")
        print("2. Save As → CSV UTF-8")
        print("3. Try again")
        return False
    
    print(f"\n✓ Detected encoding: {detected_encoding}")
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Save as UTF-8
    print(f"\nSaving to: {output_file}")
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Verify
    try:
        test_df = pd.read_csv(output_file, encoding='utf-8')
        print(f"✓ Verified: {len(test_df):,} rows")
        print(f"\nSuccessfully converted to UTF-8!")
        print(f"\nUse this file: {output_file}")
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_csv_encoding.py input.csv [output.csv]")
        print("\nExample:")
        print("  python fix_csv_encoding.py Twitter_Data.csv")
        print("  python fix_csv_encoding.py Twitter_Data.csv Twitter_Data_fixed.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("="*60)
    print("CSV ENCODING FIXER")
    print("="*60)
    
    success = fix_csv_encoding(input_file, output_file)
    
    sys.exit(0 if success else 1)