# Enhanced Training Script - User Guide

## Overview

The enhanced `main_train_enhanced.py` script provides two major improvements:

1. **Train on ALL datasets** in `data/raw/` automatically
2. **Incremental training** - add new datasets without retraining from scratch

---

## Quick Start

### First Time Training (All Datasets)

```bash
# Place all your CSV files in data/raw/
cp labeled_data.csv data/raw/
cp dataset2.csv data/raw/
cp dataset3.csv data/raw/

# Train on everything
python main_train_enhanced.py
```

This will:
- ✅ Automatically find ALL CSV files in `data/raw/`
- ✅ Combine them into one training set
- ✅ Remove duplicates
- ✅ Train models on the combined data
- ✅ Track which datasets were used

---

## Incremental Training

### Scenario: You already trained on `labeled_data.csv`, now want to add `dataset2.csv`

```bash
# Step 1: Add new dataset to data/raw/
cp dataset2.csv data/raw/

# Step 2: Run incremental training
python main_train_enhanced.py --incremental
```

**What happens:**
- ✅ Loads ONLY `dataset2.csv` (new data)
- ⏭️ Skips `labeled_data.csv` (already trained)
- ✅ Updates models with new data
- ✅ Saves training history

### Add Another Dataset Later

```bash
# Add dataset3.csv
cp dataset3.csv data/raw/

# Incremental train again
python main_train_enhanced.py --incremental
```

**Result:**
- ✅ Loads ONLY `dataset3.csv`
- ⏭️ Skips `labeled_data.csv` and `dataset2.csv`
- ✅ Models now trained on all 3 datasets

---

## How It Works

### Training History Tracking

The script creates a file: `saved_models/trained_datasets.txt`

```
labeled_data.csv
dataset2.csv
dataset3.csv
```

Each time you run `--incremental`, it:
1. Reads this file
2. Finds NEW datasets not in the list
3. Trains only on new data
4. Updates the list

### Force Full Retrain

If you want to retrain everything from scratch:

```bash
python main_train_enhanced.py --retrain
```

Or delete the tracking file:

```bash
rm saved_models/trained_datasets.txt
python main_train_enhanced.py
```

---

## Command-Line Options

### Basic Commands

```bash
# Train on all datasets (first time)
python main_train_enhanced.py

# Incremental training (add new datasets only)
python main_train_enhanced.py --incremental

# Force full retrain
python main_train_enhanced.py --retrain

# Show usage examples
python main_train_enhanced.py --usage

# List all datasets
python main_train_enhanced.py --list-datasets
```

### Advanced Options

```bash
# Skip certain phases
python main_train_enhanced.py --skip-phase1
python main_train_enhanced.py --skip-phase4

# Include deep learning models
python main_train_enhanced.py --phase5

# Include BERT (slow, ~30 min)
python main_train_enhanced.py --phase5 --use-bert

# Combine options
python main_train_enhanced.py --incremental --phase5
```

---

## Dataset Requirements

### CSV Format

Each CSV file in `data/raw/` must have:

**Required columns:**
- `tweet`: Text content (string)
- `class`: Label (integer: 0, 1, or 2)
  - 0 = Hate speech
  - 1 = Offensive language
  - 2 = Neither

**Optional columns:**
- Any other metadata (preserved in combined dataset)

**Example CSV:**

```csv
tweet,class
"I hate when my code doesn't work",2
"You're a fucking idiot",1
"I will kill you bitch",0
"Good morning everyone",2
```

### Valid Class Labels

- **0**: Hate speech
- **1**: Offensive language
- **2**: Neither

Rows with other values will be automatically filtered out.

---

## Complete Workflow Examples

### Example 1: Simple Addition

```bash
# Initial training
$ python main_train_enhanced.py
✓ Trained on: labeled_data.csv (24,783 rows)

# Add new dataset
$ cp dataset2.csv data/raw/

# Incremental update
$ python main_train_enhanced.py --incremental
✓ Found new dataset: dataset2.csv (10,000 rows)
✓ Training on 10,000 new samples
✓ Updated models
```

### Example 2: Multiple Additions

```bash
# Day 1: Initial training
$ python main_train_enhanced.py
✓ Datasets: labeled_data.csv

# Day 2: Add dataset2
$ cp dataset2.csv data/raw/
$ python main_train_enhanced.py --incremental
✓ New: dataset2.csv
✓ Total trained: labeled_data.csv, dataset2.csv

# Day 3: Add dataset3 and dataset4
$ cp dataset3.csv dataset4.csv data/raw/
$ python main_train_enhanced.py --incremental
✓ New: dataset3.csv, dataset4.csv
✓ Total trained: 4 datasets
```

### Example 3: Start Over

```bash
# Delete history
$ rm saved_models/trained_datasets.txt

# Retrain from scratch on all datasets
$ python main_train_enhanced.py
✓ Found 5 datasets in data/raw/
✓ Training on combined 50,000 rows
```

---

## Output Files

### After Training

```
saved_models/
  ├── trained_datasets.txt          # History of trained datasets
  ├── traditional_ml/
  │   ├── random_forest.pkl
  │   ├── xgboost.pkl
  │   └── ...
  └── deep_learning/
      ├── lstm_model.keras
      └── ...

saved_features/
  ├── feature_extractor.pkl
  ├── tfidf_char.pkl
  └── ...

results/
  ├── model_comparison.csv
  └── model_metadata.json
```

---

## Checking Training Status

### List All Datasets

```bash
$ python main_train_enhanced.py --list-datasets
```

**Output:**
```
📁 Datasets in data/raw/:
  📄 labeled_data.csv           (15,234 KB)
  📄 dataset2.csv               (5,432 KB)
  📄 dataset3.csv               (8,123 KB)

✅ Already trained on (2):
  ✓ labeled_data.csv
  ✓ dataset2.csv
```

### Check Training History

```bash
$ cat saved_models/trained_datasets.txt
```

---

## Tips and Best Practices

### 1. **Dataset Naming**

Use descriptive names:
```
data/raw/
  ├── twitter_2024_01.csv
  ├── reddit_comments.csv
  └── youtube_hate_speech.csv
```

### 2. **Incremental vs Full Training**

**Use Incremental When:**
- ✅ Adding small amounts of new data
- ✅ Existing model performs well
- ✅ Want to save time

**Use Full Retrain When:**
- ✅ Major data changes
- ✅ Data distribution shifts
- ✅ Model performance degrades

### 3. **Backup Before Incremental Training**

```bash
# Backup models before updating
cp -r saved_models saved_models_backup
python main_train_enhanced.py --incremental
```

### 4. **Monitor Performance**

After incremental training, check if accuracy improved:

```bash
# Before
Model Accuracy: 0.8523

# After incremental training
Model Accuracy: 0.8645  # ✅ Improved!
```

---

## Troubleshooting

### Problem: "No CSV files found"

**Solution:**
```bash
# Check data/raw exists
ls data/raw/

# Create if missing
mkdir -p data/raw

# Copy datasets
cp your_dataset.csv data/raw/
```

### Problem: "Skipping file - missing required columns"

**Solution:** Ensure CSV has `tweet` and `class` columns

```python
# Check your CSV
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.columns)  # Should include 'tweet' and 'class'
```

### Problem: Incremental training not finding new datasets

**Solution:** Check trained_datasets.txt

```bash
$ cat saved_models/trained_datasets.txt
labeled_data.csv
dataset2.csv

# Remove line if you want to retrain on specific dataset
```

---

## Advanced: Programmatic Usage

### Use in Python Scripts

```python
from main_train_enhanced import load_all_csv_datasets, IncrementalTrainer

# Load all datasets
combined_df = load_all_csv_datasets()
print(f"Total rows: {len(combined_df)}")

# Check training history
trainer = IncrementalTrainer()
new_datasets, trained = trainer.identify_new_datasets()
print(f"New: {new_datasets}")
print(f"Trained: {trained}")

# Incremental train
result = trainer.incremental_train(new_datasets_only=True)
```

---

## Comparison: Old vs Enhanced

### Old Way (`main_train.py`)

```bash
# Could only train on ONE file at a time
# Had to manually combine datasets
# No tracking of what was trained
# Full retrain every time
```

### Enhanced Way (`main_train_enhanced.py`)

```bash
# ✅ Automatically trains on ALL files in data/raw/
# ✅ Tracks training history
# ✅ Incremental updates (no full retrain needed)
# ✅ Combines datasets automatically
# ✅ Removes duplicates
```

---

## Summary

**Key Benefits:**

1. 🚀 **Automatic**: Just drop CSV files in `data/raw/`
2. ⚡ **Fast**: Incremental training saves time
3. 📊 **Smart**: Tracks what's been trained
4. 🔄 **Flexible**: Full retrain or incremental update
5. 🛡️ **Safe**: Preserves training history

**Most Common Commands:**

```bash
# First time
python main_train_enhanced.py

# Add new data
python main_train_enhanced.py --incremental

# Check status
python main_train_enhanced.py --list-datasets

# Start over
python main_train_enhanced.py --retrain
```

---

## Need Help?

Run this for examples:
```bash
python main_train_enhanced.py --usage
```

Or check the code comments in `main_train_enhanced.py` for detailed documentation.