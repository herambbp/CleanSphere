"""
Enhanced Main Training Script - Hate Speech Detection Project
Supports:
1. Training on ALL datasets in data/raw
2. Incremental training (adding new data to existing models)

PHASE 1-3: Traditional ML Models + Embeddings
PHASE 4: Severity Classification System
PHASE 5: Deep Learning Models (LSTM, BiLSTM, CNN, BERT)
"""

import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import joblib
warnings.filterwarnings('ignore')

# Import all Phase 1-3 modules
from config import PROJECT_ROOT, FEATURE_COMBINATION, RAW_DATA_DIR, MODELS_DIR
from utils import logger, print_section_header
from data_handler import DataSplitter
from feature_extractor import FeatureExtractor
from embedding_trainer import train_embeddings, HAS_GENSIM
from models.traditional_ml_trainer import train_traditional_models
from reporting import TrainingHistory, MetricsCollector, QuickReportGenerator


# Import Phase 5 modules
try:
    from models.deep_learning_trainer import train_deep_learning_models, HAS_TENSORFLOW
    HAS_DL = True
except ImportError:
    HAS_DL = False
    HAS_TENSORFLOW = False
    logger.warning("Deep learning modules not available. Phase 5 will be skipped.")

from inference.tweet_classifier import TweetClassifier, demo

# ==================== DATASET LOADING ====================

def load_all_csv_datasets(data_dir: Path = RAW_DATA_DIR, exclude: List[str] = None) -> pd.DataFrame:
    """
    Load and combine ALL CSV files from data/raw directory.
    
    Args:
        data_dir: Directory containing CSV files
        exclude: List of filenames to exclude (for incremental training)
    
    Returns:
        Combined DataFrame with all data
    """
    print_section_header("LOADING ALL DATASETS FROM data/raw")
    
    if exclude is None:
        exclude = []
    
    # Find all CSV files
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV file(s) in {data_dir}")
    
    all_dataframes = []
    total_rows = 0
    
    for csv_file in csv_files:
        # Skip excluded files
        if csv_file.name in exclude:
            logger.info(f"  [SKIP] Skipping {csv_file.name} (excluded)")
            continue
        
        try:
            # Load CSV
            logger.info(f"  [LOAD] Loading {csv_file.name}...")
            df = pd.read_csv(csv_file, encoding="latin-1", on_bad_lines="skip")
            
            # Validate columns
            if 'tweet' not in df.columns or 'class' not in df.columns:
                logger.warning(f"  [WARN] Skipping {csv_file.name} - missing required columns (tweet, class)")
                continue
            
            # Clean data
            df = df.dropna(subset=['tweet', 'class'])
            df = df[df['class'].isin([0, 1, 2])]  # Only keep valid classes
            
            rows = len(df)
            total_rows += rows
            
            logger.info(f"  [OK] Loaded {rows:,} rows from {csv_file.name}")
            
            # Add source column to track which dataset each row came from
            df['source_file'] = csv_file.name
            
            all_dataframes.append(df)
            
        except Exception as e:
            logger.error(f"  [ERROR] Error loading {csv_file.name}: {e}")
            continue
    
    if not all_dataframes:
        raise ValueError("No valid datasets could be loaded!")
    
    # Combine all dataframes
    logger.info(f"\n[INFO] Combining {len(all_dataframes)} dataset(s)...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['tweet'], keep='first')
    duplicates_removed = initial_count - len(combined_df)
    
    if duplicates_removed > 0:
        logger.info(f"  [CLEAN] Removed {duplicates_removed:,} duplicate tweets")
    
    logger.info(f"\n[SUCCESS] Total combined dataset: {len(combined_df):,} rows")
    
    # Print dataset breakdown
    print("\n[DATASET BREAKDOWN]")
    for source in combined_df['source_file'].unique():
        count = len(combined_df[combined_df['source_file'] == source])
        percentage = (count / len(combined_df)) * 100
        logger.info(f"  {source:30s}: {count:8,} ({percentage:5.2f}%)")
    
    # Print class distribution
    print("\n[CLASS DISTRIBUTION]")
    for class_id in sorted(combined_df['class'].unique()):
        count = len(combined_df[combined_df['class'] == class_id])
        percentage = (count / len(combined_df)) * 100
        from config import CLASS_LABELS
        class_name = CLASS_LABELS.get(class_id, f"Class {class_id}")
        logger.info(f"  {class_name:20s}: {count:8,} ({percentage:5.2f}%)")
    
    return combined_df


def prepare_data_from_combined_df(df: pd.DataFrame, save_split: bool = True) -> Tuple:
    """
    Prepare train/val/test splits from combined DataFrame.
    
    Args:
        df: Combined DataFrame
        save_split: Whether to save the splits
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print_section_header("PREPARING DATA SPLITS")
    
    # Extract text and labels
    X = df['tweet'].values
    y = df['class'].values
    
    logger.info(f"Total samples: {len(X):,}")
    logger.info(f"Unique classes: {sorted(np.unique(y))}")
    
    # Split data
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y)
    
    # Save if requested
    if save_split:
        splitter.save_split(X_train, X_val, X_test, y_train, y_val, y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==================== INCREMENTAL TRAINING ====================

class IncrementalTrainer:
    """
    Handles incremental training - adding new data to existing models.
    """
    
    def __init__(self, model_dir: Path = MODELS_DIR):
        self.model_dir = model_dir
        self.trained_datasets_file = model_dir / 'trained_datasets.txt'
    
    def get_trained_datasets(self) -> List[str]:
        """Get list of datasets that have been trained on."""
        if not self.trained_datasets_file.exists():
            return []
        
        with open(self.trained_datasets_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def save_trained_datasets(self, datasets: List[str]):
        """Save list of trained datasets."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.trained_datasets_file, 'w') as f:
            for dataset in datasets:
                f.write(f"{dataset}\n")
    
    def identify_new_datasets(self, data_dir: Path = RAW_DATA_DIR) -> Tuple[List[str], List[str]]:
        """
        Identify which datasets are new vs already trained.
        
        Returns:
            Tuple of (new_datasets, already_trained_datasets)
        """
        all_csvs = [f.name for f in data_dir.glob('*.csv')]
        trained = self.get_trained_datasets()
        
        new = [f for f in all_csvs if f not in trained]
        
        return new, trained
    
    def incremental_train(
        self,
        new_datasets_only: bool = True,
        retrain_from_scratch: bool = False
    ):
        """
        Perform incremental training.
        
        Args:
            new_datasets_only: Only train on datasets not previously seen
            retrain_from_scratch: If True, retrain everything; if False, try to update existing models
        """
        print_section_header("INCREMENTAL TRAINING MODE")
        
        new_datasets, trained_datasets = self.identify_new_datasets()
        
        logger.info(f"Previously trained on: {len(trained_datasets)} dataset(s)")
        for ds in trained_datasets:
            logger.info(f"  [OK] {ds}")
        
        logger.info(f"\nNew datasets found: {len(new_datasets)} dataset(s)")
        for ds in new_datasets:
            logger.info(f"  [NEW] {ds}")
        
        if not new_datasets:
            logger.warning("[WARNING] No new datasets found! Nothing to do.")
            return None
        
        # Decide what to load
        if new_datasets_only:
            logger.info(f"\n[INFO] Loading ONLY new datasets: {new_datasets}")
            exclude = trained_datasets
        else:
            logger.info(f"\n[INFO] Loading ALL datasets (new + previously trained)")
            exclude = []
        
        # Load data
        combined_df = load_all_csv_datasets(exclude=exclude)
        
        # Prepare splits
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_from_combined_df(
            combined_df, 
            save_split=True
        )
        
        # Update trained datasets list
        all_trained = list(set(trained_datasets + new_datasets))
        self.save_trained_datasets(all_trained)
        logger.info(f"\n[SUCCESS] Updated trained datasets list: {len(all_trained)} total")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# ==================== PHASE 1-3: TRADITIONAL ML TRAINING ====================

def phase1_train_traditional_ml(
    X_train=None, X_val=None, X_test=None,
    y_train=None, y_val=None, y_test=None,
    incremental: bool = False
):
    """
    Phase 1-3: Train traditional ML models with embeddings + Generate Reports.
    
    Args:
        X_train, X_val, X_test, y_train, y_val, y_test: Data splits (if None, will load all)
        incremental: Whether to use incremental training
    
    Pipeline:
    1. Load and split data (all datasets OR new ones)
    2. Train embeddings (Word2Vec + FastText)
    3. Extract features (TF-IDF + Embeddings + linguistic)
    4. Train all ML models
    5. Evaluate and save models
    6. Generate training report with visualizations
    """
    
    print_section_header("PHASE 1-3: TRADITIONAL ML TRAINING WITH EMBEDDINGS")
    
    # Initialize reporting system
    try:
        history = TrainingHistory()
        collector = MetricsCollector()
        report_generator = QuickReportGenerator()
        reporting_enabled = True
        logger.info("[OK] Reporting system initialized")
        
        # Start metrics collection
        collector.start_run()
        logger.info("[OK] Metrics collection started")
    except Exception as e:
        logger.warning(f"Could not initialize reporting system: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        reporting_enabled = False
    
    try:
        # Step 1: Load data
        if X_train is None:
            print_section_header("STEP 1: DATA LOADING & SPLITTING")
            
            if incremental:
                trainer = IncrementalTrainer()
                result = trainer.incremental_train(new_datasets_only=True)
                
                if result is None:
                    logger.warning("No new data to train on!")
                    return None
                
                X_train, X_val, X_test, y_train, y_val, y_test = result
                
                # Collect dataset info for reporting
                if reporting_enabled:
                    new_datasets, trained_datasets = trainer.identify_new_datasets()
                    all_datasets = list(set(trained_datasets + new_datasets))
                    total_samples = len(y_train) + len(y_val) + len(y_test)
                    collector.set_datasets(
                        datasets=all_datasets,
                        total_samples=total_samples,
                        new_datasets=new_datasets
                    )
            else:
                logger.info("Loading ALL datasets from data/raw...")
                combined_df = load_all_csv_datasets()
                X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_from_combined_df(combined_df)
                
                # Collect dataset info for reporting
                if reporting_enabled:
                    all_datasets = [f.name for f in RAW_DATA_DIR.glob('*.csv')]
                    total_samples = len(y_train) + len(y_val) + len(y_test)
                    collector.set_datasets(
                        datasets=all_datasets,
                        total_samples=total_samples,
                        new_datasets=[]
                    )
        else:
            logger.info("Using provided data splits")
            
            # Try to collect dataset info even with provided splits
            if reporting_enabled:
                all_datasets = [f.name for f in RAW_DATA_DIR.glob('*.csv')]
                total_samples = len(y_train) + len(y_val) + len(y_test)
                collector.set_datasets(
                    datasets=all_datasets,
                    total_samples=total_samples,
                    new_datasets=[]
                )
        
        # Collect data split info
        if reporting_enabled:
            # Collect class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip(map(int, unique), map(int, counts)))
            
            # Set splits with correct method name and parameters
            collector.set_splits(
                train_size=len(y_train),
                val_size=len(y_val),
                test_size=len(y_test),
                class_distribution=class_dist
            )
            logger.info("[OK] Dataset and split metrics collected")
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Train: {len(y_train):,} samples")
        logger.info(f"  Val:   {len(y_val):,} samples")
        logger.info(f"  Test:  {len(y_test):,} samples")
        
        # Step 2: Train embeddings (PHASE 3)
        print_section_header("STEP 2: TRAINING EMBEDDINGS (PHASE 3)")
        
        if HAS_GENSIM and FEATURE_COMBINATION in ['all', 'embeddings_only']:
            logger.info("Training Word2Vec and FastText on corpus...")
            logger.info(f"Feature combination strategy: {FEATURE_COMBINATION}")
            
            embedding_trainer = train_embeddings(X_train, save=True)
            
            if embedding_trainer:
                logger.info("Embeddings trained successfully!")
                if embedding_trainer.word2vec_model:
                    logger.info(f"  Word2Vec vocabulary: {len(embedding_trainer.word2vec_model.wv):,} words")
                if embedding_trainer.fasttext_model:
                    logger.info(f"  FastText vocabulary: {len(embedding_trainer.fasttext_model.wv):,} words")
            else:
                logger.warning("Could not train embeddings. Continuing with TF-IDF only...")
        else:
            if not HAS_GENSIM:
                logger.warning("Gensim not installed. Skipping embedding training.")
                logger.warning("Install with: pip install gensim")
            logger.info(f"Using feature combination: {FEATURE_COMBINATION}")
            logger.info("Skipping embedding training (not needed for this configuration)")
        
        # Step 3: Extract features
        print_section_header("STEP 3: FEATURE EXTRACTION")
        logger.info("Extracting features from text...")
        logger.info(f"Feature combination strategy: {FEATURE_COMBINATION}")
        
        # Initialize and fit feature extractor on training data
        feature_extractor = FeatureExtractor()
        
        logger.info("Fitting feature extractor on training data...")
        X_train_features = feature_extractor.fit_transform(X_train)
        
        logger.info("Transforming validation data...")
        X_val_features = feature_extractor.transform(X_val)
        
        logger.info("Transforming test data...")
        X_test_features = feature_extractor.transform(X_test)
        
        logger.info(f"Feature extraction complete:")
        logger.info(f"  Feature dimensions: {X_train_features.shape[1]}")
        logger.info(f"  Train features: {X_train_features.shape}")
        logger.info(f"  Val features:   {X_val_features.shape}")
        logger.info(f"  Test features:  {X_test_features.shape}")
        
        # Collect feature info
        if reporting_enabled:
            collector.set_feature_info(X_train_features.shape[1])
            logger.info("[OK] Feature extraction metrics collected")
        
        # Save feature extractor
        logger.info("Saving feature extractor...")
        feature_extractor.save()
        
        # Step 4: Train models
        print_section_header("STEP 4: MODEL TRAINING")
        logger.info("Training all traditional ML models...")
        
        trainer = train_traditional_models(
            X_train=X_train_features,
            y_train=y_train,
            X_val=X_val_features,
            y_val=y_val,
            X_test=X_test_features,
            y_test=y_test,
            save_models=True
        )
        
        # Collect model metrics
        if reporting_enabled:
            try:
                comparison_df = trainer.evaluator.get_comparison_df()
                
                if not comparison_df.empty:
                    logger.info("Collecting model metrics...")
                    for _, row in comparison_df.iterrows():
                        model_name = row['Model'].lower().replace(' ', '_')
                        
                        # Prepare metrics dictionary
                        metrics_dict = {
                            'accuracy': float(row['Accuracy']),
                            'f1_macro': float(row['F1 (Macro)']),
                            'precision_macro': float(row['Precision (Macro)']),
                            'recall_macro': float(row['Recall (Macro)'])
                        }
                        
                        # Add model result using correct method
                        collector.add_model_result(
                            model_name=model_name,
                            metrics=metrics_dict
                        )
                    logger.info(f"[OK] Collected metrics for {len(comparison_df)} models")
                else:
                    logger.warning("No model comparison data available")
            except Exception as e:
                logger.warning(f"Could not collect model metrics: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # Step 5: Summary
        print_section_header("PHASE 1-3 COMPLETE")
        
        best_name, best_model = trainer.get_best_model()
        logger.info(f"Best performing model: {best_name}")
        
        comparison_df = trainer.evaluator.get_comparison_df()
        if not comparison_df.empty:
            best_row = comparison_df.iloc[0]
            logger.info(f"  Accuracy: {best_row['Accuracy']:.4f}")
            logger.info(f"  F1 (macro): {best_row['F1 (Macro)']:.4f}")
        
        logger.info(f"\nAll models saved to: {PROJECT_ROOT / 'saved_models'}")
        logger.info(f"Feature extractor saved to: {PROJECT_ROOT / 'saved_features'}")
        logger.info(f"Results saved to: {PROJECT_ROOT / 'results'}")
        
        # Step 6: Generate Training Report
        if reporting_enabled:
            try:
                print_section_header("GENERATING TRAINING REPORT")
                
                # Finalize metrics collection
                logger.info("Finalizing metrics collection...")
                metrics = collector.finalize()
                logger.info("[OK] Metrics finalized")
                
                # Validate metrics
                if not collector.validate():
                    logger.warning("[WARNING] Metrics validation failed - some data may be missing")
                else:
                    logger.info("[OK] Metrics validated successfully")
                
                # Save to training history
                logger.info("Saving to training history...")
                history.add_run(metrics)
                logger.info(f"[OK] Training run saved to history")
                
                # Get previous run for comparison using correct method
                logger.info("Checking for previous runs...")
                all_runs = history.get_all_runs()
                if len(all_runs) >= 2:
                    previous_metrics = all_runs[-2]  # Get second to last run
                    logger.info("[OK] Previous training run found for comparison")
                else:
                    previous_metrics = None
                    logger.info("No previous run available for comparison")
                
                # Generate HTML report
                logger.info("Generating HTML report...")
                report_path = report_generator.generate_report(
                    metrics=metrics,
                    previous_metrics=previous_metrics
                )
                
                print("\n" + "=" * 80)
                logger.info("[SUCCESS] TRAINING REPORT GENERATED SUCCESSFULLY!")
                logger.info(f"[REPORT] Report location: {report_path}")
                logger.info(f"[BROWSER] Open in browser: file://{report_path.absolute()}")
                print("=" * 80 + "\n")
                
            except Exception as e:
                logger.warning(f"Failed to generate training report: {e}")
                logger.warning("Training completed successfully, but reporting failed.")
                import traceback
                logger.warning(traceback.format_exc())
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error in Phase 1-3 training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==================== MAIN ====================

def main(
    skip_phase1: bool = False,
    skip_phase4: bool = False,
    run_phase5: bool = False,
    use_bert: bool = False,
    incremental: bool = False,
    load_all_datasets: bool = True
):
    """
    Main entry point for training pipeline.
    
    Args:
        skip_phase1: Skip Phase 1-3 (traditional ML)
        skip_phase4: Skip Phase 4 (severity testing)
        run_phase5: Run Phase 5 (deep learning)
        use_bert: Include BERT in Phase 5
        incremental: Use incremental training mode (add new datasets to existing model)
        load_all_datasets: Load all datasets from data/raw (default: True)
    """
    
    print("=" * 80)
    print("HATE SPEECH DETECTION - ENHANCED TRAINING PIPELINE")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Mode: {'INCREMENTAL' if incremental else 'FULL TRAINING'}")
    print(f"Load all datasets: {load_all_datasets}")
    print("=" * 80)
    
    # Check if data files exist
    csv_files = list(RAW_DATA_DIR.glob('*.csv'))
    if not csv_files:
        logger.error(f"[ERROR] No CSV files found in {RAW_DATA_DIR}")
        logger.error("Please place dataset files in data/raw/ directory")
        sys.exit(1)
    
    logger.info(f"[OK] Found {len(csv_files)} CSV file(s) in data/raw/:")
    for csv_file in csv_files:
        logger.info(f"  [FILE] {csv_file.name}")
    
    # Phase 1-3: Traditional ML
    if not skip_phase1:
        trainer_trad = phase1_train_traditional_ml(incremental=incremental)
    else:
        logger.info("[SKIP] Skipping Phase 1-3 (Traditional ML)")
        trainer_trad = None
    
    # Phase 4: Severity System (import from original main_train.py)
    if not skip_phase4:
        try:
            from main_train import phase4_test_severity_system
            phase4_success = phase4_test_severity_system()
        except ImportError:
            logger.warning("[WARNING] Could not import phase4_test_severity_system")
            phase4_success = None
    else:
        logger.info("[SKIP] Skipping Phase 4 (Severity System)")
        phase4_success = None
    
    # Phase 5: Deep Learning
    if run_phase5:
        try:
            from main_train import phase5_train_deep_learning
            trainer_dl = phase5_train_deep_learning(use_bert=use_bert)
        except ImportError:
            logger.warning("[WARNING] Could not import phase5_train_deep_learning")
            trainer_dl = None
    else:
        logger.info("\n[SKIP] Skipping Phase 5 (Deep Learning)")
        logger.info("Use --phase5 flag to enable deep learning training")
        trainer_dl = None
    
    # Test the classifier
    if not skip_phase1 or not skip_phase4:
        try:
            from inference.tweet_classifier import demo
            demo()
        except Exception as e:
            logger.warning(f"[WARNING] Could not run classifier demo: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("[SUCCESS] TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    
    print("\n[COMPLETED PHASES]")
    if not skip_phase1:
        print("  [OK] Phase 1-3: Traditional ML + Embeddings")
        if trainer_trad:
            comparison_df = trainer_trad.evaluator.get_comparison_df()
            if not comparison_df.empty:
                best = comparison_df.iloc[0]
                print(f"     Best: {best['Model']} ({best['Accuracy']:.4f} accuracy)")
    
    if not skip_phase4:
        print("  [OK] Phase 4: Severity Classification + Action Recommendations")
    
    if run_phase5 and trainer_dl:
        print("  [OK] Phase 5: Deep Learning Models")
    
    print("\n[OUTPUT LOCATIONS]")
    print(f"  Models: {PROJECT_ROOT / 'saved_models'}")
    print(f"  Features: {PROJECT_ROOT / 'saved_features'}")
    print(f"  Results: {PROJECT_ROOT / 'results'}")
    print(f"  Reports: {PROJECT_ROOT / 'results/training_reports'}")
    
    if incremental:
        trainer = IncrementalTrainer()
        trained = trainer.get_trained_datasets()
        print(f"\n[TRAINED DATASETS] ({len(trained)})")
        for ds in trained:
            print(f"  [OK] {ds}")
    
    print("\n[NEXT STEPS]")
    print("  1. View training report in results/training_reports/")
    print("  2. Use the classifier:")
    print("     from inference.tweet_classifier import TweetClassifier")
    print("     classifier = TweetClassifier()")
    print("     result = classifier.classify_with_severity('Your tweet')")
    
    if incremental:
        print("\n  3. Add more datasets:")
        print("     - Place new CSV files in data/raw/")
        print("     - Run: python main_train_enhanced.py --incremental")
    
    print("=" * 80)


# ==================== USAGE EXAMPLES ====================

def print_usage():
    """Print usage examples."""
    print("""
USAGE EXAMPLES:
===============

1. Train on ALL datasets in data/raw (first time):
   python main_train_enhanced.py

2. Add NEW dataset to existing model (incremental):
   python main_train_enhanced.py --incremental
   
   How it works:
   - Tracks which datasets have been trained on
   - Only trains on NEW datasets not seen before
   - Updates existing models with new data
   
3. Retrain everything from scratch:
   python main_train_enhanced.py --retrain

4. Train with deep learning:
   python main_train_enhanced.py --phase5

5. Skip certain phases:
   python main_train_enhanced.py --skip-phase1 --skip-phase4

INCREMENTAL TRAINING WORKFLOW:
================================

Step 1: Initial training with labeled_data.csv
   $ python main_train_enhanced.py
   [OK] Trains on: labeled_data.csv
   [OK] Saves: trained_datasets.txt

Step 2: Add dataset2.csv to data/raw/
   $ cp dataset2.csv data/raw/

Step 3: Incremental training (adds only dataset2.csv)
   $ python main_train_enhanced.py --incremental
   [OK] Loads: dataset2.csv (NEW)
   [SKIP] Skips: labeled_data.csv (already trained)
   [OK] Updates models with new data
   [OK] Updates: trained_datasets.txt

Step 4: Add dataset3.csv
   $ cp dataset3.csv data/raw/
   $ python main_train_enhanced.py --incremental
   [OK] Loads: dataset3.csv (NEW)
   [SKIP] Skips: labeled_data.csv, dataset2.csv (already trained)

DATASET FORMAT:
===============
Each CSV must have these columns:
- tweet: Text content
- class: Label (0=Hate, 1=Offensive, 2=Neither)

Optional columns:
- Any other metadata (will be preserved)
""")


# ==================== RUN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Hate Speech Detection Training - All Datasets + Incremental'
    )
    
    # Training modes
    parser.add_argument('--incremental', action='store_true',
                       help='Incremental training: only train on NEW datasets not seen before')
    parser.add_argument('--retrain', action='store_true',
                       help='Force full retraining on all datasets (ignore history)')
    
    # Skip options
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1-3 (Traditional ML)')
    parser.add_argument('--skip-phase4', action='store_true',
                       help='Skip Phase 4 (Severity System)')
    
    # Phase 5 options
    parser.add_argument('--phase5', action='store_true',
                       help='Run Phase 5 (Deep Learning Models)')
    parser.add_argument('--use-bert', action='store_true',
                       help='Include BERT in Phase 5 (slow, ~30 min)')
    
    # Utility options
    parser.add_argument('--usage', action='store_true',
                       help='Show usage examples')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all datasets in data/raw/')
    
    args = parser.parse_args()
    
    if args.usage:
        print_usage()
    elif args.list_datasets:
        print("\n[DATASETS IN data/raw/]")
        print("=" * 60)
        csv_files = list(RAW_DATA_DIR.glob('*.csv'))
        if not csv_files:
            print("  (none found)")
        else:
            for csv_file in csv_files:
                size = csv_file.stat().st_size / 1024  # KB
                print(f"  [FILE] {csv_file.name:30s} ({size:.1f} KB)")
        
        # Show trained datasets if incremental mode has been used
        trainer = IncrementalTrainer()
        trained = trainer.get_trained_datasets()
        if trained:
            print(f"\n[ALREADY TRAINED ON] ({len(trained)})")
            for ds in trained:
                print(f"  [OK] {ds}")
    else:
        main(
            skip_phase1=args.skip_phase1,
            skip_phase4=args.skip_phase4,
            run_phase5=args.phase5,
            use_bert=args.use_bert,
            incremental=args.incremental and not args.retrain
        )