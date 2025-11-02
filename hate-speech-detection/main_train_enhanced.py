"""
Enhanced Main Training Script - Hate Speech Detection Project
COMPLETE VERSION with Traditional DL + Heavy GPU BERT + Tokenizer Fix

Supports:
1. Training on ALL datasets in data/raw
2. Incremental training (adding new data to existing models)
3. Traditional Deep Learning (LSTM, BiLSTM, CNN, Basic BERT)
4. Heavy GPU BERT models (BERT-Large, RoBERTa-Large, etc.)

PHASE 1-3: Traditional ML Models + Embeddings
PHASE 4: Severity Classification System
PHASE 5A: Traditional Deep Learning (LSTM, BiLSTM, CNN, Basic BERT)
PHASE 5B: Heavy GPU BERT (BERT-Large, RoBERTa-Large, etc.)
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


# Import Phase 5A modules - Traditional Deep Learning
try:
    from models.deep_learning_trainer import train_deep_learning_models, HAS_TENSORFLOW
    HAS_TRADITIONAL_DL = True
except ImportError:
    HAS_TRADITIONAL_DL = False
    HAS_TENSORFLOW = False
    logger.warning("Traditional deep learning modules not available.")

# Import Phase 5B modules - Heavy GPU BERT
try:
    from bert_integration import (
        train_heavy_gpu_bert,
        HeavyGPUBERTTrainer,
        quick_train_bert_large,
        quick_train_roberta_large,
        train_all_large_models,
        HAS_HEAVY_BERT
    )
    logger.info("[OK] Heavy GPU BERT modules loaded successfully")
except ImportError:
    HAS_HEAVY_BERT = False
    logger.warning("Heavy GPU BERT modules not available. Install: pip install torch transformers")

from inference.tweet_classifier import TweetClassifier, demo

# ==================== TOKENIZER FIX ====================

def fix_tokenizer_automatically():
    """
    Automatically fix tokenizer before training traditional deep learning models.
    This ensures LSTM/BiLSTM/CNN models can load the tokenizer properly.
    """
    print_section_header("CHECKING AND FIXING TOKENIZER")
    
    tokenizer_path = Path('saved_models/deep_learning/tokenizer.pkl')
    
    # Check if tokenizer exists
    if not tokenizer_path.exists():
        logger.info("[INFO] No existing tokenizer found. Will be created during training.")
        return True
    
    # Try to load and validate tokenizer
    logger.info("Validating existing tokenizer...")
    try:
        tokenizer_data = joblib.load(tokenizer_path)
        
        # Check if it's the new format (TextTokenizer wrapper)
        if hasattr(tokenizer_data, 'tokenizer') and hasattr(tokenizer_data, 'is_fitted'):
            logger.info("[OK] Tokenizer is already in correct format")
            return True
        
        # Check if it's the old format (dict with word_index)
        if isinstance(tokenizer_data, dict) and 'word_index' in tokenizer_data:
            logger.warning("[WARN] Tokenizer is in old format. Fixing...")
            
            # Import required modules
            from tensorflow.keras.preprocessing.text import Tokenizer
            try:
                from models.deep_learning.text_tokenizer import TextTokenizer
            except ImportError:
                logger.error("[ERROR] Cannot import TextTokenizer. Please ensure models/deep_learning/text_tokenizer.py exists")
                return False
            
            # Extract data
            word_index = tokenizer_data['word_index']
            config = tokenizer_data.get('config', {})
            tokenizer_config = tokenizer_data.get('tokenizer_config', {})
            
            logger.info(f"  Vocab size: {len(word_index)}")
            logger.info(f"  Max length: {config.get('max_length', 100)}")
            
            # Reconstruct Keras Tokenizer
            keras_tokenizer = Tokenizer(
                num_words=config.get('vocab_size', 20000),
                oov_token=tokenizer_config.get('oov_token', '<OOV>'),
                filters=tokenizer_config.get('filters', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
                lower=tokenizer_config.get('lower', True)
            )
            
            # Set word_index manually
            keras_tokenizer.word_index = word_index
            keras_tokenizer.index_word = tokenizer_data.get('index_word', {i: w for w, i in word_index.items()})
            
            # Create TextTokenizer wrapper
            text_tokenizer = TextTokenizer(
                vocab_size=config.get('vocab_size', 20000),
                max_length=config.get('max_length', 100)
            )
            text_tokenizer.tokenizer = keras_tokenizer
            text_tokenizer.is_fitted = True
            
            # Test tokenizer
            test_texts = ["I hate you", "Hello world"]
            sequences = text_tokenizer.texts_to_padded_sequences(test_texts)
            logger.info(f"  Test successful! Output shape: {sequences.shape}")
            
            # Save fixed tokenizer
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(text_tokenizer, tokenizer_path)
            logger.info(f"[SUCCESS] Fixed tokenizer saved to: {tokenizer_path}")
            
            return True
        
        logger.warning("[WARN] Tokenizer format is unrecognized but will try to use it")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to validate/fix tokenizer: {e}")
        logger.warning("[WARN] Deep learning models may fail. Consider deleting tokenizer.pkl and retraining.")
        import traceback
        logger.warning(traceback.format_exc())
        return False


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


# ==================== PHASE 5A: TRADITIONAL DEEP LEARNING ====================

def phase5a_train_traditional_dl(
    X_train=None, X_val=None, X_test=None,
    y_train=None, y_val=None, y_test=None,
    use_bert: bool = False
):
    """
    Phase 5A: Train traditional deep learning models (LSTM, BiLSTM, CNN, Basic BERT).
    
    Args:
        X_train, X_val, X_test, y_train, y_val, y_test: Data splits
        use_bert: Whether to include Basic BERT (not Heavy GPU BERT)
    
    Returns:
        DeepLearningTrainer instance or None
    """
    print_section_header("PHASE 5A: TRADITIONAL DEEP LEARNING (LSTM, BiLSTM, CNN)")
    
    if not HAS_TRADITIONAL_DL:
        logger.error("[ERROR] Traditional deep learning modules not available!")
        logger.error("Install with: pip install tensorflow keras")
        return None
    
    # Check if data is provided
    if X_train is None or y_train is None:
        logger.error("[ERROR] Training data not provided!")
        logger.error("Run Phase 1-3 first or provide data splits")
        return None
    
    # Fix tokenizer before training
    logger.info("\n[TOKENIZER] Checking tokenizer before training...")
    tokenizer_ok = fix_tokenizer_automatically()
    
    if not tokenizer_ok:
        logger.warning("[WARN] Tokenizer check failed but will attempt training anyway")
    
    logger.info(f"\n[INFO] Training traditional deep learning models")
    logger.info(f"[INFO] Models: LSTM, BiLSTM, CNN" + (" + Basic BERT" if use_bert else ""))
    logger.info(f"[INFO] Training samples: {len(X_train):,}")
    logger.info(f"[INFO] Validation samples: {len(X_val):,}")
    logger.info(f"[INFO] Test samples: {len(X_test):,}")
    
    # Train models
    try:
        trainer = train_deep_learning_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            use_bert=use_bert
        )
        
        if trainer:
            logger.info("\n[SUCCESS] Phase 5A Traditional DL training complete!")
            
            # Get best model
            try:
                best_model_name = trainer.get_best_model_name()
                best_metrics = trainer.get_best_metrics()
                logger.info(f"[BEST MODEL] {best_model_name}")
                logger.info(f"[TEST ACCURACY] {best_metrics.get('test_accuracy', 0):.4f}")
            except:
                logger.warning("Could not retrieve best model info")
        
        return trainer
        
    except Exception as e:
        logger.error(f"[ERROR] Phase 5A training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== PHASE 5B: HEAVY GPU BERT TRAINING ====================

def phase5b_train_heavy_gpu_bert(
    X_train=None, X_val=None, X_test=None,
    y_train=None, y_val=None, y_test=None,
    models: List[str] = None,
    use_ensemble: bool = False
):
    """
    Phase 5B: Train Heavy GPU BERT models (BERT-Large, RoBERTa-Large, etc.)
    
    Args:
        X_train, X_val, X_test, y_train, y_val, y_test: Data splits
        models: List of models to train (default: ['bert-large'])
        use_ensemble: Whether to use ensemble prediction
    
    Returns:
        HeavyGPUBERTTrainer instance
    """
    print_section_header("PHASE 5B: HEAVY GPU BERT TRAINING")
    
    if not HAS_HEAVY_BERT:
        logger.error("[ERROR] Heavy GPU BERT modules not available!")
        logger.error("Install with: pip install torch transformers")
        logger.error("Make sure bert_model_heavy_gpu.py and bert_integration.py are in project root")
        return None
    
    # Check if data is provided
    if X_train is None or y_train is None:
        logger.error("[ERROR] Training data not provided!")
        logger.error("Run Phase 1-3 first or provide data splits")
        return None
    
    # Set default models if not specified
    if models is None:
        models = ['bert-large']
        logger.info(f"Using default models: {models}")
    
    logger.info(f"\n[INFO] Training {len(models)} Heavy GPU model(s): {', '.join(models)}")
    logger.info(f"[INFO] Training samples: {len(X_train):,}")
    logger.info(f"[INFO] Validation samples: {len(X_val):,}")
    logger.info(f"[INFO] Test samples: {len(X_test):,}")
    
    # Convert numpy arrays to lists if needed
    if isinstance(X_train, np.ndarray):
        X_train = X_train.tolist()
    if isinstance(X_val, np.ndarray):
        X_val = X_val.tolist()
    if isinstance(X_test, np.ndarray):
        X_test = X_test.tolist()
    
    # Train models
    try:
        trainer = train_heavy_gpu_bert(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            model_names=models,
            use_ensemble=use_ensemble
        )
        
        if trainer:
            logger.info("\n[SUCCESS] Phase 5B Heavy GPU BERT training complete!")
            
            # Get best model
            try:
                best_name, best_model = trainer.get_best_model()
                logger.info(f"[BEST MODEL] {best_name}")
                logger.info(f"[TEST ACCURACY] {trainer.best_accuracy:.4f}")
            except:
                logger.warning("Could not retrieve best model info")
        
        return trainer
        
    except Exception as e:
        logger.error(f"[ERROR] Phase 5B training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== MAIN ====================

def main(
    skip_phase1: bool = False,
    skip_phase4: bool = False,
    run_phase5_traditional: bool = False,
    run_phase5_bert: bool = False,
    use_bert_in_traditional: bool = False,
    phase5_bert_models: List[str] = None,
    use_ensemble: bool = False,
    incremental: bool = False,
    load_all_datasets: bool = True
):
    """
    Main entry point for training pipeline.
    
    Args:
        skip_phase1: Skip Phase 1-3 (traditional ML)
        skip_phase4: Skip Phase 4 (severity testing)
        run_phase5_traditional: Run Phase 5A (LSTM, BiLSTM, CNN, Basic BERT)
        run_phase5_bert: Run Phase 5B (Heavy GPU BERT)
        use_bert_in_traditional: Include Basic BERT in Phase 5A
        phase5_bert_models: List of BERT models for Phase 5B (e.g., ['bert-large', 'roberta-large'])
        use_ensemble: Use ensemble prediction in Phase 5B
        incremental: Use incremental training mode
        load_all_datasets: Load all datasets from data/raw (default: True)
    """
    
    print("=" * 80)
    print("HATE SPEECH DETECTION - ENHANCED TRAINING PIPELINE")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Mode: {'INCREMENTAL' if incremental else 'FULL TRAINING'}")
    print(f"Load all datasets: {load_all_datasets}")
    print(f"Traditional DL available: {HAS_TRADITIONAL_DL}")
    print(f"Heavy GPU BERT available: {HAS_HEAVY_BERT}")
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
    
    # Load data (needed for multiple phases)
    X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
    
    # Phase 1-3: Traditional ML
    trainer_trad = None
    if not skip_phase1:
        trainer_trad = phase1_train_traditional_ml(incremental=incremental)
        
        # Get data splits for Phase 5
        if (run_phase5_traditional or run_phase5_bert) and trainer_trad is not None:
            try:
                from data_handler import DataSplitter
                splitter = DataSplitter()
                X_train, X_val, X_test, y_train, y_val, y_test = splitter.load_split()
                logger.info("[OK] Data splits loaded for Phase 5")
            except:
                logger.warning("Could not load data splits from Phase 1")
    else:
        logger.info("[SKIP] Skipping Phase 1-3 (Traditional ML)")
        
        # If Phase 5 will run, still need to load data
        if run_phase5_traditional or run_phase5_bert:
            if incremental:
                trainer_inc = IncrementalTrainer()
                result = trainer_inc.incremental_train(new_datasets_only=True)
                if result is not None:
                    X_train, X_val, X_test, y_train, y_val, y_test = result
            else:
                logger.info("Loading ALL datasets from data/raw for Phase 5...")
                combined_df = load_all_csv_datasets()
                X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_from_combined_df(combined_df)
    
    # Phase 4: Severity System
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
    
    # Phase 5A: Traditional Deep Learning
    trainer_trad_dl = None
    if run_phase5_traditional:
        if X_train is None:
            logger.error("[ERROR] Cannot run Phase 5A without data!")
            logger.error("Run without --skip-phase1 or ensure data can be loaded")
        else:
            trainer_trad_dl = phase5a_train_traditional_dl(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                use_bert=use_bert_in_traditional
            )
    else:
        logger.info("\n[SKIP] Skipping Phase 5A (Traditional Deep Learning)")
        logger.info("Use --phase5 flag to enable LSTM, BiLSTM, CNN training")
    
    # Phase 5B: Heavy GPU BERT
    trainer_bert = None
    if run_phase5_bert:
        if X_train is None:
            logger.error("[ERROR] Cannot run Phase 5B without data!")
            logger.error("Run without --skip-phase1 or ensure data can be loaded")
        else:
            trainer_bert = phase5b_train_heavy_gpu_bert(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                models=phase5_bert_models,
                use_ensemble=use_ensemble
            )
    else:
        logger.info("\n[SKIP] Skipping Phase 5B (Heavy GPU BERT)")
        logger.info("Use --use-bert flag to enable Heavy GPU BERT training")
        logger.info("Available models: bert-large, roberta-large, distilbert, etc.")
    
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
            try:
                comparison_df = trainer_trad.evaluator.get_comparison_df()
                if not comparison_df.empty:
                    best = comparison_df.iloc[0]
                    print(f"     Best: {best['Model']} ({best['Accuracy']:.4f} accuracy)")
            except:
                pass
    
    if not skip_phase4:
        print("  [OK] Phase 4: Severity Classification + Action Recommendations")
    
    if run_phase5_traditional and trainer_trad_dl:
        print("  [OK] Phase 5A: Traditional Deep Learning (LSTM, BiLSTM, CNN)")
        try:
            best_name = trainer_trad_dl.get_best_model_name()
            best_metrics = trainer_trad_dl.get_best_metrics()
            print(f"     Best: {best_name} ({best_metrics.get('test_accuracy', 0):.4f} accuracy)")
        except:
            pass
    
    if run_phase5_bert and trainer_bert:
        print("  [OK] Phase 5B: Heavy GPU BERT Models")
        if hasattr(trainer_bert, 'best_model_name') and trainer_bert.best_model_name:
            print(f"     Best: {trainer_bert.best_model_name} ({trainer_bert.best_accuracy:.4f} accuracy)")
    
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
    
    if HAS_HEAVY_BERT and run_phase5_bert:
        print("\n  3. Use Heavy GPU BERT model:")
        print("     from bert_model_heavy_gpu import HeavyGPUBERTModel")
        print("     model = HeavyGPUBERTModel.load('saved_models/bert_bert_large')")
        print("     predictions = model.predict(texts)")
    
    if incremental:
        print("\n  4. Add more datasets:")
        print("     - Place new CSV files in data/raw/")
        print("     - Run: python main_train_enhanced.py --incremental")
    
    print("=" * 80)


# ==================== USAGE EXAMPLES ====================

def print_usage():
    """Print usage examples."""
    print("""
USAGE EXAMPLES:
===============

BASIC USAGE (Phase 1-4 only):
1. Train traditional ML + severity system:
   python main_train_enhanced.py

PHASE 5A - TRADITIONAL DEEP LEARNING (LSTM, BiLSTM, CNN):
2. Train traditional ML + traditional DL:
   python main_train_enhanced.py --phase5

3. Train traditional ML + traditional DL + Basic BERT:
   python main_train_enhanced.py --phase5 --use-bert

4. Skip traditional ML, only train deep learning:
   python main_train_enhanced.py --skip-phase1 --phase5

PHASE 5B - HEAVY GPU BERT:
5. Train with BERT-Large:
   python main_train_enhanced.py --use-bert bert-large

6. Train multiple heavy models and compare:
   python main_train_enhanced.py --use-bert bert-large roberta-base

7. Use ensemble for best accuracy:
   python main_train_enhanced.py --use-bert bert-large roberta-base --ensemble

COMBINED PHASES:
8. Train everything (Traditional ML + Traditional DL + Heavy BERT):
   python main_train_enhanced.py --phase5 --use-bert bert-large

9. Train all DL models (Traditional DL + Heavy BERT):
   python main_train_enhanced.py --skip-phase1 --phase5 --use-bert bert-large

10. Complete pipeline with ensemble:
    python main_train_enhanced.py --phase5 --use-bert bert-large roberta-base --ensemble

SKIP OPTIONS:
11. Skip traditional ML:
    python main_train_enhanced.py --skip-phase1 --phase5

12. Skip severity system:
    python main_train_enhanced.py --skip-phase4

13. Skip both:
    python main_train_enhanced.py --skip-phase1 --skip-phase4 --phase5

INCREMENTAL TRAINING:
14. Add new dataset:
    python main_train_enhanced.py --incremental

15. Incremental with deep learning:
    python main_train_enhanced.py --incremental --phase5 --use-bert bert-large

AVAILABLE HEAVY BERT MODELS:
- bert-base          (110M params) - Fast baseline
- bert-large         (340M params) - Better accuracy [RECOMMENDED]
- roberta-base       (125M params) - Improved BERT
- roberta-large      (355M params) - Best performance [RECOMMENDED]
- distilbert         (66M params)  - Fast inference
- albert-base        (12M params)  - Efficient
- albert-large       (18M params)  - Efficient
- albert-xlarge      (60M params)  - Very efficient
- albert-xxlarge     (235M params) - Extremely efficient

DATASET FORMAT:
Each CSV must have these columns:
- tweet: Text content
- class: Label (0=Hate, 1=Offensive, 2=Neither)

GPU REQUIREMENTS:
- Traditional DL (LSTM/CNN): 4-8GB GPU
- bert-base:      8GB+ GPU
- bert-large:     16GB+ GPU [RECOMMENDED for RTX 3060]
- roberta-large:  24GB+ GPU
""")


# ==================== RUN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Hate Speech Detection Training - Complete Pipeline'
    )
    
    # Training modes
    parser.add_argument('--incremental', action='store_true',
                       help='Incremental training: only train on NEW datasets')
    parser.add_argument('--retrain', action='store_true',
                       help='Force full retraining on all datasets')
    
    # Skip options
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1-3 (Traditional ML)')
    parser.add_argument('--skip-phase4', action='store_true',
                       help='Skip Phase 4 (Severity System)')
    
    # Phase 5A options - Traditional Deep Learning
    parser.add_argument('--phase5', action='store_true',
                       help='Run Phase 5A (LSTM, BiLSTM, CNN, optionally Basic BERT)')
    
    # Phase 5B options - Heavy GPU BERT
    parser.add_argument('--use-bert', nargs='*',
                       help='Run Phase 5B with Heavy GPU BERT models (e.g., bert-large roberta-large)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble prediction in Phase 5B (requires multiple models)')
    
    # Utility options
    parser.add_argument('--usage', action='store_true',
                       help='Show usage examples')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List all datasets in data/raw/')
    parser.add_argument('--list-models', action='store_true',
                       help='List available BERT models')
    
    args = parser.parse_args()
    
    if args.usage:
        print_usage()
    elif args.list_models:
        print("\n[AVAILABLE BERT MODELS FOR HEAVY GPU]")
        print("=" * 60)
        if HAS_HEAVY_BERT:
            from bert_model_heavy_gpu import HeavyGPUBERTConfig
            for key, value in HeavyGPUBERTConfig.MODEL_OPTIONS.items():
                print(f"  {key:20s} -> {value}")
            print("\nUse with --use-bert flag:")
            print("  python main_train_enhanced.py --use-bert bert-large")
        else:
            print("  Heavy GPU BERT not available")
            print("  Install: pip install torch transformers")
    elif args.list_datasets:
        print("\n[DATASETS IN data/raw/]")
        print("=" * 60)
        csv_files = list(RAW_DATA_DIR.glob('*.csv'))
        if not csv_files:
            print("  (none found)")
        else:
            for csv_file in csv_files:
                size = csv_file.stat().st_size / 1024
                print(f"  [FILE] {csv_file.name:30s} ({size:.1f} KB)")
        
        trainer = IncrementalTrainer()
        trained = trainer.get_trained_datasets()
        if trained:
            print(f"\n[ALREADY TRAINED ON] ({len(trained)})")
            for ds in trained:
                print(f"  [OK] {ds}")
    else:
        # Determine what to run based on flags
        run_phase5_traditional = args.phase5
        run_phase5_bert = args.use_bert is not None
        use_bert_in_traditional = '--use-bert' in sys.argv and args.phase5
        phase5_bert_models = args.use_bert if args.use_bert else None
        
        main(
            skip_phase1=args.skip_phase1,
            skip_phase4=args.skip_phase4,
            run_phase5_traditional=run_phase5_traditional,
            run_phase5_bert=run_phase5_bert,
            use_bert_in_traditional=use_bert_in_traditional,
            phase5_bert_models=phase5_bert_models,
            use_ensemble=args.ensemble,
            incremental=args.incremental and not args.retrain
        )