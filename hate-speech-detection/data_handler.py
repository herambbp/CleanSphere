"""
Data loading, cleaning, and splitting for hate speech detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    LABELED_DATA_CSV, COMBINED_DATA_CSV, PROCESSED_DATA_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE, CLASS_LABELS
)
from utils import logger, print_section_header, print_class_distribution, validate_arrays

# ==================== DATA LOADER ====================

class DataLoader:
    """Load and validate hate speech datasets."""
    
    def __init__(self):
        self.df = None
        self.labeled_df = None
        self.unlabeled_df = None
    
    def load_data(self, prefer_combined: bool = True) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            prefer_combined: Try combined dataset first
        
        Returns:
            Loaded DataFrame
        """
        print_section_header("LOADING DATA")
        
        # Determine which file to load
        if prefer_combined and COMBINED_DATA_CSV.exists():
            file_path = COMBINED_DATA_CSV
            logger.info(f"Loading combined dataset: {file_path.name}")
        elif LABELED_DATA_CSV.exists():
            file_path = LABELED_DATA_CSV
            logger.info(f"Loading labeled dataset: {file_path.name}")
        else:
            raise FileNotFoundError(
                f"No dataset found. Please place data files in {LABELED_DATA_CSV.parent}"
            )
        
        # Load CSV
        try:
            df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
            logger.info(f"Successfully loaded {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        self.df = df
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate dataset structure.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Validated DataFrame
        """
        logger.info("Validating dataset...")
        
        # Check required columns
        required_cols = ['tweet', 'class']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Log column info
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Unique classes: {sorted(df['class'].dropna().unique())}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning dataset...")
        initial_count = len(df)
        
        # Remove null values
        df = df.dropna(subset=['tweet', 'class'])
        logger.info(f"Removed {initial_count - len(df)} rows with null values")
        
        # Convert types
        df['tweet'] = df['tweet'].astype(str)
        df['class'] = df['class'].astype(int)
        
        # Remove duplicates
        duplicates = df.duplicated(subset=['tweet'], keep='first').sum()
        if duplicates > 0:
            df = df.drop_duplicates(subset=['tweet'], keep='first')
            logger.info(f"Removed {duplicates} duplicate tweets")
        
        # Remove empty tweets
        empty = (df['tweet'].str.strip() == '').sum()
        if empty > 0:
            df = df[df['tweet'].str.strip() != '']
            logger.info(f"Removed {empty} empty tweets")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"Cleaned dataset: {len(df)} rows remaining")
        return df
    
    def separate_labeled_unlabeled(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate labeled and unlabeled data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (labeled_df, unlabeled_df)
        """
        logger.info("Separating labeled and unlabeled data...")
        
        # Labeled: class in [0, 1, 2]
        labeled_df = df[df['class'].isin([0, 1, 2])].copy()
        
        # Unlabeled: class == -1
        unlabeled_df = df[df['class'] == -1].copy()
        
        # Remove invalid classes
        invalid = df[~df['class'].isin([-1, 0, 1, 2])]
        if len(invalid) > 0:
            logger.warning(f"Removed {len(invalid)} rows with invalid class values")
        
        logger.info(f"Labeled samples: {len(labeled_df)}")
        logger.info(f"Unlabeled samples: {len(unlabeled_df)}")
        
        # Store
        self.labeled_df = labeled_df
        self.unlabeled_df = unlabeled_df
        
        return labeled_df, unlabeled_df
    
    def load_and_prepare(self) -> Tuple[pd.DataFrame, dict]:
        """
        Complete pipeline: load, validate, clean, separate.
        
        Returns:
            Tuple of (labeled_df, metadata)
        """
        # Load
        df = self.load_data()
        
        # Validate
        df = self.validate_data(df)
        
        # Clean
        df = self.clean_data(df)
        
        # Separate
        labeled_df, unlabeled_df = self.separate_labeled_unlabeled(df)
        
        # Print statistics
        print_section_header("DATASET SUMMARY")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Labeled samples: {len(labeled_df)}")
        logger.info(f"Unlabeled samples: {len(unlabeled_df)}")
        
        print_class_distribution(labeled_df['class'].values)
        
        # Create metadata
        metadata = {
            'total_samples': len(df),
            'labeled_samples': len(labeled_df),
            'unlabeled_samples': len(unlabeled_df)
        }
        
        return labeled_df, metadata

# ==================== DATA SPLITTER ====================

class DataSplitter:
    """Split data into train/validation/test sets."""
    
    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_state: int = RANDOM_STATE
    ):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split into train/val/test with stratification.
        
        Args:
            X: Features or text
            y: Labels
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print_section_header("SPLITTING DATA")
        
        # Validate inputs
        validate_arrays(X, y)
        
        logger.info(f"Split ratios: train={self.train_ratio:.0%}, "
                   f"val={self.val_ratio:.0%}, test={self.test_ratio:.0%}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: separate train and val
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        # Log statistics
        self._log_split_stats(y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _log_split_stats(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Log split statistics."""
        total = len(y_train) + len(y_val) + len(y_test)
        
        logger.info(f"\nTotal samples: {total}")
        logger.info(f"Training:   {len(y_train):6d} ({len(y_train)/total*100:5.2f}%)")
        logger.info(f"Validation: {len(y_val):6d} ({len(y_val)/total*100:5.2f}%)")
        logger.info(f"Test:       {len(y_test):6d} ({len(y_test)/total*100:5.2f}%)")
        
        # Per-class distribution
        logger.info("\nClass Distribution Across Splits:")
        for class_id in sorted(np.unique(y_train)):
            class_name = CLASS_LABELS.get(class_id, f"Class {class_id}")
            
            train_count = np.sum(y_train == class_id)
            val_count = np.sum(y_val == class_id)
            test_count = np.sum(y_test == class_id)
            
            logger.info(f"\n{class_name}:")
            logger.info(f"  Train: {train_count:5d} ({train_count/len(y_train)*100:5.2f}%)")
            logger.info(f"  Val:   {val_count:5d} ({val_count/len(y_val)*100:5.2f}%)")
            logger.info(f"  Test:  {test_count:5d} ({test_count/len(y_test)*100:5.2f}%)")
    
    def save_split(
        self,
        X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
    ):
        """
        Save split datasets.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Label arrays
        """
        logger.info("Saving split datasets...")
        
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        joblib.dump({'X': X_train, 'y': y_train}, PROCESSED_DATA_DIR / 'train_data.pkl')
        joblib.dump({'X': X_val, 'y': y_val}, PROCESSED_DATA_DIR / 'val_data.pkl')
        joblib.dump({'X': X_test, 'y': y_test}, PROCESSED_DATA_DIR / 'test_data.pkl')
        
        logger.info(f"Saved train/val/test data to {PROCESSED_DATA_DIR}")
    
    @staticmethod
    def load_split() -> Tuple:
        """
        Load saved split datasets.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Loading split datasets...")
        
        train_data = joblib.load(PROCESSED_DATA_DIR / 'train_data.pkl')
        val_data = joblib.load(PROCESSED_DATA_DIR / 'val_data.pkl')
        test_data = joblib.load(PROCESSED_DATA_DIR / 'test_data.pkl')
        
        logger.info(f"Loaded: {len(train_data['y'])} train, "
                   f"{len(val_data['y'])} val, {len(test_data['y'])} test samples")
        
        return (
            train_data['X'], val_data['X'], test_data['X'],
            train_data['y'], val_data['y'], test_data['y']
        )

# ==================== CONVENIENCE FUNCTIONS ====================

def load_and_split_data(save_split: bool = True) -> Tuple:
    """
    Convenience function: load data and split into train/val/test.
    
    Args:
        save_split: Whether to save the split
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load data
    loader = DataLoader()
    labeled_df, metadata = loader.load_and_prepare()
    
    # Extract text and labels
    X = labeled_df['tweet'].values
    y = labeled_df['class'].values
    
    # Split
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_data(X, y)
    
    # Save if requested
    if save_split:
        splitter.save_split(X_train, X_val, X_test, y_train, y_val, y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==================== TESTING ====================

if __name__ == "__main__":
    # Test data loading and splitting
    print("Testing data_handler.py...")
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(save_split=True)
        
        print("\nSplit successful!")
        print(f"Train samples: {len(y_train)}")
        print(f"Val samples: {len(y_val)}")
        print(f"Test samples: {len(y_test)}")
        
        print("\nSample tweets from training set:")
        for i in range(min(3, len(X_train))):
            print(f"{i+1}. {X_train[i][:80]}... (class: {y_train[i]})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()