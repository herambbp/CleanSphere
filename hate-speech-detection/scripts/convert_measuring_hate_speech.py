"""
Convert Measuring Hate Speech Dataset to 3-class format
Maps continuous hate_speech_score to: Hate Speech (0), Offensive (1), Neither (2)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, CLASS_LABELS
from utils import logger, print_section_header


class MeasuringHateSpeechConverter:
    """
    Convert Measuring Hate Speech dataset to our 3-class format
    """
    
    def __init__(self):
        self.output_dir = RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, input_path: str = None):
        """
        Load the Measuring Hate Speech dataset
        Args:
            input_path: Path to the CSV file or will download from HuggingFace
        """
        print_section_header("LOADING MEASURING HATE SPEECH DATASET")
        
        # Fix: Check if input_path is provided and file exists
        if input_path:
            # Handle both absolute and relative paths
            file_path = Path(input_path)
            
            # If path doesn't exist, try relative to project root
            if not file_path.exists():
                file_path = PROJECT_ROOT / input_path.lstrip('/')
            
            # If still doesn't exist, try without leading slash
            if not file_path.exists():
                file_path = Path(input_path.lstrip('/'))
            
            if file_path.exists():
                logger.info(f"Loading from file: {file_path}")
                try:
                    # Try different encodings
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                            logger.info(f"[OK] Dataset loaded successfully with {encoding} encoding")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # If all encodings fail, try with error handling
                        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore', low_memory=False)
                        logger.info(f"[OK] Dataset loaded with error handling")
                    
                    logger.info(f"Loaded {len(df):,} rows")
                    logger.info(f"Columns: {list(df.columns)}")
                    return df
                    
                except Exception as e:
                    logger.error(f"Error loading file: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    sys.exit(1)
            else:
                logger.error(f"File not found: {input_path}")
                logger.error(f"Tried paths:")
                logger.error(f"  - {Path(input_path)}")
                logger.error(f"  - {PROJECT_ROOT / input_path.lstrip('/')}")
                sys.exit(1)
    
        # Only try downloading if no input path provided
        logger.info("No input file provided. Attempting to download from HuggingFace...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
            df = dataset.to_pandas()
            logger.info("[OK] Dataset downloaded successfully")
            logger.info(f"Loaded {len(df):,} rows")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except ImportError:
            logger.error("Install datasets library: pip install datasets")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.error("\nPlease download manually:")
            logger.error("1. Go to: https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech")
            logger.error("2. Download the CSV file")
            logger.error("3. Run: python scripts/convert_measuring_hate_speech.py --input path/to/file.csv --strategy severity_aware --aggregate")
            sys.exit(1)

    def convert_to_3_class(self, df: pd.DataFrame, strategy: str = "balanced") -> pd.DataFrame:
        """
        Convert continuous hate_speech_score to 3 classes
        
        Strategies:
        - 'strict': High precision for hate speech detection
        - 'balanced': Balanced distribution across classes
        - 'severity_aware': Uses additional severity indicators
        - 'multi_label': Uses multiple constituent labels
        
        Score ranges:
        - > 0.5: Hate speech (original guideline)
        - -1 to 0.5: Neutral/Ambiguous
        - < -1: Counter/Supportive speech
        """
        print_section_header(f"CONVERTING TO 3-CLASS FORMAT (Strategy: {strategy})")
        
        # Remove rows without hate_speech_score
        df = df.dropna(subset=['hate_speech_score', 'text'])
        logger.info(f"After removing NaN: {len(df):,} rows")
        
        if strategy == "strict":
            df = self._strict_conversion(df)
        elif strategy == "balanced":
            df = self._balanced_conversion(df)
        elif strategy == "severity_aware":
            df = self._severity_aware_conversion(df)
        elif strategy == "multi_label":
            df = self._multi_label_conversion(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df
    
    def _strict_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strict conversion - High precision for hate speech
        
        Class 0 (Hate Speech): score > 0.5
        Class 1 (Offensive): 0 < score <= 0.5
        Class 2 (Neither): score <= 0
        """
        logger.info("Using STRICT conversion strategy")
        
        def classify(score):
            if score > 0.5:
                return 0  # Hate Speech
            elif score > 0:
                return 1  # Offensive
            else:
                return 2  # Neither
        
        df['class'] = df['hate_speech_score'].apply(classify)
        return df
    
    def _balanced_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balanced conversion - Aims for ~15-20% hate, ~50-60% offensive, ~25-30% neither
        
        Uses percentiles to create balanced classes
        """
        logger.info("Using BALANCED conversion strategy")
        
        # Calculate percentiles for balanced distribution
        p85 = df['hate_speech_score'].quantile(0.85)  # Top 15% as hate
        p30 = df['hate_speech_score'].quantile(0.30)  # Bottom 30% as neither
        
        logger.info(f"Score thresholds:")
        logger.info(f"  Hate speech threshold (85th percentile): {p85:.3f}")
        logger.info(f"  Neither threshold (30th percentile): {p30:.3f}")
        
        def classify(score):
            if score >= p85:
                return 0  # Hate Speech (top 15%)
            elif score >= p30:
                return 1  # Offensive (middle 55%)
            else:
                return 2  # Neither (bottom 30%)
        
        df['class'] = df['hate_speech_score'].apply(classify)
        return df
    
    def _severity_aware_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Severity-aware conversion using multiple constituent labels
        
        Uses violence, dehumanize, genocide for hate speech classification
        Uses insult, disrespect for offensive classification
        """
        logger.info("Using SEVERITY-AWARE conversion strategy")
        
        # Calculate severity indicators
        severity_cols = ['violence', 'dehumanize', 'genocide']
        offensive_cols = ['insult', 'humiliate', 'respect']
        
        # Check which columns exist
        available_severity = [col for col in severity_cols if col in df.columns]
        available_offensive = [col for col in offensive_cols if col in df.columns]
        
        logger.info(f"Using severity columns: {available_severity}")
        logger.info(f"Using offensive columns: {available_offensive}")
        
        if available_severity:
            df['severity_score'] = df[available_severity].mean(axis=1)
        else:
            df['severity_score'] = 0
        
        if available_offensive:
            df['offensive_score'] = df[available_offensive].mean(axis=1)
        else:
            df['offensive_score'] = 0
        
        def classify(row):
            hate_score = row['hate_speech_score']
            severity = row.get('severity_score', 0)
            offensive = row.get('offensive_score', 0)
            
            # High hate score + high severity = Hate Speech
            if hate_score > 0.3 and severity > 0.5:
                return 0  # Hate Speech
            
            # High hate score alone
            elif hate_score > 0.7:
                return 0  # Hate Speech
            
            # Moderate hate score or high offensive score = Offensive
            elif hate_score > -0.3 or offensive > 0.5:
                return 1  # Offensive
            
            # Everything else = Neither
            else:
                return 2  # Neither
        
        df['class'] = df.apply(classify, axis=1)
        
        # Clean up temporary columns
        df = df.drop(['severity_score', 'offensive_score'], axis=1, errors='ignore')
        
        return df
    
    def _multi_label_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-label conversion using all 10 constituent labels
        
        Weights different aspects:
        - Heavy: violence, genocide, dehumanize -> Hate
        - Medium: insult, humiliate, status -> Offensive
        - Light: disrespect, sentiment -> Neither
        """
        logger.info("Using MULTI-LABEL conversion strategy")
        
        # Define label weights for hate speech
        hate_indicators = {
            'violence': 0.3,
            'genocide': 0.3,
            'dehumanize': 0.25,
            'hatespeech': 0.15  # The benchmark label
        }
        
        offensive_indicators = {
            'insult': 0.3,
            'humiliate': 0.3,
            'status': 0.2,
            'respect': 0.2
        }
        
        # Calculate weighted scores
        hate_cols = [col for col in hate_indicators.keys() if col in df.columns]
        offensive_cols = [col for col in offensive_indicators.keys() if col in df.columns]
        
        logger.info(f"Hate indicators: {hate_cols}")
        logger.info(f"Offensive indicators: {offensive_cols}")
        
        if hate_cols:
            df['hate_indicator'] = sum(
                df[col] * hate_indicators[col] 
                for col in hate_cols
            ) / sum(hate_indicators[col] for col in hate_cols)
        else:
            df['hate_indicator'] = df['hate_speech_score']
        
        if offensive_cols:
            df['offensive_indicator'] = sum(
                df[col] * offensive_indicators[col] 
                for col in offensive_cols
            ) / sum(offensive_indicators[col] for col in offensive_cols)
        else:
            df['offensive_indicator'] = 0
        
        def classify(row):
            hate = row['hate_indicator']
            offensive = row['offensive_indicator']
            score = row['hate_speech_score']
            
            # Strong hate indicators
            if hate > 0.6 or score > 0.8:
                return 0  # Hate Speech
            
            # Moderate hate or strong offensive
            elif hate > 0.3 or offensive > 0.5 or score > 0.2:
                return 1  # Offensive
            
            # Neither
            else:
                return 2  # Neither
        
        df['class'] = df.apply(classify, axis=1)
        
        # Clean up
        df = df.drop(['hate_indicator', 'offensive_indicator'], axis=1, errors='ignore')
        
        return df
    
    def aggregate_by_comment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple annotations per comment
        
        The dataset has multiple annotators per comment.
        We'll use the mean hate_speech_score and majority vote for class.
        """
        print_section_header("AGGREGATING ANNOTATIONS BY COMMENT")
        
        logger.info(f"Before aggregation: {len(df):,} rows")
        logger.info(f"Unique comments: {df['comment_id'].nunique():,}")
        
        # Group by comment_id
        aggregated = df.groupby('comment_id').agg({
            'text': 'first',  # Take first text (they should all be the same)
            'hate_speech_score': 'mean',  # Average score across annotators
            'class': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Majority vote
            'annotator_id': 'count'  # Count annotators per comment
        }).reset_index()
        
        aggregated.rename(columns={'annotator_id': 'num_annotators'}, inplace=True)
        
        logger.info(f"After aggregation: {len(aggregated):,} rows")
        logger.info(f"Average annotators per comment: {aggregated['num_annotators'].mean():.1f}")
        
        return aggregated
    
    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata columns for tracking and analysis
        """
        df['source_dataset'] = 'measuring_hate_speech'
        df['original_score'] = df['hate_speech_score']
        
        # Add confidence based on score distance from thresholds
        def calculate_confidence(row):
            score = row['hate_speech_score']
            cls = row['class']
            
            if cls == 0:  # Hate
                return min(abs(score - 0.5) * 2, 1.0)  # Distance from 0.5 threshold
            elif cls == 1:  # Offensive
                return min(abs(score) * 2, 1.0)  # Distance from 0
            else:  # Neither
                return min(abs(score + 1) * 2, 1.0)  # Distance from -1
        
        df['confidence'] = df.apply(calculate_confidence, axis=1)
        
        return df
    
    def print_statistics(self, df: pd.DataFrame, name: str = "Dataset"):
        """Print detailed statistics"""
        print_section_header(f"{name} STATISTICS")
        
        logger.info(f"Total samples: {len(df):,}")
        
        # Class distribution
        logger.info("\nClass Distribution:")
        for cls in sorted(df['class'].unique()):
            count = len(df[df['class'] == cls])
            pct = (count / len(df)) * 100
            logger.info(f"  Class {cls} ({CLASS_LABELS[cls]}): {count:,} ({pct:.2f}%)")
        
        # Score statistics
        logger.info("\nHate Speech Score Statistics:")
        logger.info(f"  Mean: {df['hate_speech_score'].mean():.3f}")
        logger.info(f"  Median: {df['hate_speech_score'].median():.3f}")
        logger.info(f"  Std: {df['hate_speech_score'].std():.3f}")
        logger.info(f"  Min: {df['hate_speech_score'].min():.3f}")
        logger.info(f"  Max: {df['hate_speech_score'].max():.3f}")
        
        # Score distribution by class
        logger.info("\nScore Distribution by Class:")
        for cls in sorted(df['class'].unique()):
            scores = df[df['class'] == cls]['hate_speech_score']
            logger.info(f"  Class {cls}: mean={scores.mean():.3f}, "
                       f"median={scores.median():.3f}, "
                       f"std={scores.std():.3f}")
        
        if 'confidence' in df.columns:
            logger.info("\nConfidence Statistics:")
            logger.info(f"  Mean confidence: {df['confidence'].mean():.3f}")
            for cls in sorted(df['class'].unique()):
                conf = df[df['class'] == cls]['confidence'].mean()
                logger.info(f"  Class {cls} avg confidence: {conf:.3f}")
    
    def save_converted_dataset(self, df: pd.DataFrame, strategy: str):
        """Save converted dataset"""
        print_section_header("SAVING CONVERTED DATASET")
        
        # Prepare final dataframe with required columns
        output_df = df[['text', 'class']].copy()
        
        # Add optional metadata if you want to keep it
        metadata_cols = ['comment_id', 'hate_speech_score', 'confidence', 
                        'source_dataset', 'num_annotators']
        for col in metadata_cols:
            if col in df.columns:
                output_df[col] = df[col]
        
        # Remove duplicates
        output_df = output_df.drop_duplicates(subset=['text'])
        
        # Save
        output_file = self.output_dir / f'measuring_hate_speech_{strategy}.csv'
        output_df.to_csv(output_file, index=False)
        
        logger.info(f"[SUCCESS] Saved {len(output_df):,} samples to: {output_file}")
        logger.info(f"[INFO] File size: {output_file.stat().st_size / 1024:.1f} KB")
        
        return output_file


def main():
    """Main conversion pipeline"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert Measuring Hate Speech dataset to 3-class format'
    )
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input CSV (or will download from HuggingFace)')
    parser.add_argument('--strategy', type=str, default='severity_aware',
                       choices=['strict', 'balanced', 'severity_aware', 'multi_label'],
                       help='Conversion strategy')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate multiple annotations per comment')
    parser.add_argument('--all-strategies', action='store_true',
                       help='Try all conversion strategies')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = MeasuringHateSpeechConverter()
    
    # Load dataset
    df = converter.load_dataset(args.input)
    
    # Show original statistics
    logger.info("\n[ORIGINAL DATASET]")
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Unique comments: {df['comment_id'].nunique():,}")
    logger.info(f"Unique annotators: {df['annotator_id'].nunique():,}")
    
    strategies = ['strict', 'balanced', 'severity_aware', 'multi_label'] if args.all_strategies else [args.strategy]
    
    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING WITH STRATEGY: {strategy.upper()}")
        logger.info(f"{'='*80}")
        
        # Convert to 3-class
        df_converted = converter.convert_to_3_class(df.copy(), strategy=strategy)
        
        # Aggregate if requested
        if args.aggregate:
            df_converted = converter.aggregate_by_comment(df_converted)
        
        # Add metadata
        df_converted = converter.add_metadata(df_converted)
        
        # Print statistics
        converter.print_statistics(df_converted, name=f"{strategy.upper()} Strategy")
        
        # Save
        output_file = converter.save_converted_dataset(df_converted, strategy)
        
        logger.info(f"\n[COMPLETE] {strategy} conversion finished!")
        logger.info(f"[OUTPUT] {output_file}")


if __name__ == "__main__":
    main()