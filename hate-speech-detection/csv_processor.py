"""
CSV Processor for Hate Speech Detection
Batch process user comments and generate analytics

Features:
- Process CSV with user comments
- Analyze each user's behavior
- Generate user risk scores
- Create analytics dashboard
- Export results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from inference.explainable_classifier import ExplainableTweetClassifier
from config import CLASS_LABELS, SEVERITY_LEVELS
from utils import logger

# ==================== CSV PROCESSOR ====================

class CSVProcessor:
    """
    Process CSV files with user comments and generate analytics.
    
    Expected CSV format:
    - name/user/username: User identifier
    - timestamp/date/time: Comment timestamp
    - comment/text/message: Comment text
    """
    
    def __init__(self):
        """Initialize CSV processor with classifier."""
        logger.info("Initializing CSV Processor...")
        self.classifier = ExplainableTweetClassifier()
        self.results = []
        self.user_analytics = {}
        logger.info("CSV Processor ready")
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate CSV file.
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading CSV from: {filepath}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                logger.info(f"Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading CSV with {encoding}: {e}")
        
        if df is None:
            raise ValueError("Could not load CSV file with any encoding")
        
        # Validate columns
        df = self._standardize_columns(df)
        
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to expected format.
        
        Looks for:
        - User column: name, user, username, author
        - Timestamp column: timestamp, date, time, created_at
        - Text column: comment, text, message, content
        """
        # Create lowercase mapping
        col_lower = {col: col.lower() for col in df.columns}
        
        # Find user column
        user_cols = ['name', 'user', 'username', 'author', 'user_name']
        user_col = None
        for col in df.columns:
            if col_lower[col] in user_cols:
                user_col = col
                break
        
        # Find timestamp column
        time_cols = ['timestamp', 'date', 'time', 'created_at', 'datetime']
        time_col = None
        for col in df.columns:
            if col_lower[col] in time_cols:
                time_col = col
                break
        
        # Find text column
        text_cols = ['comment', 'text', 'message', 'content', 'tweet']
        text_col = None
        for col in df.columns:
            if col_lower[col] in text_cols:
                text_col = col
                break
        
        # Validate required columns found
        if not user_col:
            raise ValueError(f"User column not found. Expected one of: {user_cols}")
        if not time_col:
            raise ValueError(f"Timestamp column not found. Expected one of: {time_cols}")
        if not text_col:
            raise ValueError(f"Text column not found. Expected one of: {text_cols}")
        
        # Rename to standard names
        df = df.rename(columns={
            user_col: 'user',
            time_col: 'timestamp',
            text_col: 'text'
        })
        
        # Parse timestamp
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            logger.warning(f"Could not parse timestamps: {e}")
            # Keep as string if parsing fails
        
        # Remove empty texts
        df = df[df['text'].notna() & (df['text'].str.strip() != '')]
        
        logger.info("Columns standardized successfully")
        return df
    
    def process_csv(self, filepath: str, progress_callback=None) -> pd.DataFrame:
        """
        Process entire CSV file.
        
        Args:
            filepath: Path to CSV file
            progress_callback: Optional callback function(current, total)
        
        Returns:
            DataFrame with classification results
        """
        # Load CSV
        df = self.load_csv(filepath)
        
        logger.info(f"Processing {len(df)} comments from {df['user'].nunique()} users...")
        
        results = []
        
        for idx, row in df.iterrows():
            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(df))
            
            try:
                # Classify comment
                result = self.classifier.classify_with_explanation(
                    text=row['text'],
                    include_severity=True,
                    verbose=False
                )
                
                # Extract key information
                classification = {
                    'user': row['user'],
                    'timestamp': row['timestamp'],
                    'text': row['text'],
                    'prediction': result['prediction'],
                    'class': result['class'],
                    'confidence': result['confidence'],
                    'severity_level': result.get('severity', {}).get('severity_level', 1),
                    'severity_label': result.get('severity', {}).get('severity_label', 'LOW'),
                    'severity_score': result.get('severity', {}).get('severity_score', 0),
                    'action': result.get('action', {}).get('primary_action', 'NO_ACTION'),
                    'urgency': result.get('action', {}).get('urgency', 'NONE')
                }
                
                results.append(classification)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Add error entry
                results.append({
                    'user': row['user'],
                    'timestamp': row['timestamp'],
                    'text': row['text'],
                    'prediction': 'ERROR',
                    'class': -1,
                    'confidence': 0.0,
                    'severity_level': 0,
                    'severity_label': 'ERROR',
                    'severity_score': 0,
                    'action': 'ERROR',
                    'urgency': 'NONE'
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.results = results
        
        logger.info("CSV processing complete")
        
        # Generate user analytics
        self._generate_user_analytics(results_df)
        
        return results_df
    
    def _generate_user_analytics(self, df: pd.DataFrame):
        """
        Generate per-user analytics.
        
        Args:
            df: Results DataFrame
        """
        logger.info("Generating user analytics...")
        
        analytics = {}
        
        for user in df['user'].unique():
            user_df = df[df['user'] == user]
            
            # Calculate statistics
            total_comments = len(user_df)
            
            # Class distribution
            hate_count = len(user_df[user_df['class'] == 0])
            offensive_count = len(user_df[user_df['class'] == 1])
            neither_count = len(user_df[user_df['class'] == 2])
            
            # Percentages
            hate_pct = (hate_count / total_comments * 100) if total_comments > 0 else 0
            offensive_pct = (offensive_count / total_comments * 100) if total_comments > 0 else 0
            neither_pct = (neither_count / total_comments * 100) if total_comments > 0 else 0
            
            # Severity statistics
            avg_severity = user_df['severity_score'].mean()
            max_severity = user_df['severity_score'].max()
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(
                hate_pct, offensive_pct, avg_severity, max_severity
            )
            
            # Risk level
            risk_level = self._get_risk_level(risk_score)
            
            # Most severe comment
            most_severe = user_df.loc[user_df['severity_score'].idxmax()] if len(user_df) > 0 else None
            
            # Timeline
            if 'timestamp' in user_df.columns:
                first_comment = user_df['timestamp'].min()
                last_comment = user_df['timestamp'].max()
            else:
                first_comment = None
                last_comment = None
            
            # Store analytics
            analytics[user] = {
                'total_comments': total_comments,
                'hate_speech': hate_count,
                'offensive': offensive_count,
                'neither': neither_count,
                'hate_percentage': round(hate_pct, 2),
                'offensive_percentage': round(offensive_pct, 2),
                'neither_percentage': round(neither_pct, 2),
                'avg_severity_score': round(avg_severity, 2),
                'max_severity_score': round(max_severity, 2),
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level,
                'most_severe_comment': {
                    'text': most_severe['text'] if most_severe is not None else '',
                    'severity': most_severe['severity_score'] if most_severe is not None else 0,
                    'timestamp': most_severe['timestamp'] if most_severe is not None else None
                },
                'first_comment': first_comment,
                'last_comment': last_comment
            }
        
        self.user_analytics = analytics
        logger.info(f"Analytics generated for {len(analytics)} users")
    
    def _calculate_risk_score(
        self,
        hate_pct: float,
        offensive_pct: float,
        avg_severity: float,
        max_severity: float
    ) -> float:
        """
        Calculate user risk score (0-100).
        
        Factors:
        - Hate speech percentage (40%)
        - Offensive percentage (20%)
        - Average severity (25%)
        - Max severity (15%)
        """
        # Normalize percentages to 0-1
        hate_normalized = min(hate_pct / 100, 1.0)
        offensive_normalized = min(offensive_pct / 100, 1.0)
        
        # Normalize severity scores (0-100 scale)
        avg_severity_normalized = min(avg_severity / 100, 1.0)
        max_severity_normalized = min(max_severity / 100, 1.0)
        
        # Weighted combination
        risk = (
            hate_normalized * 0.40 +
            offensive_normalized * 0.20 +
            avg_severity_normalized * 0.25 +
            max_severity_normalized * 0.15
        ) * 100
        
        return risk
    
    def _get_risk_level(self, risk_score: float) -> str:
        """
        Get risk level from score.
        
        Levels:
        - CRITICAL: 80-100
        - HIGH: 60-80
        - MEDIUM: 40-60
        - LOW: 20-40
        - MINIMAL: 0-20
        """
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
    
    def get_user_analytics(self, user: str = None) -> Dict:
        """
        Get analytics for specific user or all users.
        
        Args:
            user: Username (optional, returns all if None)
        
        Returns:
            User analytics dictionary
        """
        if user:
            return self.user_analytics.get(user, {})
        return self.user_analytics
    
    def export_results(self, output_path: str):
        """
        Export detailed results to CSV.
        
        Args:
            output_path: Path for output CSV
        """
        if not self.results:
            logger.warning("No results to export")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")
    
    def export_user_analytics(self, output_path: str):
        """
        Export user analytics to CSV.
        
        Args:
            output_path: Path for output CSV
        """
        if not self.user_analytics:
            logger.warning("No analytics to export")
            return
        
        # Convert to DataFrame
        rows = []
        for user, analytics in self.user_analytics.items():
            row = {
                'user': user,
                **{k: v for k, v in analytics.items() if k != 'most_severe_comment'}
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by risk score
        df = df.sort_values('risk_score', ascending=False)
        
        df.to_csv(output_path, index=False)
        logger.info(f"User analytics exported to {output_path}")
    
    def generate_summary_report(self) -> Dict:
        """
        Generate overall summary report.
        
        Returns:
            Summary statistics
        """
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        summary = {
            'total_comments': len(df),
            'total_users': df['user'].nunique(),
            'avg_comments_per_user': round(len(df) / df['user'].nunique(), 2),
            'overall_hate_percentage': round(len(df[df['class'] == 0]) / len(df) * 100, 2),
            'overall_offensive_percentage': round(len(df[df['class'] == 1]) / len(df) * 100, 2),
            'overall_neither_percentage': round(len(df[df['class'] == 2]) / len(df) * 100, 2),
            'avg_severity_score': round(df['severity_score'].mean(), 2),
            'users_by_risk': self._count_users_by_risk()
        }
        
        return summary
    
    def _count_users_by_risk(self) -> Dict:
        """Count users by risk level."""
        counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'MINIMAL': 0
        }
        
        for analytics in self.user_analytics.values():
            risk_level = analytics['risk_level']
            counts[risk_level] = counts.get(risk_level, 0) + 1
        
        return counts


# ==================== CONVENIENCE FUNCTIONS ====================

def process_user_csv(
    csv_path: str,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to process CSV and generate reports.
    
    Args:
        csv_path: Path to input CSV
        output_dir: Directory for output files (optional)
    
    Returns:
        Tuple of (results_df, user_analytics)
    """
    processor = CSVProcessor()
    
    # Process CSV
    results_df = processor.process_csv(csv_path)
    
    # Export results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export detailed results
        processor.export_results(str(output_path / 'detailed_results.csv'))
        
        # Export user analytics
        processor.export_user_analytics(str(output_path / 'user_analytics.csv'))
        
        # Save summary
        summary = processor.generate_summary_report()
        import json
        with open(output_path / 'summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"All reports exported to {output_dir}")
    
    return results_df, processor.user_analytics


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("CSV PROCESSOR TEST")
    print("=" * 80)
    
    # Create sample CSV
    sample_data = {
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice'],
        'timestamp': [
            '2024-01-01 10:00:00',
            '2024-01-01 11:00:00',
            '2024-01-01 12:00:00',
            '2024-01-01 13:00:00',
            '2024-01-01 14:00:00',
            '2024-01-01 15:00:00'
        ],
        'comment': [
            'I hate when my code doesn\'t work',
            'You\'re a fucking idiot',
            'Good morning everyone!',
            'I will kill you bitch',
            'This is offensive language',
            'Have a great day!'
        ]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_csv = 'test_comments.csv'
    sample_df.to_csv(sample_csv, index=False)
    
    print(f"\nCreated sample CSV: {sample_csv}")
    print(f"Processing {len(sample_df)} comments from {sample_df['name'].nunique()} users...")
    
    # Process
    processor = CSVProcessor()
    
    results_df = processor.process_csv(sample_csv, 
        progress_callback=lambda c, t: print(f"Progress: {c}/{t}")
    )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(results_df[['user', 'prediction', 'severity_label', 'risk_level']].to_string())
    
    print("\n" + "=" * 80)
    print("USER ANALYTICS")
    print("=" * 80)
    
    for user, analytics in processor.user_analytics.items():
        print(f"\n{user}:")
        print(f"  Total Comments: {analytics['total_comments']}")
        print(f"  Hate Speech: {analytics['hate_percentage']}%")
        print(f"  Risk Score: {analytics['risk_score']}")
        print(f"  Risk Level: {analytics['risk_level']}")
    
    # Export
    processor.export_results('test_results.csv')
    processor.export_user_analytics('test_analytics.csv')
    
    print("\n✓ Test complete!")



