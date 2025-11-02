"""
Digital Footprint Scoring System for Hate Speech Detection
Provides comprehensive user risk profiling based on historical behavior patterns

Features:
- Temporal behavior analysis (escalation/improvement trends)
- Cross-platform aggregation
- Engagement pattern detection
- Context-aware scoring
- Behavior change detection
- Risk trajectory prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path

# Import from existing project
try:
    from config import CLASS_LABELS
    from utils import logger
except ImportError:
    CLASS_LABELS = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
    import logging
    logger = logging.getLogger(__name__)


class DigitalFootprintScorer:
    """
    Calculate comprehensive digital footprint scores for users based on:
    1. Historical content classification patterns
    2. Temporal behavior trends (improvement/deterioration)
    3. Engagement patterns and frequency
    4. Severity escalation detection
    5. Cross-platform aggregation (if applicable)
    """
    
    # Score weights (sum to 1.0)
    WEIGHTS = {
        'content_severity': 0.30,      # Severity of content posted
        'temporal_trend': 0.25,        # Is behavior improving or worsening?
        'frequency': 0.15,             # How often they post problematic content
        'recency': 0.15,               # Recent activity weighted higher
        'consistency': 0.10,           # Consistent problematic behavior vs. outliers
        'engagement_risk': 0.05        # Risk based on engagement patterns
    }
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'critical': 80,    # Immediate action required
        'high': 65,        # Close monitoring needed
        'elevated': 50,    # Watch list
        'moderate': 35,    # Standard monitoring
        'low': 20,         # Minimal concern
        'minimal': 0       # Clean record
    }
    
    def __init__(self):
        self.user_footprints = {}
        
    
    def calculate_footprint_score(
        self,
        user_id: str,
        comments_df: pd.DataFrame,
        account_age_days: Optional[int] = None,
        platform_data: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate comprehensive digital footprint score for a user.
        
        Args:
            user_id: User identifier
            comments_df: DataFrame with user's comment history (columns: timestamp, comment, prediction, confidence, severity)
            account_age_days: Age of account in days (optional)
            platform_data: Additional platform metadata (followers, interactions, etc.)
        
        Returns:
            Dictionary with footprint score and detailed breakdown
        """
        
        if len(comments_df) == 0:
            return self._create_empty_footprint(user_id)
        
        # Sort by timestamp
        comments_df = comments_df.sort_values('timestamp')
        
        # Calculate individual components
        content_score = self._calculate_content_severity_score(comments_df)
        temporal_score = self._calculate_temporal_trend_score(comments_df)
        frequency_score = self._calculate_frequency_score(comments_df, account_age_days)
        recency_score = self._calculate_recency_score(comments_df)
        consistency_score = self._calculate_consistency_score(comments_df)
        engagement_score = self._calculate_engagement_risk_score(comments_df, platform_data)
        
        # Calculate weighted total
        total_score = (
            content_score * self.WEIGHTS['content_severity'] +
            temporal_score * self.WEIGHTS['temporal_trend'] +
            frequency_score * self.WEIGHTS['frequency'] +
            recency_score * self.WEIGHTS['recency'] +
            consistency_score * self.WEIGHTS['consistency'] +
            engagement_score * self.WEIGHTS['engagement_risk']
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(total_score)
        
        # Detect behavior patterns
        patterns = self._detect_behavior_patterns(comments_df)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_score, risk_level, patterns, comments_df
        )
        
        # Create footprint profile
        footprint = {
            'user_id': user_id,
            'footprint_score': round(total_score, 2),
            'risk_level': risk_level,
            'score_breakdown': {
                'content_severity': round(content_score, 2),
                'temporal_trend': round(temporal_score, 2),
                'frequency': round(frequency_score, 2),
                'recency': round(recency_score, 2),
                'consistency': round(consistency_score, 2),
                'engagement_risk': round(engagement_score, 2)
            },
            'statistics': {
                'total_comments': len(comments_df),
                'hate_speech_count': len(comments_df[comments_df['prediction'] == 'Hate speech']),
                'offensive_count': len(comments_df[comments_df['prediction'] == 'Offensive language']),
                'neither_count': len(comments_df[comments_df['prediction'] == 'Neither']),
                'average_confidence': float(comments_df['confidence'].mean()),
                'time_span_days': (comments_df['timestamp'].max() - comments_df['timestamp'].min()).days
            },
            'patterns': patterns,
            'recommendations': recommendations,
            'account_info': {
                'account_age_days': account_age_days,
                'platform_data': platform_data or {}
            },
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache for future reference
        self.user_footprints[user_id] = footprint
        
        return footprint
    
    
    def _calculate_content_severity_score(self, df: pd.DataFrame) -> float:
        """
        Score based on severity of content (0-100).
        Higher score = more severe content.
        """
        # Map predictions to severity values
        severity_map = {
            'Hate speech': 100,
            'Offensive language': 60,
            'Neither': 0
        }
        
        # Calculate weighted average severity
        df['severity_value'] = df['prediction'].map(severity_map)
        
        # Weight by confidence
        weighted_severity = (df['severity_value'] * df['confidence']).sum() / df['confidence'].sum()
        
        return weighted_severity
    
    
    def _calculate_temporal_trend_score(self, df: pd.DataFrame) -> float:
        """
        Score based on temporal trends (0-100).
        - Improving behavior = lower score
        - Worsening behavior = higher score
        - Stable bad behavior = high score
        """
        if len(df) < 3:
            # Not enough data for trend analysis
            return self._calculate_content_severity_score(df)
        
        # Split into time windows
        n_windows = min(5, len(df))
        window_size = len(df) // n_windows
        
        window_scores = []
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size if i < n_windows - 1 else len(df)
            window_df = df.iloc[start_idx:end_idx]
            
            hate_ratio = len(window_df[window_df['prediction'] == 'Hate speech']) / len(window_df)
            window_scores.append(hate_ratio * 100)
        
        # Calculate trend (positive slope = worsening)
        if len(window_scores) >= 2:
            x = np.arange(len(window_scores))
            slope = np.polyfit(x, window_scores, 1)[0]
            
            # Recent score + trend adjustment
            recent_score = window_scores[-1]
            trend_adjustment = slope * 10  # Amplify trend impact
            
            score = recent_score + trend_adjustment
            return max(0, min(100, score))
        
        return window_scores[-1] if window_scores else 50
    
    
    def _calculate_frequency_score(self, df: pd.DataFrame, account_age_days: Optional[int]) -> float:
        """
        Score based on frequency of problematic content (0-100).
        More frequent problematic posts = higher score.
        """
        time_span_days = (df['timestamp'].max() - df['timestamp'].min()).days
        if time_span_days == 0:
            time_span_days = 1
        
        # Count problematic content
        problematic = df[df['prediction'].isin(['Hate speech', 'Offensive language'])]
        problematic_count = len(problematic)
        
        # Calculate posts per day
        posts_per_day = problematic_count / time_span_days
        
        # Score based on frequency (nonlinear scaling)
        # 0-0.5 posts/day = low score
        # 0.5-2 posts/day = medium score
        # 2+ posts/day = high score
        if posts_per_day < 0.5:
            score = posts_per_day * 40  # 0-20
        elif posts_per_day < 2:
            score = 20 + (posts_per_day - 0.5) * 40  # 20-80
        else:
            score = 80 + min(20, (posts_per_day - 2) * 10)  # 80-100
        
        return min(100, score)
    
    
    def _calculate_recency_score(self, df: pd.DataFrame) -> float:
        """
        Score based on recency of problematic content (0-100).
        Recent problematic posts weighted higher.
        """
        # Get problematic posts
        problematic = df[df['prediction'].isin(['Hate speech', 'Offensive language'])]
        
        if len(problematic) == 0:
            return 0
        
        # Calculate days since last problematic post
        most_recent = problematic['timestamp'].max()
        days_since = (datetime.now() - most_recent).days
        
        # Exponential decay: recent activity = high score
        # 0-7 days = 80-100
        # 7-30 days = 50-80
        # 30-90 days = 20-50
        # 90+ days = 0-20
        if days_since <= 7:
            score = 100 - (days_since * 2.86)  # 100 to 80
        elif days_since <= 30:
            score = 80 - ((days_since - 7) * 1.30)  # 80 to 50
        elif days_since <= 90:
            score = 50 - ((days_since - 30) * 0.5)  # 50 to 20
        else:
            score = max(0, 20 - ((days_since - 90) * 0.1))
        
        return max(0, score)
    
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """
        Score based on consistency of problematic behavior (0-100).
        Consistent bad behavior = high score
        Occasional outliers = low score
        """
        if len(df) < 5:
            # Not enough data
            hate_ratio = len(df[df['prediction'] == 'Hate speech']) / len(df)
            return hate_ratio * 100
        
        # Calculate rolling window variance in behavior
        window_size = min(10, len(df) // 2)
        
        df['is_problematic'] = df['prediction'].isin(['Hate speech', 'Offensive language']).astype(int)
        rolling_mean = df['is_problematic'].rolling(window=window_size).mean()
        
        # High consistency = low variance = high score if mean is high
        variance = rolling_mean.var()
        mean_problematic = rolling_mean.mean()
        
        # Consistency score: high mean + low variance = consistently bad
        consistency = mean_problematic * 100
        variance_penalty = variance * 50  # Reduce score if inconsistent
        
        score = consistency - variance_penalty
        return max(0, min(100, score))
    
    
    def _calculate_engagement_risk_score(
        self,
        df: pd.DataFrame,
        platform_data: Optional[Dict]
    ) -> float:
        """
        Score based on engagement patterns (0-100).
        Considers: follower count, interaction rates, amplification risk.
        """
        if not platform_data:
            # Default: assume medium risk if no data
            return 50
        
        score = 50  # Base score
        
        # Factor 1: Follower count (more followers = more amplification risk)
        followers = platform_data.get('followers', 0)
        if followers > 10000:
            score += 30
        elif followers > 1000:
            score += 20
        elif followers > 100:
            score += 10
        
        # Factor 2: Engagement rate (high engagement with problematic content)
        engagement_rate = platform_data.get('avg_engagement_rate', 0)
        if engagement_rate > 0.1:  # 10%+
            score += 15
        elif engagement_rate > 0.05:  # 5-10%
            score += 10
        
        # Factor 3: Verified/influential status
        if platform_data.get('is_verified', False):
            score += 15
        
        return min(100, score)
    
    
    def _detect_behavior_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect specific behavior patterns in user's history.
        """
        patterns = {
            'escalation_detected': False,
            'improvement_detected': False,
            'bursty_behavior': False,
            'time_of_day_pattern': None,
            'targeted_harassment': False
        }
        
        # Pattern 1: Escalation detection
        if len(df) >= 10:
            recent_10 = df.tail(10)
            earlier_10 = df.head(10)
            
            recent_hate_ratio = len(recent_10[recent_10['prediction'] == 'Hate speech']) / 10
            earlier_hate_ratio = len(earlier_10[earlier_10['prediction'] == 'Hate speech']) / 10
            
            if recent_hate_ratio > earlier_hate_ratio + 0.3:
                patterns['escalation_detected'] = True
            elif recent_hate_ratio < earlier_hate_ratio - 0.3:
                patterns['improvement_detected'] = True
        
        # Pattern 2: Bursty behavior (lots of posts in short time)
        df['hour_diff'] = df['timestamp'].diff().dt.total_seconds() / 3600
        short_intervals = len(df[df['hour_diff'] < 1])
        if short_intervals > len(df) * 0.3:
            patterns['bursty_behavior'] = True
        
        # Pattern 3: Time of day pattern
        df['hour'] = df['timestamp'].dt.hour
        problematic = df[df['prediction'].isin(['Hate speech', 'Offensive language'])]
        if len(problematic) > 0:
            most_common_hour = problematic['hour'].mode()
            if len(most_common_hour) > 0:
                patterns['time_of_day_pattern'] = f"{most_common_hour.iloc[0]:02d}:00"
        
        # Pattern 4: Targeted harassment (repeated mentions/replies if available)
        # This would require additional data about mentions/targets
        
        return patterns
    
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on score."""
        for level, threshold in sorted(
            self.RISK_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return level
        return 'minimal'
    
    
    def _generate_recommendations(
        self,
        score: float,
        risk_level: str,
        patterns: Dict,
        df: pd.DataFrame
    ) -> List[str]:
        """Generate actionable recommendations based on footprint analysis."""
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == 'critical':
            recommendations.append("🚨 IMMEDIATE ACTION: Account suspension recommended")
            recommendations.append("Escalate to human moderator review")
            recommendations.append("Review all recent content manually")
        elif risk_level == 'high':
            recommendations.append("⚠️ HIGH RISK: Restrict posting privileges")
            recommendations.append("Enable mandatory content review before publishing")
            recommendations.append("Issue final warning to user")
        elif risk_level == 'elevated':
            recommendations.append("⚡ ELEVATED RISK: Issue warning to user")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Consider temporary restrictions")
        elif risk_level == 'moderate':
            recommendations.append("📊 MODERATE RISK: Add to watch list")
            recommendations.append("Enable automated flagging for review")
        else:
            recommendations.append("✅ LOW RISK: Standard monitoring sufficient")
        
        # Pattern-based recommendations
        if patterns['escalation_detected']:
            recommendations.append("⬆️ ESCALATION DETECTED: Behavior is worsening - increase intervention")
        
        if patterns['improvement_detected']:
            recommendations.append("⬇️ IMPROVEMENT DETECTED: Consider reducing restrictions if applicable")
        
        if patterns['bursty_behavior']:
            recommendations.append("💥 BURSTY POSTING: Possible bot or coordinated behavior - investigate")
        
        if patterns['time_of_day_pattern']:
            recommendations.append(
                f"🕐 PATTERN: Most problematic posts around {patterns['time_of_day_pattern']} "
                f"- consider increased monitoring during this time"
            )
        
        # Content-based recommendations
        hate_count = len(df[df['prediction'] == 'Hate speech'])
        if hate_count > 10:
            recommendations.append(
                f"⚠️ HIGH HATE SPEECH COUNT: {hate_count} instances detected - "
                f"review account history for ToS violations"
            )
        
        return recommendations
    
    
    def _create_empty_footprint(self, user_id: str) -> Dict:
        """Create footprint for user with no data."""
        return {
            'user_id': user_id,
            'footprint_score': 0,
            'risk_level': 'minimal',
            'score_breakdown': {k: 0 for k in self.WEIGHTS.keys()},
            'statistics': {
                'total_comments': 0,
                'hate_speech_count': 0,
                'offensive_count': 0,
                'neither_count': 0,
                'average_confidence': 0,
                'time_span_days': 0
            },
            'patterns': {},
            'recommendations': ['No data available for this user'],
            'account_info': {},
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    
    def compare_footprints(
        self,
        footprint1: Dict,
        footprint2: Dict
    ) -> Dict:
        """
        Compare two user footprints or same user across time periods.
        """
        comparison = {
            'user1': footprint1['user_id'],
            'user2': footprint2['user_id'],
            'score_difference': footprint2['footprint_score'] - footprint1['footprint_score'],
            'risk_level_change': f"{footprint1['risk_level']} → {footprint2['risk_level']}",
            'behavior_trend': None,
            'component_changes': {}
        }
        
        # Determine trend
        if comparison['score_difference'] > 10:
            comparison['behavior_trend'] = 'worsening'
        elif comparison['score_difference'] < -10:
            comparison['behavior_trend'] = 'improving'
        else:
            comparison['behavior_trend'] = 'stable'
        
        # Component-wise comparison
        for component in self.WEIGHTS.keys():
            diff = (
                footprint2['score_breakdown'][component] -
                footprint1['score_breakdown'][component]
            )
            comparison['component_changes'][component] = round(diff, 2)
        
        return comparison
    
    
    def batch_analyze_users(
        self,
        users_data: Dict[str, pd.DataFrame],
        account_metadata: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """
        Analyze multiple users and return ranked DataFrame.
        
        Args:
            users_data: Dict mapping user_id to their comments DataFrame
            account_metadata: Optional dict with account info per user
        
        Returns:
            DataFrame with all users ranked by footprint score
        """
        results = []
        
        for user_id, comments_df in users_data.items():
            metadata = account_metadata.get(user_id) if account_metadata else None
            account_age = metadata.get('account_age_days') if metadata else None
            platform_data = metadata.get('platform_data') if metadata else None
            
            footprint = self.calculate_footprint_score(
                user_id=user_id,
                comments_df=comments_df,
                account_age_days=account_age,
                platform_data=platform_data
            )
            
            results.append({
                'user_id': user_id,
                'footprint_score': footprint['footprint_score'],
                'risk_level': footprint['risk_level'],
                'total_comments': footprint['statistics']['total_comments'],
                'hate_speech_count': footprint['statistics']['hate_speech_count'],
                'content_severity': footprint['score_breakdown']['content_severity'],
                'temporal_trend': footprint['score_breakdown']['temporal_trend'],
                'recency': footprint['score_breakdown']['recency'],
                'escalation_detected': footprint['patterns'].get('escalation_detected', False),
                'top_recommendation': footprint['recommendations'][0] if footprint['recommendations'] else ''
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('footprint_score', ascending=False)
        
        return df
    
    
    def export_footprint(self, footprint: Dict, filepath: Path):
        """Export footprint to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(footprint, f, indent=2)
        logger.info(f"Footprint exported to {filepath}")
    
    
    def visualize_footprint(self, footprint: Dict) -> str:
        """Generate text-based visualization of footprint (for CLI)."""
        score = footprint['footprint_score']
        risk_level = footprint['risk_level']
        
        # Create score bar
        bar_length = 50
        filled = int((score / 100) * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        viz = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     DIGITAL FOOTPRINT ANALYSIS                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║ User: {footprint['user_id']:<66} ║
║ Score: {score:>5.1f}/100  [{bar}] ║
║ Risk Level: {risk_level.upper():<60} ║
╠══════════════════════════════════════════════════════════════════════════╣
║ SCORE BREAKDOWN:                                                        ║
║   Content Severity:  {footprint['score_breakdown']['content_severity']:>5.1f} / 100                                     ║
║   Temporal Trend:    {footprint['score_breakdown']['temporal_trend']:>5.1f} / 100                                     ║
║   Frequency:         {footprint['score_breakdown']['frequency']:>5.1f} / 100                                     ║
║   Recency:           {footprint['score_breakdown']['recency']:>5.1f} / 100                                     ║
║   Consistency:       {footprint['score_breakdown']['consistency']:>5.1f} / 100                                     ║
║   Engagement Risk:   {footprint['score_breakdown']['engagement_risk']:>5.1f} / 100                                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║ STATISTICS:                                                              ║
║   Total Comments:    {footprint['statistics']['total_comments']:<52} ║
║   Hate Speech:       {footprint['statistics']['hate_speech_count']:<52} ║
║   Offensive:         {footprint['statistics']['offensive_count']:<52} ║
║   Time Span:         {footprint['statistics']['time_span_days']:} days{' ' * (48 - len(str(footprint['statistics']['time_span_days'])))} ║
╠══════════════════════════════════════════════════════════════════════════╣
║ TOP RECOMMENDATIONS:                                                     ║
"""
        
        for rec in footprint['recommendations'][:3]:
            # Wrap long recommendations
            if len(rec) > 68:
                rec = rec[:65] + '...'
            viz += f"║   • {rec:<68} ║\n"
        
        viz += "╚══════════════════════════════════════════════════════════════════════════╝"
        
        return viz


# ==================== INTEGRATION FUNCTIONS ====================

def integrate_with_csv_processor(csv_results_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Integrate with existing CSV processor to add digital footprint scores.
    
    Args:
        csv_results_df: DataFrame from CSVProcessor.process_csv()
    
    Returns:
        Dict mapping user_id to footprint data
    """
    scorer = DigitalFootprintScorer()
    
    # Group by user
    user_footprints = {}
    for user_id in csv_results_df['user'].unique():
        user_df = csv_results_df[csv_results_df['user'] == user_id].copy()
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in user_df.columns:
            # Create synthetic timestamps if not available
            user_df['timestamp'] = pd.date_range(
                end=datetime.now(),
                periods=len(user_df),
                freq='1H'
            )
        elif not pd.api.types.is_datetime64_any_dtype(user_df['timestamp']):
            user_df['timestamp'] = pd.to_datetime(user_df['timestamp'])
        
        footprint = scorer.calculate_footprint_score(
            user_id=user_id,
            comments_df=user_df
        )
        
        user_footprints[user_id] = footprint
    
    return user_footprints


def demo_footprint_analysis():
    """Demo function showing how to use the Digital Footprint Scorer."""
    print("\n" + "="*80)
    print("DIGITAL FOOTPRINT SCORING SYSTEM - DEMO")
    print("="*80 + "\n")
    
    # Create sample data
    sample_data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='D'),
        'comment': ['sample text'] * 50,
        'prediction': ['Hate speech'] * 15 + ['Offensive language'] * 20 + ['Neither'] * 15,
        'confidence': np.random.uniform(0.7, 0.95, 50)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize scorer
    scorer = DigitalFootprintScorer()
    
    # Calculate footprint
    footprint = scorer.calculate_footprint_score(
        user_id='demo_user_001',
        comments_df=df,
        account_age_days=365,
        platform_data={'followers': 5000, 'avg_engagement_rate': 0.08}
    )
    
    # Visualize
    print(scorer.visualize_footprint(footprint))
    
    print("\n📊 Full footprint data:")
    print(json.dumps(footprint, indent=2))


if __name__ == "__main__":
    demo_footprint_analysis()