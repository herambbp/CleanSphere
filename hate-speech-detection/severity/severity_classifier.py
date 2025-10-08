"""
Severity Classification System for Hate Speech Detection
Rule-based multi-level severity detection (LOW to EXTREME)

Components:
1. KeywordDetector - Detects 8 categories of harmful keywords
2. TextFeaturesAnalyzer - Analyzes text intensity features
3. ContextAnalyzer - Detects context to adjust severity
4. SeverityScorer - Main class combining all components
"""

import re
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

from config import (
    SEVERITY_LEVELS, SEVERITY_WEIGHTS, SEVERITY_THRESHOLDS,
    MAX_SEVERITY_SCORE, CONTEXT_MODIFIERS,
    VIOLENCE_KEYWORDS, THREAT_PATTERNS, DEHUMANIZATION_KEYWORDS,
    RACIAL_SLURS, LGBTQ_SLURS, SEXIST_SLURS, RELIGIOUS_SLURS,
    ABLEIST_SLURS, SARCASM_INDICATORS, EDUCATIONAL_INDICATORS,
    QUOTE_INDICATORS, MENTION_PATTERN, URL_PATTERN
)

# ==================== KEYWORD DETECTOR ====================

class KeywordDetector:
    """
    Detect various categories of harmful keywords and patterns.
    
    Detects:
    - Violence keywords
    - Explicit threat patterns
    - Dehumanization terms
    - Racial slurs
    - LGBTQ slurs
    - Sexist/misogynistic slurs
    - Religious hate terms
    - Ableist slurs
    """
    
    def __init__(self):
        """Initialize keyword detector with all keyword lists."""
        self.violence_keywords = set(k.lower() for k in VIOLENCE_KEYWORDS)
        self.threat_patterns = [p.lower() for p in THREAT_PATTERNS]
        self.dehumanization = set(k.lower() for k in DEHUMANIZATION_KEYWORDS)
        self.racial_slurs = set(k.lower() for k in RACIAL_SLURS)
        self.lgbtq_slurs = set(k.lower() for k in LGBTQ_SLURS)
        self.sexist_slurs = set(k.lower() for k in SEXIST_SLURS)
        self.religious_slurs = set(k.lower() for k in RELIGIOUS_SLURS)
        self.ableist_slurs = set(k.lower() for k in ABLEIST_SLURS)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for keyword matching.
        Handles l33t speak and common variations.
        """
        text = text.lower()
        
        # L33t speak replacements
        replacements = {
            '4': 'a', '3': 'e', '1': 'i', '0': 'o',
            '5': 's', '7': 't', '@': 'a', '$': 's'
        }
        
        for leet, normal in replacements.items():
            text = text.replace(leet, normal)
        
        # Remove common obfuscations
        text = text.replace('*', '').replace('-', '').replace('_', '')
        
        return text
    
    def _find_keywords_in_text(
        self, 
        text: str, 
        keywords: Set[str]
    ) -> Tuple[int, List[str]]:
        """
        Find keywords in text (handles partial matches and variations).
        
        Returns:
            Tuple of (count, matched_keywords)
        """
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        matched = []
        for keyword in keywords:
            # Check if keyword appears in text (word boundary aware)
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, normalized):
                matched.append(keyword)
        
        return len(matched), matched
    
    def _find_patterns_in_text(
        self, 
        text: str, 
        patterns: List[str]
    ) -> Tuple[int, List[str]]:
        """
        Find multi-word patterns in text.
        
        Returns:
            Tuple of (count, matched_patterns)
        """
        normalized = self._normalize_text(text)
        
        matched = []
        for pattern in patterns:
            if pattern in normalized:
                matched.append(pattern)
        
        return len(matched), matched
    
    def detect_violence(self, text: str) -> Dict:
        """Detect violence keywords."""
        count, matches = self._find_keywords_in_text(text, self.violence_keywords)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['violence_keywords']
        }
    
    def detect_threats(self, text: str) -> Dict:
        """Detect explicit threat patterns."""
        count, matches = self._find_patterns_in_text(text, self.threat_patterns)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['explicit_threats']
        }
    
    def detect_dehumanization(self, text: str) -> Dict:
        """Detect dehumanization terms."""
        count, matches = self._find_keywords_in_text(text, self.dehumanization)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['dehumanization']
        }
    
    def detect_racial_slurs(self, text: str) -> Dict:
        """Detect racial slurs."""
        count, matches = self._find_keywords_in_text(text, self.racial_slurs)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['racial_slurs']
        }
    
    def detect_lgbtq_slurs(self, text: str) -> Dict:
        """Detect LGBTQ slurs."""
        count, matches = self._find_keywords_in_text(text, self.lgbtq_slurs)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['lgbtq_slurs']
        }
    
    def detect_sexist_slurs(self, text: str) -> Dict:
        """Detect sexist/misogynistic slurs."""
        count, matches = self._find_keywords_in_text(text, self.sexist_slurs)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['sexist_slurs']
        }
    
    def detect_religious_slurs(self, text: str) -> Dict:
        """Detect religious hate terms."""
        count, matches = self._find_keywords_in_text(text, self.religious_slurs)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['religious_slurs']
        }
    
    def detect_ableist_slurs(self, text: str) -> Dict:
        """Detect ableist slurs."""
        count, matches = self._find_keywords_in_text(text, self.ableist_slurs)
        return {
            'count': count,
            'matches': matches,
            'score': count * SEVERITY_WEIGHTS['ableist_slurs']
        }
    
    def detect_all(self, text: str) -> Dict:
        """
        Detect all keyword categories.
        
        Returns:
            Dictionary with all detection results
        """
        results = {
            'violence': self.detect_violence(text),
            'threats': self.detect_threats(text),
            'dehumanization': self.detect_dehumanization(text),
            'racial_slurs': self.detect_racial_slurs(text),
            'lgbtq_slurs': self.detect_lgbtq_slurs(text),
            'sexist_slurs': self.detect_sexist_slurs(text),
            'religious_slurs': self.detect_religious_slurs(text),
            'ableist_slurs': self.detect_ableist_slurs(text)
        }
        
        # Calculate total score
        total_score = sum(r['score'] for r in results.values())
        
        # Count number of slur categories detected
        slur_categories = sum(
            1 for k, v in results.items() 
            if 'slur' in k and v['count'] > 0
        )
        
        # Add bonus for multiple slur categories
        if slur_categories > 1:
            multiple_slurs_bonus = (slur_categories - 1) * SEVERITY_WEIGHTS['multiple_slurs']
            total_score += multiple_slurs_bonus
        else:
            multiple_slurs_bonus = 0
        
        results['total_score'] = total_score
        results['multiple_slurs_bonus'] = multiple_slurs_bonus
        results['slur_categories_count'] = slur_categories
        
        return results

# ==================== TEXT FEATURES ANALYZER ====================

class TextFeaturesAnalyzer:
    """
    Analyze text features that indicate severity.
    
    Features:
    - CAPS ratio (excessive capitalization)
    - Repeated punctuation (!!!, ???)
    - Targeted at person (@mention + harmful content)
    """
    
    def __init__(self):
        """Initialize text features analyzer."""
        pass
    
    def get_caps_ratio(self, text: str) -> float:
        """
        Calculate ratio of uppercase characters.
        
        Returns:
            Float between 0 and 1
        """
        if not text:
            return 0.0
        
        # Remove URLs and mentions to avoid false positives
        text_clean = re.sub(URL_PATTERN, '', text)
        text_clean = re.sub(MENTION_PATTERN, '', text_clean)
        
        # Remove non-alphabetic characters
        letters = [c for c in text_clean if c.isalpha()]
        
        if not letters:
            return 0.0
        
        caps_count = sum(1 for c in letters if c.isupper())
        return caps_count / len(letters)
    
    def has_excessive_punctuation(self, text: str) -> Tuple[bool, int]:
        """
        Check for excessive repeated punctuation.
        
        Returns:
            Tuple of (has_excessive, count)
        """
        # Find repeated punctuation patterns
        patterns = [
            r'!{3,}',  # !!!
            r'\?{3,}',  # ???
            r'\.{4,}'  # ....
        ]
        
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            count += len(matches)
        
        return count > 0, count
    
    def is_targeted_at_person(self, text: str) -> Tuple[bool, int]:
        """
        Check if content is targeted at a specific person.
        Uses @mentions as indicator.
        
        Returns:
            Tuple of (is_targeted, mention_count)
        """
        mentions = re.findall(MENTION_PATTERN, text)
        return len(mentions) > 0, len(mentions)
    
    def get_text_intensity_score(self, text: str) -> int:
        """
        Calculate overall text intensity score from features.
        
        Returns:
            Score (0-30 points possible)
        """
        score = 0
        
        # CAPS ratio
        caps_ratio = self.get_caps_ratio(text)
        if caps_ratio > 0.5:  # More than 50% caps
            score += SEVERITY_WEIGHTS['all_caps_ratio']
        
        # Repeated punctuation
        has_punct, punct_count = self.has_excessive_punctuation(text)
        if has_punct:
            score += SEVERITY_WEIGHTS['repeated_punctuation']
        
        # Targeted at person
        is_targeted, mention_count = self.is_targeted_at_person(text)
        if is_targeted:
            score += SEVERITY_WEIGHTS['targeted_at_person']
        
        return score
    
    def analyze_all(self, text: str) -> Dict:
        """
        Analyze all text features.
        
        Returns:
            Dictionary with all feature analysis
        """
        caps_ratio = self.get_caps_ratio(text)
        has_punct, punct_count = self.has_excessive_punctuation(text)
        is_targeted, mention_count = self.is_targeted_at_person(text)
        total_score = self.get_text_intensity_score(text)
        
        return {
            'caps_ratio': round(caps_ratio, 2),
            'has_excessive_caps': caps_ratio > 0.5,
            'has_excessive_punctuation': has_punct,
            'punctuation_count': punct_count,
            'is_targeted_at_person': is_targeted,
            'mention_count': mention_count,
            'total_score': total_score
        }

# ==================== CONTEXT ANALYZER ====================

class ContextAnalyzer:
    """
    Analyze context to adjust severity scores.
    
    Contexts:
    - Quotes (reporting someone else's words)
    - Questions (asking vs stating)
    - Retweets (sharing vs original)
    - Sarcasm (not serious)
    - News/Educational (awareness/discussion)
    """
    
    def __init__(self):
        """Initialize context analyzer."""
        self.sarcasm_indicators = [s.lower() for s in SARCASM_INDICATORS]
        self.educational_indicators = [e.lower() for e in EDUCATIONAL_INDICATORS]
        self.quote_indicators = [q.lower() for q in QUOTE_INDICATORS]
    
    def is_quote(self, text: str) -> Tuple[bool, str]:
        """
        Check if text contains quotes (reporting someone else).
        
        Returns:
            Tuple of (is_quote, reason)
        """
        # Check for actual quote marks (NOT apostrophes)
        # Only detect proper quotation marks: " " "
        # Removed "'" to avoid catching contractions like "you're", "they're"
        if '"' in text or '"' in text or '"' in text:
            return True, "Contains quote marks"
        
        # Check for quote indicators
        text_lower = text.lower()
        for indicator in self.quote_indicators:
            if indicator in text_lower:
                return True, f"Contains '{indicator}'"
        
        return False, ""
    
    def is_question(self, text: str) -> bool:
        """Check if text is a question."""
        return '?' in text
    
    def is_retweet(self, text: str) -> bool:
        """Check if text is a retweet."""
        text_lower = text.lower().strip()
        return text_lower.startswith('rt ') or text_lower.startswith('rt:')
    
    def detect_sarcasm(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect sarcasm indicators.
        
        Returns:
            Tuple of (has_sarcasm, matched_indicators)
        """
        text_lower = text.lower()
        matches = []
        
        for indicator in self.sarcasm_indicators:
            if indicator in text_lower:
                matches.append(indicator)
        
        return len(matches) > 0, matches
    
    def has_news_context(self, text: str) -> bool:
        """Check if text contains news URL or reference."""
        # Check for URLs
        if re.search(URL_PATTERN, text):
            return True
        
        # Check for news-related terms
        news_terms = ['breaking', 'news', 'report', 'article', 'source']
        text_lower = text.lower()
        return any(term in text_lower for term in news_terms)
    
    def is_educational(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text has educational/awareness context.
        
        Returns:
            Tuple of (is_educational, matched_indicators)
        """
        text_lower = text.lower()
        matches = []
        
        for indicator in self.educational_indicators:
            if indicator in text_lower:
                matches.append(indicator)
        
        return len(matches) > 0, matches
    
    def get_context_adjustment(self, text: str) -> Dict:
        """
        Calculate total context adjustment to severity score.
        
        Returns:
            Dictionary with adjustment details
        """
        adjustment = 0
        reasons = []
        
        # Check quote
        is_quote, quote_reason = self.is_quote(text)
        if is_quote:
            adjustment += CONTEXT_MODIFIERS['has_quote_marks']
            reasons.append(f"Quote detected ({quote_reason})")
        
        # Check question
        if self.is_question(text):
            adjustment += CONTEXT_MODIFIERS['has_question_mark']
            reasons.append("Question format")
        
        # Check retweet
        if self.is_retweet(text):
            adjustment += CONTEXT_MODIFIERS['starts_with_rt']
            reasons.append("Retweet/sharing")
        
        # Check sarcasm
        has_sarcasm, sarcasm_matches = self.detect_sarcasm(text)
        if has_sarcasm:
            adjustment += CONTEXT_MODIFIERS['has_sarcasm']
            reasons.append(f"Sarcasm detected: {', '.join(sarcasm_matches)}")
        
        # Check news context
        if self.has_news_context(text):
            adjustment += CONTEXT_MODIFIERS['contains_news_url']
            reasons.append("News/article context")
        
        # Check educational
        is_edu, edu_matches = self.is_educational(text)
        if is_edu:
            adjustment += CONTEXT_MODIFIERS['is_educational']
            reasons.append(f"Educational context: {', '.join(edu_matches)}")
        
        return {
            'total_adjustment': adjustment,
            'reasons': reasons,
            'is_quote': is_quote,
            'is_question': self.is_question(text),
            'is_retweet': self.is_retweet(text),
            'has_sarcasm': has_sarcasm,
            'has_news_context': self.has_news_context(text),
            'is_educational': is_edu
        }

# ==================== SEVERITY SCORER (MAIN) ====================

class SeverityScorer:
    """
    Main severity scoring system.
    
    Combines:
    - Keyword detection (violence, threats, slurs)
    - Text features (caps, punctuation, targeting)
    - Context analysis (quotes, sarcasm, education)
    
    Outputs:
    - Severity score (0-100)
    - Severity level (1-5: LOW to EXTREME)
    - Detailed explanation
    """
    
    def __init__(self):
        """Initialize severity scorer with all components."""
        self.keyword_detector = KeywordDetector()
        self.text_analyzer = TextFeaturesAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def calculate_base_score(self, text: str) -> Tuple[int, Dict]:
        """
        Calculate base severity score before context adjustments.
        
        Returns:
            Tuple of (score, factors_dict)
        """
        # Detect keywords
        keyword_results = self.keyword_detector.detect_all(text)
        
        # Analyze text features
        text_features = self.text_analyzer.analyze_all(text)
        
        # Combine scores
        base_score = keyword_results['total_score'] + text_features['total_score']
        
        # Compile factors
        factors = {
            'violence_keywords': keyword_results['violence']['count'],
            'threat_patterns': keyword_results['threats']['count'],
            'dehumanization_terms': keyword_results['dehumanization']['count'],
            'racial_slurs': keyword_results['racial_slurs']['count'],
            'lgbtq_slurs': keyword_results['lgbtq_slurs']['count'],
            'sexist_slurs': keyword_results['sexist_slurs']['count'],
            'religious_slurs': keyword_results['religious_slurs']['count'],
            'ableist_slurs': keyword_results['ableist_slurs']['count'],
            'multiple_slurs_bonus': keyword_results['multiple_slurs_bonus'],
            'caps_ratio': text_features['caps_ratio'],
            'excessive_punctuation': text_features['has_excessive_punctuation'],
            'targeted_at_person': text_features['is_targeted_at_person']
        }
        
        # Add matched keywords for explanation
        factors['matched_keywords'] = {
            'violence': keyword_results['violence']['matches'],
            'threats': keyword_results['threats']['matches'],
            'dehumanization': keyword_results['dehumanization']['matches'],
            'racial_slurs': keyword_results['racial_slurs']['matches'],
            'lgbtq_slurs': keyword_results['lgbtq_slurs']['matches'],
            'sexist_slurs': keyword_results['sexist_slurs']['matches'],
            'religious_slurs': keyword_results['religious_slurs']['matches'],
            'ableist_slurs': keyword_results['ableist_slurs']['matches']
        }
        
        return base_score, factors
    
    def adjust_for_context(self, score: int, text: str) -> Tuple[int, Dict]:
        """
        Apply context adjustments to score.
        
        Returns:
            Tuple of (adjusted_score, context_info)
        """
        context_info = self.context_analyzer.get_context_adjustment(text)
        adjusted_score = score + context_info['total_adjustment']
        
        # Ensure score is within bounds
        adjusted_score = max(0, min(MAX_SEVERITY_SCORE, adjusted_score))
        
        return adjusted_score, context_info
    
    def score_to_level(self, score: int) -> Tuple[int, str]:
        """
        Convert severity score to level.
        
        Returns:
            Tuple of (level_number, level_name)
        """
        for level, (min_score, max_score) in SEVERITY_THRESHOLDS.items():
            if min_score <= score <= max_score:
                return level, SEVERITY_LEVELS[level]
        
        # Default to EXTREME if score is very high
        return 5, SEVERITY_LEVELS[5]
    
    def generate_explanation(
        self, 
        factors: Dict, 
        context_info: Dict,
        score: int,
        level: int
    ) -> str:
        """
        Generate human-readable explanation of severity.
        
        Returns:
            Explanation string
        """
        parts = []
        
        # Main severity factors
        if factors['violence_keywords'] > 0:
            parts.append(f"{factors['violence_keywords']} violence keyword(s)")
        
        if factors['threat_patterns'] > 0:
            parts.append(f"{factors['threat_patterns']} explicit threat(s)")
        
        if factors['dehumanization_terms'] > 0:
            parts.append(f"{factors['dehumanization_terms']} dehumanization term(s)")
        
        # Slurs
        slur_counts = []
        if factors['racial_slurs'] > 0:
            slur_counts.append(f"{factors['racial_slurs']} racial")
        if factors['lgbtq_slurs'] > 0:
            slur_counts.append(f"{factors['lgbtq_slurs']} LGBTQ")
        if factors['sexist_slurs'] > 0:
            slur_counts.append(f"{factors['sexist_slurs']} sexist")
        if factors['religious_slurs'] > 0:
            slur_counts.append(f"{factors['religious_slurs']} religious")
        if factors['ableist_slurs'] > 0:
            slur_counts.append(f"{factors['ableist_slurs']} ableist")
        
        if slur_counts:
            parts.append(f"slurs ({', '.join(slur_counts)})")
        
        # Text features
        if factors['caps_ratio'] > 0.5:
            parts.append(f"excessive caps ({factors['caps_ratio']:.0%})")
        
        if factors['excessive_punctuation']:
            parts.append("excessive punctuation")
        
        if factors['targeted_at_person']:
            parts.append("targeted at person (@mention)")
        
        # Build explanation
        if not parts:
            explanation = f"{SEVERITY_LEVELS[level]} severity - No major harmful indicators detected"
        else:
            explanation = f"{SEVERITY_LEVELS[level]} severity due to: {', '.join(parts)}"
        
        # Add context adjustments
        if context_info['reasons']:
            explanation += f". Context adjustments: {', '.join(context_info['reasons'])}"
        
        return explanation
    
    def analyze_severity(self, text: str, verbose: bool = False) -> Dict:
        """
        Complete severity analysis of text.
        
        Args:
            text: Text to analyze
            verbose: If True, print detailed analysis
        
        Returns:
            Dictionary with complete severity analysis
        """
        # Calculate base score
        base_score, factors = self.calculate_base_score(text)
        
        # Apply context adjustments
        final_score, context_info = self.adjust_for_context(base_score, text)
        
        # Convert to level
        level, level_name = self.score_to_level(final_score)
        
        # Generate explanation
        explanation = self.generate_explanation(factors, context_info, final_score, level)
        
        # Compile result
        result = {
            'severity_score': final_score,
            'base_score': base_score,
            'severity_level': level,
            'severity_label': level_name,
            'factors': factors,
            'context_adjustments': context_info,
            'explanation': explanation
        }
        
        if verbose:
            self._print_analysis(text, result)
        
        return result
    
    def _print_analysis(self, text: str, result: Dict):
        """Print detailed analysis (for debugging/testing)."""
        print("\n" + "=" * 80)
        print("SEVERITY ANALYSIS")
        print("=" * 80)
        print(f"\nText: {text[:150]}...")
        print(f"\nSeverity: {result['severity_label']} (Level {result['severity_level']})")
        print(f"Score: {result['severity_score']}/100 (Base: {result['base_score']})")
        print(f"\nExplanation: {result['explanation']}")
        print("=" * 80)

# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 80)
    print("SEVERITY CLASSIFIER TEST")
    print("=" * 80)
    
    # Initialize scorer
    scorer = SeverityScorer()
    
    # Test cases with expected severity levels
    test_cases = [
        # EXTREME (5)
        ("I will kill you and your entire family you fucking bitch", 5),
        ("All n*ggers should be exterminated from this country", 5),
        
        # SEVERE (4)
        ("You're worthless scum and deserve to die you f*ggot", 4),
        ("I hope someone shoots you dead you piece of trash", 4),
        
        # HIGH (3)
        ("You're subhuman trash and everyone hates you", 3),
        ("All [group] are vermin and parasites", 3),
        
        # MODERATE (2)
        ("You're a fucking idiot and nobody likes you bitch", 2),
        ("Shut up you dumb cunt", 2),
        
        # LOW (1)
        ("I disagree with you, this is so frustrating", 1),
        ("You're annoying and wrong", 1),
        
        # Edge cases
        ('He said "I hate you" - can you believe that?', 1),  # Quote
        ("Yeah right, like I'd ever kill someone /s", 1),  # Sarcasm
    ]
    
    print(f"\nTesting {len(test_cases)} cases...\n")
    
    correct = 0
    for i, (text, expected_level) in enumerate(test_cases, 1):
        result = scorer.analyze_severity(text, verbose=False)
        
        actual_level = result['severity_level']
        is_correct = actual_level == expected_level
        
        if is_correct:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"{i}. [{status}] Expected: {expected_level}, Got: {actual_level}")
        print(f"   Text: {text[:70]}...")
        print(f"   Score: {result['severity_score']}, Label: {result['severity_label']}")
        print(f"   {result['explanation']}")
        print()
    
    print("=" * 80)
    print(f"RESULTS: {correct}/{len(test_cases)} tests passed ({correct/len(test_cases)*100:.1f}%)")
    print("=" * 80)
    
    # Detailed analysis of one example
    print("\n\nDETAILED ANALYSIS EXAMPLE:")
    sample = "I will kill you bitch you're worthless trash"
    scorer.analyze_severity(sample, verbose=True)