"""
Dataset Augmentation Pipeline
Augment 24K → 100K with perfect class balance using 360K LLM-annotated data

Strategy:
1. Load 360K dataset + existing 24K
2. Filter by quality (high confidence samples)
3. Apply hate/offensive lexicons for validation
4. Stratified sampling for diversity
5. Perfect balance: 33,333 each class
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class AugmentationConfig:
    # Paths
    ORIGINAL_DATA = Path('onelasttry/labeled_data.csv')  # Your 24K
    NEW_DATA = Path('onelasttry/final_dataset_reannotated.csv')   # 360K dataset
    OUTPUT_DATA = Path('onelastry/augmented_100k.csv')  # Final output
    
    # Target sizes
    TOTAL_TARGET = 100000
    TARGET_PER_CLASS = 33333  # Perfect balance
    
    # Quality thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.7  # % of annotators agreeing
    MIN_TEXT_LENGTH = 10           # Minimum characters
    MAX_TEXT_LENGTH = 500          # Maximum characters
    
    # Sampling strategy
    ENSURE_DIVERSITY = True
    MAX_SIMILAR_TEXTS = 5  # Max similar texts per group

# ==================== HATE/OFFENSIVE LEXICONS ====================

class HateSpeechLexicon:
    """Comprehensive lexicons for validation."""
    
    # Group-targeting hate speech (racial, religious, gender, sexuality)
    HATE_KEYWORDS = {
        # Racial slurs
        'racial': [
            'nigger', 'nigga', 'negro', 'coon', 'monkey',
            'ch*nk', 'gook', 'spic', 'wetback', 'beaner',
            'towelhead', 'sandnigger', 'paki', 'curry',
            'kike', 'hymie', 'cracker', 'honky', 'whitey'
        ],
        
        # Religious hate
        'religious': [
            'raghead', 'terrorist', 'jihad', 'isis',
            'muslim scum', 'muslim trash', 'islam is',
            'all muslims', 'every muslim', 'muslims are',
            'jewish conspiracy', 'zionist', 'christ killer'
        ],
        
        # Gender/sexuality hate
        'lgbtq': [
            'faggot', 'fag', 'dyke', 'tranny', 'trannie',
            'sodomite', 'homo', 'queer', 'gay scum',
            'all gays', 'lgbt agenda', 'gender ideology'
        ],
        
        # Misogyny
        'gender': [
            'feminazi', 'femoid', 'roastie', 'thot',
            'all women', 'women are', 'females are',
            'bitches are', 'sluts are', 'whores are'
        ],
        
        # Dehumanization
        'dehumanization': [
            'subhuman', 'vermin', 'animals', 'scum', 'trash',
            'parasites', 'cockroaches', 'rats', 'filth',
            'inferior race', 'primitive', 'savage'
        ],
        
        # Calls for violence
        'violence': [
            'kill all', 'genocide', 'exterminate', 'eradicate',
            'purge', 'cleanse', 'eliminate', 'get rid of',
            'send them back', 'deport all', 'ban all'
        ],
        
        # Generalizations
        'generalizations': [
            'all [GROUP] are', 'every [GROUP]', '[GROUP] are all',
            'typical [GROUP]', 'those [GROUP]', 'these [GROUP]'
        ]
    }
    
    # Individual-targeting offensive language
    OFFENSIVE_KEYWORDS = {
        'profanity': [
            'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell',
            'bastard', 'asshole', 'dickhead', 'prick', 'cunt'
        ],
        
        'insults': [
            'idiot', 'stupid', 'dumb', 'moron', 'retard',
            'loser', 'pathetic', 'ugly', 'fat', 'disgusting'
        ],
        
        'vulgar': [
            'pussy', 'dick', 'cock', 'balls', 'tits',
            'whore', 'slut', 'hoe', 'skank'
        ]
    }
    
    # Neither (clean language indicators)
    NEITHER_INDICATORS = [
        'according to', 'report', 'study', 'research',
        'news', 'article', 'statement', 'official',
        'question', 'wondering', 'think', 'believe'
    ]
    
    @classmethod
    def check_hate_speech(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Check if text contains hate speech patterns.
        
        Returns:
            (is_hate, matched_keywords)
        """
        text_lower = text.lower()
        matched = []
        
        for category, keywords in cls.HATE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched.append(f"{category}:{keyword}")
        
        return len(matched) > 0, matched
    
    @classmethod
    def check_offensive(cls, text: str) -> Tuple[bool, List[str]]:
        """Check if text contains offensive language."""
        text_lower = text.lower()
        matched = []
        
        for category, keywords in cls.OFFENSIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched.append(f"{category}:{keyword}")
        
        return len(matched) > 0, matched
    
    @classmethod
    def check_neither(cls, text: str) -> bool:
        """Check if text looks like neutral content."""
        text_lower = text.lower()
        
        # Check for neutral indicators
        neutral_score = sum(1 for indicator in cls.NEITHER_INDICATORS if indicator in text_lower)
        
        # Check for absence of hate/offensive
        has_hate, _ = cls.check_hate_speech(text)
        has_offensive, _ = cls.check_offensive(text)
        
        return neutral_score > 0 and not has_hate and not has_offensive


# ==================== DATA QUALITY FILTERS ====================

class QualityFilter:
    """Filter low-quality samples."""
    
    @staticmethod
    def calculate_confidence(row) -> float:
        """
        Calculate annotation confidence.
        
        confidence = max(hate_speech, offensive_language, neither) / total_votes
        """
        total_votes = row['hate_speech'] + row['offensive_language'] + row['neither']
        if total_votes == 0:
            return 0.0
        
        max_votes = max(row['hate_speech'], row['offensive_language'], row['neither'])
        return max_votes / total_votes
    
    @staticmethod
    def is_high_quality(row, min_confidence=0.7) -> bool:
        """Check if sample meets quality criteria."""
        
        # Check confidence
        confidence = QualityFilter.calculate_confidence(row)
        if confidence < min_confidence:
            return False
        
        # Check text length
        text = str(row['tweet'])
        if len(text) < AugmentationConfig.MIN_TEXT_LENGTH:
            return False
        if len(text) > AugmentationConfig.MAX_TEXT_LENGTH:
            return False
        
        # Check for valid text (not just URLs, mentions, etc.)
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 3:  # At least 3 words
            return False
        
        return True
    
    @staticmethod
    def validate_with_lexicon(row) -> bool:
        """Validate LLM label with lexicon."""
        text = str(row['tweet'])
        predicted_class = row['class']
        
        is_hate, _ = HateSpeechLexicon.check_hate_speech(text)
        is_offensive, _ = HateSpeechLexicon.check_offensive(text)
        is_neither = HateSpeechLexicon.check_neither(text)
        
        # Validate hate speech
        if predicted_class == 0:  # Hate
            return is_hate  # Must have hate keywords
        
        # Validate offensive
        elif predicted_class == 1:  # Offensive
            # Should have offensive keywords but NOT hate
            return is_offensive and not is_hate
        
        # Validate neither
        elif predicted_class == 2:  # Neither
            # Should not have hate or offensive
            return not is_hate and not is_offensive
        
        return False


# ==================== DIVERSITY SAMPLER ====================

class DiversitySampler:
    """Ensure diverse samples (avoid repetitive content)."""
    
    @staticmethod
    def get_text_signature(text: str) -> str:
        """Get simplified text signature for similarity check."""
        # Remove URLs, mentions, numbers
        text = re.sub(r'http\S+|www\S+|@\w+|\d+', '', text.lower())
        # Keep only alphanumeric
        words = re.findall(r'\b\w{4,}\b', text)  # Words with 4+ chars
        # Sort and join
        return ' '.join(sorted(words[:10]))  # Use first 10 significant words
    
    @staticmethod
    def stratified_sample(
        df: pd.DataFrame,
        target_size: int,
        max_similar: int = 5
    ) -> pd.DataFrame:
        """
        Sample with diversity constraints.
        
        Args:
            df: DataFrame to sample from
            target_size: Number of samples needed
            max_similar: Maximum similar texts allowed
        
        Returns:
            Sampled DataFrame
        """
        print(f"  Sampling {target_size} from {len(df)} available...")
        
        if len(df) <= target_size:
            return df
        
        # Track text signatures to avoid duplicates
        signature_counter = Counter()
        selected_indices = []
        
        # Shuffle for randomness
        shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for idx, row in shuffled_df.iterrows():
            if len(selected_indices) >= target_size:
                break
            
            text_sig = DiversitySampler.get_text_signature(row['tweet'])
            
            # Check if we've seen too many similar texts
            if signature_counter[text_sig] < max_similar:
                selected_indices.append(idx)
                signature_counter[text_sig] += 1
        
        print(f"  Selected {len(selected_indices)} diverse samples")
        print(f"  Unique signatures: {len(signature_counter)}")
        
        return shuffled_df.loc[selected_indices]


# ==================== MAIN AUGMENTATION PIPELINE ====================

class DatasetAugmentor:
    """Main augmentation pipeline."""
    
    def __init__(self):
        self.original_df = None
        self.new_df = None
        self.augmented_df = None
        
        self.stats = {
            'original_size': 0,
            'new_data_size': 0,
            'filtered_size': 0,
            'final_size': 0,
            'class_distribution': {}
        }
    
    def load_data(self):
        """Load original and new datasets."""
        print("=" * 80)
        print("STEP 1: LOADING DATASETS")
        print("=" * 80)
        
        # Load original 24K
        print(f"\nLoading original dataset: {AugmentationConfig.ORIGINAL_DATA}")
        self.original_df = pd.read_csv(AugmentationConfig.ORIGINAL_DATA, encoding='latin-1')
        self.stats['original_size'] = len(self.original_df)
        
        print(f"✓ Loaded {len(self.original_df):,} samples")
        print("\nOriginal class distribution:")
        for class_id in sorted(self.original_df['class'].unique()):
            count = len(self.original_df[self.original_df['class'] == class_id])
            pct = count / len(self.original_df) * 100
            print(f"  Class {class_id}: {count:,} ({pct:.2f}%)")
        
        # Load new 360K
        print(f"\nLoading new dataset: {AugmentationConfig.NEW_DATA}")
        self.new_df = pd.read_csv(AugmentationConfig.NEW_DATA, encoding='latin-1')
        self.stats['new_data_size'] = len(self.new_df)
        
        print(f"✓ Loaded {len(self.new_df):,} samples")
        print("\nNew data class distribution:")
        for class_id in sorted(self.new_df['class'].unique()):
            count = len(self.new_df[self.new_df['class'] == class_id])
            pct = count / len(self.new_df) * 100
            print(f"  Class {class_id}: {count:,} ({pct:.2f}%)")
    
    def filter_quality(self):
        """Filter high-quality samples."""
        print("\n" + "=" * 80)
        print("STEP 2: QUALITY FILTERING")
        print("=" * 80)
        
        print(f"\nApplying quality filters to {len(self.new_df):,} samples...")
        print(f"  - Minimum confidence: {AugmentationConfig.MIN_CONFIDENCE_THRESHOLD}")
        print(f"  - Text length: {AugmentationConfig.MIN_TEXT_LENGTH}-{AugmentationConfig.MAX_TEXT_LENGTH} chars")
        print(f"  - Lexicon validation: Enabled")
        
        # Calculate confidence
        self.new_df['confidence'] = self.new_df.apply(
            QualityFilter.calculate_confidence, axis=1
        )
        
        # Apply filters
        mask_quality = self.new_df.apply(
            lambda row: QualityFilter.is_high_quality(row), axis=1
        )
        
        print(f"\n✓ Quality filter: {mask_quality.sum():,}/{len(self.new_df):,} passed")
        
        # Apply lexicon validation
        print("  Validating with hate/offensive lexicons...")
        mask_lexicon = self.new_df.apply(
            QualityFilter.validate_with_lexicon, axis=1
        )
        
        print(f"✓ Lexicon validation: {mask_lexicon.sum():,}/{len(self.new_df):,} passed")
        
        # Combine filters
        mask_combined = mask_quality & mask_lexicon
        self.new_df = self.new_df[mask_combined].copy()
        
        self.stats['filtered_size'] = len(self.new_df)
        
        print(f"\n✓ TOTAL HIGH-QUALITY SAMPLES: {len(self.new_df):,}")
        print("\nFiltered class distribution:")
        for class_id in sorted(self.new_df['class'].unique()):
            count = len(self.new_df[self.new_df['class'] == class_id])
            pct = count / len(self.new_df) * 100
            print(f"  Class {class_id}: {count:,} ({pct:.2f}%)")
    
    def calculate_sampling_needs(self) -> Dict[int, int]:
        """Calculate how many samples needed per class."""
        print("\n" + "=" * 80)
        print("STEP 3: CALCULATING SAMPLING NEEDS")
        print("=" * 80)
        
        # Current distribution in original data
        original_counts = self.original_df['class'].value_counts().to_dict()
        
        # Target: 33,333 per class
        target = AugmentationConfig.TARGET_PER_CLASS
        
        sampling_needs = {}
        
        print(f"\nTarget: {target:,} samples per class")
        print(f"Total target: {AugmentationConfig.TOTAL_TARGET:,} samples\n")
        
        print(f"{'Class':<10} {'Current':<12} {'Target':<12} {'Need':<12} {'Available':<12}")
        print("-" * 60)
        
        for class_id in [0, 1, 2]:
            current = original_counts.get(class_id, 0)
            need = max(0, target - current)
            available = len(self.new_df[self.new_df['class'] == class_id])
            
            sampling_needs[class_id] = need
            
            status = "✓" if available >= need else "⚠"
            print(f"{class_id:<10} {current:<12,} {target:<12,} {need:<12,} {available:<12,} {status}")
        
        print()
        
        # Check if we have enough
        for class_id, need in sampling_needs.items():
            available = len(self.new_df[self.new_df['class'] == class_id])
            if available < need:
                print(f"⚠ WARNING: Class {class_id} needs {need:,} but only {available:,} available!")
        
        return sampling_needs
    
    def sample_and_merge(self, sampling_needs: Dict[int, int]):
        """Sample from new data and merge with original."""
        print("\n" + "=" * 80)
        print("STEP 4: STRATIFIED SAMPLING & MERGING")
        print("=" * 80)
        
        sampled_dfs = [self.original_df]  # Start with original
        
        for class_id in [0, 1, 2]:
            need = sampling_needs[class_id]
            
            if need == 0:
                print(f"\nClass {class_id}: No additional samples needed")
                continue
            
            print(f"\nClass {class_id}: Sampling {need:,} samples...")
            
            # Get candidates for this class
            candidates = self.new_df[self.new_df['class'] == class_id]
            
            # Apply diversity sampling
            if AugmentationConfig.ENSURE_DIVERSITY:
                sampled = DiversitySampler.stratified_sample(
                    candidates,
                    target_size=need,
                    max_similar=AugmentationConfig.MAX_SIMILAR_TEXTS
                )
            else:
                sampled = candidates.sample(n=min(need, len(candidates)), random_state=42)
            
            sampled_dfs.append(sampled)
            
            print(f"  ✓ Sampled {len(sampled):,} samples for class {class_id}")
        
        # Merge all
        print("\nMerging datasets...")
        self.augmented_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Remove duplicates
        initial_size = len(self.augmented_df)
        self.augmented_df = self.augmented_df.drop_duplicates(subset=['tweet'], keep='first')
        duplicates_removed = initial_size - len(self.augmented_df)
        
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed:,} duplicates")
        
        self.stats['final_size'] = len(self.augmented_df)
        
        print(f"\n✓ FINAL DATASET SIZE: {len(self.augmented_df):,}")
    
    def analyze_final_dataset(self):
        """Analyze and display final dataset statistics."""
        print("\n" + "=" * 80)
        print("STEP 5: FINAL DATASET ANALYSIS")
        print("=" * 80)
        
        print(f"\n[FINAL CLASS DISTRIBUTION]")
        print(f"{'Class':<10} {'Count':<12} {'Percentage':<12}")
        print("-" * 35)
        
        for class_id in sorted(self.augmented_df['class'].unique()):
            count = len(self.augmented_df[self.augmented_df['class'] == class_id])
            pct = count / len(self.augmented_df) * 100
            self.stats['class_distribution'][class_id] = count
            print(f"{class_id:<10} {count:<12,} {pct:<12.2f}%")
        
        print(f"\nTotal: {len(self.augmented_df):,}")
        
        # Text length statistics
        print(f"\n[TEXT LENGTH STATISTICS]")
        self.augmented_df['text_length'] = self.augmented_df['tweet'].str.len()
        
        for class_id in sorted(self.augmented_df['class'].unique()):
            class_texts = self.augmented_df[self.augmented_df['class'] == class_id]['text_length']
            print(f"Class {class_id}:")
            print(f"  Mean: {class_texts.mean():.1f} chars")
            print(f"  Median: {class_texts.median():.1f} chars")
            print(f"  Min: {class_texts.min()} chars")
            print(f"  Max: {class_texts.max()} chars")
        
        # Lexicon validation statistics
        print(f"\n[LEXICON VALIDATION]")
        for class_id in [0, 1, 2]:
            class_df = self.augmented_df[self.augmented_df['class'] == class_id]
            
            if class_id == 0:  # Hate
                validated = class_df['tweet'].apply(
                    lambda x: HateSpeechLexicon.check_hate_speech(x)[0]
                ).sum()
                print(f"Class {class_id} (Hate): {validated:,}/{len(class_df):,} contain hate keywords ({validated/len(class_df)*100:.1f}%)")
            
            elif class_id == 1:  # Offensive
                validated = class_df['tweet'].apply(
                    lambda x: HateSpeechLexicon.check_offensive(x)[0]
                ).sum()
                print(f"Class {class_id} (Offensive): {validated:,}/{len(class_df):,} contain offensive keywords ({validated/len(class_df)*100:.1f}%)")
            
            elif class_id == 2:  # Neither
                validated = class_df['tweet'].apply(
                    HateSpeechLexicon.check_neither
                ).sum()
                print(f"Class {class_id} (Neither): {validated:,}/{len(class_df):,} validated as neutral ({validated/len(class_df)*100:.1f}%)")
    
    def save_augmented_dataset(self):
        """Save final augmented dataset."""
        print("\n" + "=" * 80)
        print("STEP 6: SAVING AUGMENTED DATASET")
        print("=" * 80)
        
        output_path = AugmentationConfig.OUTPUT_DATA
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        self.augmented_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n✓ Saved augmented dataset to: {output_path}")
        print(f"  Total samples: {len(self.augmented_df):,}")
        print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Save statistics
        stats_path = output_path.parent / 'augmentation_stats.txt'
        with open(stats_path, 'w') as f:
            f.write("DATASET AUGMENTATION STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Original dataset size: {self.stats['original_size']:,}\n")
            f.write(f"New data pool size: {self.stats['new_data_size']:,}\n")
            f.write(f"After quality filtering: {self.stats['filtered_size']:,}\n")
            f.write(f"Final augmented size: {self.stats['final_size']:,}\n\n")
            f.write("Class distribution:\n")
            for class_id, count in sorted(self.stats['class_distribution'].items()):
                pct = count / self.stats['final_size'] * 100
                f.write(f"  Class {class_id}: {count:,} ({pct:.2f}%)\n")
        
        print(f"✓ Saved statistics to: {stats_path}")
    
    def run(self):
        """Run complete augmentation pipeline."""
        print("\n" + "=" * 80)
        print("DATASET AUGMENTATION PIPELINE")
        print("24K → 100K with Perfect Class Balance")
        print("=" * 80)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Filter quality
            self.filter_quality()
            
            # Step 3: Calculate needs
            sampling_needs = self.calculate_sampling_needs()
            
            # Step 4: Sample and merge
            self.sample_and_merge(sampling_needs)
            
            # Step 5: Analyze
            self.analyze_final_dataset()
            
            # Step 6: Save
            self.save_augmented_dataset()
            
            print("\n" + "=" * 80)
            print("✓ AUGMENTATION COMPLETE!")
            print("=" * 80)
            print(f"\nFinal dataset: {AugmentationConfig.OUTPUT_DATA}")
            print(f"Ready to train with perfect class balance!")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


# ==================== MAIN ====================

def main():
    """Main entry point."""
    
    augmentor = DatasetAugmentor()
    augmentor.run()


if __name__ == "__main__":
    main()