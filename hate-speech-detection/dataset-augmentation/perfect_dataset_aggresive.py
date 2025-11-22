"""
Dataset Augmentation Pipeline - Multi-Size Version
Create 100K, 150K, 200K datasets with perfect class balance using 360K LLM-annotated data

Based on proven pipeline that achieved 93.5% accuracy with 100K dataset

Strategy:
1. Load 360K dataset + existing 24K
2. Filter by quality (high confidence samples)
3. Apply hate/offensive lexicons for validation
4. Stratified sampling for diversity
5. Create three versions: 100K, 150K, 200K (all perfectly balanced)
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

class MultiSizeAugmentationConfig:
    # Paths
    ORIGINAL_DATA = Path('onelasttry/labeled_data.csv')  # Your 24K
    NEW_DATA = Path('onelasttry/final_dataset_reannotated.csv')   # 360K dataset
    
    # Output paths for all three sizes
    OUTPUT_100K = Path('onelasttry/augmented_100k.csv')
    OUTPUT_150K = Path('onelasttry/augmented_150k.csv')
    OUTPUT_200K = Path('onelasttry/datasetsFinal/augmented_200k.csv')
    
    # Target sizes - perfectly balanced
    TARGETS = {
        '100k': {
            'total': 100000,
            'per_class': 33333,  # Will be 33333, 33333, 33334
            'output_path': OUTPUT_100K,
            'max_similar': 5  # Strict diversity
        },
        '150k': {
            'total': 150000,
            'per_class': 50000,
            'output_path': OUTPUT_150K,
            'max_similar': 7  # Balanced diversity
        },
        '200k': {
            'total': 360000,
            'per_class': 120000,  # Will be 66666, 66667, 66667
            'output_path': OUTPUT_200K,
            'max_similar': 18  # Relaxed diversity
        }
    }
    
    # Quality thresholds (PROVEN - don't change!)
    MIN_CONFIDENCE_THRESHOLD = 0.7  # % of annotators agreeing
    MIN_TEXT_LENGTH = 10           # Minimum characters
    MAX_TEXT_LENGTH = 800          # Maximum characters
    
    # Sampling strategy
    ENSURE_DIVERSITY = True

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
        if len(text) < MultiSizeAugmentationConfig.MIN_TEXT_LENGTH:
            return False
        if len(text) > MultiSizeAugmentationConfig.MAX_TEXT_LENGTH:
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
        print(f"  Sampling {target_size:,} from {len(df):,} available...")
        
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
        
        print(f"  Selected {len(selected_indices):,} diverse samples")
        print(f"  Unique signatures: {len(signature_counter):,}")
        
        return shuffled_df.loc[selected_indices]


# ==================== MAIN MULTI-SIZE AUGMENTATION ====================

class MultiSizeDatasetAugmentor:
    """Create multiple dataset sizes with same proven quality filters."""
    
    def __init__(self):
        self.original_df = None
        self.filtered_pool = None
        self.datasets = {}  # Store all three versions
        
        self.global_stats = {
            'original_size': 0,
            'new_data_size': 0,
            'filtered_size': 0
        }
        
        self.dataset_stats = {}  # Stats for each size
    
    def load_data(self):
        """Load original and new datasets (once for all)."""
        print("=" * 80)
        print("STEP 1: LOADING DATASETS")
        print("=" * 80)
        
        # Load original 24K
        print(f"\nLoading original dataset: {MultiSizeAugmentationConfig.ORIGINAL_DATA}")
        self.original_df = pd.read_csv(
            MultiSizeAugmentationConfig.ORIGINAL_DATA, 
            encoding='latin-1'
        )
        self.global_stats['original_size'] = len(self.original_df)
        
        print(f"✓ Loaded {len(self.original_df):,} samples")
        print("\nOriginal class distribution:")
        for class_id in sorted(self.original_df['class'].unique()):
            count = len(self.original_df[self.original_df['class'] == class_id])
            pct = count / len(self.original_df) * 100
            print(f"  Class {class_id}: {count:,} ({pct:.2f}%)")
        
        # Load new 360K
        print(f"\nLoading new dataset: {MultiSizeAugmentationConfig.NEW_DATA}")
        new_df = pd.read_csv(
            MultiSizeAugmentationConfig.NEW_DATA,
            encoding='latin-1'
        )
        self.global_stats['new_data_size'] = len(new_df)
        
        print(f"✓ Loaded {len(new_df):,} samples")
        print("\nNew data class distribution:")
        for class_id in sorted(new_df['class'].unique()):
            count = len(new_df[new_df['class'] == class_id])
            pct = count / len(new_df) * 100
            print(f"  Class {class_id}: {count:,} ({pct:.2f}%)")
        
        return new_df
    
    def filter_quality(self, new_df):
        """Filter high-quality samples (once for all)."""
        print("\n" + "=" * 80)
        print("STEP 2: QUALITY FILTERING (Proven Pipeline)")
        print("=" * 80)
        
        print(f"\nApplying quality filters to {len(new_df):,} samples...")
        print(f"  - Minimum confidence: {MultiSizeAugmentationConfig.MIN_CONFIDENCE_THRESHOLD}")
        print(f"  - Text length: {MultiSizeAugmentationConfig.MIN_TEXT_LENGTH}-{MultiSizeAugmentationConfig.MAX_TEXT_LENGTH} chars")
        print(f"  - Lexicon validation: Enabled")
        
        # Calculate confidence
        new_df['confidence'] = new_df.apply(
            QualityFilter.calculate_confidence, axis=1
        )
        
        # Apply quality filter
        mask_quality = new_df.apply(
            lambda row: QualityFilter.is_high_quality(row), axis=1
        )
        
        print(f"\n✓ Quality filter: {mask_quality.sum():,}/{len(new_df):,} passed")
        
        # Apply lexicon validation
        print("  Validating with hate/offensive lexicons...")
        mask_lexicon = new_df.apply(
            QualityFilter.validate_with_lexicon, axis=1
        )
        
        print(f"✓ Lexicon validation: {mask_lexicon.sum():,}/{len(new_df):,} passed")
        
        # Combine filters
        mask_combined = mask_quality & mask_lexicon
        self.filtered_pool = new_df[mask_combined].copy()
        
        self.global_stats['filtered_size'] = len(self.filtered_pool)
        
        print(f"\n✓ TOTAL HIGH-QUALITY POOL: {len(self.filtered_pool):,}")
        print("\nFiltered pool distribution:")
        for class_id in sorted(self.filtered_pool['class'].unique()):
            count = len(self.filtered_pool[self.filtered_pool['class'] == class_id])
            pct = count / len(self.filtered_pool) * 100
            print(f"  Class {class_id}: {count:,} ({pct:.2f}%)")
    
    def create_single_dataset(
        self,
        size_name: str,
        config: Dict
    ) -> pd.DataFrame:
        """Create a single dataset of specified size."""
        
        print("\n" + "=" * 80)
        print(f"CREATING {size_name.upper()} DATASET")
        print("=" * 80)
        
        total_target = config['total']
        per_class = config['per_class']
        max_similar = config['max_similar']
        
        # Calculate needs per class
        original_counts = self.original_df['class'].value_counts().to_dict()
        
        print(f"\nTarget: {per_class:,} per class (~{total_target:,} total)")
        print(f"Diversity: Max {max_similar} similar texts\n")
        
        print(f"{'Class':<10} {'Current':<12} {'Target':<12} {'Need':<12} {'Available':<12}")
        print("-" * 60)
        
        sampled_dfs = [self.original_df]  # Start with original
        sampling_needs = {}
        
        for class_id in [0, 1, 2]:
            current = original_counts.get(class_id, 0)
            need = max(0, per_class - current)
            available = len(self.filtered_pool[self.filtered_pool['class'] == class_id])
            
            sampling_needs[class_id] = need
            
            status = "✓" if available >= need else "⚠"
            print(f"{class_id:<10} {current:<12,} {per_class:<12,} {need:<12,} {available:<12,} {status}")
            
            if need > 0:
                print(f"\nClass {class_id}: Sampling {need:,} samples...")
                
                # Get candidates for this class
                candidates = self.filtered_pool[self.filtered_pool['class'] == class_id]
                
                # Apply diversity sampling
                if MultiSizeAugmentationConfig.ENSURE_DIVERSITY:
                    sampled = DiversitySampler.stratified_sample(
                        candidates,
                        target_size=need,
                        max_similar=max_similar
                    )
                else:
                    sampled = candidates.sample(
                        n=min(need, len(candidates)), 
                        random_state=42
                    )
                
                sampled_dfs.append(sampled)
                
                print(f"  ✓ Sampled {len(sampled):,} samples for class {class_id}")
        
        # Merge all
        print("\nMerging datasets...")
        dataset = pd.concat(sampled_dfs, ignore_index=True)
        
        # Remove duplicates
        initial_size = len(dataset)
        dataset = dataset.drop_duplicates(subset=['tweet'], keep='first')
        duplicates_removed = initial_size - len(dataset)
        
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed:,} duplicates")
        
        print(f"\n✓ FINAL {size_name.upper()} SIZE: {len(dataset):,}")
        
        # Store stats
        self.dataset_stats[size_name] = {
            'total': len(dataset),
            'class_distribution': {},
            'sampling_needs': sampling_needs
        }
        
        for class_id in [0, 1, 2]:
            count = len(dataset[dataset['class'] == class_id])
            self.dataset_stats[size_name]['class_distribution'][class_id] = count
        
        return dataset
    
    def analyze_all_datasets(self):
        """Analyze and compare all created datasets."""
        print("\n" + "=" * 80)
        print("STEP 4: COMPARATIVE ANALYSIS")
        print("=" * 80)
        
        print("\n[DATASET SIZE COMPARISON]")
        print(f"{'Dataset':<12} {'Total':<12} {'Hate':<12} {'Offensive':<12} {'Neither':<12}")
        print("-" * 60)
        
        for size_name in ['100k', '150k', '200k']:
            if size_name in self.datasets:
                df = self.datasets[size_name]
                total = len(df)
                hate = len(df[df['class'] == 0])
                offensive = len(df[df['class'] == 1])
                neither = len(df[df['class'] == 2])
                
                print(f"{size_name:<12} {total:<12,} {hate:<12,} {offensive:<12,} {neither:<12,}")
        
        # Balance scores
        print("\n[CLASS BALANCE ANALYSIS]")
        print(f"{'Dataset':<12} {'Hate %':<12} {'Offensive %':<12} {'Neither %':<12} {'Balance*':<12}")
        print("-" * 68)
        
        for size_name in ['100k', '150k', '200k']:
            if size_name in self.datasets:
                df = self.datasets[size_name]
                total = len(df)
                
                hate_pct = len(df[df['class'] == 0]) / total * 100
                off_pct = len(df[df['class'] == 1]) / total * 100
                nei_pct = len(df[df['class'] == 2]) / total * 100
                
                # Balance score: 100 - total deviation from 33.33%
                balance = 100 - (
                    abs(hate_pct - 33.33) + 
                    abs(off_pct - 33.33) + 
                    abs(nei_pct - 33.33)
                )
                
                print(f"{size_name:<12} {hate_pct:<12.2f} {off_pct:<12.2f} {nei_pct:<12.2f} {balance:<12.2f}")
        
        print("\n*Balance score: 100 = perfect 33.33% each class")
        
        # Lexicon validation comparison
        print("\n[LEXICON VALIDATION COMPARISON]")
        print(f"{'Dataset':<12} {'Hate Valid':<15} {'Offensive Valid':<20} {'Neither Valid':<15}")
        print("-" * 68)
        
        for size_name in ['100k', '150k', '200k']:
            if size_name in self.datasets:
                df = self.datasets[size_name]
                
                # Hate validation
                hate_df = df[df['class'] == 0]
                hate_valid = hate_df['tweet'].apply(
                    lambda x: HateSpeechLexicon.check_hate_speech(x)[0]
                ).sum()
                hate_pct = hate_valid / len(hate_df) * 100 if len(hate_df) > 0 else 0
                
                # Offensive validation
                off_df = df[df['class'] == 1]
                off_valid = off_df['tweet'].apply(
                    lambda x: HateSpeechLexicon.check_offensive(x)[0]
                ).sum()
                off_pct = off_valid / len(off_df) * 100 if len(off_df) > 0 else 0
                
                # Neither validation
                nei_df = df[df['class'] == 2]
                nei_valid = nei_df['tweet'].apply(
                    HateSpeechLexicon.check_neither
                ).sum()
                nei_pct = nei_valid / len(nei_df) * 100 if len(nei_df) > 0 else 0
                
                print(f"{size_name:<12} {hate_pct:<15.1f}% {off_pct:<20.1f}% {nei_pct:<15.1f}%")
    
    def save_all_datasets(self):
        """Save all created datasets."""
        print("\n" + "=" * 80)
        print("STEP 5: SAVING ALL DATASETS")
        print("=" * 80)
        
        for size_name in ['100k', '150k', '200k']:
            if size_name in self.datasets:
                config = MultiSizeAugmentationConfig.TARGETS[size_name]
                output_path = config['output_path']
                
                # Ensure directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save dataset
                df = self.datasets[size_name]
                df.to_csv(output_path, index=False, encoding='utf-8')
                
                size_mb = output_path.stat().st_size / 1024 / 1024
                
                print(f"\n✓ {size_name.upper():>5}: {output_path}")
                print(f"         {len(df):>7,} samples")
                print(f"         {size_mb:>7.2f} MB")
        
        # Save combined statistics
        stats_path = MultiSizeAugmentationConfig.OUTPUT_100K.parent / 'augmentation_stats_all.txt'
        
        with open(stats_path, 'w') as f:
            f.write("MULTI-SIZE DATASET AUGMENTATION STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("GLOBAL STATISTICS:\n")
            f.write(f"  Original dataset: {self.global_stats['original_size']:,}\n")
            f.write(f"  New data pool: {self.global_stats['new_data_size']:,}\n")
            f.write(f"  After filtering: {self.global_stats['filtered_size']:,}\n\n")
            
            for size_name in ['100k', '150k', '200k']:
                if size_name in self.dataset_stats:
                    stats = self.dataset_stats[size_name]
                    f.write(f"\n{size_name.upper()} DATASET:\n")
                    f.write(f"  Total: {stats['total']:,}\n")
                    f.write(f"  Class distribution:\n")
                    for class_id in sorted(stats['class_distribution'].keys()):
                        count = stats['class_distribution'][class_id]
                        pct = count / stats['total'] * 100
                        f.write(f"    Class {class_id}: {count:,} ({pct:.2f}%)\n")
        
        print(f"\n✓ Statistics saved to: {stats_path}")
    
    def run(self):
        """Run complete multi-size augmentation pipeline."""
        print("\n" + "=" * 80)
        print("MULTI-SIZE DATASET AUGMENTATION PIPELINE")
        print("Creating: 100K, 150K, 200K (All Perfectly Balanced)")
        print("Based on proven pipeline: 93.5% accuracy")
        print("=" * 80)
        
        try:
            # Step 1: Load data (once)
            new_df = self.load_data()
            
            # Step 2: Filter quality (once)
            self.filter_quality(new_df)
            
            # Check if we have enough for largest dataset
            print("\n" + "=" * 80)
            print("STEP 3: CREATING ALL DATASET SIZES")
            print("=" * 80)
            
            max_per_class = 66666  # For 200K
            sufficient = True
            
            for class_id in [0, 1, 2]:
                available = len(self.filtered_pool[self.filtered_pool['class'] == class_id])
                if available < max_per_class:
                    print(f"⚠ WARNING: Class {class_id} needs {max_per_class:,} for 200K but only {available:,} available")
                    sufficient = False
            
            if not sufficient:
                print("\nWill create datasets up to available data limits")
            
            # Step 3: Create all three sizes
            for size_name in ['100k', '150k', '200k']:
                config = MultiSizeAugmentationConfig.TARGETS[size_name]
                self.datasets[size_name] = self.create_single_dataset(size_name, config)
            
            # Step 4: Analyze all
            self.analyze_all_datasets()
            
            # Step 5: Save all
            self.save_all_datasets()
            
            print("\n" + "=" * 80)
            print("✓ ALL DATASETS CREATED SUCCESSFULLY!")
            print("=" * 80)
            
            print("\n[OUTPUT FILES]")
            for size_name in ['100k', '150k', '200k']:
                config = MultiSizeAugmentationConfig.TARGETS[size_name]
                print(f"  {size_name}: {config['output_path']}")
            
            print("\n[NEXT STEPS]")
            print("1. Train CNN on 100K (baseline - proven 93.5%)")
            print("2. Train CNN on 150K (test improvement)")
            print("3. Train CNN on 200K (maximum performance)")
            print("4. Compare results:")
            print("   python main_train_enhanced.py --phase5  # Use different datasets")
            print("5. Train BERT-Large on best performing size")
            
            print("\n[EXPECTED RESULTS]")
            print("100K: 93.5% (proven)")
            print("150K: 93.8-94.0% (predicted)")
            print("200K: 94.0-94.5% (predicted)")
            print("\nBERT-Large on 200K: 95-96% (predicted)")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


# ==================== MAIN ====================

def main():
    """Main entry point."""
    
    print("\n" + "=" * 80)
    print("MULTI-SIZE AUGMENTATION")
    print("Extending proven 93.5% accuracy pipeline to 150K and 200K")
    print("=" * 80)
    
    augmentor = MultiSizeDatasetAugmentor()
    augmentor.run()


if __name__ == "__main__":
    main()