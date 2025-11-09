"""
Download Measuring Hate Speech dataset with multiple fallback methods
"""

import pandas as pd
import requests
from pathlib import Path
import sys
import zipfile
import io

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import logger, print_section_header


class DatasetDownloader:
    """Download Measuring Hate Speech dataset with fallbacks"""
    
    def __init__(self):
        self.output_dir = PROJECT_ROOT / "data" / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "measuring_hate_speech_raw.csv"
    
    def method_1_huggingface_direct(self):
        """Method 1: Direct download from HuggingFace URLs"""
        print_section_header("METHOD 1: DIRECT HUGGINGFACE DOWNLOAD")
        
        try:
            # HuggingFace dataset files are hosted on their CDN
            url = "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/measurements_with_text.csv"
            
            logger.info(f"Downloading from: {url}")
            logger.info("This may take a few minutes (large file ~150MB)...")
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Save with progress
            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"File size: {total_size / 1024 / 1024:.1f} MB")
            
            with open(self.output_file, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            logger.info(f"[SUCCESS] Downloaded to: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Method 1 failed: {e}")
            return False
    
    def method_2_huggingface_api(self):
        """Method 2: Use HuggingFace datasets library with retry"""
        print_section_header("METHOD 2: HUGGINGFACE DATASETS API")
        
        try:
            # Fix PyArrow issues
            logger.info("Attempting to fix PyArrow compatibility...")
            import importlib
            import datasets
            
            # Reload to get fresh version
            importlib.reload(datasets)
            
            logger.info("Loading dataset...")
            from datasets import load_dataset
            
            # Try loading with different configurations
            dataset = load_dataset(
                "ucberkeley-dlab/measuring-hate-speech",
                split="train",
                download_mode="force_redownload"  # Force fresh download
            )
            
            logger.info("Converting to pandas...")
            df = dataset.to_pandas()
            
            logger.info(f"Saving to CSV...")
            df.to_csv(self.output_file, index=False)
            
            logger.info(f"[SUCCESS] Saved {len(df):,} rows to: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Method 2 failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def method_3_zenodo(self):
        """Method 3: Download from Zenodo (original source)"""
        print_section_header("METHOD 3: ZENODO DOWNLOAD")
        
        try:
            # Original dataset location
            url = "https://zenodo.org/record/5841333/files/measuring_hate_speech.csv?download=1"
            
            logger.info(f"Downloading from Zenodo: {url}")
            logger.info("This is the original dataset source...")
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(self.output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"[SUCCESS] Downloaded to: {self.output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Method 3 failed: {e}")
            return False
    
    def method_4_manual_instructions(self):
        """Method 4: Provide manual download instructions"""
        print_section_header("METHOD 4: MANUAL DOWNLOAD INSTRUCTIONS")
        
        logger.info("All automatic methods failed. Please download manually:")
        logger.info("")
        logger.info("OPTION A - HuggingFace:")
        logger.info("1. Go to: https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech")
        logger.info("2. Click 'Files and versions' tab")
        logger.info("3. Download 'measurements_with_text.csv'")
        logger.info(f"4. Save it as: {self.output_file}")
        logger.info("")
        logger.info("OPTION B - Zenodo (Original Source):")
        logger.info("1. Go to: https://zenodo.org/record/5841333")
        logger.info("2. Download 'measuring_hate_speech.csv'")
        logger.info(f"3. Save it as: {self.output_file}")
        logger.info("")
        logger.info("OPTION C - Use Sample Data:")
        logger.info("1. I can create a sample dataset for testing")
        logger.info("   (Run with --create-sample flag)")
        
        return False
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing"""
        print_section_header("CREATING SAMPLE DATASET")
        
        logger.info("Creating sample dataset with realistic examples...")
        
        # Sample data with realistic hate speech scores
        sample_data = {
            'comment_id': range(1000),
            'text': [
                # Hate speech examples (score > 0.5)
                "I hate all those people, they should die",
                "These subhuman scum need to be eliminated",
                "All [group] are vermin and parasites",
                "We should kill all [group] members",
                "They're not even human, just animals",
                # Offensive examples (0 < score <= 0.5)
                "You're such an idiot and a loser",
                "Shut up you stupid moron",
                "What a dumbass comment",
                "This is absolutely ridiculous",
                "Stop being such a jerk",
                # Neither examples (score <= 0)
                "I disagree with this policy strongly",
                "This is not a good approach in my opinion",
                "I think we need better solutions",
                "The data shows different results",
                "We should consider alternatives",
            ] * 67,  # Repeat to get ~1000 samples
            'annotator_id': [f"A{i%50}" for i in range(1000)],
        }
        
        df = pd.DataFrame(sample_data)
        df = df.iloc[:1000]  # Trim to exactly 1000
        
        # Generate hate_speech_scores based on content
        def generate_score(text):
            text_lower = text.lower()
            
            # Hate speech indicators
            if any(word in text_lower for word in ['kill', 'die', 'eliminate', 'subhuman', 'vermin']):
                return np.random.uniform(0.5, 1.0)  # Hate speech
            
            # Offensive indicators
            elif any(word in text_lower for word in ['idiot', 'stupid', 'moron', 'dumbass', 'jerk']):
                return np.random.uniform(0.0, 0.5)  # Offensive
            
            # Neither
            else:
                return np.random.uniform(-1.0, 0.0)  # Neither
        
        import numpy as np
        df['hate_speech_score'] = df['text'].apply(generate_score)
        
        # Add other required columns
        df['sentiment'] = np.random.randint(1, 6, size=len(df))
        df['respect'] = np.random.randint(1, 6, size=len(df))
        df['insult'] = np.random.randint(1, 6, size=len(df))
        df['humiliate'] = np.random.randint(1, 6, size=len(df))
        df['status'] = np.random.randint(1, 6, size=len(df))
        df['dehumanize'] = np.random.randint(1, 6, size=len(df))
        df['violence'] = np.random.randint(1, 6, size=len(df))
        df['genocide'] = np.random.randint(1, 6, size=len(df))
        df['attack_defend'] = np.random.randint(1, 6, size=len(df))
        df['hatespeech'] = np.random.randint(1, 6, size=len(df))
        
        # Save
        sample_file = self.output_dir / "measuring_hate_speech_sample.csv"
        df.to_csv(sample_file, index=False)
        
        logger.info(f"[SUCCESS] Created sample dataset: {sample_file}")
        logger.info(f"Samples: {len(df):,}")
        logger.info("Note: This is a SAMPLE for testing. Download the real dataset for production!")
        
        return sample_file
    
    def download(self, create_sample=False):
        """Try all download methods in sequence"""
        
        if create_sample:
            return self.create_sample_dataset()
        
        if self.output_file.exists():
            logger.info(f"[EXISTS] Dataset already downloaded: {self.output_file}")
            return self.output_file
        
        methods = [
            ("Direct HuggingFace URL", self.method_1_huggingface_direct),
            ("HuggingFace Datasets API", self.method_2_huggingface_api),
            ("Zenodo (Original Source)", self.method_3_zenodo),
        ]
        
        for method_name, method_func in methods:
            logger.info(f"\n[TRYING] {method_name}...")
            try:
                if method_func():
                    logger.info(f"[SUCCESS] Dataset downloaded using: {method_name}")
                    return self.output_file
            except Exception as e:
                logger.error(f"[FAILED] {method_name}: {e}")
                continue
        
        # All methods failed
        self.method_4_manual_instructions()
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Measuring Hate Speech dataset')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample dataset for testing')
    parser.add_argument('--force', action='store_true',
                       help='Force redownload even if file exists')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.force and downloader.output_file.exists():
        logger.info(f"Removing existing file: {downloader.output_file}")
        downloader.output_file.unlink()
    
    result = downloader.download(create_sample=args.create_sample)
    
    if result:
        # Verify the downloaded file
        print_section_header("VERIFYING DOWNLOADED FILE")
        try:
            df = pd.read_csv(result, nrows=5)
            logger.info(f"[OK] File is valid CSV")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"\nFirst row sample:")
            logger.info(df.iloc[0].to_string())
            
            # Get full count
            df_full = pd.read_csv(result)
            logger.info(f"\n[OK] Total rows: {len(df_full):,}")
            
            logger.info("\n[NEXT STEP] Run conversion script:")
            logger.info(f"python scripts/convert_measuring_hate_speech.py --input {result} --strategy severity_aware --aggregate")
            
        except Exception as e:
            logger.error(f"Error verifying file: {e}")
    else:
        logger.error("\n[FAILED] Could not download dataset")
        logger.error("Please follow manual instructions above")
        sys.exit(1)


if __name__ == "__main__":
    main()