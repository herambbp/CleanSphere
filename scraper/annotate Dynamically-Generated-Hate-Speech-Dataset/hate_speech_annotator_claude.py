#!/usr/bin/env python3
"""
Hate Speech Dataset Annotation System - CLAUDE VERSION
Using Anthropic Claude API (no content moderation issues for research)
"""

import pandas as pd
import json
import time
from anthropic import Anthropic
from typing import List, Dict
import os
from datetime import datetime
from tqdm import tqdm
import sys

class HateSpeechAnnotatorClaude:
    def __init__(self, api_key: str, batch_size: int = 15, checkpoint_interval: int = 1000):
        """
        Initialize the annotator with Claude
        
        Args:
            api_key: Anthropic API key
            batch_size: Number of texts to process per API call
            checkpoint_interval: Save progress every N rows
        """
        self.client = Anthropic(api_key=api_key)
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.model = "claude-sonnet-4-5-20250929"
        
        self.stats = {
            'hate_speech': 0,
            'offensive_language': 0,
            'neither': 0,
            'total_processed': 0,
            'api_calls': 0,
            'errors': 0
        }
        
    def create_annotation_prompt(self, texts_with_context: List[Dict]) -> str:
        """Create a batch prompt for multiple texts with context"""
        prompt = """You are a hate speech detection expert analyzing content for academic research purposes. Classify each text into exactly one category:

**Categories:**
0 = hate_speech - Content that expresses hate, violence, or severe prejudice against individuals or groups based on protected characteristics (race, ethnicity, religion, gender, sexual orientation, disability, etc.)
1 = offensive_language - Content that is vulgar, profane, or insulting but doesn't constitute hate speech
2 = neither - Content that is neutral or doesn't fit the above categories

**Instructions:**
- Consider the context provided for each text
- Be consistent and objective
- This is for academic research on hate speech detection
- Return ONLY a JSON array with your classifications

**Texts to classify:**

"""
        
        for i, item in enumerate(texts_with_context):
            prompt += f"\n--- Text {i+1} ---\n"
            prompt += f"ID: {item['id']}\n"
            prompt += f"Text: {item['text']}\n"
            if item.get('context'):
                prompt += f"Context: {item['context']}\n"
        
        prompt += """

**Response Format (JSON only):**
[
  {"text_id": 1, "category": 0, "confidence": 0.95},
  {"text_id": 2, "category": 1, "confidence": 0.87},
  ...
]

Return ONLY the JSON array, no other text."""
        
        return prompt
    
    def annotate_batch(self, texts_with_context: List[Dict]) -> List[Dict]:
        """Annotate a batch of texts using Claude API"""
        try:
            prompt = self.create_annotation_prompt(texts_with_context)
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                system="You are a hate speech classification expert conducting academic research. Always respond with valid JSON only.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            self.stats['api_calls'] += 1
            
            # Parse response
            response_text = message.content[0].text.strip()
            
            # Extract JSON from response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            annotations = json.loads(response_text)
            
            return annotations
            
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parsing error: {e}")
            self.stats['errors'] += 1
            return [{"text_id": i+1, "category": 2, "confidence": 0.0} for i in range(len(texts_with_context))]
            
        except Exception as e:
            print(f"\n⚠️  API error: {e}")
            self.stats['errors'] += 1
            return [{"text_id": i+1, "category": 2, "confidence": 0.0} for i in range(len(texts_with_context))]
    
    def prepare_context_string(self, row: pd.Series, exclude_cols: List[str]) -> str:
        """Prepare context string from all columns except text"""
        context_parts = []
        for col in row.index:
            if col not in exclude_cols and pd.notna(row[col]) and row[col] != '':
                context_parts.append(f"{col}={row[col]}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def process_dataset(self, input_csv: str, output_csv: str, sample_size: int = None, 
                       resume_from: str = None) -> pd.DataFrame:
        """Process the entire dataset with batch annotation"""
        print("🚀 Starting Hate Speech Annotation System (Claude)")
        print(f"📊 Model: {self.model}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"💾 Checkpoint interval: {self.checkpoint_interval} rows")
        print("-" * 70)
        
        # Load dataset
        print("\n📂 Loading dataset...")
        df = pd.read_csv(input_csv)
        original_size = len(df)
        
        if sample_size:
            df = df.head(sample_size)
            print(f"🧪 Sample mode: Processing {len(df)} rows (out of {original_size})")
        else:
            print(f"📊 Total rows: {len(df)}")
        
        # Resume from checkpoint if provided
        start_idx = 0
        if resume_from and os.path.exists(resume_from):
            print(f"\n♻️  Resuming from checkpoint: {resume_from}")
            checkpoint_df = pd.read_csv(resume_from)
            start_idx = len(checkpoint_df)
            df.iloc[:start_idx] = checkpoint_df
            print(f"✓ Resumed from row {start_idx}")
        
        # Add output columns
        if 'count' not in df.columns:
            df.insert(0, 'count', range(len(df)))
        if 'hate_speech' not in df.columns:
            df['hate_speech'] = 0
        if 'offensive_language' not in df.columns:
            df['offensive_language'] = 0
        if 'neither' not in df.columns:
            df['neither'] = 0
        if 'class' not in df.columns:
            df['class'] = -1
        if 'tweet' not in df.columns:
            df['tweet'] = df['text'] if 'text' in df.columns else ''
        
        # Identify text column
        text_col = 'text' if 'text' in df.columns else 'tweet'
        
        # Batch processing
        print("\n🔄 Starting batch processing...")
        print(f"⏱️  Estimated API calls: {(len(df) - start_idx) // self.batch_size + 1}")
        
        checkpoint_path = output_csv.replace('.csv', '_checkpoint.csv')
        
        with tqdm(total=len(df) - start_idx, desc="Processing", unit="rows") as pbar:
            for i in range(start_idx, len(df), self.batch_size):
                batch_end = min(i + self.batch_size, len(df))
                batch = df.iloc[i:batch_end]
                
                # Prepare batch for API
                texts_with_context = []
                for idx, row in batch.iterrows():
                    context = self.prepare_context_string(
                        row, 
                        exclude_cols=['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet', text_col]
                    )
                    texts_with_context.append({
                        'id': idx - i + 1,
                        'text': str(row[text_col]),
                        'context': context
                    })
                
                # Get annotations
                annotations = self.annotate_batch(texts_with_context)
                
                # Apply annotations
                for j, annotation in enumerate(annotations):
                    row_idx = i + j
                    if row_idx >= len(df):
                        break
                    
                    category = annotation.get('category', 2)
                    
                    df.at[row_idx, 'hate_speech'] = 1 if category == 0 else 0
                    df.at[row_idx, 'offensive_language'] = 1 if category == 1 else 0
                    df.at[row_idx, 'neither'] = 1 if category == 2 else 0
                    df.at[row_idx, 'class'] = category
                    
                    if category == 0:
                        self.stats['hate_speech'] += 1
                    elif category == 1:
                        self.stats['offensive_language'] += 1
                    else:
                        self.stats['neither'] += 1
                    
                    self.stats['total_processed'] += 1
                
                pbar.set_postfix({
                    'Hate': self.stats['hate_speech'],
                    'Offensive': self.stats['offensive_language'],
                    'Neither': self.stats['neither'],
                    'API calls': self.stats['api_calls']
                })
                pbar.update(len(annotations))
                
                # Checkpoint save
                if (i + self.batch_size) % self.checkpoint_interval < self.batch_size:
                    df.to_csv(checkpoint_path, index=False)
                    print(f"\n💾 Checkpoint saved at row {i + self.batch_size}")
                    self.print_live_stats()
                
                # Rate limiting
                time.sleep(0.5)
        
        # Save final output
        print(f"\n💾 Saving final output to {output_csv}...")
        
        output_cols = ['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']
        remaining_cols = [col for col in df.columns if col not in output_cols]
        final_cols = output_cols + remaining_cols
        
        df[final_cols].to_csv(output_csv, index=False)
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        print("✅ Processing complete!")
        return df
    
    def print_live_stats(self):
        """Print current annotation statistics"""
        total = self.stats['total_processed']
        if total == 0:
            return
        
        print("\n" + "=" * 70)
        print("📊 LIVE STATISTICS")
        print("=" * 70)
        print(f"Total Processed:      {total:,}")
        print(f"Hate Speech:          {self.stats['hate_speech']:,} ({self.stats['hate_speech']/total*100:.1f}%)")
        print(f"Offensive Language:   {self.stats['offensive_language']:,} ({self.stats['offensive_language']/total*100:.1f}%)")
        print(f"Neither:              {self.stats['neither']:,} ({self.stats['neither']/total*100:.1f}%)")
        print(f"API Calls Made:       {self.stats['api_calls']:,}")
        print(f"Errors:               {self.stats['errors']:,}")
        print("=" * 70 + "\n")
    
    def print_final_summary(self, df: pd.DataFrame):
        """Print final comprehensive summary"""
        print("\n" + "=" * 70)
        print("🎉 FINAL SUMMARY")
        print("=" * 70)
        
        total = len(df)
        hate_count = (df['class'] == 0).sum()
        offensive_count = (df['class'] == 1).sum()
        neither_count = (df['class'] == 2).sum()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total Rows:           {total:,}")
        print(f"  Hate Speech:          {hate_count:,} ({hate_count/total*100:.2f}%)")
        print(f"  Offensive Language:   {offensive_count:,} ({offensive_count/total*100:.2f}%)")
        print(f"  Neither:              {neither_count:,} ({neither_count/total*100:.2f}%)")
        
        print(f"\n🔧 Processing Statistics:")
        print(f"  API Calls Made:       {self.stats['api_calls']:,}")
        print(f"  Errors Encountered:   {self.stats['errors']:,}")
        
        print(f"\n💰 Cost Estimation:")
        print(f"  Model Used:           {self.model}")
        print(f"  Cost:                 ~$15-30 for 121K rows")
        
        print("=" * 70)


def main():
    """Main execution function"""
    
    API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    if not API_KEY:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        print("Get your API key from: https://console.anthropic.com")
        sys.exit(1)
    
    INPUT_CSV = "input_dataset.csv"
    OUTPUT_CSV = "annotated_dataset_claude.csv"
    SAMPLE_SIZE = 100  # Set to None for full dataset
    
    annotator = HateSpeechAnnotatorClaude(
        api_key=API_KEY,
        batch_size=15,
        checkpoint_interval=1000
    )
    
    try:
        df = annotator.process_dataset(
            input_csv=INPUT_CSV,
            output_csv=OUTPUT_CSV,
            sample_size=SAMPLE_SIZE
        )
        
        annotator.print_final_summary(df)
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{INPUT_CSV}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()