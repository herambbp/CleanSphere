"""
Balanced Comment Scraper with Live Annotation Display
Option E: Hybrid Approach with targeted scraping + iterative tracking

Features:
- Targets specific sources for each class
- Shows live annotations in real-time
- User can stop anytime if annotations look wrong
- Tracks class balance
- Saves progress incrementally
"""

import praw
import csv
import time
import os
from openai import OpenAI
from datetime import datetime
import json
import threading
import sys
from collections import defaultdict
from pathlib import Path

from pathlib import Path

def load_config(config_path="config.json"):
    """
    Load configuration from config.json file
    
    Args:
        config_path: Path to config file (default: config.json in current directory)
    
    Returns:
        dict: Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        print("📝 Creating template config.json file...")
        
        template_config = {
            "api_keys": {
                "openai_api_key": "your-openai-api-key-here",
                "reddit_client_id": "your-reddit-client-id-here",
                "reddit_client_secret": "your-reddit-client-secret-here"
            },
            "scraper_settings": {
                "target_per_class": 3333,
                "output_file": "balanced_dataset_10k.csv",
                "gpt_model": "gpt-4o-mini",
                "reddit_user_agent": "BalancedScraper/1.0"
            },
            "subreddit_sources": {
                "hate_speech": ["unpopularopinion", "TrueOffMyChest", "rant", "AmItheAsshole"],
                "offensive_language": ["gaming", "sports", "nba", "soccer", "RoastMe", "relationship_advice", "PublicFreakout"],
                "neither": ["science", "technology", "explainlikeimfive", "todayilearned", "askscience", "Documentaries", "books"]
            },
            "advanced_settings": {
                "rate_limit_delay": 1.5,
                "batch_size": 20,
                "max_retries": 3,
                "save_progress_every": 10,
                "gpt_temperature": 0.3
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        print(f"✅ Created template config file: {config_path}")
        print("📝 Please edit config.json and add your API keys")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required keys
        if config['api_keys']['openai_api_key'] == "your-openai-api-key-here":
            print("❌ Error: Please edit config.json and add your OpenAI API key")
            sys.exit(1)
        
        return config
    
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ Error: Missing required key in config: {e}")
        sys.exit(1)

class BalancedCommentScraper:
    def __init__(self, config):
        """
        Initialize the balanced scraper with config from config.json
        
        Args:
            config: Configuration dictionary loaded from config.json
        """
        # API Keys
        openai_api_key = config['api_keys']['openai_api_key']
        reddit_client_id = config['api_keys'].get('reddit_client_id')
        reddit_client_secret = config['api_keys'].get('reddit_client_secret')
        
        # Settings
        self.target_per_class = config['scraper_settings']['target_per_class']
        self.gpt_model = config['scraper_settings']['gpt_model']
        self.output_file = config['scraper_settings']['output_file']
        self.reddit_user_agent = config['scraper_settings']['reddit_user_agent']
        
        # Advanced settings
        self.rate_limit_delay = config['advanced_settings']['rate_limit_delay']
        self.batch_size = config['advanced_settings']['batch_size']
        self.gpt_temperature = config['advanced_settings']['gpt_temperature']
        
        # Sources
        self.hate_speech_sources = config['subreddit_sources']['hate_speech']
        self.offensive_language_sources = config['subreddit_sources']['offensive_language']
        self.neither_sources = config['subreddit_sources']['neither']
        
        self.client = OpenAI(api_key=openai_api_key)
        self.total_target = self.target_per_class * 3
        
        # Track counts for each class
        self.counts = {0: 0, 1: 0, 2: 0}
        self.total_scraped = 0
        self.total_annotated = 0
        self.total_kept = 0
        
        # Initialize Reddit
        if reddit_client_id and reddit_client_secret and reddit_client_id != "your-reddit-client-id-here":
            self.reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
        else:
            # Read-only mode
            self.reddit = None
        
        # Define targeted subreddits for each class (loaded from config)
        # self.hate_speech_sources, self.offensive_language_sources, self.neither_sources
        # already set above from config
        
        # Stop flag for user interruption
        self.should_stop = False
        
    def print_header(self):
        """Print a nice header"""
        print("\n" + "="*80)
        print("🎯 BALANCED COMMENT SCRAPER - LIVE ANNOTATION MODE")
        print("="*80)
        print(f"📊 Target: {self.target_per_class:,} comments per class ({self.total_target:,} total)")
        print("⚠️  Press Ctrl+C at any time to stop if annotations look wrong")
        print("="*80 + "\n")
    
    def print_progress(self):
        """Print current progress with color-coded bars"""
        print("\n" + "─"*80)
        print("📈 CURRENT PROGRESS:")
        print("─"*80)
        
        for class_id, class_name in [(0, "Hate Speech"), (1, "Offensive"), (2, "Neither")]:
            count = self.counts[class_id]
            percentage = (count / self.target_per_class) * 100
            bar_length = 40
            filled = int((count / self.target_per_class) * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"{class_name:15} [{bar}] {count:4}/{self.target_per_class} ({percentage:.1f}%)")
        
        print("─"*80)
        print(f"📊 Total Kept: {self.total_kept:,} | Scraped: {self.total_scraped:,} | "
              f"Efficiency: {(self.total_kept/max(1,self.total_scraped)*100):.1f}%")
        print("─"*80 + "\n")
    
    def annotate_comment(self, comment_text):
        """
        Annotate a comment with live display
        Uses model and temperature from config
        """
        prompt = f"""You are annotating social media comments for a hate speech detection dataset.

Analyze the following comment and provide THREE separate ratings from 3 hypothetical annotators:

Categories:
1. HATE SPEECH: Language used to express hatred towards a targeted group or to be derogatory, to humiliate, or to insult members of the group
2. OFFENSIVE LANGUAGE: Casual use of offensive terms that are not necessarily directed at a group (e.g., casual profanity)
3. NEITHER: Comments that are neither hate speech nor offensive

Comment: "{comment_text}"

For each of the 3 annotators, decide which ONE category (0, 1, or 2) they would choose:
- 0 = hate_speech
- 1 = offensive_language  
- 2 = neither

Respond ONLY with a JSON object in this exact format:
{{"annotator1": 0, "annotator2": 1, "annotator3": 1}}

The numbers should be 0, 1, or 2 representing which category each annotator chose."""

        try:
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are an expert at annotating social media content for hate speech detection research. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.gpt_temperature,
                max_tokens=100
            )
            
            result_text = response.choices[0].message.content.strip()
            annotations = json.loads(result_text)
            
            votes = [annotations["annotator1"], annotations["annotator2"], annotations["annotator3"]]
            hate_speech_count = votes.count(0)
            offensive_count = votes.count(1)
            neither_count = votes.count(2)
            
            # Determine final class (majority vote)
            if hate_speech_count >= 2:
                final_class = 0
            elif offensive_count >= 2:
                final_class = 1
            else:
                final_class = 2
            
            self.total_annotated += 1
            
            return {
                "count": 3,
                "hate_speech": hate_speech_count,
                "offensive_language": offensive_count,
                "neither": neither_count,
                "class": final_class,
                "votes": votes
            }
            
        except Exception as e:
            print(f"❌ Error annotating: {e}")
            return None
    
    def display_annotation_live(self, comment_text, annotation):
        """
        Display the annotation in real-time with visual feedback
        """
        # Truncate comment for display
        display_text = comment_text[:150] + "..." if len(comment_text) > 150 else comment_text
        
        class_names = {0: "HATE SPEECH", 1: "OFFENSIVE", 2: "NEITHER"}
        class_colors = {0: "🔴", 1: "🟡", 2: "🟢"}
        
        print("\n" + "┌" + "─"*78 + "┐")
        print(f"│ 💬 COMMENT #{self.total_annotated}")
        print("├" + "─"*78 + "┤")
        print(f"│ {display_text[:76]}")
        if len(display_text) > 76:
            print(f"│ {display_text[76:152]}")
        print("├" + "─"*78 + "┤")
        print(f"│ 📊 ANNOTATION RESULTS:")
        print(f"│    Annotator 1: {class_names[annotation['votes'][0]]}")
        print(f"│    Annotator 2: {class_names[annotation['votes'][1]]}")
        print(f"│    Annotator 3: {class_names[annotation['votes'][2]]}")
        print("├" + "─"*78 + "┤")
        print(f"│ 🎯 FINAL: {class_colors[annotation['class']]} {class_names[annotation['class']]} "
              f"(Votes: H:{annotation['hate_speech']} O:{annotation['offensive_language']} N:{annotation['neither']})")
        
        # Show if we're keeping it
        needed_class = self.get_most_needed_class()
        will_keep = self.counts[annotation['class']] < self.target_per_class
        
        if will_keep:
            print(f"│ ✅ KEEPING - This class needs more samples")
        else:
            print(f"│ ⏭️  SKIPPING - This class has enough samples")
        
        print("└" + "─"*78 + "┘")
    
    def get_most_needed_class(self):
        """Return which class needs the most samples"""
        return min(self.counts.items(), key=lambda x: x[1])[0]
    
    def is_balanced(self):
        """Check if we've reached our targets"""
        return all(count >= self.target_per_class for count in self.counts.values())
    
    def get_target_sources(self):
        """
        Get sources to scrape from based on which class needs more samples
        """
        needed_class = self.get_most_needed_class()
        
        if needed_class == 0:
            return self.hate_speech_sources, "HATE SPEECH"
        elif needed_class == 1:
            return self.offensive_language_sources, "OFFENSIVE LANGUAGE"
        else:
            return self.neither_sources, "NEITHER"
    
    def scrape_reddit_comments_targeted(self, subreddit_name, limit=50):
        """
        Scrape comments from a specific subreddit
        """
        comments = []
        
        if not self.reddit:
            print("⚠️  No Reddit API credentials. Using fallback method...")
            return []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for submission in subreddit.hot(limit=10):
                if len(comments) >= limit:
                    break
                
                submission.comments.replace_more(limit=0)
                
                for comment in submission.comments.list():
                    if len(comments) >= limit:
                        break
                    
                    if hasattr(comment, 'body') and len(comment.body) > 20:
                        if comment.body not in ['[deleted]', '[removed]']:
                            comments.append(comment.body)
                            self.total_scraped += 1
            
        except Exception as e:
            print(f"⚠️  Error scraping r/{subreddit_name}: {e}")
        
        return comments
    
    def save_to_csv(self, comment_text, annotation, output_file):
        """
        Append a single comment to CSV
        """
        file_exists = os.path.isfile(output_file)
        
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['', 'count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row = {
                '': self.total_kept,
                'count': annotation['count'],
                'hate_speech': annotation['hate_speech'],
                'offensive_language': annotation['offensive_language'],
                'neither': annotation['neither'],
                'class': annotation['class'],
                'tweet': comment_text.replace('\n', ' ').replace('\r', ' ')
            }
            writer.writerow(row)
    
    def run_balanced_scraping(self, output_file="balanced_dataset_10k.csv"):
        """
        Main scraping loop with live display
        """
        self.print_header()
        
        print("🚀 Starting balanced scraping...")
        print("💡 Tip: Watch the annotations. If they seem wrong, press Ctrl+C to stop!\n")
        
        time.sleep(2)
        
        try:
            while not self.is_balanced() and not self.should_stop:
                # Get target sources
                sources, target_class_name = self.get_target_sources()
                
                print(f"\n🎯 Targeting: {target_class_name}")
                print(f"📍 Scraping from: r/{sources[0]} (and others)\n")
                
                # Rotate through sources
                for source in sources:
                    if self.is_balanced() or self.should_stop:
                        break
                    
                    # Scrape a batch of comments (from config)
                    comments = self.scrape_reddit_comments_targeted(source, limit=self.batch_size)
                    
                    for comment in comments:
                        if self.is_balanced() or self.should_stop:
                            break
                        
                        # Annotate
                        annotation = self.annotate_comment(comment)
                        
                        if annotation:
                            # Display live
                            self.display_annotation_live(comment, annotation)
                            
                            # Keep if this class needs more samples
                            if self.counts[annotation['class']] < self.target_per_class:
                                self.save_to_csv(comment, annotation, output_file)
                                self.counts[annotation['class']] += 1
                                self.total_kept += 1
                            
                            # Show progress every 10 kept comments
                            if self.total_kept % 10 == 0:
                                self.print_progress()
                            
                            # Rate limiting (from config)
                            time.sleep(self.rate_limit_delay)
                    
                    # Pause between subreddits
                    time.sleep(3)
        
        except KeyboardInterrupt:
            print("\n\n🛑 USER STOPPED THE SCRAPING")
            print("💾 Saving progress...")
            self.should_stop = True
        
        # Final summary
        print("\n" + "="*80)
        print("✅ SCRAPING COMPLETE (or stopped)")
        print("="*80)
        self.print_progress()
        print(f"📁 Output saved to: {output_file}")
        print(f"📊 Final stats:")
        print(f"   - Total comments scraped: {self.total_scraped:,}")
        print(f"   - Total comments annotated: {self.total_annotated:,}")
        print(f"   - Total comments kept: {self.total_kept:,}")
        print(f"   - Efficiency rate: {(self.total_kept/max(1,self.total_scraped)*100):.1f}%")
        print("="*80 + "\n")


def main():
    """
    Main function - loads config and runs scraper
    """
    print("\n" + "="*80)
    print("🎯 BALANCED COMMENT SCRAPER - OPTION E")
    print("="*80)
    
    # Load configuration
    print("📝 Loading configuration from config.json...")
    config = load_config("config.json")
    
    # Display settings
    target_per_class = config['scraper_settings']['target_per_class']
    total_target = target_per_class * 3
    output_file = config['scraper_settings']['output_file']
    gpt_model = config['scraper_settings']['gpt_model']
    
    print(f"✅ Configuration loaded successfully!")
    print("="*80)
    print(f"📊 Target: {target_per_class:,} comments per class (~{total_target:,} total)")
    print(f"🤖 GPT Model: {gpt_model}")
    print(f"💰 Estimated cost: ${(total_target*0.002):.2f} - ${(total_target*0.003):.2f}")
    print(f"⏱️  Estimated time: {(total_target*1.5/60):.0f} - {(total_target*2/60):.0f} minutes")
    print(f"📁 Output file: {output_file}")
    print("="*80)
    
    # Check Reddit credentials
    if config['api_keys'].get('reddit_client_id') == "your-reddit-client-id-here":
        print("\n⚠️  WARNING: Reddit API credentials not set in config.json")
        print("The scraper will work with limited functionality.")
        print("For best results, add Reddit API credentials to config.json:")
        print("  1. Go to: https://www.reddit.com/prefs/apps")
        print("  2. Create an app and get credentials")
        print("  3. Edit config.json and update reddit_client_id and reddit_client_secret")
        print("\n⏳ Waiting 5 seconds... (press Ctrl+C to abort)")
        time.sleep(5)
    
    # Initialize scraper with config
    scraper = BalancedCommentScraper(config=config)
    
    # Run the scraping
    scraper.run_balanced_scraping(output_file=output_file)
    
    print("\n🎉 Done! Check your output file: " + output_file)


if __name__ == "__main__":
    main()