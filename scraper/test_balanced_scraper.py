"""
Test Script for Balanced Scraper
Run this FIRST to test with 30 comments (10 per class) before doing 10K
Uses config.json for settings
"""

import os
import sys
import json
from pathlib import Path
from balanced_scraper import BalancedCommentScraper, load_config

def test_run():
    """
    Test the scraper with just 30 comments (10 per class)
    """
    print("="*80)
    print("🧪 TEST MODE - BALANCED SCRAPER")
    print("="*80)
    print("📊 Target: 10 comments per class (30 total)")
    print("💰 Cost: ~$0.06-0.10")
    print("⏱️  Time: ~1-2 minutes")
    print("="*80)
    print("\n⚠️  This is a TEST RUN to verify annotations look correct")
    print("✅ If they look good, run the full scraper for 10K comments")
    print("❌ If they look wrong, press Ctrl+C and we'll adjust\n")
    
    input("Press Enter to start the test... ")
    
    # Load config
    print("\n📝 Loading configuration from config.json...")
    try:
        config = load_config("config.json")
        print("✅ Configuration loaded!")
    except SystemExit:
        return
    
    # Override target for testing
    config['scraper_settings']['target_per_class'] = 10
    config['scraper_settings']['output_file'] = "test_balanced_output.csv"
    
    print(f"📝 Overriding settings for test:")
    print(f"   - Target: 10 per class (30 total)")
    print(f"   - Output: test_balanced_output.csv")
    
    # Initialize scraper
    scraper = BalancedCommentScraper(config=config)
    
    # Run the test
    output_file = config['scraper_settings']['output_file']
    scraper.run_balanced_scraping(output_file=output_file)
    
    print("\n" + "="*80)
    print("🎉 TEST COMPLETE!")
    print("="*80)
    print(f"\n📁 Check the output: {output_file}")
    print("\n📋 REVIEW CHECKLIST:")
    print("   □ Open the CSV file")
    print("   □ Read 5-10 random comments")
    print("   □ Check if classifications make sense")
    print("   □ Verify each class has ~10 comments")
    print("\n✅ If everything looks good:")
    print("   → Run: python balanced_scraper.py")
    print("   → For the full 10K dataset")
    print("\n❌ If annotations seem wrong:")
    print("   → Let me know which ones are incorrect")
    print("   → I'll adjust the GPT prompts")
    print("="*80)

if __name__ == "__main__":
    test_run()