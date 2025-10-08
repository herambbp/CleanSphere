"""
Test all models - both Traditional ML and Deep Learning
"""

from inference.tweet_classifier import TweetClassifier
import sys

print("=" * 80)
print("COMPREHENSIVE MODEL TEST")
print("=" * 80)

# Test tweet
test_tweet = "I will kill you, you fucking bitch cunt whore"

# All available models
all_models = {
    'Traditional ML': ['mlp', 'xgboost', 'randomforest', 'gradientboosting', 'svm'],
    'Deep Learning': ['cnn', 'bilstm', 'lstm']
}

results = {}

for category, models in all_models.items():
    print(f"\n{'=' * 80}")
    print(f"Testing {category}")
    print('=' * 80)
    
    for model_name in models:
        print(f"\n--- Testing {model_name.upper()} ---")
        
        try:
            # Determine model type
            model_type = 'traditional' if category == 'Traditional ML' else 'deep_learning'
            
            # Load classifier
            classifier = TweetClassifier(model_name=model_name, model_type=model_type)
            
            # Make prediction
            result = classifier.classify_tweet(test_tweet, verbose=False)
            
            # Store result
            results[model_name] = {
                'status': 'SUCCESS',
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'model_type': result['model_type']
            }
            
            print(f"✓ Status: SUCCESS")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.1%}")
            print(f"  Model Type: {result['model_type']}")
            
        except Exception as e:
            results[model_name] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"✗ Status: FAILED")
            print(f"  Error: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

successful = [m for m, r in results.items() if r['status'] == 'SUCCESS']
failed = [m for m, r in results.items() if r['status'] == 'FAILED']

print(f"\nSuccessful: {len(successful)}/{len(results)}")
for model in successful:
    print(f"  ✓ {model}: {results[model]['prediction']} ({results[model]['confidence']:.1%})")

if failed:
    print(f"\nFailed: {len(failed)}/{len(results)}")
    for model in failed:
        print(f"  ✗ {model}: {results[model]['error']}")

# Recommend best working model
if successful:
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"Use one of these working models: {', '.join(successful)}")
    print(f"Best Traditional ML: mlp (fastest and reliable)")
    if 'cnn' in successful:
        print(f"Best Deep Learning: cnn (good balance of speed and accuracy)")