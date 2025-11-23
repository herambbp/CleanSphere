"""
Simple BERT Tester - Minimal code for quick testing
Based on official tweet_classifier usage guide

Usage:
    python test_bert_simple.py
"""

from inference.explainable_classifier import ExplainableTweetClassifier

# Initialize BERT classifier (automatic model selection)
print("Loading BERT model...")
classifier = ExplainableTweetClassifier(model_name='bert-base', model_type='auto')
print(f"✓ Model loaded: {classifier.model_name} ({classifier.model_type})\n")

# Test single sentence with full analysis
text = "man I love you so much"

result = classifier.classify_with_explanation(
    text=text,
    include_severity=True,
    verbose=False
)

# Print results
print(f"{'='*70}")
print(f"Text: {text}")
print(f"{'='*70}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nProbabilities:")
for label, prob in result['probabilities'].items():
    print(f"  {label:20s}: {prob:.2%}")

if 'severity' in result:
    print(f"\nSeverity: {result['severity']['severity_label']} ({result['severity']['severity_score']}/100)")
    print(f"Explanation: {result['severity']['explanation']}")

if 'action' in result:
    print(f"\nAction: {result['action']['action_string']}")
    print(f"Urgency: {result['action']['urgency']}")

print(f"\nModel: {result['model_info']['name']} ({result['model_info']['type']})")
print(f"{'='*70}\n")

# Test another example
print("\nTesting another example...")
text2 = "You are awesome and kind"
result2 = classifier.classify_with_explanation(text=text2, include_severity=True, verbose=False)

print(f"{'='*70}")
print(f"Text: {text2}")
print(f"{'='*70}")
print(f"Prediction: {result2['prediction']}")
print(f"Confidence: {result2['confidence']:.2%}")
print(f"{'='*70}\n")