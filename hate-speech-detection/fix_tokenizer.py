"""
Fix the tokenizer - reconstruct it properly
"""
import joblib
from pathlib import Path
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

print("=" * 70)
print("FIXING TOKENIZER")
print("=" * 70)

# Load the saved tokenizer data
tokenizer_path = Path('saved_models/deep_learning/tokenizer.pkl')
data = joblib.load(tokenizer_path)

print(f"\n✓ Loaded tokenizer data")
print(f"  Keys: {list(data.keys())}")

# Extract what we need
word_index = data['word_index']
config = data['config']
tokenizer_config = data['tokenizer_config']

print(f"\n✓ Extracted data:")
print(f"  Vocab size: {len(word_index)}")
print(f"  Max length: {config.get('max_length', 100)}")

# Reconstruct Keras Tokenizer from word_index
from tensorflow.keras.preprocessing.text import Tokenizer

keras_tokenizer = Tokenizer(
    num_words=config.get('vocab_size', 20000),
    oov_token=tokenizer_config.get('oov_token', '<OOV>'),
    filters=tokenizer_config.get('filters', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
    lower=tokenizer_config.get('lower', True)
)

# Set the word_index manually
keras_tokenizer.word_index = word_index
keras_tokenizer.index_word = data['index_word']

print(f"\n✓ Reconstructed Keras Tokenizer")

# Create TextTokenizer wrapper
from models.deep_learning.text_tokenizer import TextTokenizer

text_tokenizer = TextTokenizer(
    vocab_size=config.get('vocab_size', 20000),
    max_length=config.get('max_length', 100)
)
text_tokenizer.tokenizer = keras_tokenizer
text_tokenizer.is_fitted = True

print(f"\n✓ Created TextTokenizer wrapper")

# Test it
test_texts = ["I hate you", "Hello world"]
try:
    sequences = text_tokenizer.texts_to_padded_sequences(test_texts)
    print(f"\n✓ Test successful!")
    print(f"  Input: {test_texts}")
    print(f"  Output shape: {sequences.shape}")
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Save the fixed tokenizer
print(f"\n✓ Saving fixed tokenizer...")
joblib.dump(text_tokenizer, tokenizer_path)
print(f"  Saved to: {tokenizer_path}")

print("\n" + "=" * 70)
print("TOKENIZER FIXED!")
print("=" * 70)
print("\nNow deep learning models should work!")