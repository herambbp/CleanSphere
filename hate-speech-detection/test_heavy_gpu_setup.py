"""
Test Script for Heavy GPU BERT Enhancement
==========================================

This script tests the heavy GPU BERT implementation to ensure
everything is set up correctly before training.
"""

import sys
from pathlib import Path

print("=" * 80)
print("HEAVY GPU BERT SETUP TEST")
print("=" * 80)

# Test 1: Check Python imports
print("\n[TEST 1] Checking Python imports...")
try:
    import numpy as np
    import pandas as pd
    print("✓ NumPy and Pandas installed")
except ImportError as e:
    print(f"✗ NumPy/Pandas missing: {e}")
    sys.exit(1)

# Test 2: Check PyTorch
print("\n[TEST 2] Checking PyTorch...")
try:
    import torch
    print(f"✓ PyTorch installed: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️ CUDA not available - will run on CPU (very slow)")
        print("  Install CUDA-enabled PyTorch:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("✗ PyTorch not installed")
    print("  Install with:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Test 3: Check Transformers
print("\n[TEST 3] Checking Transformers...")
try:
    import transformers
    print(f"✓ Transformers installed: {transformers.__version__}")
except ImportError:
    print("✗ Transformers not installed")
    print("  Install with: pip install transformers")
    sys.exit(1)

# Test 4: Check sklearn
print("\n[TEST 4] Checking scikit-learn...")
try:
    import sklearn
    print(f"✓ scikit-learn installed: {sklearn.__version__}")
except ImportError:
    print("⚠️ scikit-learn not installed (needed for metrics)")
    print("  Install with: pip install scikit-learn")

# Test 5: Check Heavy GPU BERT module
print("\n[TEST 5] Checking Heavy GPU BERT module...")
try:
    from bert_model_heavy_gpu import (
        HeavyGPUBERTModel,
        HeavyGPUBERTConfig,
        HAS_TORCH,
        HAS_TRANSFORMERS
    )
    print("✓ bert_model_heavy_gpu.py found")
    print(f"  PyTorch available: {HAS_TORCH}")
    print(f"  Transformers available: {HAS_TRANSFORMERS}")
except ImportError as e:
    print(f"✗ bert_model_heavy_gpu.py not found: {e}")
    print("  Make sure bert_model_heavy_gpu.py is in the current directory")
    sys.exit(1)

# Test 6: Check Integration module
print("\n[TEST 6] Checking Integration module...")
try:
    from bert_integration import (
        HeavyGPUBERTTrainer,
        train_heavy_gpu_bert,
        HAS_HEAVY_BERT
    )
    print("✓ bert_integration.py found")
    print(f"  Heavy BERT available: {HAS_HEAVY_BERT}")
except ImportError as e:
    print(f"✗ bert_integration.py not found: {e}")
    print("  Make sure bert_integration.py is in the current directory")
    sys.exit(1)

# Test 7: Test model creation
print("\n[TEST 7] Testing model creation...")
try:
    config = HeavyGPUBERTConfig(
        model_name='bert-base',  # Use small model for testing
        batch_size=8,
        epochs=1
    )
    print("✓ Configuration created")
    
    model = HeavyGPUBERTModel(config=config, num_classes=3)
    print("✓ Model instance created")
    
    model.build_model()
    print("✓ Model built successfully")
    
    # Get model info
    info = model.get_model_info()
    print(f"  Model: {info['model_name']}")
    print(f"  Parameters: {info['total_params']:,}")
    print(f"  Device: {info['device']}")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test tokenization
print("\n[TEST 8] Testing tokenization...")
try:
    test_texts = [
        "I hate you",
        "Good morning everyone",
        "You are an idiot"
    ]
    
    encoded = model.tokenize_texts(test_texts)
    print(f"✓ Tokenization works")
    print(f"  Input shape: {encoded['input_ids'].shape}")
    
except Exception as e:
    print(f"✗ Tokenization failed: {e}")
    sys.exit(1)

# Test 9: Test prediction
print("\n[TEST 9] Testing prediction...")
try:
    predictions = model.predict(test_texts)
    print(f"✓ Prediction works: {predictions}")
    
    probabilities = model.predict_proba(test_texts)
    print(f"✓ Probability prediction works: {probabilities.shape}")
    
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    sys.exit(1)

# Test 10: Check project structure
print("\n[TEST 10] Checking project structure...")
expected_files = [
    'bert_model_heavy_gpu.py',
    'bert_integration.py',
    'main_train_enhanced_heavy_gpu.py',
    'HEAVY_GPU_BERT_README.md'
]

for file in expected_files:
    if Path(file).exists():
        print(f"✓ {file} found")
    else:
        print(f"⚠️ {file} not found (optional)")

# Test 11: List available models
print("\n[TEST 11] Available BERT models...")
print("-" * 60)
for key, value in HeavyGPUBERTConfig.MODEL_OPTIONS.items():
    print(f"  {key:20s} -> {value}")

# Test 12: GPU Memory recommendation
print("\n[TEST 12] GPU Memory recommendations...")
print("-" * 60)
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Your GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {gpu_memory:.1f} GB")
    print("\nRecommended models:")
    
    if gpu_memory >= 40:
        print("  ✓ bert-large (batch_size=128)")
        print("  ✓ roberta-large (batch_size=64)")
        print("  ✓ All models with large batches")
        print("  ✓ Train multiple models simultaneously")
    elif gpu_memory >= 24:
        print("  ✓ bert-large (batch_size=64)")
        print("  ✓ roberta-large (batch_size=32-48)")
        print("  ✓ Most models work well")
    elif gpu_memory >= 16:
        print("  ✓ bert-large (batch_size=32-48)")
        print("  ✓ roberta-base (batch_size=64)")
        print("  ⚠️ roberta-large (batch_size=16-32, may need adjustment)")
    elif gpu_memory >= 8:
        print("  ✓ bert-base (batch_size=32)")
        print("  ✓ roberta-base (batch_size=32)")
        print("  ✓ distilbert (batch_size=64)")
        print("  ⚠️ Large models may not fit")
    else:
        print("  ⚠️ Limited memory - use smaller models")
        print("  ✓ distilbert (batch_size=16)")
        print("  ✓ albert-base (batch_size=32)")
else:
    print("No GPU detected - CPU training will be very slow")
    print("Consider using Google Colab or cloud GPU")

# Final summary
print("\n" + "=" * 80)
print("SETUP TEST COMPLETE")
print("=" * 80)

all_tests_passed = True
if not torch.cuda.is_available():
    print("\n⚠️ WARNING: No GPU detected")
    print("  Training will be VERY SLOW on CPU")
    print("  Consider installing CUDA or using Google Colab")
    all_tests_passed = False

if HAS_TORCH and HAS_TRANSFORMERS and HAS_HEAVY_BERT:
    print("\n✓ ALL TESTS PASSED!")
    print("\n[READY TO TRAIN]")
    print("=" * 80)
    print("\nQuick start commands:")
    print("  1. Train BERT-Large:")
    print("     python main_train_enhanced_heavy_gpu.py --phase5 --models bert-large")
    print("\n  2. Train multiple models:")
    print("     python main_train_enhanced_heavy_gpu.py --phase5 --models bert-large roberta-base")
    print("\n  3. See all options:")
    print("     python main_train_enhanced_heavy_gpu.py --usage")
    print("\n  4. List available models:")
    print("     python main_train_enhanced_heavy_gpu.py --list-models")
    print("=" * 80)
else:
    print("\n✗ SOME TESTS FAILED")
    print("\nPlease install missing dependencies:")
    if not HAS_TORCH:
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    if not HAS_TRANSFORMERS:
        print("  pip install transformers")
    all_tests_passed = False

if not all_tests_passed:
    sys.exit(1)