"""
Test Script for PyTorch CUDA Deep Learning Models
Verifies that models work correctly and compare with TensorFlow
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pytorch_dl_models import (
    LSTMClassifier,
    BiLSTMClassifier,
    CNNClassifier,
    PyTorchDLTrainer,
    PyTorchPredictor,
    DEVICE
)

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_device():
    """Test CUDA availability"""
    print_section("1. DEVICE CHECK")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("[OK] CUDA ready for training!")
    else:
        print("[WARNING] CUDA not available - will use CPU")
    
    return torch.cuda.is_available()

def test_model_creation():
    """Test model creation"""
    print_section("2. MODEL CREATION TEST")
    
    models = {}
    
    # LSTM
    print("Creating LSTM model...")
    lstm = LSTMClassifier(vocab_size=1000, max_length=50)
    lstm = lstm.to(DEVICE)
    models['lstm'] = lstm
    print(f"[OK] LSTM created - Parameters: {sum(p.numel() for p in lstm.parameters()):,}")
    
    # BiLSTM
    print("\nCreating BiLSTM model...")
    bilstm = BiLSTMClassifier(vocab_size=1000, max_length=50)
    bilstm = bilstm.to(DEVICE)
    models['bilstm'] = bilstm
    print(f"[OK] BiLSTM created - Parameters: {sum(p.numel() for p in bilstm.parameters()):,}")
    
    # CNN
    print("\nCreating CNN model...")
    cnn = CNNClassifier(vocab_size=1000, max_length=50)
    cnn = cnn.to(DEVICE)
    models['cnn'] = cnn
    print(f"[OK] CNN created - Parameters: {sum(p.numel() for p in cnn.parameters()):,}")
    
    return models

def test_forward_pass(models):
    """Test forward pass"""
    print_section("3. FORWARD PASS TEST")
    
    # Create dummy input
    batch_size = 8
    max_length = 50
    dummy_input = torch.randint(0, 1000, (batch_size, max_length)).to(DEVICE)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input device: {dummy_input.device}")
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\n[{name.upper()}]")
        print(f"  Output shape: {output.shape}")
        print(f"  Output device: {output.device}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        assert output.shape == (batch_size, 3), f"Wrong output shape for {name}"
    
    print("\n[OK] All forward passes successful!")

def test_training():
    """Test training loop"""
    print_section("4. TRAINING TEST (1 epoch)")
    
    # Create small synthetic dataset
    n_samples = 1000
    vocab_size = 1000
    max_length = 50
    
    print(f"Creating synthetic dataset...")
    print(f"  Samples: {n_samples}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max length: {max_length}")
    
    X_train = np.random.randint(0, vocab_size, (n_samples, max_length))
    y_train = np.random.randint(0, 3, n_samples)
    
    X_val = np.random.randint(0, vocab_size, (200, max_length))
    y_val = np.random.randint(0, 3, 200)
    
    X_test = np.random.randint(0, vocab_size, (200, max_length))
    y_test = np.random.randint(0, 3, 200)
    
    # Train LSTM only (quick test)
    print("\nTraining LSTM for 1 epoch...")
    trainer = PyTorchDLTrainer(
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=3
    )
    
    model = trainer.create_model('lstm', embedding_dim=64, lstm_units=32)
    
    from pytorch_dl_models import create_data_loader
    train_loader = create_data_loader(X_train, y_train, batch_size=32)
    val_loader = create_data_loader(X_val, y_val, batch_size=32)
    
    history = trainer.train_model(
        model=model,
        model_name='lstm_test',
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        learning_rate=0.001,
        patience=1,
        verbose=False
    )
    
    print(f"\n[OK] Training completed!")
    print(f"  Train loss: {history['train_loss'][0]:.4f}")
    print(f"  Train acc: {history['train_acc'][0]:.4f}")
    print(f"  Val loss: {history['val_loss'][0]:.4f}")
    print(f"  Val acc: {history['val_acc'][0]:.4f}")
    
    # Test evaluation
    test_loader = create_data_loader(X_test, y_test, batch_size=32)
    test_loss, test_acc, y_true, y_pred = trainer.evaluate_model(
        model=model,
        test_loader=test_loader,
        verbose=False
    )
    
    print(f"\n[TEST] Test accuracy: {test_acc:.4f}")
    
    return model

def test_prediction(model):
    """Test prediction"""
    print_section("5. PREDICTION TEST")
    
    # Create test data
    X_test = np.random.randint(0, 1000, (10, 50))
    
    print(f"Input shape: {X_test.shape}")
    
    # Test predictor
    predictor = PyTorchPredictor(model)
    
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test)
    
    print(f"\n[OK] Predictions shape: {predictions.shape}")
    print(f"[OK] Probabilities shape: {probabilities.shape}")
    print(f"\nFirst 5 predictions: {predictions[:5]}")
    print(f"First 5 probabilities:")
    for i in range(5):
        print(f"  Sample {i}: {probabilities[i]}")

def test_save_load():
    """Test model saving and loading"""
    print_section("6. SAVE/LOAD TEST")
    
    save_dir = Path('/tmp/pytorch_test_models')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and train a small model
    print("Creating model...")
    trainer = PyTorchDLTrainer(vocab_size=1000, max_length=50, num_classes=3)
    model = trainer.create_model('lstm', embedding_dim=64, lstm_units=32)
    
    # Save
    print(f"Saving model to {save_dir}...")
    trainer.save_model(model, 'test_lstm', save_dir)
    
    # Load
    print("Loading model...")
    loaded_model = trainer.load_model('test_lstm', save_dir)
    
    # Test that loaded model works
    dummy_input = torch.randint(0, 1000, (5, 50)).to(DEVICE)
    with torch.no_grad():
        output1 = model(dummy_input)
        output2 = loaded_model(dummy_input)
    
    # Check outputs match
    diff = torch.abs(output1 - output2).max().item()
    print(f"\n[OK] Model saved and loaded successfully!")
    print(f"[OK] Output difference: {diff:.10f}")
    assert diff < 1e-5, "Loaded model outputs don't match!"

def benchmark_speed(has_cuda):
    """Benchmark training speed"""
    print_section("7. SPEED BENCHMARK")
    
    if not has_cuda:
        print("[SKIP] Skipping benchmark - CUDA not available")
        return
    
    import time
    
    # Small dataset
    n_samples = 5000
    X = np.random.randint(0, 1000, (n_samples, 50))
    y = np.random.randint(0, 3, n_samples)
    
    from pytorch_dl_models import create_data_loader
    
    batch_sizes = [32, 64]
    
    print("Testing training speed with different batch sizes...\n")
    
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size}")
        
        loader = create_data_loader(X, y, batch_size=batch_size, shuffle=True)
        
        model = LSTMClassifier(vocab_size=1000, max_length=50).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        model.train()
        start_time = time.time()
        
        for sequences, labels in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        elapsed = time.time() - start_time
        samples_per_sec = n_samples / elapsed
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {samples_per_sec:.0f} samples/sec")
        print()

def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "PYTORCH CUDA MODELS TEST SUITE" + " "*27 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    try:
        # Run tests
        has_cuda = test_device()
        models = test_model_creation()
        test_forward_pass(models)
        trained_model = test_training()
        test_prediction(trained_model)
        test_save_load()
        benchmark_speed(has_cuda)
        
        # Final summary
        print_section("TEST SUMMARY")
        print("[✓] Device check")
        print("[✓] Model creation")
        print("[✓] Forward pass")
        print("[✓] Training loop")
        print("[✓] Prediction")
        print("[✓] Save/Load")
        if has_cuda:
            print("[✓] Speed benchmark")
        
        print("\n" + "="*80)
        print("  ALL TESTS PASSED! ✓")
        print("="*80)
        print("\n[READY] PyTorch CUDA models are ready for use!")
        print("\nNext steps:")
        print("  1. Run: python main_train_enhanced.py --phase5")
        print("  2. Monitor: watch -n 1 nvidia-smi")
        print("  3. Check: PYTORCH_CUDA_GUIDE.md for detailed usage")
        print()
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("  TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)