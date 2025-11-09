"""
Quick Test and Demo Script for Advanced Models
===============================================

Run this to verify installation and see capabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sklearn': 'Scikit-learn',
        'scipy': 'SciPy',
        'numpy': 'NumPy'
    }
    
    missing = []
    
    for package, name in required.items():
        try:
            __import__(package)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[MISSING] {name}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install -r models/advanced/requirements.txt")
        return False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n[OK] GPU available: {torch.cuda.get_device_name(0)}")
            print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("\n[WARNING] No GPU detected - training will be very slow")
            print("          Consider using Google Colab or cloud GPU")
    except:
        pass
    
    return True


def list_available_models():
    """List all available specialized models."""
    print("\n" + "=" * 80)
    print("AVAILABLE SPECIALIZED MODELS")
    print("=" * 80)
    
    try:
        from models.advanced.specialized_transformers import list_available_models
        
        models = list_available_models()
        
        print(f"\n{'Model':<20} {'Parameters':<12} {'Expected Gain':<15} Description")
        print("-" * 80)
        
        for model_type, config in models.items():
            print(f"{model_type:<20} {config['params']:<12} {config['expected_gain']:<15} {config['description'][:40]}")
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return True


def show_recommendations():
    """Show recommended configurations."""
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 80)
    
    configs = [
        {
            'name': 'Quick Start (10 min)',
            'models': ['hatebert'],
            'ensemble': False,
            'expected': '+2-4%'
        },
        {
            'name': 'Best Single Model (25 min)',
            'models': ['deberta-v3-large'],
            'ensemble': False,
            'expected': '+4-6%'
        },
        {
            'name': 'Multi-Model Ensemble (45 min)',
            'models': ['hatebert', 'deberta-v3-large', 'roberta-large'],
            'ensemble': True,
            'expected': '+5-7%'
        },
        {
            'name': 'Maximum Accuracy (60+ min)',
            'models': ['hatebert', 'deberta-v3-large'],
            'ensemble': True,
            'domain_adapt': True,
            'cv': True,
            'expected': '+6-8%'
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Models: {', '.join(config['models'])}")
        if config.get('ensemble'):
            print(f"   Ensemble: Yes")
        if config.get('domain_adapt'):
            print(f"   Domain Adaptation: Yes")
        if config.get('cv'):
            print(f"   Cross-Validation: Yes")
        print(f"   Expected improvement: {config['expected']}")


def show_usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n1. Train HateBERT (simplest):")
    print("-" * 40)
    print("""
from models.advanced.specialized_transformers import SpecializedTransformerModel

model = SpecializedTransformerModel('hatebert')
model.train(X_train, y_train, X_val, y_val)
metrics = model.evaluate(X_test, y_test)
print(f"F1 Score: {metrics['f1_macro']:.4f}")
""")
    
    print("\n2. Create Multi-Model Ensemble:")
    print("-" * 40)
    print("""
from models.advanced.train_advanced import train_advanced_models

trainer = train_advanced_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_types=['hatebert', 'deberta-v3-large'],
    create_ensemble=True
)
""")
    
    print("\n3. Full Pipeline with Domain Adaptation:")
    print("-" * 40)
    print("""
trainer = train_advanced_models(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_types=['hatebert', 'deberta-v3-large'],
    use_domain_adaptation=True,
    create_ensemble=True,
    create_cv_ensemble=True
)
""")


def run_simple_test():
    """Run a simple test with dummy data."""
    print("\n" + "=" * 80)
    print("RUNNING SIMPLE TEST")
    print("=" * 80)
    
    try:
        import numpy as np
        from models.advanced.specialized_transformers import SpecializedTransformerModel
        
        print("\nCreating dummy data...")
        X_train = ["I hate you"] * 50 + ["Good morning"] * 50
        y_train = np.array([0] * 50 + [2] * 50)
        X_val = ["I hate you"] * 10 + ["Good morning"] * 10
        y_val = np.array([0] * 10 + [2] * 10)
        
        print(f"Training: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        
        print("\nInitializing HateBERT...")
        model = SpecializedTransformerModel('hatebert')
        
        print("\nModel initialized successfully!")
        print("To actually train:")
        print("  model.train(X_train, y_train, X_val, y_val)")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("ADVANCED MODELS - QUICK TEST & DEMO")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # List models
    if not list_available_models():
        return
    
    # Show recommendations
    show_recommendations()
    
    # Show usage examples
    show_usage_examples()
    
    # Run simple test
    print("\n" + "=" * 80)
    print("Would you like to run a simple initialization test? (y/n)")
    
    try:
        response = input("> ").strip().lower()
        if response == 'y':
            run_simple_test()
    except:
        print("\nSkipping test")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare your data (X_train, y_train, X_val, y_val, X_test, y_test)")
    print("2. Choose a configuration from recommendations above")
    print("3. Run training:")
    print("   python -m models.advanced.train_advanced")
    print("\nFor full documentation:")
    print("   See models/advanced/README.md")
    print("=" * 80)


if __name__ == "__main__":
    main()