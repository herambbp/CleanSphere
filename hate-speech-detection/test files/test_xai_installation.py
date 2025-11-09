# test_xai_installation.py
print("Testing XAI Installation...")
print("=" * 70)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from explainability.explainability_engine import ComprehensiveExplainer, KeywordExplainer
    print("   ✓ explainability_engine imported successfully")
except ImportError as e:
    print(f"   ✗ Error: {e}")

try:
    from inference.explainable_classifier import ExplainableTweetClassifier
    print("   ✓ explainable_classifier imported successfully")
except ImportError as e:
    print(f"   ✗ Error: {e}")

# Test 2: Check LIME
print("\n2. Checking LIME...")
# robust LIME version detection (replace the old lime.__version__ usage)
import importlib
import importlib.metadata
import pkg_resources
import lime

def _get_pkg_version(pkg_name: str, module) -> str:
    # 1) common attribute
    ver = getattr(module, "__version__", None)
    if ver:
        return ver
    # 2) importlib.metadata (py3.8+)
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        pass
    # 3) pkg_resources (setuptools)
    try:
        return pkg_resources.get_distribution(pkg_name).version
    except Exception:
        pass
    return "unknown"

print(f"   ✓ LIME installed (version {_get_pkg_version('lime', lime)})")

# Test 3: Check SHAP (optional)
print("\n3. Checking SHAP (optional)...")
try:
    import shap
    print(f"   ✓ SHAP installed (version {shap.__version__})")
except ImportError:
    print("   ⚠ SHAP not installed (optional)")

# Test 4: Keyword explainer (no model required)
print("\n4. Testing keyword explainer...")
try:
    from explainability.explainability_engine import KeywordExplainer
    
    explainer = KeywordExplainer()
    result = explainer.explain("I will kill you bitch", predicted_class=0)
    
    print("   ✓ Keyword explainer works!")
    print(f"   Found {result['total_categories']} keyword categories")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Full classifier (requires trained model)
print("\n5. Testing explainable classifier...")
try:
    from inference.explainable_classifier import ExplainableTweetClassifier
    
    classifier = ExplainableTweetClassifier()
    result = classifier.classify_with_explanation(
        "Good morning everyone",
        verbose=False
    )
    
    print("   ✓ Explainable classifier works!")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Methods available: {result['explanation']['methods_used']}")
except FileNotFoundError:
    print("   ⚠ No trained model found - train models first")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 70)
print("Installation test complete!")