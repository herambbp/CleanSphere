"""
Deep Learning Models for Hate Speech Detection

This module provides deep learning models for improved accuracy:
- LSTM: Long Short-Term Memory networks
- BiLSTM: Bidirectional LSTM networks
- CNN: Convolutional Neural Networks for text
- BERT: Transformer-based models (optional)

Requirements:
- TensorFlow 2.13+
- Transformers 4.30+ (for BERT)
- Torch 2.0+ (for BERT)
"""

__version__ = '1.0.0'

# Check TensorFlow availability
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
    TF_VERSION = tf.__version__
except ImportError:
    HAS_TENSORFLOW = False
    TF_VERSION = None

# Check Transformers availability (for BERT)
try:
    import transformers
    HAS_TRANSFORMERS = True
    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    HAS_TRANSFORMERS = False
    TRANSFORMERS_VERSION = None

# Import components if TensorFlow is available
if HAS_TENSORFLOW:
    try:
        from .text_tokenizer import TextTokenizer
        from .lstm_model import LSTMModel
        from .bilstm_model import BiLSTMModel
        from .cnn_model import CNNModel
        
        __all__ = [
            'TextTokenizer',
            'LSTMModel',
            'BiLSTMModel',
            'CNNModel',
            'HAS_TENSORFLOW',
            'HAS_TRANSFORMERS',
            'TF_VERSION',
            'TRANSFORMERS_VERSION'
        ]
        
        # Import BERT if transformers is available
        if HAS_TRANSFORMERS:
            try:
                from .bert_model import BERTModel
                __all__.append('BERTModel')
            except ImportError as e:
                # BERT model file might not exist yet
                pass
    
    except ImportError as e:
        # Model files might not exist yet during development
        __all__ = [
            'HAS_TENSORFLOW',
            'HAS_TRANSFORMERS',
            'TF_VERSION',
            'TRANSFORMERS_VERSION'
        ]
else:
    __all__ = [
        'HAS_TENSORFLOW',
        'HAS_TRANSFORMERS',
        'TF_VERSION',
        'TRANSFORMERS_VERSION'
    ]

# Helper function to check requirements
def check_requirements(require_bert: bool = False):
    """
    Check if required libraries are installed.
    
    Args:
        require_bert: Whether BERT is required
    
    Returns:
        Tuple of (tensorflow_available, transformers_available)
    
    Raises:
        ImportError: If required libraries are missing
    """
    if not HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required for deep learning models.\n"
            "Install with: pip install tensorflow==2.13.0 keras==2.13.1"
        )
    
    if require_bert and not HAS_TRANSFORMERS:
        raise ImportError(
            "Transformers and PyTorch are required for BERT.\n"
            "Install with: pip install transformers==4.30.0 torch==2.0.1"
        )
    
    return HAS_TENSORFLOW, HAS_TRANSFORMERS

# Print library status (for debugging)
def print_library_status():
    """Print status of required libraries."""
    print("=" * 70)
    print("DEEP LEARNING LIBRARIES STATUS")
    print("=" * 70)
    
    if HAS_TENSORFLOW:
        print(f"✅ TensorFlow: {TF_VERSION}")
        
        # Check for GPU
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU Available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  GPU: Not available (will use CPU)")
    else:
        print("❌ TensorFlow: Not installed")
        print("   Install: pip install tensorflow==2.13.0")
    
    if HAS_TRANSFORMERS:
        print(f"✅ Transformers: {TRANSFORMERS_VERSION}")
        
        # Check for PyTorch
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"✅ CUDA Available: {torch.cuda.device_count()} device(s)")
            else:
                print("⚠️  CUDA: Not available (BERT will use CPU)")
        except ImportError:
            print("❌ PyTorch: Not installed (needed for BERT)")
    else:
        print("❌ Transformers: Not installed (BERT unavailable)")
        print("   Install: pip install transformers==4.30.0 torch==2.0.1")
    
    print("=" * 70)

# Testing function
if __name__ == "__main__":
    print_library_status()
    
    # Try to check requirements
    try:
        check_requirements(require_bert=False)
        print("\n✅ All basic requirements satisfied!")
        print("   LSTM, BiLSTM, and CNN models are available.")
    except ImportError as e:
        print(f"\n❌ Missing requirements: {e}")
    
    try:
        check_requirements(require_bert=True)
        print("✅ BERT requirements satisfied!")
        print("   All models including BERT are available.")
    except ImportError as e:
        print(f"\n⚠️  BERT requirements not satisfied: {e}")
        print("   LSTM, BiLSTM, and CNN will still work.")