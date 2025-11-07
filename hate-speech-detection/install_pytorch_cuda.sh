#!/bin/bash
# PyTorch CUDA Installation Script
# For RTX 3060 and similar GPUs

echo "=========================================="
echo "PyTorch CUDA Installation Script"
echo "=========================================="
echo ""

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "[CHECK] NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "[INFO] CUDA Version: $CUDA_VERSION"
    echo ""
else
    echo "[WARNING] nvidia-smi not found. No NVIDIA GPU detected."
    echo "[INFO] Will install CPU-only version of PyTorch"
    echo ""
fi

# Ask user which version to install
echo "Select PyTorch installation:"
echo "1. PyTorch with CUDA 11.8 (Recommended for RTX 30 series)"
echo "2. PyTorch with CUDA 12.1 (Latest, for RTX 40 series)"
echo "3. PyTorch CPU only (No GPU acceleration)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "[INSTALL] Installing PyTorch with CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo ""
        echo "[INSTALL] Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo ""
        echo "[INSTALL] Installing PyTorch CPU only..."
        pip install torch torchvision torchaudio
        ;;
    *)
        echo "[ERROR] Invalid choice"
        exit 1
        ;;
esac

# Verify installation
echo ""
echo "=========================================="
echo "Verifying PyTorch Installation"
echo "=========================================="
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("Running on CPU")
EOF

# Test PyTorch functionality
echo ""
echo "=========================================="
echo "Testing PyTorch CUDA Functionality"
echo "=========================================="
python3 << EOF
import torch
import torch.nn as nn

# Create a simple tensor operation
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print("[SUCCESS] CUDA operations working!")
    print(f"[TEST] Result shape: {z.shape}")
    print(f"[TEST] Device: {z.device}")
else:
    print("[INFO] Running on CPU - GPU not available")
EOF

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify PyTorch files are in your project:"
echo "   - pytorch_dl_models.py"
echo "   - pytorch_trainer.py"
echo ""
echo "2. Train models with PyTorch CUDA:"
echo "   python main_train_enhanced.py --phase5"
echo ""
echo "3. Monitor GPU usage while training:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "4. Check PYTORCH_CUDA_GUIDE.md for detailed usage"
echo ""