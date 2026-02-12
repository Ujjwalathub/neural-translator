#!/usr/bin/env python3
# verify_install.py

try:
    import torch
    print(f"✅ PyTorch successfully imported!")
    print(f"📦 Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"🎯 CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️ GPU NOT detected (Code will run slow on CPU)")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")
