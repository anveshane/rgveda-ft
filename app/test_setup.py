"""
Quick setup verification script.

Run this to verify your environment is correctly configured before training.
"""

import sys
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import sentence_transformers
        print(f"✓ Sentence Transformers {sentence_transformers.__version__}")
        
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
        
        from huggingface_hub import HfApi
        print("✓ HuggingFace Hub")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("✓ CUDA available")
        print(f"  Device: {device_name}")
        print(f"  Memory: {memory_gb:.2f} GB")
        
        # Test if we can allocate memory
        try:
            test_tensor = torch.zeros(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✓ GPU memory allocation test passed")
            return True
        except Exception as e:
            print(f"✗ GPU memory allocation failed: {e}")
            return False
    else:
        print("✗ CUDA not available - training will be very slow")
        print("  If you have a GPU, ensure CUDA drivers are installed")
        return False


def test_hf_auth():
    """Test HuggingFace authentication."""
    print("\nTesting HuggingFace authentication...")
    
    from huggingface_hub import HfApi
    
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Authenticated as: {user['name']}")
        return True
    except Exception:
        print("✗ Not authenticated with HuggingFace")
        print("  Run: uv run huggingface-cli login")
        return False


def test_model_access():
    """Test if we can access EmbeddingGemma model."""
    print("\nTesting EmbeddingGemma access...")
    
    from huggingface_hub import HfApi
    
    try:
        api = HfApi()
        api.model_info("google/embeddinggemma-300m")
        print("✓ Can access EmbeddingGemma model")
        return True
    except Exception:
        print("✗ Cannot access EmbeddingGemma model")
        print("  Visit: https://huggingface.co/google/embeddinggemma-300m")
        print("  Click 'Access repository' and accept the terms")
        return False


def test_dataset_access():
    """Test if we can access the Sanskrit triplets dataset."""
    print("\nTesting dataset access...")
    
    from huggingface_hub import HfApi
    
    try:
        api = HfApi()
        api.dataset_info("indhic-ai/sanskrit-triplets")
        print("✓ Can access Sanskrit triplets dataset")
        return True
    except Exception:
        print("⚠ Cannot access dataset")
        print("  This might be okay if the dataset is private/restricted")
        return True  # Don't fail on this


def main():
    """Run all tests."""
    print("="*60)
    print("Setup Verification for rgveda-ft")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("HuggingFace Auth", test_hf_auth()))
    results.append(("Model Access", test_model_access()))
    results.append(("Dataset Access", test_dataset_access()))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("="*60 + "\n")
    
    if all_passed:
        print("🎉 All tests passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. uv run split-dataset")
        print("  2. uv run train-embeddings")
    else:
        print("⚠ Some tests failed. Please fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()

