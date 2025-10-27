# Transfer Instructions for Windows Machine

This document provides instructions for transferring this project from your Mac to your Windows machine with RTX 3060.

## Files to Transfer

Transfer the entire `rgveda-ft` directory to your Windows machine. The project contains:

```
rgveda-ft/
├── app/
│   ├── __init__.py
│   ├── cli.py
│   ├── data_preparation.py
│   ├── train.py
│   └── test_setup.py
├── .gitignore
├── pyproject.toml
├── README.md
├── QUICKSTART.md
└── TRANSFER_INSTRUCTIONS.md (this file)
```

## Transfer Methods

### Option 1: Git (Recommended)

On Mac:
```bash
cd /Users/ganaraj.permunda/Projects/rgveda-ft
git init
git add .
git commit -m "Initial commit: EmbeddingGemma fine-tuning pipeline"
git remote add origin <your-repo-url>
git push -u origin main
```

On Windows:
```powershell
git clone <your-repo-url>
cd rgveda-ft
```

### Option 2: Direct Transfer

Use any of these methods:
- USB drive
- Cloud storage (Google Drive, Dropbox, OneDrive)
- Network share
- Email/file transfer service (if project is small)

## Setup on Windows

### 1. Install Prerequisites

**Python 3.10+:**
Download from https://www.python.org/downloads/

During installation:
- ✓ Check "Add Python to PATH"
- ✓ Check "Install pip"

**NVIDIA CUDA Drivers:**
Download from https://developer.nvidia.com/cuda-downloads
- Select Windows > x86_64 > your Windows version
- Install CUDA Toolkit 11.8 or later

Verify GPU:
```powershell
nvidia-smi
```

You should see your RTX 3060 listed.

**Install uv:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell after installation.

### 2. Project Setup

```powershell
# Navigate to project
cd rgveda-ft

# Install dependencies
uv sync

# This will install:
# - PyTorch with CUDA support
# - sentence-transformers
# - transformers
# - datasets
# - huggingface-hub
# - All other dependencies
```

### 3. Verify Installation

```powershell
uv run test-setup
```

This will verify:
- All packages installed correctly
- CUDA/GPU is detected
- HuggingFace authentication (you'll set this up next)

### 4. HuggingFace Authentication

```powershell
uv run huggingface-cli login
```

When prompted, paste your HuggingFace token.

Get token from: https://huggingface.co/settings/tokens
- Click "New token"
- Name: "rgveda-ft"
- Type: "Write" (needed to push dataset and models)
- Copy the token

**Accept EmbeddingGemma License:**
1. Visit: https://huggingface.co/google/embeddinggemma-300m
2. Click "Access repository"
3. Read and accept the terms

### 5. Run Setup Verification Again

```powershell
uv run test-setup
```

All checks should now pass.

## Training Workflow

### One-time: Split Dataset

```powershell
uv run split-dataset
```

This creates a 90/10 train/test split (~52k train, ~5.6k test) and pushes it to HuggingFace.

### Train the Model

```powershell
# Default settings (recommended for RTX 3060)
uv run train-embeddings

# With options:
uv run train-embeddings --epochs 5 --batch-size 32
```

Expected training time: **30 minutes to 2 hours**

### Monitor Training

The script will display:
- GPU information
- Dataset loading progress
- Training progress bar
- Evaluation metrics every 500 steps
- Final results and model location

### After Training

Your fine-tuned model will be in:
```
rgveda-ft/models/embeddinggemma-sanskrit-ft/
```

Test it:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./models/embeddinggemma-sanskrit-ft")
text = "धर्मक्षेत्रे कुरुक्षेत्रे"
embedding = model.encode(text)
print(f"Embedding shape: {embedding.shape}")  # Should be (768,)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```powershell
uv run train-embeddings --batch-size 32
# or
uv run train-embeddings --batch-size 16
```

### PyTorch Not Using GPU

Reinstall PyTorch with CUDA:
```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Permission Errors

Run PowerShell as Administrator:
- Right-click PowerShell
- Select "Run as Administrator"

### Antivirus Blocking

If Windows Defender or antivirus blocks uv/Python:
1. Add exception for the project folder
2. Add exception for uv executable
3. Temporarily disable during installation

### Network/Firewall Issues

If model downloads fail:
- Check firewall settings
- Ensure ports 443, 80 are not blocked
- Try different network if corporate firewall is blocking

## Performance Tips

### Optimize for RTX 3060

1. **Use default batch size (64)** - Should fit in 12GB VRAM
2. **Keep mixed precision enabled** (default) - Uses bfloat16 for efficiency
3. **Close other GPU applications** - Browsers, games, etc.
4. **Monitor GPU usage**: `nvidia-smi -l 1` (updates every second)

### Expected Memory Usage

- Model: ~1.2 GB
- Gradients: ~1.2 GB  
- Optimizer: ~2.4 GB
- Batch (64): ~5-6 GB
- **Total: ~10-11 GB** out of 12 GB available

## Next Steps

After successful training:

1. **Evaluate the model** on your specific tasks
2. **Push to HuggingFace Hub** (optional):
   ```powershell
   uv run train-embeddings --push-to-hub --hub-model-id "your-username/embeddinggemma-sanskrit"
   ```
3. **Share with team** or use in production
4. **Fine-tune further** if needed with more epochs or different hyperparameters

## Support

- **README.md**: Comprehensive documentation
- **QUICKSTART.md**: Quick reference for Windows
- **Test setup**: `uv run test-setup` to diagnose issues

## Project Information

- **Model**: EmbeddingGemma-300m (300M parameters)
- **Dataset**: Sanskrit triplets (57,625 examples)
- **Training**: Full fine-tune (not LoRA)
- **Hardware**: Optimized for RTX 3060 12GB
- **Framework**: sentence-transformers + PyTorch

