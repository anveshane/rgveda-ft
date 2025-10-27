# Quick Start Guide

## For Windows Users with RTX 3060

### Prerequisites
1. Install uv package manager
2. Clone this repository to your Windows machine
3. Ensure CUDA drivers are installed for RTX 3060

### Setup (One-time)

```powershell
# 1. Install dependencies
uv sync

# 2. Authenticate with HuggingFace
uv run huggingface-cli login

# 3. Accept EmbeddingGemma license
# Visit: https://huggingface.co/google/embeddinggemma-300m
# Click "Access repository" and accept terms
```

### Step 1: Split Dataset (One-time)

```powershell
uv run split-dataset
```

Expected output: ~52k train, ~5.6k test examples

### Step 2: Train Model

```powershell
# Default settings (recommended for RTX 3060)
uv run train-embeddings

# If you get out of memory errors, reduce batch size:
uv run train-embeddings --batch-size 32

# Train for more epochs:
uv run train-embeddings --epochs 5

# Push to HuggingFace Hub after training:
uv run train-embeddings --push-to-hub --hub-model-id "your-username/embeddinggemma-sanskrit"
```

### Expected Training Time
- **30 minutes to 2 hours** depending on configuration
- Default (batch_size=64, 3 epochs): ~1 hour

### After Training

Your model will be saved in: `./models/embeddinggemma-sanskrit-ft/`

Test it:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./models/embeddinggemma-sanskrit-ft")

# IMPORTANT: Always use task prompts with EmbeddingGemma!
query_prompt = "task: search result | query: "
doc_prompt = "title: none | text: "

# For search queries
query_emb = model.encode(query_prompt + "धर्मक्षेत्रे कुरुक्षेत्रे")

# For documents
doc_emb = model.encode(doc_prompt + "भगवद्गीता महाभारतस्य भागः")

print(query_emb.shape)  # (768,)
```

**Note:** The training pipeline automatically adds these prompts during training. You must use the same prompts when using the fine-tuned model!

## Troubleshooting

### Out of Memory
```powershell
uv run train-embeddings --batch-size 32
# or even lower:
uv run train-embeddings --batch-size 16
```

### CUDA Not Detected
- Verify GPU: `nvidia-smi` in PowerShell
- Reinstall PyTorch with CUDA support if needed

### Permission Errors on Windows
- Run PowerShell as Administrator if needed
- Check antivirus isn't blocking uv or Python

## File Locations

- **Trained models**: `models/embeddinggemma-sanskrit-ft/`
- **Checkpoints during training**: Created in output directory
- **Logs**: Console output (redirect to file if needed)

## Next Steps

See the full [README.md](README.md) for:
- Detailed usage examples
- Advanced configuration options
- Using the model for search and retrieval
- Pushing models to HuggingFace Hub

