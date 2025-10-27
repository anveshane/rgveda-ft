# rgveda-ft: Fine-tuning EmbeddingGemma for Sanskrit

Fine-tuning pipeline for Google's [EmbeddingGemma-300m](https://huggingface.co/google/embeddinggemma-300m) on Sanskrit text triplets. This project creates specialized embeddings optimized for Sanskrit semantic similarity, search, and retrieval tasks.

## Overview

EmbeddingGemma-300m is a 300M parameter state-of-the-art embedding model from Google DeepMind. This project fine-tunes it on the [Sanskrit triplets dataset](https://huggingface.co/datasets/indhic-ai/sanskrit-triplets) using full parameter training (not LoRA) to create embeddings specifically optimized for Sanskrit text.

## Features

- **Full Fine-tuning**: All 300M parameters trained for optimal Sanskrit understanding
- **Triplet Loss Training**: Learns that similar texts should be closer than dissimilar texts in embedding space
- **Evaluation-driven**: Saves best model based on triplet accuracy on held-out test set
- **Efficient Training**: Optimized for NVIDIA RTX 3060 12GB GPU with mixed precision (bfloat16)
- **Easy CLI**: Simple commands for dataset preparation and training

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or equivalent
- **RAM**: 16GB+ recommended
- **Storage**: ~5GB for model checkpoints

### Software
- **Python**: 3.10 or higher
- **uv**: Package manager (installation instructions below)
- **CUDA**: Compatible with your GPU (CUDA 11.8+ recommended)
  - The project automatically installs PyTorch with CUDA 11.8 support
  - Ensure you have compatible NVIDIA drivers installed (run `nvidia-smi` to check)

## Installation

### 1. Install uv (if not already installed)

**On Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup Project

```bash
cd rgveda-ft
uv sync
```

This will install all dependencies defined in `pyproject.toml`, including PyTorch with CUDA 11.8 support for GPU acceleration.

**Note:** The project is configured to automatically install the CUDA-enabled version of PyTorch. The configuration in `pyproject.toml` specifies:
- PyTorch CUDA 11.8 index for GPU support
- Automatic installation of `torch`, `torchvision`, and `torchaudio` with CUDA capabilities

If you need a different CUDA version, modify the `[[tool.uv.index]]` section in `pyproject.toml`.

### 3. Authenticate with HuggingFace

You need HuggingFace authentication to:
- Access the EmbeddingGemma model (requires accepting license)
- Download and push datasets
- Push trained models (optional)

**Authenticate:**
```bash
uv run huggingface-cli login
```

Follow the prompts and paste your HuggingFace token. Get your token from: https://huggingface.co/settings/tokens

**Accept EmbeddingGemma License:**
Visit https://huggingface.co/google/embeddinggemma-300m and click "Access repository" to accept the terms.

## Usage

### Step 0: Verify Setup (Optional but Recommended)

Before starting, verify your environment is correctly configured:

```bash
uv run test-setup
```

This will check:
- All required packages are installed
- CUDA/GPU is available and working
- HuggingFace authentication is set up
- You have access to EmbeddingGemma model

### Step 1: Fine-tune Model

The Sanskrit triplets dataset already includes train/test splits, so you can start training directly with optimized defaults:

```bash
uv run train-embeddings
```

This uses the **optimized defaults** for RTX 3060:
- Loss: MNRL (MultipleNegativesRankingLoss) - 2-3x faster than TripletLoss
- Batch size: 64 (can increase to 96-128 with MNRL's lower memory usage)
- Epochs: 3
- Mixed precision: bfloat16

**Dataset Information:**
- The dataset uses columns: `query`, `positive_verse`, `negative_verse`
- Training set: ~51,000 triplets
- Test set: ~5,700 triplets

**Common Options:**

```bash
uv run train-embeddings --help

# Use default optimized settings (MNRL loss)
uv run train-embeddings

# Increase batch size for faster training (MNRL uses less VRAM)
uv run train-embeddings --batch-size 96

# Adjust batch size if you get OOM errors
uv run train-embeddings --batch-size 32

# Train for more epochs
uv run train-embeddings --epochs 5

# Use traditional TripletLoss (slower but proven)
uv run train-embeddings --loss triplet --batch-size 64

# Custom output directory
uv run train-embeddings --output-dir ./my-models/sanskrit-embeddings

# Push to HuggingFace Hub after training
uv run train-embeddings --push-to-hub --hub-model-id "your-username/embeddinggemma-sanskrit"

# Make the Hub model private
uv run train-embeddings --push-to-hub --hub-model-id "your-username/model" --hub-private

# Disable mixed precision (if compatibility issues)
uv run train-embeddings --no-amp

# Resume training from a checkpoint
uv run train-embeddings --resume-from-checkpoint ./models/embeddinggemma-sanskrit-ft

# Resume from a HuggingFace model
uv run train-embeddings --resume-from-checkpoint "your-username/embeddinggemma-sanskrit"
```

**Training Time:**
- **With MNRL (Recommended)**: ~30-50 minutes for 3 epochs with batch size 64
- **With MNRL + Larger Batch**: ~20-40 minutes for 3 epochs with batch size 96-128
- **With TripletLoss**: ~1-2 hours for 3 epochs
- RTX 3060 12GB VRAM

**What Happens During Training:**
1. Loads dataset with existing train/test splits from HuggingFace
2. Identifies triplet columns: `query`, `positive_verse`, `negative_verse`
3. Initializes EmbeddingGemma-300m with mixed precision (bfloat16)
4. Trains using selected loss function (MNRL by default) with task-specific prompts
5. Evaluates every 500 steps on test set
6. Saves best checkpoint based on triplet accuracy
7. Optionally pushes to HuggingFace Hub

**Performance Optimizations:**
- **MNRL (MultipleNegativesRankingLoss)**: Default loss function, 2-3x faster than TripletLoss
  - Encodes only 2N samples per batch instead of 3N
  - Uses in-batch negatives automatically (all other samples in batch serve as negatives)
  - Lower VRAM usage allows larger batch sizes (96-128 vs 64 for TripletLoss)
  - Better scaling with larger batch sizes for improved convergence

**Evaluation Metric:**
The model is evaluated using **triplet accuracy**: the percentage of test triplets where the positive example is closer to the anchor than the negative example. Higher is better (aim for 85%+).

## Using the Fine-tuned Model

After training completes, use your fine-tuned model:

```python
from sentence_transformers import SentenceTransformer

# Load the fine-tuned model
model = SentenceTransformer("./models/embeddinggemma-sanskrit-ft")

# Or load from HuggingFace Hub (if you pushed it)
# model = SentenceTransformer("your-username/embeddinggemma-sanskrit")

# IMPORTANT: EmbeddingGemma requires task-specific prompts for best performance
query_prompt = "task: search result | query: "
doc_prompt = "title: none | text: "

# Encode Sanskrit texts for similarity comparison (use doc prompt for both)
texts = [
    doc_prompt + "धर्मक्षेत्रे कुरुक्षेत्रे",
    doc_prompt + "भगवद्गीता महाभारतस्य भागः"
]
embeddings = model.encode(texts)

# Compute similarity
similarity = model.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")

# Search/retrieval example (most common use case)
query = "गीता"
documents = [
    "भगवद्गीता महाभारतस्य भागः",
    "रामायणं वाल्मीकेः कृतम्",
    "वेदः प्राचीनतमः ग्रन्थः"
]

# Add appropriate prompts: query prompt for search, doc prompt for documents
query_embedding = model.encode(query_prompt + query)
doc_embeddings = model.encode([doc_prompt + doc for doc in documents])

# Get similarities and rank
similarities = model.similarity(query_embedding, doc_embeddings)
ranked_indices = similarities.argsort(descending=True)

print("Search results:")
for idx in ranked_indices:
    print(f"  {documents[idx]}: {similarities[idx]:.4f}")
```

## Project Structure

```
rgveda-ft/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── .gitignore                 # Git ignore rules
├── app/
│   ├── __init__.py           # Package initialization
│   ├── cli.py                # CLI entry points
│   ├── data_preparation.py   # Dataset splitting and loading
│   └── train.py              # Training pipeline
└── models/                    # Output directory for trained models
    └── embeddinggemma-sanskrit-ft/  # Fine-tuned model
```

## Performance Tuning

### Training Speed Optimization

If training seems slow, here are the key factors and solutions:

**1. Batch Size (Most Important)**
```bash
# Increase batch size for faster training (MNRL allows larger batches)
uv run train-embeddings --batch-size 96  # or 128
```
- Larger batches = fewer steps per epoch = faster training
- MNRL uses less VRAM, allowing batch sizes of 96-128 on RTX 3060
- TripletLoss limited to batch size 64

**2. Data Loading (Now Optimized)**
- ✅ **Parallel workers**: Uses 4 workers on Linux/Mac (Windows uses single-threaded for compatibility)
- ✅ **Pin memory**: Enabled for faster GPU transfer
- ✅ **Data caching**: Preprocessed examples cached to disk (`.cache/` directory)
  - First run: Creates cache
  - Subsequent runs: Instant loading from cache

**3. Model Encoding Speed**
- The model encoding is the main bottleneck (not data loading)
- MNRL encodes 2N samples vs 3N for TripletLoss = **33% faster**
- bfloat16 precision provides ~1.5-2x speedup vs float32

**4. Expected Training Times (RTX 3060 12GB)**
- MNRL + batch 64: ~30-50 minutes for 3 epochs
- MNRL + batch 96: ~25-35 minutes for 3 epochs
- MNRL + batch 128: ~20-30 minutes for 3 epochs
- TripletLoss + batch 64: ~1-2 hours for 3 epochs

**5. Diagnostic Commands**
```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Check if GPU is being utilized (should be 90-100%)
# If GPU usage is low, increase batch size
```

**Common Issues:**
- **Low GPU utilization (<50%)**: Increase batch size
- **OOM errors**: Decrease batch size
- **Slow first epoch**: Normal - data is being cached for subsequent epochs

## Troubleshooting

### CUDA Not Available

If the setup test shows "CUDA not available":

1. **Check NVIDIA drivers**: Run `nvidia-smi` to verify your GPU is detected
2. **Verify PyTorch CUDA installation**: Run `uv run python -c "import torch; print(torch.cuda.is_available())"`
3. **Reinstall with CUDA support**: The project is configured to use CUDA 11.8 by default. If you need a different version:
   - Edit `pyproject.toml` and change the `[[tool.uv.index]]` URL (e.g., `cu121` for CUDA 12.1)
   - Delete `uv.lock`
   - Run `uv sync`

### Out of Memory (OOM) Errors

If you get CUDA out of memory errors, reduce the batch size:

```bash
uv run train-embeddings --batch-size 32
# or even lower
uv run train-embeddings --batch-size 16
```

### Dataset Not Split

If you see "Dataset does not have train/test splits", run:

```bash
uv run split-dataset
```

### Model Download Issues

If EmbeddingGemma download fails:
1. Ensure you've accepted the license: https://huggingface.co/google/embeddinggemma-300m
2. Verify authentication: `uv run huggingface-cli whoami`
3. Check internet connection

### Mixed Precision Issues

If you encounter bfloat16 compatibility issues:

```bash
uv run train-embeddings --no-amp
```

This uses full float32 precision (uses more memory but more compatible).

## Training Optimization Guide

### Loss Functions

The training pipeline supports two loss functions:

#### 1. **MNRL (MultipleNegativesRankingLoss)** - Recommended ⚡

**Default and recommended** for faster training and better performance.

```bash
uv run train-embeddings --loss mnrl --batch-size 64
```

**How it works:**
- Only requires (anchor, positive) pairs
- Uses all other samples in the batch as negatives automatically
- Encodes 2N samples per batch instead of 3N

**Benefits:**
- **2-3x faster** than TripletLoss
- **Better scaling** with larger batch sizes
- **Lower VRAM usage** - can use larger batches
- **State-of-the-art** performance on retrieval tasks

**Best for:** Fast training, large batch sizes, production use

#### 2. **TripletLoss** - Traditional

Classic triplet-based training with explicit negatives.

```bash
uv run train-embeddings --loss triplet --batch-size 64
```

**How it works:**
- Uses explicit (anchor, positive, negative) triplets
- Encodes 3N samples per batch
- Calculates distances for all triplets

**Benefits:**
- **Well-established** and proven approach
- **Direct control** over negative samples

**Best for:** Research, comparison studies, specific use cases

### Batch Size Optimization

MNRL's lower memory usage allows you to use larger batch sizes for faster training:

```bash
# Default batch size
uv run train-embeddings --batch-size 64

# Larger batch size for faster training (recommended for RTX 3060)
uv run train-embeddings --batch-size 96

# Maximum batch size for RTX 3060 12GB
uv run train-embeddings --batch-size 128
```

**Benefits of larger batch sizes with MNRL:**
- **More in-batch negatives**: Batch size of 128 provides 127 negative samples per anchor
- **Faster training**: Fewer steps per epoch
- **Better convergence**: Stronger training signal from more diverse negatives
- **No extra cost**: MNRL's efficiency makes this possible

**Recommended settings for RTX 3060 12GB:**
- MNRL: `--batch-size 96` or `--batch-size 128` (optimal) **← Recommended**
- TripletLoss: `--batch-size 64` (maximum for 12GB VRAM)

### Optimal Training Command

For best speed and performance on RTX 3060:

```bash
# Recommended: MNRL with larger batch size
uv run train-embeddings --batch-size 96

# Maximum performance (if VRAM allows)
uv run train-embeddings --batch-size 128

# Or use defaults (MNRL with batch size 64)
uv run train-embeddings
```

This gives you:
- ⚡ **2-3x faster training** than TripletLoss
- 💪 **Larger batch sizes** (96-128) for better convergence
- 🎯 **State-of-the-art performance** on Sanskrit embeddings
- ⏱️ **~20-40 minutes** total training time with batch size 96-128

## Advanced Usage

### Creating Custom Train/Test Splits (Optional)

The Sanskrit triplets dataset already includes train/test splits. However, if you want to create custom splits with different ratios, you can use:

```bash
uv run split-dataset --help

# Create a custom split (e.g., 80/20)
uv run split-dataset --test-size 0.2

# Only split locally without pushing to HuggingFace
uv run split-dataset --skip-push
```

**Note:** This will overwrite the existing splits if you push to HuggingFace.

### Training with Custom Dataset

To use a different dataset, it must have triplet structure (anchor, positive, negative) or (query, positive, negative):

```bash
uv run train-embeddings --dataset "your-username/your-triplet-dataset"
```

### Resuming Training from Checkpoint

You can resume training from a previously saved checkpoint or fine-tuned model:

**From a local checkpoint:**
```bash
uv run train-embeddings --resume-from-checkpoint ./models/embeddinggemma-sanskrit-ft
```

**From a HuggingFace model:**
```bash
uv run train-embeddings --resume-from-checkpoint "your-username/embeddinggemma-sanskrit"
```

**Use cases:**
- **Continue interrupted training**: If training was stopped, resume from the last saved checkpoint
- **Further fine-tuning**: Take an already fine-tuned model and train it more
- **Transfer learning**: Start from a related fine-tuned model instead of the base model

**Example - Train for 3 more epochs:**
```bash
# Initial training
uv run train-embeddings --epochs 3

# Continue for 3 more epochs from the checkpoint
uv run train-embeddings --resume-from-checkpoint ./models/embeddinggemma-sanskrit-ft --epochs 3
```

**Note:** When resuming, the model architecture and configuration are loaded from the checkpoint, but you can still adjust training hyperparameters like batch size, learning rate warmup, and number of epochs.

### Task-Specific Prompts (IMPORTANT!)

**EmbeddingGemma was trained with task-specific prompts and performs significantly better when you use them.** The fine-tuning pipeline automatically adds these prompts during training, and you **must** use them when encoding text with your fine-tuned model.

**Prompts used during training:**
- **Queries/Searches**: `"task: search result | query: "`
- **Documents**: `"title: none | text: "`

**Usage examples:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./models/embeddinggemma-sanskrit-ft")

# Define the same prompts used during training
query_prompt = "task: search result | query: "
doc_prompt = "title: none | text: "

# For search/retrieval (most common):
query = "भगवद्गीता के बारे में"
documents = ["भगवद्गीता महाभारतस्य भागः", "रामायणं वाल्मीकेः कृतम्"]

query_emb = model.encode(query_prompt + query)
doc_embs = model.encode([doc_prompt + doc for doc in documents])

similarities = model.similarity(query_emb, doc_embs)

# For document-to-document similarity (use doc prompt for both):
doc1 = doc_prompt + "धर्मक्षेत्रे कुरुक्षेत्रे"
doc2 = doc_prompt + "भगवद्गीता महाभारतस्य भागः"
similarity = model.similarity(model.encode(doc1), model.encode(doc2))
```

**Other available task prompts from EmbeddingGemma:**
- Question Answering: `"task: question answering | query: "`
- Fact Verification: `"task: fact checking | query: "`
- Classification: `"task: classification | query: "`
- Clustering: `"task: clustering | query: "`
- Semantic Similarity: `"task: sentence similarity | query: "`
- Code Retrieval: `"task: code retrieval | query: "`

See the [EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m#prompt-instructions) for complete details.

## Technical Details

### Training Configuration

- **Loss Function**: MultipleNegativesRankingLoss (MNRL) by default, TripletLoss optional
  - MNRL: Uses in-batch negatives, 2-3x faster
  - TripletLoss: Explicit negatives with margin 0.5
- **Optimizer**: AdamW (default from sentence-transformers)
- **Learning Rate Schedule**: Warmup for 10% of steps, then linear decay
- **Precision**: bfloat16 mixed precision (float32 for embeddings)
- **Batch Size**: 64 (default), 96-128 recommended for MNRL on RTX 3060
- **Epochs**: 3 (default, adjustable)
- **Evaluation**: Every 500 steps using TripletEvaluator
- **Task Prompts**: Automatically adds EmbeddingGemma-specific prompts
  - Anchors: `"task: search result | query: "`
  - Documents: `"title: none | text: "`

### Model Architecture

- **Base**: EmbeddingGemma-300m (Gemma 3 with T5Gemma initialization)
- **Parameters**: 300 million
- **Embedding Dimension**: 768 (supports MRL truncation to 512, 256, 128)
- **Context Length**: 2048 tokens
- **Training**: Full fine-tune (all parameters)

## References

- [EmbeddingGemma Paper](https://arxiv.org/abs/2509.20354)
- [EmbeddingGemma Model Card](https://huggingface.co/google/embeddinggemma-300m)
- [Sanskrit Triplets Dataset](https://huggingface.co/datasets/indhic-ai/sanskrit-triplets)
- [Sentence Transformers Documentation](https://www.sbert.net/)

## License

This project uses:
- **EmbeddingGemma**: Gemma License (see https://ai.google.dev/gemma/terms)
- **Sanskrit Triplets Dataset**: Check dataset repository for license
- **This Code**: MIT License

## Citation

If you use this fine-tuned model, please cite:

```bibtex
@article{embedding_gemma_2025,
    title={Sanskrit EmbeddingGemma},
    author={Ganaraj Permunda & Bharat Shetty},
    publisher={Indhic AI},
    year={2025}
}
```

## Support

For issues related to:
- **This pipeline**: Open an issue in this repository
- **EmbeddingGemma model**: See https://huggingface.co/google/embeddinggemma-300m
- **Sanskrit dataset**: See https://huggingface.co/datasets/indhic-ai/sanskrit-triplets
- **sentence-transformers**: See https://github.com/UKPLab/sentence-transformers

