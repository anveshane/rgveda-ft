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

This will install all dependencies defined in `pyproject.toml`.

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

### Step 1: Split Dataset

The original Sanskrit triplets dataset isn't split into train/test sets. Run this command once to create a 90/10 split and push it back to HuggingFace:

```bash
uv run split-dataset
```

**Options:**
```bash
uv run split-dataset --help

# Use a different test size (e.g., 20%)
uv run split-dataset --test-size 0.2

# Only split locally without pushing (for testing)
uv run split-dataset --skip-push
```

**Output:**
- Training set: ~52,000 triplets
- Test set: ~5,600 triplets

### Step 2: Fine-tune Model

Train EmbeddingGemma-300m on the Sanskrit triplets:

```bash
uv run train-embeddings
```

**Common Options:**

```bash
uv run train-embeddings --help

# Adjust batch size if you get OOM errors
uv run train-embeddings --batch-size 32

# Train for more epochs
uv run train-embeddings --epochs 5

# Custom output directory
uv run train-embeddings --output-dir ./my-models/sanskrit-embeddings

# Push to HuggingFace Hub after training
uv run train-embeddings --push-to-hub --hub-model-id "your-username/embeddinggemma-sanskrit"

# Make the Hub model private
uv run train-embeddings --push-to-hub --hub-model-id "your-username/model" --hub-private

# Disable mixed precision (if compatibility issues)
uv run train-embeddings --no-amp
```

**Training Time:**
- Expected: **30 minutes to 2 hours** (depending on batch size and epochs)
- RTX 3060 with batch size 64: ~1 hour for 3 epochs

**What Happens During Training:**
1. Loads pre-split dataset from HuggingFace
2. Initializes EmbeddingGemma-300m with mixed precision (bfloat16)
3. Trains using TripletLoss
4. Evaluates every 500 steps on test set
5. Saves best checkpoint based on triplet accuracy
6. Optionally pushes to HuggingFace Hub

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

## Troubleshooting

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

## Advanced Usage

### Training with Custom Dataset

To use a different dataset, it must have triplet structure (anchor, positive, negative):

```bash
uv run train-embeddings --dataset "your-username/your-triplet-dataset"
```

### Resuming Training

Currently, the pipeline trains from scratch. To resume training:

```python
from sentence_transformers import SentenceTransformer
from app.train import train_embedding_model

# Load your checkpoint
model = SentenceTransformer("./models/embeddinggemma-sanskrit-ft")

# Continue training with the loaded model
# You'll need to modify train.py to accept a pre-loaded model
```

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

- **Loss Function**: TripletLoss with default margin (0.5)
- **Optimizer**: AdamW (default from sentence-transformers)
- **Learning Rate Schedule**: Warmup for 10% of steps, then linear decay
- **Precision**: bfloat16 mixed precision (float32 for embeddings)
- **Batch Size**: 64 (adjustable)
- **Epochs**: 3 (adjustable)
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
    title={EmbeddingGemma: Powerful and Lightweight Text Representations},
    author={Schechter Vera, Henrique and others},
    publisher={Google DeepMind},
    year={2025},
    url={https://arxiv.org/abs/2509.20354}
}
```

## Support

For issues related to:
- **This pipeline**: Open an issue in this repository
- **EmbeddingGemma model**: See https://huggingface.co/google/embeddinggemma-300m
- **Sanskrit dataset**: See https://huggingface.co/datasets/indhic-ai/sanskrit-triplets
- **sentence-transformers**: See https://github.com/UKPLab/sentence-transformers

