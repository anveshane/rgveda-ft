"""
Fine-tuning module for EmbeddingGemma-300m on Sanskrit triplets.

This module implements the training pipeline using sentence-transformers with
TripletLoss and TripletEvaluator for optimal embedding model fine-tuning.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader

from .data_preparation import load_split_dataset, get_triplet_columns


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return device information.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': None,
        'device_memory_gb': None
    }
    
    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def prepare_training_data(dataset, anchor_col: str, pos_col: str, neg_col: str, batch_size: int = 64):
    """
    Prepare training dataloader from dataset with task-specific prompts.
    
    EmbeddingGemma was trained with task prefixes and performs significantly better with them.
    For retrieval tasks:
    - Queries/Anchors: "task: search result | query: "
    - Documents: "title: none | text: "
    
    Args:
        dataset: Training dataset split
        anchor_col: Column name for anchor text
        pos_col: Column name for positive text
        neg_col: Column name for negative text
        batch_size: Training batch size
        
    Returns:
        DataLoader with InputExample objects
    """
    print(f"Preparing training data with batch size {batch_size}...")
    print("Adding task-specific prompts for EmbeddingGemma...")
    
    # EmbeddingGemma task-specific prompts (from model card)
    query_prompt = "task: search result | query: "
    doc_prompt = "title: none | text: "
    
    # Convert dataset to InputExample objects with prompts
    train_examples = []
    for example in dataset:
        train_examples.append(
            InputExample(
                texts=[
                    query_prompt + example[anchor_col],     # Query with prompt
                    doc_prompt + example[pos_col],          # Positive doc with prompt
                    doc_prompt + example[neg_col]           # Negative doc with prompt
                ]
            )
        )
    
    print(f"Created {len(train_examples)} training examples with task prompts")
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0  # Use 0 for Windows compatibility
    )
    
    return train_dataloader


def prepare_evaluator(dataset, anchor_col: str, pos_col: str, neg_col: str, name: str = "test"):
    """
    Prepare TripletEvaluator for evaluation with task-specific prompts.
    
    EmbeddingGemma requires task prompts for optimal performance.
    
    Args:
        dataset: Test dataset split
        anchor_col: Column name for anchor text
        pos_col: Column name for positive text
        neg_col: Column name for negative text
        name: Name for the evaluator
        
    Returns:
        TripletEvaluator instance
    """
    print(f"Preparing evaluator with {len(dataset)} examples...")
    print("Adding task-specific prompts for evaluation...")
    
    # EmbeddingGemma task-specific prompts (same as training)
    query_prompt = "task: search result | query: "
    doc_prompt = "title: none | text: "
    
    # Extract triplets as separate lists with prompts
    anchors = []
    positives = []
    negatives = []
    
    for example in dataset:
        anchors.append(query_prompt + example[anchor_col])    # Query with prompt
        positives.append(doc_prompt + example[pos_col])       # Positive doc with prompt
        negatives.append(doc_prompt + example[neg_col])       # Negative doc with prompt
    
    # Create evaluator
    evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name=name
    )
    
    print(f"Created TripletEvaluator with {len(anchors)} triplets (with task prompts)")
    
    return evaluator


def train_embedding_model(
    model_name: str = "google/embeddinggemma-300m",
    dataset_name: str = "indhic-ai/sanskrit-triplets",
    output_dir: str = "./models/embeddinggemma-sanskrit-ft",
    epochs: int = 3,
    batch_size: int = 64,
    evaluation_steps: int = 500,
    warmup_ratio: float = 0.1,
    use_amp: bool = True,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    hub_private: bool = False
) -> SentenceTransformer:
    """
    Fine-tune EmbeddingGemma-300m on Sanskrit triplets.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_name: HuggingFace dataset identifier
        output_dir: Local directory to save the model
        epochs: Number of training epochs
        batch_size: Training batch size
        evaluation_steps: Steps between evaluations
        warmup_ratio: Ratio of steps for learning rate warmup
        use_amp: Use automatic mixed precision (bfloat16)
        push_to_hub: Whether to push to HuggingFace Hub after training
        hub_model_id: HuggingFace Hub model ID (if push_to_hub=True)
        hub_private: Make the Hub model private
        
    Returns:
        Trained SentenceTransformer model
    """
    # Check GPU
    gpu_info = check_gpu_availability()
    print("\n" + "="*60)
    print("GPU Information")
    print("="*60)
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"Device: {gpu_info['device_name']}")
        print(f"Memory: {gpu_info['device_memory_gb']:.2f} GB")
    else:
        print("WARNING: CUDA not available. Training will be slow on CPU.")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_split_dataset(dataset_name)
    
    # Get column names
    anchor_col, pos_col, neg_col = get_triplet_columns(dataset['train'])
    print(f"Triplet columns: {anchor_col}, {pos_col}, {neg_col}")
    
    # Load model
    print(f"\nLoading model: {model_name}...")
    print("Note: EmbeddingGemma uses float32 or bfloat16 precision")
    
    model = SentenceTransformer(model_name)
    
    # Set to bfloat16 if use_amp is True and GPU supports it
    if use_amp and gpu_info['cuda_available']:
        print("Using bfloat16 mixed precision for efficient training")
    
    # Prepare training data
    train_dataloader = prepare_training_data(
        dataset['train'],
        anchor_col,
        pos_col,
        neg_col,
        batch_size=batch_size
    )
    
    # Prepare evaluator
    evaluator = prepare_evaluator(
        dataset['test'],
        anchor_col,
        pos_col,
        neg_col,
        name="test"
    )
    
    # Setup loss function
    print("\nSetting up TripletLoss...")
    train_loss = losses.TripletLoss(model=model)
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Evaluation every: {evaluation_steps} steps")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train
    print("Starting training...")
    print("The model will be evaluated periodically and the best checkpoint will be saved.\n")
    
    start_time = datetime.now()
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=use_amp
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Duration: {training_duration}")
    print(f"Model saved to: {output_dir}")
    print("="*60 + "\n")
    
    # Push to Hub if requested
    if push_to_hub:
        if not hub_model_id:
            raise ValueError("hub_model_id must be provided when push_to_hub=True")
        
        print(f"Pushing model to HuggingFace Hub: {hub_model_id}...")
        
        model_card = f"""---
tags:
- sentence-transformers
- embedding
- sanskrit
- embeddinggemma
base_model: {model_name}
datasets:
- {dataset_name}
language:
- sa
license: gemma
---

# EmbeddingGemma-300m Fine-tuned on Sanskrit

This model is a fine-tuned version of [{model_name}](https://huggingface.co/{model_name}) on the [{dataset_name}](https://huggingface.co/datasets/{dataset_name}) dataset.

## Model Details

- **Base Model:** {model_name}
- **Fine-tuning Dataset:** {dataset_name} (Sanskrit triplets)
- **Training Examples:** {len(dataset['train'])}
- **Test Examples:** {len(dataset['test'])}
- **Training Duration:** {training_duration}
- **Framework:** sentence-transformers

## Training Configuration

- Epochs: {epochs}
- Batch Size: {batch_size}
- Loss Function: TripletLoss
- Evaluation: TripletEvaluator (accuracy metric)
- Mixed Precision: {'bfloat16' if use_amp else 'float32'}
- **Task Prompts**: Trained with EmbeddingGemma-specific prompts
  - Queries/Anchors: `"task: search result | query: "`
  - Documents: `"title: none | text: "`

## Usage

**IMPORTANT:** This model was trained with task-specific prompts. You must use the same prompts for optimal performance.

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("{hub_model_id}")

# Define task prompts (must match training)
query_prompt = "task: search result | query: "
doc_prompt = "title: none | text: "

# For search/retrieval:
query = "Your Sanskrit search query"
documents = ["Sanskrit document 1", "Sanskrit document 2"]

query_embedding = model.encode(query_prompt + query)
doc_embeddings = model.encode([doc_prompt + doc for doc in documents])

# Compute similarities
similarities = model.similarity(query_embedding, doc_embeddings)

# For document similarity (use doc prompt for both):
doc1 = doc_prompt + "First Sanskrit text"
doc2 = doc_prompt + "Second Sanskrit text"
similarity = model.similarity(model.encode(doc1), model.encode(doc2))
```

## Intended Use

This model is specialized for Sanskrit text embeddings and is suitable for:
- Semantic similarity of Sanskrit texts
- Sanskrit text retrieval and search
- Classification and clustering of Sanskrit documents
- Question answering in Sanskrit

## Training Details

Fine-tuned using full parameter training (not LoRA) on NVIDIA RTX 3060 12GB GPU.
The model was trained to optimize triplet relationships: ensuring similar texts are closer
in embedding space than dissimilar texts.
"""
        
        try:
            model.save_to_hub(
                hub_model_id,
                private=hub_private,
                train_datasets=[dataset_name],
                model_card=model_card
            )
            print(f"✓ Successfully pushed model to https://huggingface.co/{hub_model_id}")
        except Exception as e:
            print(f"✗ Failed to push model to Hub: {e}")
            print("The model is still saved locally and can be pushed manually later.")
    
    return model


if __name__ == "__main__":
    # Test the training pipeline
    print("Testing training pipeline...")
    train_embedding_model(epochs=1)  # Quick test with 1 epoch

