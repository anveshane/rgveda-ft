"""
CLI entry points for dataset preparation and model training.
"""

import sys
import argparse
from pathlib import Path

from .data_preparation import (
    load_and_split_dataset,
    push_split_dataset,
    check_hf_authentication
)
from .train import train_embedding_model


def split_dataset():
    """
    CLI command to split the Sanskrit triplets dataset and push to HuggingFace.
    
    Usage: uv run split-dataset
    """
    parser = argparse.ArgumentParser(
        description="Split Sanskrit triplets dataset into train/test and push to HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="indhic-ai/sanskrit-triplets",
        help="HuggingFace dataset ID (default: indhic-ai/sanskrit-triplets)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Proportion of data for test set (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on HuggingFace"
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip pushing to HuggingFace (only split locally for testing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Sanskrit Triplets Dataset Splitter")
    print("="*60 + "\n")
    
    # Check authentication if not skipping push
    if not args.skip_push:
        if not check_hf_authentication():
            print("✗ ERROR: Not authenticated with HuggingFace")
            print("\nPlease authenticate by running:")
            print("  huggingface-cli login")
            print("\nOr set HF_TOKEN environment variable")
            sys.exit(1)
        print("✓ HuggingFace authentication verified\n")
    
    try:
        # Load and split dataset
        split_dataset_obj = load_and_split_dataset(
            dataset_name=args.dataset,
            test_size=args.test_size,
            seed=args.seed
        )
        
        # Push to HuggingFace if not skipped
        if not args.skip_push:
            push_split_dataset(
                split_dataset_obj,
                repo_id=args.dataset,
                private=args.private
            )
            print("\n✓ Dataset successfully split and pushed!")
            print(f"  View at: https://huggingface.co/datasets/{args.dataset}")
        else:
            print("\n✓ Dataset split created (not pushed to HuggingFace)")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)


def train_embeddings():
    """
    CLI command to fine-tune EmbeddingGemma-300m on Sanskrit triplets.
    
    Usage: uv run train-embeddings
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune EmbeddingGemma-300m on Sanskrit triplets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Base model to fine-tune (default: google/embeddinggemma-300m)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="indhic-ai/sanskrit-triplets",
        help="HuggingFace dataset ID (default: indhic-ai/sanskrit-triplets)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/embeddinggemma-sanskrit-ft",
        help="Local directory to save model (default: ./models/embeddinggemma-sanskrit-ft)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64, reduce if OOM)"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluation frequency in steps (default: 500)"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (use full float32)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        help="HuggingFace Hub model ID (required if --push-to-hub)"
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make Hub model private"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("EmbeddingGemma-300m Fine-tuning for Sanskrit")
    print("="*60 + "\n")
    
    # Validate arguments
    if args.push_to_hub and not args.hub_model_id:
        print("✗ ERROR: --hub-model-id is required when --push-to-hub is set")
        sys.exit(1)
    
    if args.push_to_hub and not check_hf_authentication():
        print("✗ ERROR: Not authenticated with HuggingFace")
        print("\nPlease authenticate by running:")
        print("  huggingface-cli login")
        sys.exit(1)
    
    try:
        # Train model
        train_embedding_model(
            model_name=args.model,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            evaluation_steps=args.eval_steps,
            warmup_ratio=args.warmup_ratio,
            use_amp=not args.no_amp,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            hub_private=args.hub_private
        )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nModel saved locally at: {args.output_dir}")
        
        if args.push_to_hub:
            print(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")
        
        print("\nTo use the model:")
        print("```python")
        print("from sentence_transformers import SentenceTransformer")
        if args.push_to_hub:
            print(f"model = SentenceTransformer('{args.hub_model_id}')")
        else:
            print(f"model = SentenceTransformer('{args.output_dir}')")
        print("embeddings = model.encode(['Your Sanskrit text here'])")
        print("```")
        print()
        
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("This module provides CLI entry points.")
    print("Use 'uv run split-dataset' or 'uv run train-embeddings'")

