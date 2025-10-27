"""
Dataset preparation module for Sanskrit triplets.

This module handles loading, splitting, and pushing the Sanskrit triplets dataset
to HuggingFace Hub.
"""

import os
from typing import Tuple
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import HfApi


# Default dataset configuration
DEFAULT_DATASET_NAME = "indhic-ai/sanskrit-triplets"


def check_hf_authentication() -> bool:
    """
    Check if user is authenticated with HuggingFace.
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    try:
        api = HfApi()
        api.whoami()
        return True
    except Exception:
        return False


def load_and_split_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    test_size: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    """
    Load the Sanskrit triplets dataset and split it into train/test sets.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        test_size: Proportion of dataset to use for testing (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    print(f"Loading dataset: {dataset_name}...")
    
    # Load the original dataset
    dataset = load_dataset(dataset_name)
    
    # The dataset might be loaded as a dict with a single split
    # Get the actual data (usually in 'train' split or the only split available)
    if isinstance(dataset, DatasetDict):
        # If already split, get the first available split
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
    else:
        data = dataset
    
    print(f"Total examples: {len(data)}")
    print(f"Dataset features: {data.features}")
    
    # Verify the dataset has the expected structure
    # Triplet datasets typically have columns like: anchor, positive, negative
    # or query, positive, negative, or similar variations
    print("\nFirst example:")
    print(data[0])
    
    # Split the dataset
    print(f"\nSplitting dataset: {1-test_size:.0%} train, {test_size:.0%} test...")
    split_dataset = data.train_test_split(test_size=test_size, seed=seed)
    
    print(f"Training examples: {len(split_dataset['train'])}")
    print(f"Test examples: {len(split_dataset['test'])}")
    
    return split_dataset


def push_split_dataset(
    split_dataset: DatasetDict,
    repo_id: str = DEFAULT_DATASET_NAME,
    private: bool = False
) -> None:
    """
    Push the split dataset back to HuggingFace Hub.
    
    Args:
        split_dataset: DatasetDict with train/test splits
        repo_id: HuggingFace repository ID
        private: Whether to make the repo private
    """
    if not check_hf_authentication():
        raise ValueError(
            "Not authenticated with HuggingFace. Please run: huggingface-cli login"
        )
    
    print(f"\nPushing split dataset to {repo_id}...")
    
    try:
        split_dataset.push_to_hub(
            repo_id,
            private=private,
            commit_message="Add train/test split (90/10)"
        )
        print(f"✓ Successfully pushed dataset to {repo_id}")
        print(f"  - Train split: {len(split_dataset['train'])} examples")
        print(f"  - Test split: {len(split_dataset['test'])} examples")
    except Exception as e:
        print(f"✗ Failed to push dataset: {e}")
        raise


def load_split_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME
) -> DatasetDict:
    """
    Load the dataset from HuggingFace Hub.
    
    If the dataset already has train/test splits, use them directly.
    Otherwise, raise an error with instructions.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        
    Returns:
        DatasetDict with 'train' and 'test' splits
    """
    print(f"Loading dataset from {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    # Check if the dataset has train/test splits
    if 'train' in dataset and 'test' in dataset:
        print(f"✓ Loaded dataset with {len(dataset['train'])} train and {len(dataset['test'])} test examples")
        return dataset
    elif 'train' in dataset and 'validation' in dataset:
        # Some datasets use 'validation' instead of 'test'
        print(f"✓ Loaded dataset with {len(dataset['train'])} train and {len(dataset['validation'])} validation examples")
        print("  Note: Using 'validation' split as test set")
        return DatasetDict({
            'train': dataset['train'],
            'test': dataset['validation']
        })
    else:
        raise ValueError(
            f"Dataset {dataset_name} does not have train/test splits. "
            f"Available splits: {list(dataset.keys())}\n"
            "Please run 'uv run split-dataset' to create splits, or use a dataset with existing splits."
        )


def get_triplet_columns(dataset: Dataset) -> Tuple[str, str, str]:
    """
    Identify the column names for anchor, positive, and negative in the dataset.
    
    Args:
        dataset: A dataset split
        
    Returns:
        Tuple of (anchor_col, positive_col, negative_col)
    """
    columns = dataset.column_names
    
    # Common naming patterns
    if 'anchor' in columns and 'positive' in columns and 'negative' in columns:
        return 'anchor', 'positive', 'negative'
    elif 'query' in columns and 'positive_verse' in columns and 'negative_verse' in columns:
        return 'query', 'positive_verse', 'negative_verse'
    elif 'query' in columns and 'positive' in columns and 'negative' in columns:
        return 'query', 'positive', 'negative'
    elif 'sentence1' in columns and 'sentence2' in columns and 'sentence3' in columns:
        return 'sentence1', 'sentence2', 'sentence3'
    else:
        raise ValueError(
            f"Could not identify triplet columns. Available columns: {columns}\n"
            "Expected patterns: (anchor, positive, negative), (query, positive, negative), or (query, positive_verse, negative_verse)"
        )


if __name__ == "__main__":
    # Test the module
    split_dataset = load_and_split_dataset()
    print("\nDataset split completed successfully!")
    
    # Show column structure
    anchor_col, pos_col, neg_col = get_triplet_columns(split_dataset['train'])
    print(f"\nTriplet columns: {anchor_col}, {pos_col}, {neg_col}")

