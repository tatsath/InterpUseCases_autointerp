import random
from collections import Counter

def count_categories(examples):
    """Count the number of examples in each category based on label value

    Args:
        examples: List of examples, each with a 'label' field

    Returns:
        dict: Dictionary with label values as keys and counts as values
    """
    labels = [ex['label'] for ex in examples]
    return Counter(labels)

def load_multi_class_dataset(dataset_source, progress_callback, max_samples=5000, custom_file=None, tf_splits=None):
    """Load multi-class dataset with progress updates"""
    examples = []

    if dataset_source == "custom" and custom_file is not None:
        progress_callback(0.1, "Loading custom multi-class dataset...",
                          "Processing uploaded CSV file")
        try:
            import pandas as pd
            custom_df = pd.read_csv(custom_file)

            if 'statement' not in custom_df.columns or 'label' not in custom_df.columns:
                progress_callback(1.0, "Error: CSV must contain 'statement' and 'label' columns",
                                  "Please check your CSV format and try again")
                return []

            # Clean and validate data
            custom_df = custom_df.dropna(subset=['statement', 'label'])

            # Check for multi-class labels (more than 2 unique labels)
            unique_labels = sorted(custom_df['label'].unique())
            num_classes = len(unique_labels)
            
            if num_classes < 2:
                progress_callback(1.0, "Error: Dataset must have at least 2 classes",
                                  "Please check your labels and try again")
                return []
            
            progress_callback(0.2, f"Detected {num_classes} classes",
                              f"Classes: {unique_labels}")

            # Process each row
            for idx, row in enumerate(custom_df.itertuples()):
                if idx % 10 == 0:
                    progress = 0.2 + (idx / len(custom_df)) * 0.8
                    progress_callback(progress, f"Processing custom example {idx+1}/{len(custom_df)}",
                                      f"Statement: {row.statement[:50]}...")

                examples.append({
                    "text": row.statement,
                    "label": int(row.label)
                })

                # Limit dataset size if needed
                if len(examples) >= max_samples:
                    break

            # Show class distribution
            label_counts = count_categories(examples)
            progress_callback(0.9, f"Class distribution: {dict(label_counts)}",
                              f"Total examples: {len(examples)}")

            progress_callback(1.0, f"Loaded multi-class dataset: {len(examples)} examples",
                              f"Classes: {num_classes}, Distribution: {dict(label_counts)}")

            return examples

        except Exception as e:
            progress_callback(1.0, f"Error loading custom dataset: {str(e)}",
                              "Please check your CSV format and try again")
            return []

    # Handle other dataset sources (HuggingFace, etc.)
    else:
        progress_callback(1.0, "Multi-class dataset loading not implemented for this source",
                          "Please use a custom CSV file with multi-class labels")
        return []
