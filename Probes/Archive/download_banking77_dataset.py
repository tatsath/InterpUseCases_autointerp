#!/usr/bin/env python3
"""
Download and Convert Banking77 Dataset
Downloads the Banking77 dataset and converts it to the same format as other datasets
"""

import pandas as pd
from datasets import load_dataset
import os
import json

def download_and_convert_banking77():
    """Download Banking77 dataset and convert to standard format"""
    
    print("🔄 Downloading Banking77 dataset...")
    
    # Load the dataset
    dataset = load_dataset("mteb/banking77")
    
    print(f"✅ Dataset loaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Get number of classes
    unique_labels = set(item['label'] for item in dataset['train'])
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    print(f"Label range: {min(unique_labels)} to {max(unique_labels)}")
    
    # Convert to our format
    def convert_to_standard_format(split_name):
        data = dataset[split_name]
        converted_data = []
        
        for item in data:
            # Banking77 format: {'text': '...', 'label': 0-76}
            # Our format: {'statement': '...', 'label': 0/1}
            
            # For Banking77, we'll use the text as statement and convert label to binary
            # We'll use a simple approach: even labels = 1, odd labels = 0
            # Or we can use: label < 39 = 1, label >= 39 = 0
            
            binary_label = 1 if item['label'] < 39 else 0  # Split classes roughly in half
            
            converted_data.append({
                'statement': item['text'],
                'label': binary_label
            })
        
        return converted_data
    
    # Convert both splits
    train_data = convert_to_standard_format('train')
    test_data = convert_to_standard_format('test')
    
    print(f"✅ Converted to standard format!")
    print(f"Train: {len(train_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Check label distribution
    train_labels = [item['label'] for item in train_data]
    test_labels = [item['label'] for item in test_data]
    
    train_pos = sum(train_labels)
    test_pos = sum(test_labels)
    
    print(f"Train - Positive: {train_pos}, Negative: {len(train_data) - train_pos}")
    print(f"Test - Positive: {test_pos}, Negative: {len(test_data) - test_pos}")
    
    # Save as CSV files
    output_dir = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/Probe/LLMProbe/datasets"
    
    # Save training data
    train_df = pd.DataFrame(train_data)
    train_file = os.path.join(output_dir, "banking77_train.csv")
    train_df.to_csv(train_file, index=False)
    print(f"✅ Saved training data to: {train_file}")
    
    # Save test data
    test_df = pd.DataFrame(test_data)
    test_file = os.path.join(output_dir, "banking77_test.csv")
    test_df.to_csv(test_file, index=False)
    print(f"✅ Saved test data to: {test_file}")
    
    # Save combined data
    all_data = train_data + test_data
    all_df = pd.DataFrame(all_data)
    all_file = os.path.join(output_dir, "banking77_combined.csv")
    all_df.to_csv(all_file, index=False)
    print(f"✅ Saved combined data to: {all_file}")
    
    # Save metadata
    metadata = {
        "dataset_name": "banking77",
        "original_classes": num_classes,
        "label_range": f"{min(unique_labels)} to {max(unique_labels)}",
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "total_samples": len(all_data),
        "binary_mapping": "label < 39 = 1, label >= 39 = 0",
        "description": "Banking77 dataset converted to binary classification"
    }
    
    metadata_file = os.path.join(output_dir, "banking77_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to: {metadata_file}")
    
    # Show sample data
    print(f"\n📋 Sample Data:")
    print(f"Statement: {all_data[0]['statement']}")
    print(f"Label: {all_data[0]['label']}")
    print(f"Statement: {all_data[1]['statement']}")
    print(f"Label: {all_data[1]['label']}")
    
    return {
        "train_file": train_file,
        "test_file": test_file,
        "combined_file": all_file,
        "metadata_file": metadata_file,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "total_samples": len(all_data)
    }

def create_banking77_subsets():
    """Create smaller subsets for testing"""
    
    print("\n🔄 Creating Banking77 subsets...")
    
    # Load the combined data
    combined_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/Probe/LLMProbe/datasets/banking77_combined.csv"
    df = pd.read_csv(combined_file)
    
    # Create 10-sample subset
    subset_10 = df.sample(n=min(10, len(df)), random_state=42)
    subset_10_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/Probe/LLMProbe/datasets/banking77_10.csv"
    subset_10.to_csv(subset_10_file, index=False)
    print(f"✅ Created 10-sample subset: {subset_10_file}")
    
    # Create 100-sample subset
    subset_100 = df.sample(n=min(100, len(df)), random_state=42)
    subset_100_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/Probe/LLMProbe/datasets/banking77_100.csv"
    subset_100.to_csv(subset_100_file, index=False)
    print(f"✅ Created 100-sample subset: {subset_100_file}")
    
    # Create 1000-sample subset
    subset_1000 = df.sample(n=min(1000, len(df)), random_state=42)
    subset_1000_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Probes/Probe/LLMProbe/datasets/banking77_1000.csv"
    subset_1000.to_csv(subset_1000_file, index=False)
    print(f"✅ Created 1000-sample subset: {subset_1000_file}")
    
    return {
        "subset_10": subset_10_file,
        "subset_100": subset_100_file,
        "subset_1000": subset_1000_file
    }

if __name__ == "__main__":
    print("🚀 Banking77 Dataset Download and Conversion")
    print("=" * 60)
    
    # Download and convert
    results = download_and_convert_banking77()
    
    # Create subsets
    subsets = create_banking77_subsets()
    
    print(f"\n✅ All done!")
    print(f"📁 Files created:")
    print(f"  - Training: {results['train_file']}")
    print(f"  - Test: {results['test_file']}")
    print(f"  - Combined: {results['combined_file']}")
    print(f"  - Metadata: {results['metadata_file']}")
    print(f"  - Subset 10: {subsets['subset_10']}")
    print(f"  - Subset 100: {subsets['subset_100']}")
    print(f"  - Subset 1000: {subsets['subset_1000']}")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Total samples: {results['total_samples']}")
    print(f"  - Train samples: {results['train_samples']}")
    print(f"  - Test samples: {results['test_samples']}")
