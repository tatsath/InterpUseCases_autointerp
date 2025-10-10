#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')

import torch
from probetrain.standalone_probe_system import load_dataset_from_csv, train_and_evaluate_multi_class

# Test with minimal data
print("Testing probetrain with minimal data...")

# Load dataset
dataset = load_dataset_from_csv('/home/nvidia/Documents/Hariom/probetrain/probetrain/sample_multi_class_dataset.csv', max_samples=5)
print(f"Loaded {len(dataset)} examples")

# Create dummy hidden states with correct shape
batch_size = len(dataset)
num_layers = 32  # Llama-2-7b has 32 layers
hidden_dim = 4096  # Llama-2-7b hidden dimension

# Create dummy hidden states
dummy_hidden_states = torch.randn(batch_size, num_layers, hidden_dim)
dummy_labels = torch.tensor([ex['label'] for ex in dataset])

print(f"Hidden states shape: {dummy_hidden_states.shape}")
print(f"Labels shape: {dummy_labels.shape}")
print(f"Unique labels: {torch.unique(dummy_labels)}")

# Test training on layer 16
try:
    results = train_and_evaluate_multi_class(
        dummy_hidden_states, dummy_labels,
        dummy_hidden_states, dummy_labels,  # Use same data for train/test
        num_layers=1,  # Only test one layer
        use_control_tasks=False,
        progress_callback=None,
        epochs=5,
        lr=0.01,
        device=torch.device("cpu")
    )
    print("✅ Training successful!")
    print(f"Results: {results}")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
