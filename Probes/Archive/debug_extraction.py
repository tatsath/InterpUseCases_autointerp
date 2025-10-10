#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/nvidia/Documents/Hariom/probetrain')

import torch
from transformers import AutoModel, AutoTokenizer
from probetrain.standalone_probe_system import load_dataset_from_csv

print("Testing hidden states extraction...")

# Load model
model_name = 'meta-llama/Llama-2-7b-hf'
print(f"Loading model: {model_name}")

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully")

# Load dataset
dataset = load_dataset_from_csv('/home/nvidia/Documents/Hariom/probetrain/probetrain/sample_multi_class_dataset.csv', max_samples=3)
print(f"Loaded {len(dataset)} examples")

# Test extraction
all_hidden_states_list = []

for i, example in enumerate(dataset):
    print(f"\nProcessing example {i+1}: {example['text'][:50]}...")
    
    # Tokenize
    inputs = tokenizer(
        example['text'], 
        return_tensors='pt', 
        padding=True, 
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        print(f"  Hidden states shape: {hidden_states[0].shape}")
        print(f"  Number of layers: {len(hidden_states)}")
        
        # Process each layer
        example_layers = []
        for layer_idx, layer in enumerate(hidden_states[1:]):  # Skip embedding layer
            # Use mean pooling
            token_repr = layer[0].mean(dim=0)  # [hidden_dim]
            example_layers.append(token_repr)
            print(f"  Layer {layer_idx}: {token_repr.shape}")
        
        # Stack layers for this example
        example_tensor = torch.stack(example_layers, dim=0)
        print(f"  Example tensor shape: {example_tensor.shape}")
        all_hidden_states_list.append(example_tensor)

# Stack all examples
try:
    all_hidden_states = torch.stack(all_hidden_states_list, dim=0)
    print(f"\n✅ Final hidden states shape: {all_hidden_states.shape}")
except Exception as e:
    print(f"\n❌ Error stacking tensors: {e}")
    print("Individual tensor shapes:")
    for i, tensor in enumerate(all_hidden_states_list):
        print(f"  Example {i}: {tensor.shape}")
