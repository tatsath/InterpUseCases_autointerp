#!/usr/bin/env python3
"""
Banking77 Multi-Layer Linear Probe - Training and Testing
Linear probe for meta-llama/Llama-2-7b-hf using mteb/banking77 dataset
Tests accuracy across ALL layers
"""

import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

class LinearProbe(nn.Module):
    """Simple linear probe for classification"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear(self.dropout(x))

class Banking77MultiLayerProbe:
    """Multi-layer linear probe trainer and tester for Banking77 dataset"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.probes = None
        self.num_classes = None
        self.num_layers = None
        
    def load_model(self):
        """Load the language model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True  # Enable hidden states output
        )
        self.model.eval()
        
        # Get number of layers
        self.num_layers = self.model.config.num_hidden_layers
        print(f"Model loaded on device: {self.device}")
        print(f"Number of layers: {self.num_layers}")
        
    def load_banking77_dataset(self, max_samples=None, split='train'):
        """Load and preprocess the Banking77 dataset"""
        print(f"Loading Banking77 {split} dataset...")
        dataset = load_dataset("mteb/banking77")
        
        # Get the specified split
        data = dataset[split]
        
        # Convert to list of examples
        examples = []
        
        for item in data:
            examples.append({
                'text': item['text'],
                'label': item['label']  # Banking77 labels are 0-indexed (0-76)
            })
            
        # Limit dataset size if needed
        if max_samples and len(examples) > max_samples:
            examples = examples[:max_samples]
            
        print(f"Loaded {len(examples)} examples from Banking77 {split}")
        return examples
    
    def extract_representations(self, examples, batch_size=8, max_length=512):
        """Extract hidden states from ALL layers of the model"""
        print("Extracting representations from ALL layers...")
        
        all_representations = []  # Will be [num_examples, num_layers, hidden_dim]
        all_labels = []
        
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i+batch_size]
            batch_texts = [ex['text'] for ex in batch_examples]
            batch_labels = [ex['label'] for ex in batch_examples]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Get hidden states from ALL layers
            with torch.no_grad():
                outputs = self.model(**inputs)
                # outputs.hidden_states is a tuple of [batch_size, seq_len, hidden_dim] for each layer
                # We want [batch_size, hidden_dim] for each layer (using CLS token)
                layer_representations = []
                for layer_idx in range(len(outputs.hidden_states)):
                    # Use CLS token (first token) representation
                    layer_repr = outputs.hidden_states[layer_idx][:, 0, :]  # [batch_size, hidden_dim]
                    layer_representations.append(layer_repr.cpu())
                
                # Stack to get [batch_size, num_layers, hidden_dim]
                batch_layer_repr = torch.stack(layer_representations, dim=1)
                all_representations.append(batch_layer_repr)
                all_labels.extend(batch_labels)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_examples)}/{len(examples)} examples")
        
        # Concatenate all representations
        representations = torch.cat(all_representations, dim=0)  # [num_examples, num_layers, hidden_dim]
        labels = torch.tensor(all_labels, dtype=torch.long)
        
        print(f"Extracted representations shape: {representations.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Number of layers: {representations.shape[1]}")
        
        return representations, labels
    
    def train_probes(self, train_representations, train_labels, test_representations, test_labels, 
                    num_classes, epochs=100, lr=0.001):
        """Train linear probes for ALL layers"""
        print("Training linear probes for ALL layers...")
        
        self.num_classes = num_classes
        num_layers = train_representations.shape[1]
        self.probes = []
        
        layer_accuracies = []
        layer_results = []
        
        # Train a probe for each layer
        for layer_idx in range(num_layers):
            print(f"\n--- Training probe for Layer {layer_idx} ---")
            
            # Get representations for this layer
            train_layer_repr = train_representations[:, layer_idx, :]  # [num_examples, hidden_dim]
            test_layer_repr = test_representations[:, layer_idx, :]
            
            # Initialize probe for this layer
            input_dim = train_layer_repr.shape[1]
            probe = LinearProbe(input_dim, num_classes).to(self.device)
            probe = probe.to(train_layer_repr.dtype)
            
            # Move data to device
            train_layer_repr = train_layer_repr.to(self.device)
            train_labels = train_labels.to(self.device)
            test_layer_repr = test_layer_repr.to(self.device)
            test_labels = test_labels.to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
            
            # Training loop
            train_losses = []
            train_accuracies = []
            
            for epoch in range(epochs):
                # Training
                probe.train()
                optimizer.zero_grad()
                
                outputs = probe(train_layer_repr)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == train_labels).float().mean().item()
                    
                train_losses.append(loss.item())
                train_accuracies.append(accuracy)
                
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")
            
            # Final evaluation on test set
            probe.eval()
            with torch.no_grad():
                test_outputs = probe(test_layer_repr)
                test_predictions = torch.argmax(test_outputs, dim=1)
                test_accuracy = (test_predictions == test_labels).float().mean().item()
                
            print(f"  Layer {layer_idx} Test Accuracy: {test_accuracy:.4f}")
            
            # Store results
            layer_accuracies.append(test_accuracy)
            layer_results.append({
                'layer': layer_idx,
                'accuracy': test_accuracy,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'test_predictions': test_predictions.cpu().numpy(),
                'test_labels': test_labels.cpu().numpy()
            })
            
            self.probes.append(probe)
        
        # Print summary
        print(f"\n{'='*60}")
        print("ACCURACY ACROSS ALL LAYERS:")
        print(f"{'='*60}")
        for layer_idx, accuracy in enumerate(layer_accuracies):
            print(f"Layer {layer_idx:2d}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        best_layer = np.argmax(layer_accuracies)
        best_accuracy = layer_accuracies[best_layer]
        print(f"{'='*60}")
        print(f"Best Layer: {best_layer} with accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"{'='*60}")
        
        return {
            'layer_accuracies': layer_accuracies,
            'layer_results': layer_results,
            'best_layer': best_layer,
            'best_accuracy': best_accuracy
        }
    
    def generate_layer_visualizations(self, results, save_dir):
        """Generate visualization plots for all layers"""
        print("Generating layer-wise visualizations...")
        
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Accuracy by Layer
        plt.figure(figsize=(12, 8))
        layers = list(range(len(results['layer_accuracies'])))
        accuracies = results['layer_accuracies']
        
        plt.plot(layers, accuracies, marker='o', linewidth=2, markersize=8)
        plt.title('Banking77 Linear Probe Accuracy by Layer', fontsize=16)
        plt.xlabel('Layer Number', fontsize=14)
        plt.ylabel('Test Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(layers)
        
        # Highlight best layer
        best_layer = results['best_layer']
        best_accuracy = results['best_accuracy']
        plt.scatter([best_layer], [best_accuracy], color='red', s=100, zorder=5)
        plt.annotate(f'Best: Layer {best_layer}\nAccuracy: {best_accuracy:.4f}', 
                    xy=(best_layer, best_accuracy), xytext=(10, 10),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_by_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Layer Performance Comparison
        plt.figure(figsize=(15, 8))
        bars = plt.bar(layers, accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.title('Banking77 Linear Probe Performance by Layer', fontsize=16)
        plt.xlabel('Layer Number', fontsize=14)
        plt.ylabel('Test Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Color the best layer differently
        bars[best_layer].set_color('red')
        bars[best_layer].set_alpha(1.0)
        
        # Add value labels on bars
        for i, (layer, acc) in enumerate(zip(layers, accuracies)):
            plt.text(layer, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'layer_performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Layer visualizations saved to: {plots_dir}")
    
    def save_results(self, results, save_dir):
        """Save results and probe weights"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save all probe weights
        for layer_idx, probe in enumerate(self.probes):
            torch.save(probe.state_dict(), os.path.join(save_dir, f'probe_layer_{layer_idx}.pt'))
        
        # Save detailed results
        results_to_save = {
            'layer_accuracies': results['layer_accuracies'],
            'best_layer': results['best_layer'],
            'best_accuracy': results['best_accuracy'],
            'num_layers': len(results['layer_accuracies']),
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'probe_type': 'multi_layer_linear'
        }
        
        with open(os.path.join(save_dir, 'layer_results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Save per-layer detailed results
        for layer_result in results['layer_results']:
            layer_idx = layer_result['layer']
            layer_save_dir = os.path.join(save_dir, f'layer_{layer_idx}')
            os.makedirs(layer_save_dir, exist_ok=True)
            
            # Save confusion matrix for this layer
            if 'test_predictions' in layer_result and 'test_labels' in layer_result:
                cm = confusion_matrix(layer_result['test_labels'], layer_result['test_predictions'])
                np.save(os.path.join(layer_save_dir, 'confusion_matrix.npy'), cm)
                
                # Save classification report
                report = classification_report(
                    layer_result['test_labels'], 
                    layer_result['test_predictions'], 
                    output_dict=True
                )
                with open(os.path.join(layer_save_dir, 'classification_report.json'), 'w') as f:
                    json.dump(report, f, indent=2)
        
        print(f"Results saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Banking77 Multi-Layer Linear Probe - Train and Test')
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', help='Model name')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum samples to use')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--save_dir', default='banking77_multi_layer_probe_results', help='Save directory')
    
    args = parser.parse_args()
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}_{timestamp}"
    
    print("🚀 Banking77 Multi-Layer Linear Probe - Training and Testing")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print(f"Epochs per layer: {args.epochs}")
    print(f"Save directory: {save_dir}")
    print("=" * 70)
    
    # Initialize probe
    probe = Banking77MultiLayerProbe(model_name=args.model)
    
    # Load model
    probe.load_model()
    
    print("\n📚 TRAINING PHASE")
    print("-" * 30)
    
    # Load training dataset
    train_examples = probe.load_banking77_dataset(max_samples=args.max_samples, split='train')
    test_examples = probe.load_banking77_dataset(max_samples=args.max_samples//4, split='test')
    
    # Get number of classes from our sample
    unique_labels = set(ex['label'] for ex in train_examples)
    num_classes = len(unique_labels)
    print(f"Number of classes in sample: {num_classes}")
    
    # Create label mapping to 0-indexed
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    print(f"Label mapping: {dict(list(label_mapping.items())[:5])}...")  # Show first 5 mappings
    
    # Extract representations from ALL layers
    train_repr, train_labels = probe.extract_representations(train_examples, batch_size=args.batch_size)
    test_repr, test_labels = probe.extract_representations(test_examples, batch_size=args.batch_size)
    
    # Apply label mapping to make labels 0-indexed
    train_labels = torch.tensor([label_mapping[label.item()] for label in train_labels])
    test_labels = torch.tensor([label_mapping.get(label.item(), 0) for label in test_labels])  # Use 0 for unseen labels
    
    print(f"Training set: {train_repr.shape[0]} examples")
    print(f"Test set: {test_repr.shape[0]} examples")
    print(f"Number of layers: {train_repr.shape[1]}")
    
    # Train probes for all layers
    results = probe.train_probes(
        train_repr, train_labels, test_repr, test_labels, 
        num_classes, epochs=args.epochs, lr=args.lr
    )
    
    # Generate visualizations
    probe.generate_layer_visualizations(results, save_dir)
    
    # Save results
    probe.save_results(results, save_dir)
    
    print(f"\n✅ Training completed!")
    print(f"Best layer: {results['best_layer']} with accuracy: {results['best_accuracy']:.4f}")
    print(f"💾 All results saved to: {save_dir}")

if __name__ == "__main__":
    main()



