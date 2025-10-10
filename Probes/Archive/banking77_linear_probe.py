#!/usr/bin/env python3
"""
Banking77 Linear Probe - Training and Testing
Simple linear probe for meta-llama/Llama-2-7b-hf using mteb/banking77 dataset
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

class Banking77LinearProbe:
    """Linear probe trainer and tester for Banking77 dataset"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.probe = None
        self.num_classes = None
        
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
            device_map="auto"
        )
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
        
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
        """Extract hidden states from the model for all examples"""
        print("Extracting representations from model...")
        
        all_representations = []
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
            
            # Get hidden states
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state (CLS token representation)
                hidden_states = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                
            all_representations.append(hidden_states.cpu())
            all_labels.extend(batch_labels)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_examples)}/{len(examples)} examples")
        
        # Concatenate all representations
        representations = torch.cat(all_representations, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)
        
        print(f"Extracted representations shape: {representations.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return representations, labels
    
    def train_probe(self, train_representations, train_labels, test_representations, test_labels, 
                   num_classes, epochs=100, lr=0.001):
        """Train the linear probe"""
        print("Training linear probe...")
        
        self.num_classes = num_classes
        
        # Initialize probe
        input_dim = train_representations.shape[1]
        self.probe = LinearProbe(input_dim, num_classes).to(self.device)
        
        # Ensure probe uses same dtype as model
        self.probe = self.probe.to(train_representations.dtype)
        
        # Move data to device
        train_representations = train_representations.to(self.device)
        train_labels = train_labels.to(self.device)
        test_representations = test_representations.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=lr)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            # Training
            self.probe.train()
            optimizer.zero_grad()
            
            outputs = self.probe(train_representations)
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
                print(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")
        
        # Final evaluation
        self.probe.eval()
        with torch.no_grad():
            test_outputs = self.probe(test_representations)
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_accuracy = (test_predictions == test_labels).float().mean().item()
            
        print(f"Final test accuracy: {test_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'test_predictions': test_predictions.cpu().numpy(),
            'test_labels': test_labels.cpu().numpy()
        }
    
    def test_probe(self, test_representations, test_labels):
        """Test the probe on the test set"""
        print("Testing probe...")
        
        # Move data to device
        test_representations = test_representations.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.probe(test_representations)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
        # Calculate metrics
        accuracy = (predictions == test_labels).float().mean().item()
        
        # Convert to numpy for sklearn metrics
        predictions_np = predictions.cpu().numpy()
        labels_np = test_labels.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        
        return {
            'accuracy': accuracy,
            'predictions': predictions_np,
            'labels': labels_np,
            'probabilities': probabilities_np
        }
    
    def load_probe(self, probe_path, num_classes):
        """Load a trained probe"""
        print(f"Loading probe from: {probe_path}")
        
        # Get input dimension from the first layer weight
        checkpoint = torch.load(probe_path, map_location=self.device)
        input_dim = checkpoint['linear.weight'].shape[1]
        
        # Initialize probe
        self.probe = LinearProbe(input_dim, num_classes).to(self.device)
        self.probe.load_state_dict(checkpoint)
        self.probe.eval()
        self.num_classes = num_classes
        
        print(f"Probe loaded: input_dim={input_dim}, num_classes={num_classes}")
    
    def generate_visualizations(self, results, save_dir):
        """Generate visualization plots"""
        print("Generating visualizations...")
        
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(results['labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Banking77 Linear Probe')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Accuracy by Class
        class_report = classification_report(results['labels'], results['predictions'], output_dict=True)
        
        # Extract per-class F1 scores
        class_f1_scores = []
        class_names = []
        for i in range(len(class_report) - 3):  # Exclude 'accuracy', 'macro avg', 'weighted avg'
            class_name = f"Class {i}"
            if class_name in class_report:
                class_f1_scores.append(class_report[class_name]['f1-score'])
                class_names.append(class_name)
        
        plt.figure(figsize=(15, 8))
        plt.bar(class_names, class_f1_scores)
        plt.title('F1-Score by Class')
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'f1_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction Confidence Distribution
        if 'probabilities' in results and results['probabilities'] is not None:
            max_probs = np.max(results['probabilities'], axis=1)
            # Filter out NaN values
            max_probs = max_probs[~np.isnan(max_probs)]
            if len(max_probs) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
                plt.title('Distribution of Maximum Prediction Probabilities')
                plt.xlabel('Maximum Probability')
                plt.ylabel('Frequency')
                plt.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("Warning: All probabilities are NaN, skipping confidence distribution plot")
        
        print(f"Visualizations saved to: {plots_dir}")
    
    def save_results(self, results, save_dir, is_training=True):
        """Save results and probe weights"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save probe weights if training
        if is_training and self.probe is not None:
            torch.save(self.probe.state_dict(), os.path.join(save_dir, 'probe_weights.pt'))
        
        # Save detailed results
        results_to_save = {
            'accuracy': results.get('accuracy', results.get('test_accuracy', 0.0)),
            'num_samples': len(results.get('labels', [])),
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'probe_type': 'linear'
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Save classification report
        if 'labels' in results and 'predictions' in results:
            report = classification_report(
                results['labels'], 
                results['predictions'], 
                output_dict=True
            )
        else:
            report = {}
        
        with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save confusion matrix
        if 'labels' in results and 'predictions' in results:
            cm = confusion_matrix(results['labels'], results['predictions'])
            np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
            
            # Save predictions
            np.save(os.path.join(save_dir, 'predictions.npy'), results['predictions'])
            np.save(os.path.join(save_dir, 'labels.npy'), results['labels'])
            if 'probabilities' in results:
                np.save(os.path.join(save_dir, 'probabilities.npy'), results['probabilities'])
        
        print(f"Results saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Banking77 Linear Probe - Train and Test')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both', 
                       help='Mode: train, test, or both')
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', help='Model name')
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum samples to use')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--save_dir', default='banking77_linear_probe_results', help='Save directory')
    parser.add_argument('--probe_path', help='Path to trained probe weights (for test mode)')
    parser.add_argument('--num_classes', type=int, default=77, help='Number of classes (for test mode)')
    
    args = parser.parse_args()
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}_{timestamp}"
    
    print("🚀 Banking77 Linear Probe - Training and Testing")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print(f"Save directory: {save_dir}")
    print("=" * 60)
    
    # Initialize probe
    probe = Banking77LinearProbe(model_name=args.model)
    
    # Load model
    probe.load_model()
    
    if args.mode in ['train', 'both']:
        print("\n📚 TRAINING PHASE")
        print("-" * 30)
        
        # Load training dataset
        train_examples = probe.load_banking77_dataset(max_samples=args.max_samples, split='train')
        test_examples = probe.load_banking77_dataset(max_samples=args.max_samples//4, split='test')
        
        # Get number of classes from our sample
        unique_labels = set(ex['label'] for ex in train_examples)
        num_classes = len(unique_labels)
        print(f"Number of classes in sample: {num_classes}")
        print(f"Unique labels in training sample: {len(unique_labels)}")
        print(f"Label range: {min(unique_labels)} to {max(unique_labels)}")
        
        # Create label mapping to 0-indexed
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"Label mapping: {dict(list(label_mapping.items())[:5])}...")  # Show first 5 mappings
        
        # Extract representations
        train_repr, train_labels = probe.extract_representations(train_examples, batch_size=args.batch_size)
        test_repr, test_labels = probe.extract_representations(test_examples, batch_size=args.batch_size)
        
        # Apply label mapping to make labels 0-indexed
        train_labels = torch.tensor([label_mapping[label.item()] for label in train_labels])
        test_labels = torch.tensor([label_mapping.get(label.item(), 0) for label in test_labels])  # Use 0 for unseen labels
        
        print(f"Training set: {train_repr.shape[0]} examples")
        print(f"Test set: {test_repr.shape[0]} examples")
        
        # Train probe
        results = probe.train_probe(
            train_repr, train_labels, test_repr, test_labels, 
            num_classes, epochs=args.epochs, lr=args.lr
        )
        
        # Save training results
        probe.save_results(results, save_dir, is_training=True)
        
        print(f"✅ Training completed! Test accuracy: {results['test_accuracy']:.4f}")
    
    if args.mode in ['test', 'both']:
        print("\n🧪 TESTING PHASE")
        print("-" * 30)
        
        # Load probe if not trained in this session
        if args.mode == 'test':
            if not args.probe_path:
                print("Error: --probe_path required for test mode")
                return
            probe.load_probe(args.probe_path, args.num_classes)
        
        # Load test dataset
        test_examples = probe.load_banking77_dataset(max_samples=args.max_samples, split='test')
        
        # Extract representations
        test_repr, test_labels = probe.extract_representations(test_examples, batch_size=args.batch_size)
        
        # Test probe
        results = probe.test_probe(test_repr, test_labels)
        
        # Generate visualizations
        probe.generate_visualizations(results, save_dir)
        
        # Save test results
        probe.save_results(results, save_dir, is_training=False)
        
        print(f"✅ Testing completed! Test accuracy: {results['accuracy']:.4f}")
    
    print(f"\n💾 All results saved to: {save_dir}")

if __name__ == "__main__":
    main()
