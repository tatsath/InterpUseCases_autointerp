#!/usr/bin/env python3
"""
Banking77 Attention Probe - Training and Testing
Attention-weighted probe for meta-llama/Llama-2-7b-hf using mteb/banking77 dataset
Based on concepts from https://github.com/EleutherAI/attention-probes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

class AttentionProbe(nn.Module):
    """Attention-weighted probe for classification"""
    def __init__(self, input_dim, num_classes, n_heads=4, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, mask=None):
        """
        Forward pass with attention weighting
        x: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        """
        # Handle both sequence and single vector inputs
        if x.dim() == 2:
            # Single vector input - add sequence dimension
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            single_vector = True
        else:
            single_vector = False
        
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
        
        # Residual connection and layer norm
        x = self.layer_norm(x + attn_output)
        
        # Global average pooling if we have sequence input
        if not single_vector:
            x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Classification
        output = self.classifier(x)
        
        return output, attn_weights

class Banking77AttentionProbe:
    """Attention probe trainer and tester for Banking77 dataset"""
    
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
                'label': item['label']  # Banking77 labels are already 0-indexed (0-76)
            })
            
        # Limit dataset size if needed
        if max_samples and len(examples) > max_samples:
            examples = examples[:max_samples]
            
        print(f"Loaded {len(examples)} examples from Banking77 {split}")
        return examples
    
    def extract_representations(self, examples, batch_size=8, max_length=512, use_sequence=True):
        """Extract hidden states from the model for all examples"""
        print("Extracting representations from model...")
        
        all_representations = []
        all_labels = []
        all_attention_masks = []
        
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
                
                if use_sequence:
                    # Use all token representations for attention
                    hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                    attention_mask = inputs['attention_mask']  # [batch_size, seq_len]
                else:
                    # Use only CLS token representation
                    hidden_states = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                    attention_mask = None
                
            all_representations.append(hidden_states.cpu())
            all_labels.extend(batch_labels)
            if attention_mask is not None:
                all_attention_masks.append(attention_mask.cpu())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_examples)}/{len(examples)} examples")
        
        # Concatenate all representations
        representations = torch.cat(all_representations, dim=0)
        labels = torch.tensor(all_labels, dtype=torch.long)
        attention_masks = torch.cat(all_attention_masks, dim=0) if all_attention_masks else None
        
        print(f"Extracted representations shape: {representations.shape}")
        print(f"Labels shape: {labels.shape}")
        if attention_masks is not None:
            print(f"Attention masks shape: {attention_masks.shape}")
        
        return representations, labels, attention_masks
    
    def train_probe(self, train_representations, train_labels, test_representations, test_labels, 
                   num_classes, train_attention_masks=None, test_attention_masks=None,
                   epochs=100, lr=0.001, n_heads=4):
        """Train the attention probe"""
        print("Training attention probe...")
        
        self.num_classes = num_classes
        
        # Initialize probe
        input_dim = train_representations.shape[-1]
        self.probe = AttentionProbe(input_dim, num_classes, n_heads=n_heads).to(self.device)
        
        # Ensure probe uses same dtype as model
        self.probe = self.probe.to(train_representations.dtype)
        
        # Move data to device
        train_representations = train_representations.to(self.device)
        train_labels = train_labels.to(self.device)
        test_representations = test_representations.to(self.device)
        test_labels = test_labels.to(self.device)
        
        if train_attention_masks is not None:
            train_attention_masks = train_attention_masks.to(self.device)
        if test_attention_masks is not None:
            test_attention_masks = test_attention_masks.to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        best_accuracy = 0
        
        for epoch in range(epochs):
            # Training
            self.probe.train()
            optimizer.zero_grad()
            
            outputs, attn_weights = self.probe(train_representations, train_attention_masks)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == train_labels).float().mean().item()
                
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)
            
            # Learning rate scheduling
            scheduler.step(loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")
        
        # Final evaluation
        self.probe.eval()
        with torch.no_grad():
            test_outputs, test_attn_weights = self.probe(test_representations, test_attention_masks)
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_accuracy = (test_predictions == test_labels).float().mean().item()
            
        print(f"Final test accuracy: {test_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracy': test_accuracy,
            'test_predictions': test_predictions.cpu().numpy(),
            'test_labels': test_labels.cpu().numpy(),
            'attention_weights': test_attn_weights.cpu().numpy() if test_attn_weights is not None else None
        }
    
    def test_probe(self, test_representations, test_labels, test_attention_masks=None):
        """Test the probe on the test set"""
        print("Testing attention probe...")
        
        # Move data to device
        test_representations = test_representations.to(self.device)
        test_labels = test_labels.to(self.device)
        if test_attention_masks is not None:
            test_attention_masks = test_attention_masks.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs, attn_weights = self.probe(test_representations, test_attention_masks)
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
            'probabilities': probabilities_np,
            'attention_weights': attn_weights.cpu().numpy() if attn_weights is not None else None
        }
    
    def load_probe(self, probe_path, num_classes):
        """Load a trained probe"""
        print(f"Loading probe from: {probe_path}")
        
        # Load checkpoint
        checkpoint = torch.load(probe_path, map_location=self.device)
        
        # Initialize probe with same architecture
        input_dim = checkpoint['classifier.3.weight'].shape[1]  # Get from final layer
        n_heads = checkpoint.get('attention.num_heads', 4)  # Default to 4 if not found
        
        self.probe = AttentionProbe(input_dim, num_classes, n_heads=n_heads).to(self.device)
        self.probe.load_state_dict(checkpoint)
        self.probe.eval()
        self.num_classes = num_classes
        
        print(f"Attention probe loaded: input_dim={input_dim}, num_classes={num_classes}, n_heads={n_heads}")
    
    def generate_visualizations(self, results, save_dir):
        """Generate visualization plots including attention weights"""
        print("Generating visualizations...")
        
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(results['labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Banking77 Attention Probe')
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
        plt.title('F1-Score by Class - Attention Probe')
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'f1_by_class.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction Confidence Distribution
        max_probs = np.max(results['probabilities'], axis=1)
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Maximum Prediction Probabilities - Attention Probe')
        plt.xlabel('Maximum Probability')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Attention Weights Visualization (if available)
        if results['attention_weights'] is not None:
            attn_weights = results['attention_weights']
            # Average attention weights across heads and samples
            avg_attention = np.mean(attn_weights, axis=(0, 1))  # Average over batch and heads
            
            plt.figure(figsize=(12, 8))
            plt.imshow(avg_attention, cmap='viridis', aspect='auto')
            plt.colorbar(label='Attention Weight')
            plt.title('Average Attention Weights - Attention Probe')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'attention_weights.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {plots_dir}")
    
    def save_results(self, results, save_dir, is_training=True):
        """Save results and probe weights"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save probe weights if training
        if is_training and self.probe is not None:
            torch.save(self.probe.state_dict(), os.path.join(save_dir, 'attention_probe_weights.pt'))
        
        # Save detailed results
        results_to_save = {
            'accuracy': results.get('accuracy', results.get('test_accuracy', 0.0)),
            'num_samples': len(results['labels']),
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'probe_type': 'attention'
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Save classification report
        report = classification_report(
            results['labels'], 
            results['predictions'], 
            output_dict=True
        )
        
        with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save confusion matrix
        cm = confusion_matrix(results['labels'], results['predictions'])
        np.save(os.path.join(save_dir, 'confusion_matrix.npy'), cm)
        
        # Save predictions and attention weights
        np.save(os.path.join(save_dir, 'predictions.npy'), results['predictions'])
        np.save(os.path.join(save_dir, 'labels.npy'), results['labels'])
        np.save(os.path.join(save_dir, 'probabilities.npy'), results['probabilities'])
        
        if results['attention_weights'] is not None:
            np.save(os.path.join(save_dir, 'attention_weights.npy'), results['attention_weights'])
        
        print(f"Results saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Banking77 Attention Probe - Train and Test')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both', 
                       help='Mode: train, test, or both')
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', help='Model name')
    parser.add_argument('--max_samples', type=int, default=5000, help='Maximum samples to use')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--use_sequence', action='store_true', default=True, 
                       help='Use sequence representations (vs CLS token only)')
    parser.add_argument('--save_dir', default='banking77_attention_probe_results', help='Save directory')
    parser.add_argument('--probe_path', help='Path to trained probe weights (for test mode)')
    parser.add_argument('--num_classes', type=int, default=77, help='Number of classes (for test mode)')
    
    args = parser.parse_args()
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{args.save_dir}_{timestamp}"
    
    print("🚀 Banking77 Attention Probe - Training and Testing")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples}")
    print(f"Attention heads: {args.n_heads}")
    print(f"Use sequence: {args.use_sequence}")
    print(f"Save directory: {save_dir}")
    print("=" * 60)
    
    # Initialize probe
    probe = Banking77AttentionProbe(model_name=args.model)
    
    # Load model
    probe.load_model()
    
    if args.mode in ['train', 'both']:
        print("\n📚 TRAINING PHASE")
        print("-" * 30)
        
        # Load training dataset
        train_examples = probe.load_banking77_dataset(max_samples=args.max_samples, split='train')
        test_examples = probe.load_banking77_dataset(max_samples=args.max_samples//4, split='test')
        
        # Get number of classes - Banking77 has 77 classes total
        num_classes = 77  # Banking77 dataset has exactly 77 classes
        print(f"Number of classes: {num_classes}")
        
        # Check if we have all classes in our sample
        unique_labels = set(ex['label'] for ex in train_examples)
        print(f"Unique labels in training sample: {len(unique_labels)}")
        print(f"Label range: {min(unique_labels)} to {max(unique_labels)}")
        
        # Extract representations
        train_repr, train_labels, train_masks = probe.extract_representations(
            train_examples, batch_size=args.batch_size, use_sequence=args.use_sequence)
        test_repr, test_labels, test_masks = probe.extract_representations(
            test_examples, batch_size=args.batch_size, use_sequence=args.use_sequence)
        
        print(f"Training set: {train_repr.shape[0]} examples")
        print(f"Test set: {test_repr.shape[0]} examples")
        
        # Train probe
        results = probe.train_probe(
            train_repr, train_labels, test_repr, test_labels, 
            num_classes, train_masks, test_masks,
            epochs=args.epochs, lr=args.lr, n_heads=args.n_heads
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
        test_repr, test_labels, test_masks = probe.extract_representations(
            test_examples, batch_size=args.batch_size, use_sequence=args.use_sequence)
        
        # Test probe
        results = probe.test_probe(test_repr, test_labels, test_masks)
        
        # Generate visualizations
        probe.generate_visualizations(results, save_dir)
        
        # Save test results
        probe.save_results(results, save_dir, is_training=False)
        
        print(f"✅ Testing completed! Test accuracy: {results['accuracy']:.4f}")
    
    print(f"\n💾 All results saved to: {save_dir}")

if __name__ == "__main__":
    main()
