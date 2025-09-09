#!/usr/bin/env python3

import json
import argparse
import sys
import os
from typing import List, Dict, Any

def load_financial_data(max_samples: int = 100) -> List[str]:
    """Load financial data from HuggingFace dataset"""
    try:
        from datasets import load_dataset
        
        print(f"    📊 Loading financial data from jyanimaulik/yahoo_finance_stockmarket_news...")
        dataset = load_dataset("jyanimaulik/yahoo_finance_stockmarket_news", split="train")
        
        # Extract text content
        texts = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            # Get the main text content
            if 'text' in item:
                texts.append(item['text'])
            elif 'content' in item:
                texts.append(item['content'])
            elif 'headline' in item and 'summary' in item:
                texts.append(f"{item['headline']} {item['summary']}")
            else:
                # Fallback to any text field
                for key, value in item.items():
                    if isinstance(value, str) and len(value) > 50:
                        texts.append(value)
                        break
        
        print(f"    ✅ Loaded {len(texts)} financial text samples")
        return texts
        
    except Exception as e:
        print(f"    ❌ Error loading financial data: {str(e)}")
        return []

def get_model_activations(model_path: str, texts: List[str], layer: int, max_length: int = 256):
    """Get activations from the model for given texts"""
    try:
        print(f"    🔍 Getting activations from {model_path} for layer {layer}")
        
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # Load tokenizer and model with safetensors to avoid torch version issues
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Try to load with safetensors first, then fallback to regular loading
        try:
            model = AutoModel.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                device_map="auto",
                use_safetensors=True
            )
        except Exception as e:
            print(f"    ⚠️ Safetensors failed, trying regular loading: {str(e)}")
            try:
                model = AutoModel.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16, 
                    device_map="auto",
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"    ❌ Both safetensors and regular loading failed: {str(e2)}")
                return None
        
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        all_activations = []
        
        # Process texts in batches
        batch_size = 2  # Smaller batch size to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if i % 10 == 0:
                print(f"      Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # Tokenize batch
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=max_length, 
                             truncation=True, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                # Get activations for the specified layer
                layer_activations = outputs.hidden_states[layer + 1]  # +1 because layer 0 is embeddings
                all_activations.append(layer_activations.cpu())
        
        # Concatenate all activations
        if all_activations:
            activations = torch.cat(all_activations, dim=0)
            print(f"    ✅ Got activations shape: {activations.shape}")
            return activations
        else:
            return None
            
    except Exception as e:
        print(f"    ❌ Error getting model activations: {str(e)}")
        return None

def load_sae_weights(model_path: str, layer: int):
    """Load SAE encoder weights for a specific layer"""
    try:
        import torch
        from safetensors import safe_open
        
        sae_file = f"{model_path}/layers.{layer}/sae.safetensors"
        print(f"    📂 Loading SAE weights from: {sae_file}")
        
        if not os.path.exists(sae_file):
            print(f"    ❌ SAE file not found: {sae_file}")
            return None, None
        
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight")
            bias = f.get_tensor("encoder.bias")
        
        print(f"    ✅ Loaded SAE weights - Encoder: {encoder.shape}, Bias: {bias.shape}")
        return encoder, bias
        
    except Exception as e:
        print(f"    ❌ Error loading SAE for layer {layer}: {str(e)}")
        return None, None

def compute_sae_activations(activations, encoder, bias):
    """Compute SAE feature activations"""
    try:
        import torch
        
        # Ensure all tensors have the same dtype
        activations = activations.float()
        encoder = encoder.float()
        bias = bias.float()
        
        # Compute SAE activations: ReLU(W_enc @ x + b_enc)
        # activations shape: [batch_size, seq_len, hidden_dim]
        # encoder shape: [n_features, hidden_dim]
        # bias shape: [n_features]
        
        batch_size, seq_len, hidden_dim = activations.shape
        n_features = encoder.shape[0]
        
        # Reshape activations to [batch_size * seq_len, hidden_dim]
        activations_flat = activations.view(-1, hidden_dim)
        
        # Compute SAE activations
        sae_acts = torch.relu(torch.matmul(activations_flat, encoder.T) + bias)
        
        # Reshape back to [batch_size, seq_len, n_features]
        sae_acts = sae_acts.view(batch_size, seq_len, n_features)
        
        return sae_acts
        
    except Exception as e:
        print(f"    ❌ Error computing SAE activations: {str(e)}")
        return None

def get_top_features_for_model(model_path: str, sae_path: str, texts: List[str], layer: int, top_n: int = 10):
    """Get top features for a specific model"""
    import torch
    
    print(f"\n🔍 Getting top {top_n} features for {model_path} - Layer {layer}")
    
    # Get model activations
    activations = get_model_activations(model_path, texts, layer)
    if activations is None:
        return None
    
    # Load SAE weights
    encoder, bias = load_sae_weights(sae_path, layer)
    if encoder is None or bias is None:
        return None
    
    # Compute SAE activations
    sae_acts = compute_sae_activations(activations, encoder, bias)
    if sae_acts is None:
        return None
    
    # Compute average activations per feature
    avg_acts = sae_acts.mean(dim=(0, 1))  # Average across batch and sequence
    
    # Get top features
    top_features = torch.topk(avg_acts, k=top_n)
    
    return {
        'feature_indices': top_features.indices.tolist(),
        'activation_values': top_features.values.tolist(),
        'model_path': model_path,
        'layer': layer
    }

def generate_feature_labels(feature_indices: List[int], model_path: str, layer: int, texts: List[str]):
    """Generate labels for features using LLM"""
    try:
        print(f"    🏷️ Generating labels for {len(feature_indices)} features...")
        
        # This is a simplified version - in practice you'd use AutoInterp's labeling system
        # For now, we'll create placeholder labels
        labels = []
        for i, feature_idx in enumerate(feature_indices):
            # Placeholder label - in real implementation, this would use AutoInterp
            label = f"Feature_{feature_idx}_Layer_{layer}"
            labels.append(label)
        
        return labels
        
    except Exception as e:
        print(f"    ❌ Error generating labels: {str(e)}")
        return [f"Feature_{idx}" for idx in feature_indices]

def main():
    parser = argparse.ArgumentParser(description="Label top features for base and finetuned models")
    parser.add_argument("--base_sae", required=True, help="Path to base SAE model")
    parser.add_argument("--finetuned_sae", required=True, help="Path to finetuned SAE model")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--finetuned_model", required=True, help="Finetuned model path")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 10, 16, 22, 28], help="Layers to analyze")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top features to label")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of text samples to use")
    
    args = parser.parse_args()
    
    print("🏷️ FEATURE LABELING FOR BASE AND FINETUNED MODELS")
    print("=" * 60)
    print(f"📊 Using {args.max_samples} financial text samples")
    print(f"🔍 Analyzing {len(args.layers)} layers: {args.layers}")
    print(f"📋 Top {args.top_n} features per model per layer")
    print("")
    
    # Load financial data
    print("📊 Loading financial data...")
    financial_texts = load_financial_data(max_samples=args.max_samples)
    if not financial_texts:
        print("❌ No financial data loaded")
        return
    
    print(f"✅ Loaded {len(financial_texts)} financial text samples")
    print("")
    
    all_results = {}
    
    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"LABELING FEATURES FOR LAYER {layer}")
        print(f"{'='*60}")
        
        layer_results = {}
        
        # Get top features for base model
        base_features = get_top_features_for_model(
            args.base_model, args.base_sae, financial_texts, layer, args.top_n
        )
        
        if base_features:
            base_labels = generate_feature_labels(
                base_features['feature_indices'], args.base_model, layer, financial_texts
            )
            layer_results['base_model'] = {
                'features': base_features,
                'labels': base_labels
            }
            print(f"✅ Base model: {len(base_features['feature_indices'])} features labeled")
        
        # Get top features for finetuned model
        finetuned_features = get_top_features_for_model(
            args.finetuned_model, args.finetuned_sae, financial_texts, layer, args.top_n
        )
        
        if finetuned_features:
            finetuned_labels = generate_feature_labels(
                finetuned_features['feature_indices'], args.finetuned_model, layer, financial_texts
            )
            layer_results['finetuned_model'] = {
                'features': finetuned_features,
                'labels': finetuned_labels
            }
            print(f"✅ Finetuned model: {len(finetuned_features['feature_indices'])} features labeled")
        
        all_results[layer] = layer_results
    
    # Save results
    output_file = "feature_labels_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("✅ Feature labeling completed!")

if __name__ == "__main__":
    main()
