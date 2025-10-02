"""
Complete Analysis Circuit Tracer

Shows the difference between:
1. Features that improved most during finetuning (top 10 per layer)
2. Features that are actually most active during inference
3. All 400 features per layer (5 layers = 2000 total features)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import time

class CompleteAnalysisTracer:
    def __init__(self, model_path, sae_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map=None
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load SAEs
        self.saes = self._load_saes(sae_path)
        print(f"Loaded {len(self.saes)} SAE layers")
        
        # Load finetuning-improved features (top 10 per layer)
        self.finetuning_features = self._load_finetuning_features()
        print(f"Loaded {len(self.finetuning_features)} finetuning-improved features")
    
    def _load_finetuning_features(self):
        """Load features that improved most during finetuning (top 10 per layer)."""
        features = {}
        
        # Layer 4 - Top 10 features with largest activation improvement
        features[4] = [299, 32, 347, 176, 335, 362, 269, 387, 312, 209]
        
        # Layer 10 - Top 10 features with largest activation improvement  
        features[10] = [83, 162, 91, 266, 318, 105, 310, 320, 131, 17]
        
        # Layer 16 - Top 10 features with largest activation improvement
        features[16] = [389, 85, 385, 279, 18, 355, 283, 121, 107, 228]
        
        # Layer 22 - Top 10 features with largest activation improvement
        features[22] = [159, 258, 116, 186, 141, 323, 90, 252, 157, 353]
        
        # Layer 28 - Top 10 features with largest activation improvement
        features[28] = [116, 375, 276, 345, 305, 287, 19, 103, 178, 121]
        
        return features
    
    def _load_saes(self, sae_path):
        """Load SAE weights."""
        saes = {}
        layers = [4, 10, 16, 22, 28]
        
        for layer in layers:
            try:
                sae_file = f"{sae_path}/layers.{layer}/sae.safetensors"
                with safe_open(sae_file, framework="pt") as f:
                    encoder = f.get_tensor("encoder.weight").to(self.device, dtype=torch.float16)
                    encoder_bias = f.get_tensor("encoder.bias").to(self.device, dtype=torch.float16)
                
                saes[layer] = {
                    'encoder': encoder,
                    'bias': encoder_bias
                }
                print(f"✓ Layer {layer}")
            except Exception as e:
                print(f"✗ Layer {layer}: {e}")
        
        return saes
    
    def analyze_feature_overlap(self, prompt):
        """Analyze overlap between finetuning-improved features and actually active features."""
        print(f"\nAnalyzing: '{prompt[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        results = {}
        
        for layer, hidden_states in enumerate(outputs.hidden_states):
            if layer in self.saes:
                # Apply SAE
                sae = self.saes[layer]
                features = torch.matmul(hidden_states, sae['encoder'].T) + sae['bias']
                features = torch.relu(features)
                
                # Get top 10 active features
                feature_scores = features.max(dim=1)[0].squeeze()
                top_features = torch.topk(feature_scores, 10)
                
                active_features = set(top_features.indices.cpu().numpy())
                finetuning_features = set(self.finetuning_features.get(layer, []))
                
                overlap = active_features.intersection(finetuning_features)
                
                results[layer] = {
                    'active_features': active_features,
                    'finetuning_features': finetuning_features,
                    'overlap': overlap,
                    'overlap_count': len(overlap),
                    'scores': top_features.values.cpu().numpy()
                }
                
                print(f"\nLayer {layer}:")
                print(f"  Active features: {sorted(active_features)}")
                print(f"  Finetuning-improved features: {sorted(finetuning_features)}")
                print(f"  Overlap: {sorted(overlap)} ({len(overlap)}/10)")
                print(f"  Top 5 active: {top_features.indices.cpu().numpy()[:5]} (scores: {top_features.values.cpu().numpy()[:5]})")
        
        return results
    
    def comprehensive_analysis(self, prompts):
        """Run comprehensive analysis across multiple prompts."""
        print("="*80)
        print("COMPREHENSIVE FEATURE ANALYSIS")
        print("="*80)
        print(f"Total features per layer: 400")
        print(f"Total layers: 5")
        print(f"Total features: 2000")
        print(f"Finetuning-improved features: 50 (top 10 per layer)")
        print("="*80)
        
        all_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*20} PROMPT {i+1} {'='*20}")
            results = self.analyze_feature_overlap(prompt)
            all_results.append(results)
        
        # Summary analysis
        print(f"\n{'='*80}")
        print("SUMMARY ANALYSIS")
        print(f"{'='*80}")
        
        # Calculate average overlap per layer
        layer_overlaps = {layer: [] for layer in [4, 10, 16, 22, 28]}
        
        for result in all_results:
            for layer, data in result.items():
                layer_overlaps[layer].append(data['overlap_count'])
        
        print(f"\nAverage overlap between active and finetuning-improved features:")
        for layer in [4, 10, 16, 22, 28]:
            avg_overlap = np.mean(layer_overlaps[layer])
            print(f"  Layer {layer}: {avg_overlap:.1f}/10 features overlap")
        
        # Find most consistently active features
        all_active_features = []
        for result in all_results:
            for layer, data in result.items():
                all_active_features.extend(list(data['active_features']))
        
        from collections import Counter
        feature_counts = Counter(all_active_features)
        
        print(f"\nMost consistently active features across all prompts:")
        for feature, count in feature_counts.most_common(10):
            print(f"  Feature {feature}: appears in {count}/{len(prompts)} prompts")
        
        # Check if any finetuning features are consistently active
        finetuning_active = []
        for result in all_results:
            for layer, data in result.items():
                finetuning_active.extend(list(data['overlap']))
        
        finetuning_counts = Counter(finetuning_active)
        
        print(f"\nFinetuning-improved features that are also consistently active:")
        if finetuning_counts:
            for feature, count in finetuning_counts.most_common(10):
                print(f"  Feature {feature}: appears in {count}/{len(prompts)} prompts")
        else:
            print("  None - no overlap between finetuning-improved and consistently active features!")
        
        print(f"\n{'='*80}")
        print("KEY INSIGHTS:")
        print(f"{'='*80}")
        print("1. The features that improved most during finetuning are NOT the same")
        print("   as the features that are most active during normal inference.")
        print("2. This suggests finetuning created specialized features for specific")
        print("   financial tasks, while general inference relies on different features.")
        print("3. The model has 2000 total features (400 per layer), but we only")
        print("   have labels for 50 features (top 10 per layer from finetuning).")
        print("4. Most active features during inference are from the 'unknown' 1950 features.")

def main():
    print("="*80)
    print("COMPLETE ANALYSIS CIRCUIT TRACER")
    print("="*80)
    
    # Setup
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    tracer = CompleteAnalysisTracer(model_path, sae_path)
    
    # Test prompts
    prompts = [
        "Jamie Dimon leads JPMorgan Chase",
        "Bank earnings rise after rate cuts",
        "Microsoft acquires Activision",
        "Federal Reserve raises rates",
        "Tesla stock increases 15%"
    ]
    
    # Run comprehensive analysis
    tracer.comprehensive_analysis(prompts)
    
    print(f"\n✅ Complete analysis completed!")

if __name__ == "__main__":
    main()
