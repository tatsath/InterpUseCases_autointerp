"""
Simple and Fast Circuit Tracer

A lightweight version that focuses on speed and simplicity.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

@dataclass
class SimpleSAE:
    """Simple SAE wrapper for fast computation."""
    layer: int
    encoder: torch.Tensor
    encoder_bias: Optional[torch.Tensor] = None
    
    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fast SAE encoding."""
        # Simple matrix multiplication + ReLU
        features = torch.matmul(hidden_states, self.encoder.T)
        if self.encoder_bias is not None:
            features = features + self.encoder_bias
        return torch.relu(features)

class FastCircuitTracer:
    """Fast and simple circuit tracer."""
    
    def __init__(self, model_path: str, sae_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model (single GPU, no device mapping)
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load SAEs (simplified)
        self.saes = self._load_saes(sae_path)
        print(f"Loaded {len(self.saes)} SAE layers")
        
        # Cache for activations
        self.activations = {}
        self._hooks = []
    
    def _load_saes(self, sae_path: str) -> Dict[int, SimpleSAE]:
        """Load SAE weights quickly."""
        saes = {}
        layers = [4, 10, 16, 22, 28]  # Fixed layers
        
        for layer in layers:
            try:
                sae_file = f"{sae_path}/layers.{layer}/sae.safetensors"
                from safetensors import safe_open
                
                with safe_open(sae_file, framework="pt") as f:
                    encoder = f.get_tensor("encoder.weight").to(self.device, dtype=torch.float16)
                    encoder_bias = f.get_tensor("encoder.bias").to(self.device, dtype=torch.float16)
                
                saes[layer] = SimpleSAE(layer=layer, encoder=encoder, encoder_bias=encoder_bias)
                print(f"✓ Loaded SAE layer {layer}")
            except Exception as e:
                print(f"⚠️  Could not load SAE layer {layer}: {e}")
        
        return saes
    
    def _register_hooks(self):
        """Register hooks to capture hidden states."""
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self.activations[layer_idx] = hidden.detach()
            return hook
        
        # Register hooks for SAE layers
        for layer_idx in self.saes.keys():
            if hasattr(self.model.model, 'layers') and layer_idx < len(self.model.model.layers):
                hook = make_hook(layer_idx)
                self._hooks.append(
                    self.model.model.layers[layer_idx].register_forward_hook(hook)
                )
    
    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def trace_simple(self, prompt: str, max_tokens: int = 50) -> Dict:
        """Simple and fast circuit tracing."""
        print(f"Tracing: '{prompt[:50]}...'")
        start_time = time.time()
        
        # Clear previous activations
        self.activations = {}
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Tokenize and run forward pass
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print("Running forward pass...")
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get SAE features
            features = {}
            for layer_idx, sae in self.saes.items():
                if layer_idx in self.activations:
                    hidden = self.activations[layer_idx]
                    feat = sae.encode(hidden)
                    features[layer_idx] = feat.cpu().numpy()
                    print(f"Layer {layer_idx}: {feat.shape}")
            
            # Simple circuit analysis
            circuits = self._find_simple_circuits(features)
            
            elapsed = time.time() - start_time
            print(f"✓ Tracing completed in {elapsed:.2f}s")
            
            return {
                "prompt": prompt,
                "features": features,
                "circuits": circuits,
                "time": elapsed
            }
            
        finally:
            self._clear_hooks()
    
    def _find_simple_circuits(self, features: Dict[int, np.ndarray]) -> List[Dict]:
        """Find simple circuits between layers."""
        circuits = []
        
        if len(features) < 2:
            return circuits
        
        layers = sorted(features.keys())
        
        # Find top features in each layer
        for i, layer in enumerate(layers):
            feat = features[layer]
            # Get top 5 features by max activation
            top_features = np.argsort(feat.max(axis=(0, 1)))[-5:][::-1]
            
            print(f"Layer {layer} top features: {top_features}")
            
            # Simple connection to next layer
            if i < len(layers) - 1:
                next_layer = layers[i + 1]
                next_feat = features[next_layer]
                next_top = np.argsort(next_feat.max(axis=(0, 1)))[-5:][::-1]
                
                # Create simple circuit
                circuit = {
                    "from_layer": layer,
                    "to_layer": next_layer,
                    "from_features": top_features.tolist(),
                    "to_features": next_top.tolist(),
                    "strength": float(np.mean(feat.max(axis=(0, 1))[top_features]))
                }
                circuits.append(circuit)
        
        return circuits
    
    def analyze_financial_prompts(self, prompts: List[str]) -> Dict:
        """Analyze multiple financial prompts quickly."""
        results = {
            "total_prompts": len(prompts),
            "successful_traces": 0,
            "failed_traces": 0,
            "total_time": 0,
            "circuits": []
        }
        
        for i, prompt in enumerate(prompts):
            print(f"\n--- Analyzing prompt {i+1}/{len(prompts)} ---")
            try:
                result = self.trace_simple(prompt)
                results["successful_traces"] += 1
                results["total_time"] += result["time"]
                results["circuits"].append(result)
            except Exception as e:
                print(f"❌ Failed: {e}")
                results["failed_traces"] += 1
        
        results["avg_time"] = results["total_time"] / max(results["successful_traces"], 1)
        return results

def main():
    """Test the fast circuit tracer."""
    print("="*60)
    print("FAST CIRCUIT TRACER TEST")
    print("="*60)
    
    # Configuration
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    # Initialize tracer
    tracer = FastCircuitTracer(model_path, sae_path)
    
    # Test prompts
    test_prompts = [
        "Jamie Dimon leads JPMorgan Chase.",
        "Bank earnings rise after rate cuts.",
        "Microsoft acquires Activision for $68 billion.",
        "Federal Reserve raises interest rates.",
        "Tesla stock price increases 15%."
    ]
    
    # Run analysis
    print(f"\nAnalyzing {len(test_prompts)} financial prompts...")
    results = tracer.analyze_financial_prompts(test_prompts)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts: {results['total_prompts']}")
    print(f"Successful traces: {results['successful_traces']}")
    print(f"Failed traces: {results['failed_traces']}")
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Average time per prompt: {results['avg_time']:.2f}s")
    
    # Show sample circuits
    if results['circuits']:
        print(f"\nSample circuits from first prompt:")
        for i, circuit in enumerate(results['circuits'][0]['circuits'][:2]):
            print(f"  Circuit {i+1}: Layer {circuit['from_layer']} → Layer {circuit['to_layer']}")
            print(f"    Strength: {circuit['strength']:.4f}")
    
    print(f"\n✅ Fast circuit tracing completed!")

if __name__ == "__main__":
    main()
