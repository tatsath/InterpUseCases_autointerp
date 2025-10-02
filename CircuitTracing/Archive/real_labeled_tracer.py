"""
Real Labeled Circuit Tracer

Uses the actual "Label (Base Model)" from the finetuning analysis README.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import time

class RealLabeledTracer:
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
        
        # Load REAL feature labels from finetuning analysis
        self.feature_labels = self._load_real_labels()
        print(f"Loaded REAL labels for {len(self.feature_labels)} features")
    
    def _load_real_labels(self):
        """Load REAL feature labels from the finetuning analysis README."""
        labels = {}
        
        # Layer 4 - REAL labels from README (Base Model)
        labels[4] = {
            299: "Intellectual or professional achievements and experience",
            32: "Punctuation and syntax markers in language",
            347: "Investment advice or guidance",
            176: "Technology and Innovation",
            335: "Financial Market Indicators",
            362: "Recognition of names and titles as indicators of respect",
            269: "Financial or Business Terminology",
            387: "Representation of possessive or contracted forms in language",
            312: "Financial market symbols and punctuation",
            209: "Cryptocurrency market instability and skepticism"
        }
        
        # Layer 10 - REAL labels from README (Base Model)
        labels[10] = {
            83: "Specific textual references or citations",
            162: "Economic growth and inflation trends in the tech industry",
            91: "A transitional or explanatory phrase indicating a change",
            266: "Financial dividend payout terminology",
            318: "Symbolic representations of monetary units or financial concepts",
            105: "Relationship between entities",
            310: "Article title references",
            320: "Time frame or duration",
            131: "Financial news sources and publications",
            17: "Financial market terminology and stock-related jargon"
        }
        
        # Layer 16 - REAL labels from README (Base Model)
        labels[16] = {
            389: "Specific numerical values associated with financial data",
            85: "Dates and financial numbers in business and economic contexts",
            385: "Financial Market Analysis",
            279: "Comma-separated clauses or phrases indicating transactions",
            18: "Quotation marks indicating direct speech or quotes",
            355: "Financial Market News and Analysis",
            283: "Quantifiable aspects of change or occurrence",
            121: "Temporal progression or continuation of a process",
            107: "Market-related terminology",
            228: "Company names and stock-related terminology"
        }
        
        # Layer 22 - REAL labels from README (Base Model)
        labels[22] = {
            159: "Article titles and stock market-related keywords",
            258: "Temporal relationships and causal connections between events",
            116: "Names or Identifiers are being highlighted",
            186: "Relationship or Connection between entities",
            141: "Business relationships or partnerships",
            323: "Comparative relationships and transitional concepts",
            90: "Temporal Market Dynamics",
            252: "Geographic or Topographic Features and Names",
            157: "Temporal or sequential relationships between events",
            353: "Financial concepts and metrics are represented"
        }
        
        # Layer 28 - REAL labels from README (Base Model)
        labels[28] = {
            116: "Prepositional phrases indicating direction or relationship",
            375: "Punctuation marks and word boundaries",
            276: "Assertion of existence or state",
            345: "Financial Market and Business Terminology",
            305: "Continuity or persistence in economic trends",
            287: "Patterns of linguistic and semantic relationships",
            19: "Acronyms and abbreviations for technology and business",
            103: "Prepositions and conjunctions indicating relationships",
            178: "Connection between entities or concepts",
            121: "Specific entities or concepts related to the context"
        }
        
        return labels
    
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
    
    def trace(self, prompt, top_k=10):
        """Trace with REAL feature labels from finetuning analysis."""
        print(f"\nTracing: '{prompt[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get SAE features for each layer
        results = {}
        for layer, hidden_states in enumerate(outputs.hidden_states):
            if layer in self.saes:
                # Apply SAE
                sae = self.saes[layer]
                features = torch.matmul(hidden_states, sae['encoder'].T) + sae['bias']
                features = torch.relu(features)
                
                # Get top features
                feature_scores = features.max(dim=1)[0].squeeze()
                top_features = torch.topk(feature_scores, top_k)
                
                results[layer] = {
                    'features': top_features.indices.cpu().numpy(),
                    'scores': top_features.values.cpu().numpy()
                }
                
                print(f"\nLayer {layer}:")
                for i, (feature, score) in enumerate(zip(top_features.indices.cpu().numpy()[:5], top_features.values.cpu().numpy()[:5])):
                    label = self.feature_labels.get(layer, {}).get(int(feature), "Unknown feature")
                    print(f"  {i+1}. Feature {feature}: {score:.1f} - {label}")
        
        return results
    
    def find_connections(self, results):
        """Find connections with REAL labels."""
        print("\n=== FEATURE CONNECTIONS (REAL LABELS) ===")
        
        # Get all features across layers
        all_features = {}
        for layer, data in results.items():
            for i, feature in enumerate(data['features']):
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append((layer, data['scores'][i]))
        
        # Find features that appear in multiple layers
        connections = []
        for feature, appearances in all_features.items():
            if len(appearances) > 1:
                layers = [layer for layer, score in appearances]
                scores = [score for layer, score in appearances]
                connections.append({
                    'feature': int(feature),
                    'layers': layers,
                    'scores': scores,
                    'strength': sum(scores)
                })
        
        # Sort by strength
        connections.sort(key=lambda x: x['strength'], reverse=True)
        
        print(f"Found {len(connections)} features active across multiple layers:")
        for conn in connections[:10]:
            # Get REAL label from the first layer where it appears
            first_layer = conn['layers'][0]
            label = self.feature_labels.get(first_layer, {}).get(conn['feature'], "Unknown feature")
            print(f"Feature {conn['feature']}: {label}")
            print(f"  → Active in layers {conn['layers']} (strength: {conn['strength']:.2f})")
        
        return connections

def main():
    print("="*70)
    print("REAL LABELED CIRCUIT TRACER (Using Base Model Labels)")
    print("="*70)
    
    # Setup
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    tracer = RealLabeledTracer(model_path, sae_path)
    
    # Test prompts
    prompts = [
        "Jamie Dimon leads JPMorgan Chase",
        "Bank earnings rise after rate cuts",
        "Microsoft acquires Activision",
        "Federal Reserve raises rates",
        "Tesla stock increases 15%"
    ]
    
    # Trace each prompt
    all_connections = []
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*20} PROMPT {i+1} {'='*20}")
        results = tracer.trace(prompt)
        connections = tracer.find_connections(results)
        all_connections.extend(connections)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (Using REAL Base Model Labels)")
    print(f"{'='*70}")
    print(f"Analyzed {len(prompts)} prompts")
    print(f"Found {len(all_connections)} total feature connections")
    
    # Most common features with REAL labels
    feature_counts = {}
    for conn in all_connections:
        feature = conn['feature']
        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    print(f"\nMost common features across all prompts:")
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        # Get REAL label from any layer
        label = "Unknown"
        for layer in [4, 10, 16, 22, 28]:
            if feature in tracer.feature_labels.get(layer, {}):
                label = tracer.feature_labels[layer][feature]
                break
        print(f"Feature {feature}: {label} (appears in {count} prompts)")
    
    print(f"\n✅ Real labeled tracing completed!")

if __name__ == "__main__":
    main()
