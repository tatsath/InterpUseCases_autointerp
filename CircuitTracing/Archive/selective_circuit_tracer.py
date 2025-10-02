"""
Selective Circuit Tracer

Filters out always-on features and focuses on features that actually vary with different prompts.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import time
from collections import defaultdict

class SelectiveCircuitTracer:
    def __init__(self, model_path, sae_path, always_on_threshold=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.always_on_threshold = always_on_threshold  # Features active in >95% of cases are "always on"
        print(f"Using device: {self.device}")
        print(f"Always-on threshold: {self.always_on_threshold*100}%")
        
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
        
        # Load feature labels
        self.feature_labels = self._load_feature_labels()
        print(f"Loaded labels for {sum(len(labels) for labels in self.feature_labels.values())} features")
    
    def _load_feature_labels(self):
        """Load feature labels from finetuning analysis."""
        labels = {}
        
        # Layer 4
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
        
        # Layer 10
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
        
        # Layer 16
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
        
        # Layer 22
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
        
        # Layer 28
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
    
    def identify_always_on_features(self, prompts):
        """Identify features that are always on across different prompts."""
        print(f"\n{'='*80}")
        print(f"IDENTIFYING ALWAYS-ON FEATURES")
        print(f"{'='*80}")
        
        feature_activation_counts = defaultdict(lambda: defaultdict(int))
        total_activations = 0
        
        for i, prompt in enumerate(prompts):
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get SAE features for each layer
            for layer, hidden_states in enumerate(outputs.hidden_states):
                if layer in self.saes:
                    # Apply SAE
                    sae = self.saes[layer]
                    features = torch.matmul(hidden_states, sae['encoder'].T) + sae['bias']
                    features = torch.relu(features)
                    
                    # Get top features
                    feature_scores = features.max(dim=1)[0].squeeze()
                    top_features = torch.topk(feature_scores, 10)
                    
                    for feature in top_features.indices.cpu().numpy():
                        feature_activation_counts[layer][feature] += 1
                    total_activations += 1
        
        # Identify always-on features
        always_on_features = {}
        selective_features = {}
        
        for layer in [4, 10, 16, 22, 28]:
            layer_always_on = []
            layer_selective = []
            
            for feature, count in feature_activation_counts[layer].items():
                activation_rate = count / len(prompts)
                if activation_rate >= self.always_on_threshold:
                    layer_always_on.append(feature)
                else:
                    layer_selective.append((feature, count, activation_rate))
            
            always_on_features[layer] = layer_always_on
            # Sort selective features by activation count
            layer_selective.sort(key=lambda x: x[1], reverse=True)
            selective_features[layer] = layer_selective
            
            print(f"\nLayer {layer}:")
            print(f"  Always-on features ({len(layer_always_on)}): {layer_always_on}")
            print(f"  Selective features ({len(layer_selective)}): {[f[0] for f in layer_selective[:10]]}")
        
        return always_on_features, selective_features
    
    def trace_selective_features(self, prompts, top_k=10):
        """Trace only selective features (excluding always-on ones)."""
        print(f"\n{'='*80}")
        print(f"SELECTIVE CIRCUIT TRACING")
        print(f"{'='*80}")
        
        # First identify always-on features
        always_on_features, selective_features = self.identify_always_on_features(prompts)
        
        all_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*20} PROMPT {i+1}/{len(prompts)} {'='*20}")
            print(f"Tracing: '{prompt[:50]}...'")
            
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
                    top_features = torch.topk(feature_scores, 20)  # Get more to filter
                    
                    # Filter out always-on features
                    selective_top_features = []
                    selective_scores = []
                    
                    for feature, score in zip(top_features.indices.cpu().numpy(), top_features.values.cpu().numpy()):
                        if feature not in always_on_features[layer]:
                            selective_top_features.append(feature)
                            selective_scores.append(score)
                            if len(selective_top_features) >= top_k:
                                break
                    
                    results[layer] = {
                        'features': np.array(selective_top_features),
                        'scores': np.array(selective_scores),
                        'always_on_count': len(always_on_features[layer])
                    }
                    
                    print(f"\n🔹 LAYER {layer} (Top {len(selective_top_features)} Selective Features):")
                    print(f"{'─'*60}")
                    print(f"  Filtered out {len(always_on_features[layer])} always-on features")
                    
                    for j, (feature, score) in enumerate(zip(selective_top_features, selective_scores)):
                        label = self.feature_labels.get(layer, {}).get(int(feature), "Unknown feature")
                        status = "📋 LABELED" if label != "Unknown feature" else "❓ UNKNOWN"
                        print(f"  {j+1:2d}. Feature {feature:3d}: {score:6.1f} - {status}")
                        print(f"      {label}")
                        print()
            
            all_results.append(results)
        
        # Summary analysis
        self._print_selective_summary(all_results, always_on_features, selective_features, prompts)
        
        return all_results, always_on_features, selective_features
    
    def _print_selective_summary(self, all_results, always_on_features, selective_features, prompts):
        """Print summary of selective circuit tracing."""
        print(f"\n{'='*80}")
        print(f"SELECTIVE CIRCUIT TRACING SUMMARY")
        print(f"{'='*80}")
        
        # Count selective feature appearances
        selective_feature_counts = defaultdict(int)
        labeled_selective_counts = defaultdict(int)
        
        for result in all_results:
            for layer, data in result.items():
                for feature in data['features']:
                    selective_feature_counts[feature] += 1
                    if feature in self.feature_labels.get(layer, {}):
                        labeled_selective_counts[feature] += 1
        
        print(f"\n📊 MOST FREQUENTLY ACTIVATED SELECTIVE FEATURES:")
        print(f"{'─'*60}")
        for feature, count in sorted(selective_feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            total_possible = len(prompts) * 5
            percentage = (count / total_possible) * 100
            print(f"  Feature {feature:3d}: {count:2d}/{total_possible} activations ({percentage:5.1f}%)")
        
        print(f"\n📋 LABELED SELECTIVE FEATURES:")
        print(f"{'─'*60}")
        if labeled_selective_counts:
            for feature, count in sorted(labeled_selective_counts.items(), key=lambda x: x[1], reverse=True):
                total_possible = len(prompts) * 5
                percentage = (count / total_possible) * 100
                print(f"  Feature {feature:3d}: {count:2d}/{total_possible} activations ({percentage:5.1f}%)")
        else:
            print("  No labeled features in selective set")
        
        # Always-on vs selective comparison
        total_always_on = sum(len(features) for features in always_on_features.values())
        total_selective = sum(len(features) for features in selective_features.values())
        
        print(f"\n🔍 FILTERING RESULTS:")
        print(f"{'─'*60}")
        print(f"  Always-on features filtered out: {total_always_on}")
        print(f"  Selective features analyzed: {total_selective}")
        print(f"  Filtering effectiveness: {(total_always_on/(total_always_on+total_selective)*100):.1f}% of features filtered")
        
        print(f"\n{'='*80}")
        print(f"KEY INSIGHTS:")
        print(f"{'='*80}")
        print(f"• Filtered out {total_always_on} always-on features that don't vary with prompts")
        print(f"• Focused on {total_selective} selective features that show meaningful variation")
        print(f"• Circuit tracing now shows features that actually respond to different financial topics")
        print(f"• This gives a clearer picture of how the model processes different types of financial content")

def main():
    print("="*80)
    print("SELECTIVE CIRCUIT TRACER (Filtering Always-On Features)")
    print("="*80)
    
    # Setup
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    tracer = SelectiveCircuitTracer(model_path, sae_path, always_on_threshold=0.95)
    
    # Test prompts - diverse financial topics
    prompts = [
        "Jamie Dimon leads JPMorgan Chase and oversees major acquisitions",
        "Bank earnings rise after Federal Reserve rate cuts boost lending",
        "Microsoft acquires Activision Blizzard for $68.7 billion in gaming deal",
        "Federal Reserve raises interest rates to combat inflation pressures",
        "Tesla stock increases 15% following strong quarterly earnings report",
        "Goldman Sachs reports record profits from investment banking division",
        "Apple announces $90 billion stock buyback program for shareholders",
        "Cryptocurrency market crashes as Bitcoin drops below $30,000 threshold",
        "Amazon expands cloud computing services to capture enterprise market",
        "Warren Buffett's Berkshire Hathaway increases stake in energy sector"
    ]
    
    # Run selective circuit tracing
    tracer.trace_selective_features(prompts)
    
    print(f"\n✅ Selective circuit tracing completed!")

if __name__ == "__main__":
    main()
