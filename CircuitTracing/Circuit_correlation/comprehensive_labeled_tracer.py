"""
Comprehensive Labeled Circuit Tracer

Traces multiple prompts across all layers with available feature labels in a nice format.
Now with unique feature tracking across layers (feature_id_layer) and improved charts.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class ComprehensiveLabeledTracer:
    def __init__(self, model_path, sae_path, always_on_threshold=0.8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.always_on_threshold = always_on_threshold  # Features active in >80% of cases are "always on"
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
        
        # Load comprehensive feature labels
        self.feature_labels = self._load_comprehensive_labels()
        print(f"Loaded labels for {sum(len(labels) for labels in self.feature_labels.values())} features")
        
        # Track unique features across layers
        self.unique_features = set()
        self.layer_feature_mapping = defaultdict(set)  # layer -> set of feature_ids
        self.feature_layer_mapping = defaultdict(set)  # feature_id -> set of layers
    
    def _load_comprehensive_labels(self):
        """Load comprehensive feature labels from finetuning analysis."""
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
    
    def identify_always_on_features(self, prompts):
        """Identify features that are always on across different prompts."""
        print(f"\n{'='*80}")
        print(f"IDENTIFYING ALWAYS-ON FEATURES")
        print(f"{'='*80}")
        
        feature_activation_counts = defaultdict(lambda: defaultdict(int))
        
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
            print(f"  Filtering out {len(layer_always_on)}/{len(layer_always_on)+len(layer_selective)} features ({len(layer_always_on)/(len(layer_always_on)+len(layer_selective))*100:.1f}%)")
        
        return always_on_features, selective_features
    
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
    
    def trace_prompt(self, prompt, top_k=10, always_on_features=None):
        """Trace a single prompt with nice formatting, optionally filtering always-on features."""
        print(f"\n{'='*80}")
        print(f"TRACING: '{prompt}'")
        print(f"{'='*80}")
        
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
                
                # Filter out always-on features if provided
                if always_on_features is not None:
                    selective_features = []
                    selective_scores = []
                    
                    for feature, score in zip(top_features.indices.cpu().numpy(), top_features.values.cpu().numpy()):
                        if feature not in always_on_features.get(layer, []):
                            selective_features.append(feature)
                            selective_scores.append(score)
                            if len(selective_features) >= top_k:
                                break
                    
                    final_features = np.array(selective_features)
                    final_scores = np.array(selective_scores)
                    filtered_count = len(always_on_features.get(layer, []))
                else:
                    final_features = top_features.indices.cpu().numpy()[:top_k]
                    final_scores = top_features.values.cpu().numpy()[:top_k]
                    filtered_count = 0
                
                # Additional filtering: remove features that are too common across all prompts
                if len(final_features) > 0 and hasattr(self, '_prompt_count') and hasattr(self, '_feature_global_counts'):
                    # Keep only features that appear in less than 70% of prompts
                    max_common_threshold = int(self._prompt_count * 0.7)
                    very_common_features = set()
                    for feature in final_features:
                        if self._feature_global_counts.get(feature, 0) >= max_common_threshold:
                            very_common_features.add(feature)
                    
                    if very_common_features:
                        mask = ~np.isin(final_features, list(very_common_features))
                        final_features = final_features[mask]
                        final_scores = final_scores[mask]
                        filtered_count += len(very_common_features)
                
                results[layer] = {
                    'features': final_features,
                    'scores': final_scores
                }
                
                print(f"\n🔹 LAYER {layer} (Top {len(final_features)} Features):")
                print(f"{'─'*60}")
                if filtered_count > 0:
                    print(f"  Filtered out {filtered_count} always-on features")
                
                for i, (feature, score) in enumerate(zip(final_features, final_scores)):
                    # Create unique feature ID: feature_id_layer
                    unique_feature_id = f"{feature}_{layer}"
                    self.unique_features.add(unique_feature_id)
                    self.layer_feature_mapping[layer].add(feature)
                    self.feature_layer_mapping[feature].add(layer)
                    
                    label = self.feature_labels.get(layer, {}).get(int(feature), "Unknown feature")
                    status = "📋 LABELED" if label != "Unknown feature" else "❓ UNKNOWN"
                    print(f"  {i+1:2d}. Feature {feature:3d} (ID: {unique_feature_id}): {score:6.1f} - {status}")
                    print(f"      {label}")
                    print()
        
        return results
    
    def analyze_multiple_prompts(self, prompts, filter_always_on=True):
        """Analyze multiple prompts and show comprehensive results."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE CIRCUIT TRACING ANALYSIS")
        print(f"{'='*80}")
        print(f"Analyzing {len(prompts)} prompts across 5 layers")
        print(f"Total features per layer: 400")
        print(f"Labeled features per layer: 10")
        if filter_always_on:
            print(f"Filtering always-on features (threshold: {self.always_on_threshold*100}%)")
        print(f"{'='*80}")
        
        # First identify always-on features if filtering is enabled
        always_on_features = None
        if filter_always_on:
            always_on_features, selective_features = self.identify_always_on_features(prompts)
        
        # Track global feature counts across all prompts
        self._prompt_count = len(prompts)
        self._feature_global_counts = defaultdict(int)
        
        all_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*20} PROMPT {i+1}/{len(prompts)} {'='*20}")
            results = self.trace_prompt(prompt, always_on_features=always_on_features)
            all_results.append(results)
            
            # Track global feature counts
            for layer, data in results.items():
                for feature in data['features']:
                    self._feature_global_counts[feature] += 1
        
        # Summary analysis
        self._print_summary_analysis(all_results, prompts, always_on_features)
        
        # Create improved charts
        self._create_improved_charts(all_results, prompts, always_on_features)
        
        return all_results
    
    def _print_summary_analysis(self, all_results, prompts, always_on_features=None):
        """Print comprehensive summary analysis."""
        print(f"\n{'='*80}")
        print(f"SUMMARY ANALYSIS")
        print(f"{'='*80}")
        
        # Count feature appearances across all prompts and layers
        feature_counts = defaultdict(int)
        labeled_feature_counts = defaultdict(int)
        layer_feature_counts = defaultdict(lambda: defaultdict(int))
        
        for result in all_results:
            for layer, data in result.items():
                for feature in data['features']:
                    feature_counts[feature] += 1
                    layer_feature_counts[layer][feature] += 1
                    
                    # Check if this feature has a label
                    if feature in self.feature_labels.get(layer, {}):
                        labeled_feature_counts[feature] += 1
        
        # Most common features overall
        print(f"\n📊 MOST FREQUENTLY ACTIVATED FEATURES:")
        print(f"{'─'*60}")
        for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            total_possible = len(prompts) * 5  # 5 layers
            percentage = (count / total_possible) * 100
            print(f"  Feature {feature:3d}: {count:2d}/{total_possible} activations ({percentage:5.1f}%)")
        
        # Labeled features that are active
        print(f"\n📋 LABELED FEATURES THAT ARE ACTIVE:")
        print(f"{'─'*60}")
        if labeled_feature_counts:
            for feature, count in sorted(labeled_feature_counts.items(), key=lambda x: x[1], reverse=True):
                total_possible = len(prompts) * 5
                percentage = (count / total_possible) * 100
                print(f"  Feature {feature:3d}: {count:2d}/{total_possible} activations ({percentage:5.1f}%)")
        else:
            print("  No labeled features are consistently active across prompts")
        
        # Always-on vs selective comparison
        if always_on_features is not None:
            total_always_on = sum(len(features) for features in always_on_features.values())
            print(f"\n🔍 FILTERING RESULTS:")
            print(f"{'─'*60}")
            print(f"  Always-on features filtered out: {total_always_on}")
            print(f"  Selective features analyzed: {len(feature_counts)}")
            print(f"  Filtering effectiveness: {(total_always_on/(total_always_on+len(feature_counts))*100):.1f}% of features filtered")
        
        # Layer-specific analysis
        print(f"\n🔹 LAYER-SPECIFIC FEATURE PATTERNS:")
        print(f"{'─'*60}")
        for layer in [4, 10, 16, 22, 28]:
            layer_features = layer_feature_counts[layer]
            top_features = sorted(layer_features.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Layer {layer:2d}: {[f[0] for f in top_features]}")
        
        # Circuit flow analysis - show layer-specific feature details
        print(f"\n🔄 CIRCUIT FLOW ANALYSIS:")
        print(f"{'─'*60}")
        
        # Find features that appear in multiple layers
        multi_layer_features = defaultdict(set)
        for result in all_results:
            for layer, data in result.items():
                for feature in data['features']:
                    multi_layer_features[feature].add(layer)
        
        circuit_features = {f: layers for f, layers in multi_layer_features.items() if len(layers) > 1}
        
        print(f"  Features active across multiple layers: {len(circuit_features)}")
        print(f"\n  ⚠️  IMPORTANT: Same feature numbers across layers represent DIFFERENT concepts!")
        print(f"  {'─'*60}")
        
        for feature, layers in sorted(circuit_features.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            layer_list = sorted(list(layers))
            print(f"\n  🔹 Feature {feature:3d} (appears in {len(layers)} layers: {layer_list}):")
            print(f"      ⚠️  WARNING: Feature {feature} in Layer {layer_list[0]} is NOT the same as")
            print(f"          Feature {feature} in Layer {layer_list[-1]} - they are completely different!")
            
            # Show what this feature represents in each layer
            for layer in layer_list:
                label = self.feature_labels.get(layer, {}).get(feature, "Unknown feature")
                status = "📋 LABELED" if label != "Unknown feature" else "❓ UNKNOWN"
                print(f"    Layer {layer:2d}: {status} - {label}")
                print(f"         (This is a DIFFERENT feature from the same number in other layers)")
        
        # Show most active features per layer
        print(f"\n  🎯 MOST ACTIVE FEATURES PER LAYER:")
        print(f"  {'─'*60}")
        for layer in [4, 10, 16, 22, 28]:
            layer_features = layer_feature_counts[layer]
            if layer_features:
                top_feature = max(layer_features.items(), key=lambda x: x[1])
                feature_id, count = top_feature
                label = self.feature_labels.get(layer, {}).get(feature_id, "Unknown feature")
                status = "📋 LABELED" if label != "Unknown feature" else "❓ UNKNOWN"
                print(f"    Layer {layer:2d}: Feature {feature_id:3d} ({count:2d} activations) - {status}")
                print(f"            {label}")
        
        print(f"\n{'='*80}")
        print(f"KEY INSIGHTS:")
        print(f"{'='*80}")
        print(f"• Total features analyzed: {len(feature_counts)}")
        print(f"• Labeled features available: {sum(len(labels) for labels in self.feature_labels.values())}")
        print(f"• Features with circuit flow: {len(circuit_features)}")
        if feature_counts:
            print(f"• Most active feature: {max(feature_counts.items(), key=lambda x: x[1])[0]}")
        if always_on_features is not None:
            total_always_on = sum(len(features) for features in always_on_features.values())
            print(f"• Always-on features filtered: {total_always_on}")
            print(f"• Circuit tracing now focuses on features that actually vary with prompts!")
        else:
            print(f"• Circuit tracing successfully identifies feature flow across layers!")
    
    def _create_improved_charts(self, all_results, prompts, always_on_features=None):
        """Create improved charts for visualization."""
        print(f"\n{'='*80}")
        print(f"CREATING IMPROVED CHARTS")
        print(f"{'='*80}")
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Circuit Tracing Analysis - Improved Charts', fontsize=20, fontweight='bold')
        
        # 1. Feature Distribution Across Layers
        ax1 = plt.subplot(2, 3, 1)
        self._plot_feature_distribution(ax1, all_results)
        
        # 2. Layer-specific Feature Activity Heatmap
        ax2 = plt.subplot(2, 3, 2)
        self._plot_layer_heatmap(ax2, all_results)
        
        # 3. Circuit Flow Network
        ax3 = plt.subplot(2, 3, 3)
        self._plot_circuit_flow(ax3, all_results)
        
        # 4. Feature Activation Frequency
        ax4 = plt.subplot(2, 3, 4)
        self._plot_activation_frequency(ax4, all_results)
        
        # 5. Filtering Effectiveness
        ax5 = plt.subplot(2, 3, 5)
        self._plot_filtering_effectiveness(ax5, always_on_features)
        
        # 6. Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_stats(ax6, all_results, prompts)
        
        plt.tight_layout()
        plt.savefig('circuit_tracing_improved_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Improved charts saved as 'circuit_tracing_improved_charts.png'")
    
    def _plot_feature_distribution(self, ax, all_results):
        """Plot feature distribution across layers."""
        ax.set_title('Feature Distribution Across Layers', fontsize=14, fontweight='bold')
        
        # Count features per layer
        layer_counts = defaultdict(int)
        for result in all_results:
            for layer, data in result.items():
                layer_counts[layer] += len(data['features'])
        
        layers = sorted(layer_counts.keys())
        counts = [layer_counts[layer] for layer in layers]
        
        bars = ax.bar([f'L{layer}' for layer in layers], counts, 
                     color='skyblue', alpha=0.7, edgecolor='navy')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Number of Active Features')
        ax.grid(True, alpha=0.3)
    
    def _plot_layer_heatmap(self, ax, all_results):
        """Plot layer-specific feature activity heatmap."""
        ax.set_title('Layer-Specific Feature Activity', fontsize=14, fontweight='bold')
        
        # Create feature-layer matrix
        all_features = set()
        for result in all_results:
            for layer, data in result.items():
                all_features.update(data['features'])
        
        all_features = sorted(list(all_features))
        layers = [4, 10, 16, 22, 28]
        
        # Create activity matrix
        activity_matrix = np.zeros((len(all_features), len(layers)))
        
        for result in all_results:
            for layer, data in result.items():
                layer_idx = layers.index(layer)
                for feature in data['features']:
                    feature_idx = all_features.index(feature)
                    activity_matrix[feature_idx, layer_idx] += 1
        
        # Create heatmap
        im = ax.imshow(activity_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{layer}' for layer in layers])
        ax.set_yticks(range(0, len(all_features), max(1, len(all_features)//10)))
        ax.set_yticklabels([f'F{all_features[i]}' for i in range(0, len(all_features), max(1, len(all_features)//10))])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Activation Count')
    
    def _plot_circuit_flow(self, ax, all_results):
        """Plot circuit flow network."""
        ax.set_title('Circuit Flow Network\n⚠️ Same feature numbers = DIFFERENT concepts!', fontsize=14, fontweight='bold')
        
        # Find features that appear in multiple layers
        multi_layer_features = defaultdict(set)
        for result in all_results:
            for layer, data in result.items():
                for feature in data['features']:
                    multi_layer_features[feature].add(layer)
        
        circuit_features = {f: layers for f, layers in multi_layer_features.items() if len(layers) > 1}
        
        # Draw layers as vertical lines
        layers = [4, 10, 16, 22, 28]
        layer_positions = {layer: i for i, layer in enumerate(layers)}
        
        for i, layer in enumerate(layers):
            ax.axvline(x=i, ymin=0, ymax=1, color='black', linewidth=2, alpha=0.7)
            ax.text(i, -0.05, f'L{layer}', ha='center', va='top', fontweight='bold')
        
        # Draw feature flows
        colors = plt.cm.Set3(np.linspace(0, 1, len(circuit_features)))
        
        for i, (feature, layer_set) in enumerate(circuit_features.items()):
            color = colors[i]
            y_pos = 0.9 - (i * 0.08)  # Spread features vertically
            
            # Convert set to sorted list for proper ordering
            layer_list = sorted(list(layer_set))
            
            # Draw horizontal lines for features
            for j in range(len(layer_list) - 1):
                start_layer = layer_list[j]
                end_layer = layer_list[j + 1]
                start_x = layer_positions[start_layer]
                end_x = layer_positions[end_layer]
                
                ax.plot([start_x, end_x], [y_pos, y_pos], 
                       color=color, linewidth=3, alpha=0.8)
                
                # Add arrow
                ax.annotate('', xy=(end_x, y_pos), xytext=(start_x, y_pos),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2))
            
            # Add feature label with warning
            ax.text(-0.1, y_pos, f'F{feature}\n(DIFFERENT\nper layer)', ha='right', va='center', 
                   fontweight='bold', color=color, fontsize=8)
        
        ax.set_xlim(-0.4, len(layers) - 0.7)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'L{layer}' for layer in layers])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    def _plot_activation_frequency(self, ax, all_results):
        """Plot feature activation frequency."""
        ax.set_title('Top Features by Activation Frequency', fontsize=14, fontweight='bold')
        
        # Count feature activations
        feature_counts = defaultdict(int)
        for result in all_results:
            for layer, data in result.items():
                for feature in data['features']:
                    feature_counts[feature] += 1
        
        # Get top features
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        features = [f[0] for f in top_features]
        counts = [f[1] for f in top_features]
        
        bars = ax.bar(range(len(features)), counts, color='lightgreen', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Activation Count')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([f'F{feature}' for feature in features], rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_filtering_effectiveness(self, ax, always_on_features):
        """Plot filtering effectiveness."""
        ax.set_title('Filtering Effectiveness', fontsize=14, fontweight='bold')
        
        if always_on_features is not None:
            total_always_on = sum(len(features) for features in always_on_features.values())
            total_selective = len(self.unique_features) - total_always_on
            
            categories = ['Always-on\nFeatures', 'Selective\nFeatures']
            counts = [total_always_on, total_selective]
            colors = ['lightcoral', 'lightgreen']
            
            bars = ax.bar(categories, counts, color=colors, alpha=0.7)
            
            # Add percentage labels
            total = sum(counts)
            for bar, count in zip(bars, counts):
                percentage = (count / total) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12)
            
            ax.set_ylabel('Number of Features')
            ax.set_ylim(0, max(counts) * 1.2)
        else:
            ax.text(0.5, 0.5, 'No filtering applied', ha='center', va='center', 
                   fontsize=14, transform=ax.transAxes)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_stats(self, ax, all_results, prompts):
        """Plot summary statistics."""
        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Calculate statistics
        total_features = len(self.unique_features)
        total_prompts = len(prompts)
        total_layers = 5
        
        # Find most active feature
        feature_counts = defaultdict(int)
        for result in all_results:
            for layer, data in result.items():
                for feature in data['features']:
                    feature_counts[feature] += 1
        
        most_active = max(feature_counts.items(), key=lambda x: x[1])[0] if feature_counts else "N/A"
        
        stats_text = f"""CIRCUIT TRACING STATISTICS:
        
Total Unique Features: {total_features}
Total Prompts Analyzed: {total_prompts}
Total Layers: {total_layers}

Most Active Feature: {most_active}
Features with Circuit Flow: {len([f for f, layers in self.feature_layer_mapping.items() if len(layers) > 1])}

LAYER COVERAGE:
{chr(10).join([f"Layer {layer}: {len(self.layer_feature_mapping[layer])} features" for layer in [4, 10, 16, 22, 28]])}

⚠️  CRITICAL UNDERSTANDING:
• Feature 56 in Layer 4 ≠ Feature 56 in Layer 10
• Same feature numbers = DIFFERENT concepts per layer
• Each layer has its own SAE with different weights
• Circuit flow shows COINCIDENTAL feature number matches

KEY INSIGHTS:
• Features are tracked with unique IDs (feature_id_layer)
• Each feature ID represents a different concept per layer
• "Circuit flow" is misleading - these are separate features
• Filtering focuses on selective, meaningful features"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def main():
    print("="*80)
    print("COMPREHENSIVE LABELED CIRCUIT TRACER - IMPROVED VERSION")
    print("="*80)
    
    # Setup
    model_path = "meta-llama/Llama-2-7b-hf"
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    tracer = ComprehensiveLabeledTracer(model_path, sae_path)
    
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
    
    # Run comprehensive analysis
    tracer.analyze_multiple_prompts(prompts)
    
    print(f"\n✅ Comprehensive labeled tracing with improved charts completed!")

if __name__ == "__main__":
    main()