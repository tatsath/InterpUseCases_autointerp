import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

class FeatureActivationTracker:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sae_weights = {}
        self.activation_data = {}
        
        # Load feature data from the finetuned model analysis
        self.feature_data = self._load_finetuned_feature_data()
        
    def _load_finetuned_feature_data(self) -> Dict:
        """Load feature data for base Llama-2-7B model based on analysis results."""
        return {
            4: {
                "features": [
                    {"id": 25, "label": "General language patterns and common words", "f1_score": 0.85, "activation_improvement": 0.0},
                    {"id": 299, "label": "Intellectual or professional achievements", "f1_score": 0.82, "activation_improvement": 0.0},
                    {"id": 32, "label": "Punctuation and syntax markers", "f1_score": 0.76, "activation_improvement": 0.0},
                    {"id": 347, "label": "Investment advice or guidance", "f1_score": 0.61, "activation_improvement": 0.0},
                    {"id": 176, "label": "Technology and Innovation", "f1_score": 0.78, "activation_improvement": 0.0},
                    {"id": 335, "label": "Financial Market Indicators", "f1_score": 0.92, "activation_improvement": 0.0},
                    {"id": 362, "label": "Recognition of names and titles", "f1_score": 0.76, "activation_improvement": 0.0},
                    {"id": 269, "label": "Financial or Business Terminology", "f1_score": 0.84, "activation_improvement": 0.0},
                    {"id": 387, "label": "Possessive or contracted forms", "f1_score": 0.68, "activation_improvement": 0.0},
                    {"id": 312, "label": "Financial market symbols and punctuation", "f1_score": 0.82, "activation_improvement": 0.0}
                ]
            },
            10: {
                "features": [
                    {"id": 389, "label": "Specific numerical values and financial data", "f1_score": 0.79, "activation_improvement": 0.0},
                    {"id": 83, "label": "Specific textual references or citations", "f1_score": 0.69, "activation_improvement": 0.0},
                    {"id": 162, "label": "Economic growth and inflation trends", "f1_score": 0.00, "activation_improvement": 0.0},
                    {"id": 91, "label": "Transitional or explanatory phrases", "f1_score": 0.76, "activation_improvement": 0.0},
                    {"id": 266, "label": "Financial dividend payout terminology", "f1_score": 0.66, "activation_improvement": 0.0},
                    {"id": 318, "label": "Symbolic representations of monetary units", "f1_score": 0.79, "activation_improvement": 0.0},
                    {"id": 105, "label": "Relationship between entities", "f1_score": 0.68, "activation_improvement": 0.0},
                    {"id": 310, "label": "Article title references", "f1_score": 0.83, "activation_improvement": 0.0},
                    {"id": 320, "label": "Time frame or duration", "f1_score": 0.64, "activation_improvement": 0.0},
                    {"id": 131, "label": "Financial news sources and publications", "f1_score": 0.80, "activation_improvement": 0.0}
                ]
            },
            16: {
                "features": [
                    {"id": 214, "label": "General language processing patterns", "f1_score": 0.85, "activation_improvement": 0.0},
                    {"id": 389, "label": "Specific numerical values and financial data", "f1_score": 0.79, "activation_improvement": 0.0},
                    {"id": 85, "label": "Dates and financial numbers in business", "f1_score": 0.69, "activation_improvement": 0.0},
                    {"id": 385, "label": "Financial Market Analysis", "f1_score": 0.80, "activation_improvement": 0.0},
                    {"id": 279, "label": "Comma-separated clauses or phrases", "f1_score": 0.89, "activation_improvement": 0.0},
                    {"id": 18, "label": "Quotation marks indicating direct speech", "f1_score": 0.53, "activation_improvement": 0.0},
                    {"id": 355, "label": "Financial Market News and Analysis", "f1_score": 0.33, "activation_improvement": 0.0},
                    {"id": 283, "label": "Quantifiable aspects of change", "f1_score": 0.83, "activation_improvement": 0.0},
                    {"id": 121, "label": "Temporal progression or continuation", "f1_score": 0.78, "activation_improvement": 0.0},
                    {"id": 107, "label": "Market-related terminology", "f1_score": 0.91, "activation_improvement": 0.0}
                ]
            },
            22: {
                "features": [
                    {"id": 290, "label": "Advanced language understanding patterns", "f1_score": 0.88, "activation_improvement": 0.0},
                    {"id": 294, "label": "Complex semantic relationships", "f1_score": 0.85, "activation_improvement": 0.0},
                    {"id": 159, "label": "Article titles and market keywords", "f1_score": 0.84, "activation_improvement": 0.0},
                    {"id": 258, "label": "Temporal relationships and causal connections", "f1_score": 0.80, "activation_improvement": 0.0},
                    {"id": 116, "label": "Names or Identifiers being highlighted", "f1_score": 0.68, "activation_improvement": 0.0},
                    {"id": 186, "label": "Relationship or Connection between entities", "f1_score": 0.76, "activation_improvement": 0.0},
                    {"id": 141, "label": "Business relationships or partnerships", "f1_score": 0.72, "activation_improvement": 0.0},
                    {"id": 323, "label": "Comparative relationships and transitions", "f1_score": 0.75, "activation_improvement": 0.0},
                    {"id": 90, "label": "Temporal Market Dynamics", "f1_score": 0.76, "activation_improvement": 0.0},
                    {"id": 252, "label": "Geographic or Topographic Features", "f1_score": 0.31, "activation_improvement": 0.0}
                ]
            },
            28: {
                "features": [
                    {"id": 134, "label": "High-level semantic understanding", "f1_score": 0.90, "activation_improvement": 0.0},
                    {"id": 294, "label": "Complex semantic relationships", "f1_score": 0.85, "activation_improvement": 0.0},
                    {"id": 116, "label": "Prepositional phrases indicating direction", "f1_score": 0.63, "activation_improvement": 0.0},
                    {"id": 375, "label": "Punctuation marks and word boundaries", "f1_score": 0.90, "activation_improvement": 0.0},
                    {"id": 276, "label": "Assertion of existence or state", "f1_score": 0.78, "activation_improvement": 0.0},
                    {"id": 345, "label": "Financial Market and Business Terminology", "f1_score": 0.84, "activation_improvement": 0.0},
                    {"id": 305, "label": "Continuity or persistence in trends", "f1_score": 0.76, "activation_improvement": 0.0},
                    {"id": 287, "label": "Patterns of linguistic relationships", "f1_score": 0.79, "activation_improvement": 0.0},
                    {"id": 19, "label": "Acronyms and abbreviations", "f1_score": 0.00, "activation_improvement": 0.0},
                    {"id": 103, "label": "Prepositions and conjunctions", "f1_score": 0.84, "activation_improvement": 0.0}
                ]
            }
        }
    
    @st.cache_resource
    def load_model_and_tokenizer(_self):
        """Load the model and tokenizer with caching."""
        with st.spinner("Loading model and tokenizer..."):
            model_path = "meta-llama/Llama-2-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set padding token for Llama tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            return model, tokenizer
    
    def load_sae_weights(_self, layer_idx: int):
        """Load SAE weights for a specific layer with caching."""
        sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        layer_path = os.path.join(sae_path, f"layers.{layer_idx}")
        
        try:
            # Load SAE weights from safetensors file
            sae_file = os.path.join(layer_path, "sae.safetensors")
            
            if os.path.exists(sae_file):
                with safe_open(sae_file, framework="pt", device="cpu") as f:
                    encoder = f.get_tensor("encoder.weight")
                    encoder_bias = f.get_tensor("encoder.bias")
                    decoder = f.get_tensor("W_dec")
                    decoder_bias = f.get_tensor("b_dec")
                
                return {
                    "encoder": encoder,
                    "encoder_bias": encoder_bias,
                    "decoder": decoder,
                    "decoder_bias": decoder_bias
                }
            else:
                st.warning(f"SAE weights not found for layer {layer_idx} at {sae_file}")
                return None
        except Exception as e:
            st.warning(f"Error loading SAE weights for layer {layer_idx}: {str(e)}")
            return None
    
    def get_activations(self, text: str, layers: List[int], aggregation_method: str = "max") -> Dict[int, np.ndarray]:
        """Get feature activations for the given text across specified layers."""
        if self.model is None or self.tokenizer is None:
            return {}
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Pre-load SAE weights for all layers to avoid loading in hook
        sae_weights_cache = {}
        for layer in layers:
            sae_weights = self.load_sae_weights(layer)
            if sae_weights is not None:
                sae_weights_cache[layer] = sae_weights
                # Debug: Check SAE weights for layer 28
                if layer == 28:
                    encoder = sae_weights["encoder"]
                    print(f"DEBUG: SAE weights loaded for layer 28")
                    print(f"DEBUG: Encoder shape: {encoder.shape}")
                    print(f"DEBUG: Encoder dtype: {encoder.dtype}")
                    print(f"DEBUG: Encoder max: {encoder.max().item():.4f}, min: {encoder.min().item():.4f}")
                    print(f"DEBUG: Encoder mean: {encoder.mean().item():.4f}, std: {encoder.std().item():.4f}")
                    
                    # Check if encoder row 116 is extreme
                    if 116 < encoder.shape[0]:
                        encoder_row_116 = encoder[116]
                        print(f"DEBUG: Encoder row 116 stats:")
                        print(f"DEBUG:   Max: {encoder_row_116.max().item():.4f}, Min: {encoder_row_116.min().item():.4f}")
                        print(f"DEBUG:   Mean: {encoder_row_116.mean().item():.4f}, Std: {encoder_row_116.std().item():.4f}")
                        print(f"DEBUG:   Norm: {torch.norm(encoder_row_116).item():.4f}")
                        
                        # Compare with other rows
                        other_rows = encoder[encoder != encoder_row_116.unsqueeze(0)]
                        if len(other_rows) > 0:
                            print(f"DEBUG: Other rows mean norm: {torch.norm(encoder, dim=1).mean().item():.4f}")
                            print(f"DEBUG: Row 116 norm vs others: {torch.norm(encoder_row_116).item():.4f} vs {torch.norm(encoder, dim=1).mean().item():.4f}")
            else:
                print(f"DEBUG: Failed to load SAE weights for layer {layer}")
        
        activations = {}
        
        # Forward pass with hidden states output
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states_list = outputs.hidden_states
        
        # Process each requested layer
        print(f"DEBUG: Hidden states list length: {len(hidden_states_list)}")
        print(f"DEBUG: Requested layers: {layers}")
        print(f"DEBUG: SAE weights cache keys: {list(sae_weights_cache.keys())}")
        
        for layer_idx in layers:
            print(f"DEBUG: Processing layer {layer_idx}")
            if layer_idx in sae_weights_cache and layer_idx < len(hidden_states_list):
                # Get hidden states for this layer (layer_idx + 1 because layer 0 is embeddings)
                hidden_states = hidden_states_list[layer_idx + 1]
                print(f"DEBUG: Hidden states shape for layer {layer_idx}: {hidden_states.shape}")
                
                # Use pre-loaded SAE weights
                sae_weights = sae_weights_cache[layer_idx]
                encoder = sae_weights["encoder"]
                encoder = encoder.to(hidden_states.device).to(hidden_states.dtype)
                
                # Compute activations: (batch_size, seq_len, n_features)
                feature_activations = torch.matmul(hidden_states, encoder.T)
                
                # Add encoder bias if available
                if "encoder_bias" in sae_weights:
                    encoder_bias = sae_weights["encoder_bias"].to(hidden_states.device).to(hidden_states.dtype)
                    feature_activations = feature_activations + encoder_bias
                
                # Exclude the first token (BOS token) as it often has always-on features
                # that don't vary with content
                content_activations = feature_activations[:, 1:, :]  # Skip first token
                
                # Apply different aggregation methods based on user choice
                if aggregation_method == "max":
                    # Use max activation across content tokens - preserves most important activations
                    layer_activations = content_activations.max(dim=1)[0].squeeze(0)
                elif aggregation_method == "mean":
                    # Use mean activation across content tokens
                    layer_activations = content_activations.mean(dim=1).squeeze(0)
                elif aggregation_method == "sum":
                    # Use sum activation across content tokens
                    layer_activations = content_activations.sum(dim=1).squeeze(0)
                elif aggregation_method == "combined":
                    # Use weighted combination of mean and max
                    mean_activations = content_activations.mean(dim=1).squeeze(0)
                    max_activations = content_activations.max(dim=1)[0].squeeze(0)
                    layer_activations = 0.7 * max_activations + 0.3 * mean_activations
                elif aggregation_method == "normalized":
                    # Use max activation but normalize to reduce always-on feature dominance
                    max_activations = content_activations.max(dim=1)[0].squeeze(0)
                    # Normalize by subtracting mean and dividing by std
                    mean_act = max_activations.mean()
                    std_act = max_activations.std()
                    layer_activations = (max_activations - mean_act) / (std_act + 1e-8)
                else:
                    # Default to max
                    layer_activations = content_activations.max(dim=1)[0].squeeze(0)
                
                activations[layer_idx] = layer_activations.detach().cpu().numpy()
        
        return activations
    
    def get_top_features_per_layer(self, activations: Dict[int, np.ndarray], top_k: int = 10, threshold: float = 0.0, filter_always_on: bool = True) -> Dict:
        """Get top activated features for each layer based on actual activations for the prompt."""
        top_features = {}
        
        # Define always-on features that should be filtered out (based on base Llama analysis)
        always_on_features = {25, 389, 214, 290, 294, 134}
        
        for layer, layer_activations in activations.items():
            # Get feature information for lookup
            layer_features = {f["id"]: f for f in self.feature_data[layer]["features"]}
            
            # Only consider features that we have proper labels for
            available_feature_ids = list(layer_features.keys())
            
            # Filter out always-on features if requested
            if filter_always_on:
                available_feature_ids = [fid for fid in available_feature_ids if fid not in always_on_features]
                print(f"DEBUG: Filtered out always-on features. Remaining features: {len(available_feature_ids)}")
            
            if len(available_feature_ids) == 0:
                top_features[layer] = []
                continue
            
            # Get activations only for features we have labels for
            available_activations = layer_activations[available_feature_ids]
            
            # Filter features above threshold
            above_threshold = available_activations >= threshold
            filtered_activations = available_activations[above_threshold]
            filtered_feature_ids = np.array(available_feature_ids)[above_threshold]
            
            if len(filtered_activations) == 0:
                top_features[layer] = []
                continue
            
            # Get the actual top k most activated features for this specific prompt
            # Sort by activation value and take top k
            sorted_indices = np.argsort(filtered_activations)
            sorted_indices = np.flip(sorted_indices).copy()  # Reverse and copy to avoid negative strides
            top_k_actual = min(top_k, len(filtered_activations))
            top_indices = sorted_indices[:top_k_actual]
            top_values = filtered_activations[top_indices]
            top_feature_ids = filtered_feature_ids[top_indices]
            
            # Create feature list with actual top activations
            analyzed_features = []
            for feature_id, value in zip(top_feature_ids, top_values):
                # Use the predefined feature data
                feature_info = layer_features[feature_id].copy()
                feature_info["activation"] = float(value)
                analyzed_features.append(feature_info)
            
            top_features[layer] = analyzed_features
        
        return top_features
    
    def get_positive_negative_features(self, activations: Dict[int, np.ndarray], top_k: int = 10, threshold: float = 0.0, filter_always_on: bool = True) -> Dict:
        """Separate positive and negative activations for each layer."""
        separated_features = {}
        
        # Define always-on features that should be filtered out (based on base Llama analysis)
        always_on_features = {25, 389, 214, 290, 294, 134}
        
        for layer, layer_activations in activations.items():
            # Get feature information for lookup
            layer_features = {f["id"]: f for f in self.feature_data[layer]["features"]}
            
            # Only consider features that we have proper labels for
            available_feature_ids = list(layer_features.keys())
            
            # Filter out always-on features if requested
            if filter_always_on:
                available_feature_ids = [fid for fid in available_feature_ids if fid not in always_on_features]
            
            if len(available_feature_ids) == 0:
                separated_features[layer] = {"positive": [], "negative": []}
                continue
            
            # Get activations only for features we have labels for
            available_activations = layer_activations[available_feature_ids]
            
            # Separate positive and negative activations
            positive_mask = available_activations >= threshold
            negative_mask = available_activations < -threshold  # Only show strongly negative
            
            positive_activations = available_activations[positive_mask]
            positive_feature_ids = np.array(available_feature_ids)[positive_mask]
            
            negative_activations = available_activations[negative_mask]
            negative_feature_ids = np.array(available_feature_ids)[negative_mask]
            
            # Process positive activations
            positive_features = []
            if len(positive_activations) > 0:
                sorted_indices = np.argsort(positive_activations)
                sorted_indices = np.flip(sorted_indices).copy()  # Reverse and copy to avoid negative strides
                top_k_pos = min(top_k, len(positive_activations))
                top_pos_indices = sorted_indices[:top_k_pos]
                top_pos_values = positive_activations[top_pos_indices]
                top_pos_feature_ids = positive_feature_ids[top_pos_indices]
                
                for feature_id, value in zip(top_pos_feature_ids, top_pos_values):
                    feature_info = layer_features[feature_id].copy()
                    feature_info["activation"] = float(value)
                    positive_features.append(feature_info)
            
            # Process negative activations
            negative_features = []
            if len(negative_activations) > 0:
                sorted_indices = np.argsort(negative_activations)  # Most negative first
                sorted_indices = sorted_indices.copy()  # Ensure copy to avoid negative strides
                top_k_neg = min(top_k, len(negative_activations))
                top_neg_indices = sorted_indices[:top_k_neg]
                top_neg_values = negative_activations[top_neg_indices]
                top_neg_feature_ids = negative_feature_ids[top_neg_indices]
                
                for feature_id, value in zip(top_neg_feature_ids, top_neg_values):
                    feature_info = layer_features[feature_id].copy()
                    feature_info["activation"] = float(value)
                    negative_features.append(feature_info)
            
            separated_features[layer] = {
                "positive": positive_features,
                "negative": negative_features
            }
        
        return separated_features
    
    def create_activation_bars(self, top_features: Dict) -> go.Figure:
        """Create horizontal bar charts showing top activated features for each layer."""
        layers = list(top_features.keys())
        n_layers = len(layers)
        
        # Create subplots
        fig = make_subplots(
            rows=n_layers, cols=1,
            subplot_titles=[f"Layer {layer} - Top Activated Features" for layer in layers],
            vertical_spacing=0.05
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, layer in enumerate(layers):
            features = top_features[layer]
            
            # Sort by activation value (highest to lowest)
            features = sorted(features, key=lambda x: x["activation"], reverse=True)
            
            # Prepare data - reverse the order for display (highest at top)
            features = list(reversed(features))  # Reverse the features list itself
            
            labels = [f"F{feat['id']}: {feat['label'][:60]}{'...' if len(feat['label']) > 60 else ''}" for feat in features]
            activations = [feat["activation"] for feat in features]
            f1_scores = [feat["f1_score"] for feat in features]
            
            # Create hover text
            hover_text = []
            for feat in features:
                hover_text.append(
                    f"Feature {feat['id']}<br>"
                    f"Label: {feat['label']}<br>"
                    f"Activation: {feat['activation']:.4f}<br>"
                    f"F1 Score: {feat['f1_score']:.3f}<br>"
                    f"Activation Improvement: {feat['activation_improvement']:.4f}"
                )
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    y=labels,
                    x=activations,
                    orientation='h',
                    name=f"Layer {layer}",
                    marker_color=colors[i % len(colors)],
                    text=[f"{act:.3f}" for act in activations],
                    textposition='outside',
                    hovertemplate=hover_text,
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=200 * n_layers,
            title="Top Activated Features by Layer",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update x-axis for all subplots
        for i in range(n_layers):
            fig.update_xaxes(title_text="Activation Value", row=i+1, col=1)
            fig.update_yaxes(title_text="Features", row=i+1, col=1)
        
        return fig
    
    def create_positive_negative_bars(self, separated_features: Dict) -> go.Figure:
        """Create separate bar charts for positive and negative activations."""
        layers = list(separated_features.keys())
        n_layers = len(layers)
        
        # Create subplots with 2 columns (positive and negative)
        fig = make_subplots(
            rows=n_layers, cols=2,
            subplot_titles=[f"Layer {layer} - Positive Activations" for layer in layers] + 
                          [f"Layer {layer} - Negative Activations" for layer in layers],
            vertical_spacing=0.05,
            horizontal_spacing=0.1
        )
        
        colors_pos = px.colors.qualitative.Set2
        colors_neg = px.colors.qualitative.Pastel2
        
        for i, layer in enumerate(layers):
            layer_data = separated_features[layer]
            pos_features = layer_data["positive"]
            neg_features = layer_data["negative"]
            
            # Positive activations (left column) - sort highest to lowest
            if pos_features:
                pos_features = sorted(pos_features, key=lambda x: x["activation"], reverse=True)
                pos_features = list(reversed(pos_features))  # Reverse to show highest at top
                
                pos_labels = [f"F{feat['id']}: {feat['label'][:50]}{'...' if len(feat['label']) > 50 else ''}" for feat in pos_features]
                pos_activations = [feat["activation"] for feat in pos_features]
                
                pos_hover_text = []
                for feat in pos_features:
                    pos_hover_text.append(
                        f"Feature {feat['id']}<br>"
                        f"Label: {feat['label']}<br>"
                        f"Activation: {feat['activation']:.4f}<br>"
                        f"F1 Score: {feat['f1_score']:.3f}<br>"
                        f"Activation Improvement: {feat['activation_improvement']:.4f}"
                    )
                
                fig.add_trace(
                    go.Bar(
                        y=pos_labels,
                        x=pos_activations,
                        orientation='h',
                        name=f"Layer {layer} Positive",
                        marker_color=colors_pos[i % len(colors_pos)],
                        text=[f"{act:.3f}" for act in pos_activations],
                        textposition='outside',
                        hovertemplate=pos_hover_text,
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
            
            # Negative activations (right column) - sort most negative to least negative
            if neg_features:
                neg_features = sorted(neg_features, key=lambda x: x["activation"], reverse=True)  # Most negative first
                neg_features = list(reversed(neg_features))  # Reverse to show most negative at top
                
                neg_labels = [f"F{feat['id']}: {feat['label'][:50]}{'...' if len(feat['label']) > 50 else ''}" for feat in neg_features]
                neg_activations = [feat["activation"] for feat in neg_features]
                
                neg_hover_text = []
                for feat in neg_features:
                    neg_hover_text.append(
                        f"Feature {feat['id']}<br>"
                        f"Label: {feat['label']}<br>"
                        f"Activation: {feat['activation']:.4f}<br>"
                        f"F1 Score: {feat['f1_score']:.3f}<br>"
                        f"Activation Improvement: {feat['activation_improvement']:.4f}"
                    )
                
                fig.add_trace(
                    go.Bar(
                        y=neg_labels,
                        x=neg_activations,
                        orientation='h',
                        name=f"Layer {layer} Negative",
                        marker_color=colors_neg[i % len(colors_neg)],
                        text=[f"{act:.3f}" for act in neg_activations],
                        textposition='outside',
                        hovertemplate=neg_hover_text,
                        showlegend=False
                    ),
                    row=i+1, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=200 * n_layers,
            title="Positive and Negative Feature Activations by Layer",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update axes
        for i in range(n_layers):
            # Positive activations (left column)
            fig.update_xaxes(title_text="Activation Value", row=i+1, col=1)
            fig.update_yaxes(title_text="Features", row=i+1, col=1)
            
            # Negative activations (right column)
            fig.update_xaxes(title_text="Activation Value", row=i+1, col=2)
            fig.update_yaxes(title_text="Features", row=i+1, col=2)
        
        return fig
    
    def create_layer_comparison_chart(self, top_features: Dict) -> go.Figure:
        """Create a comparison chart showing top features across all layers."""
        # Collect all features with their layer info
        all_features = []
        for layer, features in top_features.items():
            for feat in features:
                all_features.append({
                    "layer": layer,
                    "feature_id": feat["id"],
                    "label": feat["label"],
                    "activation": feat["activation"],
                    "f1_score": feat["f1_score"],
                    "activation_improvement": feat["activation_improvement"]
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        layers = sorted(df["layer"].unique())
        colors = px.colors.qualitative.Set3
        
        for i, layer in enumerate(layers):
            layer_data = df[df["layer"] == layer].nlargest(5, "activation")
            
            fig.add_trace(go.Bar(
                name=f"Layer {layer}",
                x=[f"F{feat['feature_id']}: {feat['label'][:40]}{'...' if len(feat['label']) > 40 else ''}" for _, feat in layer_data.iterrows()],
                y=layer_data["activation"],
                marker_color=colors[i % len(colors)],
                text=[f"{act:.3f}" for act in layer_data["activation"]],
                textposition='outside',
                hovertemplate="<b>Layer %{customdata[0]}</b><br>" +
                             "Feature %{customdata[1]}<br>" +
                             "Label: %{customdata[2]}<br>" +
                             "Activation: %{y:.4f}<br>" +
                             "F1 Score: %{customdata[3]:.3f}<br>" +
                             "Activation Improvement: %{customdata[4]:.4f}<extra></extra>",
                customdata=[[layer, feat["feature_id"], feat["label"], feat["f1_score"], feat["activation_improvement"]] 
                           for _, feat in layer_data.iterrows()]
            ))
        
        fig.update_layout(
            title="Top 5 Features Comparison Across Layers",
            xaxis_title="Features",
            yaxis_title="Activation Value",
            barmode='group',
            height=600,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def get_token_level_activations(self, text: str, layers: List[int], aggregation_method: str = "max") -> Dict:
        """Get token-level feature activations for detailed analysis."""
        if self.model is None or self.tokenizer is None:
            return {}
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Pre-load SAE weights for all layers
        sae_weights_cache = {}
        for layer in layers:
            sae_weights = self.load_sae_weights(layer)
            if sae_weights is not None:
                sae_weights_cache[layer] = sae_weights
        
        # Forward pass with hidden states output
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states_list = outputs.hidden_states
        
        token_activations = {}
        
        # Process each requested layer
        for layer_idx in layers:
            if layer_idx in sae_weights_cache and layer_idx < len(hidden_states_list):
                # Get hidden states for this layer
                hidden_states = hidden_states_list[layer_idx + 1]
                
                # Use pre-loaded SAE weights
                sae_weights = sae_weights_cache[layer_idx]
                encoder = sae_weights["encoder"]
                encoder = encoder.to(hidden_states.device).to(hidden_states.dtype)
                
                # Compute activations: (batch_size, seq_len, n_features)
                feature_activations = torch.matmul(hidden_states, encoder.T)
                
                # Add encoder bias if available
                if "encoder_bias" in sae_weights:
                    encoder_bias = sae_weights["encoder_bias"].to(hidden_states.device).to(hidden_states.dtype)
                    feature_activations = feature_activations + encoder_bias
                
                # Get activations for each token
                token_activations[layer_idx] = {
                    'tokens': tokens,
                    'activations': feature_activations[0].detach().cpu().numpy()  # Shape: (seq_len, n_features)
                }
        
        return token_activations
    
    def create_token_level_heatmap(self, token_activations: Dict, layer: int, top_k: int = 10) -> go.Figure:
        """Create a heatmap showing token-level feature activations."""
        if layer not in token_activations:
            return go.Figure()
        
        data = token_activations[layer]
        tokens = data['tokens']
        activations = data['activations']  # Shape: (seq_len, n_features)
        
        # Get top features across all tokens for this layer
        max_activations = np.max(activations, axis=0)
        top_feature_indices = np.argsort(max_activations)[-top_k:][::-1]
        
        # Create heatmap data
        heatmap_data = activations[:, top_feature_indices]
        
        # Create feature labels
        feature_labels = [f"F{idx}" for idx in top_feature_indices]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.T,  # Transpose so features are on y-axis
            x=tokens,
            y=feature_labels,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate="Token: %{x}<br>Feature: %{y}<br>Activation: %{z:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Token-Level Feature Activations - Layer {layer}",
            xaxis_title="Tokens",
            yaxis_title="Top Features",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_activation_heatmap(self, top_features: Dict) -> go.Figure:
        """Create a heatmap showing activation patterns across layers and features."""
        # Prepare data for heatmap
        layers = sorted(top_features.keys())
        all_feature_ids = set()
        
        for features in top_features.values():
            for feat in features:
                all_feature_ids.add(feat["id"])
        
        all_feature_ids = sorted(list(all_feature_ids))
        
        # Create activation matrix
        activation_matrix = np.zeros((len(layers), len(all_feature_ids)))
        
        for i, layer in enumerate(layers):
            layer_features = {feat["id"]: feat for feat in top_features[layer]}
            for j, feature_id in enumerate(all_feature_ids):
                if feature_id in layer_features:
                    activation_matrix[i, j] = layer_features[feature_id]["activation"]
        
        # Create feature labels for heatmap
        feature_labels = []
        for fid in all_feature_ids:
            # Find the label for this feature
            label = f"F{fid}"
            for layer, data in top_features.items():
                for feat in data:
                    if feat["id"] == fid:
                        label = f"F{fid}: {feat['label'][:25]}{'...' if len(feat['label']) > 25 else ''}"
                        break
                if label != f"F{fid}":
                    break
            feature_labels.append(label)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=activation_matrix,
            x=feature_labels,
            y=[f"Layer {layer}" for layer in layers],
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate="Layer: %{y}<br>Feature: %{x}<br>Activation: %{z:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Feature Activation Heatmap Across Layers",
            xaxis_title="Features",
            yaxis_title="Layers",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig

def main():
    st.set_page_config(
        page_title="Feature Activation Tracker",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Feature Activation Tracker")
    st.markdown("**Analyze feature activations in Llama-2-7B using Sparse Autoencoders (SAEs)**")
    
    # Initialize the tracker
    if 'tracker' not in st.session_state:
        st.session_state.tracker = FeatureActivationTracker()
    
    tracker = st.session_state.tracker
    
    # Instructions
    st.markdown("### Enter text below and click 'Analyze Text' to see feature activations")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Text input
    # Initialize text input in session state
    if 'current_text' not in st.session_state:
        st.session_state.current_text = "The company reported strong quarterly earnings with revenue growth of 15% year-over-year, driven by increased demand in the technology sector."
    
    text_input = st.sidebar.text_area(
        "Enter text to analyze:",
        value=st.session_state.current_text,
        height=100,
        help="Enter text you want to analyze for feature activations",
        key="text_input_area"
    )
    
    # Update session state when text changes
    if text_input != st.session_state.current_text:
        st.session_state.current_text = text_input
    
    # Analyze button below text input
    analyze_button = st.sidebar.button("🔍 Analyze Text", type="primary", help="Click to analyze the current text and see feature activations")
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    top_k = st.sidebar.slider("Number of top features to show:", 5, 30, 10)
    threshold = st.sidebar.slider("Activation threshold:", -2.0, 5.0, 0.0, 0.1, 
                                 help="Features with activation below this threshold will be filtered out. Set to negative values (e.g., -1.0) to see negative activations.")
    
    # Aggregation method selection
    aggregation_method = st.sidebar.selectbox(
        "Activation Aggregation Method:",
        ["max", "mean", "sum", "combined", "normalized"],
        index=0,
        help="How to aggregate activations across sequence length:\n"
             "• max: Use maximum activation (best for finding most relevant features)\n"
             "• mean: Use average activation (original method)\n"
             "• sum: Use sum of activations\n"
             "• combined: 70% max + 30% mean\n"
             "• normalized: Max activation normalized (reduces always-on feature dominance)"
    )
    
    # Filter always-on features
    filter_always_on = st.sidebar.checkbox(
        "Filter Always-On Features",
        value=True,
        help="Filter out features that are always highly activated regardless of content. "
             "These features (25, 389, 214, 290, 294, 134) "
             "appear in 80%+ of texts and may not be content-specific."
    )
    
    layers_to_analyze = st.sidebar.multiselect(
        "Select layers to analyze:",
        [4, 10, 16, 22, 28],
        default=[4, 10, 16, 22, 28]
    )
    
    # Visualization mode
    viz_mode = st.sidebar.radio(
        "Visualization Mode:",
        ["Combined", "Positive/Negative Split", "Token-Level Analysis"],
        help="Combined: Show all features together. Split: Show positive and negative activations separately. Token-Level: Show which features activate on each token."
    )
    
    # Auto-load model on first run
    if tracker.model is None or tracker.tokenizer is None:
        with st.spinner("Loading model..."):
            tracker.model, tracker.tokenizer = tracker.load_model_and_tokenizer()
            st.success("Model loaded successfully!")
    
    # Test prompts
    st.sidebar.subheader("Test Prompts")
    test_prompts = [
        "The stock market is performing well today with strong gains in technology stocks",
        "The Federal Reserve announced a 0.25% interest rate increase to combat inflation",
        "Bitcoin reached a new all-time high of $100,000 amid institutional adoption",
        "The company's quarterly earnings exceeded expectations with 15% revenue growth",
        "Gold prices surged as investors sought safe-haven assets during market volatility",
        "The merger between the two banks created the largest financial institution in the region",
        "Hedge funds increased their short positions in the energy sector",
        "The IPO was oversubscribed by 300% indicating strong investor confidence",
        "Credit default swaps widened as concerns about corporate debt grew",
        "The central bank's quantitative easing program injected $2 trillion into the economy"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        if st.sidebar.button(f"Test {i}: {prompt[:30]}...", key=f"test_{i}"):
            st.session_state.test_text = prompt
            st.rerun()
    
    # Use test text if available
    if 'test_text' in st.session_state:
        st.session_state.current_text = st.session_state.test_text
        text_input = st.session_state.current_text
    
    # Force analysis when analyze button is clicked
    if tracker.model is not None and tracker.tokenizer is not None and analyze_button:
        # Get the current text from session state
        current_text = st.session_state.current_text.strip()
        
        if not current_text:
            st.error("Please enter some text to analyze!")
            st.stop()
        
        # Always recalculate - no caching
        with st.spinner("Analyzing feature activations..."):
            # Get activations
            activations = tracker.get_activations(current_text, layers_to_analyze, aggregation_method)
            
            if activations:
                # Get features based on visualization mode
                if viz_mode == "Combined":
                    top_features = tracker.get_top_features_per_layer(activations, top_k, threshold, filter_always_on)
                    st.session_state.top_features = top_features
                    st.session_state.separated_features = None
                    st.session_state.token_activations = None
                elif viz_mode == "Positive/Negative Split":
                    separated_features = tracker.get_positive_negative_features(activations, top_k, threshold, filter_always_on)
                    st.session_state.separated_features = separated_features
                    st.session_state.top_features = None
                    st.session_state.token_activations = None
                else:  # Token-Level Analysis
                    token_activations = tracker.get_token_level_activations(current_text, layers_to_analyze, aggregation_method)
                    st.session_state.token_activations = token_activations
                    st.session_state.top_features = None
                    st.session_state.separated_features = None
                
                # Store in session state
                st.session_state.activations = activations
                st.session_state.text_input = current_text
                st.session_state.threshold = threshold
                st.session_state.viz_mode = viz_mode
                
                # Analysis complete - no message needed
                
                # Show activation statistics (commented out)
                # st.subheader("Activation Statistics")
                # for layer in layers_to_analyze:
                #     if layer in activations:
                #         layer_activations = activations[layer]
                #         pos_count = np.sum(layer_activations >= threshold)
                #         neg_count = np.sum(layer_activations < -threshold)
                #         total_count = len(layer_activations)
                #         max_act = np.max(layer_activations)
                #         min_act = np.min(layer_activations)
                #         mean_act = np.mean(layer_activations)
                #         std_act = np.std(layer_activations)
                #         
                #         col1, col2, col3, col4, col5 = st.columns(5)
                #         with col1:
                #             st.metric(f"Layer {layer} - Positive", pos_count)
                #         with col2:
                #             st.metric(f"Layer {layer} - Negative", neg_count)
                #         with col3:
                #             st.metric(f"Layer {layer} - Max", f"{max_act:.3f}")
                #         with col4:
                #             st.metric(f"Layer {layer} - Mean", f"{mean_act:.3f}")
                #         with col5:
                #             st.metric(f"Layer {layer} - Std", f"{std_act:.3f}")
                #         
                #         # Show activation range for verification
                #         st.caption(f"Layer {layer} activation range: [{min_act:.3f}, {max_act:.3f}] (mean: {mean_act:.3f}, std: {std_act:.3f})")
                
                # Debug output moved to end
            else:
                st.error("No activations found. Please check your model and SAE weights.")
    
    # Note: Analysis happens automatically for every text change
    
    # Display results
    if 'top_features' in st.session_state and st.session_state.top_features:
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Features by Layer")
            activation_bars = tracker.create_activation_bars(st.session_state.top_features)
            st.plotly_chart(activation_bars, use_container_width=True)
        
        with col2:
            st.subheader("Layer Comparison")
            comparison_chart = tracker.create_layer_comparison_chart(st.session_state.top_features)
            st.plotly_chart(comparison_chart, use_container_width=True)
        
        # Heatmap
        st.subheader("Activation Heatmap")
        heatmap = tracker.create_activation_heatmap(st.session_state.top_features)
        st.plotly_chart(heatmap, use_container_width=True)
        
        # Detailed feature tables
        st.subheader("Detailed Feature Analysis")
        
        for layer, features in st.session_state.top_features.items():
            with st.expander(f"Layer {layer} - Top {len(features)} Features"):
                # Create DataFrame
                df = pd.DataFrame(features)
                # Ensure all required columns exist
                required_columns = ['id', 'label', 'activation', 'f1_score', 'activation_improvement']
                available_columns = [col for col in required_columns if col in df.columns]
                df = df[available_columns]
                
                # Rename columns for display
                column_mapping = {
                    'id': 'Feature ID',
                    'label': 'Label', 
                    'activation': 'Activation',
                    'f1_score': 'F1 Score',
                    'activation_improvement': 'Activation Improvement'
                }
                df = df.rename(columns=column_mapping)
                df = df.round(4)
                
                # Display table
                st.dataframe(df, use_container_width=True)
                
                # Feature descriptions
                st.markdown("**Feature Descriptions:**")
                for feat in features:
                    st.markdown(f"**Feature {feat['id']}**: {feat['label']}")
                    st.markdown(f"  - Activation: {feat['activation']:.4f} | F1 Score: {feat['f1_score']:.3f} | Activation Improvement: {feat['activation_improvement']:.4f}")
                    st.markdown("")
    
    elif 'separated_features' in st.session_state and st.session_state.separated_features:
        
        # Create positive/negative visualization
        st.subheader("Positive and Negative Activations")
        pos_neg_chart = tracker.create_positive_negative_bars(st.session_state.separated_features)
        st.plotly_chart(pos_neg_chart, use_container_width=True)
        
        # Feature details tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positive Activations")
            pos_features = []
            for layer, layer_data in st.session_state.separated_features.items():
                for feat in layer_data["positive"]:
                    pos_features.append({
                        "Layer": layer,
                        "Feature ID": feat["id"],
                        "Label": feat["label"],
                        "Activation": feat["activation"],
                        "F1 Score": feat["f1_score"],
                        "Activation Improvement": feat["activation_improvement"]
                    })
            
            if pos_features:
                pos_df = pd.DataFrame(pos_features)
                st.dataframe(pos_df, use_container_width=True)
            else:
                st.info("No positive activations above threshold.")
        
        with col2:
            st.subheader("Negative Activations")
            neg_features = []
            for layer, layer_data in st.session_state.separated_features.items():
                for feat in layer_data["negative"]:
                    neg_features.append({
                        "Layer": layer,
                        "Feature ID": feat["id"],
                        "Label": feat["label"],
                        "Activation": feat["activation"],
                        "F1 Score": feat["f1_score"],
                        "Activation Improvement": feat["activation_improvement"]
                    })
            
            if neg_features:
                neg_df = pd.DataFrame(neg_features)
                st.dataframe(neg_df, use_container_width=True)
            else:
                st.info("No negative activations below threshold.")
    
    elif 'token_activations' in st.session_state and st.session_state.token_activations:
        
        # Token-level analysis display
        st.subheader("Token-Level Feature Analysis")
        
        # Show tokens
        if layers_to_analyze:
            first_layer = layers_to_analyze[0]
            if first_layer in st.session_state.token_activations:
                tokens = st.session_state.token_activations[first_layer]['tokens']
                st.write(f"**Tokens:** {tokens}")
                st.write(f"**Number of tokens:** {len(tokens)}")
        
        # Layer selection for token analysis
        selected_layer = st.selectbox(
            "Select layer for token-level analysis:",
            layers_to_analyze,
            index=0
        )
        
        if selected_layer in st.session_state.token_activations:
            # Create token-level heatmap
            st.subheader(f"Token-Level Heatmap - Layer {selected_layer}")
            heatmap = tracker.create_token_level_heatmap(st.session_state.token_activations, selected_layer, top_k)
            st.plotly_chart(heatmap, use_container_width=True)
            
            # Detailed token analysis
            st.subheader("Detailed Token Analysis")
            
            data = st.session_state.token_activations[selected_layer]
            tokens = data['tokens']
            activations = data['activations']  # Shape: (seq_len, n_features)
            
            # Create a table showing top features for each token
            token_data = []
            for token_idx, token in enumerate(tokens):
                token_acts = activations[token_idx]
                top5_indices = np.argsort(token_acts)[-5:][::-1]
                top5_values = token_acts[top5_indices]
                
                token_data.append({
                    'Token': token,
                    'Position': token_idx,
                    'Top Feature 1': f"F{top5_indices[0]} ({top5_values[0]:.2f})",
                    'Top Feature 2': f"F{top5_indices[1]} ({top5_values[1]:.2f})",
                    'Top Feature 3': f"F{top5_indices[2]} ({top5_values[2]:.2f})",
                    'Top Feature 4': f"F{top5_indices[3]} ({top5_values[3]:.2f})",
                    'Top Feature 5': f"F{top5_indices[4]} ({top5_values[4]:.2f})"
                })
            
            df = pd.DataFrame(token_data)
            st.dataframe(df, use_container_width=True)
            
            # Feature frequency analysis
            st.subheader("Feature Frequency Analysis")
            
            # Count how often each feature appears in top 5 across all tokens
            feature_counts = {}
            for token_idx in range(len(tokens)):
                token_acts = activations[token_idx]
                top5_indices = np.argsort(token_acts)[-5:][::-1]
                for feat_idx in top5_indices:
                    feature_counts[feat_idx] = feature_counts.get(feat_idx, 0) + 1
            
            # Sort by frequency
            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
            
            freq_data = []
            for feat_idx, count in sorted_features[:20]:  # Top 20 most frequent
                percentage = (count / len(tokens)) * 100
                freq_data.append({
                    'Feature': f"F{feat_idx}",
                    'Frequency': count,
                    'Percentage': f"{percentage:.1f}%"
                })
            
            freq_df = pd.DataFrame(freq_data)
            st.dataframe(freq_df, use_container_width=True)
    
    else:
        st.info("👆 **How to use:** Enter text in the sidebar, adjust parameters, then click '🔍 Analyze Text' to see feature activations.")
        
        # Show current text input
        current_display_text = st.session_state.current_text.strip()
        if current_display_text:
            st.markdown("**Current text to analyze:**")
            st.code(current_display_text)
            st.markdown("**Click the '🔍 Analyze Text' button in the sidebar to start analysis.**")
        else:
            st.markdown("**No text entered yet.** Please enter some text in the sidebar.")
    
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Model:** `meta-llama/Llama-2-7b-hf` | "
        "**SAE Layers:** 4, 10, 16, 22, 28 | "
        "**Features:** Top activated features with base model analysis"
    )

if __name__ == "__main__":
    main()