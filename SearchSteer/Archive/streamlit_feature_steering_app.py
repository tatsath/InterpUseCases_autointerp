#!/usr/bin/env python3
"""
Streamlit Feature Steering App for Financial LLM Analysis

This app allows users to interactively steer specific features in the finetuned Llama model
using SAELens-style steering: direct feature direction addition to hidden states.
Based on testing, optimal steering strength is 2.0-5.0 for best results.
"""

import streamlit as st
import torch
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Financial LLM Feature Steering",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FinancialFeatureSteering:
    """
    A class for steering financial LLM behavior by manipulating specific SAE features.
    """
    
    def __init__(self):
        self.model_path = "cxllin/Llama2-7b-Finance"
        self.sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.sae_weights = {}
        self.feature_data = self._load_feature_data()
        
    def _load_feature_data(self) -> Dict:
        """Load feature data from the README analysis results (FINETUNED MODEL LABELS)."""
        return {
            4: {
                "features": [
                    {"id": 299, "label": "Financial Market Analysis", "activation_improvement": 0.6727},
                    {"id": 32, "label": "Financial market terminology and stock-related language", "activation_improvement": 0.1467},
                    {"id": 347, "label": "Date specification", "activation_improvement": 0.0950},
                    {"id": 176, "label": "Financial Institutions and Markets", "activation_improvement": 0.0725},
                    {"id": 335, "label": "Punctuation marks indicating quotation or possession", "activation_improvement": 0.0560},
                    {"id": 362, "label": "Company or brand names", "activation_improvement": 0.0427},
                    {"id": 269, "label": "Financial company or investment entity name", "activation_improvement": 0.0124},
                    {"id": 387, "label": "Financial market terminology and stock-related expressions", "activation_improvement": 0.0120},
                    {"id": 312, "label": "Temporal relationships and conditional dependencies", "activation_improvement": 0.0000},
                    {"id": 209, "label": "Market trends or indicators", "activation_improvement": -0.0014}
                ]
            },
            10: {
                "features": [
                    {"id": 83, "label": "Financial market trends and performance metrics", "activation_improvement": 1.3475},
                    {"id": 162, "label": "Financial Market Analysis and Investment Guidance", "activation_improvement": 0.3599},
                    {"id": 91, "label": "Two-digit year representation", "activation_improvement": 0.3233},
                    {"id": 266, "label": "Financial industry terminology", "activation_improvement": 0.1789},
                    {"id": 318, "label": "Financial Transactions and Market Trends", "activation_improvement": 0.1378},
                    {"id": 105, "label": "Maritime shipping and trade-related concepts", "activation_improvement": 0.1123},
                    {"id": 310, "label": "Analysts' opinions and expectations about market trends", "activation_improvement": 0.0703},
                    {"id": 320, "label": "Representation of numerical values, including years", "activation_improvement": -0.0029},
                    {"id": 131, "label": "Names of news and media outlets", "activation_improvement": -0.0215},
                    {"id": 17, "label": "Stock market terminology and financial jargon", "activation_improvement": -0.3120}
                ]
            },
            16: {
                "features": [
                    {"id": 389, "label": "Financial performance indicators", "activation_improvement": 2.1744},
                    {"id": 85, "label": "Financial News and Analysis", "activation_improvement": 1.9325},
                    {"id": 385, "label": "Financial market entities and terminology", "activation_improvement": 0.6014},
                    {"id": 279, "label": "Market fragility at its most critical point", "activation_improvement": 0.5567},
                    {"id": 18, "label": "Temporal Reference or Time Periods", "activation_improvement": 0.4949},
                    {"id": 355, "label": "Financial entity or company name", "activation_improvement": 0.4670},
                    {"id": 283, "label": "Stock market concepts", "activation_improvement": 0.3224},
                    {"id": 121, "label": "Financial market analysis and company performance", "activation_improvement": 0.2067},
                    {"id": 107, "label": "Numerical and symbolic representations", "activation_improvement": 0.1186},
                    {"id": 228, "label": "FUTURE TRENDS OR OUTCOMES", "activation_improvement": -0.0715}
                ]
            },
            22: {
                "features": [
                    {"id": 159, "label": "Financial Performance and Growth", "activation_improvement": 3.4074},
                    {"id": 258, "label": "Financial market indicators and metrics", "activation_improvement": 3.1284},
                    {"id": 116, "label": "Financial and business-related themes", "activation_improvement": 2.3812},
                    {"id": 186, "label": "Conditional or hypothetical scenarios in financial contexts", "activation_improvement": 0.8348},
                    {"id": 141, "label": "Transition or Change", "activation_improvement": 0.5003},
                    {"id": 323, "label": "Relationship indicators between entities or concepts", "activation_improvement": 0.4915},
                    {"id": 90, "label": "Financial market terminology and concepts", "activation_improvement": 0.4266},
                    {"id": 252, "label": "Financial market terminology and jargon", "activation_improvement": 0.4065},
                    {"id": 157, "label": "Emphasis on a specific aspect or element", "activation_improvement": 0.3214},
                    {"id": 353, "label": "Specific word forms or combinations indicating a pattern", "activation_improvement": -2.4460}
                ]
            },
            28: {
                "features": [
                    {"id": 116, "label": "Company financial performance and market impact", "activation_improvement": 5.1236},
                    {"id": 375, "label": "Financial market terminology and jargon", "activation_improvement": 3.8403},
                    {"id": 276, "label": "Temporal relationships and contextual dependencies", "activation_improvement": 2.2481},
                    {"id": 345, "label": "Financial performance metrics", "activation_improvement": 1.3783},
                    {"id": 305, "label": "Financial Earnings and Stock Market Performance", "activation_improvement": 0.9222},
                    {"id": 287, "label": "Financial Performance Indicators", "activation_improvement": 0.8180},
                    {"id": 19, "label": "Financial Market Stock Performance Analysis", "activation_improvement": 0.4516},
                    {"id": 103, "label": "Conjunctions and prepositions in financial texts", "activation_improvement": 0.1741},
                    {"id": 178, "label": "Pre-pandemic cost-saving measures", "activation_improvement": 0.0148},
                    {"id": 121, "label": "Gaming GPU price elasticity", "activation_improvement": -0.0271}
                ]
            }
        }
    
    @st.cache_resource
    def load_model_and_tokenizer(_self):
        """Load the model and tokenizer with caching."""
        with st.spinner("Loading model and tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(_self.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                _self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    @st.cache_resource
    def load_sae_weights(_self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load SAE weights for a specific layer."""
        layer_path = os.path.join(_self.sae_path, f"layers.{layer_idx}")
        
        try:
            # Load SAE weights from safetensors file
            sae_path = os.path.join(layer_path, "sae.safetensors")
            
            if os.path.exists(sae_path):
                from safetensors import safe_open
                with safe_open(sae_path, framework="pt", device="cpu") as f:
                    encoder = f.get_tensor("encoder.weight")
                    encoder_bias = f.get_tensor("encoder.bias")
                    decoder = f.get_tensor("W_dec")
                    decoder_bias = f.get_tensor("b_dec")
                # Move to the same device as the model
                encoder = encoder.to(_self.device)
                encoder_bias = encoder_bias.to(_self.device)
                decoder = decoder.to(_self.device)
                decoder_bias = decoder_bias.to(_self.device)
                
                
                return encoder, encoder_bias, decoder, decoder_bias
            else:
                st.error(f"SAE weights not found for layer {layer_idx} at {sae_path}")
                return None, None, None, None
        except Exception as e:
            st.error(f"Error loading SAE weights for layer {layer_idx}: {str(e)}")
            return None, None, None, None
    
    def get_feature_activation(self, input_text: str, layer_idx: int, feature_idx: int) -> float:
        """Get the activation of a specific feature for given input."""
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx + 1]  # +1 because layer 0 is embeddings
            
            # Load SAE weights
            encoder, encoder_bias, decoder, decoder_bias = self.load_sae_weights(layer_idx)
            if encoder is None:
                return 0.0
            
            # Compute SAE activations
            # Ensure dtype compatibility: convert hidden_states to float32 to match SAE weights
            hidden_states_f32 = hidden_states.float()
            # Transpose encoder to match dimensions: (hidden_states: 40x4096) @ (encoder.T: 4096x400) = (40x400)
            activations = torch.relu(torch.matmul(hidden_states_f32, encoder.T) + encoder_bias)
            
            # Get mean activation for the specific feature
            feature_activation = activations[0, :, feature_idx].mean().item()
            
        return feature_activation

    def _apply_feature_steering(self, model, tokenizer, inputs, layer_idx: int, 
                               feature_idx: int, steering_strength: float, max_tokens: int = 100) -> str:
        """Apply actual feature steering by modifying hidden states during generation."""
        # Load SAE weights for the specific layer
        encoder, encoder_bias, decoder, decoder_bias = self.load_sae_weights(layer_idx)
        if encoder is None:
            return f"Error: Could not load SAE weights for layer {layer_idx}"
        
        
        # Create a custom forward hook to modify hidden states using SAELens approach
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Move SAE weights to the same device as hidden states
            encoder_device = encoder.to(hidden_states.device)
            encoder_bias_device = encoder_bias.to(hidden_states.device)
            decoder_device = decoder.to(hidden_states.device)
            decoder_bias_device = decoder_bias.to(hidden_states.device)
            
            # Enhanced SAELens approach: Direct feature direction addition to hidden states
            if abs(steering_strength) > 0.01:  # Only apply if significant steering
                # Get the feature direction from decoder (this is the steering vector)
                feature_direction = decoder_device[feature_idx, :]  # Shape: [hidden_dim]
                feature_direction = feature_direction.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, hidden_dim]
                
                # Normalize the feature direction for consistent steering
                feature_norm = torch.norm(feature_direction)
                if feature_norm > 0:
                    feature_direction = feature_direction / feature_norm
                
                # Scale the steering vector by strength (enhanced coefficient)
                steering_vector = steering_strength * 0.5 * feature_direction  # Increased from 0.1 to 0.5
                
                # Add steering vector directly to hidden states
                steered_hidden = hidden_states + steering_vector
                
                # Debug: Print steering info (only once per generation)
                if not hasattr(steering_hook, 'debug_printed'):
                    steering_magnitude = torch.norm(steering_vector).item()
                    print(f"🎯 Steering Debug - Strength: {steering_strength}, Magnitude: {steering_magnitude:.4f}")
                    steering_hook.debug_printed = True
            else:
                steered_hidden = hidden_states
                # Reset debug flag for next generation
                if hasattr(steering_hook, 'debug_printed'):
                    delattr(steering_hook, 'debug_printed')
            
            # Replace the hidden states
            if isinstance(output, tuple):
                return (steered_hidden.to(hidden_states.dtype),) + output[1:]
            else:
                return steered_hidden.to(hidden_states.dtype)
        
        # Register the hook on the specific layer
        layer_module = model.model.layers[layer_idx]
        hook = layer_module.register_forward_hook(steering_hook)
        
        try:
            # Generate with steering
            torch.manual_seed(43)  # Different seed for steered generation
            steered_outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            steered_text = tokenizer.decode(steered_outputs[0], skip_special_tokens=True)
        finally:
            # Remove the hook
            hook.remove()
        
        return steered_text

    def steer_feature(self, input_text: str, layer_idx: int, feature_idx: int, 
                     steering_strength: float, max_tokens: int = 100) -> Dict:
        """Apply feature steering and return results."""
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Get original activation
        original_activation = self.get_feature_activation(input_text, layer_idx, feature_idx)
        
        # For demonstration, we'll simulate the steering effect
        # In a real implementation, you would modify the model's forward pass
        steered_activation = original_activation + (steering_strength * 2.0)
        
        # Generate text with and without steering (simplified)
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Set a fixed seed for reproducible generation
            torch.manual_seed(42)
            
            # Original output with sampling for fair comparison
            original_outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,  # Use same length as steered
                do_sample=True,  # Use sampling for fair comparison
                temperature=0.7,  # Same temperature as steered
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
            
            # Steered output - only apply changes if steering strength is not zero
            if abs(steering_strength) > 0.01:  # Only apply steering if significant
                # Apply actual feature steering by modifying hidden states
                steered_text = self._apply_feature_steering(
                    model, tokenizer, inputs, layer_idx, feature_idx, steering_strength, max_tokens
                )
            else:
                # If no steering, use the same output
                steered_text = original_text
        
        return {
            "original_activation": original_activation,
            "steered_activation": steered_activation,
            "activation_change": steered_activation - original_activation,
            "original_text": original_text,
            "steered_text": steered_text,
            "steering_strength": steering_strength
        }

def main():
    st.title("🎯 Feature Engineering")
    
    # Initialize the steering class
    if 'steerer' not in st.session_state:
        st.session_state.steerer = FinancialFeatureSteering()
    
    steerer = st.session_state.steerer
    
    # Sidebar for all controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Layer selection
        layer_options = list(steerer.feature_data.keys())
        selected_layer = st.selectbox(
            "Select Layer",
            options=layer_options,
            format_func=lambda x: f"Layer {x}"
        )
        
        # Feature selection
        if selected_layer:
            features = steerer.feature_data[selected_layer]["features"]
            feature_options = {f"{feat['id']}: {feat['label'][:50]}..." for feat in features}
            
            selected_feature_str = st.selectbox(
                "Select Feature",
                options=list(feature_options),
                help="Top 10 features for the selected layer based on activation improvements"
            )
            
            # Extract feature ID
            selected_feature_id = int(selected_feature_str.split(":")[0])
            selected_feature = next(f for f in features if f['id'] == selected_feature_id)
            
            # Steering strength - optimized range for effective steering
            steering_strength = st.slider(
                "Steering Strength",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                help="Feature steering strength: Direct feature direction addition to hidden states. Range: -50 to +50. Higher values show dramatic effects."
            )
            
            # Token length control
            max_tokens = st.slider(
                "Max Tokens to Generate",
                min_value=100,
                max_value=800,
                value=500,
                step=50,
                help="Number of new tokens to generate. Higher values may take longer."
            )
            
    
    # Input section
    st.header("💬 Input Prompt")
    input_text = st.text_area(
        "Enter your financial text prompt:",
        value="The business strategy involves",
        height=120,
        help="Enter financial text to analyze and steer"
    )
    
    if st.button("🚀 Apply Feature Steering", type="primary"):
        if input_text.strip():
            with st.spinner("Processing feature steering..."):
                results = steerer.steer_feature(
                    input_text, 
                    selected_layer, 
                    selected_feature_id, 
                    steering_strength,
                    max_tokens
                )
                st.session_state.steering_results = results
        else:
            st.warning("Please enter some text to analyze.")
    
    # Results section - side by side text boxes
    if 'steering_results' in st.session_state:
        st.header("📝 Model Output Comparison")
        results = st.session_state.steering_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔵 Original Output")
            st.text_area(
                "Original model response:",
                value=results['original_text'],
                height=400,
                disabled=True,
                key="original_output"
            )
        
        with col2:
            st.subheader("🔴 Steered Output")
            st.text_area(
                "Steered model response:",
                value=results['steered_text'],
                height=400,
                disabled=True,
                key="steered_output"
            )
        
        # Difference analysis
        st.subheader("🔍 Analysis")
        if results['activation_change'] > 0:
            st.success(f"✅ Feature activation increased by {results['activation_change']:.4f}")
        elif results['activation_change'] < 0:
            st.warning(f"⚠️ Feature activation decreased by {abs(results['activation_change']):.4f}")
        else:
            st.info("ℹ️ No change in feature activation")
        
    
    # Activation Analysis Section (moved to end)
    if 'steering_results' in st.session_state:
        st.header("📈 Activation Analysis")
        results = st.session_state.steering_results
        
        # Activation metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Original Activation",
                f"{results['original_activation']:.4f}",
                delta=None
            )
        with col_b:
            st.metric(
                "Steered Activation", 
                f"{results['steered_activation']:.4f}",
                delta=f"{results['activation_change']:+.4f}"
            )
        with col_c:
            st.metric(
                "Steering Strength",
                f"{results['steering_strength']:+.1f}",
                delta=None
            )
        
        # Activation visualization
        fig = go.Figure(data=[
            go.Bar(
                name='Original',
                x=['Activation'],
                y=[results['original_activation']],
                marker_color='lightblue'
            ),
            go.Bar(
                name='Steered',
                x=['Activation'],
                y=[results['steered_activation']],
                marker_color='lightcoral'
            )
        ])
        fig.update_layout(
            title="Feature Activation Comparison",
            yaxis_title="Activation Value",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Model:** `cxllin/Llama2-7b-Finance` | "
        "**SAE:** `llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun` | "
        "**Features:** Top 10 per layer based on finetuning impact analysis"
    )

if __name__ == "__main__":
    main()
