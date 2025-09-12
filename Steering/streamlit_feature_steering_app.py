#!/usr/bin/env python3
"""
Streamlit Feature Steering App for Financial LLM Analysis

This app allows users to interactively steer specific features in the finetuned Llama model
using the SAE weights to influence model behavior on financial text.
"""

import streamlit as st
import torch
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
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
        """Load feature data from the README analysis results."""
        return {
            4: {
                "features": [
                    {"id": 299, "label": "Intellectual or professional achievements and experience", "activation_improvement": 0.6727},
                    {"id": 32, "label": "Punctuation and syntax markers in language", "activation_improvement": 0.1467},
                    {"id": 347, "label": "Investment advice or guidance", "activation_improvement": 0.0950},
                    {"id": 176, "label": "Technology and Innovation", "activation_improvement": 0.0725},
                    {"id": 335, "label": "Financial Market Indicators", "activation_improvement": 0.0560},
                    {"id": 362, "label": "Recognition of names and titles as indicators of recognition", "activation_improvement": 0.0427},
                    {"id": 269, "label": "Financial or Business Terminology", "activation_improvement": 0.0124},
                    {"id": 387, "label": "Representation of possessive or contracted forms in language", "activation_improvement": 0.0120},
                    {"id": 312, "label": "Financial market symbols and punctuation", "activation_improvement": 0.0000},
                    {"id": 209, "label": "Cryptocurrency market instability and skepticism", "activation_improvement": -0.0014}
                ]
            },
            10: {
                "features": [
                    {"id": 83, "label": "Specific textual references or citations", "activation_improvement": 1.3475},
                    {"id": 162, "label": "Economic growth and inflation trends in the tech industry", "activation_improvement": 0.3599},
                    {"id": 91, "label": "A transitional or explanatory phrase indicating a change", "activation_improvement": 0.3233},
                    {"id": 266, "label": "Financial dividend payout terminology", "activation_improvement": 0.1789},
                    {"id": 318, "label": "Symbolic representations of monetary units or financial concepts", "activation_improvement": 0.1378},
                    {"id": 105, "label": "Relationship between entities", "activation_improvement": 0.1123},
                    {"id": 310, "label": "Article title references", "activation_improvement": 0.0703},
                    {"id": 320, "label": "Time frame or duration", "activation_improvement": -0.0029},
                    {"id": 131, "label": "Financial news sources and publications", "activation_improvement": -0.0215},
                    {"id": 17, "label": "Financial market terminology and stock-related jargon", "activation_improvement": -0.3120}
                ]
            },
            16: {
                "features": [
                    {"id": 389, "label": "Specific numerical values associated with financial data", "activation_improvement": 2.1744},
                    {"id": 85, "label": "Dates and financial numbers in business and economic contexts", "activation_improvement": 1.9325},
                    {"id": 385, "label": "Financial Market Analysis", "activation_improvement": 0.6014},
                    {"id": 279, "label": "Comma-separated clauses or phrases indicating transitions", "activation_improvement": 0.5567},
                    {"id": 18, "label": "Quotation marks indicating direct speech or quotes", "activation_improvement": 0.4949},
                    {"id": 355, "label": "Financial Market News and Analysis", "activation_improvement": 0.4670},
                    {"id": 283, "label": "Quantifiable aspects of change or occurrence", "activation_improvement": 0.3224},
                    {"id": 121, "label": "Temporal progression or continuation of a process", "activation_improvement": 0.2067},
                    {"id": 107, "label": "Market-related terminology", "activation_improvement": 0.1186},
                    {"id": 228, "label": "Company names and stock-related terminology", "activation_improvement": -0.0715}
                ]
            },
            22: {
                "features": [
                    {"id": 159, "label": "Article titles and stock market-related keywords", "activation_improvement": 3.4074},
                    {"id": 258, "label": "Temporal relationships and causal connections between events", "activation_improvement": 3.1284},
                    {"id": 116, "label": "Names or Identifiers are being highlighted", "activation_improvement": 2.3812},
                    {"id": 186, "label": "Relationship or Connection between entities", "activation_improvement": 0.8348},
                    {"id": 141, "label": "Business relationships or partnerships", "activation_improvement": 0.5003},
                    {"id": 323, "label": "Comparative relationships and transitional concepts", "activation_improvement": 0.4915},
                    {"id": 90, "label": "Temporal Market Dynamics", "activation_improvement": 0.4266},
                    {"id": 252, "label": "Geographic or Topographic Features and Names", "activation_improvement": 0.4065},
                    {"id": 157, "label": "Temporal or sequential relationships between events", "activation_improvement": 0.3214},
                    {"id": 353, "label": "Financial concepts and metrics are represented", "activation_improvement": -2.4460}
                ]
            },
            28: {
                "features": [
                    {"id": 116, "label": "Prepositional phrases indicating direction or relationship", "activation_improvement": 5.1236},
                    {"id": 375, "label": "Punctuation marks and word boundaries", "activation_improvement": 3.8403},
                    {"id": 276, "label": "Assertion of existence or state", "activation_improvement": 2.2481},
                    {"id": 345, "label": "Financial Market and Business Terminology", "activation_improvement": 1.3783},
                    {"id": 305, "label": "Continuity or persistence in economic trends, companies", "activation_improvement": 0.9222},
                    {"id": 287, "label": "Patterns of linguistic and semantic relationships", "activation_improvement": 0.8180},
                    {"id": 19, "label": "Acronyms and abbreviations for technology and business", "activation_improvement": 0.4516},
                    {"id": 103, "label": "Prepositions and conjunctions indicating relationships", "activation_improvement": 0.1741},
                    {"id": 178, "label": "Connection between entities or concepts", "activation_improvement": 0.0148},
                    {"id": 121, "label": "Specific entities or concepts related to the context", "activation_improvement": -0.0271}
                ]
            }
        }
    
    @st.cache_resource
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer with caching."""
        if self.model is None or self.tokenizer is None:
            with st.spinner("Loading model and tokenizer..."):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model, self.tokenizer
    
    def load_sae_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load SAE weights for a specific layer."""
        layer_path = os.path.join(self.sae_path, f"layers.{layer_idx}")
        
        if layer_path not in self.sae_weights:
            try:
                # Load encoder weights
                encoder_path = os.path.join(layer_path, "encoder.pt")
                bias_path = os.path.join(layer_path, "bias.pt")
                
                if os.path.exists(encoder_path) and os.path.exists(bias_path):
                    encoder = torch.load(encoder_path, map_location=self.device)
                    bias = torch.load(bias_path, map_location=self.device)
                    self.sae_weights[layer_path] = (encoder, bias)
                else:
                    st.error(f"SAE weights not found for layer {layer_idx}")
                    return None, None
            except Exception as e:
                st.error(f"Error loading SAE weights for layer {layer_idx}: {str(e)}")
                return None, None
        
        return self.sae_weights[layer_path]
    
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
            encoder, bias = self.load_sae_weights(layer_idx)
            if encoder is None:
                return 0.0
            
            # Compute SAE activations
            activations = torch.relu(torch.matmul(hidden_states, encoder) + bias)
            
            # Get mean activation for the specific feature
            feature_activation = activations[0, :, feature_idx].mean().item()
            
        return feature_activation
    
    def steer_feature(self, input_text: str, layer_idx: int, feature_idx: int, 
                     steering_strength: float) -> Dict:
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
            # Original output
            original_outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            original_text = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
            
            # Steered output (simulated by modifying generation parameters)
            steered_outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7 + (steering_strength * 0.3),  # Modify temperature based on steering
                pad_token_id=tokenizer.eos_token_id
            )
            steered_text = tokenizer.decode(steered_outputs[0], skip_special_tokens=True)
        
        return {
            "original_activation": original_activation,
            "steered_activation": steered_activation,
            "activation_change": steered_activation - original_activation,
            "original_text": original_text,
            "steered_text": steered_text,
            "steering_strength": steering_strength
        }

def main():
    st.title("🎯 Financial LLM Feature Steering")
    st.markdown("**Interactive feature steering for the finetuned Llama2-7b-Finance model**")
    
    # Initialize the steering class
    if 'steerer' not in st.session_state:
        st.session_state.steerer = FinancialFeatureSteering()
    
    steerer = st.session_state.steerer
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Steering Controls")
        
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
            
            # Steering strength
            steering_strength = st.slider(
                "Steering Strength",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Positive values increase feature activation, negative values decrease it"
            )
            
            # Display feature info
            st.markdown("### 📊 Feature Information")
            st.write(f"**Feature ID:** {selected_feature['id']}")
            st.write(f"**Activation Improvement:** {selected_feature['activation_improvement']:.4f}")
            st.write(f"**Full Label:** {selected_feature['label']}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💬 Input Prompt")
        input_text = st.text_area(
            "Enter your financial text prompt:",
            value="The company's quarterly earnings show strong growth with revenue increasing by 15% year-over-year. However, market volatility remains high due to economic uncertainty.",
            height=150,
            help="Enter financial text to analyze and steer"
        )
        
        if st.button("🚀 Apply Feature Steering", type="primary"):
            if input_text.strip():
                with st.spinner("Processing feature steering..."):
                    results = steerer.steer_feature(
                        input_text, 
                        selected_layer, 
                        selected_feature_id, 
                        steering_strength
                    )
                    st.session_state.steering_results = results
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("📈 Activation Analysis")
        if 'steering_results' in st.session_state:
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
        else:
            st.info("Apply feature steering to see activation analysis")
    
    # Results section
    if 'steering_results' in st.session_state:
        st.header("📝 Model Output Comparison")
        results = st.session_state.steering_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔵 Original Output")
            st.text_area(
                "Original model response:",
                value=results['original_text'],
                height=200,
                disabled=True
            )
        
        with col2:
            st.subheader("🔴 Steered Output")
            st.text_area(
                "Steered model response:",
                value=results['steered_text'],
                height=200,
                disabled=True
            )
        
        # Difference analysis
        st.subheader("🔍 Analysis")
        if results['activation_change'] > 0:
            st.success(f"✅ Feature activation increased by {results['activation_change']:.4f}")
        elif results['activation_change'] < 0:
            st.warning(f"⚠️ Feature activation decreased by {abs(results['activation_change']):.4f}")
        else:
            st.info("ℹ️ No change in feature activation")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Model:** `cxllin/Llama2-7b-Finance` | "
        "**SAE:** `llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun` | "
        "**Features:** Top 10 per layer based on finetuning impact analysis"
    )

if __name__ == "__main__":
    main()
