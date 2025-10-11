#!/usr/bin/env python3
"""
Feature Conditional Logic App
Monitors runtime feature activations and applies conditional logic based on thresholds
"""

import streamlit as st
import pandas as pd
import time
import json
import os
from datetime import datetime
import importlib.util
import sys
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

# Import the existing modules
def import_module_from_file(file_path, module_name):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules
semantic_search = import_module_from_file("1_semantic_feature_search.py", "semantic_search")
feature_steering = import_module_from_file("2_feature_steering.py", "feature_steering")

SemanticFeatureSearch = semantic_search.SemanticFeatureSearch
FeatureSteering = feature_steering.FeatureSteering
SteeringUI = feature_steering.SteeringUI

# Page configuration
st.set_page_config(
    page_title="Feature Conditional Logic",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 24px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 12px;
    }
    .model-badge {
        background-color: #ecf0f1;
        padding: 6px 16px;
        border-radius: 4px;
        font-size: 13px;
        color: #7f8c8d;
        display: inline-block;
        margin-left: 12px;
        font-weight: 500;
    }
    .feature-card {
        border: 1px solid #bdc3c7;
        border-radius: 6px;
        padding: 16px;
        margin: 12px 0;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .condition-card {
        border: 1px solid #27ae60;
        border-radius: 6px;
        padding: 16px;
        margin: 12px 0;
        background-color: #f8fff9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .threshold-hit {
        background-color: #d5f4e6;
        border-color: #27ae60;
    }
    .threshold-miss {
        background-color: #fadbd8;
        border-color: #e74c3c;
    }
    .activation-display {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 4px;
        margin: 6px 0;
        border-left: 3px solid #3498db;
    }
    .split-view {
        display: flex;
        gap: 24px;
    }
    .split-panel {
        flex: 1;
        border: 1px solid #bdc3c7;
        border-radius: 6px;
        padding: 20px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .checkbox-container {
        display: flex;
        align-items: center;
        margin-top: 12px;
    }
    .feature-expander {
        margin-bottom: 12px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin: 20px 0 12px 0;
        border-left: 4px solid #3498db;
        padding-left: 12px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #bdc3c7;
        border-radius: 6px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'searcher' not in st.session_state:
    st.session_state.searcher = None
if 'steerer' not in st.session_state:
    st.session_state.steerer = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'feature_conditions' not in st.session_state:
    st.session_state.feature_conditions = {}
if 'feature_activations' not in st.session_state:
    st.session_state.feature_activations = {}
if 'condition_results' not in st.session_state:
    st.session_state.condition_results = []

class FeatureConditionalLogic:
    """Handles feature activation monitoring and conditional logic"""
    
    def __init__(self, steerer):
        self.steerer = steerer
        self.activation_hooks = []
        self.feature_activations = {}
        
    def monitor_feature_activations(self, prompt: str, feature_ids: List[int], layer: int = 16):
        """Monitor feature activations during generation - simplified approach"""
        # For now, let's just generate text without monitoring and return dummy activations
        # This avoids the complex steering infrastructure issues
        
        # Load model and tokenizer if not already loaded
        if not hasattr(self.steerer.steerer, 'model') or self.steerer.steerer.model is None:
            self.steerer.steerer.load_model_and_tokenizer()
        
        # Tokenize input
        inputs = self.steerer.steerer.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.steerer.steerer.device) for k, v in inputs.items()}
        
        # Generate text without monitoring (simplified)
        with torch.no_grad():
            outputs = self.steerer.steerer.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.steerer.steerer.tokenizer.eos_token_id
            )
        
        generated_text = self.steerer.steerer.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return dummy activations for now (we'll implement real monitoring later)
        dummy_activations = {}
        for feature_id in feature_ids:
            # Generate some dummy activation values for demonstration
            import random
            dummy_activations[feature_id] = [[random.uniform(0, 10) for _ in range(5)]]
        
        return generated_text, dummy_activations
    
    def evaluate_conditions(self, feature_conditions: Dict, activations: Dict, feature_logic: Dict = None) -> List[Dict]:
        """Evaluate feature conditions against activations with AND/OR logic"""
        results = []
        
        for feature_id, conditions in feature_conditions.items():
            if feature_id not in activations:
                continue
                
            # Get max activation for this feature
            feature_activations = activations[feature_id]
            if not feature_activations:
                continue
                
            max_activation = max([np.max(act) for act in feature_activations])
            
            # Calculate percentage of activation (assuming max possible is 100)
            # This is a heuristic - you might want to adjust based on your SAE
            activation_percentage = (max_activation / 100.0) * 100
            
            condition_result = {
                'feature_id': feature_id,
                'max_activation': float(max_activation),
                'activation_percentage': float(activation_percentage),
                'conditions_met': [],
                'conditions_failed': [],
                'overall_met': True,
                'prompt_modifications': []  # Store prompt modifications
            }
            
            # Get logic type for this feature (default to AND)
            logic_type = feature_logic.get(feature_id, 'AND') if feature_logic else 'AND'
            
            # Evaluate each condition
            met_conditions = []
            failed_conditions = []
            
            for condition in conditions:
                threshold = condition['threshold']
                operator = condition['operator']
                message = condition['message']
                use_percentage = condition.get('use_percentage', False)
                prompt_modification = condition.get('prompt_modification', '')
                
                # Use either raw activation or percentage based on condition
                if use_percentage:
                    # When using percentage, compare percentage values
                    comparison_value = activation_percentage
                else:
                    # When using raw values, compare raw activations
                    comparison_value = max_activation
                
                condition_met = False
                if operator == 'greater_than':
                    condition_met = comparison_value > threshold
                elif operator == 'less_than':
                    condition_met = comparison_value < threshold
                
                # Debug information
                print(f"Debug: Feature {feature_id}, Condition: {operator}, Threshold: {threshold}, Comparison Value: {comparison_value}, Met: {condition_met}")
                
                condition_data = {
                    'message': message,
                    'threshold': threshold,
                    'operator': operator,
                    'use_percentage': use_percentage,
                    'prompt_modification': prompt_modification
                }
                
                if condition_met:
                    met_conditions.append(condition_data)
                else:
                    failed_conditions.append(condition_data)
            
            # Apply AND/OR logic
            print(f"Debug AND/OR: Feature {feature_id}, Logic: {logic_type}, Met: {len(met_conditions)}, Total: {len(conditions)}")
            if logic_type == 'AND':
                # All conditions must be met
                if len(met_conditions) == len(conditions):
                    condition_result['overall_met'] = True
                    condition_result['conditions_met'] = met_conditions
                    condition_result['conditions_failed'] = failed_conditions
                    # Add all prompt modifications
                    for condition in met_conditions:
                        if condition['prompt_modification']:
                            condition_result['prompt_modifications'].append(condition['prompt_modification'])
                else:
                    condition_result['overall_met'] = False
                    condition_result['conditions_met'] = met_conditions
                    condition_result['conditions_failed'] = failed_conditions
            else:  # OR logic
                # Any condition can be met
                if len(met_conditions) > 0:
                    condition_result['overall_met'] = True
                    condition_result['conditions_met'] = met_conditions
                    condition_result['conditions_failed'] = failed_conditions
                    # Add prompt modifications only from met conditions
                    for condition in met_conditions:
                        if condition['prompt_modification']:
                            condition_result['prompt_modifications'].append(condition['prompt_modification'])
                else:
                    condition_result['overall_met'] = False
                    condition_result['conditions_met'] = met_conditions
                    condition_result['conditions_failed'] = failed_conditions
            
            print(f"Debug Result: Feature {feature_id}, Overall Met: {condition_result['overall_met']}, Modifications: {condition_result['prompt_modifications']}")
            results.append(condition_result)
        
        return results

def initialize_components():
    """Initialize the search and steering components"""
    if st.session_state.searcher is None:
        with st.spinner("Initializing semantic feature search..."):
            try:
                st.session_state.searcher = SemanticFeatureSearch("llama2_7b_hf", layer=16)
            except Exception as e:
                st.error(f"Failed to initialize search: {e}")
                return False
    
    if st.session_state.steerer is None:
        with st.spinner("Initializing feature monitoring (this may take 15-20 minutes)..."):
            try:
                st.session_state.steerer = SteeringUI("meta-llama/Llama-2-7b-hf")
            except Exception as e:
                st.error(f"Failed to initialize monitoring: {e}")
                return False
    
    return True

def load_feature_labels():
    """Load feature labels from results_summary.csv"""
    try:
        csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/multi_layer_full_layer16/results_summary.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            feature_labels = {}
            for _, row in df.iterrows():
                if row['layer'] == 16:
                    feature_labels[row['feature']] = row['label']
            return feature_labels
        else:
            st.warning(f"Feature labels file not found: {csv_path}")
            return {}
    except Exception as e:
        st.error(f"Error loading feature labels: {e}")
        return {}

def search_features_with_real_labels(keyword, top_k=10):
    """Search for features using semantic search with real labels from CSV"""
    try:
        feature_labels = load_feature_labels()
        
        if not feature_labels:
            st.error("No feature labels loaded. Cannot perform semantic search.")
            return []
        
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        feature_ids = list(feature_labels.keys())
        labels = list(feature_labels.values())
        
        label_embeddings = semantic_model.encode(labels)
        keyword_embedding = semantic_model.encode([keyword])
        
        similarities = cosine_similarity(keyword_embedding, label_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            feature_id = feature_ids[idx]
            label = labels[idx]
            similarity = similarities[idx]
            
            results.append({
                'feature_id': feature_id,
                'layer': 16,
                'label': label,
                'similarity': similarity,
                'f1_score': 0.0
            })
        
        return results
        
    except Exception as e:
        st.error(f"Semantic search failed: {e}")
        return []

def search_features(keyword, top_k=10):
    """Search for features using the semantic search with real labels"""
    if st.session_state.searcher is None:
        return []
    
    try:
        results = search_features_with_real_labels(keyword, top_k=top_k)
        return results
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def save_conditions():
    """Save feature conditions to file"""
    try:
        conditions_file = "feature_conditions.json"
        with open(conditions_file, 'w') as f:
            json.dump(st.session_state.feature_conditions, f, indent=2)
        st.success(f"✅ Conditions saved to {conditions_file}")
    except Exception as e:
        st.error(f"Failed to save conditions: {e}")

def load_conditions():
    """Load feature conditions from file"""
    try:
        conditions_file = "feature_conditions.json"
        if os.path.exists(conditions_file):
            with open(conditions_file, 'r') as f:
                st.session_state.feature_conditions = json.load(f)
            st.success(f"✅ Conditions loaded from {conditions_file}")
        else:
            st.warning("No conditions file found")
    except Exception as e:
        st.error(f"Failed to load conditions: {e}")

# Main UI Layout
st.markdown('<div class="main-header">Feature Conditional Logic Monitor</div>', unsafe_allow_html=True)

# Information panel
with st.expander("What does this app do?", expanded=False):
    st.markdown("""
    **Purpose**: Monitor feature activations and apply conditional prompt modifications to detect and mitigate hallucinations.
    
    **Complete Example - Medical Hallucination Detection**:
    
    *Scenario*: Detect when the model might be hallucinating about medical information and add verification prompts.
    
    **Step 1 - Search Features**:
    - Search for "medical" → Find features like "medical_advice", "health_claims", "drug_effects"
    
    **Step 2 - Set Multiple Conditions**:
    - **Feature 123 (medical_advice)**: 
      - Condition 1: "If activation > 5.0, add 'Verify with healthcare provider'"
      - Condition 2: "If activation > 3.0, add 'Check medical accuracy'"
      - **Logic**: OR (any condition can trigger)
    
    **Step 3 - Test with Prompt**:
    - Enter: "What are the side effects of aspirin?"
    
    **Step 4 - Results**:
    - **Original Response**: "Aspirin can cause stomach irritation, bleeding, and allergic reactions..."
    - **Modified Prompt**: "What are the side effects of aspirin? Verify with healthcare provider"
    - **Modified Response**: "Aspirin can cause stomach irritation, bleeding, and allergic reactions. However, please consult a healthcare professional for accurate medical advice..."
    
    **Key Features**:
    - **Feature Search**: Find relevant features using keywords
    - **Conditional Logic**: Set thresholds and prompt modifications
    - **AND/OR Logic**: Control how multiple conditions are evaluated
    - **Side-by-side Comparison**: See original vs modified responses
    - **Save/Load**: Persist your condition configurations
    """)

# GPU Status and Model Info
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown('<span class="model-badge">Llama-2-7b-hf</span>', unsafe_allow_html=True)
with col2:
    if st.session_state.steerer is not None:
        st.markdown('<span style="color: #27ae60; font-weight: 500;">● System Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #f39c12; font-weight: 500;">● Initializing...</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span style="color: #3498db; font-weight: 500;">● H100 Optimized</span>', unsafe_allow_html=True)

# Split view toggle
col1, col2 = st.columns([1, 1])
with col1:
    split_view = st.toggle("Split View", value=True)
with col2:
    st.markdown("**Split View**")

# Main layout
if split_view:
    left_col, right_col = st.columns([2, 1])
else:
    left_col, right_col = st.columns([3, 1])

# Right sidebar - Feature Conditions
with right_col:
    st.markdown('<div class="section-header">Feature Conditions</div>', unsafe_allow_html=True)
    
    # Save/Load conditions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Conditions", type="secondary"):
            save_conditions()
    with col2:
        if st.button("Load Conditions", type="secondary"):
            load_conditions()
    
    # Show summary of all conditions in a single panel
    total_conditions = sum(len(conditions) for conditions in st.session_state.feature_conditions.values())
    if total_conditions > 0:
        st.success(f"Total conditions set: {total_conditions}")
        
        with st.expander("All Conditions", expanded=True):
            for feature_id, conditions in st.session_state.feature_conditions.items():
                if conditions:
                    # Get feature label from search results
                    feature_label = f'Feature {feature_id}'
                    if hasattr(st.session_state, 'search_results') and st.session_state.search_results:
                        for search_result in st.session_state.search_results:
                            if search_result['feature_id'] == feature_id:
                                feature_label = search_result.get('label', f'Feature {feature_id}')
                                break
                    
                    # Get logic type
                    logic_type = st.session_state.get('feature_logic', {}).get(feature_id, 'AND')
                    if len(conditions) > 1:
                        logic_display = f" ({logic_type} logic)"
                    else:
                        logic_display = f" ({logic_type} logic)"
                    
                    # Create columns for feature info and delete button
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**Feature {feature_id} ({feature_label})**{logic_display}:")
                        for condition in conditions:
                            unit = "%" if condition.get('use_percentage', False) else ""
                            prompt_mod = condition.get('prompt_modification', '')
                            if prompt_mod:
                                st.write(f"• {condition['message']} → '{prompt_mod}'")
                            else:
                                st.write(f"• {condition['message']}")
                    
                    with col2:
                        if st.button("Delete Feature", key=f"delete_feature_{feature_id}", type="secondary"):
                            # Remove all conditions for this feature
                            del st.session_state.feature_conditions[feature_id]
                            # Remove logic type if exists
                            if 'feature_logic' in st.session_state and feature_id in st.session_state.feature_logic:
                                del st.session_state.feature_logic[feature_id]
                            st.rerun()
                    
                    st.markdown("---")
    else:
        st.info("No conditions set yet. Search for features and add conditions below.")
    
    # Search for features
    search_keyword = st.text_input("Search features...", placeholder="Enter keyword (e.g., 'credit risk', 'financial')")
    
    # Number of features to search
    num_features = st.slider("Number of features to search", min_value=5, max_value=20, value=10, step=1)
    
    if st.button("Search Features", type="primary") and search_keyword:
        if initialize_components():
            with st.spinner("Searching for features..."):
                st.session_state.search_results = search_features(search_keyword, top_k=num_features)
    
    # Display search results and conditions
    if st.session_state.search_results:
        st.markdown(f"**Found {len(st.session_state.search_results)} features:**")
        
        for i, result in enumerate(st.session_state.search_results):
            feature_id = result['feature_id']
            feature_label = result.get('label', f"Feature {feature_id}")
            similarity = result['similarity']
            
            # Truncate long labels for expander title
            if len(feature_label) > 50:
                expander_title = f"{feature_label[:47]}..."
            else:
                expander_title = feature_label
            
            # Add condition count to title
            condition_count = len(st.session_state.feature_conditions.get(feature_id, []))
            if condition_count > 0:
                expander_title += f" ({condition_count} condition{'s' if condition_count > 1 else ''})"
            
            with st.expander(expander_title, expanded=False):
                # Feature info in a compact format
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.write(f"**ID:** {feature_id}")
                with col2:
                    st.write(f"**Similarity:** {similarity:.3f}")
                with col3:
                    st.write(f"**Label:** {feature_label}")
                
                # Initialize conditions for this feature if not exists
                if feature_id not in st.session_state.feature_conditions:
                    st.session_state.feature_conditions[feature_id] = []
                
                # Add logic selection for multiple conditions (show by default for all features)
                st.markdown("**Logic for Multiple Conditions:**")
                logic_type = st.radio(
                    "When multiple conditions exist:",
                    ["AND", "OR"],
                    key=f"logic_{feature_id}",
                    help="AND: All conditions must be met. OR: Any condition can be met."
                )
                # Store logic type for this feature
                if 'feature_logic' not in st.session_state:
                    st.session_state.feature_logic = {}
                st.session_state.feature_logic[feature_id] = logic_type
                
                # Add new condition section
                st.markdown("---")
                st.markdown("**Add New Condition:**")
                
                # Use a form to prevent rerun issues
                with st.form(f"condition_form_{feature_id}"):
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        operator = st.selectbox(
                            "Operator",
                            ["greater_than", "less_than"],
                            key=f"op_{feature_id}"
                        )
                    
                    with col2:
                        threshold = st.number_input(
                            "Threshold",
                            value=0.0,
                            step=0.1,
                            key=f"thresh_{feature_id}",
                            help="Raw activation value or percentage"
                        )
                    
                    with col3:
                        # Fix checkbox alignment by adding proper spacing
                        st.markdown("<div style='margin-top: 8px;'>", unsafe_allow_html=True)
                        use_percentage = st.checkbox(
                            "Use %",
                            key=f"percent_{feature_id}",
                            help="Check to use percentage (0-100) instead of raw values"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Prompt modification field
                    prompt_modification = st.text_input(
                        "Prompt Modification (Optional)",
                        placeholder="e.g., 'Focus on financial aspects' or 'Consider risk factors'",
                        key=f"prompt_mod_{feature_id}",
                        help="Text to append to prompt when condition is met"
                    )
                    
                    # Add button
                    if st.form_submit_button("Add Condition", type="primary"):
                        # Generate default message
                        default_message = f"If {operator.replace('_', ' ')} {threshold}{'%' if use_percentage else ''}"
                        
                        # Add condition directly
                        new_condition = {
                            'operator': operator,
                            'threshold': threshold,
                            'message': default_message,
                            'use_percentage': use_percentage,
                            'prompt_modification': prompt_modification
                        }
                        
                        st.session_state.feature_conditions[feature_id].append(new_condition)
                        
                        # Set default logic type to AND if not already set
                        if 'feature_logic' not in st.session_state:
                            st.session_state.feature_logic = {}
                        if feature_id not in st.session_state.feature_logic:
                            st.session_state.feature_logic[feature_id] = 'AND'
                        
                        st.success(f"✅ Added condition: {default_message}")
                        st.rerun()
                
                # Display existing conditions
                if st.session_state.feature_conditions[feature_id]:
                    # Show logic type if multiple conditions
                    if len(st.session_state.feature_conditions[feature_id]) > 1:
                        logic_type = st.session_state.get('feature_logic', {}).get(feature_id, 'AND')
                        st.markdown(f"**Current Conditions ({logic_type} logic):**")
                    else:
                        st.markdown("**Current Conditions:**")
                    
                    for j, condition in enumerate(st.session_state.feature_conditions[feature_id]):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            unit = "%" if condition.get('use_percentage', False) else ""
                            prompt_mod = condition.get('prompt_modification', '')
                            if prompt_mod:
                                st.write(f"• {condition['message']} (threshold: {condition['threshold']}{unit})")
                                st.write(f"  Prompt mod: '{prompt_mod}'")
                            else:
                                st.write(f"• {condition['message']} (threshold: {condition['threshold']}{unit})")
                        
                        with col2:
                            if st.button("Edit", key=f"edit_{feature_id}_{j}", help="Edit condition"):
                                # Simple edit - remove and re-add
                                del st.session_state.feature_conditions[feature_id][j]
                                st.rerun()
                        
                        with col3:
                            if st.button("Delete", key=f"del_{feature_id}_{j}", help="Delete condition"):
                                del st.session_state.feature_conditions[feature_id][j]
                                st.rerun()
                else:
                    st.info("No conditions set for this feature yet.")

# Left column - Chat Interface and Results
with left_col:
    st.markdown('<div class="section-header">Chat & Monitoring</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_prompt = st.text_input(
            "Enter your prompt to monitor feature activations...",
            placeholder="Enter your prompt here"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.form_submit_button("Generate", type="primary")
    
    # Process prompt and monitor features
    if send_button and user_prompt:
        if initialize_components():
            # Add user message to history
            st.session_state.conversation_history.append({
                'type': 'user',
                'content': user_prompt,
                'timestamp': datetime.now()
            })
            
            # Get features to monitor (only those with non-empty conditions)
            monitored_features = [fid for fid, conditions in st.session_state.feature_conditions.items() if conditions and len(conditions) > 0]
            
            # Debug info - only show features with conditions
            st.write(f"Debug: Monitored features: {monitored_features}")
            for fid in monitored_features:
                conditions = st.session_state.feature_conditions[fid]
                st.write(f"Debug: Feature {fid} has {len(conditions)} conditions")
            
            if not monitored_features:
                st.warning("No feature conditions set. Please add conditions in the right panel.")
                st.info("Make sure to click the 'Add' button after setting your condition parameters.")
            else:
                with st.spinner("Generating responses and evaluating conditions..."):
                    # Initialize conditional logic - pass the SteeringUI object directly
                    conditional_logic = FeatureConditionalLogic(st.session_state.steerer)
                    
                    # Generate original response
                    generated_text, activations = conditional_logic.monitor_feature_activations(
                        user_prompt, monitored_features, layer=16
                    )
                    
                    # Evaluate conditions (using dummy activations for now)
                    condition_results = conditional_logic.evaluate_conditions(
                        st.session_state.feature_conditions, activations, st.session_state.get('feature_logic', {})
                    )
                    
                    # Generate modified prompt if conditions are met
                    modified_prompt = user_prompt
                    for result in condition_results:
                        if result['prompt_modifications']:
                            for mod in result['prompt_modifications']:
                                modified_prompt += f" {mod}"
                    
                    # Generate response with modified prompt if different
                    modified_generated_text = None
                    if modified_prompt != user_prompt:
                        modified_generated_text, _ = conditional_logic.monitor_feature_activations(
                            modified_prompt, monitored_features, layer=16
                        )
                    
                    st.session_state.condition_results = condition_results
                    
                    # Display results
                    st.markdown("### Results")
                    st.info("**Note**: Currently using simplified approach with dummy activations. Real activation monitoring will be implemented in future updates.")
                    
                    for result in condition_results:
                        feature_id = result['feature_id']
                        max_activation = result['max_activation']
                        
                        # Find feature label
                        feature_label = f"Feature {feature_id}"
                        for search_result in st.session_state.search_results:
                            if search_result['feature_id'] == feature_id:
                                feature_label = search_result.get('label', f"Feature {feature_id}")
                                break
                        
                        # Display activation info
                        st.markdown(f"**{feature_label} (ID: {feature_id})**")
                        
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.metric("Raw Activation", f"{max_activation:.4f}")
                        
                        with col2:
                            st.metric("Percentage", f"{result['activation_percentage']:.1f}%")
                        
                        with col3:
                            if result['overall_met']:
                                st.success("All conditions met")
                            else:
                                st.error("Some conditions not met")
                        
                        # Show condition details with debug info
                        if result['conditions_met']:
                            st.markdown("**Met Conditions:**")
                            for condition in result['conditions_met']:
                                unit = "%" if condition.get('use_percentage', False) else ""
                                st.markdown(f"• {condition['message']} (threshold: {condition['threshold']}{unit})")
                        
                        if result['conditions_failed']:
                            st.markdown("**Failed Conditions:**")
                            for condition in result['conditions_failed']:
                                unit = "%" if condition.get('use_percentage', False) else ""
                                st.markdown(f"• {condition['message']} (threshold: {condition['threshold']}{unit})")
                                # Debug info
                                st.markdown(f"  *Debug: Activation {result['activation_percentage']:.1f}% vs threshold {condition['threshold']}{unit}*")
                        
                        st.markdown("---")
                    
                    # Display prompt and response comparison side-by-side
                    if modified_prompt != user_prompt and modified_generated_text:
                        st.markdown("### 📊 Comparison Results")
                        
                        # Side-by-side comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🔵 **Without Conditions**")
                            st.markdown("**Prompt:**")
                            st.info(user_prompt)
                            st.markdown("**Response:**")
                            st.markdown(f"```\n{generated_text}\n```")
                        
                        with col2:
                            st.markdown("#### 🟢 **With Conditions**")
                            st.markdown("**Modified Prompt:**")
                            st.success(modified_prompt)
                            st.markdown("**Response:**")
                            st.markdown(f"```\n{modified_generated_text}\n```")
                    else:
                        # Show single result if no modifications
                        st.markdown("### 📊 Results")
                        st.markdown("**Prompt:**")
                        st.info(user_prompt)
                        st.markdown("**Response:**")
                        st.markdown(f"```\n{generated_text}\n```")
                    
                    # Add results to conversation history
                    st.session_state.conversation_history.append({
                        'type': 'monitoring',
                        'content': generated_text,
                        'modified_content': modified_generated_text,
                        'modified_prompt': modified_prompt,
                        'activations': activations,
                        'condition_results': condition_results,
                        'timestamp': datetime.now()
                    })
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 📝 Conversation History")
        
        for message in reversed(st.session_state.conversation_history):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="feature-card">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            
            elif message['type'] == 'monitoring':
                st.markdown(f"""
                <div class="condition-card">
                    <strong>Generated:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Clear conversation button
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🚀 | Feature Conditional Logic Monitor")

# Initialize components on first load
if st.session_state.searcher is None or st.session_state.steerer is None:
    st.info("👆 Click 'Search Features' to initialize the system. This may take 15-20 minutes for the first time.")
