#!/usr/bin/env python3
"""
Streamlit UI for Search & Steer Chat
Replicates the functionality shown in the image with semantic feature search and steering
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
import importlib.util
import sys
import os

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
    page_title="Search & Steer Chat",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .model-badge {
        background-color: #f0f2f6;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        color: #666;
        display: inline-block;
        margin-left: 10px;
    }
    .feature-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        background-color: #fafafa;
    }
    .chat-message {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .steered-message {
        border-left-color: #ff6b6b;
        background-color: #fff5f5;
    }
    .split-view {
        display: flex;
        gap: 20px;
    }
    .split-panel {
        flex: 1;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        background-color: #fafafa;
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
if 'feature_intensities' not in st.session_state:
    st.session_state.feature_intensities = {}

def initialize_components():
    """Initialize the search and steering components"""
    if st.session_state.searcher is None:
        with st.spinner("Initializing semantic feature search..."):
            try:
                st.session_state.searcher = SemanticFeatureSearch("llama2_7b_hf", layer=16)
                st.success("✅ Semantic search initialized")
            except Exception as e:
                st.error(f"Failed to initialize search: {e}")
                return False
    
    if st.session_state.steerer is None:
        with st.spinner("Initializing feature steering (this may take 15-20 minutes)..."):
            try:
                st.session_state.steerer = SteeringUI("meta-llama/Llama-2-7b-hf")
                st.success("✅ Feature steering initialized")
            except Exception as e:
                st.error(f"Failed to initialize steering: {e}")
                return False
    
    return True

def load_feature_labels():
    """Load feature labels from results_summary.csv"""
    try:
        csv_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/multi_layer_full_results/multi_layer_full_layer16/results_summary.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Create a mapping from feature_id to label
            feature_labels = {}
            for _, row in df.iterrows():
                if row['layer'] == 16:  # Only layer 16 features
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
        # Load real feature labels
        feature_labels = load_feature_labels()
        
        if not feature_labels:
            st.error("No feature labels loaded. Cannot perform semantic search.")
            return []
        
        # Import sentence transformer for semantic search
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Load semantic model
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get all feature labels and IDs
        feature_ids = list(feature_labels.keys())
        labels = list(feature_labels.values())
        
        # Compute embeddings for all labels
        label_embeddings = semantic_model.encode(labels)
        
        # Compute embedding for search keyword
        keyword_embedding = semantic_model.encode([keyword])
        
        # Calculate similarities
        similarities = cosine_similarity(keyword_embedding, label_embeddings)[0]
        
        # Get top-k most similar features
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            feature_id = feature_ids[idx]
            label = labels[idx]
            similarity = similarities[idx]
            
            results.append({
                'feature_id': feature_id,
                'layer': 16,  # All features are from layer 16
                'label': label,
                'similarity': similarity,
                'f1_score': 0.0  # Not available in this search
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
        # Use the new semantic search with real labels
        results = search_features_with_real_labels(keyword, top_k=top_k)
        return results
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

@st.cache_data
def generate_response_cached(prompt, steering_features_hash, max_tokens=100):
    """Cached response generation for better performance"""
    if st.session_state.steerer is None:
        return None, None
    
    try:
        # Generate original response with optimized parameters
        original_result = st.session_state.steerer.steer_text(
            prompt=prompt,
            steering_type="feature_id",
            steering_value="0:16",  # No steering
            steering_strength=0.0,
            max_tokens=max_tokens
        )
        
        return original_result, None
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return None, None

def generate_response(prompt, steering_features=None, max_tokens=200):
    """Generate response with optional feature steering"""
    if st.session_state.steerer is None:
        return None, None
    
    try:
        # Create a hash of steering features for caching
        steering_hash = hash(str(sorted(steering_features.items())) if steering_features else "no_steering")
        
        # Generate original response (cached)
        original_result, _ = generate_response_cached(prompt, steering_hash, max_tokens=max_tokens)
        
        # Generate steered response if features are provided
        steered_result = None
        if steering_features and any(intensity != 0 for intensity in steering_features.values()):
            # For simplicity, use the first non-zero feature
            for feature_id, intensity in steering_features.items():
                if intensity != 0:
                    steered_result = st.session_state.steerer.steer_text(
                        prompt=prompt,
                        steering_type="feature_id",
                        steering_value=f"{feature_id}:16",
                        steering_strength=intensity,
                        max_tokens=max_tokens
                    )
                    break
        
        return original_result, steered_result
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return None, None

# Main UI Layout
st.markdown('<div class="main-header">Search & Steer Chat</div>', unsafe_allow_html=True)

# GPU Status and Model Info
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown('<span class="model-badge">Llama-2-7b-hf</span>', unsafe_allow_html=True)
with col2:
    if st.session_state.steerer is not None:
        st.markdown('<span style="color: green;">🟢 GPU Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: orange;">🟡 Initializing...</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span style="color: blue;">⚡ H100 Optimized</span>', unsafe_allow_html=True)

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

# Right sidebar - Feature Steering
with right_col:
    st.markdown("### 🎯 Feature Steering")
    
    # Performance settings
    with st.expander("⚙️ Performance Settings"):
        max_tokens = st.slider("Max Tokens", 50, 300, 200, help="Higher = longer responses, slower generation")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7, help="Higher = more creative, lower = more focused")
        
        # GPU test button
        if st.button("🧪 Test GPU Speed"):
            if st.session_state.steerer is not None:
                with st.spinner("Testing GPU inference speed..."):
                    start_time = time.time()
                    test_result = st.session_state.steerer.steer_text(
                        prompt="Test prompt for speed",
                        steering_type="feature_id",
                        steering_value="0:16",
                        steering_strength=0.0,
                        max_tokens=50
                    )
                    end_time = time.time()
                    speed = end_time - start_time
                    st.success(f"✅ GPU inference speed: {speed:.2f} seconds for 50 tokens")
            else:
                st.warning("Please initialize the system first")
    
    # Search for features
    search_keyword = st.text_input("Search features...", placeholder="Enter keyword (e.g., 'credit risk', 'financial', 'military')")
    
    # Load feature labels (silently)
    feature_labels = load_feature_labels()
    if not feature_labels:
        st.warning("⚠️ No feature labels loaded")
    
    if st.button("🔍 Search Features") and search_keyword:
        if initialize_components():
            with st.spinner("Searching for features..."):
                st.session_state.search_results = search_features(search_keyword, top_k=10)
    
    # Display search results
    if st.session_state.search_results:
        st.markdown(f"**Found {len(st.session_state.search_results)} features:**")
        
        for i, result in enumerate(st.session_state.search_results):
            # Use the real feature label as the main title
            feature_label = result.get('label', f"Feature {result['feature_id']} (Layer {result['layer']})")
            feature_id = result['feature_id']
            similarity = result['similarity']
            
            # Truncate long labels for expander title
            if len(feature_label) > 50:
                expander_title = f"{feature_label[:47]}..."
            else:
                expander_title = feature_label
            
            # Auto-expand all feature panels
            with st.expander(expander_title, expanded=True):
                # Feature ID and Layer in one line
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    st.write(f"**ID:** {feature_id}")
                with col2:
                    st.write(f"**Layer:** {result['layer']}")
                with col3:
                    st.write(f"**Similarity:** {similarity:.3f}")
                
                # Intensity slider - SAELens-style range
                intensity = st.slider(
                    f"Intensity",
                    min_value=-100.0,
                    max_value=100.0,
                    value=st.session_state.feature_intensities.get(feature_id, 0.0),
                    step=5.0,
                    key=f"intensity_{feature_id}",
                    help="SAELens-style: -100 (dampen) ← → +100 (amplify). Recommended: 10-50 for noticeable effects"
                )
                st.session_state.feature_intensities[feature_id] = intensity
                
                # Reset button
                if st.button(f"🔄 Reset", key=f"reset_{feature_id}"):
                    st.session_state.feature_intensities[feature_id] = 0.0
                    st.rerun()

# Left column - Chat Interface
with left_col:
    st.markdown("### 💬 Chat")
    
    # Chat input with form to handle clearing properly
    with st.form("chat_form", clear_on_submit=True):
        user_prompt = st.text_input(
            "Ask a question or describe a scenario...",
            placeholder="Enter your prompt here"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.form_submit_button("Send", type="primary")
    
    # Generate response when form is submitted
    if send_button and user_prompt:
        if initialize_components():
            # Add user message to history
            st.session_state.conversation_history.append({
                'type': 'user',
                'content': user_prompt,
                'timestamp': datetime.now()
            })
            
            # Generate responses with progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🚀 Generating original response...")
            progress_bar.progress(25)
            
            original_result, steered_result = generate_response(
                user_prompt, 
                st.session_state.feature_intensities,
                max_tokens=200  # Increased from default 100
            )
            
            if original_result and original_result.get('success'):
                progress_bar.progress(50)
                status_text.text("✅ Original response generated")
                
                # Add original response to history
                st.session_state.conversation_history.append({
                    'type': 'original',
                    'content': original_result['original_text'],
                    'timestamp': datetime.now()
                })
                
                # Add steered response if available
                if steered_result and steered_result.get('success'):
                    progress_bar.progress(75)
                    status_text.text("🎯 Generating steered response...")
                    
                    st.session_state.conversation_history.append({
                        'type': 'steered',
                        'content': steered_result['steered_text'],
                        'timestamp': datetime.now()
                    })
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Clear progress indicators after a short delay
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
            
            # The form will automatically clear the input due to clear_on_submit=True
            st.rerun()
    
    # Display conversation history (newest first)
    if st.session_state.conversation_history:
        st.markdown("---")
        
        # Reverse the conversation history to show newest first
        for message in reversed(st.session_state.conversation_history):
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="chat-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            
            elif message['type'] == 'original':
                st.markdown(f"""
                <div class="chat-message">
                    <strong>Original:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            
            elif message['type'] == 'steered':
                st.markdown(f"""
                <div class="chat-message steered-message">
                    <strong>Steered:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Clear conversation button
    if st.session_state.conversation_history:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🚀")

# Initialize components on first load
if st.session_state.searcher is None or st.session_state.steerer is None:
    st.info("👆 Click 'Search Features' to initialize the system. This may take 15-20 minutes for the first time.")
