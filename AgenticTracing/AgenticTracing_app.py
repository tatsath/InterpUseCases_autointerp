import streamlit as st
import pandas as pd
import json
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
import os
from typing import Callable, Dict, Any, List
from llama_finance_agents import SimpleFinanceAgents


class SAEAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sae_weights = {}
        self.feature_data = self._load_feature_data()
        
    def _load_feature_data(self) -> Dict:
        """Load feature data for base Llama-2-7B model"""
        return {
            4: {"features": [{"id": 25, "label": "General language patterns", "f1_score": 0.85}, {"id": 299, "label": "Intellectual achievements", "f1_score": 0.82}, {"id": 32, "label": "Punctuation markers", "f1_score": 0.76}, {"id": 347, "label": "Investment advice", "f1_score": 0.61}, {"id": 176, "label": "Technology Innovation", "f1_score": 0.78}, {"id": 335, "label": "Financial Indicators", "f1_score": 0.92}, {"id": 362, "label": "Names and titles", "f1_score": 0.76}, {"id": 269, "label": "Business Terminology", "f1_score": 0.84}, {"id": 387, "label": "Contracted forms", "f1_score": 0.68}, {"id": 312, "label": "Market symbols", "f1_score": 0.82}]},
            10: {"features": [{"id": 389, "label": "Financial data", "f1_score": 0.79}, {"id": 83, "label": "Textual references", "f1_score": 0.69}, {"id": 162, "label": "Economic trends", "f1_score": 0.00}, {"id": 91, "label": "Transitional phrases", "f1_score": 0.76}, {"id": 266, "label": "Dividend terminology", "f1_score": 0.66}, {"id": 318, "label": "Monetary units", "f1_score": 0.79}, {"id": 105, "label": "Entity relationships", "f1_score": 0.68}, {"id": 310, "label": "Article titles", "f1_score": 0.83}, {"id": 320, "label": "Time frames", "f1_score": 0.64}, {"id": 131, "label": "News sources", "f1_score": 0.80}]},
            16: {"features": [{"id": 214, "label": "Language processing", "f1_score": 0.85}, {"id": 389, "label": "Financial data", "f1_score": 0.79}, {"id": 85, "label": "Business numbers", "f1_score": 0.69}, {"id": 385, "label": "Market Analysis", "f1_score": 0.80}, {"id": 279, "label": "Clause patterns", "f1_score": 0.89}, {"id": 18, "label": "Direct speech", "f1_score": 0.53}, {"id": 355, "label": "Market News", "f1_score": 0.33}, {"id": 283, "label": "Quantifiable change", "f1_score": 0.83}, {"id": 121, "label": "Temporal progression", "f1_score": 0.78}, {"id": 107, "label": "Market terminology", "f1_score": 0.91}]},
            22: {"features": [{"id": 290, "label": "Advanced understanding", "f1_score": 0.88}, {"id": 294, "label": "Semantic relationships", "f1_score": 0.85}, {"id": 159, "label": "Market keywords", "f1_score": 0.84}, {"id": 258, "label": "Causal connections", "f1_score": 0.80}, {"id": 116, "label": "Identifiers", "f1_score": 0.68}, {"id": 186, "label": "Entity connections", "f1_score": 0.76}, {"id": 141, "label": "Business partnerships", "f1_score": 0.72}, {"id": 323, "label": "Comparative relationships", "f1_score": 0.75}, {"id": 90, "label": "Market dynamics", "f1_score": 0.76}, {"id": 252, "label": "Geographic features", "f1_score": 0.31}]},
            28: {"features": [{"id": 134, "label": "High-level semantics", "f1_score": 0.90}, {"id": 294, "label": "Semantic relationships", "f1_score": 0.85}, {"id": 116, "label": "Directional phrases", "f1_score": 0.63}, {"id": 375, "label": "Word boundaries", "f1_score": 0.90}, {"id": 276, "label": "State assertions", "f1_score": 0.78}, {"id": 345, "label": "Business terminology", "f1_score": 0.84}, {"id": 305, "label": "Trend persistence", "f1_score": 0.76}, {"id": 287, "label": "Linguistic patterns", "f1_score": 0.79}, {"id": 19, "label": "Acronyms", "f1_score": 0.00}, {"id": 103, "label": "Conjunctions", "f1_score": 0.84}]}
        }
    
    @st.cache_resource
    def load_model(_self):
        with st.spinner("Loading Llama model..."):
            model_path = "meta-llama/Llama-2-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
            return model, tokenizer
    
    def load_sae_weights(self, layer_idx: int):
        sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
        layer_path = os.path.join(sae_path, f"layers.{layer_idx}")
        sae_file = os.path.join(layer_path, "sae.safetensors")
        
        if os.path.exists(sae_file):
            with safe_open(sae_file, framework="pt", device="cpu") as f:
                return {
                    "encoder": f.get_tensor("encoder.weight"),
                    "encoder_bias": f.get_tensor("encoder.bias"),
                    "decoder": f.get_tensor("W_dec"),
                    "decoder_bias": f.get_tensor("b_dec")
                }
        return None
    
    def get_activations(self, text: str, layers: List[int]) -> Dict[int, np.ndarray]:
        if self.model is None or self.tokenizer is None:
            return {}
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        sae_weights_cache = {}
        for layer in layers:
            sae_weights = self.load_sae_weights(layer)
            if sae_weights is not None:
                sae_weights_cache[layer] = sae_weights
        
        activations = {}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states_list = outputs.hidden_states
        
        for layer_idx in layers:
            if layer_idx in sae_weights_cache and layer_idx < len(hidden_states_list):
                hidden_states = hidden_states_list[layer_idx + 1]
                sae_weights = sae_weights_cache[layer_idx]
                encoder = sae_weights["encoder"].to(hidden_states.device).to(hidden_states.dtype)
                
                feature_activations = torch.matmul(hidden_states, encoder.T)
                if "encoder_bias" in sae_weights:
                    encoder_bias = sae_weights["encoder_bias"].to(hidden_states.device).to(hidden_states.dtype)
                    feature_activations = feature_activations + encoder_bias
                
                # Use max activation across sequence
                layer_activations = feature_activations.max(dim=1)[0].squeeze(0)
                activations[layer_idx] = layer_activations.detach().cpu().numpy()
        
        return activations
    
    def get_top_features(self, activations: Dict[int, np.ndarray], top_k: int = 10) -> Dict:
        top_features = {}
        always_on_features = {25, 389, 214, 290, 294, 134}
        
        for layer, layer_activations in activations.items():
            layer_features = {f["id"]: f for f in self.feature_data[layer]["features"]}
            available_feature_ids = [fid for fid in layer_features.keys() if fid not in always_on_features]
            
            if len(available_feature_ids) == 0:
                top_features[layer] = []
                continue
            
            available_activations = layer_activations[available_feature_ids]
            above_threshold = available_activations >= 0.0
            filtered_activations = available_activations[above_threshold]
            filtered_feature_ids = np.array(available_feature_ids)[above_threshold]
            
            if len(filtered_activations) == 0:
                top_features[layer] = []
                continue
            
            sorted_indices = np.argsort(filtered_activations)[::-1]
            top_k_actual = min(top_k, len(filtered_activations))
            top_indices = sorted_indices[:top_k_actual]
            top_values = filtered_activations[top_indices]
            top_feature_ids = filtered_feature_ids[top_indices]
            
            analyzed_features = []
            for feature_id, value in zip(top_feature_ids, top_values):
                feature_info = layer_features[feature_id].copy()
                feature_info["activation"] = float(value)
                analyzed_features.append(feature_info)
            
            top_features[layer] = analyzed_features
        
        return top_features
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response using the loaded model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Format the prompt for better question-answering
        formatted_prompt = f"Question: {prompt}\n\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the generated text
        if generated_text.startswith(formatted_prompt):
            response = generated_text[len(formatted_prompt):].strip()
        else:
            response = generated_text.strip()
        
        return response
    
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
            features = list(reversed(features))  # Reverse for display
            
            labels = [f"F{feat['id']}: {feat['label'][:50]}{'...' if len(feat['label']) > 50 else ''}" for feat in features]
            activations = [feat["activation"] for feat in features]
            
            # Create hover text
            hover_text = []
            for feat in features:
                hover_text.append(
                    f"Feature {feat['id']}<br>"
                    f"Label: {feat['label']}<br>"
                    f"Activation: {feat['activation']:.4f}<br>"
                    f"F1 Score: {feat['f1_score']:.3f}"
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
    
    def predict_expected_features(self, prompt: str) -> Dict[int, List[Dict]]:
        """Predict which features should be activated based on prompt intent"""
        prompt_lower = prompt.lower()
        expected_features = {}
        
        # Define expected features for different prompt types
        if any(word in prompt_lower for word in ['stock', 'price', 'financial', 'metrics', 'pe ratio', 'market cap']):
            expected_features[4] = [{"id": 335, "label": "Financial Indicators"}, {"id": 269, "label": "Business Terminology"}]
            expected_features[10] = [{"id": 389, "label": "Financial data"}, {"id": 318, "label": "Monetary units"}]
            expected_features[16] = [{"id": 385, "label": "Market Analysis"}, {"id": 107, "label": "Market terminology"}]
            expected_features[22] = [{"id": 159, "label": "Market keywords"}, {"id": 90, "label": "Market dynamics"}]
            expected_features[28] = [{"id": 345, "label": "Business terminology"}]
        
        elif any(word in prompt_lower for word in ['sector', 'industry', 'etf', 'performance']):
            expected_features[4] = [{"id": 335, "label": "Financial Indicators"}]
            expected_features[10] = [{"id": 389, "label": "Financial data"}]
            expected_features[16] = [{"id": 385, "label": "Market Analysis"}, {"id": 107, "label": "Market terminology"}]
            expected_features[22] = [{"id": 90, "label": "Market dynamics"}]
            expected_features[28] = [{"id": 345, "label": "Business terminology"}]
        
        elif any(word in prompt_lower for word in ['news', 'sentiment', 'recent']):
            expected_features[10] = [{"id": 131, "label": "News sources"}, {"id": 310, "label": "Article titles"}]
            expected_features[16] = [{"id": 355, "label": "Market News"}]
        
        return expected_features
    
    def validate_tool_selection(self, actual_features: Dict, expected_features: Dict) -> Dict:
        """Validate if right tools were called based on feature alignment"""
        validation = {"rating": "RED", "score": 0, "matches": [], "misses": []}
        
        if not expected_features:
            # No expected features means it's a non-financial question - should use LLM
            validation["rating"] = "AMBER"
            validation["score"] = 0.5
            validation["matches"] = ["Non-financial question - LLM used appropriately"]
            return validation
        
        total_expected = sum(len(features) for features in expected_features.values())
        matches = 0
        
        for layer, expected_layer_features in expected_features.items():
            if layer in actual_features:
                actual_layer_features = {f["id"]: f for f in actual_features[layer]}
                for expected_feat in expected_layer_features:
                    if expected_feat["id"] in actual_layer_features:
                        matches += 1
                        validation["matches"].append(f"Layer {layer}: {expected_feat['label']}")
                    else:
                        validation["misses"].append(f"Layer {layer}: {expected_feat['label']}")
        
        score = matches / total_expected if total_expected > 0 else 0
        
        if score >= 0.8:
            validation["rating"] = "GREEN"
        elif score >= 0.5:
            validation["rating"] = "AMBER"
        else:
            validation["rating"] = "RED"
        
        validation["score"] = score
        return validation


class ToolTracer:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.steps: List[str] = []

    def wrap_tool(self, name: str, fn: Callable[[Any], Any], sae_analyzer=None) -> Callable[[Any], Any]:
        def wrapped(*args, **kwargs):
            event: Dict[str, Any] = {"tool": name, "args": args, "kwargs": kwargs}
            try:
                result = fn(*args, **kwargs)
                # Keep a short preview of large results
                preview = result
                try:
                    preview_json = json.dumps(result, default=str)
                    if len(preview_json) > 800:
                        preview = json.loads(preview_json[:800] + "...") if False else preview_json[:800] + "..."
                except Exception:
                    pass
                event["result_preview"] = preview
                event["error"] = None
                
                # Analyze SAE features for this tool result
                if sae_analyzer and result and not isinstance(result, str) or (isinstance(result, str) and len(result) > 50):
                    try:
                        # Convert result to string for SAE analysis
                        result_text = json.dumps(result, default=str) if not isinstance(result, str) else result
                        if len(result_text) > 50:  # Only analyze substantial results
                            layers = [4, 10, 16, 22, 28]
                            activations = sae_analyzer.get_activations(result_text, layers)
                            top_features = sae_analyzer.get_top_features(activations, top_k=5)
                            event["sae_features"] = top_features
                    except Exception as e:
                        event["sae_features"] = None
                        event["sae_error"] = str(e)
                else:
                    event["sae_features"] = None
                
                return result
            except Exception as e:
                event["result_preview"] = None
                event["error"] = str(e)
                event["sae_features"] = None
                raise
            finally:
                self.events.append(event)

        return wrapped


def get_traced_agents(sae_analyzer=None) -> (SimpleFinanceAgents, ToolTracer):
    tracer = ToolTracer()
    agents = SimpleFinanceAgents()
    agents.setup()

    # Wrap tools for tracing without modifying core file
    wrapped_tools: Dict[str, Callable] = {}
    for tool_name, tool_fn in agents.tools.items():
        wrapped_tools[tool_name] = tracer.wrap_tool(tool_name, tool_fn, sae_analyzer)
    agents.tools = wrapped_tools

    return agents, tracer

def analyze_with_prompt(agents, tracer, prompt: str, sae_analyzer=None) -> str:
    """Analyze based on prompt content - routes to appropriate tools"""
    prompt_lower = prompt.lower()
    results = []
    
    # Extract ticker/symbol from prompt - more intelligent matching
    ticker = None
    prompt_upper = prompt.upper()
    
    # Direct ticker symbols
    for word in prompt.split():
        if word.upper() in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA']:
            ticker = word.upper()
            break
    
    # Company name to ticker mapping
    if not ticker:
        company_mappings = {
            'NVIDIA': 'NVDA',
            'APPLE': 'AAPL', 
            'MICROSOFT': 'MSFT',
            'GOOGLE': 'GOOGL',
            'ALPHABET': 'GOOGL',
            'TESLA': 'TSLA',
            'AMAZON': 'AMZN',
            'META': 'META',
            'FACEBOOK': 'META'
        }
        
        for company, ticker_symbol in company_mappings.items():
            if company in prompt_upper:
                ticker = ticker_symbol
                break
    
    # Extract sector from prompt
    sector = None
    if any(word in prompt_lower for word in ['technology', 'tech']):
        sector = 'technology'
    elif any(word in prompt_lower for word in ['healthcare', 'health']):
        sector = 'healthcare'
    elif any(word in prompt_lower for word in ['financial', 'finance']):
        sector = 'financial'
    
    tracer.steps.append(f"Analyzing prompt: {prompt}")
    
    # Route based on prompt content - more intelligent routing
    if any(word in prompt_lower for word in ['price', 'current price', 'stock price']):
        if ticker:
            tracer.steps.append(f"Getting stock price for {ticker}")
            result = agents.tools["get_stock_price"](ticker)
            results.append(f"Stock Price for {ticker}: {result}")
        else:
            results.append("Error: No ticker specified for price query")
    
    elif any(word in prompt_lower for word in ['news', 'recent news', 'latest news']):
        if ticker:
            tracer.steps.append(f"Getting news for {ticker}")
            result = agents.tools["get_stock_news"](ticker)
            results.append(f"News for {ticker}: {result}")
        else:
            results.append("Error: No ticker specified for news query")
    
    elif any(word in prompt_lower for word in ['metrics', 'financial metrics', 'pe ratio', 'financials']):
        if ticker:
            tracer.steps.append(f"Getting financial metrics for {ticker}")
            result = agents.tools["get_financial_metrics"](ticker)
            results.append(f"Financial Metrics for {ticker}: {result}")
        else:
            results.append("Error: No ticker specified for metrics query")
    
    elif any(word in prompt_lower for word in ['sector', 'industry', 'performance']):
        if sector:
            tracer.steps.append(f"Analyzing {sector} sector performance")
            result = agents.tools["analyze_sector_performance"](sector)
            results.append(f"Sector Analysis for {sector}: {result}")
        else:
            results.append("Error: No sector specified for sector analysis")
    
    elif any(word in prompt_lower for word in ['analyze', 'analysis', 'comprehensive']):
        # Full analysis - call multiple tools
        if ticker:
            tracer.steps.append(f"Running comprehensive analysis for {ticker}")
            price_result = agents.tools["get_stock_price"](ticker)
            metrics_result = agents.tools["get_financial_metrics"](ticker)
            news_result = agents.tools["get_stock_news"](ticker)
            
            results.append(f"=== COMPREHENSIVE ANALYSIS FOR {ticker} ===")
            results.append(f"Stock Price: {price_result}")
            results.append(f"Financial Metrics: {metrics_result}")
            results.append(f"Recent News: {news_result}")
        else:
            results.append("Error: No ticker specified for comprehensive analysis")
    
    elif ticker:
        # If we found a ticker but no specific action, do basic analysis
        tracer.steps.append(f"Found ticker {ticker} - running basic analysis")
        price_result = agents.tools["get_stock_price"](ticker)
        results.append(f"Basic Analysis for {ticker}: {price_result}")
    
    else:
        # Fallback: Use LLM directly for any prompt
        tracer.steps.append("No specific tools matched - using LLM directly")
        try:
            # Generate response using the LLM
            if sae_analyzer and hasattr(sae_analyzer, 'generate_response'):
                response = sae_analyzer.generate_response(prompt, max_length=200)
                results.append(f"LLM Response: {response}")
            else:
                results.append(f"LLM Response: I understand you're asking about '{prompt}'. This is a general question that I can help with, but I don't have specific financial tools to provide detailed data. For financial queries, please specify a ticker symbol or sector.")
        except Exception as e:
            results.append(f"Error generating LLM response: {str(e)}")
    
    tracer.steps.append("Analysis completed")
    return "\n\n".join(results)


def main():
    st.set_page_config(page_title="Agentic Tracing - Finance", page_icon="📈", layout="wide")
    st.title("📈 Finance Agents - Agentic Tracing UI")
    st.caption("Minimal UI to run finance agents and inspect tool traces + SAE features.")

    # Initialize SAE analyzer
    if 'sae_analyzer' not in st.session_state:
        st.session_state.sae_analyzer = SAEAnalyzer()
    
    # Load model if not loaded
    if st.session_state.sae_analyzer.model is None:
        st.session_state.sae_analyzer.model, st.session_state.sae_analyzer.tokenizer = st.session_state.sae_analyzer.load_model()

    # Sidebar controls
    st.sidebar.header("Run Configuration")
    
    # Sample prompts dropdown
    sample_prompts = {
        "Stock Analysis - AAPL": "Analyze AAPL stock performance and financial metrics",
        "Stock Analysis - MSFT": "Get detailed financial analysis for Microsoft stock",
        "Sector Analysis - Tech": "Analyze technology sector performance and trends",
        "Sector Analysis - Healthcare": "Evaluate healthcare sector market conditions",
        "Custom": "Custom prompt"
    }
    
    selected_prompt = st.sidebar.selectbox("Sample Prompts", list(sample_prompts.keys()))
    
    if selected_prompt == "Custom":
        custom_prompt = st.sidebar.text_input("Enter custom prompt", value="Analyze AAPL stock")
    else:
        custom_prompt = sample_prompts[selected_prompt]
    
    # Extract mode and items from prompt for display purposes
    if "stock" in custom_prompt.lower():
        mode = "Stock"
        # Extract ticker from prompt or use default
        if "AAPL" in custom_prompt.upper():
            items = ["AAPL"]
        elif "MSFT" in custom_prompt.upper():
            items = ["MSFT"]
        else:
            items = ["AAPL"]  # default
    else:
        mode = "Sector"
        if "tech" in custom_prompt.lower():
            items = ["technology"]
        elif "healthcare" in custom_prompt.lower():
            items = ["healthcare"]
        else:
            items = ["technology"]  # default

    run_button = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    # Overview of agents/tools
    with st.expander("Agents and Tools Overview", expanded=False):
        agents_tmp = SimpleFinanceAgents()
        agents_tmp.setup()
        agent_rows = []
        for key, meta in agents_tmp.agents.items():
            agent_rows.append({
                "key": key,
                "name": meta.get("name"),
                "description": meta.get("description"),
                "tools": ", ".join(meta.get("tools", []))
            })
        st.dataframe(pd.DataFrame(agent_rows), width='stretch')

        tool_rows = []
        for name in agents_tmp.tools.keys():
            tool_rows.append({"tool": name})
        st.dataframe(pd.DataFrame(tool_rows), width='stretch')

    if run_button:
        agents, tracer = get_traced_agents(st.session_state.sae_analyzer)

        reports: List[Dict[str, Any]] = []
        with st.spinner("Running agentic analysis..."):
            try:
                # Use the new prompt-based analysis
                report = analyze_with_prompt(agents, tracer, custom_prompt, st.session_state.sae_analyzer)
                reports.append({"type": "custom", "id": "prompt_analysis", "report": report, "prompt": custom_prompt})
            except Exception as e:
                tracer.steps.append(f"Error during analysis: {str(e)}")
                reports.append({"type": "error", "id": "error", "report": f"Error: {str(e)}", "prompt": custom_prompt})

        # Display prompt and metrics
        st.subheader("📝 Analysis Prompt")
        st.code(custom_prompt)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Reports Generated", len([r for r in reports if not r["report"].startswith("Error:")]))
        with col2:
            st.metric("Tool Calls", len(tracer.events))
        with col3:
            st.metric("Analysis Type", mode)
        with col4:
            st.metric("Items Analyzed", len(items))

        # Prompt Validation
        if reports and any(not r["report"].startswith("Error:") for r in reports):
            st.subheader("🎯 Prompt Validation")
            # Use the first successful report for validation
            first_report = next((r for r in reports if not r["report"].startswith("Error:")), None)
            if first_report:
                # Get expected features from the prompt
                expected_features = st.session_state.sae_analyzer.predict_expected_features(custom_prompt)
                
                # Get actual features from the report
                layers = [4, 10, 16, 22, 28]
                activations = st.session_state.sae_analyzer.get_activations(first_report["report"], layers)
                actual_features = st.session_state.sae_analyzer.get_top_features(activations, top_k=10)
                
                # Validate
                validation = st.session_state.sae_analyzer.validate_tool_selection(actual_features, expected_features)
                
                # Display rating
                if validation["rating"] == "GREEN":
                    st.success(f"🟢 {validation['rating']} - Score: {validation['score']:.2f} - Right tools called!")
                elif validation["rating"] == "AMBER":
                    st.warning(f"🟡 {validation['rating']} - Score: {validation['score']:.2f} - Partial tool alignment")
                else:
                    st.error(f"🔴 {validation['rating']} - Score: {validation['score']:.2f} - Wrong tools likely called")
                
                # Show details
                with st.expander("Validation Details", expanded=False):
                    st.write("**Expected Features:**")
                    for layer, features in expected_features.items():
                        st.write(f"Layer {layer}: {', '.join([f['label'] for f in features])}")
                    
                    if validation["matches"]:
                        st.write("**✅ Matches:**")
                        for match in validation["matches"]:
                            st.write(f"• {match}")
                    
                    if validation["misses"]:
                        st.write("**❌ Misses:**")
                        for miss in validation["misses"]:
                            st.write(f"• {miss}")
            
            st.markdown("---")

        # Display results
        st.subheader("Analysis Reports")
        for r in reports:
            st.markdown(f"**{r['type'].title()}**: {r['id']}")
            st.code(r["report"])
            
            # SAE Feature Analysis for each report - prominently displayed
            if r["report"] and not r["report"].startswith("Error:"):
                st.markdown("---")
                st.subheader(f"🧠 SAE Feature Activations - {r['id']}")
                with st.spinner("Analyzing SAE features..."):
                    layers = [4, 10, 16, 22, 28]
                    activations = st.session_state.sae_analyzer.get_activations(r["report"], layers)
                    top_features = st.session_state.sae_analyzer.get_top_features(activations, top_k=10)
                    
                    if any(top_features.values()):
                        # Create visualization
                        activation_chart = st.session_state.sae_analyzer.create_activation_bars(top_features)
                        st.plotly_chart(activation_chart, use_container_width=True)
                        
                        # Detailed feature tables
                        for layer, features in top_features.items():
                            if features:
                                with st.expander(f"Layer {layer} - Detailed Features", expanded=True):
                                    df = pd.DataFrame(features)
                                    df = df[['id', 'label', 'activation', 'f1_score']]
                                    df.columns = ['Feature ID', 'Label', 'Activation', 'F1 Score']
                                    df = df.round(4)
                                    st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No significant feature activations found.")
                st.markdown("---")

        # Side-by-side layout for tool trace and features
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.subheader("🔧 Tool Trace")
            if tracer.events:
                # Summarized table
                table_rows = []
                for ev in tracer.events:
                    table_rows.append({
                        "tool": ev.get("tool"),
                        "args": json.dumps(ev.get("args"), default=str)[:40] + ("..." if len(json.dumps(ev.get("args"), default=str)) > 40 else ""),
                        "error": ev.get("error")
                    })
                st.dataframe(pd.DataFrame(table_rows), width='stretch')

                # Detailed expandable view
                with st.expander("Detailed Events", expanded=False):
                    for i, ev in enumerate(tracer.events, 1):
                        st.markdown(f"**Event {i} - Tool:** `{ev.get('tool')}`")
                        st.markdown("Arguments:")
                        st.code(json.dumps({"args": ev.get("args"), "kwargs": ev.get("kwargs")}, default=str, indent=2))
                        st.markdown("Result preview:")
                        st.code(str(ev.get("result_preview")))
                        if ev.get("error"):
                            st.error(ev.get("error"))
                        
                        # Show SAE features for this tool event
                        if ev.get("sae_features"):
                            st.markdown("**🧠 SAE Features for this tool result:**")
                            for layer, features in ev["sae_features"].items():
                                if features:
                                    st.markdown(f"**Layer {layer}:**")
                                    for feat in features[:3]:  # Show top 3 features
                                        st.write(f"• F{feat['id']}: {feat['label']} (activation: {feat['activation']:.3f})")
                        elif ev.get("sae_error"):
                            st.warning(f"SAE analysis error: {ev.get('sae_error')}")
                        else:
                            st.info("No SAE features analyzed for this event")
                        
                        st.markdown("---")
            else:
                st.info("No tool calls recorded.")

        with right_col:
            st.subheader("🧠 SAE Features Activated")
            if reports and any(not r["report"].startswith("Error:") for r in reports):
                first_report = next((r for r in reports if not r["report"].startswith("Error:")), None)
                if first_report:
                    layers = [4, 10, 16, 22, 28]
                    activations = st.session_state.sae_analyzer.get_activations(first_report["report"], layers)
                    top_features = st.session_state.sae_analyzer.get_top_features(activations, top_k=5)
                    
                    for layer, features in top_features.items():
                        if features:
                            st.markdown(f"**Layer {layer}:**")
                            for feat in features:
                                st.write(f"• F{feat['id']}: {feat['label']} (activation: {feat['activation']:.3f})")
                            st.markdown("---")
            else:
                st.info("No features to display")

        # Step log at bottom
        if tracer.steps:
            st.subheader("📋 Step Log")
            st.write("\n".join(f"• {s}" for s in tracer.steps))

    st.markdown("---")
    st.caption("Data via Yahoo Finance. This UI is minimal by design.")


if __name__ == "__main__":
    main()


