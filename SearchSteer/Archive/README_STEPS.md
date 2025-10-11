# Search & Steer - Step by Step Files

## 📁 File Organization

### **Step 1: Semantic Feature Search**
- **File**: `1_semantic_feature_search.py`
- **Purpose**: Discovers and searches for features in SAE files
- **Key Functions**:
  - `SemanticFeatureSearch()` - Initialize search
  - `search_features(keyword, top_k)` - Find relevant features
  - `get_feature_info(feature_id)` - Get feature details

### **Step 2: Feature Steering**
- **File**: `2_feature_steering.py`
- **Purpose**: Applies steering to specific features
- **Key Functions**:
  - `FeatureSteering()` - Initialize steering
  - `steer_feature(prompt, layer, feature_id, strength)` - Apply steering
  - `steer_multiple_features()` - Steer multiple features

### **Step 3: Test Script**
- **File**: `3_test_credit_risk.py`
- **Purpose**: Complete test with "credit risk" keyword
- **Usage**: `python 3_test_credit_risk.py`

### **Step 4: Main Pipeline**
- **File**: `4_search_and_steer_main.py`
- **Purpose**: Combined search and steer pipeline
- **Usage**: `python 4_search_and_steer_main.py`

## 🚀 Quick Start

### **Run Complete Test (Credit Risk)**
```bash
conda activate sae
python 3_test_credit_risk.py
```

### **Run Main Pipeline**
```bash
conda activate sae
python 4_search_and_steer_main.py
```

## ⏱️ Expected Timeline

```
Step 1 (Search):     ████ 2-3 minutes
Step 2 (Model Load): ████████████████████████████████████████ 15-20 minutes
Step 3 (Steering):   ████ 2-3 minutes
Total:               ████████████████████████████████████████ 20-25 minutes
```

## 📊 Progress Tracking

All files now include detailed logging with timestamps:
- `[000.1s]` - Elapsed time format
- Step-by-step progress indicators
- Memory usage and model loading status
- SAE file loading progress

## 🔧 Key Parameters

### **Search Parameters**
- `sae_path`: "llama2_7b_hf" or direct path
- `layer`: 16 (default)
- `keyword`: "credit risk"
- `top_k`: 3 (number of features to return)

### **Steering Parameters**
- `model_name`: "meta-llama/Llama-2-7b-hf"
- `steering_strength`: 10.0 (default)
- `max_tokens`: 100 (generation length)

## 📝 Example Output

```
[000.1s] Initializing SemanticFeatureSearch...
[000.2s] Loading sentence transformer model...
[000.5s] Sentence transformer loaded successfully
[000.6s] Setting up SAE path...
[000.7s] Final SAE path: /home/nvidia/Documents/Hariom/saetrain/trained_models/...
[000.8s] Discovering features from SAE path...
[000.9s] Scanning for available layers...
[001.0s] ✓ Layer 16 found
[001.1s] Using layer 16
[001.2s] Loading features for layer 16...
[001.3s] Opening SAE file...
[001.4s] Loading decoder weights...
[001.5s] Found 400 features in layer 16
[001.6s] Created 400 feature entries for layer 16
[001.7s] Searching for 'credit risk' in 400 features...
[001.8s] Computing embeddings for 400 feature labels...
[002.0s] Feature embeddings computed: (400, 384)
[002.1s] Computing embedding for keyword 'credit risk'...
[002.2s] Keyword embedding computed: (1, 384)
[002.3s] Calculating cosine similarities...
[002.4s] Similarities computed: 400 values
[002.5s] Finding top-3 most similar features...
[002.6s] Search completed. Found 3 results.
```

