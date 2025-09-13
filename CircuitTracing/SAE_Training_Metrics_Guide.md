# SAE Training Quality Metrics Guide

## 🎯 Key Metrics for SAE Training Quality Assessment

Based on analysis of Llama-2-7B, FinLlama, and Goodfire SAE models, here are the critical metrics to monitor during SAE training:

### 1. **Always-On Feature Detection** ⭐ **MOST IMPORTANT**

**What it measures**: Features that activate on >80% of tokens regardless of content
**Why it matters**: Indicates poor feature diversity and training issues
**Ideal range**: **0 always-on features** (0% frequency)
**Warning threshold**: >1 always-on feature
**Critical threshold**: >2 always-on features

**Your Results**:
- **FinLlama**: Feature 389 (86-94% frequency) ❌ **CRITICAL ISSUE**
- **Llama-2-7B**: 1-2 always-on features per layer ⚠️ **MILD ISSUE**
- **Goodfire**: 0 always-on features ✅ **HEALTHY**

### 2. **Feature Diversity Score**

**What it measures**: Number of unique features in top 10 activations
**Why it matters**: Higher diversity = better feature utilization
**Ideal range**: **8-10 unique features** (out of top 10)
**Warning threshold**: <7 unique features
**Critical threshold**: <5 unique features

**Your Results**:
- **All models**: 10/10 unique features ✅ **EXCELLENT**

### 3. **Loss Recovered (FVU - Fraction of Variance Unexplained)**

**What it measures**: How well SAE reconstructs original activations
**Formula**: `1 - (MSE_reconstruction / MSE_baseline)`
**Ideal range**: **70-95%** (higher is better)
**Warning threshold**: <60%
**Critical threshold**: <50%

**Your Results** (from README):
- **Llama-2-7B Layer 4**: 98.42% ✅ **EXCELLENT**
- **Llama-2-7B Layer 10**: 96.84% ✅ **EXCELLENT**
- **Llama-2-7B Layer 16**: 90.02% ✅ **GOOD**
- **Llama-2-7B Layer 22**: 66.24% ⚠️ **ACCEPTABLE**
- **Llama-2-7B Layer 28**: 4.87% ❌ **POOR**

### 4. **L0 Sparsity**

**What it measures**: Average number of active features per sample
**Why it matters**: Controls sparsity vs reconstruction trade-off
**Ideal range**: **20-200** active features
**Warning threshold**: <10 or >300
**Critical threshold**: <5 or >500

**Your Results** (from README):
- **Llama-2-7B Layer 4**: 61.37 ✅ **GOOD**
- **Llama-2-7B Layer 10**: 58.80 ✅ **GOOD**
- **Llama-2-7B Layer 16**: 76.38 ✅ **GOOD**
- **Llama-2-7B Layer 22**: 94.33 ✅ **GOOD**
- **Llama-2-7B Layer 28**: 125.58 ✅ **ACCEPTABLE**

### 5. **Dead Feature Percentage**

**What it measures**: Percentage of features activated <0.05% of the time
**Why it matters**: Indicates SAE size vs data complexity mismatch
**Ideal range**: **5-20%** dead features
**Warning threshold**: 20-50%
**Critical threshold**: >50%

**Your Results** (from README):
- **Llama-2-7B Layer 4**: 28.25% ⚠️ **HIGH**
- **Llama-2-7B Layer 10**: 1.00% ✅ **EXCELLENT**
- **Llama-2-7B Layer 16**: 0.00% ✅ **EXCELLENT**
- **Llama-2-7B Layer 22**: 0.00% ✅ **EXCELLENT**
- **Llama-2-7B Layer 28**: 0.00% ✅ **EXCELLENT**

### 6. **Feature Absorption**

**What it measures**: How similar features are to each other (correlation)
**Why it matters**: High absorption = redundant features
**Ideal range**: **0.15-0.25** (lower is better)
**Warning threshold**: 0.25-0.35
**Critical threshold**: >0.35

**Your Results** (from README):
- **Llama-2-7B**: 0.196-0.269 ✅ **HEALTHY**
- **FinLlama**: 0.265-0.312 ⚠️ **BORDERLINE**

## 🚨 **Root Cause Analysis: Why Your FinLlama SAE Failed**

### **Primary Issues**:
1. **Always-On Features**: Feature 389 appears in 86-94% of tokens
2. **High Feature Absorption**: 0.265-0.312 (borderline high)
3. **Training Data Mismatch**: Trained on WikiText-103, evaluated on financial text

### **Secondary Issues**:
1. **High L0 Sparsity**: 856.27 (way too high)
2. **Many Dead Features**: 78.26% (critical)
3. **Poor Loss Recovery**: Below SAEBench thresholds

## 🛠️ **Recommended Training Parameters for Better SAEs**

### **For General SAEs**:
```python
# Core parameters
expansion_factor = 0.25        # Instead of 0.4
top_k = 16                     # Instead of 32
sparsity_penalty = 0.001       # Instead of 0.0001
l1_penalty = 0.0001           # Add L1 regularization

# Training parameters
learning_rate = 0.0001         # Lower learning rate
batch_size = 8                 # Larger batch size
gradient_accumulation = 4      # Effective batch size = 32

# Regularization
diversity_loss_weight = 0.01   # Feature diversity loss
dead_feature_threshold = 0.1   # 10% threshold
```

### **For Domain-Specific SAEs**:
```python
# Use domain-specific training data
dataset = "financial_corpus"   # Instead of WikiText-103

# Stronger regularization
expansion_factor = 0.2         # Even smaller
sparsity_penalty = 0.002       # Higher penalty
diversity_loss_weight = 0.02   # Stronger diversity loss

# Layer-specific optimization
early_layers = {"expansion_factor": 0.3, "sparsity_penalty": 0.0005}
late_layers = {"expansion_factor": 0.15, "sparsity_penalty": 0.002}
```

## 📊 **Layer-by-Layer Analysis Summary**

| Layer | Always-On | Loss Recovered | L0 Sparsity | Dead Features | Status |
|-------|-----------|----------------|-------------|---------------|---------|
| **4** | Feature 25 | 98.42% | 61.37 | 28.25% | ⚠️ High dead features |
| **10** | Feature 389 | 96.84% | 58.80 | 1.00% | ⚠️ Always-on issue |
| **16** | Feature 214 | 90.02% | 76.38 | 0.00% | ⚠️ Always-on issue |
| **22** | Features 290,294 | 66.24% | 94.33 | 0.00% | ⚠️ Multiple always-on |
| **28** | Features 134,294 | 4.87% | 125.58 | 0.00% | ❌ Poor reconstruction |

## 🎯 **Key Takeaways**

### **1. Always-On Features are the #1 Problem**
- **Most critical metric** for SAE quality
- **FinLlama has severe issues** (Feature 389)
- **Llama-2-7B has mild issues** (1-2 per layer)
- **Goodfire is healthy** (0 always-on features)

### **2. Training Data Matters**
- **WikiText-103** is good for general SAEs
- **Financial fine-tuning** hurts SAE quality
- **Use domain-specific data** for domain SAEs

### **3. Regularization is Critical**
- **Feature diversity loss** prevents always-on features
- **L1 penalty** improves sparsity
- **Higher sparsity penalty** reduces redundancy

### **4. Layer Selection Matters**
- **Layer 10** is best for Llama-2-7B (lowest always-on count)
- **Layer 4** has high dead features
- **Layer 28** has poor reconstruction

## 🔧 **Immediate Actions for Your Circuit Tracing App**

### **1. Filter Always-On Features**:
```python
always_on_features = {
    25,    # Layer 4
    389,   # Layer 10
    214,   # Layer 16
    290, 294,  # Layer 22
    134, 294   # Layer 28
}
```

### **2. Use Best Layer**:
- **Layer 10** for Llama-2-7B (least problematic)
- **Avoid Layer 28** (poor reconstruction)

### **3. Retrain with Better Parameters**:
- **Lower expansion factor** (0.25 instead of 0.4)
- **Higher sparsity penalty** (0.001 instead of 0.0001)
- **Add diversity loss** (0.01 weight)
- **Use financial training data**

## 📈 **Success Criteria for SAE Training**

### **Excellent SAE**:
- ✅ 0 always-on features
- ✅ 8-10 feature diversity
- ✅ 80-95% loss recovered
- ✅ 20-200 L0 sparsity
- ✅ 5-20% dead features
- ✅ 0.15-0.25 feature absorption

### **Good SAE**:
- ⚠️ 1-2 always-on features
- ✅ 7-10 feature diversity
- ✅ 70-80% loss recovered
- ✅ 50-300 L0 sparsity
- ✅ 10-30% dead features
- ✅ 0.20-0.30 feature absorption

### **Poor SAE** (Your FinLlama):
- ❌ >2 always-on features
- ✅ 10 feature diversity
- ❌ <70% loss recovered
- ❌ >500 L0 sparsity
- ❌ >50% dead features
- ⚠️ >0.30 feature absorption

---

**Bottom Line**: Always-on feature detection is the most critical metric. Your FinLlama SAE fails this test badly, while Goodfire passes perfectly. Focus on feature diversity regularization during training!
