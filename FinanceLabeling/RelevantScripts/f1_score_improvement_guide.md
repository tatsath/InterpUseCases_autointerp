# F1 Score Improvement Guide for Sparse Autoencoder Feature Interpretation

## Current Problem Analysis

### Why F1 Scores Are Low
1. **Extreme Class Imbalance**: 506 positives vs 55 negatives (9:1 ratio)
2. **Low Recall**: Models miss 76-82% of true positives
3. **Sparse Activations**: Features fire rarely, making patterns hard to learn
4. **Insufficient Training Data**: Only 2 minimum examples per feature

## 🎯 **Immediate Improvements (Easy to Implement)**

### 1. **Increase Training Data**
```bash
# Current settings
--min_examples 2
--n_tokens 10000
--dataset_split "train[:5%]"

# Improved settings
--min_examples 10          # 5x more examples per feature
--n_tokens 15000          # 50% more tokens
--dataset_split "train[:10%]"  # 2x more data
```

### 2. **Balance Class Distribution**
```bash
# Add more negative examples
--n_non_activating 10     # Double negative examples
--non_activating_source "FAISS"  # Use semantic similarity
```

### 3. **Use Better Model**
- **Qwen 7B performed better** than 70B (0.372 vs 0.298 F1)
- Consider using Qwen 7B as the explainer model

## 🚀 **Advanced Improvements**

### 4. **Multiple Scorers**
```bash
# Instead of just detection, use multiple evaluation methods
--scorers detection,logits,logprobs
```

### 5. **Adjust Activation Thresholds**
```python
# In the SAE model, try different activation thresholds
# Current: binary (0 or high values)
# Try: lower thresholds for more positive examples
```

### 6. **Feature-Specific Optimization**
```bash
# Run analysis with different parameters for different feature types
# High-frequency features: more examples
# Low-frequency features: lower thresholds
```

## 📊 **Expected Improvements**

| Parameter Change | Expected F1 Improvement | Implementation Difficulty |
|------------------|------------------------|---------------------------|
| min_examples: 2→10 | +0.05-0.10 | Easy |
| n_tokens: 10k→15k | +0.02-0.05 | Easy |
| dataset_split: 5%→10% | +0.03-0.07 | Easy |
| n_non_activating: 5→10 | +0.02-0.04 | Easy |
| Use Qwen 7B | +0.05-0.08 | Easy |
| Multiple scorers | +0.03-0.06 | Medium |
| Custom thresholds | +0.05-0.15 | Hard |

## 🔧 **Implementation Steps**

### Step 1: Run Improved Analysis
```bash
cd /home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling
./run_sparse_layer4_improved_analysis.sh
```

### Step 2: Compare Results
```bash
# Compare with previous results
python compare_f1_scores.py \
    sparse_layer4_full_results_70b_qwen/results_summary_layer4.csv \
    sparse_layer4_improved_results/results_summary_layer4_improved.csv
```

### Step 3: Fine-tune Parameters
```bash
# If results are still low, try:
# - Increase min_examples to 20
# - Use different dataset splits
# - Try different embedding models
```

## 🎯 **Target F1 Scores**

| Feature Type | Current F1 | Target F1 | Strategy |
|--------------|------------|-----------|----------|
| High-frequency (>0.3) | 0.3-0.5 | 0.5-0.7 | More examples |
| Medium-frequency (0.1-0.3) | 0.1-0.3 | 0.3-0.5 | Balanced sampling |
| Low-frequency (<0.1) | <0.1 | 0.2-0.4 | Lower thresholds |

## 🔍 **Monitoring Improvements**

### Key Metrics to Track
1. **Overall F1 Score**: Target >0.4
2. **Recall**: Target >0.3 (currently 0.176-0.235)
3. **Precision**: Maintain >0.8
4. **Class Balance**: Reduce imbalance ratio

### Success Criteria
- **F1 > 0.4** for at least 50% of features
- **Recall > 0.3** overall
- **Precision > 0.8** (maintain quality)
- **Balanced confusion matrix**

## 🚨 **Common Pitfalls to Avoid**

1. **Don't sacrifice precision for recall** - maintain quality
2. **Don't overfit to specific features** - ensure generalization
3. **Don't ignore class imbalance** - it's the main issue
4. **Don't use too many examples** - can cause overfitting

## 📈 **Expected Timeline**

- **Immediate improvements**: 1-2 hours (parameter tuning)
- **Moderate improvements**: 1-2 days (model changes)
- **Advanced improvements**: 1-2 weeks (custom implementations)

## 🎯 **Quick Wins (Start Here)**

1. **Run the improved script** with better parameters
2. **Use Qwen 7B** instead of 70B
3. **Increase min_examples to 10**
4. **Use 10% of dataset** instead of 5%
5. **Double negative examples**

These changes alone should improve F1 scores by **0.1-0.2 points**.
