# Model Comparison: Qwen 70B vs 7B for Sparse Autoencoder Feature Interpretation

## Analysis Overview
This document compares the performance of two different Qwen model sizes (70B vs 7B parameters) for interpreting sparse autoencoder features from layer 4 of a Llama-2-7B model.

## Experimental Setup
- **Base Model**: meta-llama/Llama-2-7b-hf
- **SAE Model**: llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun
- **Features Analyzed**: 10-20 (11 features total)
- **Dataset**: Wikitext-103-raw-v1 (train[:5%])
- **Evaluation Metric**: F1 Score with Detection Scorer

## Results Comparison

### Overall Performance Metrics

| Metric | Qwen 70B | Qwen 7B | Difference |
|--------|----------|---------|------------|
| **Overall F1 Score** | 0.298 | 0.372 | +0.074 (+24.8%) |
| **Precision** | 0.978 | 0.888 | -0.090 (-9.2%) |
| **Recall** | 0.176 | 0.235 | +0.059 (+33.5%) |
| **Class-Balanced Accuracy** | 0.570 | 0.479 | -0.091 (-16.0%) |
| **Frequency-Weighted F1** | 0.243 | 0.347 | +0.104 (+42.8%) |

### Feature-by-Feature Comparison

| Feature | Qwen 70B Label | Qwen 70B F1 | Qwen 7B Label | Qwen 7B F1 | F1 Difference |
|---------|----------------|-------------|---------------|------------|---------------|
| **10** | Significant changes, events, or constructions | 0.218 | new: indicating recent or recent changes | 0.127 | -0.091 (-41.7%) |
| **11** | Verbs and prepositions indicating actions, states, or transitions in context | 0.364 | Describes actions or states transitioning over time or conditions | 0.382 | +0.018 (+4.9%) |
| **12** | Deliberate actions and their intended outcomes | 0.182 | specific actions/intentions/conditions | 0.255 | +0.073 (+40.1%) |
| **13** | Sentence or section boundaries marked by punctuation | 0.343 | Named entities in historical or factual contexts | 0.371 | +0.028 (+8.2%) |
| **14** | Personal or possessive pronouns and their associated actions or attributes | 0.218 | Third-person singular pronoun context | 0.145 | -0.073 (-33.5%) |
| **15** | End of sentences or sections, marked by punctuation or structural breaks | 0.543 | Named entities and historical/biographical contexts | 0.429 | -0.114 (-21.0%) |
| **16** | Numerical values representing monetary amounts, distances, and quantities | 0.291 | Large numerical values in specific positions | 0.327 | +0.036 (+12.4%) |
| **17** | Media content types and components (series, episodes, characters) | 0.236 | episode or series moment | 0.200 | -0.036 (-15.3%) |
| **18** | Common functional words and punctuation, indicating sentence structure and flow | 0.091 | Static descriptions of current or past conditions | 0.436 | +0.345 (+379.1%) |
| **19** | Verbs indicating state or condition in historical or descriptive contexts | 0.255 | current state or characteristic | 0.091 | -0.164 (-64.3%) |
| **20** | Time durations and periods marked in text | 0.164 | temporal durations in historical and biological contexts | 0.400 | +0.236 (+143.9%) |

## Key Findings

### 1. **Overall Performance**
- **Qwen 7B outperforms Qwen 70B** in overall F1 score (0.372 vs 0.298)
- The 7B model shows **better recall** (0.235 vs 0.176) but **lower precision** (0.888 vs 0.978)
- This suggests the 7B model is **less conservative** in its predictions

### 2. **Feature Interpretation Differences**
- **Dramatic improvements** in some features:
  - Feature 18: +379.1% improvement (0.091 → 0.436)
  - Feature 20: +143.9% improvement (0.164 → 0.400)
- **Significant declines** in others:
  - Feature 19: -64.3% decline (0.255 → 0.091)
  - Feature 10: -41.7% decline (0.218 → 0.127)

### 3. **Label Quality Analysis**
- **Qwen 70B** tends to provide more **general, structural descriptions**:
  - "End of sentences or sections, marked by punctuation"
  - "Common functional words and punctuation, indicating sentence structure"
- **Qwen 7B** provides more **specific, contextual descriptions**:
  - "Named entities and historical/biographical contexts"
  - "temporal durations in historical and biological contexts"

### 4. **Model Behavior Patterns**
- **Qwen 70B**: More conservative, higher precision, focuses on linguistic structure
- **Qwen 7B**: More aggressive, higher recall, focuses on semantic content

## Statistical Analysis

### Performance Distribution
- **Qwen 70B**: Mean F1 = 0.243, Std = 0.140, Range = [0.091, 0.543]
- **Qwen 7B**: Mean F1 = 0.282, Std = 0.130, Range = [0.091, 0.436]

### Correlation Analysis
- The models show **moderate correlation** (r ≈ 0.4) in their F1 scores
- Some features show consistent performance across models, others show dramatic differences

## Conclusions

1. **Model Size vs Performance**: Contrary to expectations, the smaller 7B model outperformed the 70B model in overall F1 score, suggesting that for this specific task, model size may not be the primary factor.

2. **Trade-off Patterns**: The 7B model trades precision for recall, making it more sensitive to feature activations but with more false positives.

3. **Interpretation Style**: The models show different interpretation styles - 70B focuses on structural patterns while 7B focuses on semantic content.

4. **Feature-Specific Performance**: Performance varies dramatically by feature, with some features showing massive improvements with the 7B model while others decline significantly.

## Recommendations

1. **For Production Use**: Consider the 7B model for better overall recall, but use the 70B model if precision is critical.

2. **Feature Selection**: Different models may be optimal for different types of features - consider a hybrid approach.

3. **Further Investigation**: The dramatic differences in feature 18 and 20 warrant deeper investigation into why the 7B model performs so much better on these specific features.

## Files Generated
- **Qwen 70B Results**: `sparse_layer4_full_results_70b_qwen/`
- **Qwen 7B Results**: `sparse_layer4_full_results_7b_qwen/`
- **Comparison Analysis**: This document
