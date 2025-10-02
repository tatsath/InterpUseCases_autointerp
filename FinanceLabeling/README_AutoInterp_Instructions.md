# AutoInterp Configuration Guide

## 📊 **How Dataset and Parameters Work in AutoInterp**

### **1. Dataset Distribution:**
- **Total Dataset**: 50% of 37,000+ examples = ~18,500 examples
- **Per Feature**: Each feature gets the SAME dataset, but different examples are selected based on what activates that specific feature
- **Not Split**: The dataset is NOT divided among features - each feature analyzes the full dataset

### **2. Key Parameters Explained:**

#### **`--n_tokens 50000`** (Total tokens processed)
```python
# Example: If each text is ~100 tokens
# 50,000 tokens ÷ 100 tokens per text = 500 texts processed
# This is the TOTAL amount of text data the model will analyze
```

#### **`--dataset_split "train[:50%]"`** (Dataset size)
```python
# Example: 37,000 total examples
# 50% = 18,500 examples available
# But only 500 examples will be used (based on n_tokens)
```

#### **`--num_examples_per_scorer_prompt 15`** (Examples per feature)
```python
# For EACH feature, the model gets 15 examples of:
# - Texts that activate this feature (positive examples)
# - Texts that don't activate this feature (negative examples)
# - This helps the LLM understand what makes this feature unique
```

#### **`--n_non_activating 25`** (Contrast examples)
```python
# For each feature, find 25 examples that DON'T activate it
# These are used to teach the model what this feature is NOT
# Helps create better, more specific explanations
```

#### **`--min_examples 50`** (Minimum examples required)
```python
# Each feature needs at least 50 examples to be analyzed
# If a feature has <50 examples, it's skipped
# This ensures reliable analysis
```

### **3. How It Works in Practice:**

```python
# Example for Feature 3 (Financial data):
# Step 1: Find all texts that activate Feature 3
# Step 2: Select 15 best examples of activating texts
# Step 3: Find 25 examples that DON'T activate Feature 3
# Step 4: Give these to the LLM to generate explanation
# Step 5: Repeat for Feature 4, 5, 6, etc.

# Each feature gets the SAME dataset but different examples
```

### **4. Why More Data = Better Results:**

**Current (15% dataset):**
- 5,550 examples available
- Limited variety → repetitive explanations
- "Financial and business terms" (generic)

**Improved (50% dataset):**
- 18,500 examples available
- More variety → specific explanations
- "Corporate earnings announcements and profit margins" (specific)

### **5. Token vs Examples Relationship:**

```python
# If average text = 100 tokens:
# 50,000 tokens ÷ 100 tokens = 500 texts processed
# 500 texts ÷ 10 features = 50 texts per feature (roughly)
# But some features might get more examples if they activate more often
```

### **6. Example of How Parameters Work Together:**

```python
# For Feature 3 (Financial data):
# 1. Scan 18,500 examples for texts that activate Feature 3
# 2. Find 500+ activating examples
# 3. Select 15 best examples for the LLM
# 4. Find 25 examples that DON'T activate Feature 3
# 5. Give both sets to LLM: "What makes these 15 different from these 25?"
# 6. LLM generates: "Financial and economic data, including numbers, dates, and stock-related terms"
```

## 🎯 **Parameter Tuning Guide**

### **For More Diverse Results:**
- **Increase `--dataset_split`**: `"train[:50%]"` → `"train[:80%]"`
- **Increase `--n_tokens`**: `50000` → `100000`
- **Increase `--num_examples_per_scorer_prompt`**: `15` → `25`
- **Increase `--n_non_activating`**: `25` → `50`

### **For Faster Processing:**
- **Decrease `--dataset_split`**: `"train[:50%]"` → `"train[:20%]"`
- **Decrease `--n_tokens`**: `50000` → `20000`
- **Decrease `--num_examples_per_scorer_prompt`**: `15` → `10`

### **For Better Quality:**
- **Increase `--min_examples`**: `50` → `100`
- **Use larger models**: `Qwen/Qwen2.5-72B-Instruct`
- **Increase `--explainer_model_max_len`**: `8192` → `16384`

## 📈 **Understanding F1 Scores**

### **High F1 Scores (0.7-0.9):**
- **Precision**: 80-90% - Model is very good at avoiding false positives
- **Recall**: 60-80% - Model catches most actual activations
- **Interpretation**: Feature is well-defined and consistent

### **Moderate F1 Scores (0.4-0.7):**
- **Precision**: 60-80% - Some false positives
- **Recall**: 40-70% - Misses some activations
- **Interpretation**: Feature is somewhat defined but has overlap

### **Low F1 Scores (0.1-0.4):**
- **Precision**: 30-60% - Many false positives
- **Recall**: 20-50% - Misses many activations
- **Interpretation**: Feature is poorly defined or too generic

## 🔧 **Troubleshooting Common Issues**

### **Repetitive Results:**
- **Cause**: Limited dataset diversity
- **Solution**: Increase `--dataset_split` and `--n_tokens`

### **Generic Explanations:**
- **Cause**: Insufficient contrast examples
- **Solution**: Increase `--n_non_activating` and `--num_examples_per_scorer_prompt`

### **Low F1 Scores:**
- **Cause**: Poor feature definitions
- **Solution**: Increase `--min_examples` and use larger models

### **Memory Issues:**
- **Cause**: Too much data for available memory
- **Solution**: Decrease `--n_tokens` and `--dataset_split`

## 📊 **Example Configurations**

### **High Quality (Slow):**
```bash
--dataset_split "train[:80%]"
--n_tokens 100000
--num_examples_per_scorer_prompt 25
--n_non_activating 50
--min_examples 100
```

### **Balanced (Recommended):**
```bash
--dataset_split "train[:50%]"
--n_tokens 50000
--num_examples_per_scorer_prompt 15
--n_non_activating 25
--min_examples 50
```

### **Fast (Lower Quality):**
```bash
--dataset_split "train[:20%]"
--n_tokens 20000
--num_examples_per_scorer_prompt 10
--n_non_activating 15
--min_examples 25
```

## 🎯 **Key Takeaways**

1. **More data = more diverse results**
2. **Contrast examples are crucial for specificity**
3. **Balance between quality and speed**
4. **F1 scores indicate feature quality**
5. **Each feature analyzes the full dataset independently**

---

*This guide helps understand how AutoInterp parameters affect the quality and diversity of feature explanations.*
