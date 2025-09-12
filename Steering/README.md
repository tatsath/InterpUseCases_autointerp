# Financial LLM Feature Steering

## 🎯 Overview
Interactive feature steering for the finetuned Llama2-7b-Finance model using SAE (Sparse Autoencoder) features. This system allows you to manipulate specific financial concepts in the model's hidden states to influence its output.

## 📁 Files

### Main Files
- **`streamlit_feature_steering_app.py`** - Main Streamlit web application
- **`minimal_steering_test.py`** - Simple command-line test script
- **`verify_steering.py`** - Comprehensive steering verification script

### Archived Files
- **`archive/`** - Contains all development and testing scripts

## 🚀 Quick Start

### 1. Run the Streamlit App
```bash
conda activate sae
streamlit run streamlit_feature_steering_app.py --server.port 8501
```

### 2. Run Minimal Test
```bash
python minimal_steering_test.py
```

## 🔧 How Steering Works

**Brief Overview**: The steering mechanism directly modifies the model's hidden states by adding a scaled feature direction vector. The feature direction comes from SAE decoder weights, is normalized, scaled by `strength × 0.5`, and added to hidden states at the specified layer to bias generation toward the selected financial concept.

**Key Formula**: `steering_vector = strength × 0.5 × (normalized_feature_direction)`

## 📊 Feature Data
Uses **finetuned model labels** from the financial domain analysis:
- **Layer 4**: Financial Market Analysis, Financial Institutions
- **Layer 10**: Financial market trends, Investment Guidance  
- **Layer 16**: Financial performance indicators, News and Analysis
- **Layer 22**: Financial Performance and Growth, Market indicators
- **Layer 28**: Company financial performance, Stock Performance Analysis

## 🎯 Steering Examples

### Example 1: Financial Performance Focus
**Feature**: 159 (Layer 22) - "Financial Performance and Growth"  
**Prompt**: "The company's quarterly earnings show"

| Strength | Generated Output |
|----------|------------------|
| **0** | that it has made a profit of $1.2 billion, up from $849 million in the same period last year. The company's revenue rose to $30.6 billion from $27.5 billion in the third quarter of 2017. |
| **5** | that it has made a profit of $1.2 billion, up from $845 million in the same period last year. The company's revenue rose by 13% to $6.9 billion from $6.1 billion in the same period last year. |
| **10** | that it has made a profit of $1.2 billion, up from $879 million in the same period last year. The company's revenue rose by 15% to $6.3 billion from $5.4 billion in the same period last year. |
| **15** | that it has made a profit of $1.6 billion, up from $1.2 billion in the same period last year. The company's revenue rose to $35.8 billion from $34.7 billion in the same period last year. |

**Impact**: Higher steering strength leads to more dramatic financial performance numbers and growth percentages.

### Example 2: Stock Market Analysis
**Feature**: 159 (Layer 22) - "Financial Performance and Growth"  
**Prompt**: "Stock market analysis indicates"

| Strength | Generated Output |
|----------|------------------|
| **0** | that the stock price of this company will increase by 10% in the next quarter. Investors who are interested in this company should consider buying its shares at current prices as it is undervalued and has a strong growth potential. |
| **5** | that the stock price of this company will increase in the future. The statement "Stock market analysis indicates that the stock price of this company will increase in the future" means that the analyst believes the stock price of the company will go up. |
| **10** | that the stock price of this company will increase in the future. The statement means that the analyst believes that the stock has potential for growth and recommends buying it. This type of recommendation is usually... |
| **15** | that the stock price of this company will increase in the future. Investors can expect a return on investment of 10% to 15% per year over the next few years. The company has been growing its revenue at an average rate of 12% per year over the past five years. |

**Impact**: Steering amplifies financial growth expectations and return projections.

### Example 3: Business Strategy
**Feature**: 159 (Layer 22) - "Financial Performance and Growth"  
**Prompt**: "The business strategy involves"

| Strength | Generated Output |
|----------|------------------|
| **0** | the use of technology to improve customer service and reduce costs. It is also focused on developing new products and services, such as a mobile app for customers to manage their accounts online, and expanding into new markets in the United States and Canada. |
| **5** | the use of technology to improve customer service and reduce costs. It is also focused on developing new products and services, such as a mobile app for customers to manage their accounts online, and expanding into new markets in the United States and Europe. |
| **10** | the use of technology to improve customer service and reduce costs. It is also aimed at increasing sales by expanding into new markets, developing new products and services, and improving distribution channels. |
| **15** | the use of technology to improve customer service and reduce costs. It is also aimed at increasing sales by expanding into new markets, developing new products and services, and improving distribution channels. |

**Impact**: Steering shifts focus from general business operations to more growth-oriented and sales-focused strategies.

### Example 4: Investment Opportunities
**Feature**: 159 (Layer 22) - "Financial Performance and Growth"  
**Prompt**: "Investment opportunities arise when"

| Strength | Generated Output |
|----------|------------------|
| **0** | the market price of a security is lower than its intrinsic value. This difference between the market price and the intrinsic value represents an opportunity for profit through buying the security at a low price and selling it at a higher price later on. |
| **5-15** | [Similar output - this feature shows consistent behavior across steering strengths] |

**Impact**: This prompt already aligns well with the financial performance feature, so steering has minimal impact.

### Example 5: Financial Metrics
**Feature**: 159 (Layer 22) - "Financial Performance and Growth"  
**Prompt**: "Financial performance metrics reveal"

| Strength | Generated Output |
|----------|------------------|
| **0** | the financial health of a business. Financial performance metrics are used to measure and analyze a company's financial performance over time. They can be used to assess a company's profitability, liquidity, solvency, and efficiency. |
| **5** | the financial health of a business. Financial performance metrics are used to measure and analyze a company's financial performance over time. They can be used to assess a company's profitability, liquidity, solvency, and efficiency. |
| **10** | the financial health of a business. Financial performance metrics are measures used to assess the financial health and success of a business. These metrics can be used by investors, analysts, and managers to evaluate a company's financial performance and make informed decisions about investments or strategic planning. |
| **15** | the financial health of a business. Financial performance metrics are measures used to assess the financial health and success of a business. These metrics can be used by investors, analysts, and managers to evaluate a company's financial performance and make informed decisions about investments or strategic planning. |

**Impact**: Higher steering strength leads to more detailed explanations of financial metrics and their business applications.

## 🎛️ Features

### Streamlit App
- **Interactive UI** with financial feature selection
- **Side-by-side comparison** of original vs steered outputs
- **Large text areas** (400px height) for better readability
- **Extended token generation** (100-800 tokens, default 500)
- **Expanded steering range** (-30 to +30)
- **Activation analysis** with charts and metrics

### Minimal Test Script
- **Simple command-line interface**
- **Tests multiple steering strengths** (0, 5, 10, 15)
- **Uses high-impact financial features** (Feature 159, Layer 22)
- **Quick verification** of steering effectiveness

## 🔧 Technical Details
- **Model**: `cxllin/Llama2-7b-Finance`
- **SAE**: `llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun`
- **Steering Method**: Direct feature direction addition to hidden states
- **Coefficient**: 0.5x (effective range)
- **Optimal Range**: 2.0-15.0
- **Feature Dimensions**: 4096 (hidden state size)
- **Steering Vector Shape**: [1, 1, 4096] (broadcasted to [batch, seq_len, hidden_dim])

## 🎯 Usage Tips
1. **Start with moderate steering** (5-10) for noticeable effects
2. **Use financial prompts** for best results
3. **Compare side-by-side** to see steering impact
4. **Check activation metrics** to verify feature engagement
5. **Higher strengths** (10-15) show more dramatic changes
6. **Different features** will have different impacts on the same prompt

---

## 💻 Technical Implementation Details

### Core Steering Code
```python
def steering_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
    
    if strength > 0:
        # 1. Get feature direction from SAE decoder weights
        feature_direction = decoder[feature_idx, :]  # Shape: [hidden_dim]
        feature_direction = feature_direction.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        # 2. Normalize for consistent steering magnitude
        feature_norm = torch.norm(feature_direction)
        if feature_norm > 0:
            feature_direction = feature_direction / feature_norm
        
        # 3. Apply steering: strength * 0.5 * normalized_direction
        steering_vector = strength * 0.5 * feature_direction
        
        # 4. Add directly to hidden states
        steered_hidden = hidden_states + steering_vector
        
        return (steered_hidden.to(hidden_states.dtype),) + output[1:]
    else:
        return output
```

### Why 0.5 Coefficient?
The **0.5 coefficient** is an **empirical tuning parameter** that provides optimal balance between:
- **Effectiveness**: Strong enough to show clear steering effects
- **Stability**: Not so strong that it breaks generation quality  
- **Controllability**: Allows fine-grained control over steering intensity

**Without 0.5**: Raw feature directions can be too strong, causing degraded text quality
**With 0.5**: Provides the "sweet spot" for effective, stable steering

### Comparison with SAELens
Our implementation differs from the [SAELens steering approach](https://github.com/jbloomAus/SAELens/blob/main/tutorials/using_an_sae_as_a_steering_vector.ipynb):

| Aspect | SAELens | Our Implementation |
|--------|---------|-------------------|
| **Normalization** | Often skips normalization | Always normalizes feature direction |
| **Coefficient** | Smaller values (0.1-0.3) | Larger value (0.5) |
| **Steering Method** | Feature activation-based | Direct hidden state addition |
| **Intensity** | More conservative | More aggressive |
| **Stability** | Very stable, subtle effects | Balanced stability and visibility |

**Our 0.5 coefficient** is larger than typical SAELens values because:
1. **We normalize first**, so the raw magnitude is controlled
2. **We want more visible effects** for the interactive UI
3. **Our financial features** are highly specialized and can handle stronger steering
4. **User control** allows fine-tuning the strength slider (0-30)

### Steering Process Flow
```
Input Prompt → Model Forward Pass → Hidden States (Layer 22)
                                           ↓
                                    Steering Hook Applied
                                           ↓
                              Feature Direction (4096D) ← SAE Decoder Weights
                                           ↓
                              Normalize & Scale (strength × 0.5)
                                           ↓
                              Add to Hidden States
                                           ↓
                              Modified Hidden States → Continue Generation
                                           ↓
                              Steered Output Text
```

### Mathematical Formula
```
steering_vector = strength × 0.5 × (decoder[feature_idx, :] / ||decoder[feature_idx, :]||)
steered_hidden = original_hidden + steering_vector
```

Where:
- `strength`: User-defined steering intensity (0-30)
- `decoder[feature_idx, :]`: 4096-dimensional feature direction from SAE
- `||·||`: L2 normalization
- `0.5`: Empirical coefficient for effective steering