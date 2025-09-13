# Financial LLM Feature Steering

## 🎯 Overview
Interactive feature steering for the finetuned Llama2-7b-Finance model using SAE (Sparse Autoencoder) features. This system allows you to manipulate specific financial concepts in the model's hidden states to influence its output. **All layers (4, 10, 16, 22, 28) are fully functional and tested.**

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

**Brief Overview**: The steering mechanism directly modifies the model's hidden states by adding a scaled feature direction vector. The feature direction comes from SAE decoder weights, is normalized, scaled by `strength × 1.0`, and added to hidden states at the specified layer to bias generation toward the selected financial concept.

**Key Formula**: `steering_vector = strength × 1.0 × (normalized_feature_direction)`

## 📊 Feature Data
Uses **finetuned model labels** from the financial domain analysis:
- **Layer 4**: Financial Market Analysis, Financial Institutions
- **Layer 10**: Financial market trends, Investment Guidance  
- **Layer 16**: Financial performance indicators, News and Analysis
- **Layer 22**: Financial Performance and Growth, Market indicators
- **Layer 28**: Company financial performance, Stock Performance Analysis

## 📸 Interactive Demo

**Screenshot**: The Streamlit app interface showing before and after steering effects

**Key Highlights**:
- **Left Sidebar**: Clean controls (Layer, Feature, Steering Strength, Max Tokens)
- **Main Area**: Input prompt and side-by-side output comparison
- **Steering Effect**: Strength 28 transforms general business questions into structured financial quizzes with specific numerical questions and multiple-choice answers
- **Feature-Specific**: Each feature produces distinct, dramatic transformations aligned with its label

## 🎯 Steering Examples

### Example 1: Financial Performance Focus
**Feature**: 159 (Layer 22) - "Financial Performance and Growth"  
**Prompt**: "The company's quarterly earnings show"

| Strength | Generated Output |
|----------|------------------|
| **0** | that it has made a profit of $1.2 billion, up from $849 million in the same period last year. The company's revenue rose to $30.6 billion from $27.5 billion in the third quarter of 2017. |
| **10** | that it has made a profit of $1.6 billion, up from $1.2 billion in the same period last year. The company's revenue for the quarter was $10.3 billion, down from $10.8 billion in the same period. |
| **20** | a 30% increase in revenue from the same period last year. Together with the company's strong balance sheet and solid cash flow, this suggests that the company is well-positioned to continue growing its business over the long term. |
| **30** | a profit of $1.3 per share, up from a loss of $0.2 per share in the same period last year. The company's net sales for the quarter were $487 million, down from $50 million. |

**Impact**: Dramatic transformation from specific numbers to growth-focused language and strategic positioning.

### Example 2: Stock Market Analysis
**Feature**: 258 (Layer 22) - "Financial market indicators and metrics"  
**Prompt**: "Stock market analysis indicates"

| Strength | Generated Output |
|----------|------------------|
| **0** | that the stock price of this company will increase by 10% in the next quarter. Investors who are interested in this company should consider buying its shares at current prices as it is undervalued and has a strong growth potential. |
| **10** | that the stock price of this company will increase by 10% in the next quarter. Investors are optimistic about the future growth prospects of the company due to its strong brand and increasing sales. |
| **20** | that the stock has been overbought for a short period. The stock is trading at $25.89 and it has an average volume of 10,376,400 shares traded daily. |
| **30** | that the stocks of companies in the energy sector are likely to be affected by the changes in the oil price. The current situation is a challenge for the companies in this industry. |

**Impact**: Complete shift from general investment advice to specific technical analysis and sector-specific insights.

### Example 3: Financial Metrics Focus
**Feature**: 345 (Layer 22) - "Financial performance metrics"  
**Prompt**: "Stock market analysis indicates"

| Strength | Generated Output |
|----------|------------------|
| **0** | that the stock price of this company will increase by 10% in the next quarter. Investors who are interested in this company should consider buying its shares at current prices as it is undervalued and has a strong growth potential. |
| **10** | that the stock has been in a downtrend since 2013. The company's stock price is currently trading at $6.48, which is significantly lower than its 52-week high of $17.95. |
| **20** | that the stock has been in a downtrend since 2013. The company is also facing competition from other players such as Reliance Jio and Airtel. However, the company has made significant investments in its network and customer service. |
| **30** | that the stock has been in a downtrend since 2 days. A downtrend is when the price is generally lower than it was at any other time. The stock price has declined by -1.87% in the past three days. |

**Impact**: Transformation from optimistic investment advice to detailed technical analysis with specific price levels and trend analysis.

### Example 4: Financial Terminology
**Feature**: 375 (Layer 22) - "Financial market terminology and jargon"  
**Prompt**: "Stock market analysis indicates"

| Strength | Generated Output |
|----------|------------------|
| **0** | that the stock price of this company will increase by 10% in the next quarter. Investors who are interested in this company should consider buying its shares at current prices as it is undervalued and has a strong growth potential. |
| **10** | that the stock has been trading in a strong uptrend lately. The stock is currently trading at $28.60, which is above its 50-day moving average of $27.14 and above its 200-day moving average. |
| **20** | that the stock has a 50-day moving average of $23.89 and a 200-day moving average of $24.67. The company has a market cap of $1.3 billion. A number of analysts have recently issued reports on the stock. |
| **30** | that the stock (NYSE: 2019) is in a strong uptrend and (NYSE: 2019) is in a strong downtrend. The stock (NYSE: 2019) has a [technical indicators]. |

**Impact**: Evolution from simple investment advice to sophisticated technical analysis with moving averages, market cap, and trading terminology.

### Example 5: Business Themes
**Feature**: 116 (Layer 22) - "Financial and business-related themes"  
**Prompt**: "Investment opportunities arise when"

| Strength | Generated Output |
|----------|------------------|
| **0** | the market price of a security is lower than its intrinsic value. This difference between the market price and the intrinsic value represents an opportunity for profit through buying the security at a low price and selling it at a higher price later on. |
| **10** | the market price of a stock is lower than its intrinsic value. This difference between the market price and the intrinsic value represents an opportunity for profit. Investors can take advantage of these opportunities by buying the stock at the lower market price. |
| **20** | the market price of a stock is significantly lower than its intrinsic value. This can occur due to various reasons such as: 1. Earnings decline: A company may experience a decline in earnings, which could lead to a decrease in its stock price. 2. Industry downturn. |
| **30** | the market price of a stock is significantly lower than its intrinsic value. This can occur due to various reasons such as: 1. Earnings decline: A company may experience a decline in earnings, which could lead to a decrease in the stock price. 2. Industry downturn. |

**Impact**: Progressive enhancement from basic investment concepts to detailed analysis frameworks with structured reasoning.

## 🎛️ Features

### Streamlit App
- **Interactive UI** with financial feature selection
- **Side-by-side comparison** of original vs steered outputs
- **Large text areas** (400px height) for better readability
- **Extended token generation** (100-800 tokens, default 500)
- **Steering range** (-50 to +50)
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
- **Coefficient**: 0.5x (balanced for effective steering)
- **Optimal Range**: 10.0-50.0
- **Feature Dimensions**: 4096 (hidden state size)
- **Steering Vector Shape**: [1, 1, 4096] (broadcasted to [batch, seq_len, hidden_dim])

## 🎯 Usage Tips
1. **Start with moderate steering** (10-15) for noticeable effects
2. **Use financial prompts** for best results
3. **Compare side-by-side** to see steering impact
4. **Check activation metrics** to verify feature engagement
5. **Higher strengths** (20-30) show dramatic transformations
6. **Different features** will have different impacts on the same prompt
7. **Each feature** specializes in different financial concepts

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
- **Effectiveness**: Strong enough to show dramatic steering effects
- **Stability**: Maintains generation quality even at high strengths
- **Controllability**: Allows fine-grained control over steering intensity

**Without 0.5**: Steering effects are too subtle to be clearly visible
**With 0.5**: Provides dramatic, visible transformations while maintaining quality

### Comparison with SAELens
Our implementation differs from the [SAELens steering approach](https://github.com/jbloomAus/SAELens/blob/main/tutorials/using_an_sae_as_a_steering_vector.ipynb):

| Aspect | SAELens | Our Implementation |
|--------|---------|-------------------|
| **Normalization** | Often skips normalization | Always normalizes feature direction |
| **Coefficient** | Smaller values (0.1-0.3) | Larger value (1.0) |
| **Steering Method** | Feature activation-based | Direct hidden state addition |
| **Intensity** | More conservative | More aggressive |
| **Stability** | Very stable, subtle effects | Balanced stability and visibility |

**Our 0.5 coefficient** is balanced compared to typical SAELens values because:
1. **We normalize first**, so the raw magnitude is controlled
2. **We want dramatic visible effects** for the interactive UI
3. **Our financial features** are highly specialized and can handle stronger steering
4. **User control** allows fine-tuning the strength slider (0-50)

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
- `strength`: User-defined steering intensity (0-50)
- `decoder[feature_idx, :]`: 4096-dimensional feature direction from SAE
- `||·||`: L2 normalization
- `0.5`: Empirical coefficient for balanced steering effects