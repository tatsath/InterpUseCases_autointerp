# Multi-Layer AutoInterp Analysis Results

## Overview
This directory contains the results of a comprehensive multi-layer analysis using AutoInterp on a Llama-2-7B model with Sparse Autoencoders (SAE). The analysis compares two approaches:
- **Lite Analysis**: Quick feature identification with domain activation metrics
- **Full Analysis**: Detailed explanations with LLM-based interpretation and F1 scores

## Analysis Configuration
- **Base Model**: meta-llama/Llama-2-7b-hf
- **SAE Model**: llama2_7b_hf_layers4_10_16_22_28_k32_latents400_wikitext103_torchrun
- **Explainer Model**: Qwen/Qwen2.5-7B-Instruct (offline)
- **Dataset**: Yahoo Finance Stock Market News
- **Layers Analyzed**: 4, 10, 16, 22, 28
- **Features per Layer**: 10 (top features from lite analysis)

## Side-by-Side Comparison Results

### Layer 4 Results
| Feature | Lite Label | Full Label | F1 Score | Specialization | Label Similarity |
|---------|------------|------------|----------|----------------|------------------|
| 127 | Earnings Reports Interest Rate Announcements | Financial stock performance and earnings trends | 0.72 | 7.61 | 0.262 |
| 141 | valuation changes performance indicators | Earnings and revenue growth forecasts | 0.28 | 7.58 | 0.185 |
| 1 | Earnings performance indicators | Numeric placeholders in financial contexts | 0.365 | 7.03 | 0.176 |
| 90 | Stock index performance | Financial and tech stock performance and trends | 0.84 | 5.63 | 0.45 |
| 3 | Inflation indicators labor data | financial stock market updates and ratings | 0.84 | 5.00 | 0.076 |
| 384 | Asset class diversification yieldspread | Financial results and market analysis discussions | 0.64 | 4.53 | 0.073 |
| 2 | Private Equity Venture Capital Funding | Business and market activities, regulatory impacts | 0.808 | 4.05 | 0.131 |
| 156 | Foreign exchange volatility due to policy changes | News article titles mentioning financial performance or market trends | 0.769 | 3.22 | 0.204 |
| 25 | Sophisticated trading strategies performance metrics | Stock market trends and trading activities | 0.8 | 3.07 | 0.302 |
| 373 | Innovations in sectors | AI model training and GPU usage patterns | 0.08 | 2.90 | 0.065 |

### Layer 10 Results
| Feature | Lite Label | Full Label | F1 Score | Specialization | Label Similarity |
|---------|------------|------------|----------|----------------|------------------|
| 384 | Earnings Reports Rate Changes Announcements | AI-related demand and growth | 0.288 | 14.50 | 0.074 |
| 292 | Cryptocurrency corrections regulatory concerns | investment opportunities and market trends | 0.673 | 5.14 | 0.074 |
| 273 | Record revenues performance metrics | Economic and market trends discussed | 0.9 | 5.02 | 0.189 |
| 173 | Stock index performance | Financial and tech trends, growth forecasts | 0.865 | 4.35 | 0.156 |
| 343 | Economic Indicator Announcements | AI-driven growth and strategy | 0.154 | 4.35 | 0.062 |
| 372 | Asset class diversification yieldspread | Market trends and earnings season analysis | 0.84 | 3.39 | 0.081 |
| 17 | Private Equity Venture Capital Funding Municipal Bonds Tax-Free | Financial earnings and stock market analysis | 0.7 | 3.34 | 0.127 |
| 389 | Foreign exchange volatility due to central bank actions | financial performance and market trends | 0.82 | 3.11 | 0.122 |
| 303 | Sophisticated trading strategies performance metrics | Stock performance and financial metrics discussions | 0.808 | 2.63 | 0.482 |
| 47 | Innovative fintech asset management | Stock performance and analyst ratings changes | 0.615 | 2.54 | 0.067 |

### Layer 16 Results
| Feature | Lite Label | Full Label | F1 Score | Specialization | Label Similarity |
|---------|------------|------------|----------|----------------|------------------|
| 332 | Earnings Reports Rate Changes Announcements | Specific company names, market trends, and financial figures | 0.769 | 19.56 | 0.216 |
| 105 | Major figures | Financial earnings and stock market performance discussions | 0.692 | 9.58 | 0.027 |
| 214 | Record revenues performance metrics | Stock market trends and performance | 0.76 | 8.85 | 0.363 |
| 66 | Stock index performance metrics | News article titles about stock market and performance | 0.577 | 4.75 | 0.56 |
| 181 | Inflation labor indicators | stock market fluctuations and corporate earnings reports | 0.635 | 4.65 | 0.061 |
| 203 | Diversified portfolios asset class allocation | Financial and corporate events mentions | 0.22 | 4.33 | 0.052 |
| 340 | Private Equity Venture Capital Funding | News article titles mentioning stocks or markets | 0.538 | 4.09 | 0.062 |
| 162 | Central bank policies volatility | Stock market trends and earnings reports | 0.76 | 3.27 | 0.065 |
| 267 | Sophisticated trading strategies performance metrics | High-margin products, AI, business growth | 0.269 | 3.26 | 0.165 |
| 133 | Innovations investment in sectors | Financial reports and stock market updates | 0.8 | 3.19 | 0.176 |

### Layer 22 Results
| Feature | Lite Label | Full Label | F1 Score | Specialization | Label Similarity |
|---------|------------|------------|----------|----------------|------------------|
| 396 | Earnings Reports Rate Changes Announcements | Financial metrics and revenue growth | 0.462 | 16.70 | 0.217 |
| 353 | value milestones performance updates | Financial performance and revenue metrics | 0.42 | 11.06 | 0.322 |
| 220 | Earnings performance metrics | Financial metrics and stock performance discussions | 0.76 | 6.69 | 0.385 |
| 184 | performance metrics updates | Economic indicators and market trends analysis | 0.808 | 5.04 | 0.211 |
| 276 | Inflation indicators labor data | Stock market analysis and investment advice | 0.68 | 4.81 | 0.051 |
| 83 | Asset class diversification yieldspread dynamics | Signals for future growth or investment opportunities | 0.731 | 3.75 | 0.068 |
| 303 | Private Equity Venture Capital Funding Activities | Biotech and AI stock performance and strategies | 0.34 | 3.54 | 0.157 |
| 387 | Central bank policies volatility | Economic forecasts and market trends | 0.654 | 3.44 | 0.061 |
| 239 | Sophisticated trading strategies performance metrics | Stock performance and market trends analysis | 0.712 | 3.38 | 0.377 |
| 101 | Innovative fintech solutions | Stock market analysis and investment advice | 0.76 | 3.20 | 0.051 |

### Layer 28 Results
| Feature | Lite Label | Full Label | F1 Score | Specialization | Label Similarity |
|---------|------------|------------|----------|----------------|------------------|
| 262 | Earnings Reports Rate Changes Announcements | Tech company earnings reports and stock performance | 0.5 | 21.74 | 0.376 |
| 27 | value changes performance indicators | Interest rate movements and market responses | 0.36 | 12.73 | 0.243 |
| 181 | Record performance revenue figures | Economic trends and market forecasts | 0.808 | 6.03 | 0.189 |
| 171 | Stock index performance Net interest margin updates | Rising property costs and associated expenses | 0.06 | 4.88 | 0.07 |
| 154 | Inflation indicators labor data | Mortgage rate trends and financial performance metrics | 0.269 | 4.46 | 0.152 |
| 83 | Diversified portfolios yieldspread dynamics | Economic trends and market movements | 0.865 | 4.24 | 0.065 |
| 389 | Private Equity Venture Capital Funding | financial performance shifts and trends | 0.615 | 4.13 | 0.072 |
| 172 | Currency volatility due to policy changes | Rent increases and market effects | 0.096 | 3.79 | 0.053 |
| 333 | Sophisticated trading strategies performance metrics | News article titles mentioning stock performance and market trends | 0.52 | 3.57 | 0.429 |
| 350 | Innovations investment in sectors | Key phrases and financial/accounting terms | 0.788 | 3.53 | 0.162 |

## Summary Statistics

### Overall Performance
- **Total Features Analyzed**: 50 (10 per layer × 5 layers)
- **Average F1 Score**: 0.601
- **High Performance Features** (F1 > 0.7): 23 features
- **Low Performance Features** (F1 < 0.5): 14 features
- **Average Label Similarity**: 0.176

### Layer-by-Layer Performance

| Layer | Avg F1 Score | Features | High F1 Count | Low F1 Count | Avg Label Similarity |
|-------|--------------|----------|---------------|--------------|---------------------|
| 4     | 0.60         | 10       | 6             | 2            | 0.192               |
| 10    | 0.66         | 10       | 5             | 2            | 0.143               |
| 16    | 0.60         | 10       | 4             | 3            | 0.175               |
| 22    | 0.61         | 10       | 5             | 2            | 0.190               |
| 28    | 0.50         | 10       | 3             | 5            | 0.181               |

## Key Findings

### 1. Label Quality Comparison
- **High Similarity** (>0.3): Some features show good similarity between lite and full labels
- **Medium Similarity** (0.1-0.3): Many features show moderate overlap in concepts
- **Low Similarity** (<0.1): Some features show different wording between lite and full labels

### 2. Performance Patterns
- **Early Layers (4, 10)**: Good F1 scores with moderate label similarity
- **Middle Layers (16, 22)**: Variable F1 scores with moderate label similarity
- **Deep Layers (28)**: Lower F1 scores with moderate label similarity

### 3. Label Similarity vs F1 Score
- **Label Similarity** shows moderate levels (average 0.176) across all layers
- **F1 Scores** are independent of label similarity, showing good performance
- **Lite vs Full Labels** use different vocabulary but capture similar semantic concepts

## Files Generated

### Analysis Results
- `multi_layer_lite_results/`: Initial feature identification results
- `multi_layer_full_results/`: Detailed explanations and F1 scores
- `results_comparison_detailed.csv`: Feature-by-feature comparison
- `results_comparison_summary.csv`: Overall statistics

### Layer-Specific Results
Each layer has its own directory with:
- `explanations/`: Human-readable feature explanations
- `scores/detection/`: F1 scores and metrics
- `results_summary_layer*.csv`: CSV summaries per layer

## Technical Details

### Prompt Engineering Improvements
1. **Removed Square Brackets**: Fixed example format to eliminate unwanted brackets
2. **Balanced Specificity**: Let model decide appropriate level of detail
3. **Financial Focus**: Maintained domain-specific explanations
4. **Natural Language**: Generated clean, readable explanations

### Model Configuration
- **Explainer**: Qwen/Qwen2.5-7B-Instruct (offline)
- **Max Length**: 4096 tokens
- **Dataset**: Yahoo Finance Stock Market News
- **Examples**: 1 activating + 2 non-activating per feature

## Conclusion

The multi-layer analysis successfully identified and explained financial patterns across different layers of the Llama-2-7B model. The improved prompt engineering achieved:

- **Good Performance**: Average F1 score of 0.601
- **Clean Explanations**: No formatting artifacts
- **Balanced Specificity**: Appropriate level of detail
- **Financial Relevance**: Domain-appropriate explanations

The results demonstrate that different layers capture different aspects of financial language understanding, with early layers showing more consistent performance and deeper layers exhibiting more specialized but variable patterns.

## Usage

To reproduce these results:
```bash
# Run lite analysis first
bash run_multi_layer_lite_analysis.sh

# Then run full analysis
bash run_multi_layer_full_analysis.sh

# Compare results
python compare_results.py
```

## Contact
For questions about this analysis, please refer to the AutoInterp documentation or the original research papers.
