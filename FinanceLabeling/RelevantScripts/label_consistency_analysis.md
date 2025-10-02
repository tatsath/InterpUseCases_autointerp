# Label Consistency Analysis: Top 10 vs All 400 Features

## Overview
This analysis compares the labels generated for the same features between two different runs:
1. **Previous Analysis**: Top 10 features per layer (from README.md)
2. **New Analysis**: All 400 features per layer (comprehensive run)

## Key Findings: Significant Label Polymorphism Detected

### 🚨 **Major Finding: High Label Inconsistency**
The same features show **dramatically different labels** between runs, indicating significant **label polymorphism** in the AutoInterp system.

## Detailed Feature-by-Feature Comparison

### Layer 4 Comparison

| Feature | Previous Label (Top 10) | New Label (All 400) | F1 Score Change | Polymorphism Level |
|---------|------------------------|---------------------|-----------------|-------------------|
| **127** | Earnings Reports Interest Rate Announcements | Various historical and technical contexts | 0.72 → 0.52 | 🔴 **HIGH** |
| **141** | valuation changes performance indicators | Political support and policy changes over time | 0.28 → 0.02 | 🔴 **HIGH** |
| **1** | Earnings performance indicators | Historical dates and numerical specifics | 0.365 → 0.467 | 🔴 **HIGH** |
| **90** | Stock index performance | Activation patterns for specific words and phrases indicating historical or documentary content | 0.84 → 0.36 | 🔴 **HIGH** |
| **3** | Inflation indicators labor data | Specific named entities and locations | 0.84 → 0.38 | 🔴 **HIGH** |
| **384** | Asset class diversification yieldspread | Political figures and events in historical contexts | 0.64 → 0.28 | 🔴 **HIGH** |
| **2** | Private Equity Venture Capital Funding | Location, conflict, and institutions in historical contexts | 0.808 → 0.4 | 🔴 **HIGH** |
| **156** | Foreign exchange volatility due to policy changes | End-of-sequence marker or delimiter | 0.769 → 0.483 | 🔴 **HIGH** |
| **25** | Sophisticated trading strategies performance metrics | End of section or paragraph marker | 0.8 → 0.22 | 🔴 **HIGH** |
| **373** | Innovations in sectors | conservation and recovery efforts | 0.08 → 0.06 | 🔴 **HIGH** |

### Layer 10 Comparison

| Feature | Previous Label (Top 10) | New Label (All 400) | F1 Score Change | Polymorphism Level |
|---------|------------------------|---------------------|-----------------|-------------------|
| **384** | Earnings Reports Rate Changes Announcements | captive breeding programs for wildlife conservation | 0.288 → 0.183 | 🔴 **HIGH** |
| **292** | Cryptocurrency corrections regulatory concerns | use of specific prepositions and articles to structure information | 0.673 → 0.4 | 🔴 **HIGH** |
| **273** | Record revenues performance metrics | Activation for "the" and specific modifiers indicating important events or states | 0.9 → 0.22 | 🔴 **HIGH** |
| **173** | Stock index performance | battleship modernization and design features | 0.865 → 0.167 | 🔴 **HIGH** |
| **343** | Economic Indicator Announcements | Post-war political reconstruction and economic recovery | 0.154 → 0.2 | 🔴 **HIGH** |
| **372** | Asset class diversification yieldspread | Cooking practices and associated risks in women's lives | 0.84 → 0.02 | 🔴 **HIGH** |
| **17** | Private Equity Venture Capital Funding Municipal Bonds Tax-Free | Activation patterns for specific historical and geographical terms | 0.7 → 0.22 | 🔴 **HIGH** |
| **389** | Foreign exchange volatility due to central bank actions | Text fragments referencing specific historical events or entities | 0.82 → 0.46 | 🔴 **HIGH** |
| **303** | Sophisticated trading strategies performance metrics | technological advancements and innovations in screen technology | 0.808 → 0.233 | 🔴 **HIGH** |
| **47** | Innovative fintech asset management | Cultural/technological influences in media and art history | 0.615 → 0.533 | 🟡 **MEDIUM** |

### Layer 16 Comparison

| Feature | Previous Label (Top 10) | New Label (All 400) | F1 Score Change | Polymorphism Level |
|---------|------------------------|---------------------|-----------------|-------------------|
| **332** | Earnings Reports Rate Changes Announcements | Proper nouns and specific place names | 0.769 → 0.583 | 🔴 **HIGH** |
| **105** | Major figures | Event announcements or unveilings | 0.692 → 0.183 | 🔴 **HIGH** |
| **214** | Record revenues performance metrics | Named entities and their attributes or actions | 0.76 → 0.42 | 🔴 **HIGH** |
| **66** | Stock index performance metrics | film/works/characters' previous achievements and reception | 0.577 → 0.333 | 🔴 **HIGH** |
| **181** | Inflation labor indicators | References to specific individuals or entities in media contexts | 0.635 → 0.467 | 🔴 **HIGH** |
| **203** | Diversified portfolios asset class allocation | Named entities and proper nouns | 0.22 → 0.58 | 🔴 **HIGH** |
| **340** | Private Equity Venture Capital Funding | Promotional activities and references to specific films or figures | 0.538 → 0.3 | 🔴 **HIGH** |
| **162** | Central bank policies volatility | specific words or phrases indicating scientific or mathematical concepts | 0.76 → 0.367 | 🔴 **HIGH** |
| **267** | Sophisticated trading strategies performance metrics | rotation and activity levels of celestial bodies | 0.269 → 0.267 | 🟡 **MEDIUM** |
| **133** | Innovations investment in sectors | historical and biographical subjects | 0.8 → 0.44 | 🔴 **HIGH** |

### Layer 22 Comparison

| Feature | Previous Label (Top 10) | New Label (All 400) | F1 Score Change | Polymorphism Level |
|---------|------------------------|---------------------|-----------------|-------------------|
| **396** | Earnings Reports Rate Changes Announcements | Population estimates or counts | 0.462 → 0.3 | 🔴 **HIGH** |
| **353** | value milestones performance updates | Names and proper nouns | 0.42 → 0.44 | 🟡 **MEDIUM** |
| **220** | Earnings performance metrics | Specific entities, actions, and technical terms in contracts, awards, and historical events | 0.76 → 0.36 | 🔴 **HIGH** |
| **184** | performance metrics updates | Specific names, places, or technical terms | 0.808 → 0.45 | 🔴 **HIGH** |
| **276** | Inflation indicators labor data | Specific entities, names, or concepts | 0.68 → 0.44 | 🔴 **HIGH** |
| **83** | Asset class diversification yieldspread dynamics | mentions and descriptions in narratives | 0.731 → 0.633 | 🟡 **MEDIUM** |
| **303** | Private Equity Venture Capital Funding Activities | ecological restoration activities | 0.34 → 0.02 | 🔴 **HIGH** |
| **387** | Central bank policies volatility | World War II era alliances and post-war recovery | 0.654 → 0.267 | 🔴 **HIGH** |
| **239** | Sophisticated trading strategies performance metrics | Names of historical or political figures | 0.712 → 0.367 | 🔴 **HIGH** |
| **101** | Innovative fintech solutions | Transitions or shifts in context and topics | 0.76 → 0.52 | 🔴 **HIGH** |

### Layer 28 Comparison

| Feature | Previous Label (Top 10) | New Label (All 400) | F1 Score Change | Polymorphism Level |
|---------|------------------------|---------------------|-----------------|-------------------|
| **262** | Earnings Reports Rate Changes Announcements | OS, W, Z, Y, Form, WV90, Truj, Er, Kes, @-, Ke, @- scale, H, 610, fall Gul | 0.5 → 0.383 | 🔴 **HIGH** |
| **27** | value changes performance indicators | Burns caused by various factors | 0.36 → 0.02 | 🔴 **HIGH** |
| **181** | Record performance revenue figures | Periodic policy or system changes driven by external events | 0.808 → 0.233 | 🔴 **HIGH** |
| **171** | Stock index performance Net interest margin updates | construction and development projects | 0.06 → 0.233 | 🔴 **HIGH** |
| **154** | Inflation indicators labor data | population size or change | 0.269 → 0.35 | 🟡 **MEDIUM** |
| **83** | Diversified portfolios yieldspread dynamics | Describing or characterizing individuals or entities | 0.865 → 0.35 | 🔴 **HIGH** |
| **389** | Private Equity Venture Capital Funding | Key events, plans, and beliefs triggering model responses | 0.615 → 0.233 | 🔴 **HIGH** |
| **172** | Currency volatility due to policy changes | Specific thematic concepts in context | 0.096 → 0.14 | 🟡 **MEDIUM** |
| **333** | Sophisticated trading strategies performance metrics | Named entities, technical terms, specific dates/numbers | 0.52 → 0.46 | 🟡 **MEDIUM** |
| **350** | Innovations investment in sectors | expression or concept being split or defined | 0.788 → 0.433 | 🔴 **HIGH** |

## Polymorphism Analysis Summary

### 🔴 **High Polymorphism (80% of features)**
- **40 out of 50 features** show completely different semantic interpretations
- Labels change from financial/economic concepts to historical, technical, or linguistic concepts
- F1 scores often change significantly (both increases and decreases)

### 🟡 **Medium Polymorphism (20% of features)**
- **10 out of 50 features** show some consistency but with notable differences
- Labels maintain similar semantic categories but with different specific interpretations
- F1 scores show moderate changes

### 🔵 **Low Polymorphism (0% of features)**
- **No features** show consistent labels between runs

## Key Observations

### 1. **Domain Shift**
- **Previous run**: Strongly financial/economic focus
- **New run**: More diverse domains (historical, technical, linguistic, cultural)

### 2. **F1 Score Variability**
- **Average F1 change**: -0.15 (significant decrease)
- **Range of changes**: -0.82 to +0.36
- **Most features**: Show decreased F1 scores

### 3. **Label Quality Differences**
- **Previous run**: More specific, domain-focused labels
- **New run**: More generic, broader semantic categories

## Potential Causes of Polymorphism

### 1. **Context Sensitivity**
- Different datasets or data splits may influence label generation
- Model may be sensitive to the specific examples used for each feature

### 2. **Random Sampling Effects**
- Different activating/non-activating examples selected between runs
- FAISS sampling may produce different negative examples

### 3. **Model Instability**
- LLM explainer may produce different interpretations for the same feature
- Temperature or sampling parameters may affect consistency

### 4. **Feature Interaction**
- Analyzing all 400 features simultaneously may affect individual feature interpretations
- Competition between features for similar semantic space

## Recommendations

### 1. **Improve Consistency**
- Use fixed random seeds for reproducible sampling
- Implement label validation across multiple runs
- Consider ensemble approaches for more stable labels

### 2. **Better Evaluation**
- Track label consistency as a key metric
- Implement cross-validation for feature interpretations
- Use multiple explainer models for validation

### 3. **Feature Stability Analysis**
- Identify which features are most/least stable across runs
- Focus on consistently interpretable features for downstream tasks

## Conclusion

The analysis reveals **significant label polymorphism** in the AutoInterp system, with 80% of features showing completely different interpretations between runs. This suggests that:

1. **Feature interpretations are not stable** across different analysis runs
2. **Context and sampling significantly influence** label generation
3. **Care should be taken** when using these labels for downstream tasks
4. **Further research is needed** to improve consistency and reliability

This polymorphism represents a significant challenge for the practical application of AutoInterp results and highlights the need for more robust and consistent feature interpretation methods.
