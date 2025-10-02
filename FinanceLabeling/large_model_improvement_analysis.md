# Large Model Improvement Analysis: 72B vs Previous Analysis

## Overview
Comparison of Layer 4 top 10 features between previous analysis (all 400 features) and large model analysis (Qwen 2.5-72B-Instruct).

## Results Summary

| Feature | Previous F1 | Large Model F1 | Improvement | Previous Label | Large Model Label |
|---------|-------------|----------------|-------------|----------------|-------------------|
| **373** | 0.060 | 0.780 | +0.720 | conservation and recovery efforts | Complex, multi-part words and technical terms related to specific contexts or domains |
| **141** | 0.020 | 0.680 | +0.660 | Political support and policy changes over time | Mixed content with numbers, dates, and proper nouns in various contexts |
| **384** | 0.280 | 0.580 | +0.300 | Political figures and events in historical contexts | Names, dates, and specific entities in historical and institutional contexts |
| **127** | 0.520 | 0.800 | +0.280 | Various historical and technical contexts | Numerical and structural elements in text, including dates, numbers, and punctuation |
| **2** | 0.400 | 0.554 | +0.154 | Location, conflict, and institutions in historical contexts | Specific institutional, organizational, and geographical entities |
| **90** | 0.360 | 0.492 | +0.132 | Activation patterns for specific words and phrases indicating historical or documentary content | Contextual markers in text, such as dates, locations, and specific terms |
| **25** | 0.220 | 0.300 | +0.080 | End of section or paragraph marker | Markers of sentence structure and punctuation in text |
| **3** | 0.380 | 0.420 | +0.040 | Specific named entities and locations | Dates, numbers, and specific events in historical and cultural contexts |
| **156** | 0.483 | 0.431 | -0.052 | End-of-sequence marker or delimiter | HTML tags and section markers in text |
| **1** | 0.467 | 0.385 | -0.082 | Historical dates and numerical specifics | Highlighting digits, especially zeros, in numerical sequences |

## Summary Statistics
- **Features Improved**: 8/10 (80%)
- **Average F1 Improvement**: +0.223
- **Best Improvement**: +0.720 (Feature 373)
- **Worst Change**: -0.082 (Feature 1)