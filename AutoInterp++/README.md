# AutoInterp++: Delphi-Free Interpretability Pipeline

Complete end-to-end feature discovery, labeling, and evaluation system for LLM interpretability.

## Key Steps

### 1. **Feature Discovery**
- Rank candidates by coverage × lift
- Select top features for analysis

### 2. **Contrastive Explanation**
- Generate labels using Meta 7B/8B models
- Compare positives vs hard negatives
- Extract include/exclude cues and risks

### 3. **Clustering & Polysemanticity**
- Cluster exemplars using SBERT + HDBSCAN
- Calculate polysemanticity index
- Identify multi-meaning features

### 4. **Thresholding & Scoring**
- Rule-based scorer from LLM cues
- Optional logistic regression scorer
- Selectivity gate for hard negatives

### 5. **Comprehensive Metrics**
- F1, precision, recall, selectivity
- Confusion matrix, PR curve points
- Robustness testing (fuzzing)
- Self-consistency validation

### 6. **Feature Cards**
- JSON artifacts with all metrics
- Audit-ready documentation
- Decision: accept/split/revise

## Quick Start

```python
from auto_interp_pipeline import build_feature_card, MetaLLMExplainer

# Initialize LLM (optional)
llm_explainer = MetaLLMExplainer("meta-llama/Llama-2-7b-hf")

# Build feature card
card, path = build_feature_card(
    base_model_id="llama-2-7b",
    sae_meta={"layer": 4, "l0": 121},
    latent_id=127,
    domain_name="finance",
    positives_train=positives,
    negatives_train=negatives,
    positives_val=val_positives,
    negatives_val=val_negatives,
    bg_texts_for_hard_negs=background_corpus,
    llm_explainer=llm_explainer
)
```

## Dependencies

```bash
pip install sentence-transformers faiss-cpu hdbscan scikit-learn orjson numpy rapidfuzz transformers torch
```

## Output

Each feature generates a comprehensive JSON card with:
- **Labeling**: Label, definition, cues, risks
- **Metrics**: F1, precision, recall, selectivity
- **Clustering**: Polysemanticity analysis
- **Robustness**: Fuzzing test results
- **Decision**: Accept/split/revise recommendation

## Features vs Delphi

✅ **Parity**: F1, selectivity, clustering, robustness, self-consistency  
✅ **Fast**: No external dependencies  
✅ **Flexible**: Works with any LLM provider  
⚠️ **Delphi advantages**: Battle-tested, standardized splits, operational tools
