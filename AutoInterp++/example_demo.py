#!/usr/bin/env python3
"""
AutoInterp++ Demo Script
Demonstrates the complete interpretability pipeline with sample data
"""

import os
import json
from auto_interp_pipeline import (
    build_feature_card, 
    process_latent_batch, 
    MetaLLMExplainer,
    set_seed
)

def create_sample_data():
    """Create sample data for demonstration"""
    
    # Sample latent features with financial domain data
    latent_data = {
        127: {  # Earnings feature
            "positives_train": [
                "The company reported strong quarterly earnings growth of 25%",
                "Revenue increased by 15% year-over-year to $2.5 billion",
                "Profit margins expanded significantly in Q3",
                "Financial performance exceeded analyst expectations",
                "Earnings per share beat estimates by $0.15",
                "Operating income improved by 18%",
                "The firm posted record quarterly profits",
                "Revenue growth accelerated to 12%",
                "Net income rose substantially",
                "Financial results were outstanding"
            ],
            "negatives_train": [
                "The stock price declined 5% today",
                "Market volatility increased significantly",
                "Trading volume was unusually low",
                "Technical indicators show weakness",
                "The market closed lower",
                "Investor sentiment turned negative",
                "Share prices fell across the board",
                "Trading was light today",
                "Market conditions were challenging",
                "The sector underperformed"
            ],
            "positives_val": [
                "Earnings per share beat estimates by $0.20",
                "Revenue growth accelerated to 15%",
                "Operating income improved significantly",
                "Quarterly profits reached new highs",
                "Financial metrics exceeded targets"
            ],
            "negatives_val": [
                "Share price fell sharply in after-hours trading",
                "Market sentiment turned negative",
                "Trading volume dried up",
                "Technical analysis shows bearish signals",
                "The stock underperformed the market"
            ]
        },
        
        141: {  # Valuation feature
            "positives_train": [
                "The company's valuation increased by 30%",
                "Price-to-earnings ratio expanded significantly",
                "Market cap reached $50 billion",
                "Valuation metrics improved substantially",
                "The stock is trading at premium multiples",
                "Enterprise value increased to $45 billion",
                "Valuation growth accelerated",
                "Price targets were raised by analysts",
                "The company's worth increased dramatically",
                "Market valuation exceeded expectations"
            ],
            "negatives_train": [
                "The company reported a loss this quarter",
                "Revenue declined by 10% year-over-year",
                "Operating expenses increased significantly",
                "The business faced headwinds",
                "Cost structure deteriorated",
                "Margins compressed due to inflation",
                "The company struggled with profitability",
                "Financial performance was disappointing",
                "The firm missed revenue targets",
                "Earnings were below expectations"
            ],
            "positives_val": [
                "Valuation metrics improved across all measures",
                "Price-to-sales ratio expanded",
                "Market cap growth accelerated",
                "Valuation reached new highs",
                "Enterprise value increased substantially"
            ],
            "negatives_val": [
                "Valuation metrics deteriorated",
                "Price multiples compressed",
                "Market cap declined significantly",
                "Valuation growth stalled",
                "Enterprise value decreased"
            ]
        }
    }
    
    # Background corpus for hard negatives
    bg_corpus = [
        "The weather is beautiful today with clear skies",
        "I went to the grocery store to buy food",
        "The movie was entertaining and well-directed",
        "Technology advances rapidly in the modern world",
        "Education is important for personal development",
        "The food at the restaurant was delicious",
        "Music can be very relaxing and therapeutic",
        "Sports require dedication and practice",
        "The book was interesting and informative",
        "Travel broadens one's perspective on life",
        "Art can express complex emotions and ideas",
        "Science helps us understand the world",
        "Friendship is valuable and meaningful",
        "Exercise is good for physical health",
        "Reading improves vocabulary and knowledge",
        "Cooking can be creative and enjoyable",
        "Nature provides peace and tranquility",
        "Learning new skills is rewarding",
        "Hobbies provide relaxation and fulfillment",
        "Community service helps others in need"
    ]
    
    return latent_data, bg_corpus

def run_demo():
    """Run the complete AutoInterp++ demo"""
    
    print("🚀 AutoInterp++ Demo Starting...")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(1337)
    
    # Create sample data
    print("📊 Creating sample data...")
    latent_data, bg_corpus = create_sample_data()
    print(f"   Created {len(latent_data)} latent features")
    print(f"   Background corpus: {len(bg_corpus)} texts")
    
    # Initialize LLM explainer (optional)
    print("\n🤖 Initializing LLM explainer...")
    llm_explainer = None
    try:
        # Try to load Meta model
        llm_explainer = MetaLLMExplainer("meta-llama/Llama-2-7b-hf")
        print("   ✅ Meta LLM loaded successfully")
    except Exception as e:
        print(f"   ⚠️  Could not load Meta model: {str(e)}")
        print("   Using fallback explainer")
    
    # Process the batch
    print("\n🔍 Processing latent features...")
    results = process_latent_batch(
        latent_data=latent_data,
        base_model_id="llama-2-7b",
        sae_meta={"layer": 4, "l0": 121, "sha": "abc123def456"},
        domain_name="finance",
        bg_corpus=bg_corpus,
        outdir="feature_cards",
        llm_explainer=llm_explainer,
        min_selectivity_gate=0.7,
        min_f1_gate=0.6
    )
    
    # Display results
    print(f"\n📋 Results Summary")
    print("=" * 60)
    print(f"Processed {len(results)} features successfully")
    
    for i, (card, path) in enumerate(results, 1):
        print(f"\n{i}. Feature: {card['labeling']['label']}")
        print(f"   Definition: {card['labeling']['definition']}")
        print(f"   F1 Score: {card['metrics']['f1']:.3f}")
        print(f"   Precision: {card['metrics']['precision']:.3f}")
        print(f"   Recall: {card['metrics']['recall']:.3f}")
        print(f"   Selectivity: {card['metrics']['selectivity']:.3f}")
        print(f"   Polysemanticity: {card['clustering']['polysemanticity_index']:.3f}")
        print(f"   Decision: {card['decision']}")
        print(f"   Include Cues: {', '.join(card['labeling']['include_cues'][:5])}...")
        print(f"   Exclude Cues: {', '.join(card['labeling']['exclude_cues'][:3])}...")
        print(f"   Saved to: {path}")
    
    # Show feature cards directory
    print(f"\n📁 Feature Cards saved to: feature_cards/")
    if os.path.exists("feature_cards"):
        files = os.listdir("feature_cards")
        print(f"   Generated {len(files)} JSON files")
        for file in files:
            print(f"   - {file}")
    
    print("\n✅ Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
