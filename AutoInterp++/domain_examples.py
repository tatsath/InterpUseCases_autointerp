#!/usr/bin/env python3
"""
Domain-Agnostic AutoInterp++ Examples
Demonstrates the interpretability pipeline across different domains
"""

import os
import json
from auto_interp_pipeline import (
    build_feature_card, 
    process_latent_batch, 
    MetaLLMExplainer,
    set_seed
)

def create_tech_domain_data():
    """Create sample data for technology domain"""
    
    latent_data = {
        42: {  # AI/ML feature
            "positives_train": [
                "The new AI model achieved state-of-the-art performance",
                "Machine learning algorithm improved accuracy by 15%",
                "Deep learning neural network processed the data efficiently",
                "Artificial intelligence system made accurate predictions",
                "The model used advanced algorithms to solve the problem",
                "Neural network architecture optimized for speed and accuracy",
                "AI-powered solution delivered impressive results",
                "Machine learning pipeline processed millions of samples"
            ],
            "negatives_train": [
                "The traditional software worked as expected",
                "Manual data entry took several hours to complete",
                "Basic spreadsheet analysis showed some patterns",
                "Simple database query returned the results",
                "Standard programming approach was sufficient",
                "Conventional method handled the task adequately",
                "Regular software update fixed the issue",
                "Basic computer program ran without problems"
            ],
            "positives_val": [
                "AI model outperformed all previous benchmarks",
                "Machine learning approach revolutionized the process",
                "Neural network achieved breakthrough results"
            ],
            "negatives_val": [
                "Standard software solution worked fine",
                "Basic programming method was adequate"
            ]
        },
        
        73: {  # Sports feature
            "positives_train": [
                "The team won the championship with a spectacular victory",
                "Player scored a hat-trick in the final match",
                "Olympic athlete broke the world record",
                "Team achieved a perfect season with no losses",
                "Championship game ended with a dramatic overtime win",
                "Athlete won gold medal in the competition",
                "Team secured playoff berth with impressive performance",
                "Player set new league record for goals scored"
            ],
            "negatives_train": [
                "The team practiced for two hours today",
                "Player attended training session regularly",
                "Athlete maintained fitness routine throughout season",
                "Team prepared for upcoming match",
                "Player worked on improving basic skills",
                "Athlete followed standard training protocol",
                "Team completed regular practice drills",
                "Player focused on fundamental techniques"
            ],
            "positives_val": [
                "Championship victory was celebrated by thousands",
                "Olympic gold medalist made history",
                "Team's perfect season will be remembered forever"
            ],
            "negatives_val": [
                "Regular practice session went as planned",
                "Standard training routine was completed"
            ]
        }
    }
    
    # Background corpus for different domains
    background_corpus = [
        # Technology
        "Software development requires careful planning and testing",
        "Computer programming involves writing code and debugging",
        "Technology companies invest heavily in research and development",
        "Digital transformation is changing how businesses operate",
        "Cloud computing provides scalable infrastructure solutions",
        
        # Sports
        "Athletes train hard to improve their performance",
        "Sports teams compete in various leagues and tournaments",
        "Physical fitness is important for athletic success",
        "Coaching staff helps players develop their skills",
        "Sports fans support their favorite teams passionately",
        
        # General
        "Education plays a crucial role in personal development",
        "Healthcare professionals work to improve patient outcomes",
        "Environmental protection requires collective action",
        "Art and culture enrich our daily lives",
        "Science and research drive human progress"
    ]
    
    return latent_data, background_corpus

def create_medical_domain_data():
    """Create sample data for medical domain"""
    
    latent_data = {
        156: {  # Medical diagnosis feature
            "positives_train": [
                "Patient diagnosed with acute myocardial infarction",
                "CT scan revealed malignant tumor in the lung",
                "Blood test confirmed diabetes mellitus type 2",
                "MRI showed evidence of multiple sclerosis",
                "Biopsy confirmed presence of cancerous cells",
                "ECG indicated atrial fibrillation",
                "X-ray revealed fractured tibia",
                "Ultrasound detected gallstones in the bladder"
            ],
            "negatives_train": [
                "Patient reported feeling generally well",
                "Routine checkup showed normal vital signs",
                "Annual physical examination was unremarkable",
                "Patient maintained healthy lifestyle habits",
                "Regular exercise routine kept patient fit",
                "Balanced diet contributed to good health",
                "Patient had no significant medical history",
                "Preventive care measures were effective"
            ],
            "positives_val": [
                "Emergency diagnosis saved patient's life",
                "Critical condition required immediate intervention",
                "Severe symptoms indicated serious illness"
            ],
            "negatives_val": [
                "Routine screening showed no abnormalities",
                "Preventive care maintained good health"
            ]
        }
    }
    
    background_corpus = [
        "Medical professionals provide essential healthcare services",
        "Patient care requires compassion and expertise",
        "Healthcare systems aim to improve population health",
        "Medical research advances treatment options",
        "Preventive medicine helps avoid serious conditions",
        "Emergency medicine saves lives in critical situations",
        "Public health initiatives promote community wellness",
        "Medical technology enhances diagnostic capabilities"
    ]
    
    return latent_data, background_corpus

def run_domain_examples():
    """Run examples across different domains"""
    
    print("🌍 Domain-Agnostic AutoInterp++ Examples")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize LLM explainer
    print("🤖 Initializing LLM explainer...")
    explainer = MetaLLMExplainer()
    
    # Test different domains
    domains = [
        ("Technology", create_tech_domain_data),
        ("Medical", create_medical_domain_data)
    ]
    
    all_results = []
    
    for domain_name, data_func in domains:
        print(f"\n📊 Processing {domain_name} Domain...")
        print("-" * 40)
        
        latent_data, background_corpus = data_func()
        
        # Process each latent feature
        for latent_id, data in latent_data.items():
            print(f"🔍 Processing latent {latent_id}...")
            
            try:
                # Build feature card using process_latent_batch
                results = process_latent_batch(
                    latent_data={latent_id: data},
                    base_model_id="llama-2-7b",
                    sae_meta={"layer": 4, "l0": 121, "sha": "abc123def456"},
                    domain_name=domain_name.lower(),
                    bg_corpus=background_corpus,
                    outdir="feature_cards",
                    llm_explainer=explainer,
                    min_selectivity_gate=0.7,
                    min_f1_gate=0.6
                )
                
                if results:
                    card, path = results[0]
                else:
                    card = None
                
                if card:
                    all_results.append((domain_name, latent_id, card))
                    print(f"✅ Processed latent {latent_id}: {card['labeling']['label']}")
                else:
                    print(f"❌ Failed to process latent {latent_id}")
                    
            except Exception as e:
                print(f"❌ Error processing latent {latent_id}: {str(e)}")
    
    # Display results summary
    print(f"\n📋 Results Summary")
    print("=" * 60)
    print(f"Processed {len(all_results)} features across {len(domains)} domains")
    
    for domain, latent_id, card in all_results:
        print(f"\n{domain} - Latent {latent_id}: {card['labeling']['label']}")
        print(f"  Definition: {card['labeling']['definition']}")
        print(f"  F1 Score: {card['metrics']['f1']:.3f}")
        print(f"  Precision: {card['metrics']['precision']:.3f}")
        print(f"  Recall: {card['metrics']['recall']:.3f}")
        print(f"  Decision: {card['decision']}")
        print(f"  Include Cues: {', '.join(card['labeling']['include_cues'][:5])}...")
        print(f"  Exclude Cues: {', '.join(card['labeling']['exclude_cues'][:3])}...")
    
    print(f"\n✅ Domain examples completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_domain_examples()
