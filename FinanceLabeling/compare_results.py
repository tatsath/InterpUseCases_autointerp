#!/usr/bin/env python3
"""
Comparison script for multi-layer lite vs full results
Generates a comprehensive comparison showing feature numbers, labels, similarity scores, and F1 scores
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path

def calculate_label_similarity(lite_label, full_label):
    """Calculate similarity between two labels using improved semantic approaches"""
    if pd.isna(lite_label) or pd.isna(full_label):
        return 0.0
    
    # Convert to lowercase and clean
    lite_clean = str(lite_label).lower().strip()
    full_clean = str(full_label).lower().strip()
    
    if not lite_clean or not full_clean:
        return 0.0
    
    # Remove common stop words and punctuation
    import re
    lite_clean = re.sub(r'[^\w\s]', ' ', lite_clean)
    full_clean = re.sub(r'[^\w\s]', ' ', full_clean)
    
    lite_words = set(lite_clean.split())
    full_words = set(full_clean.split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
    lite_words = lite_words - stop_words
    full_words = full_words - stop_words
    
    if not lite_words or not full_words:
        return 0.0
    
    # Method 1: Enhanced Jaccard similarity with stemming
    def simple_stem(word):
        """Simple stemming to group related words"""
        if len(word) <= 3:
            return word
        if word.endswith('ing'):
            return word[:-3]
        if word.endswith('ed'):
            return word[:-2]
        if word.endswith('s') and len(word) > 4:
            return word[:-1]
        if word.endswith('ly'):
            return word[:-2]
        return word
    
    lite_stemmed = {simple_stem(word) for word in lite_words}
    full_stemmed = {simple_stem(word) for word in full_words}
    
    intersection = lite_stemmed.intersection(full_stemmed)
    union = lite_stemmed.union(full_stemmed)
    jaccard_sim = len(intersection) / len(union) if union else 0.0
    
    # Method 2: Semantic word groups (synonyms and related terms)
    semantic_groups = {
        'financial': {'financial', 'finance', 'monetary', 'economic', 'fiscal'},
        'earnings': {'earnings', 'revenue', 'profit', 'income', 'returns'},
        'market': {'market', 'trading', 'exchange', 'stock', 'equity'},
        'performance': {'performance', 'results', 'metrics', 'indicators', 'measures'},
        'trends': {'trends', 'patterns', 'movements', 'changes', 'shifts'},
        'analysis': {'analysis', 'reports', 'studies', 'evaluation', 'assessment'},
        'growth': {'growth', 'expansion', 'increase', 'rise', 'development'},
        'forecasts': {'forecasts', 'predictions', 'projections', 'outlook', 'expectations'},
        'volatility': {'volatility', 'fluctuation', 'variation', 'instability', 'uncertainty'},
        'policies': {'policies', 'regulations', 'rules', 'guidelines', 'standards'},
        'rates': {'rates', 'interest', 'yields', 'returns', 'pricing'},
        'business': {'business', 'corporate', 'company', 'enterprise', 'firm'},
        'investment': {'investment', 'funding', 'capital', 'financing', 'backing'},
        'trading': {'trading', 'transactions', 'deals', 'operations', 'activities'},
        'indicators': {'indicators', 'signals', 'metrics', 'measures', 'gauges'},
        'announcements': {'announcements', 'reports', 'statements', 'releases', 'updates'}
    }
    
    # Find semantic matches
    lite_semantic = set()
    full_semantic = set()
    
    for word in lite_words:
        for group_name, group_words in semantic_groups.items():
            if word in group_words:
                lite_semantic.add(group_name)
                break
    
    for word in full_words:
        for group_name, group_words in semantic_groups.items():
            if word in group_words:
                full_semantic.add(group_name)
                break
    
    semantic_intersection = lite_semantic.intersection(full_semantic)
    semantic_union = lite_semantic.union(full_semantic)
    semantic_sim = len(semantic_intersection) / len(semantic_union) if semantic_union else 0.0
    
    # Method 3: Character-level similarity for partial matches
    def char_similarity(s1, s2):
        """Calculate character-level similarity"""
        if not s1 or not s2:
            return 0.0
        
        # Use longest common subsequence
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(s1, s2)
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 0.0
    
    char_sim = char_similarity(lite_clean, full_clean)
    
    # Method 4: TF-IDF with better preprocessing
    try:
        # Combine both texts for better vocabulary
        combined_text = [lite_clean, full_clean]
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 3),  # Include trigrams
            min_df=1,
            max_features=1000,
            lowercase=True
        )
        tfidf_matrix = vectorizer.fit_transform(combined_text)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0.0
    
    # Combine all methods with adjusted weights
    # Give more weight to semantic similarity and character similarity
    combined_similarity = (
        0.3 * jaccard_sim +      # Word overlap with stemming
        0.3 * semantic_sim +     # Semantic group matching
        0.2 * cosine_sim +       # TF-IDF cosine similarity
        0.2 * char_sim           # Character-level similarity
    )
    
    # Boost similarity if both labels contain financial terms
    financial_terms = {'financial', 'market', 'stock', 'earnings', 'revenue', 'performance', 'trading', 'investment', 'economic', 'business'}
    lite_has_financial = any(term in lite_clean for term in financial_terms)
    full_has_financial = any(term in full_clean for term in financial_terms)
    
    if lite_has_financial and full_has_financial:
        combined_similarity = min(1.0, combined_similarity + 0.1)  # Boost by 0.1
    
    return round(combined_similarity, 3)

def compare_layer_results(layer_num):
    """Compare lite vs full results for a specific layer"""
    
    # Read lite results
    lite_file = f"multi_layer_lite_results/features_layer{layer_num}.csv"
    full_file = f"multi_layer_full_results/results_summary_layer{layer_num}.csv"
    
    if not os.path.exists(lite_file) or not os.path.exists(full_file):
        print(f"Warning: Missing files for layer {layer_num}")
        return None
    
    # Read lite results
    lite_df = pd.read_csv(lite_file)
    lite_features = lite_df['feature'].tolist()
    lite_labels = lite_df['llm_label'].tolist()
    lite_specialization = lite_df['specialization'].tolist()
    
    # Read full results
    full_df = pd.read_csv(full_file)
    full_features = full_df['feature'].tolist()
    full_labels = full_df['label'].tolist()
    full_f1_scores = full_df['f1_score'].tolist()
    
    # Create comparison data
    comparison_data = []
    
    for i, feature in enumerate(full_features):
        if i < len(lite_features):
            lite_feature = lite_features[i]
            lite_label = lite_labels[i]
            lite_spec = lite_specialization[i]
            full_label = full_labels[i]
            full_f1 = full_f1_scores[i]
            
            # Calculate label similarity
            label_similarity = calculate_label_similarity(lite_label, full_label)
            
            comparison_data.append({
                'layer': layer_num,
                'feature': feature,
                'lite_label': lite_label,
                'full_label': full_label,
                'f1_score': full_f1,
                'specialization': lite_spec,
                'label_similarity': label_similarity
            })
    
    return pd.DataFrame(comparison_data)

def main():
    """Main comparison function"""
    
    layers = [4, 10, 16, 22, 28]
    all_comparisons = []
    
    print("🔍 Comparing Lite vs Full Results...")
    print("=" * 50)
    
    for layer in layers:
        print(f"📊 Processing Layer {layer}...")
        layer_comparison = compare_layer_results(layer)
        
        if layer_comparison is not None:
            all_comparisons.append(layer_comparison)
            print(f"   ✅ Layer {layer}: {len(layer_comparison)} features compared")
        else:
            print(f"   ❌ Layer {layer}: Failed to process")
    
    if not all_comparisons:
        print("❌ No comparisons could be made")
        return
    
    # Combine all comparisons
    combined_df = pd.concat(all_comparisons, ignore_index=True)
    
    # Calculate summary statistics
    avg_similarity = combined_df['label_similarity'].mean()
    avg_f1_score = combined_df['f1_score'].mean()
    
    # Save detailed comparison
    comparison_file = "results_comparison_detailed.csv"
    combined_df.to_csv(comparison_file, index=False)
    
    # Create summary statistics
    summary_stats = {
        'total_features_compared': len(combined_df),
        'average_label_similarity': avg_similarity,
        'average_f1_score': avg_f1_score,
        'layers_analyzed': layers,
        'high_f1_features': len(combined_df[combined_df['f1_score'] > 0.7]),
        'low_f1_features': len(combined_df[combined_df['f1_score'] < 0.5]),
        'high_similarity_features': len(combined_df[combined_df['label_similarity'] > 0.6]),
        'low_similarity_features': len(combined_df[combined_df['label_similarity'] < 0.3])
    }
    
    # Save summary
    summary_file = "results_comparison_summary.csv"
    pd.DataFrame([summary_stats]).to_csv(summary_file, index=False)
    
    print(f"\n📈 Summary Statistics:")
    print(f"   Total features compared: {summary_stats['total_features_compared']}")
    print(f"   Average label similarity: {avg_similarity:.3f}")
    print(f"   Average F1 score: {avg_f1_score:.3f}")
    print(f"   High F1 features (>0.7): {summary_stats['high_f1_features']}")
    print(f"   Low F1 features (<0.5): {summary_stats['low_f1_features']}")
    print(f"   High similarity features (>0.6): {summary_stats['high_similarity_features']}")
    print(f"   Low similarity features (<0.3): {summary_stats['low_similarity_features']}")
    
    print(f"\n📁 Files generated:")
    print(f"   📊 Detailed comparison: {comparison_file}")
    print(f"   📈 Summary statistics: {summary_file}")
    
    # Print layer-by-layer summary
    print(f"\n📊 Layer-by-Layer Summary:")
    for layer in layers:
        layer_data = combined_df[combined_df['layer'] == layer]
        if not layer_data.empty:
            avg_f1 = layer_data['f1_score'].mean()
            avg_sim = layer_data['label_similarity'].mean()
            high_f1 = len(layer_data[layer_data['f1_score'] > 0.7])
            high_sim = len(layer_data[layer_data['label_similarity'] > 0.6])
            print(f"   Layer {layer}: F1={avg_f1:.3f}, Similarity={avg_sim:.3f}, High F1={high_f1}, High Sim={high_sim}")
    
    return combined_df, summary_stats

if __name__ == "__main__":
    main()