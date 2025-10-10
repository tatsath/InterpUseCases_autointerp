#!/usr/bin/env python3
"""
SAE Logistic Regression Classifier for Financial Sentiment
Uses top 40 SAE features from layer 16 to train a logistic regression classifier
"""

import pandas as pd
import numpy as np
import torch
import os
import logging
from typing import Dict, List, Tuple
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAELogisticClassifier:
    """Logistic regression classifier using top SAE features"""
    
    def __init__(self, layer: int = 16, device: str = "cuda:1"):
        self.layer = layer
        self.device = device
        self.model = None
        self.feature_indices = None
        self.sae_model = None
        self.tokenizer = None
        self.llama_model = None
        
    def _load_models(self):
        """Load SAE model and Llama model"""
        logger.info("Loading SAE model...")
        
        # Load SAE model
        sae_path = f"/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun/layers.{self.layer}/sae.safetensors"
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE model not found at {sae_path}")
        
        self.sae_model = {}
        with safe_open(sae_path, framework="pt", device=self.device) as f:
            for key in f.keys():
                self.sae_model[key] = f.get_tensor(key).to(self.device)
        
        logger.info("Loading Llama model...")
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.llama_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _get_top_features(self, top_n: int = 40) -> List[int]:
        """Get top N features from layer 16 based on F1 score"""
        analysis_path = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/FinanceLabeling/combined_analysis_results.xlsx"
        df = pd.read_excel(analysis_path)
        
        # Get top features from layer 16
        layer16_features = df[df['layer'] == 16].sort_values('f1_score', ascending=False).head(top_n)
        feature_ids = layer16_features['feature'].tolist()
        
        logger.info(f"Selected top {top_n} features from layer {self.layer}: {feature_ids[:10]}...")
        return feature_ids
    
    def _extract_sae_features(self, text: str) -> np.ndarray:
        """Extract SAE features for a given text"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.llama_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer + 1]  # +1 because layer 0 is embedding
        
        # Convert to float32 to match SAE weights
        hidden_states = hidden_states.float()
        
        # Apply SAE encoder
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)
        
        # SAE forward pass
        W_enc = self.sae_model["encoder.weight"].float()  # [400, 4096]
        b_enc = self.sae_model["encoder.bias"].float()     # [400]
        
        # Encoder: x -> h (sparse activations)
        h = torch.relu(hidden_flat @ W_enc.T + b_enc)  # [seq_len, 400]
        
        # Average across sequence length to get per-text activations
        activations = h.mean(dim=0)  # [400] - one activation per SAE feature
        
        return activations.cpu().numpy()
    
    def _load_financial_data(self, max_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """Load financial dataset"""
        logger.info("Loading financial dataset...")
        df = pd.read_csv("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Trading/financial_three_class.csv")
        
        # Limit samples if specified
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        texts = df['statement'].tolist()
        labels = df['label'].tolist()
        
        logger.info(f"Loaded {len(texts)} samples with labels: {set(labels)}")
        return texts, labels
    
    def train(self, max_samples: int = 1000, test_size: float = 0.2, random_state: int = 42):
        """Train the logistic regression classifier"""
        logger.info("Starting training...")
        
        # Load models
        self._load_models()
        
        # Get top features
        self.feature_indices = self._get_top_features(40)
        
        # Load data
        texts, labels = self._load_financial_data(max_samples)
        
        # Extract features
        logger.info("Extracting SAE features...")
        features = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processing {i}/{len(texts)} samples...")
            
            try:
                sae_features = self._extract_sae_features(text)
                # Select only the top features
                selected_features = sae_features[self.feature_indices]
                features.append(selected_features)
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                # Use zero features as fallback
                features.append(np.zeros(len(self.feature_indices)))
        
        features = np.array(features)
        labels = np.array(labels)
        
        logger.info(f"Feature matrix shape: {features.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train logistic regression
        logger.info("Training logistic regression...")
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            multi_class='ovr'  # One-vs-Rest for multi-class
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Detailed evaluation
        print("\n" + "="*60)
        print("SAE LOGISTIC REGRESSION CLASSIFIER RESULTS")
        print("="*60)
        print(f"Model: Logistic Regression (One-vs-Rest)")
        print(f"Features: Top 40 SAE features from layer {self.layer}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred, target_names=['Down (0)', 'Neutral (1)', 'Up (2)']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, test_pred)
        print(cm)
        
        # Feature importance (coefficients)
        print(f"\nTop 10 Most Important Features:")
        feature_importance = np.abs(self.model.coef_).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        for i, idx in enumerate(top_features_idx):
            feature_id = self.feature_indices[idx]
            importance = feature_importance[idx]
            print(f"  {i+1:2d}. Feature {feature_id:3d}: {importance:.4f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, test_pred, output_dict=True),
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance.tolist(),
            'selected_features': self.feature_indices
        }
    
    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        sae_features = self._extract_sae_features(text)
        selected_features = sae_features[self.feature_indices].reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = self.model.predict(selected_features)[0]
        probabilities = self.model.predict_proba(selected_features)[0]
        
        class_labels = ['Down (0)', 'Neutral (1)', 'Up (2)']
        
        return {
            'text': text,
            'predicted_class': int(prediction),
            'predicted_label': class_labels[prediction],
            'confidence': float(np.max(probabilities)),
            'class_probabilities': {
                class_labels[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'raw_probabilities': probabilities.tolist()
        }
    
    def save_model(self, output_dir: str):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sklearn model
        model_path = os.path.join(output_dir, "sae_logistic_model.joblib")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'layer': self.layer,
            'feature_indices': self.feature_indices,
            'model_type': 'LogisticRegression',
            'num_features': len(self.feature_indices),
            'num_classes': 3,
            'class_labels': ['Down (0)', 'Neutral (1)', 'Up (2)']
        }
        
        metadata_path = os.path.join(output_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """Load a trained model"""
        model_path = os.path.join(model_dir, "sae_logistic_model.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model files not found in {model_dir}")
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.layer = metadata['layer']
        self.feature_indices = metadata['feature_indices']
        
        # Load SAE model for feature extraction
        self._load_models()
        
        logger.info(f"Model loaded from {model_dir}")

def main():
    """Main function to train and test the classifier"""
    # Initialize classifier
    classifier = SAELogisticClassifier(layer=16, device="cuda:1")
    
    # Train the model
    results = classifier.train(max_samples=1000, test_size=0.2)
    
    # Save the model
    output_dir = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Trading/sae_logistic_results"
    classifier.save_model(output_dir)
    
    # Test with some examples
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE TEXTS")
    print("="*60)
    
    test_texts = [
        "Apple stock surged 8% after strong earnings report",
        "Tesla shares dropped 3% following production delays", 
        "Microsoft stock remained stable during market volatility"
    ]
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\nText: '{result['text']}'")
        print(f"Predicted: {result['predicted_class']} ({result['predicted_label']})")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.3f}")

if __name__ == "__main__":
    main()
