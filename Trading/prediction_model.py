"""
Prediction Model for Trading
Creates and trains models to predict price direction using SAE features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import joblib
import os

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingPredictionModel:
    """Creates and trains models for predicting price direction"""
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the prediction model
        
        Args:
            model_type: Type of model to use ('logistic', 'random_forest', 'xgboost', 'ensemble')
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.training_history = {}
        
        # Initialize models based on type
        if model_type == 'ensemble':
            self.models = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'xgboost': xgb.XGBClassifier(random_state=42, n_jobs=-1),
                'gradient_boosting': GradientBoostingClassifier(random_state=42)
            }
        elif model_type == 'logistic':
            self.models['main'] = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.models['main'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'xgboost':
            self.models['main'] = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_target_variable(self, 
                             df: pd.DataFrame, 
                             price_col: str = 'close',
                             horizon: int = 1,
                           threshold: float = 0.001) -> pd.DataFrame:
        """
        Create target variable for price direction prediction
        
        Args:
            df: DataFrame with price data
            price_col: Name of price column
            horizon: Prediction horizon in periods
            threshold: Minimum return threshold for classification
            
        Returns:
            DataFrame with target variable added
        """
        df = df.copy()
        
        # Calculate future returns
        future_returns = df[price_col].pct_change(horizon).shift(-horizon)
        
        # Create binary target: 1 for up, 0 for down (handle NaN values)
        df['target'] = (future_returns > threshold).astype('Int64')  # Use nullable integer type
        
        # Create multi-class target for more nuanced predictions
        df['target_3class'] = pd.cut(
            future_returns, 
            bins=[-np.inf, -threshold, threshold, np.inf], 
            labels=[0, 1, 2]  # 0: down, 1: neutral, 2: up
        ).astype('Int64')  # Use nullable integer type
        
        # Create continuous target for regression
        df['target_continuous'] = future_returns
        
        # Add volatility-adjusted target
        volatility = df[price_col].pct_change().rolling(20).std()
        df['target_vol_adj'] = future_returns / (volatility + 1e-8)
        
        logger.info(f"Target variable created with horizon {horizon}")
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df
    
    def prepare_training_data(self, 
                            df: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str = 'target',
                            test_size: float = 0.2,
                            validation_size: float = 0.1) -> Dict[str, Any]:
        """
        Prepare data for training with time series split
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
            
        Returns:
            Dictionary with training data splits
        """
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_col])
        
        # Sort by index (timestamp)
        df_clean = df_clean.sort_index()
        
        # Calculate split indices
        n_samples = len(df_clean)
        train_end = int(n_samples * (1 - test_size - validation_size))
        val_end = int(n_samples * (1 - test_size))
        
        # Split data
        train_data = df_clean.iloc[:train_end]
        val_data = df_clean.iloc[train_end:val_end]
        test_data = df_clean.iloc[val_end:]
        
        # Prepare features and targets
        X_train = train_data[feature_cols].fillna(0)
        X_train = X_train.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
        y_train = train_data[target_col]
        
        X_val = val_data[feature_cols].fillna(0)
        X_val = X_val.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
        y_val = val_data[target_col]
        
        X_test = test_data[feature_cols].fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0)  # Replace infinite values with 0
        y_test = test_data[target_col]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        data_splits = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'feature_cols': feature_cols
        }
        
        logger.info(f"Data prepared: Train={len(X_train_scaled)}, Val={len(X_val_scaled)}, Test={len(X_test_scaled)}")
        
        return data_splits
    
    def train_models(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            data_splits: Dictionary with training data
            
        Returns:
            Dictionary with training results
        """
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        training_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
            val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            train_accuracy = (train_pred == y_train).mean()
            val_accuracy = (val_pred == y_val).mean()
            
            train_auc = roc_auc_score(y_train, train_proba) if train_proba is not None else None
            val_auc = roc_auc_score(y_val, val_proba) if val_proba is not None else None
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(
                    data_splits['feature_cols'], 
                    model.feature_importances_
                ))
            elif hasattr(model, 'coef_'):
                self.feature_importance[model_name] = dict(zip(
                    data_splits['feature_cols'], 
                    np.abs(model.coef_[0])
                ))
            
            training_results[model_name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'train_predictions': train_pred,
                'val_predictions': val_pred,
                'train_probabilities': train_proba,
                'val_probabilities': val_proba
            }
            
            logger.info(f"{model_name} - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            if train_auc is not None:
                logger.info(f"{model_name} - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        
        self.training_history = training_results
        return training_results
    
    def create_ensemble_predictions(self, 
                                  X: pd.DataFrame, 
                                  method: str = 'voting') -> np.ndarray:
        """
        Create ensemble predictions
        
        Args:
            X: Feature matrix
            method: Ensemble method ('voting', 'averaging', 'weighted')
            
        Returns:
            Ensemble predictions
        """
        if method == 'voting':
            # Simple majority voting
            predictions = []
            for model_name, results in self.training_history.items():
                pred = results['model'].predict(X)
                predictions.append(pred)
            
            # Majority vote
            predictions = np.array(predictions)
            ensemble_pred = np.round(predictions.mean(axis=0)).astype(int)
            
        elif method == 'averaging':
            # Average probabilities
            probabilities = []
            for model_name, results in self.training_history.items():
                if hasattr(results['model'], 'predict_proba'):
                    proba = results['model'].predict_proba(X)[:, 1]
                    probabilities.append(proba)
            
            if probabilities:
                avg_proba = np.mean(probabilities, axis=0)
                ensemble_pred = (avg_proba > 0.5).astype(int)
            else:
                ensemble_pred = np.zeros(len(X))
        
        elif method == 'weighted':
            # Weighted average based on validation performance
            weights = []
            probabilities = []
            
            for model_name, results in self.training_history.items():
                weight = results['val_accuracy']
                weights.append(weight)
                
                if hasattr(results['model'], 'predict_proba'):
                    proba = results['model'].predict_proba(X)[:, 1]
                    probabilities.append(proba * weight)
            
            if probabilities:
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                weighted_proba = np.sum(probabilities, axis=0) / weights.sum()
                ensemble_pred = (weighted_proba > 0.5).astype(int)
            else:
                ensemble_pred = np.zeros(len(X))
        
        return ensemble_pred
    
    def evaluate_models(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all models on test data
        
        Args:
            data_splits: Dictionary with test data
            
        Returns:
            Dictionary with evaluation results
        """
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        evaluation_results = {}
        
        # Evaluate individual models
        for model_name, results in self.training_history.items():
            model = results['model']
            
            # Test predictions
            test_pred = model.predict(X_test)
            test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            test_accuracy = (test_pred == y_test).mean()
            test_auc = roc_auc_score(y_test, test_proba) if test_proba is not None else None
            
            # Classification report
            class_report = classification_report(y_test, test_pred, output_dict=True)
            
            evaluation_results[model_name] = {
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'test_predictions': test_pred,
                'test_probabilities': test_proba,
                'classification_report': class_report
            }
            
            logger.info(f"{model_name} - Test Acc: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
        
        # Evaluate ensemble
        if len(self.training_history) > 1:
            ensemble_pred = self.create_ensemble_predictions(X_test)
            ensemble_accuracy = (ensemble_pred == y_test).mean()
            
            evaluation_results['ensemble'] = {
                'test_accuracy': ensemble_accuracy,
                'test_predictions': ensemble_pred,
                'test_probabilities': None
            }
            
            logger.info(f"Ensemble - Test Acc: {ensemble_accuracy:.4f}")
        
        return evaluation_results
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance for a specific model or all models
        
        Args:
            model_name: Name of specific model (None for all)
            
        Returns:
            DataFrame with feature importance
        """
        if model_name:
            if model_name in self.feature_importance:
                importance_df = pd.DataFrame([
                    {'feature': feature, 'importance': importance}
                    for feature, importance in self.feature_importance[model_name].items()
                ]).sort_values('importance', ascending=False)
                return importance_df
            else:
                logger.warning(f"No feature importance available for {model_name}")
                return pd.DataFrame()
        else:
            # Combine all models
            all_importance = []
            for model_name, importance_dict in self.feature_importance.items():
                for feature, importance in importance_dict.items():
                    all_importance.append({
                        'model': model_name,
                        'feature': feature,
                        'importance': importance
                    })
            
            if all_importance:
                importance_df = pd.DataFrame(all_importance)
                return importance_df
            else:
                return pd.DataFrame()
    
    def predict(self, X: pd.DataFrame, model_name: str = 'ensemble') -> Dict[str, np.ndarray]:
        """
        Make predictions using specified model
        
        Args:
            X: Feature matrix
            model_name: Name of model to use ('ensemble' for ensemble)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if model_name == 'ensemble' and len(self.training_history) > 1:
            predictions = self.create_ensemble_predictions(X)
            probabilities = None  # Ensemble probabilities not implemented
        elif model_name in self.training_history:
            model = self.training_history[model_name]['model']
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        else:
            raise ValueError(f"Model {model_name} not found")
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.model_type = model_data['model_type']
        logger.info(f"Model loaded from {filepath}")

def main():
    """Example usage of the prediction model"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Create synthetic features
    n_features = 20
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    sample_data = pd.DataFrame(
        np.random.randn(1000, n_features),
        columns=feature_cols,
        index=dates
    )
    
    # Add price data
    sample_data['close'] = 50000 + np.cumsum(np.random.randn(1000) * 10)
    sample_data['returns'] = sample_data['close'].pct_change()
    
    # Initialize model
    model = TradingPredictionModel(model_type='ensemble')
    
    # Create target variable
    sample_data = model.create_target_variable(sample_data)
    
    # Prepare training data
    data_splits = model.prepare_training_data(sample_data, feature_cols)
    
    # Train models
    training_results = model.train_models(data_splits)
    
    # Evaluate models
    evaluation_results = model.evaluate_models(data_splits)
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    print("Feature Importance (Top 10):")
    print(importance_df.head(10))
    
    # Make predictions
    predictions = model.predict(data_splits['X_test'])
    print(f"Predictions shape: {predictions['predictions'].shape}")

if __name__ == "__main__":
    main()


