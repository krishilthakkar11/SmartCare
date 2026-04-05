"""
SmartCare SHAP Explainability
Model interpretation using SHAP (SHapley Additive exPlanations)
"""

import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class ExplainabilityAnalyzer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self):
        """Create SHAP explainer"""
        print("Creating SHAP explainer...")
        
        # Use TreeExplainer for tree-based models (XGBoost)
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fall back to KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                shap.sample(self.X_train, 100)
            )
        
        print("✓ SHAP explainer created")
        return self.explainer
    
    def explain_predictions(self, X_test, y_test):
        """Generate SHAP explanations"""
        print("Generating SHAP explanations...")
        
        if self.explainer is None:
            self.create_explainer()
        
        self.shap_values = self.explainer.shap_values(X_test)
        
        print("✓ SHAP values computed")
        return self.shap_values
    
    def get_feature_importance(self):
        """Get feature importance from SHAP values"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions first.")
        
        # Handle different output formats
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(self.shap_values[0]).mean(axis=0)
        else:
            shap_vals = np.abs(self.shap_values).mean(axis=0)
        
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'SHAP_Value': shap_vals
        }).sort_values('SHAP_Value', ascending=False)
        
        return feature_importance
    
    def plot_summary(self, filepath=None):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions first.")
        
        plt.figure(figsize=(10, 6))
        
        # Handle different output formats
        if isinstance(self.shap_values, list):
            shap.summary_plot(self.shap_values[0], self.X_train, show=False)
        else:
            shap.summary_plot(self.shap_values, self.X_train, show=False)
        
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Summary plot saved to {filepath}")
        
        plt.close()
    
    def explain_single_prediction(self, sample_index, X_test):
        """Explain a single prediction"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_predictions first.")
        
        # Handle different output formats
        if isinstance(self.shap_values, list):
            sample_shap = self.shap_values[0][sample_index]
        else:
            sample_shap = self.shap_values[sample_index]
        
        explanation = pd.DataFrame({
            'Feature': X_test.columns,
            'Value': X_test.iloc[sample_index].values,
            'SHAP_Value': sample_shap
        }).sort_values('SHAP_Value', ascending=False, key=abs)
        
        return explanation
    
    def save_explainer(self, filepath=SHAP_EXPLAINER):
        """Save SHAP explainer"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.explainer, f)
        print(f"✓ SHAP explainer saved to {filepath}")
    
    def load_explainer(self, filepath=SHAP_EXPLAINER):
        """Load SHAP explainer"""
        with open(filepath, 'rb') as f:
            self.explainer = pickle.load(f)
        print(f"✓ SHAP explainer loaded from {filepath}")
        return self.explainer


def main():
    """Test explainability module"""
    from model_training import ModelTrainer
    from data_generator import HealthcareDataGenerator
    from feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Load or generate data
    print("Preparing data...")
    generator = HealthcareDataGenerator()
    df = generator.generate_data(num_samples=2000, days=DAYS_TO_GENERATE)
    df = generator.add_temporal_features(df)
    
    fe = FeatureEngineer()
    X, y, df_processed = fe.prepare_features(df, fit=True)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )
    
    # Train best model
    print("\nTraining model for explainability analysis...")
    trainer = ModelTrainer()
    trainer.train_xgboost(X_train, y_train, X_val, y_val)
    best_model = trainer.models['XGBoost']
    
    # Create explainer
    print("\nCreating explainability analysis...")
    analyzer = ExplainabilityAnalyzer(best_model, X_train)
    analyzer.create_explainer()
    
    # Get SHAP values for test set
    shap_values = analyzer.explain_predictions(X_test.head(100), y_test.head(100))
    
    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (SHAP VALUES)")
    print("="*70)
    importance_df = analyzer.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
    
    # Save explainer
    analyzer.save_explainer()
    
    print("\n✓ Explainability analysis complete!")
    return analyzer


if __name__ == '__main__':
    analyzer = main()
