"""
SmartCare Model Training and Evaluation
Trains Linear Regression, Random Forest, and XGBoost models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression baseline model"""
        print("\n" + "="*60)
        print("TRAINING LINEAR REGRESSION")
        print("="*60)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.models['Linear Regression'] = model
        print("✓ Linear Regression trained")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['Random Forest'] = model
        print("✓ Random Forest trained")
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.models['XGBoost'] = model
        print("✓ XGBoost trained")
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        self.results[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'predictions': y_pred
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return mae, rmse, r2
    
    def train_all_models(self, X_train, y_train, X_test, y_test, X_val=None, y_val=None):
        """Train and evaluate all models"""
        print("\n" + "="*70)
        print("MODEL TRAINING AND EVALUATION")
        print("="*70)
        
        # Train models
        self.train_linear_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate all models
        print("\n" + "="*70)
        print("MODEL EVALUATION ON TEST SET")
        print("="*70)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model, X_test, y_test)
        
        return self.models, self.results
    
    def get_best_model(self):
        """Return best model based on R² score"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['R2'])
        best_model = self.models[best_model_name]
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model_name}")
        print("="*70)
        
        return best_model_name, best_model
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        with open(LINEAR_REGRESSION_MODEL, 'wb') as f:
            pickle.dump(self.models.get('Linear Regression'), f)
        print(f"✓ Linear Regression model saved")
        
        with open(RANDOM_FOREST_MODEL, 'wb') as f:
            pickle.dump(self.models.get('Random Forest'), f)
        print(f"✓ Random Forest model saved")
        
        with open(XGBOOST_MODEL, 'wb') as f:
            pickle.dump(self.models.get('XGBoost'), f)
        print(f"✓ XGBoost model saved")
    
    def load_models(self):
        """Load trained models"""
        with open(LINEAR_REGRESSION_MODEL, 'rb') as f:
            self.models['Linear Regression'] = pickle.load(f)
        
        with open(RANDOM_FOREST_MODEL, 'rb') as f:
            self.models['Random Forest'] = pickle.load(f)
        
        with open(XGBOOST_MODEL, 'rb') as f:
            self.models['XGBoost'] = pickle.load(f)
        
        print("✓ Models loaded")
        return self.models
    
    def print_comparison(self):
        """Print model comparison table"""
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        
        comparison_df = pd.DataFrame(self.results).T
        print(comparison_df[['MAE', 'RMSE', 'R2']].to_string())
        print()


def main():
    """Main execution function"""
    from src.data_generator import HealthcareDataGenerator
    from src.feature_engineering import FeatureEngineer
    
    # Generate data
    print("Loading data...")
    generator = HealthcareDataGenerator()
    df = generator.generate_data(num_samples=DATASET_SIZE, days=DAYS_TO_GENERATE)
    df = generator.add_temporal_features(df)
    
    # Prepare features
    fe = FeatureEngineer()
    X, y, df_processed = fe.prepare_features(df, fit=True)
    fe.save_scaler()
    fe.save_label_encoders()
    
    # Split data into train, validation, test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    
    # Train models
    trainer = ModelTrainer()
    models, results = trainer.train_all_models(X_train, y_train, X_test, y_test, X_val, y_val)
    
    # Print comparison
    trainer.print_comparison()
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model()
    print(f"Best Model: {best_model_name}")
    print(f"R² Score: {results[best_model_name]['R2']:.4f}")
    
    # Save models
    trainer.save_models()
    
    return trainer, X_test, y_test, df_processed


if __name__ == '__main__':
    trainer, X_test, y_test, df = main()
