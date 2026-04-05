"""
SmartCare Feature Engineering
Prepares features for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN
        
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical variables"""
        categorical_cols = ['department', 'patient_type']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[f'{col}_encoded'] = le.transform(df[col])
        
        return df
    
    def prepare_features(self, df, fit=True):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        df = df.copy()
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Ensure all required features exist
        required_features = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_holiday',
            'month', 'prev_slot_load', 'rolling_avg_load_3h',
            'rolling_avg_load_24h', 'doctor_availability', 'dept_avg_load'
        ]
        
        # Add hour_of_day if not present
        if 'hour_of_day' not in df.columns:
            if 'appointment_hour' in df.columns:
                df['hour_of_day'] = df['appointment_hour']
            else:
                df['hour_of_day'] = df['appointment_datetime'].dt.hour
        
        # Add department_encoded to features
        if 'department_encoded' in df.columns:
            required_features.append('department_encoded')
        
        # Ensure all features are present
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Scale numerical features
        numerical_features = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'month',
            'prev_slot_load', 'rolling_avg_load_3h', 'rolling_avg_load_24h',
            'doctor_availability', 'dept_avg_load'
        ]
        
        if fit:
            scaled_features = self.scaler.fit_transform(df[numerical_features])
        else:
            scaled_features = self.scaler.transform(df[numerical_features])
        
        # Create scaled dataframe
        scaled_df = pd.DataFrame(
            scaled_features,
            columns=[f'{col}_scaled' for col in numerical_features],
            index=df.index
        )
        
        # Combine scaled and non-scaled features
        X = pd.concat([
            scaled_df,
            df[['is_holiday', 'department_encoded'] if 'department_encoded' in df.columns else ['is_holiday']]
        ], axis=1)
        
        # Use original features for explainability
        X = df[[col for col in numerical_features + ['is_holiday'] if col in df.columns]]
        if 'department_encoded' in df.columns:
            X = pd.concat([X, df[['department_encoded']]], axis=1)
        
        y = df[self.target_column] if self.target_column in df.columns else None
        
        print(f"✓ Features prepared. Shape: {X.shape}")
        return X, y, df
    
    def save_scaler(self, filepath=SCALER_MODEL):
        """Save the scaler for future use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {filepath}")
    
    def load_scaler(self, filepath=SCALER_MODEL):
        """Load saved scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {filepath}")
        return self.scaler
    
    def save_label_encoders(self, filepath=None):
        """Save label encoders for categorical features"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'label_encoders.pkl')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"✓ Label encoders saved to {filepath}")
    
    def load_label_encoders(self, filepath=None):
        """Load label encoders for categorical features"""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, 'label_encoders.pkl')
        with open(filepath, 'rb') as f:
            self.label_encoders = pickle.load(f)
        print(f"✓ Label encoders loaded from {filepath}")
        return self.label_encoders


def main():
    """Test feature engineering"""
    from data_generator import HealthcareDataGenerator
    
    # Generate data
    generator = HealthcareDataGenerator()
    df = generator.generate_data(num_samples=1000)
    df = generator.add_temporal_features(df)
    
    # Prepare features
    fe = FeatureEngineer()
    X, y, df_processed = fe.prepare_features(df, fit=True)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING RESULTS")
    print("="*60)
    print(f"Input shape: {df.shape}")
    print(f"Output shape: {X.shape}")
    print(f"\nFeature columns:\n{list(X.columns)}")
    print(f"\nTarget statistics:")
    print(y.describe())
    
    # Save scaler
    fe.save_scaler()
    
    return X, y


if __name__ == '__main__':
    X, y = main()
