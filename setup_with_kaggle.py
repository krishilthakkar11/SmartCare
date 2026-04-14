"""
SmartCare Setup with Kaggle Data Support
Supports both synthetic and real Kaggle appointment data
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def ask_data_source():
    """Ask user which data source to use"""
    print_header("Data Source Selection")
    print("\n1. SYNTHETIC DATA (fast, clean, for prototyping)")
    print("   - 10,000 generated records")
    print("   - Perfect distribution")
    print("   - Best for quick testing")
    
    print("\n2. KAGGLE DATA (real, complex, for production)")
    print("   - Real appointment records")
    print("   - Missing values, outliers, imbalance")
    print("   - Better model generalization")
    print("   - Requires: data/KaggleV2-May-2016.csv")
    
    choice = input("\nSelect data source (1 or 2): ").strip()
    return choice in ['2', 'kaggle', 'KAGGLE', 'Kaggle']

def check_kaggle_file(filepath='data/KaggleV2-May-2016.csv'):
    """Check if Kaggle dataset exists"""
    if os.path.exists(filepath):
        print(f"✓ Found Kaggle dataset at {filepath}")
        return True
    
    print(f"\n✗ Kaggle dataset not found at {filepath}")
    print("\nTo use Kaggle data:")
    print("1. Download from: https://www.kaggle.com/joniarroba/noshowappointments")
    print("2. Extract 'KaggleV2-May-2016.csv' to the 'data/' directory")
    print("3. Run this script again")
    return False

def generate_synthetic_data():
    """Generate synthetic healthcare dataset"""
    print_header("Generating Synthetic Healthcare Data")
    
    try:
        from src.data_generator import HealthcareDataGenerator
        
        generator = HealthcareDataGenerator()
        df = generator.generate_data(num_samples=10000, days=180)
        df = generator.add_temporal_features(df)
        generator.save_dataset(df)
        
        print("\n✓ Synthetic data generated successfully!")
        return df
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_kaggle_data(filepath='data/KaggleV2-May-2016.csv'):
    """Load and process Kaggle data"""
    print_header("Loading and Processing Kaggle Data")
    
    try:
        from src.kaggle_loader import KaggleDataLoader
        
        loader = KaggleDataLoader(filepath)
        print("\n1. Loading data...")
        loader.load_data()
        
        print("\n2. Cleaning data...")
        loader.clean_data()
        
        print("\n3. Transforming to SmartCare format...")
        loader.transform_to_smartcare_format()
        
        print("\n4. Adding temporal features...")
        loader.add_temporal_features()
        
        print("\n5. Handling class imbalance...")
        loader.handle_class_imbalance(method='oversample')
        
        print("\n6. Saving processed data...")
        loader.save_processed_data()
        
        df = loader.get_processed_data()
        print("\n✓ Kaggle data processed successfully!")
        return df
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def prepare_features(df):
    """Prepare features for modeling"""
    print_header("Preparing Features for Modeling")
    
    try:
        from src.feature_engineering import FeatureEngineer
        from config import FEATURE_COLUMNS, TARGET_COLUMN
        import pickle
        
        engineer = FeatureEngineer()
        
        # Prepare features
        X = engineer.prepare_features(df, fit=True)
        y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else df['load_category']
        
        # Encode target if necessary
        from sklearn.preprocessing import LabelEncoder
        if isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            with open('models/target_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(engineer.scaler, f)
        
        print(f"✓ Features prepared: {X.shape}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        
        return X, y, engineer
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def train_models(X, y):
    """Train all models"""
    print_header("Training Machine Learning Models")
    
    try:
        from src.model_training import ModelTrainer
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train models
        trainer = ModelTrainer()
        print("\n→ Training Linear Regression...")
        trainer.train_linear_regression(X_train, y_train)
        
        print("→ Training Random Forest...")
        trainer.train_random_forest(X_train, y_train)
        
        print("→ Training XGBoost...")
        trainer.train_xgboost(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating models...")
        trainer.evaluate_model(X_test, y_test, "Linear Regression", trainer.lr)
        trainer.evaluate_model(X_test, y_test, "Random Forest", trainer.rf)
        trainer.evaluate_model(X_test, y_test, "XGBoost", trainer.xgb)
        
        print("\n✓ Models trained successfully!")
        return trainer, X_test, y_test
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_prediction():
    """Test prediction system"""
    print_header("Testing Prediction System")
    
    try:
        from src.predictor import PatientLoadPredictor
        from datetime import datetime
        
        predictor = PatientLoadPredictor()
        predictor.load_model()
        predictor.load_scaler()
        
        # Test prediction
        pred_load, category = predictor.predict_load(
            datetime.now(), 14, 'Cardiology'
        )
        print(f"\n✓ Test Prediction:")
        print(f"  Time: 2:00 PM")
        print(f"  Department: Cardiology")
        print(f"  Predicted Load: {pred_load} patients")
        print(f"  Category: {category}")
        return True
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def print_completion_summary(data_source):
    """Print completion summary"""
    print_header("Setup Complete!")
    print(f"\nData Source: {'KAGGLE (Real)' if data_source else 'SYNTHETIC'}")
    print("\nNext Steps:")
    print("1. Run the dashboard: python run_dashboard.py")
    print("2. Open http://localhost:8503 in your browser")
    print("3. Explore predictions and insights!")
    print("\n" + "="*70)

def main():
    """Main setup pipeline"""
    print("\n" + "="*70)
    print("  SmartCare Healthcare Prediction System")
    print("  Setup and Training Pipeline")
    print("="*70)
    
    # Ask user preference
    use_kaggle = ask_data_source()
    print(f"\nSelected: {'KAGGLE DATA' if use_kaggle else 'SYNTHETIC DATA'}")
    
    # Load/generate data
    if use_kaggle:
        if not check_kaggle_file():
            print("\n⚠️  Falling back to synthetic data...")
            df = generate_synthetic_data()
            use_kaggle = False
        else:
            df = load_kaggle_data()
    else:
        df = generate_synthetic_data()
    
    if df is None:
        print("\n✗ Failed to load/generate data. Exiting.")
        return False
    
    # Show data statistics
    print_header("Data Summary")
    print(f"Total Records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPatient Load Distribution:")
    if 'load_category' in df.columns:
        print(df['load_category'].value_counts())
    elif 'patient_load' in df.columns:
        print(df['patient_load'].describe())
    
    # Prepare features
    X, y, engineer = prepare_features(df)
    if X is None:
        print("\n✗ Failed to prepare features. Exiting.")
        return False
    
    # Train models
    trainer, X_test, y_test = train_models(X, y)
    if trainer is None:
        print("\n✗ Failed to train models. Exiting.")
        return False
    
    # Test prediction
    test_prediction()
    
    # Print summary
    print_completion_summary(use_kaggle)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
