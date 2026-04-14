"""
SmartCare Setup with Kaggle Data Support
Supports both synthetic and real Kaggle appointment data
"""

import sys
import os

def ask_data_source():
    """Ask user which data source to use"""
    print("\n" + "="*70)
    print("  Data Source Selection")
    print("="*70)
    print("\n1. SYNTHETIC DATA (fast, clean, for prototyping)")
    print("   - 10,000 generated records")
    print("   - Perfect distribution")
    print("   - Best for quick testing")
    
    print("\n2. KAGGLE DATA (real, complex, for production)")
    print("   - Real appointment records (~110k)")
    print("   - Missing values, outliers, imbalance")
    print("   - Better model generalization")
    print("   - Requires: data/KaggleV2-May-2016.csv")
    
    choice = input("\nSelect data source (1 or 2): ").strip()
    return choice in ['2', 'kaggle', 'KAGGLE', 'Kaggle']

def check_kaggle_file(filepath='data/KaggleV2-May-2016.csv'):
    """Check if Kaggle dataset exists"""
    if os.path.exists(filepath):
        print(f"✓ Found Kaggle dataset: {filepath}")
        return True
    
    print(f"\n✗ Kaggle dataset not found at {filepath}")
    print("\nTo use Kaggle data:") 
    print("  1. Download: https://www.kaggle.com/joniarroba/noshowappointments")
    print("  2. Extract 'KaggleV2-May-2016.csv' to data/")
    print("  3. Run this script again")
    return False

def main():
    """Main setup pipeline"""
    print("\n" + "="*70)
    print("  SmartCare + Kaggle Setup Pipeline")
    print("="*70)
    
    # Ask which data source
    use_kaggle = ask_data_source()
    print(f"\nSelected: {'KAGGLE DATA' if use_kaggle else 'SYNTHETIC DATA'}")
    
    # Check if Kaggle file exists if needed
    if use_kaggle and not check_kaggle_file():
        print("\n⚠️  Falling back to synthetic data...")
        use_kaggle = False
    
    # Load data
    if use_kaggle:
        print("\n" + "="*70)
        print("  Loading Kaggle Data")
        print("="*70)
        try:
            from src.kaggle_loader import load_kaggle_dataset
            df = load_kaggle_dataset('data/KaggleV2-May-2016.csv', use_balancing=True)
            print(f"✓ Kaggle dataset loaded: {len(df)} records")
        except Exception as e:
            print(f"✗ Error loading Kaggle data: {e}")
            print("  Falling back to synthetic...")
            use_kaggle = False
    
    if not use_kaggle:
        print("\n" + "="*70)
        print("  Generating Synthetic Data")
        print("="*70)
        try:
            from src.data_generator import HealthcareDataGenerator
            gen = HealthcareDataGenerator()
            df = gen.generate_data(num_samples=10000, days=180)
            df = gen.add_temporal_features(df)
            gen.save_dataset(df)
            print(f"✓ Synthetic data generated: {len(df)} records")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    # Prepare features
    print("\n" + "="*70)
    print("  Preparing Features")
    print("="*70)
    try:
        from src.feature_engineering import FeatureEngineer
        from sklearn.model_selection import train_test_split
        
        engineer = FeatureEngineer()
        X, y, _ = engineer.prepare_features(df, fit=True)
        
        # Handle missing target
        if y is None:
            if 'patient_load' in df.columns:
                y = df['patient_load']
            else:
                raise ValueError("No target variable found")
        
        # Encode categorical targets
        from sklearn.preprocessing import LabelEncoder
        if hasattr(y.iloc[0], 'lower'):  # Check if string
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        engineer.save_scaler()
        print(f"✓ Features prepared: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Train models
    print("\n" + "="*70)
    print("  Training Models")
    print("="*70)
    try:
        from src.model_training import ModelTrainer
        
        trainer = ModelTrainer()
        models, results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        print("\n✓ Training Complete!")
        print("\nPerformance Summary:")
        for name, metrics in results.items():
            print(f"{name}: MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Success
    print("\n" + "="*70)
    print("  ✓ Setup Complete!")
    print("="*70)
    print(f"\nData: {'Kaggle (Real)' if use_kaggle else 'Synthetic'}")
    print("\nNext: python run_dashboard.py")
    print("      http://localhost:8503")
    print("="*70 + "\n")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
