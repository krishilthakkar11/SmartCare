"""
SmartCare Setup and Full Pipeline Runner
Generates data, trains models, and prepares the system for dashboard use
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'sklearn', 'xgboost', 'shap'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing dependencies...")
        os.system(f"pip install -r requirements.txt")
    else:
        print("\n✓ All dependencies are installed!")
    
    return len(missing_packages) == 0

def generate_data():
    """Generate synthetic healthcare dataset"""
    print_header("Generating Synthetic Healthcare Data")
    
    try:
        from src.data_generator import HealthcareDataGenerator, main
        df = main()
        print("\n✓ Data generation completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error generating data: {str(e)}")
        return False

def prepare_features():
    """Prepare features for modeling"""
    print_header("Preparing Features")
    
    try:
        from src.feature_engineering import main
        X, y = main()
        print("\n✓ Feature preparation completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error preparing features: {str(e)}")
        return False

def train_models():
    """Train all models"""
    print_header("Training Machine Learning Models")
    
    try:
        from src.model_training import main
        trainer, X_test, y_test, df = main()
        print("\n✓ Model training completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error training models: {str(e)}")
        return False

def test_prediction():
    """Test prediction system"""
    print_header("Testing Prediction System")
    
    try:
        from src.predictor import PatientLoadPredictor, SmartScheduler
        from datetime import datetime
        
        predictor = PatientLoadPredictor()
        predictor.load_model()
        predictor.load_scaler()
        
        # Test prediction
        pred_load, category = predictor.predict_load(
            datetime.now(), 14, 'General Practice'
        )
        print(f"\n✓ Test Prediction:")
        print(f"  Predicted Load: {pred_load} patients")
        print(f"  Load Category: {category}")
        
        # Test scheduling
        scheduler = SmartScheduler(predictor)
        recommendations = scheduler.recommend_time_slots(datetime.now(), 'General Practice', num_slots=3)
        
        print(f"\n✓ Test Scheduling Recommendations:")
        print(recommendations.to_string(index=False))
        
        return True
    except Exception as e:
        print(f"\n✗ Error testing predictions: {str(e)}")
        return False

def test_shap():
    """Test SHAP explainability"""
    print_header("Testing SHAP Explainability")
    
    try:
        from src.explainability import main
        analyzer = main()
        print("\n✓ SHAP explainability is ready!")
        return True
    except Exception as e:
        print(f"\n✗ Error with SHAP: {str(e)}")
        # This is optional, so don't fail the whole pipeline
        return True

def main():
    """Run full setup and training pipeline"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║     🏥 SmartCare — Healthcare Appointment & Load Prediction 🏥║")
    print("║                                                                ║")
    print("║         Complete Setup & Model Training Pipeline              ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return False
    
    # Step 2: Generate data
    if not generate_data():
        print("\n❌ Data generation failed. Aborting.")
        return False
    
    # Step 3: Prepare features
    if not prepare_features():
        print("\n❌ Feature preparation failed. Aborting.")
        return False
    
    # Step 4: Train models
    if not train_models():
        print("\n❌ Model training failed. Aborting.")
        return False
    
    # Step 5: Test prediction
    if not test_prediction():
        print("\n❌ Prediction test failed. Aborting.")
        return False
    
    # Step 6: Test SHAP (optional)
    test_shap()
    
    # Success message
    print_header("✓ Setup Complete!")
    
    print("""
    🎉 All systems are ready! 

    Next Steps:
    
    1. Start the Dashboard:
       streamlit run app.py
    
    2. Explore the EDA Notebook:
       jupyter notebook notebooks/smartcare_eda.ipynb
    
    3. View Model Performance:
       Navigate to "Model Performance" page in the dashboard
    
    4. Make Predictions:
       Go to "Predict Load" page for individual predictions
    
    5. Get Scheduling Recommendations:
       Use "Schedule Optimization" for optimal appointment times
    
    ═══════════════════════════════════════════════════════════════
    
    📊 Project Structure:
       - config.py                 : Configuration & constants
       - app.py                   : Streamlit dashboard
       - src/data_generator.py    : Data generation
       - src/feature_engineering.py : Feature preparation
       - src/model_training.py    : Model training
       - src/explainability.py    : SHAP analysis
       - src/predictor.py         : Predictions & scheduling
       - notebooks/smartcare_eda.ipynb : EDA notebook
    
    📝 Documentation:
       - README.md                : Full project documentation
       - requirements.txt         : Python dependencies
    
    ═══════════════════════════════════════════════════════════════
    
    Happy predicting! 🚀
    """)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
