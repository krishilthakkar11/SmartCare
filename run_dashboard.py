"""
SmartCare Quick Start Script
Run this to immediately launch the dashboard if models are already trained
"""

import sys
import os
import subprocess
from pathlib import Path

def check_models_exist():
    """Check if trained models exist"""
    models_dir = Path('models')
    required_files = [
        'xgboost_model.pkl',
        'linear_regression_model.pkl',
        'random_forest_model.pkl',
        'scaler.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def main():
    """Run the Streamlit dashboard"""
    print("\n" + "="*70)
    print("  SmartCare — Healthcare Load Prediction Dashboard")
    print("="*70 + "\n")
    
    # Check if models exist
    models_exist, missing_files = check_models_exist()
    
    if not models_exist:
        print("⚠️ WARNING: Trained models not found!")
        print(f"\nMissing files: {', '.join(missing_files)}\n")
        print("To train models, run:")
        print("   python setup_and_train.py\n")
        
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("Aborted.")
            return False
    
    print("🚀 Starting Streamlit dashboard...\n")
    
    try:
        # Run streamlit app
        os.system('streamlit run app.py')
        return True
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {str(e)}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
