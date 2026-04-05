# SmartCare Configuration File
import os
from datetime import datetime

# Project Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Dataset Configuration
DATASET_FILE = os.path.join(DATA_DIR, 'healthcare_appointments.csv')
DATASET_SIZE = 10000
RANDOM_SEED = 42

# Time-related Configuration
HOSPITAL_OPEN_HOUR = 8
HOSPITAL_CLOSE_HOUR = 18
APPOINTMENT_SLOT_MINUTES = 30
DAYS_TO_GENERATE = 180  # 6 months of data

# Departments
DEPARTMENTS = [
    'General Practice', 
    'Cardiology', 
    'Orthopedics', 
    'Pediatrics', 
    'Dermatology',
    'ENT',
    'Gastroenterology'
]

# Load Categories
LOAD_CATEGORIES = {
    'Low': (0, 10),      # 0-10 patients
    'Medium': (11, 20),  # 11-20 patients
    'High': (21, 50)     # 21-50 patients
}

# Model Configuration
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Feature Names
FEATURE_COLUMNS = [
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'is_holiday',
    'month',
    'prev_slot_load',
    'rolling_avg_load_3h',
    'rolling_avg_load_24h',
    'doctor_availability_count',
    'dept_avg_load',
]

TARGET_COLUMN = 'patient_load'

# Model Paths
LINEAR_REGRESSION_MODEL = os.path.join(MODELS_DIR, 'linear_regression_model.pkl')
RANDOM_FOREST_MODEL = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
XGBOOST_MODEL = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
SCALER_MODEL = os.path.join(MODELS_DIR, 'scaler.pkl')
SHAP_EXPLAINER = os.path.join(MODELS_DIR, 'shap_explainer.pkl')

# Dashboard Configuration
STREAMLIT_PORT = 8501
STREAMLIT_THEME = 'light'

print("✓ Configuration loaded successfully")
