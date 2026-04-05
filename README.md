# SmartCare — Healthcare Appointment & Load Prediction System

A complete end-to-end machine learning project for predicting patient load in hospital time slots and optimizing appointment scheduling.

## 📋 Project Overview

SmartCare uses machine learning to:
- **Predict** the number of patients arriving in each hospital time slot
- **Categorize** loads as Low, Medium, or High
- **Recommend** optimal appointment times to reduce waiting times
- **Suggest** rescheduling when appointments fall during peak hours
- **Provide** interactive visualizations and analytics through a Streamlit dashboard

## 🎯 Key Features

### 1. **Predictive Models**
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost (primary model) - Gradient boosting with best performance

### 2. **Smart Scheduling**
- Automatic recommendation of low-load time slots
- Peak hour identification and avoidance
- Alternative appointment suggestions with rescheduling logic

### 3. **Explainability**
- SHAP (SHapley Additive exPlanations) for feature importance
- Individual prediction explanation
- Interactive feature dependence plots

### 4. **Interactive Dashboard**
- Real-time load predictions
- Department-specific analysis
- Weekly load heatmaps
- Model performance metrics

## 📁 Project Structure

```
Healthcare/
├── config.py                          # Configuration & constants
├── app.py                            # Streamlit dashboard application
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── data/
│   └── (generated healthcare data)
├── models/
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
├── notebooks/
│   └── smartcare_eda.ipynb          # Exploratory Data Analysis notebook
└── src/
    ├── data_generator.py             # Synthetic data generation
    ├── feature_engineering.py        # Feature preprocessing
    ├── model_training.py             # Model training & evaluation
    ├── explainability.py             # SHAP explainability
    └── predictor.py                  # Prediction & scheduling logic
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd Healthcare

# Create a virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data & Train Models

```bash
# Generate synthetic dataset
python src/data_generator.py

# Prepare features
python src/feature_engineering.py

# Train all models
python src/model_training.py
```

### 3. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### 4. Explore the EDA Notebook

```bash
jupyter notebook notebooks/smartcare_eda.ipynb
```

## 📊 Data Structure

### Generated Dataset
- **Size**: 10,000 appointment records
- **Time Range**: 180 days (6 months)
- **Features**: 20+ engineered features

### Key Columns
| Column | Type | Description |
|--------|------|-------------|
| appointment_datetime | DateTime | Date and time of appointment |
| appointment_date | Date | Appointment date |
| appointment_hour | Int | Hour of day (8-17) |
| department | String | Hospital department |
| patient_type | String | Walk-in or Booked |
| doctor_availability | Int | Number of available doctors |
| patient_load | Int | **TARGET**: Number of patients |
| is_weekend | Int | Weekend flag (0/1) |
| day_of_week | Int | Day of week (0-6) |
| prev_slot_load | Float | Load from previous time slot |
| rolling_avg_load_3h | Float | 3-hour rolling average |
| rolling_avg_load_24h | Float | 24-hour rolling average |
| dept_avg_load | Float | Department average load |

## 🤖 Models & Performance

### Model Comparison

| Model | Type | Strengths | Best For |
|-------|------|-----------|----------|
| **Linear Regression** | Baseline | Fast, interpretable | Quick baseline |
| **Random Forest** | Ensemble | Robust, handles non-linearity | Feature importance |
| **XGBoost** | Gradient Boosting | Best accuracy, handles patterns | Production predictions |

### Evaluation Metrics
- **MAE** (Mean Absolute Error): Average absolute deviation
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R² Score**: Fraction of variance explained

## 📈 Feature Engineering

### Temporal Features
- `hour_of_day` - Hour when appointment occurs
- `day_of_week` - Day of week (0-6)
- `is_weekend` - Binary weekend indicator
- `month` - Month of year (1-12)
- `is_holiday` - Holiday indicator (for future enhancement)

### Historical Features
- `prev_slot_load` - Patient count from previous time slot
- `rolling_avg_load_3h` - 3-hour rolling average load
- `rolling_avg_load_24h` - 24-hour rolling average load

### Operational Features
- `doctor_availability` - Number of available doctors
- `dept_avg_load` - Department's average patient load

### Categorical Features (Encoded)
- `department` - Hospital department (label encoded)
- `patient_type` - Walk-in vs Booked

## 🔍 Smart Scheduling Algorithm

### Recommendation Logic

1. **Get Best Slots**: Predict load for all hours on requested date
2. **Sort by Load**: rank slots from lowest to highest predicted load
3. **Filter by Category**: Prioritize Low and Medium load slots
4. **Return Top N**: Provide top N recommendations

### Rescheduling Logic

1. **Check Current Load**: Predict patient count for current appointment
2. **Assess Category**: If High load, suggest alternatives
3. **Search Future Dates**: Look ahead 7 days for better options
4. **Rank Alternatives**: Sort by predicted load (lowest first)

## 📊 Dashboard Features

### Pages

#### 1. **Dashboard** 
   - Quick statistics and metrics
   - Today's expected load by hour
   - Department overview
   - Load category definitions

#### 2. **Predict Load**
   - Input: Date, Time, Department
   - Output: Predicted load & category
   - Advanced options for custom parameters
   - Recommendation indicator

#### 3. **Schedule Optimization**
   - **Recommend Slots**: Get best appointment times
   - **Reschedule**: Find alternatives for high-load times
   - **Weekly View**: Heatmap of loads across week

#### 4. **Model Performance**
   - Model comparison metrics
   - Feature importance visualization
   - Model status and descriptions

#### 5. **About**
   - Project overview
   - Technical stack
   - Usage guide
   - Support information

## 🎓 Machine Learning Pipeline

```
Raw Data
   ↓
Data Generation (synthetic healthcare data)
   ↓
Feature Engineering (20+ features)
   ↓
Data Splitting (80% train, 20% test)
   ↓
Model Training
   ├── Linear Regression
   ├── Random Forest
   └── XGBoost
   ↓
Model Evaluation (MAE, RMSE, R²)
   ↓
SHAP Explainability Analysis
   ↓
Deployment (Streamlit Dashboard)
```

## 🔧 Configuration

Edit `config.py` to customize:
- Number of records to generate
- Hospital hours
- Departments
- Load categories
- Features to use
- Model paths

### Key Parameters
```python
DATASET_SIZE = 10000              # Number of synthetic records
DAYS_TO_GENERATE = 180            # Date range (6 months)
HOSPITAL_OPEN_HOUR = 8            # Hospital opening time
HOSPITAL_CLOSE_HOUR = 18          # Hospital closing time
APPOINTMENT_SLOT_MINUTES = 30     # Appointment duration
```

## 📌 Load Categories

The system classifies patient load as:

- **Low**: 0-10 patients
- **Medium**: 11-20 patients  
- **High**: 21-50+ patients

Adjustable in `config.py` under `LOAD_CATEGORIES`.

## 🔐 Security & Best Practices

- Models are pickled and saved locally
- No patient personal information in generated data
- Input validation on all user inputs
- Error handling with user-friendly messages

## 🚀 Advanced Usage

### Custom Predictions
```python
from src.predictor import PatientLoadPredictor

predictor = PatientLoadPredictor()
predictor.load_model()
load, category = predictor.predict_load('2024-04-10', 14, 'Cardiology')
print(f"Predicted load: {load} ({category})")
```

### Batch Predictions
```python
predictions_df = predictor.batch_predict(data_df)
```

### Smart Scheduling
```python
from src.predictor import SmartScheduler

scheduler = SmartScheduler(predictor)
recommendations = scheduler.recommend_time_slots('2024-04-10', 'General Practice')
```

### SHAP Explainability
```python
from src.explainability import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(xgboost_model, X_train)
analyzer.create_explainer()
shap_values = analyzer.explain_predictions(X_test)
importance_df = analyzer.get_feature_importance()
```

## 📚 Data Sources

Currently uses **synthetic data** generated to mimic realistic healthcare patterns:
- Real temporal patterns (peak hours, weekday/weekend variations)
- Department-specific distributions
- Correlated features (doctor availability, load patterns)
- Realistic ranges for patient loads

Can be extended to use real datasets with appropriate data preprocessing.

## 🐛 Troubleshooting

### Models Not Found
```bash
python src/model_training.py  # Train models first
```

### Dashboard Not Loading
```bash
pip install --upgrade streamlit
streamlit run app.py --logger.level=debug
```

### Feature Mismatch
Ensure feature names in config match those in code.

## 📈 Future Enhancements

- [ ] Real-time data integration
- [ ] Patient demographic analysis
- [ ] Appointment cancellation prediction
- [ ] Doctor-specific load balancing
- [ ] Multi-hospital comparison
- [ ] Mobile app integration
- [ ] Automated email notifications
- [ ] Advanced seasonality analysis

## 📄 License

This project is open for educational and research purposes.

## ✍️ Author

Built as a comprehensive end-to-end machine learning project.

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

---

**Built with ❤️ for better healthcare management**

*Last Updated: April 2026*
