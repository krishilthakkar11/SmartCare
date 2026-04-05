# QUICK START GUIDE - SmartCare

## 🚀 Get Started in 2 Minutes

### Option 1: Full Setup (Recommended for First Time)
This will generate data, train models, and prepare everything.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full setup and training
python setup_and_train.py

# 3. Launch the dashboard
streamlit run app.py
```

### Option 2: Quick Start (Models Already Trained)
If you already have trained models, just launch the dashboard.

```bash
python run_dashboard.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

---

## 📊 What to Do Next

### 1. **Explore the Dashboard**
   - **Dashboard Page**: View today's expected load and hospital overview
   - **Predict Load**: Make predictions for specific dates/times/departments
   - **Schedule Optimization**: Get recommended appointment times
   - **Model Performance**: View metrics and feature importance

### 2. **Explore the EDA Notebook**
   Run the Jupyter notebook to see detailed data analysis:
   ```bash
   jupyter notebook notebooks/smartcare_eda.ipynb
   ```

### 3. **Try Making Predictions**
   In Python:
   ```python
   from src.predictor import PatientLoadPredictor
   from datetime import datetime
   
   predictor = PatientLoadPredictor()
   predictor.load_model()
   predictor.load_scaler()
   
   load, category = predictor.predict_load(
       datetime.now(),        # date
       14,                   # hour (2 PM)
       'General Practice'    # department
   )
   print(f"Predicted: {load} patients ({category})")
   ```

### 4. **Get Scheduling Recommendations**
   ```python
   from src.predictor import SmartScheduler
   
   scheduler = SmartScheduler(predictor)
   recommendations = scheduler.recommend_time_slots(
       '2024-04-10',
       'Cardiology',
       num_slots=5
   )
   print(recommendations)
   ```

---

## 🎯 Project Features Overview

### Load Prediction
- Predict number of patients for any time slot
- Classify as Low/Medium/High
- Works for any department and date

### Smart Scheduling
- Get top 5-10 best appointment times
- Identify peak hours to avoid
- Suggest alternatives for high-load times

### Interactive Dashboard
- Real-time predictions
- Department-specific analytics
- Weekly load heatmaps
- Model metrics and feature importance

### Explainability
- SHAP feature importance
- Individual prediction explanations
- Feature dependence analysis

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (main interface) |
| `config.py` | Configuration and constants |
| `setup_and_train.py` | Full setup & training pipeline |
| `run_dashboard.py` | Quick dashboard launcher |
| `src/data_generator.py` | Generate synthetic data |
| `src/feature_engineering.py` | Prepare features |
| `src/model_training.py` | Train ML models |
| `src/predictor.py` | Make predictions & recommendations |
| `src/explainability.py` | SHAP analysis |
| `notebooks/smartcare_eda.ipynb` | Data exploration notebook |
| `README.md` | Full documentation |

---

## 🔧 Customization

### Change Hospital Hours
Edit `config.py`:
```python
HOSPITAL_OPEN_HOUR = 8      # Open at 8 AM
HOSPITAL_CLOSE_HOUR = 18    # Close at 6 PM
```

### Add More Departments
Edit `config.py`:
```python
DEPARTMENTS = [
    'General Practice',
    'Cardiology',
    'Orthopedics',
    # Add your departments here
]
```

### Adjust Load Categories
Edit `config.py`:
```python
LOAD_CATEGORIES = {
    'Low': (0, 10),        # 0-10 patients
    'Medium': (11, 20),    # 11-20 patients
    'High': (21, 50)       # 21+ patients
}
```

### Generate More/Less Data
Edit `config.py`:
```python
DATASET_SIZE = 10000       # Number of records
DAYS_TO_GENERATE = 180     # Time period (days)
```

---

## ⚠️ Troubleshooting

### Dashboard won't start
```bash
# Update Streamlit
pip install --upgrade streamlit

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### Models not found
```bash
# Train models
python setup_and_train.py
```

### Import errors
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Memory issues with large datasets
Edit `config.py` and reduce `DATASET_SIZE`:
```python
DATASET_SIZE = 5000  # Instead of 10000
```

---

## 📈 Expected Results

After setup, you should see:

1. **Generated Data**: 10,000 appointment records
2. **Trained Models**:
   - Linear Regression (baseline)
   - Random Forest
   - XGBoost (best performance)
3. **Predictions**: Accuracy with R² > 0.85 (on XGBoost)
4. **Smart Scheduling**: Recommendations for optimal slots
5. **Interactive Dashboard**: Full visualization suite

---

## 🎓 Learning Resources

- **EDA Notebook**: Comprehensive data analysis
- **README.md**: Full technical documentation
- **Code Comments**: Detailed in-code explanations
- **Dashboard Help**: Hover tooltips and descriptions

---

## 🚀 Next Steps

After exploring the basic features:

1. Train with your own data (modify `data_generator.py`)
2. Adjust department and time configurations
3. Fine-tune model hyperparameters in `model_training.py`
4. Add custom scheduling rules in `predictor.py`
5. Deploy to production (use Docker/cloud services)

---

## 📞 Need Help?

Check the **README.md** for:
- Detailed feature descriptions
- Technical architecture
- Advanced usage examples
- Troubleshooting guide

---

**Happy predicting! 🏥📊**
