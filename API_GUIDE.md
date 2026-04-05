# SmartCare API Guide - Using Prediction & Scheduling Modules

## Overview

The SmartCare system provides powerful Python APIs for making predictions and recommendations programmatically, without using the Streamlit dashboard.

## Installation & Setup

```python
# 1. Add project to path
import sys
sys.path.append('path/to/SmartCare')

# 2. Import required modules
from src.predictor import PatientLoadPredictor, SmartScheduler
from datetime import datetime, timedelta
```

---

## PatientLoadPredictor - API Reference

### Initialize the Predictor

```python
from src.predictor import PatientLoadPredictor

# Create predictor instance
predictor = PatientLoadPredictor()

# Load the trained model
predictor.load_model()

# Load the scaler for feature normalization
predictor.load_scaler()
```

### Make a Single Prediction

```python
# Basic prediction
predicted_load, load_category = predictor.predict_load(
    appointment_date='2024-04-10',      # Required: date as string (YYYY-MM-DD) or datetime
    appointment_time=14,                # Required: hour (0-23) or string (HH:MM)
    department='General Practice',      # Required: department name
    doctor_availability=3,              # Optional: number of doctors (default=3)
    prev_slot_load=15,                  # Optional: previous slot load (default=15)
    rolling_avg_3h=12,                  # Optional: 3-hour avg (default=12)
    rolling_avg_24h=18,                 # Optional: 24-hour avg (default=18)
    dept_avg_load=16                    # Optional: dept avg (default=16)
)

print(f"Predicted Load: {predicted_load} patients")
print(f"Load Category: {load_category}")  # 'Low', 'Medium', or 'High'
```

### Different Date Formats

```python
from datetime import datetime

# Option 1: String format (YYYY-MM-DD)
load, category = predictor.predict_load('2024-04-10', 14, 'Cardiology')

# Option 2: Datetime object
date_obj = datetime(2024, 4, 10)
load, category = predictor.predict_load(date_obj, 14, 'Cardiology')

# Option 3: Using datetime.now()
load, category = predictor.predict_load(datetime.now(), 14, 'Cardiology')

# Option 4: Relative dates
from datetime import timedelta
tomorrow = datetime.now() + timedelta(days=1)
load, category = predictor.predict_load(tomorrow, 14, 'Cardiology')
```

### Different Time Formats

```python
# Option 1: Hour as integer (0-23)
load, category = predictor.predict_load(date, 14, 'Cardiology')  # 2 PM

# Option 2: Time as string (HH:MM)
load, category = predictor.predict_load(date, '14:00', 'Cardiology')

# Morning (8 AM)
load, category = predictor.predict_load(date, 8, 'Cardiology')

# Evening (5 PM)
load, category = predictor.predict_load(date, 17, 'Cardiology')
```

### Batch Predictions

```python
import pandas as pd

# Create DataFrame with multiple predictions
predictions_data = pd.DataFrame({
    'appointment_date': ['2024-04-10', '2024-04-11', '2024-04-12'],
    'appointment_hour': [9, 14, 16],
    'department': ['Cardiology', 'General Practice', 'Orthopedics'],
    'doctor_availability': [3, 4, 2],
    'prev_slot_load': [12, 18, 10]
})

# Make batch predictions
results = predictor.batch_predict(predictions_data)

# View results
print(results)
# Output:
#   appointment_date appointment_hour    department  predicted_load load_category
# 0        2024-04-10                9      Cardiology              15       Medium
# 1        2024-04-11               14 General Practice              22         High
# 2        2024-04-12               16      Orthopedics               8         Low
```

### Load Categories

```python
# Categories are defined in config.py
# Low:    0-10 patients
# Medium: 11-20 patients
# High:   21-50+ patients

# Customize load categorization
from config import LOAD_CATEGORIES
print(LOAD_CATEGORIES)
# Output: {'Low': (0, 10), 'Medium': (11, 20), 'High': (21, 50)}

# Manual categorization
def categorize(load):
    for category, (min_load, max_load) in LOAD_CATEGORIES.items():
        if min_load <= load <= max_load:
            return category
    return 'High'

print(categorize(5))    # 'Low'
print(categorize(15))   # 'Medium'
print(categorize(25))   # 'High'
```

---

## SmartScheduler - API Reference

### Initialize the Scheduler

```python
from src.predictor import SmartScheduler

# Create scheduler with predictor
scheduler = SmartScheduler(predictor)
```

### Recommend Best Time Slots

```python
from datetime import datetime

# Get 5 best slots for a given date and department
recommendations = scheduler.recommend_time_slots(
    appointment_date='2024-04-10',
    department='Cardiology',
    num_slots=5
)

# View recommendations
print(recommendations)
# Output:
#   time_slot  predicted_load load_category
# 0     08:00              8         Low
# 1     09:00              9         Low
# 2     17:00             11       Medium
# 3     10:00             13       Medium
# 4     15:00             15       Medium
```

### Find Peak Hours

```python
# Get hours with high patient load
peak_hours = scheduler.get_peak_hours(
    appointment_date='2024-04-10',
    department='General Practice'
)

print(f"Peak hours to avoid: {peak_hours}")
# Output: Peak hours to avoid: ['11:00', '12:00', '14:00', '15:00']
```

### Get Rescheduling Recommendations

```python
# If a patient has an appointment during high load
recommendation = scheduler.reschedule_recommendation(
    appointment_date='2024-04-10',
    appointment_hour=14,              # Current appointment at 2 PM
    department='Cardiology',
    max_days_ahead=7                  # Search up to 7 days ahead
)

# View recommendations
print(recommendation)
# Output:
# {
#     'current_load': 28,
#     'current_category': 'High',
#     'recommendation': 'Current slot is High. Consider rescheduling.',
#     'alternatives': [
#         {
#             'suggested_date': '2024-04-11',
#             'suggested_time': '09:00',
#             'predicted_load': 8,
#             'load_category': 'Low'
#         },
#         ...
#     ]
# }

# Access specific information
print(f"Current load: {recommendation['current_load']}")
print(f"Current category: {recommendation['current_category']}")
print(f"Recommendation: {recommendation['recommendation']}")

if recommendation['alternatives']:
    best_alternative = recommendation['alternatives'][0]
    print(f"Best alternative: {best_alternative['suggested_date']} at {best_alternative['suggested_time']}")
```

### Department-wide Recommendations

```python
# Get recommended slots for ALL departments on a date
department_recs = scheduler.get_department_recommendations('2024-04-10')

for dept, slots in department_recs.items():
    print(f"\n{dept}:")
    print(slots[['time_slot', 'predicted_load', 'load_category']].to_string(index=False))
```

---

## Advanced Usage Examples

### Example 1: Find Best Appointment Day

```python
from datetime import datetime, timedelta

def find_best_appointment_day(department, duration_days=7):
    """Find the day with the lowest average load"""
    best_day = None
    best_avg_load = float('inf')
    
    for day_offset in range(duration_days):
        current_date = datetime.now() + timedelta(days=day_offset)
        recommendations = scheduler.recommend_time_slots(current_date, department, 10)
        avg_load = recommendations['predicted_load'].mean()
        
        if avg_load < best_avg_load:
            best_avg_load = avg_load
            best_day = current_date
    
    return best_day, best_avg_load

best_date, avg_load = find_best_appointment_day('Cardiology')
print(f"Best day: {best_date.strftime('%Y-%m-%d')} (avg load: {avg_load:.1f})")
```

### Example 2: Optimize Multiple Appointments

```python
def schedule_multiple_appointments(appointments, departments):
    """Schedule multiple appointments optimally"""
    scheduled = []
    
    for i, dept in enumerate(departments):
        date = datetime.now() + timedelta(days=i)
        recommendations = scheduler.recommend_time_slots(date, dept, num_slots=1)
        
        if len(recommendations) > 0:
            best_slot = recommendations.iloc[0]
            scheduled.append({
                'department': dept,
                'date': date.strftime('%Y-%m-%d'),
                'time': best_slot['time_slot'],
                'predicted_load': int(best_slot['predicted_load']),
                'category': best_slot['load_category']
            })
    
    return pd.DataFrame(scheduled)

# Example: Schedule appointments for 3 departments
departments = ['Cardiology', 'General Practice', 'Orthopedics']
schedule = schedule_multiple_appointments(3, departments)
print(schedule)
```

### Example 3: Monitor Load Trends

```python
def get_load_trend(department, days=14):
    """Get average load trend over days"""
    trend = []
    
    for day_offset in range(days):
        current_date = datetime.now() + timedelta(days=day_offset)
        recommendations = scheduler.recommend_time_slots(current_date, department, 10)
        avg_load = recommendations['predicted_load'].mean()
        
        trend.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'avg_load': avg_load,
            'day_name': current_date.strftime('%A')
        })
    
    return pd.DataFrame(trend)

trend = get_load_trend('General Practice')
print(trend)
```

### Example 4: Peak Hour Analysis

```python
def analyze_peak_hours(department, date):
    """Detailed peak hour analysis"""
    analysis = {
        'peak_hours': [],
        'off_peak_hours': [],
        'avg_load': 0,
        'load_variance': 0
    }
    
    hourly_predictions = []
    for hour in range(8, 18):  # 8 AM to 5 PM
        load, category = predictor.predict_load(date, hour, department)
        hourly_predictions.append({'hour': hour, 'load': load, 'category': category})
    
    df = pd.DataFrame(hourly_predictions)
    analysis['avg_load'] = df['load'].mean()
    analysis['load_variance'] = df['load'].std()
    analysis['peak_hours'] = df[df['category'] == 'High']['hour'].tolist()
    analysis['off_peak_hours'] = df[df['category'] == 'Low']['hour'].tolist()
    
    return analysis

analysis = analyze_peak_hours('Cardiology', '2024-04-10')
print(f"Average load: {analysis['avg_load']:.1f}")
print(f"Peak hours: {analysis['peak_hours']}")
print(f"Off-peak hours: {analysis['off_peak_hours']}")
```

---

## Error Handling

```python
from src.predictor import PatientLoadPredictor, SmartScheduler

try:
    predictor = PatientLoadPredictor()
    predictor.load_model()
    predictor.load_scaler()
    
    # Make prediction
    load, category = predictor.predict_load(
        '2024-04-10', 14, 'Cardiology'
    )
    
except FileNotFoundError as e:
    print(f"Error: Model files not found. Train models first.")
    print(f"Run: python setup_and_train.py")
    
except ValueError as e:
    print(f"Input error: {str(e)}")
    
except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

---

## Performance Tips

```python
# 1. Load model once, reuse multiple times
predictor = PatientLoadPredictor()
predictor.load_model()
predictor.load_scaler()

# Good: Reuse predictor
for date in dates:
    load, _ = predictor.predict_load(date, 14, 'Cardiology')

# Bad: Don't reload every time
for date in dates:
    predictor.load_model()  # Inefficient!
    load, _ = predictor.predict_load(date, 14, 'Cardiology')

# 2. Use batch predictions for multiple rows
# Good: Batch processing
results = predictor.batch_predict(large_df)

# Bad: Row-by-row in a loop
for _, row in large_df.iterrows():
    predictor.predict_load(...)  # Slow!
```

---

## Configuration Reference

Key settings in `config.py`:

```python
# Hospital hours
HOSPITAL_OPEN_HOUR = 8
HOSPITAL_CLOSE_HOUR = 18

# Load categories
LOAD_CATEGORIES = {
    'Low': (0, 10),
    'Medium': (11, 20),
    'High': (21, 50)
}

# Available departments
DEPARTMENTS = [
    'General Practice',
    'Cardiology',
    'Orthopedics',
    # ... more departments
]

# Model paths
XGBOOST_MODEL = 'models/xgboost_model.pkl'
SCALER_MODEL = 'models/scaler.pkl'
```

---

## Troubleshooting

### "Models not found" Error
```bash
python setup_and_train.py  # Train models first
```

### "Module not found" Error
```python
import sys
sys.path.append('/path/to/SmartCare')  # Add project to path
```

### Inconsistent Predictions
```python
# Ensure using same scaler
predictor.load_scaler()  # Must call after load_model()
```

---

## Full Example Script

```python
#!/usr/bin/env python
"""Complete example using SmartCare APIs"""

import sys
sys.path.append('.')  # Add current directory

from src.predictor import PatientLoadPredictor, SmartScheduler
from datetime import datetime, timedelta

# Initialize
print("Loading SmartCare...")
predictor = PatientLoadPredictor()
predictor.load_model()
predictor.load_scaler()
scheduler = SmartScheduler(predictor)

# Example 1: Single prediction
print("\n=== SINGLE PREDICTION ===")
load, category = predictor.predict_load('2024-04-10', 14, 'Cardiology')
print(f"Cardiology at 2 PM on Apr 10: {load} patients ({category})")

# Example 2: Recommendations
print("\n=== RECOMMENDATIONS ===")
recs = scheduler.recommend_time_slots('2024-04-10', 'Cardiology', 5)
print(recs[['time_slot', 'predicted_load', 'load_category']])

# Example 3: Peak hours
print("\n=== PEAK HOURS ===")
peaks = scheduler.get_peak_hours('2024-04-10', 'Cardiology')
print(f"Peak hours: {peaks}")

# Example 4: Rescheduling
print("\n=== RESCHEDULE RECOMMENDATION ===")
reschedule = scheduler.reschedule_recommendation('2024-04-10', 14, 'Cardiology')
print(f"Current: {reschedule['current_load']} ({reschedule['current_category']})")
if reschedule['alternatives']:
    alt = reschedule['alternatives'][0]
    print(f"Suggestion: {alt['suggested_date']} at {alt['suggested_time']}")

print("\n✓ SmartCare API demonstration complete!")
```

---

**For more information, see README.md and the Streamlit dashboard!**
