"""
SmartCare Prediction and Smart Scheduling Logic
Predicts patient load and recommends optimal appointment slots
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class PatientLoadPredictor:
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self.label_encoders = {}
        self.load_categories = LOAD_CATEGORIES
        
    def load_model(self, model_path=XGBOOST_MODEL):
        """Load trained model"""
        try:
            print(f"Attempting to load model from: {model_path}")
            print(f"Model path exists: {os.path.exists(model_path)}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            file_size = os.path.getsize(model_path)
            print(f"Model file size: {file_size} bytes")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            print(f"✓ Model loaded successfully from {model_path}")
            print(f"Model type: {type(self.model)}")
            return self.model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_scaler(self, scaler_path=SCALER_MODEL):
        """Load scaler"""
        try:
            print(f"Attempting to load scaler from: {scaler_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded successfully from {scaler_path}")
            return self.scaler
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            raise
    
    def load_label_encoders(self, encoders_path=None):
        """Load label encoders"""
        if encoders_path is None:
            encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
        try:
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print(f"✓ Label encoders loaded from {encoders_path}")
        except FileNotFoundError:
            print(f"⚠️ Label encoders not found. Using default encoding.")
            self.label_encoders = {}
        return self.label_encoders
    
    def create_prediction_input(self, appointment_date, appointment_time, department,
                               doctor_availability=3, prev_slot_load=15, 
                               rolling_avg_3h=12, rolling_avg_24h=18,
                               dept_avg_load=16):
        """
        Create input features for prediction
        
        Args:
            appointment_date: datetime object or string (YYYY-MM-DD)
            appointment_time: string (HH:MM) or hour integer
            department: string (department name)
            doctor_availability: int (number of available doctors)
            prev_slot_load: int (patient load in previous slot)
            rolling_avg_3h: float (3-hour rolling average)
            rolling_avg_24h: float (24-hour rolling average)
            dept_avg_load: float (department average load)
        
        Returns:
            DataFrame with input features
        """
        if isinstance(appointment_date, str):
            appointment_date = pd.to_datetime(appointment_date)
        
        if isinstance(appointment_time, str):
            hour = int(appointment_time.split(':')[0])
        else:
            hour = appointment_time
        
        day_of_week = appointment_date.weekday()
        is_weekend = 1 if day_of_week in [5, 6] else 0
        month = appointment_date.month
        
        # Encode department
        dept_encoded = 0
        if 'department' in self.label_encoders:
            try:
                dept_encoded = self.label_encoders['department'].transform([department])[0]
            except (ValueError, IndexError):
                dept_encoded = 0
        
        # Create input data with features in exact order expected by model
        input_data = pd.DataFrame({
            'hour_of_day': [hour],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'month': [month],
            'prev_slot_load': [prev_slot_load],
            'rolling_avg_load_3h': [rolling_avg_3h],
            'rolling_avg_load_24h': [rolling_avg_24h],
            'doctor_availability': [doctor_availability],
            'dept_avg_load': [dept_avg_load],
            'is_holiday': [0],
            'department_encoded': [dept_encoded]
        })
        
        # Ensure column order matches training
        expected_columns = ['hour_of_day', 'day_of_week', 'is_weekend', 'month', 
                          'prev_slot_load', 'rolling_avg_load_3h', 'rolling_avg_load_24h',
                          'doctor_availability', 'dept_avg_load', 'is_holiday', 'department_encoded']
        input_data = input_data[expected_columns]
        
        return input_data
    
    def predict_load(self, appointment_date, appointment_time, department,
                    doctor_availability=3, prev_slot_load=15,
                    rolling_avg_3h=12, rolling_avg_24h=18,
                    dept_avg_load=16):
        """
        Predict patient load for a given time slot
        
        Returns:
            predicted_load, load_category
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first. Check if model files exist in the models/ directory.")
        
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Please call load_scaler() first.")
        
        # Create input features
        X = self.create_prediction_input(
            appointment_date, appointment_time, department,
            doctor_availability, prev_slot_load,
            rolling_avg_3h, rolling_avg_24h, dept_avg_load
        )
        
        # Scale features using the same scaler from training
        numerical_features = ['hour_of_day', 'day_of_week', 'is_weekend', 'month',
                            'prev_slot_load', 'rolling_avg_load_3h', 'rolling_avg_load_24h',
                            'doctor_availability', 'dept_avg_load']
        
        # Scale only numerical features
        X_numerical = X[numerical_features].copy()
        X_numerical = pd.DataFrame(
            self.scaler.transform(X_numerical),
            columns=numerical_features
        )
        
        # Keep non-numerical features as-is
        X_scaled = X_numerical.copy()
        X_scaled['is_holiday'] = X['is_holiday'].values[0]
        X_scaled['department_encoded'] = X['department_encoded'].values[0]
        
        # Reorder columns to match training order
        expected_columns = ['hour_of_day', 'day_of_week', 'is_weekend', 'month', 
                          'prev_slot_load', 'rolling_avg_load_3h', 'rolling_avg_load_24h',
                          'doctor_availability', 'dept_avg_load', 'is_holiday', 'department_encoded']
        X_scaled = X_scaled[expected_columns]
        
        # Predict
        predicted_load = max(1, int(self.model.predict(X_scaled)[0]))
        
        # Categorize load
        load_category = self.categorize_load(predicted_load)
        
        return predicted_load, load_category
    
    def categorize_load(self, patient_load):
        """Categorize patient load as Low/Medium/High"""
        for category, (min_load, max_load) in self.load_categories.items():
            if min_load <= patient_load <= max_load:
                return category
        return 'High'  # Default to high if beyond defined ranges
    
    def batch_predict(self, predictions_df):
        """
        Make predictions for multiple time slots
        
        Args:
            predictions_df: DataFrame with columns
                - appointment_date
                - appointment_hour
                - department
                - doctor_availability (optional)
                - prev_slot_load (optional)
                - rolling_avg_load_3h (optional)
                - rolling_avg_load_24h (optional)
                - dept_avg_load (optional)
        
        Returns:
            DataFrame with predictions
        """
        predictions = []
        
        for idx, row in predictions_df.iterrows():
            pred_load, category = self.predict_load(
                row['appointment_date'],
                row['appointment_hour'],
                row['department'],
                row.get('doctor_availability', 3),
                row.get('prev_slot_load', 15),
                row.get('rolling_avg_load_3h', 12),
                row.get('rolling_avg_load_24h', 18),
                row.get('dept_avg_load', 16)
            )
            
            predictions.append({
                'appointment_date': row['appointment_date'],
                'appointment_hour': row['appointment_hour'],
                'department': row['department'],
                'predicted_load': pred_load,
                'load_category': category
            })
        
        return pd.DataFrame(predictions)


class SmartScheduler:
    def __init__(self, predictor):
        self.predictor = predictor
        self.hospital_open = HOSPITAL_OPEN_HOUR
        self.hospital_close = HOSPITAL_CLOSE_HOUR
        self.slot_duration = APPOINTMENT_SLOT_MINUTES
        
    def recommend_time_slots(self, appointment_date, department, num_slots=3):
        """
        Recommend best time slots (low patient load) for a given date and department
        
        Args:
            appointment_date: datetime object or string (YYYY-MM-DD)
            department: string (department name)
            num_slots: int (number of slots to recommend)
        
        Returns:
            DataFrame with recommended slots and predicted loads
        """
        if isinstance(appointment_date, str):
            appointment_date = pd.to_datetime(appointment_date)
        
        # Generate all time slots for the day
        slot_predictions = []
        
        for hour in range(self.hospital_open, self.hospital_close):
            predicted_load, category = self.predictor.predict_load(
                appointment_date, hour, department
            )
            
            slot_predictions.append({
                'time_slot': f"{hour:02d}:00",
                'hour': hour,
                'predicted_load': predicted_load,
                'load_category': category
            })
        
        slots_df = pd.DataFrame(slot_predictions)
        
        # Sort by predicted load and get top num_slots with lowest load
        recommended = slots_df.nsmallest(num_slots, 'predicted_load')
        
        return recommended[['time_slot', 'predicted_load', 'load_category']].reset_index(drop=True)
    
    def get_peak_hours(self, appointment_date, department):
        """
        Get peak hours (high load) for a department on a given date
        
        Returns:
            List of hours with 'High' load category
        """
        peak_hours = []
        
        for hour in range(self.hospital_open, self.hospital_close):
            _, category = self.predictor.predict_load(
                appointment_date, hour, department
            )
            
            if category == 'High':
                peak_hours.append(f"{hour:02d}:00")
        
        return peak_hours
    
    def get_department_recommendations(self, appointment_date):
        """
        Get recommended time slots for each department on a given date
        
        Args:
            appointment_date: datetime object or string (YYYY-MM-DD)
        
        Returns:
            Dictionary with department as key and recommended slots as value
        """
        recommendations = {}
        
        for dept in DEPARTMENTS:
            best_slots = self.recommend_time_slots(appointment_date, dept, num_slots=3)
            recommendations[dept] = best_slots
        
        return recommendations
    
    def reschedule_recommendation(self, appointment_date, appointment_hour, 
                                  department, max_days_ahead=7):
        """
        Suggest alternative appointment slots if current slot is heavily loaded
        
        Args:
            appointment_date: datetime object or string (YYYY-MM-DD)
            appointment_hour: int (current appointment hour)
            department: string (department name)
            max_days_ahead: int (how many days ahead to search)
        
        Returns:
            Dictionary with alternative recommendations
        """
        current_load, current_category = self.predictor.predict_load(
            appointment_date, appointment_hour, department
        )
        
        if current_category != 'High':
            return {
                'current_load': current_load,
                'current_category': current_category,
                'recommendation': 'Current slot is good',
                'alternatives': None
            }
        
        # Find better alternatives
        alternatives = []
        current_date = pd.to_datetime(appointment_date)
        
        for day_offset in range(1, max_days_ahead + 1):
            future_date = current_date + timedelta(days=day_offset)
            
            # Get best slot for this date
            best_slots = self.recommend_time_slots(future_date, department, num_slots=1)
            
            if len(best_slots) > 0:
                alternatives.append({
                    'suggested_date': future_date.strftime('%Y-%m-%d'),
                    'suggested_time': best_slots.iloc[0]['time_slot'],
                    'predicted_load': int(best_slots.iloc[0]['predicted_load']),
                    'load_category': best_slots.iloc[0]['load_category']
                })
                
                # Return if we found a good alternative
                if best_slots.iloc[0]['load_category'] == 'Low':
                    break
        
        return {
            'current_load': current_load,
            'current_category': current_category,
            'recommendation': f'Current slot is {current_category}. Consider rescheduling.',
            'alternatives': alternatives[:3]  # Return top 3 alternatives
        }


def main():
    """Test prediction and scheduling"""
    # Create predictor
    predictor = PatientLoadPredictor()
    
    # Note: Model and scaler will be loaded from disk in Streamlit app
    print("SmartCare Prediction and Scheduling modules ready!")
    print("Use load_model() and load_scaler() to load trained models.")
    
    # Example usage (if model exists)
    try:
        predictor.load_model()
        predictor.load_scaler()
        
        # Create scheduler
        scheduler = SmartScheduler(predictor)
        
        # Example predictions
        print("\n" + "="*70)
        print("EXAMPLE: RECOMMEND TIME SLOTS FOR TODAY")
        print("="*70)
        today = datetime.now()
        recommendations = scheduler.recommend_time_slots(today, 'General Practice', num_slots=5)
        print(recommendations.to_string(index=False))
        
        print("\n" + "="*70)
        print("EXAMPLE: RESCHEDULE RECOMMENDATION")
        print("="*70)
        reschedule_result = scheduler.reschedule_recommendation(today, 14, 'Cardiology')
        print(f"Current Load: {reschedule_result['current_load']} ({reschedule_result['current_category']})")
        print(f"Recommendation: {reschedule_result['recommendation']}")
        if reschedule_result['alternatives']:
            print("\nAlternatives:")
            for alt in reschedule_result['alternatives']:
                print(f"  {alt['suggested_date']} at {alt['suggested_time']}: "
                      f"{alt['predicted_load']} patients ({alt['load_category']})")
        
    except FileNotFoundError:
        print("⚠ Models not found. Train models first using model_training.py")
    
    return predictor, scheduler if 'scheduler' in locals() else None


if __name__ == '__main__':
    predictor, scheduler = main()
