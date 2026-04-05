"""
SmartCare Data Generator
Generates synthetic healthcare appointment and patient load data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class HealthcareDataGenerator:
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.departments = DEPARTMENTS
        self.hospital_open = HOSPITAL_OPEN_HOUR
        self.hospital_close = HOSPITAL_CLOSE_HOUR
        self.slot_duration = APPOINTMENT_SLOT_MINUTES
        
    def generate_data(self, num_samples=DATASET_SIZE, days=DAYS_TO_GENERATE):
        """Generate synthetic healthcare appointment data"""
        print(f"Generating {num_samples} healthcare appointment records...")
        
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(num_samples):
            # Random date within the period
            appointment_date = start_date + timedelta(
                days=int(np.random.randint(0, days)),
                hours=int(np.random.randint(self.hospital_open, self.hospital_close)),
                minutes=int(np.random.choice(range(0, 60, self.slot_duration)))
            )
            
            department = np.random.choice(self.departments)
            patient_type = np.random.choice(['walk-in', 'booked'], p=[0.3, 0.7])
            
            # Doctor availability (1-5 doctors per department)
            doctor_availability = np.random.randint(1, 6)
            
            # Patient load per slot (influenced by hour and department)
            hour = appointment_date.hour
            base_load = np.random.normal(20, 8)
            
            # Peak hours (9-11 AM, 2-4 PM)
            if hour in [9, 10, 14, 15]:
                base_load *= 1.5
            elif hour in [8, 17]:  # Off-peak
                base_load *= 0.6
                
            patient_load = max(1, int(base_load))
            
            data.append({
                'appointment_datetime': appointment_date,
                'appointment_date': appointment_date.date(),
                'appointment_hour': appointment_date.hour,
                'appointment_minute': appointment_date.minute,
                'department': department,
                'patient_type': patient_type,
                'doctor_availability': doctor_availability,
                'patient_load': patient_load,
                'day_of_week': appointment_date.weekday(),
            })
        
        df = pd.DataFrame(data)
        
        # Sort by datetime
        df = df.sort_values('appointment_datetime').reset_index(drop=True)
        
        # Add derived features
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['appointment_date'].apply(lambda x: x.month)
        df['is_holiday'] = 0  # Can add holiday logic later
        
        print(f"✓ Generated {len(df)} records")
        return df
    
    def add_temporal_features(self, df):
        """Add temporal and rolling features"""
        print("Adding temporal features...")
        
        # Previous slot load (grouped by department and hour)
        df['prev_slot_load'] = df.groupby(['department', 'appointment_hour'])['patient_load'].shift(1).fillna(0)
        
        # Rolling averages
        df['rolling_avg_load_3h'] = df.groupby('department')['patient_load'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(drop=True)
        
        df['rolling_avg_load_24h'] = df.groupby('department')['patient_load'].rolling(
            window=24, min_periods=1
        ).mean().reset_index(drop=True)
        
        # Department average load
        dept_avg = df.groupby('department')['patient_load'].mean()
        df['dept_avg_load'] = df['department'].map(dept_avg)
        
        # Fill any remaining NaNs
        df = df.fillna(method='bfill').fillna(0)
        
        print("✓ Temporal features added")
        return df
    
    def save_dataset(self, df, filepath=DATASET_FILE):
        """Save dataset to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Dataset saved to {filepath}")
        return filepath


def main():
    """Main execution function"""
    generator = HealthcareDataGenerator()
    
    # Generate data
    df = generator.generate_data(num_samples=DATASET_SIZE, days=DAYS_TO_GENERATE)
    
    # Add features
    df = generator.add_temporal_features(df)
    
    # Display statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total Records: {len(df)}")
    print(f"Date Range: {df['appointment_date'].min()} to {df['appointment_date'].max()}")
    print(f"Departments: {df['department'].nunique()}")
    print(f"\nPatient Load Statistics:")
    print(df['patient_load'].describe())
    print(f"\nLoads by Department:")
    print(df.groupby('department')['patient_load'].describe().round(2))
    print(f"\nLoads by Hour of Day:")
    print(df.groupby('appointment_hour')['patient_load'].agg(['mean', 'std', 'min', 'max']).round(2))
    
    # Save dataset
    generator.save_dataset(df)
    
    print("\n✓ Data generation complete!")
    return df


if __name__ == '__main__':
    df = main()
