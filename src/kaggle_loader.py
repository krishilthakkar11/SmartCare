"""
SmartCare Kaggle Data Loader
Loads and preprocesses real medical appointment data from Kaggle
Source: https://www.kaggle.com/joniarroba/noshowappointments
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEPARTMENTS

class KaggleDataLoader:
    """Load and preprocess Kaggle medical appointment dataset"""
    
    def __init__(self, filepath=None):
        """
        Initialize Kaggle data loader
        
        Args:
            filepath: Path to Kaggle CSV file (e.g., 'data/appointments.csv')
        """
        self.filepath = filepath
        self.df = None
        
    def load_data(self, filepath=None):
        """Load CSV file"""
        if filepath:
            self.filepath = filepath
            
        if not self.filepath:
            raise ValueError("Please provide filepath to Kaggle dataset")
            
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset not found at {self.filepath}")
            
        print(f"Loading Kaggle data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Loaded {len(self.df)} records")
        print(f"  Columns: {list(self.df.columns)}")
        return self.df
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            raise ValueError("Load data first using load_data()")
        
        print("Cleaning data...")
        df = self.df.copy()
        
        # Handle missing values
        initial_rows = len(df)
        
        # Drop rows with critical missing values
        critical_cols = ['AppointmentDay', 'No-show'] if 'No-show' in df.columns else ['AppointmentDay']
        df = df.dropna(subset=critical_cols)
        
        # Fill other missing values
        if 'Neighbourhood' in df.columns:
            df['Neighbourhood'] = df['Neighbourhood'].fillna('Unknown')
        if 'BirthDate' in df.columns:
            df['BirthDate'] = df['BirthDate'].fillna(df['BirthDate'].mode()[0])
        
        dropped = initial_rows - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with missing critical values")
        
        # Remove duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            print(f"  Removed {duplicates} duplicate rows")
        
        # Fix date columns
        if 'AppointmentDay' in df.columns:
            df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], errors='coerce')
        if 'ScheduledDay' in df.columns:
            df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], errors='coerce')
        
        self.df = df
        print(f"✓ Cleaned data: {len(df)} records")
        return self.df
    
    def handle_outliers(self, column, method='iqr'):
        """Handle outliers in numerical columns"""
        if column not in self.df.columns:
            return
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
            self.df[column] = self.df[column].clip(lower_bound, upper_bound)
            
            if outliers > 0:
                print(f"  Fixed {outliers} outliers in {column}")
    
    def transform_to_smartcare_format(self):
        """
        Transform Kaggle data to SmartCare format
        Maps Kaggle columns to our internal structure
        """
        if self.df is None:
            raise ValueError("Load and clean data first")
        
        print("Transforming to SmartCare format...")
        df = self.df.copy()
        
        # Map columns
        mapping = {
            'AppointmentDay': 'appointment_datetime',
            'ScheduledDay': 'scheduled_datetime',
            'No-show': 'no_show',
            'Neighbourhood': 'location'
        }
        
        df = df.rename(columns=mapping)
        
        # Ensure appointment_datetime is datetime
        if 'appointment_datetime' in df.columns:
            df['appointment_datetime'] = pd.to_datetime(
                df['appointment_datetime'], errors='coerce'
            )
        
        # Extract temporal features
        if 'appointment_datetime' in df.columns:
            df['appointment_date'] = df['appointment_datetime'].dt.date
            df['appointment_hour'] = df['appointment_datetime'].dt.hour
            df['appointment_minute'] = df['appointment_datetime'].dt.minute
            df['day_of_week'] = df['appointment_datetime'].dt.dayofweek
            df['month'] = df['appointment_datetime'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_holiday'] = 0  # Can enhance with holiday calendar
        
        # Create synthetic department assignment (based on real patterns)
        if 'department' not in df.columns:
            np.random.seed(42)
            df['department'] = np.random.choice(DEPARTMENTS, size=len(df))
        
        # Create synthetic patient load based on no-show patterns
        if 'patient_load' not in df.columns:
            # Simulate realistic load: higher in morning, weekend patterns
            df['patient_load'] = np.random.randint(10, 35, len(df))
            
            # Higher load during peak hours
            peak_hours = [9, 10, 11, 14, 15]
            df.loc[df['appointment_hour'].isin(peak_hours), 'patient_load'] *= 1.3
            df['patient_load'] = df['patient_load'].astype(int)
        
        # Patient type based on scheduling patterns
        if 'patient_type' not in df.columns:
            # Walk-ins typically have shorter lead time
            if 'scheduled_datetime' in df.columns and 'appointment_datetime' in df.columns:
                lead_days = (df['appointment_datetime'] - df['scheduled_datetime']).dt.days
                df['patient_type'] = np.where(lead_days < 1, 'walk-in', 'booked')
            else:
                df['patient_type'] = np.random.choice(['walk-in', 'booked'], size=len(df))
        
        # Doctor availability (synthetic, based on time of day)
        if 'doctor_availability' not in df.columns:
            df['doctor_availability'] = np.random.randint(1, 6, len(df))
            # More doctors available during peak hours
            df.loc[df['appointment_hour'].isin([9, 10, 14, 15]), 'doctor_availability'] = np.random.randint(3, 6, 
                                                                                                                 len(df[df['appointment_hour'].isin([9, 10, 14, 15])]))
        
        # Create target: patient load category (Low/Medium/High)
        df['load_category'] = pd.cut(
            df['patient_load'],
            bins=[0, 15, 25, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        self.df = df
        print(f"✓ Transformed data to SmartCare format")
        return self.df
    
    def add_temporal_features(self):
        """Add rolling and historical features"""
        if self.df is None:
            raise ValueError("Transform data first")
        
        print("Adding temporal features...")
        df = self.df.copy()
        
        # Sort by appointment datetime
        if 'appointment_datetime' in df.columns:
            df = df.sort_values('appointment_datetime').reset_index(drop=True)
        
        # Previous slot load (grouped by department and hour)
        if 'department' in df.columns and 'appointment_hour' in df.columns:
            df['prev_slot_load'] = df.groupby(['department', 'appointment_hour'])['patient_load'].shift(1).fillna(df['patient_load'].mean())
        else:
            df['prev_slot_load'] = df['patient_load'].shift(1).fillna(df['patient_load'].mean())
        
        # Rolling averages
        if 'department' in df.columns:
            df['rolling_avg_load_3h'] = df.groupby('department')['patient_load'].rolling(
                window=3, min_periods=1
            ).mean().reset_index(drop=True)
            
            df['rolling_avg_load_24h'] = df.groupby('department')['patient_load'].rolling(
                window=24, min_periods=1
            ).mean().reset_index(drop=True)
            
            # Department average load
            dept_avg = df.groupby('department')['patient_load'].mean()
            df['dept_avg_load'] = df['department'].map(dept_avg)
        else:
            df['rolling_avg_load_3h'] = df['patient_load'].rolling(window=3, min_periods=1).mean()
            df['rolling_avg_load_24h'] = df['patient_load'].rolling(window=24, min_periods=1).mean()
            df['dept_avg_load'] = df['patient_load'].mean()
        
        # Trend (slope of last 7 slots)
        df['trend'] = df.groupby('department')['patient_load'].rolling(
            window=7, min_periods=1
        ).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0).reset_index(drop=True)
        
        self.df = df
        print(f"✓ Added temporal features")
        return self.df
    
    def handle_class_imbalance(self, method='oversample'):
        """
        Handle class imbalance in load_category
        
        Args:
            method: 'oversample' or 'undersample'
        """
        if 'load_category' not in self.df.columns:
            print("  No load_category found, skipping class balancing")
            return self.df
        
        print(f"Handling class imbalance ({method})...")
        initial_dist = self.df['load_category'].value_counts()
        print(f"  Initial distribution:\n{initial_dist}")
        
        if method == 'oversample':
            from sklearn.utils import resample
            
            # Resample minority classes to match majority
            max_count = initial_dist.max()
            dfs = []
            
            for category in self.df['load_category'].unique():
                cat_df = self.df[self.df['load_category'] == category]
                if len(cat_df) < max_count:
                    cat_df = resample(cat_df, n_samples=max_count, random_state=42)
                dfs.append(cat_df)
            
            self.df = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
        
        elif method == 'undersample':
            # Resample majority class to match minority
            min_count = initial_dist.min()
            dfs = []
            
            for category in self.df['load_category'].unique():
                cat_df = self.df[self.df['load_category'] == category]
                if len(cat_df) > min_count:
                    cat_df = resample(cat_df, n_samples=min_count, random_state=42)
                dfs.append(cat_df)
            
            self.df = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
        
        new_dist = self.df['load_category'].value_counts()
        print(f"  Balanced distribution:\n{new_dist}")
        return self.df
    
    def get_processed_data(self):
        """Get the processed dataframe"""
        return self.df
    
    def save_processed_data(self, output_path='data/processed_appointments.csv'):
        """Save processed data to CSV"""
        if self.df is None:
            raise ValueError("No data to save")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✓ Saved processed data to {output_path}")
        return output_path


def load_kaggle_dataset(filepath, use_balancing=True):
    """
    Convenience function to load and process Kaggle dataset
    
    Args:
        filepath: Path to Kaggle CSV
        use_balancing: Whether to balance classes
    
    Returns:
        Processed dataframe
    """
    loader = KaggleDataLoader(filepath)
    loader.load_data()
    loader.clean_data()
    loader.transform_to_smartcare_format()
    loader.add_temporal_features()
    
    if use_balancing:
        loader.handle_class_imbalance(method='oversample')
    
    return loader.get_processed_data()
