"""
SmartCare Dashboard - Main Streamlit Application
Interactive frontend for the Healthcare Appointment & Load Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from config import *
from src.predictor import PatientLoadPredictor, SmartScheduler
from src.feature_engineering import FeatureEngineer

# Set page config
st.set_page_config(
    page_title="SmartCare — Healthcare Load Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🏥 SmartCare — Healthcare Appointment & Load Prediction System")
st.markdown("Predict patient load and optimize appointment scheduling to reduce waiting times")

# Initialize session state
if 'predictor' not in st.session_state:
    try:
        st.session_state.predictor = PatientLoadPredictor()
        st.session_state.predictor.load_model()
        st.session_state.predictor.load_scaler()
        st.session_state.predictor.load_label_encoders()
        st.session_state.scheduler = SmartScheduler(st.session_state.predictor)
        print("✓ All models loaded successfully")
    except FileNotFoundError as e:
        st.error(f"⚠️ Models not found: {e}")
        st.error("Please train models first by running: python src/model_training.py")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.error("Please check the models directory and ensure all model files exist.")
        st.stop()

# Sidebar Navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select a page:", [
    "Dashboard",
    "Predict Load",
    "Schedule Optimization",
    "Model Performance",
    "About"
])

# Page: Dashboard
if page == "Dashboard":
    st.header("📊 Hospital Load Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Departments", len(DEPARTMENTS))
    with col2:
        st.metric("Operating Hours", f"{HOSPITAL_OPEN_HOUR}:00 - {HOSPITAL_CLOSE_HOUR}:00")
    with col3:
        st.metric("Slot Duration", f"{APPOINTMENT_SLOT_MINUTES} minutes")
    with col4:
        st.metric("Load Categories", "3 (Low/Med/High)")
    
    st.markdown("---")
    
    # Quick statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Load Categories")
        categories_data = {
            'Low': f"0-{LOAD_CATEGORIES['Low'][1]} patients",
            'Medium': f"{LOAD_CATEGORIES['Medium'][0]}-{LOAD_CATEGORIES['Medium'][1]} patients",
            'High': f"{LOAD_CATEGORIES['High'][0]}-{LOAD_CATEGORIES['High'][1]}+ patients"
        }
        for category, range_text in categories_data.items():
            st.write(f"**{category}**: {range_text}")
    
    with col2:
        st.subheader("🏢 Departments")
        dept_col1, dept_col2 = st.columns(2)
        for i, dept in enumerate(DEPARTMENTS):
            if i % 2 == 0:
                with dept_col1:
                    st.write(f"• {dept}")
            else:
                with dept_col2:
                    st.write(f"• {dept}")
    
    st.markdown("---")
    
    # Today's expected load
    st.subheader("📅 Today's Expected Load by Hour")
    today = datetime.now()
    today_predictions = []
    
    for hour in range(HOSPITAL_OPEN_HOUR, HOSPITAL_CLOSE_HOUR):
        pred_load, category = st.session_state.predictor.predict_load(
            today, hour, DEPARTMENTS[0]
        )
        today_predictions.append({
            'Hour': f"{hour:02d}:00",
            'Predicted Load': pred_load,
            'Category': category
        })
    
    today_df = pd.DataFrame(today_predictions)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    bar_colors = [colors[cat] for cat in today_df['Category']]
    ax.bar(today_df['Hour'], today_df['Predicted Load'], color=bar_colors, alpha=0.7)
    ax.set_title("Expected Patient Load by Hour Today", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Patient Load")
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.write(today_df.to_string(index=False))

# Page: Predict Load
elif page == "Predict Load":
    st.header("🔮 Patient Load Prediction")
    st.markdown("Predict the patient load for a specific date, time, and department")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        appointment_date = st.date_input("Select Appointment Date", value=datetime.now().date())
    
    with col2:
        appointment_hour = st.selectbox("Select Hour", 
                                        range(HOSPITAL_OPEN_HOUR, HOSPITAL_CLOSE_HOUR),
                                        format_func=lambda x: f"{x:02d}:00")
    
    with col3:
        department = st.selectbox("Select Department", DEPARTMENTS)
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            doctor_availability = st.slider("Doctor Availability", 1, 5, 3)
        
        with col2:
            prev_slot_load = st.slider("Previous Slot Load", 0, 50, 15)
        
        with col3:
            rolling_avg_3h = st.slider("Rolling Avg (3H)", 0, 50, 12)
        
        col1, col2 = st.columns(2)
        with col1:
            rolling_avg_24h = st.slider("Rolling Avg (24H)", 0, 50, 18)
        
        with col2:
            dept_avg_load = st.slider("Department Avg Load", 0, 50, 16)
    
    # Make prediction
    if st.button("🔍 Predict Load", key="predict_button"):
        try:
            predicted_load, load_category = st.session_state.predictor.predict_load(
                appointment_date, appointment_hour, department,
                doctor_availability, prev_slot_load,
                rolling_avg_3h, rolling_avg_24h, dept_avg_load
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Load", f"{predicted_load} patients")
            
            with col2:
                category_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                st.metric("Load Category", f"{category_colors[load_category]} {load_category}")
            
            with col3:
                if load_category == 'Low':
                    recommendation = "✅ Good time for appointment"
                elif load_category == 'Medium':
                    recommendation = "⚠️ Moderate wait expected"
                else:
                    recommendation = "❌ High wait time expected"
                st.metric("Recommendation", recommendation)
            
            # Show details
            st.markdown("---")
            st.subheader("📋 Prediction Details")
            details_df = pd.DataFrame({
                'Parameter': ['Date', 'Hour', 'Department', 'Predicted Load', 'Category'],
                'Value': [appointment_date, f"{appointment_hour:02d}:00", department, 
                         f"{predicted_load} patients", load_category]
            })
            st.dataframe(details_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Page: Schedule Optimization
elif page == "Schedule Optimization":
    st.header("📅 Smart Scheduling & Recommendations")
    st.markdown("Get optimal appointment slots and rescheduling recommendations")
    
    tab1, tab2, tab3 = st.tabs(["Recommend Slots", "Reschedule", "Weekly View"])
    
    with tab1:
        st.subheader("🎯 Recommend Best Time Slots")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rec_date = st.date_input("Select Date", value=datetime.now().date(), key="rec_date")
        
        with col2:
            rec_department = st.selectbox("Select Department", DEPARTMENTS, key="rec_dept")
        
        with col3:
            num_slots = st.slider("Number of Recommendations", 3, 10, 5)
        
        if st.button("Get Recommendations", key="rec_button"):
            try:
                recommendations = st.session_state.scheduler.recommend_time_slots(
                    rec_date, rec_department, num_slots
                )
                
                st.subheader(f"✅ Recommended Time Slots for {rec_date.strftime('%A, %B %d')}")
                
                # Display top recommendations
                for idx, row in recommendations.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.write(f"**⏰ {row['time_slot']}**")
                    with col2:
                        st.write(f"👥 {int(row['predicted_load'])}")
                    with col3:
                        category_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                        st.write(f"{category_colors[row['load_category']]} {row['load_category']}")
                
                # Show avoidance hours
                st.markdown("---")
                peak_hours = st.session_state.scheduler.get_peak_hours(rec_date, rec_department)
                if peak_hours:
                    st.warning(f"⚠️ Peak Hours to Avoid: {', '.join(peak_hours)}")
                else:
                    st.success("✅ No peak hours identified for this date")
                    
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
    
    with tab2:
        st.subheader("🔄 Reschedule Recommendation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            curr_date = st.date_input("Current Appointment Date", value=datetime.now().date(), 
                                     key="curr_date")
        
        with col2:
            curr_hour = st.selectbox("Current Hour", 
                                    range(HOSPITAL_OPEN_HOUR, HOSPITAL_CLOSE_HOUR),
                                    format_func=lambda x: f"{x:02d}:00",
                                    key="curr_hour")
        
        with col3:
            curr_dept = st.selectbox("Department", DEPARTMENTS, key="curr_dept")
        
        if st.button("Get Reschedule Options", key="reschedule_button"):
            try:
                result = st.session_state.scheduler.reschedule_recommendation(
                    curr_date, curr_hour, curr_dept, max_days_ahead=7
                )
                
                st.subheader("Current Appointment Status")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Load", f"{result['current_load']} patients")
                with col2:
                    category_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                    st.metric("Category", f"{category_colors[result['current_category']]} {result['current_category']}")
                
                st.info(result['recommendation'])
                
                if result['alternatives']:
                    st.subheader("📌 Alternative Appointments")
                    for i, alt in enumerate(result['alternatives'], 1):
                        with st.expander(f"Option {i}: {alt['suggested_date']} at {alt['suggested_time']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Load", f"{alt['predicted_load']} patients")
                            with col2:
                                st.metric("Category", alt['load_category'])
                            
                            if alt['load_category'] == 'Low':
                                st.success("✅ This slot has low patient load")
                            
            except Exception as e:
                st.error(f"Error getting reschedule options: {str(e)}")
    
    with tab3:
        st.subheader("📊 Weekly Load Overview")
        
        view_date = st.date_input("Start Date", value=datetime.now().date(), key="weekly_date")
        view_dept = st.selectbox("Department", DEPARTMENTS, key="weekly_dept")
        
        # Generate weekly predictions
        weekly_data = []
        for day_offset in range(7):
            current_date = pd.to_datetime(view_date) + timedelta(days=day_offset)
            day_name = current_date.strftime('%A')
            
            for hour in range(HOSPITAL_OPEN_HOUR, HOSPITAL_CLOSE_HOUR):
                pred_load, category = st.session_state.predictor.predict_load(
                    current_date, hour, view_dept
                )
                weekly_data.append({
                    'Date': day_name,
                    'Hour': f"{hour:02d}:00",
                    'Load': pred_load,
                    'Category': category
                })
        
        weekly_df = pd.DataFrame(weekly_data)
        
        # Pivot for heatmap
        pivot_df = weekly_df.pivot_table(values='Load', index='Hour', columns='Date', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Patient Load'})
        ax.set_title(f"Weekly Patient Load Heatmap - {view_dept}", fontsize=14, fontweight='bold')
        st.pyplot(fig)

# Page: Model Performance
elif page == "Model Performance":
    st.header("📈 Model Performance Metrics")
    st.markdown("View model evaluation metrics and feature importance")
    
    try:
        from src.model_training import ModelTrainer
        
        st.info("💡 Tip: Train models using `python src/model_training.py` to see performance metrics here")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Expected Metrics")
            metrics_data = {
                'Metric': ['MAE', 'RMSE', 'R² Score'],
                'Description': [
                    'Mean Absolute Error',
                    'Root Mean Squared Error',
                    'Coefficient of Determination'
                ],
                'Best Model': ['XGBoost', 'XGBoost', 'XGBoost']
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🎯 Model Comparison")
            models_data = {
                'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
                'Type': ['Baseline', 'Ensemble', 'Gradient Boosting'],
                'Status': ['Trained', 'Trained', 'Trained (Best)']
            }
            models_df = pd.DataFrame(models_data)
            st.dataframe(models_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("📌 Feature Importance (Expected)")
        feature_importance_data = {
            'Feature': [
                'hour_of_day',
                'rolling_avg_load_24h',
                'department',
                'day_of_week',
                'doctor_availability',
                'rolling_avg_load_3h',
                'dept_avg_load',
                'is_weekend',
                'month'
            ],
            'Importance': [25, 20, 18, 15, 10, 5, 4, 2, 1]
        }
        imp_df = pd.DataFrame(feature_importance_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Importance Score')
        ax.set_title('Expected Feature Importance (XGBoost Model)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading performance data: {str(e)}")

# Page: About
elif page == "About":
    st.header("ℹ️ About SmartCare")
    
    st.markdown("""
    ## SmartCare — Healthcare Appointment & Load Prediction System
    
    ### 🎯 Project Overview
    SmartCare is an end-to-end machine learning system designed to predict patient loads 
    in hospital time slots and optimize appointment scheduling to:
    
    - **Reduce waiting times** for patients
    - **Balance doctor workload** across departments
    - **improve resource allocation** and planning
    - **Enhance patient experience** with data-driven scheduling
    
    ### 🔧 Technical Stack
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, Seaborn
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Explainability**: SHAP
    - **Frontend**: Streamlit
    
    ### 📊 Models Used
    1. **Linear Regression** - Baseline model for comparison
    2. **Random Forest** - Ensemble learning approach
    3. **XGBoost** - Gradient boosting (primary model)
    
    ### ✨ Key Features
    - **Load Prediction**: Predict patient count for any time slot
    - **Smart Scheduling**: Automatic recommendation of optimal appointment times
    - **Rescheduling Suggestions**: Suggest alternatives for high-load slots
    - **Visual Analytics**: Dashboard with real-time load visualization
    - **Department Insights**: Department-specific load patterns and trends
    - **Feature Importance**: SHAP-based explainability of predictions
    
    ### 📈 Data Features
    - **Temporal Features**: Hour of day, day of week, weekend flag, month
    - **Historical Features**: Previous slot load, rolling averages
    - **Department Features**: Department-specific load patterns
    - **Resource Features**: Doctor availability, patient type
    
    ### 🚀 How to Use
    1. **Prediction**: Navigate to "Predict Load" page to predict patient count
    2. **Scheduling**: Use "Schedule Optimization" for appointment recommendations
    3. **Analytics**: View "Model Performance" for insights and feature importance
    
    ---
    
    **Built with ❤️ for better healthcare management**
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Support & Documentation")
    st.markdown("""
    - **GitHub**: [SmartCare Repository](https://github.com)
    - **Documentation**: Check the `/notebooks/smartcare_eda.ipynb` for detailed analysis
    - **Model Training**: Run `python src/model_training.py` to train models
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 2rem 0;">
    <p>SmartCare © 2024 | Healthcare Load Prediction System</p>
    <p>Powered by Machine Learning & Data Analytics</p>
</div>
""", unsafe_allow_html=True)
