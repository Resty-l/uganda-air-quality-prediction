import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- 1. MODEL ARCHITECTURE ---
class AdvancedCNNLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. CACHED MODEL & DATA LOADING ---
@st.cache_resource
def load_model():
    model = AdvancedCNNLSTM(input_size=4)
    model.load_state_dict(torch.load('aqi_cnn_lstm_model.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def get_data():
    df_viz = pd.read_csv('training_data_with_satellite.csv')
    df_map = pd.read_csv('air-quality-data-uganda.csv')
    return df_viz, df_map

# --- 3. PAGE SETUP ---
st.set_page_config(page_title="Uganda Air Quality Predictor", layout="wide")
st.title("🌍 Uganda Air Quality Forecasting")
st.markdown("Predicting $PM_{2.5}$ using **Sentinel-5P Satellite data** and **WHO Health Standards**.")

# Sidebar for Inputs
st.sidebar.header("Manual Prediction Inputs")
st.sidebar.info("Adjust the sliders to simulate current conditions.")

s5p_val = st.sidebar.slider("Satellite Aerosol Index", 0.0, 1.0, 0.2)
hour_val = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_val = st.sidebar.selectbox("Day of Week", options=[0,1,2,3,4,5,6], 
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

# --- 4. PREDICTION LOGIC & WHO STATUS ---
if st.sidebar.button("Run Prediction"):
    model = load_model()
    
    # Create dummy 24-hour sequence (1, 24, 4)
    dummy_input = np.random.rand(1, 24, 4).astype(np.float32) 
    dummy_input[0, -1, 1] = s5p_val 
    dummy_input[0, -1, 2] = hour_val 
    dummy_input[0, -1, 3] = day_val 
    
    with torch.no_grad():
        prediction = model(torch.from_numpy(dummy_input))
        # Scaled for Uganda context
        final_pm25 = max(0, prediction.item() * 150) 
    
    # WHO Status Logic
    if final_pm25 <= 15:
        status, color, note = "GOOD (WHO Safe)", "#00e400", "Fresh air. Safe for everyone."
        text_color = "white"
    elif 15 < final_pm25 <= 35:
        status, color, note = "MODERATE", "#ffff00", "Acceptable quality. Sensitive groups should monitor."
        text_color = "black"
    elif 35 < final_pm25 <= 55:
        status, color, note = "UNHEALTHY (Sensitive Groups)", "#ff7e00", "Children/Elderly should limit outdoor time."
        text_color = "white"
    else:
        status, color, note = "HAZARDOUS / POOR", "#ff0000", "Everyone should avoid outdoor exertion and wear masks."
        text_color = "white"

    # Display Metrics
    st.metric(label="Predicted PM2.5 Concentration", value=f"{final_pm25:.2f} µg/m³")
    
    # Color-coded Progress Bar
    progress_percentage = min(100, int((final_pm25 / 150) * 100))
    st.markdown(f"""
        <div style="background-color: #eee; border-radius: 10px; width: 100%; height: 25px; margin-bottom: 10px;">
            <div style="background-color: {color}; width: {progress_percentage}%; height: 25px; border-radius: 10px; transition: width 0.5s;">
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Status Advice Box
    st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; color:{text_color};">
            <h3 style="margin:0;">Status: {status}</h3>
            <p style="margin:0; font-size:18px;">{note}</p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. VISUALIZATION SECTION ---
st.divider()
st.subheader("📊 Historical Trends & Sensor Map")

try:
    df_viz, df_map = get_data()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Ground PM2.5 vs. Satellite Index Over Time**")
        st.line_chart(df_viz[['pm2_5_ground', 's5p_index']])
        st.caption("Blue: Ground Sensor | Light Blue: Satellite Aerosol Index")

    with col2:
        st.write("**Sensor Network Coverage**")
        st.map(df_map[['latitude', 'longitude']].dropna())
except:
    st.warning("Data files (CSV) not found. Please ensure they are uploaded to the GitHub repo.")
