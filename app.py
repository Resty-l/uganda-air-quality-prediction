import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

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

# --- 2. LOADING & UI SETUP ---
@st.cache_resource
def load_model():
    model = AdvancedCNNLSTM(input_size=4)
    model.load_state_dict(torch.load('aqi_cnn_lstm_model.pth', map_location='cpu'))
    model.eval()
    return model

st.set_page_config(page_title="WHO-Aligned AQI Tracker", layout="wide")
st.title("🌍 Uganda Air Quality Tracker (WHO Aligned)")
st.markdown(f"**Current WHO Recommended 24h Mean:** <span style='color:#00e400'>15 µg/m³</span>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Data Inputs")
s5p_val = st.sidebar.slider("Satellite Index", 0.0, 1.0, 0.2)
hour_val = st.sidebar.slider("Hour", 0, 23, 12)
day_val = st.sidebar.selectbox("Day", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

# --- 3. PREDICTION & WHO LOGIC ---
if st.sidebar.button("Analyze Air Quality"):
    model = load_model()
    dummy_input = np.random.rand(1, 24, 4).astype(np.float32) 
    dummy_input[0, -1, 1] = s5p_val 
    dummy_input[0, -1, 2] = hour_val 
    dummy_input[0, -1, 3] = day_val 
    
    with torch.no_grad():
        prediction = model(torch.from_numpy(dummy_input))
        final_pm25 = max(0, prediction.item() * 150) 

    # WHO Categorization (2021 Standards)
    if final_pm25 <= 15:
        level, color, text_c = "SAFE (Within WHO Guidelines)", "#00e400", "white"
        advice = "Air quality is ideal. Safe for all outdoor activities."
    elif 15 < final_pm25 <= 25:
        level, color, text_c = "MODERATE (Interim Target 4)", "#ffff00", "black"
        advice = "Slightly above WHO guidelines. Sensitive individuals should monitor symptoms."
    elif 25 < final_pm25 <= 37.5:
        level, color, text_c = "UNHEALTHY (Interim Target 3)", "#ff7e00", "white"
        advice = "Increased risk of respiratory symptoms. Children and elderly should limit outdoor time."
    else:
        level, color, text_c = "HAZARDOUS (High Risk)", "#ff0000", "white"
        advice = "High risk of cardiovascular and respiratory issues. Avoid outdoor exercise and wear masks."

    # DISPLAY
    st.metric("Predicted PM2.5", f"{final_pm25:.2f} µg/m³")
    
    # Progress Bar
    bar_width = min(100, int((final_pm25 / 100) * 100))
    st.markdown(f"""
        <div style="background-color: #ddd; border-radius: 15px; width: 100%; height: 30px;">
            <div style="background-color: {color}; width: {bar_width}%; height: 30px; border-radius: 15px;"></div>
        </div>
        <div style="background-color:{color}; padding:20px; border-radius:10px; color:{text_c}; margin-top:20px;">
            <h2 style="margin:0;">{level}</h2>
            <p style="font-size:18px;">{advice}</p>
        </div>
        """, unsafe_allow_html=True)

# --- 4. VISUALIZATION ---
st.divider()
try:
    df_viz = pd.read_csv('training_data_with_satellite.csv')
    st.write("**Historical Pollution Trends (Ground vs Satellite)**")
    st.line_chart(df_viz[['pm2_5_ground', 's5p_index']])
    # WHO Line on Chart
    st.caption("WHO Safety Threshold: 15 µg/m³")
except:
    st.info("Upload CSV files to view historical trends.")
