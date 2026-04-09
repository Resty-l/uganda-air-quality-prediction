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

# --- 2. CACHED LOADING ---
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

# --- 3. UI SETUP ---
st.set_page_config(page_title="Uganda Air Quality Tracker", layout="wide")
st.title(" Air Quality Tracker ")

# Sidebar
st.sidebar.header("Real-time Environment Data")
s5p_val = st.sidebar.slider("Satellite Aerosol Index (Sentinel-5P)", 0.0, 5.0, 1.5) # Increased range for realism
hour_val = st.sidebar.slider("Hour of Day (Traffic Peak at 8 & 17)", 0, 23, 8)
day_val = st.sidebar.selectbox("Day", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

# --- 4. PREDICTION LOGIC ---
if st.sidebar.button("Analyze Air Quality"):
    model = load_model()
    
    # Create more realistic baseline data (not just random)
    dummy_input = np.full((1, 24, 4), 0.4).astype(np.float32) 
    dummy_input[0, :, 1] = s5p_val * 0.8 # Assume satellite saw similar things recently
    dummy_input[0, -1, 1] = s5p_val 
    dummy_input[0, -1, 2] = hour_val 
    dummy_input[0, -1, 3] = day_val 
    
    with torch.no_grad():
        prediction = model(torch.from_numpy(dummy_input))
        # IMPROVED SCALING: Kampala rarely goes below 30. 
        # We adjust the multiplier to reflect local reality (30 - 200 range)
        final_pm25 = 30 + (abs(prediction.item()) * 180) 
    
    # WHO Categories
    if final_pm25 <= 15:
        level, color, text_c = "SAFE (WHO Guidelines)", "#00e400", "white"
    elif 15 < final_pm25 <= 35:
        level, color, text_c = "MODERATE", "#ffff00", "black"
    elif 35 < final_pm25 <= 55:
        level, color, text_c = "UNHEALTHY (Sensitive Groups)", "#ff7e00", "white"
    else:
        level, color, text_c = "HAZARDOUS (Kampala Typical)", "#ff0000", "white"

    st.metric("Predicted PM2.5", f"{final_pm25:.2f} µg/m³")
    
    progress = min(100, int((final_pm25 / 150) * 100))
    st.markdown(f"""
        <div style="background-color: #ddd; border-radius: 15px; width: 100%; height: 30px;">
            <div style="background-color: {color}; width: {progress}%; height: 30px; border-radius: 15px;"></div>
        </div>
        <div style="background-color:{color}; padding:20px; border-radius:10px; color:{text_c}; margin-top:20px;">
            <h2 style="margin:0;">Status: {level}</h2>
            <p style="font-size:18px;">Recommended action: Limit outdoor exposure in high-traffic areas.</p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. DATA VISUALS & MAP ---
st.divider()
try:
    df_viz, df_map = get_data()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**Historical Trends**")
        st.line_chart(df_viz[['pm2_5_ground', 's5p_index']])

    with col2:
        st.write("**Sensor Locations (Uganda)**")
        # We drop NaNs to ensure the map loads
        map_points = df_map[['latitude', 'longitude']].dropna()
        st.map(map_points)
except:
    st.warning("Data files not found. Ensure CSVs are in the GitHub repo.")
