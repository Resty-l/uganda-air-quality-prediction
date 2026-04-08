
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

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

@st.cache_resource
def load_model():
    model = AdvancedCNNLSTM(input_size=4)
    model.load_state_dict(torch.load('aqi_cnn_lstm_model.pth', map_location='cpu'))
    model.eval()
    return model

st.set_page_config(page_title="Uganda Air Quality Predictor", layout="wide")
st.title("🌍 Uganda Air Quality Forecasting")

st.sidebar.header("Manual Prediction Inputs")
s5p_val = st.sidebar.slider("Satellite Aerosol Index", 0.0, 1.0, 0.2)
hour_val = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_val = st.sidebar.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

if st.sidebar.button("Run Prediction"):
    model = load_model()
    dummy_input = np.random.rand(1, 24, 4).astype(np.float32) 
    dummy_input[0, -1, 1] = s5p_val
    dummy_input[0, -1, 2] = hour_val
    dummy_input[0, -1, 3] = day_val
    
    with torch.no_grad():
        prediction = model(torch.from_numpy(dummy_input))
        final_pm25 = prediction.item() * 100 
    
    st.metric(label="Predicted PM2.5 Level", value=f"{final_pm25:.2f} µg/m³")

st.divider()
st.subheader("📊 Historical Trends")

@st.cache_data
def get_viz_data():
    df = pd.read_csv('training_data_with_satellite.csv')
    return df

try:
    df_viz = get_viz_data()
    st.line_chart(df_viz[['pm2_5_ground', 's5p_index']])
except:
    st.warning("Data file not found.")
