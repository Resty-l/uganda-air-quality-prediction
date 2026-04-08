# 🌍 Uganda Air Quality Forecasting System
### *A Hybrid CNN-LSTM approach for Predicting PM2.5 via Satellite Fusion*

This capstone project addresses the gap in air quality monitoring across Uganda by leveraging **Sentinel-5P satellite data** and **ground-level sensor measurements**. It provides a scalable solution to estimate air pollution ($PM_{2.5}$) in regions without physical sensors.

---

## 🔗 Live Application
**Check out the live dashboard here:** [Your Streamlit App Link](https://streamlit.io)

---

## 📌 Project Overview
*   **The Problem:** High-cost ground sensors limit air quality monitoring to specific urban hubs. 
*   **The Solution:** A "Satellite-to-Ground" estimation model that uses aerosol indices from space to predict pollution levels on the ground.
*   **Health Impact:** Predictions are mapped directly to **WHO 2021 Global Air Quality Guidelines** (15 $\mu g/m^3$ safety threshold) to provide actionable health advice.

## 🛠️ Technical Methodology
1.  **Data Fusion:** Merging ground-level sensor data (AirQo) with Sentinel-5P satellite imagery via Google Earth Engine.
2.  **Model Architecture:** A **Hybrid CNN-LSTM**.
    *   **CNN (Convolutional Neural Network):** Extracts spatial features from the satellite's aerosol index pixels.
    *   **LSTM (Long Short-Term Memory):** Learns the temporal/seasonal patterns of pollution over 24-hour windows.
3.  **Performance:** The model achieved a **Mean Absolute Error (MAE) of 6.30 µg/m³**, allowing for reliable classification of air quality categories.

## 📈 Key Features
- **Real-time Inference:** Adjust satellite aerosol indices and time parameters via an interactive sidebar.
- **Color-coded Health Bar:** Instant visual feedback on pollution severity (Green → Red).
- **Geospatial Mapping:** Interactive map showing sensor distribution across Uganda.
- **Historical Analysis:** Integrated charts showing the correlation between ground truth and satellite observations.

## 📂 Repository Structure
```text
├── app.py                         # Streamlit web application
├── requirements.txt               # Project dependencies
├── aqi_cnn_lstm_model.pth         # Trained PyTorch model weights
├── training_data_with_satellite.csv # Merged dataset for visualizations
└── air-quality-data-uganda.csv    # Raw sensor data for geospatial mapping

## 🛰️ Data Sources

This project utilizes **Multi-Sensor Data Fusion** to bridge the gap between space-based observations and ground-level reality:

1. **Ground Truth Data (AirQo):**
   - **Source:** [AirQo](https://airqo.africa) - Makerere University.
   - **Details:** High-resolution $PM_{2.5}$ and $PM_{10}$ measurements from a network of low-cost sensors deployed across Kampala and other Ugandan cities.
   - **Role:** Provided the "Ground Truth" labels for training the machine learning model.

2. **Satellite Imagery (Sentinel-5P):**
   - **Source:** [European Space Agency (ESA) / Copernicus Program](https://copernicus.eu).
   - **Platform:** Accessed via **Google Earth Engine (GEE)**.
   - **Dataset:** `COPERNICUS/S5P/OFFL/L3_AER_AI` (Aerosol Index).
   - **Role:** Provided global coverage of UV-absorbing particles (dust/smoke), serving as the primary predictor for areas without physical sensors.

3. **Temporal Features:**
   - **Details:** Engineered time-based signals (Hour of Day, Day of Week).
   - **Role:** Captured the "Rush Hour" and "Cooking Time" pollution patterns common in Ugandan urban centers.

## 🏁 Conclusion
This project successfully demonstrates the potential of **Multi-Sensor Data Fusion**—combining ground-level sensors with satellite observations—to predict air quality in resource-constrained environments like Uganda. By achieving a **Mean Absolute Error of 6.30 µg/m³**, the model provides a reliable, low-cost alternative to physical monitoring stations. Aligning these predictions with **WHO 2021 Global Air Quality Guidelines** ensures that raw data is translated into actionable health insights for the community.

---

## 🔮 Future Work & Scalability
To further enhance the system's accuracy and utility, future iterations could include:
*   **Meteorological Integration:** Incorporating real-time weather data (wind speed, temperature, and humidity), as these factors heavily influence pollution dispersion.
*   **Mobile Notifications:** Developing a lightweight mobile application to send automated push alerts when air quality exceeds WHO safe limits ($15 µg/m³$).
*   **Hyper-Local Models:** Training specialized models for specific high-traffic zones like Kampala's central business district vs. rural residential areas.
*   **Real-Time Data Pipelines:** Automating the data fetch from Google Earth Engine to ensure the dashboard reflects the very latest satellite pass.
