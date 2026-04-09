# Spatiotemporal Air Quality Estimation in Uganda
### *A Computational Framework Integrating Satellite-Derived Aerosol Indices and Ground-Level PM2.5 Measurements*

## Abstract
This capstone project investigates the efficacy of using remote sensing data as a proxy for ground-level atmospheric monitoring in resource-constrained environments. By fusing Sentinel-5P satellite observations with localized sensor data from the AirQo network, this study develops a hybrid CNN-LSTM architecture to estimate PM2.5 concentrations across the Ugandan landscape, specifically addressing the infrastructure gap in traditional monitoring networks.

## 🔗 Technical Deployment
**Web-based Inference Dashboard:** [View Application](https://streamlit.io)


## 1. Problem Statement
Air quality monitoring in Sub-Saharan Africa is hindered by the high capital expenditure required for stationary monitoring stations. In Uganda, urban centers like Kampala experience significant particulate matter fluctuations, yet rural and peri-urban areas remain unmonitored. This research proposes a scalable, cost-effective alternative by calibrating satellite-derived Aerosol Index (AI) data against localized ground truth measurements.

## 2. Methodology & Architecture
The predictive engine utilizes a dual-stage deep learning approach:
*   **Feature Extraction (CNN):** A 1D-Convolutional layer processes the multispectral satellite signals to identify high-frequency patterns in aerosol density.
*   **Sequence Modeling (LSTM):** Long Short-Term Memory units capture the temporal dependencies and diurnal cycles of pollution, utilizing a 24-hour look-back window.
*   **Normalization & Optimization:** Data was pre-processed using Min-Max scaling and optimized via a Huber Loss function to maintain robustness against sensor noise and outliers.

## 3. Data Inventory & Fusion
The research utilizes two primary data streams:
*   **In-Situ Measurements (Ground Truth):** Longitudinal PM2.5 data provided by Makerere University’s **AirQo** project.
*   **Remote Sensing (Predictors):** UV-absorbing aerosol index data sourced from the **Copernicus Sentinel-5P** mission, accessed through the Google Earth Engine (GEE) API.
*   **Validation:** The model was evaluated on a 20% hold-out test set, achieving a Mean Absolute Error (MAE) of 6.30 µg/m³.

## 4. Public Health Alignment
In accordance with the **WHO 2021 Global Air Quality Guidelines**, the system classifies output based on the recommended 24-hour mean threshold of 15 µg/m³. The framework categorizes air quality into distinct risk tiers, facilitating data-driven decision-making for public health interventions.

## 5. Repository Inventory
*   `app.py`: The computational interface for model inference.
*   `aqi_cnn_lstm_model.pth`: Serialized state dictionary of the trained neural network.
*   `requirements.txt`: Environment configuration and dependency manifest.
*   `training_data_with_satellite.csv`: Curated dataset containing aligned spatiotemporal features.

## 6. Conclusion
The results indicate that satellite-ground data fusion provides a statistically significant estimation of localized air quality. This methodology demonstrates that remote sensing can effectively augment ground-level monitoring networks, providing a viable pathway for regional environmental policy-making and respiratory health risk mitigation.

**Investigator:** Resty Lalam
**Institutional Context:** Capstone Research Project 2026 
