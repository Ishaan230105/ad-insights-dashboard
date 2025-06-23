# 📊 Ad Insights Dashboard
A **Streamlit-based interactive dashboard** for analyzing ad campaign performance, forecasting revenue, detecting anomalies, and predicting Click-Through Rate (CTR) using machine learning.

## 🌐 Live Demo

Check out the live app here: https://ad-insights-dashboard-yifxw9rzqtdtmu5zyf7ctn.streamlit.app/

## 🚀 Features

- ✅ Interactive filters (Location, Ad Type)
- 📈 CTR & Conversion Rate visualizations using Altair
- 📉 Revenue simulation + 7-day linear forecast
- 🧠 CTR Prediction using Random Forest Regressor
- 🚨 Anomaly detection in revenue trends
- 📋 Filtered data preview and export

## 🗂️ Dataset

Dataset_Ads.csv(borrowed from Mendeley data) includes:

- Location, Age, Gender, Income
- Ad Type, Ad Topic, Ad Placement
- Clicks, CTR, Conversion Rate, Click Time

**Revenue** does not exist in the dataset and is simulated as:
- Revenue = Clicks * 0.05

## Getting Started
You can use the following commands to get a local copy of this project up and running:
- git clone https://github.com/Ishaan230105/ad-insights-dashboard.git
- cd ad-insights-dashboard
- pip install -r requirements.txt

Once the dependencies are installed, you can launch the Streamlit app by running:
- streamlit run dashboard.py


