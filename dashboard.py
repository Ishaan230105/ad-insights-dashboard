import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Ad Insights Dashboard", layout="wide")
st.title("📊 Ad Data Visualization & Insights")

# Loading the dataset here
@st.cache_data
def load_data():
    return pd.read_csv("Dataset_Ads.csv")

df = load_data()

# Datetime conversion
if 'Click Time' in df.columns:
    df['Click Time'] = pd.to_datetime(df['Click Time'], errors='coerce')

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    selected_location = st.multiselect("Select Location(s)", df['Location'].unique(), default=df['Location'].unique())
    selected_ad_type = st.multiselect("Select Ad Type(s)", df['Ad Type'].unique(), default=df['Ad Type'].unique())

# Applying filters
filtered_df = df[
    (df['Location'].isin(selected_location)) &
    (df['Ad Type'].isin(selected_ad_type))
].copy()

# Numeric types
filtered_df["Clicks"] = pd.to_numeric(filtered_df["Clicks"], errors="coerce")
filtered_df["CTR"] = pd.to_numeric(filtered_df["CTR"], errors="coerce")
filtered_df["Conversion Rate"] = pd.to_numeric(filtered_df["Conversion Rate"], errors="coerce")

# derived metrics
total_clicks = filtered_df['Clicks'].sum()
avg_ctr = filtered_df['CTR'].mean()
avg_cvr = filtered_df['Conversion Rate'].mean()

st.metric("Total Clicks", total_clicks)
st.metric("Average CTR", f"{avg_ctr:.2f}%")
st.metric("Average Conversion Rate", f"{avg_cvr:.2f}%")

# Visualizing CTR by Ad type
st.subheader("🎯 CTR by Ad Type")
ctr_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Ad Type:N',
    y='mean(CTR):Q',
    color='Ad Type:N',
    tooltip=['Ad Type', 'mean(CTR)']
).properties(width=600, height=400)
st.altair_chart(ctr_chart, use_container_width=True)

# Visualizing Cinversion rate by location
st.subheader("🌍 Conversion Rate by Location")
cvr_chart = alt.Chart(filtered_df).mark_bar().encode(
    x='Location:N',
    y='mean(Conversion Rate):Q',
    color='Location:N',
    tooltip=['Location', 'mean(Conversion Rate)']
).properties(width=600, height=400)
st.altair_chart(cvr_chart, use_container_width=True)

# Clicks over time
if 'Click Time' in filtered_df.columns:
    st.subheader("⏱️ Clicks Over Time")
    time_series = filtered_df.groupby(filtered_df['Click Time'].dt.date)['Clicks'].sum().reset_index()
    time_series.columns = ['Date', 'Clicks']
    st.line_chart(time_series.set_index('Date'))

# simulating revenue(Revenue does not exist in dataset)
filtered_df["Revenue"] = filtered_df["Clicks"] * 0.05 

# Forecasting revenue
st.subheader("📈 Revenue Forecast (Linear Regression)")

if "Click Time" in filtered_df.columns and "Revenue" in filtered_df.columns:
    time_df = filtered_df.groupby(filtered_df["Click Time"].dt.date)["Revenue"].sum().reset_index()
    time_df.columns = ["Date", "Revenue"]

    if len(time_df) >= 2:
        time_df["DateOrdinal"] = pd.to_datetime(time_df["Date"]).map(lambda x: x.toordinal())
        X = time_df["DateOrdinal"].values.reshape(-1, 1)
        y = time_df["Revenue"].values

        model = LinearRegression()
        model.fit(X, y)

        future_dates = pd.date_range(start=time_df["Date"].max(), periods=8)[1:]
        future_ordinals = future_dates.map(lambda x: x.toordinal()).values.reshape(-1, 1)
        forecast = model.predict(future_ordinals)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Revenue": forecast
        })

        st.line_chart(forecast_df.set_index("Date"))
        st.dataframe(forecast_df.round(2))
    else:
        st.warning("Not enough historical data for forecasting.")
else:
    st.warning("Missing 'Click Time' or 'Revenue' column.")

# Auto insights
st.subheader("📌 Auto Insights")

# Top location by revenue
top_location = filtered_df.groupby("Location")["Revenue"].sum().idxmax()
top_location_value = filtered_df.groupby("Location")["Revenue"].sum().max()
st.markdown(f"🗺️ **Top Location by Revenue**: `{top_location}` with ₹{top_location_value:.2f}`")

# Best ad placement
if "Ad Placement" in filtered_df.columns:
    top_device = filtered_df.groupby("Ad Placement")["CTR"].mean().idxmax()
    st.markdown(f"📱 **Best Ad Placement by CTR**: `{top_device}`")

# Highest revenue day
filtered_df["Click Date"] = pd.to_datetime(filtered_df["Click Time"]).dt.date
top_day = filtered_df.groupby("Click Date")["Revenue"].sum().idxmax()
top_day_revenue = filtered_df.groupby("Click Date")["Revenue"].sum().max()
st.markdown(f"📅 **Highest Revenue Day**: `{top_day}` with ₹{top_day_revenue:.2f}`")

# Average CTR & Conversion Rate
st.markdown(f"📊 **Average CTR**: `{avg_ctr:.2%}`")
st.markdown(f"🔁 **Average Conversion Rate**: `{avg_cvr:.2%}`")

# Anomaly Detection
st.subheader("🚨 Anomaly Alerts")
daily_revenue = filtered_df.groupby("Click Date")["Revenue"].sum()
mean_rev = daily_revenue.mean()
std_rev = daily_revenue.std()

anomalies = daily_revenue[(daily_revenue > mean_rev + 2*std_rev) | (daily_revenue < mean_rev - 2*std_rev)]

if anomalies.empty:
    st.success("✅ No revenue anomalies detected.")
else:
    st.warning("⚠️ Revenue anomalies detected on the following dates:")
    st.dataframe(anomalies.reset_index().rename(columns={"Revenue": "Anomalous Revenue"}))

# CTR Prediction Model
st.subheader("🧠 CTR Prediction Model (on Filtered Data)")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Features and target
features = ["Age", "Gender", "Income", "Location", "Ad Type", "Ad Topic", "Ad Placement"]
target = "CTR"

model_df = filtered_df[features + [target]].dropna()

if not model_df.empty:
    categorical_cols = model_df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_cats = encoder.fit_transform(model_df[categorical_cols])
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

    numeric_df = model_df.drop(columns=categorical_cols)
    X = pd.concat([numeric_df.drop(columns=[target]).reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)
    y = numeric_df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"📉 **RMSE**: `{rmse:.4f}`")
    st.markdown(f"🧮 **R² Score**: `{r2:.4f}`")

    results_df = pd.DataFrame({"Actual CTR": y_test, "Predicted CTR": y_pred})
    st.dataframe(results_df.round(4))
else:
    st.info("Not enough data for CTR prediction model.")

# Showing filtered data
st.subheader("📋 Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

# Export button(CSV)
st.download_button(
    label="📥 Download Insights as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='ad_campaign_insights.csv',
    mime='text/csv'
)
