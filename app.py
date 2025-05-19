import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Page setup
st.set_page_config(page_title="📊 Sales Forecast App", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1abc9c;
        margin-bottom: 20px;
    }
    .footer {
        font-size: 14px;
        color: gray;
        text-align: center;
        margin-top: 50px;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .section {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
    }
    .subheader {
        color: #34495e;
        font-size: 24px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📈 Sales Forecasting</div>', unsafe_allow_html=True)
st.markdown("Upload your CSV and forecast sales trends using Prophet.")

# File upload
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="subheader">📂 Upload Your Sales Data</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("CSV file with 'ds' (date) and 'y' (sales)", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ds' in df.columns and 'y' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        st.success("✅ Data loaded successfully!")

        with st.expander("📋 Preview Data"):
            st.dataframe(df.head())

        # Forecast settings
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="subheader">⚙️ Forecast Settings</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        periods = col1.number_input("📅 Days to Forecast", 7, 365, 30, step=7)
        yearly = col2.checkbox("📆 Yearly Seasonality", True)
        weekly = col2.checkbox("📈 Weekly Seasonality", True)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("⏳ Training model..."):
            model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly)
            model.fit(df)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

        # Forecast Plot
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="subheader">📊 Forecast Visualization</div>', unsafe_allow_html=True)

        forecast_future = forecast[forecast['ds'] > df['ds'].max()]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['ds'], df['y'], label='Actual Sales', color='#2c3e50')
        ax.plot(forecast_future['ds'], forecast_future['yhat'], label='Forecast', color='#27ae60')
        ax.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'],
                        color='#2ecc71', alpha=0.3, label='Confidence Interval')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title("Sales Forecast")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Trend & Seasonality
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="subheader">📉 Trend & Seasonality</div>', unsafe_allow_html=True)
        st.pyplot(model.plot_components(forecast))
        st.markdown('</div>', unsafe_allow_html=True)

        # Forecast Table
        with st.expander("📁 Forecast Table"):
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

    else:
        st.error("❌ Your file must contain 'ds' and 'y' columns.")
else:
    st.info("🗂️ Upload a CSV file to begin.")

# Footer
st.markdown(f"""
    <div class="footer">
        Created by <strong>KHIZAR BSE-22S-057</strong> | Streamlit Sales Forecast App
    </div>
""", unsafe_allow_html=True)
