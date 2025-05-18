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
    .title { text-align: center; font-size: 36px; font-weight: bold; color: #2c3e50; }
    .footer { font-size: 14px; color: gray; text-align: center; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📈 Seasonal Sales Forecasting</div>', unsafe_allow_html=True)
st.markdown("Upload your CSV and forecast sales trends using Prophet.")

# File upload
st.subheader("📂 Upload Your Sales Data")
uploaded_file = st.file_uploader("CSV file with 'ds' (date) and 'y' (sales)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ds' in df.columns and 'y' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        st.success("✅ Data loaded successfully!")

        with st.expander("📋 Preview Data"):
            st.write(df.head())

        # Forecast settings
        st.subheader("⚙️ Forecast Settings")
        col1, col2 = st.columns(2)
        periods = col1.number_input("Days to Forecast", 7, 365, 30, step=7)
        yearly = col2.checkbox("Yearly Seasonality", True)
        weekly = col2.checkbox("Weekly Seasonality", True)

        with st.spinner("Training model..."):
            model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly)
            model.fit(df)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

        # Forecast Plot
        st.subheader("📊 Forecast Visualization")
        forecast_future = forecast[forecast['ds'] > df['ds'].max()]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['ds'], df['y'], label='Actual Sales', color='black')
        ax.plot(forecast_future['ds'], forecast_future['yhat'], label='Forecast', color='green')
        ax.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'],
                        color='green', alpha=0.3, label='Confidence Interval')
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title("Sales Forecast")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Trend & Seasonality
        st.subheader("📉 Trend & Seasonality")
        st.pyplot(model.plot_components(forecast))

        # Forecast Table
        with st.expander("📁 Forecast Table"):
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
    else:
        st.error("❌ Your file must contain 'ds' and 'y' columns.")
else:
    st.info("Upload a CSV file to begin.")

# Footer
st.markdown(f"""
    <div class="footer">
        Built with ❤️ using Streamlit & Prophet • Last updated: {datetime.today().strftime('%Y-%m-%d')}
    </div>
""", unsafe_allow_html=True)
