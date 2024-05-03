import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Generating synthetic historical data
np.random.seed(0)
data = np.random.normal(1200, 300, 100)  # Generating 100 data points
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
historical_data = pd.Series(data, index=dates)

# Set up the Streamlit interface
st.title("Enhanced Power Demand Forecasting Tool")

# Display historical data
st.write("Historical Power Demand Data:")
st.line_chart(historical_data)

# ARIMA model fitting
model = ARIMA(historical_data, order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next 30 days
forecast = model_fit.forecast(steps=30)
forecast_dates = pd.date_range(start=dates[-1], periods=31, freq="D")[1:]

# Display forecast
st.write("30-Day Power Demand Forecast:")
plt.figure(figsize=(10, 5))
plt.plot(historical_data, label="Historical Data")
plt.plot(forecast_dates, forecast, label="Forecasted Data", color="red")
plt.title("Forecast vs Historical")
plt.xlabel("Date")
plt.ylabel("Power Demand (kW)")
plt.legend()
st.pyplot(plt)
