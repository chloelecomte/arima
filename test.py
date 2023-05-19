import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('ARIMA Model\Data_for_ARIMA_Model.csv')

# Assuming "Demand", "Year", and "Month" are the column names
demand = data['Demand']
year = data['Year']
month = data['Month']

# Combine Year and Month columns into a single date format
dates = pd.to_datetime(year.astype(str) + '-' + month.astype(str), format='%Y-%m')

# Fit ARIMA model
model = ARIMA(demand, order=(1, 1, 1))  # Adjust the order as per your requirement
model_fit = model.fit()

# Forecast future month demand
forecast_steps = 12  # Change this to the desired number of forecasted months
forecast = model_fit.forecast(steps=forecast_steps)

# Plot historical demand and forecast
plt.plot(dates, demand, color='blue', linestyle='dashed', label='Historical Demand')
plt.plot(pd.date_range(start=dates.iloc[-1], periods=forecast_steps, freq='M'), forecast, color='red', label='Forecasted Demand')
plt.xlabel('Month')
plt.ylabel('Demand')
plt.legend()
plt.show()