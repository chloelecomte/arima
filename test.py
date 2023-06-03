import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('ARIMA Model/Data_for_ARIMA_Model.csv')

# Assuming "Demand", "Year", and "Month" are the column names
demand = data['Demand']
year = data['Year']
month = data['Month']

# Combine Year and Month columns into a single date format
dates = pd.to_datetime(year.astype(str) + '-' + month.astype(str), format='%Y-%m')

# function tha fits ARIMA model and forecasts future demand with 80% and 90% confidence interval
def arimaModel():
    # Fit the ARIMA model
    arima = ARIMA(demand, order=(5, 1, 0))
    model = arima.fit()

    # Forecast future demand
    forecast_steps = 12
    forecast = model.forecast(steps=forecast_steps)
    conf_int = model.get_forecast(steps=forecast_steps).conf_int(alpha=0.2)

    # Create a dataframe with the forecasted demand and confidence interval
    forecast_df = pd.DataFrame({'Forecasted Demand': forecast, 'Lower Confidence Interval': conf_int['lower Demand'], 'Upper Confidence Interval': conf_int['upper Demand']})

    # Add the dates to the dataframe
    forecast_df['Date'] = dates[60:]
    forecast_df.set_index('Date', inplace=True)

    # Plot the forecasted demand and confidence interval
    plt.figure(figsize=(10, 5))
    plt.plot(dates, demand, color='blue', linestyle='dashed', label='Historical Demand')
    plt.plot(pd.date_range(start=dates.iloc[-1], periods=forecast_steps, freq='M'), forecast_df['Forecasted Demand'], color='blue', label='Forecasted Demand')
    plt.plot(pd.date_range(start=dates.iloc[-1], periods=forecast_steps, freq='M'), forecast_df['Lower Confidence Interval'], color='red', label='Lower Confidence Interval')
    plt.plot(pd.date_range(start=dates.iloc[-1], periods=forecast_steps, freq='M'), forecast_df['Upper Confidence Interval'], color='green', label='Upper Confidence Interval')
    plt.xlabel('Month')
    plt.ylabel('Demand')
    plt.legend(loc='upper left')
    plt.show()

arimaModel()
