import pandas as pd

# Load datasets
train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

# Merge train and features
data = pd.merge(train, features,
                on=["Store", "Date", "IsHoliday"],
                how="left")

# Merge with stores
data = pd.merge(data, stores,
                on="Store",
                how="left")

print("Merged Dataset")
print(data.head())

print("\nShape of Dataset:")
print(data.shape)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values with 0
data = data.fillna(0)

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Convert Date column
data["Date"] = pd.to_datetime(data["Date"])

# Group sales by date
sales = data.groupby("Date")["Weekly_Sales"].sum().reset_index()

# Rename columns for Prophet
sales = sales.rename(columns={
    "Date": "ds",
    "Weekly_Sales": "y"
})

print("\nSales Data for Prophet:")
print(sales.head())

from prophet import Prophet

model = Prophet()

model.fit(sales)

future = model.make_future_dataframe(periods=52, freq="W")

forecast = model.predict(future)

print(forecast[["ds", "yhat"]].tail())

import matplotlib.pyplot as plt

model.plot(forecast)

plt.title("Sales Forecast")

plt.show()

import matplotlib.pyplot as plt

# Merge actual and forecast
compare = sales.merge(
    forecast[['ds', 'yhat']],
    on='ds',
    how='left'
)

plt.figure(figsize=(12,6))

plt.plot(compare['ds'], compare['y'],
         label="Actual Sales")

plt.plot(compare['ds'], compare['yhat'],
         label="Forecast Sales")

plt.title("Actual vs Forecast Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(compare['y'], compare['yhat'])

print("\nMean Absolute Error:", mae)

model.plot_components(forecast)

plt.show()

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(compare["y"], compare["yhat"])

print("\nMean Absolute Error (MAE):", mae)

from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(compare["y"], compare["yhat"]))

print("Root Mean Squared Error (RMSE):", rmse)