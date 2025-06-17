# store-sales-prophet-forecasting

Time series forecasting of grocery sales using Prophet on Kaggle's Store Sales dataset. Features regressors and seasonality for supply chain analytics.
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
from prophet.diagnostics import cross_validation

%matplotlib inline



# Load Kaggle dataset files
    train = pd.read_csv('train.csv')
    stores = pd.read_csv('stores.csv')
    holidays = pd.read_csv('holidays_events.csv')
    oil = pd.read_csv('oil.csv')
    transactions = pd.read_csv('transactions.csv')


# Rename 'type' columns to avoid conflicts
stores = stores.rename(columns={'type': 'store_type'})
holidays = holidays.rename(columns={'type': 'holiday_type'})


# Ensure date columns are in datetime format
    train['date'] = pd.to_datetime(train['date'])
    oil['date'] = pd.to_datetime(oil['date'])
    holidays['date'] = pd.to_datetime(holidays['date'])
    transactions['date'] = pd.to_datetime(transactions['date'])

# Merge datasets
data = train.merge(stores, on='store_nbr', how='left')
print("Columns after train-stores merge:", data.columns.tolist())
if 'date' not in data.columns or 'date' not in oil.columns:
    print("Error: 'date' column missing in data or oil DataFrame.")
    exit(1)
data = data.merge(oil, on='date', how='left')
data = data.merge(transactions, on=['date', 'store_nbr'], how='left')
data = data.merge(holidays, on='date', how='left')

# Preprocess data
data['dcoilwtico'] = data['dcoilwtico'].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')  # Handle missing oil prices
data['transactions'] = data['transactions'].fillna(data['transactions'].mean())  # Fill missing transactions
# Create holiday indicator
if 'holiday_type' in data.columns:
    data['holiday'] = data['holiday_type'].notnull().astype(int)
else:
    print("Warning: 'holiday_type' column missing, setting holiday to 0")
    data['holiday'] = 0
data['onpromotion'] = data['onpromotion'].fillna(0)  # Ensure no missing promotions


# Select a single store-item combination for demonstration (Store 1, BEVERAGES)
prophet_data = data[(data['store_nbr'] == 1) & (data['family'] == 'BEVERAGES')].copy()
prophet_data = prophet_data[['date', 'sales', 'onpromotion', 'dcoilwtico', 'holiday', 'transactions']].rename(columns={'date': 'ds', 'sales': 'y'})


# Feature engineering
prophet_data['month'] = prophet_data['ds'].dt.month  # Add month as a regressor
prophet_data['day_of_week'] = prophet_data['ds'].dt.dayofweek  # Add day of week
prophet_data['lag1'] = prophet_data['y'].shift(1)  # Lagged sales (1 day)
prophet_data = prophet_data.dropna()  # Remove rows with NaN


model = Prophet(
    growth='linear',
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)


# Add regressors
regressors = ['onpromotion', 'dcoilwtico', 'holiday', 'transactions', 'month', 'day_of_week', 'lag1']
for regressor in regressors:
    model.add_regressor(regressor, mode='additive')


# Fit model
model.fit(prophet_data)


# Function to calculate RMSSE
def calculate_rmsse(actual, predicted, train, horizon):
    """
    Calculate RMSSE for a single time series.
    actual: Array of actual values in forecast period
    predicted: Array of predicted values
    train: Array of training values (historical sales)
    horizon: Forecast horizon (number of periods)
    """
    # Mean squared error of forecast
    mse = np.mean((actual - predicted) ** 2)
    
    # Mean squared error of naive forecast (using previous day's sales)
    naive_mse = np.mean((train[1:] - train[:-1]) ** 2) if len(train) > 1 else 1e-10  # Avoid division by zero
    
    # RMSSE
    rmsse = np.sqrt(mse / naive_mse) if naive_mse != 0 else np.inf
    return rmsse


#using prophet built-in cross validation

cv_results = cross_validation(
    model,
    initial='1095 days',
    period='180 days',
    horizon='16 days',
    parallel="processes"
)

print(cv_results)

rmsse_scores = []
for cutoff in cv_results['cutoff'].unique():
    cv_subset = cv_results[cv_results['cutoff'] == cutoff]
    actual = cv_subset['y'].values
    predicted = cv_subset['yhat'].values
    # Use training data up to cutoff for naive forecast denominator
    train_subset = prophet_data[prophet_data['ds'] <= cutoff]['y'].values
    rmsse = calculate_rmsse(actual, predicted, train_subset, horizon=len(actual))
    rmsse_scores.append(rmsse)

# Average RMSSE
mean_rmsse = np.mean(rmsse_scores)
print(f"Mean RMSSE: {mean_rmsse:.4f}")


# Create future dataframe for forecasting (30 days ahead)
future = model.make_future_dataframe(periods=30, freq='D')
future = future.merge(prophet_data[['ds', 'onpromotion', 'dcoilwtico', 'holiday', 'transactions', 'month', 'day_of_week', 'lag1']], on='ds', how='left')


# Fill future regressors
future['onpromotion'] = future['onpromotion'].fillna(prophet_data['onpromotion'].mean())
future['dcoilwtico'] = future['dcoilwtico'].fillna(prophet_data['dcoilwtico'].mean())
future['holiday'] = future['holiday'].fillna(0)
future['transactions'] = future['transactions'].fillna(prophet_data['transactions'].mean())
future['month'] = future['ds'].dt.month
future['day_of_week'] = future['ds'].dt.dayofweek
future['lag1'] = future['lag1'].fillna(prophet_data['y'].iloc[-1])




# Generate forecast
forecast = model.predict(future)

print(forecast)










