# 1. Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    """ Load data from a CSV file and prepare it for training. """
    # Load data
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Feature engineering
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df['MonthNumeric'] = df['Month'].dt.year * 100 + df['Month'].dt.month
    df['Year'] = df['Date'].dt.year
    df['MonthOfYear'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Lag features
    df['Lag1'] = df['Sales'].shift(1)
    df['Lag2'] = df['Sales'].shift(2)
    df['Lag3'] = df['Sales'].shift(3)

    # Rolling mean
    df['RollingMean3'] = df['Sales'].rolling(window=3).mean()

    # Drop NaN values
    df.dropna(inplace=True)
    return df

def train_model(df):
    """ Train the model using the prepared data. """
    # Features (X) and Target (Y)
    X = df[['MonthNumeric', 'Year', 'MonthOfYear', 'DayOfYear', 'Lag1', 'Lag2', 'Lag3', 'RollingMean3']]
    Y = df['Sales']

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

    # Model Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, Y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    metrics = {
        'mse': mean_squared_error(Y_test, y_pred),
        'mae': mean_absolute_error(Y_test, y_pred),
        'r2': r2_score(Y_test, y_pred)
    }

    # Save the predictions
    results_df = X_test.copy()
    results_df['ActualSales'] = Y_test.values
    results_df['PredictedSales'] = y_pred
    results_df['Date'] = df.loc[X_test.index, 'Date']
    results_df = results_df[['Date', 'ActualSales', 'PredictedSales']]
    
    return pipeline, results_df, metrics

def save_predictions(results_df, file_path):
    """ Save predictions to a CSV file. """
    results_df.to_csv(file_path, index=False)

def plot_predictions(results_df, save_path):
    """ Plot predictions and save the graph as an image. """
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Date'], results_df['ActualSales'], color='red', label="Actual Sales")
    plt.plot(results_df['Date'], results_df['PredictedSales'], color='blue', linewidth=2, label="Predicted Sales")
    plt.fill_between(results_df['Date'], results_df['PredictedSales'] - np.std(results_df['ActualSales']), 
                     results_df['PredictedSales'] + np.std(results_df['ActualSales']), color='blue', alpha=0.2, label="Prediction Confidence")
    plt.xlabel("Month (YYYY-MM)")
    plt.ylabel("Sales")
    plt.title('Sales Forecast with Random Forest')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
