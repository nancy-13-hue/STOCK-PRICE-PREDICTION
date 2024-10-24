# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fetch stock data using yfinance
def get_stock_data(ticker):
    stock = yf.download(ticker, start="2010-01-01", end="2024-01-01")
    return stock

# Prepare data by adding features and labels for machine learning
def prepare_data(stock_df):
    stock_df['Prediction'] = stock_df[['Close']].shift(-30)  # Predicting the next 30 days

    # Features (closing prices)
    X = np.array(stock_df[['Close']])
    X = X[:-30]  # Remove the last 30 rows

    # Labels (predicted closing prices)
    y = np.array(stock_df['Prediction'])
    y = y[:-30]  # Remove the last 30 rows
    
    return X, y

# Train model and make predictions
def train_and_predict(X, y, stock_df):
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the Linear Regression model
    lr_model = LinearRegression()

    # Train the model
    lr_model.fit(X_train, y_train)

    # Test the model
    predictions = lr_model.predict(X_test)

    # Calculate model performance
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Predict the next 30 days
    next_30_days = stock_df[['Close']][-30:].to_numpy()
    future_prediction = lr_model.predict(next_30_days)
    
    return future_prediction

# Plot stock prices and predictions
def plot_predictions(stock_df, future_prediction):
    plt.figure(figsize=(12, 6))
    plt.title('Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Close Price USD ($)')
    plt.plot(stock_df['Close'], label='Historical Prices')
    
    # Plot future prediction
    future_dates = pd.date_range(stock_df.index[-1], periods=31, closed='right')
    plt.plot(future_dates, future_prediction, label='Predicted Prices', color='red')
    
    plt.legend()
    plt.show()

# Main function to execute the steps
def main():
    ticker = 'AAPL'  # Apple stock
    stock_data = get_stock_data(ticker)
    
    # Prepare data for prediction
    X, y = prepare_data(stock_data)
    
    # Train model and make predictions
    future_prediction = train_and_predict(X, y, stock_data)
    
    # Plot results
    plot_predictions(stock_data, future_prediction)

# Run the stock price prediction
if __name__ == "__main__":
    main()
