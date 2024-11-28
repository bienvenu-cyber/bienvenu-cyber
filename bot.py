import requests
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import os

# Initialize Telegram parameters
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# List of cryptocurrencies to monitor
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "ADA-USD"]  

# Performance tracking file
PERFORMANCE_LOG = "trading_performance.csv"

# Function to fetch historical data using yfinance
def fetch_crypto_data(crypto_id, period="1y"):
    data = yf.download(crypto_id, period=period)
    return data['Close'].values

# Function to train a machine learning model (can be improved)
def train_ml_model(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Function to calculate technical indicators
def calculate_indicators(prices):
    sma_short = prices[-10:].mean()
    sma_long = prices[-20:].mean()
    return sma_short, sma_long

# Function to analyze signals with the ML model
def analyze_signals(prices, model):
    indicators = calculate_indicators(prices)
    features = np.array(indicators).reshape(1, -1)
    prediction = model.predict(features)
    buy_signal = prediction[0] == 1
    stop_loss = 0  # Example value
    take_profit = 0  # Example value
    return buy_signal, stop_loss, take_profit

# Function to analyze a cryptocurrency
def analyze_crypto(crypto, model):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal, stop_loss, take_profit = analyze_signals(prices, model)
        if buy_signal:
            bot.send_message(
                CHAT_ID,
                f"Recommended to buy {crypto} with Stop-Loss: {stop_loss}, Take-Profit: {take_profit}",
            )

# Main function
def main():
    data = fetch_crypto_data("BTC-USD", "5y")
    features = calculate_indicators(data)
    targets = np.random.choice([0, 1], size=len(features))  # Example targets
    model = train_ml_model(features, targets)

    while True:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda crypto: analyze_crypto(crypto, model), CRYPTO_LIST)
        time.sleep(300)  # Wait 5 minutes before the next analysis

if __name__ == "__main__":
    main()