import os
import requests
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from telegram.error import TelegramError

# Initialize Telegram parameters
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  
CHAT_ID = os.getenv("CHAT_ID") 
bot = Bot(token=TELEGRAM_TOKEN)

# Fetch cryptocurrency data
def fetch_crypto_data(ticker, period):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=0&period2=9999999999&interval=1d&events=history"
    response = requests.get(url)
    data = []
    if response.status_code == 200:
        lines = response.text.split("\n")[1:]
        for line in lines:
            try:
                close_price = float(line.split(",")[4])
                data.append(close_price)
            except (ValueError, IndexError):
                continue
    return np.array(data)

# Calculate indicators
def calculate_indicators(prices):
    if len(prices) < 20:
        raise ValueError("Not enough data to calculate indicators.")
    sma_short = prices[-10:].mean()
    sma_long = prices[-20:].mean()
    return np.array([sma_short, sma_long])

# Train machine learning model
def train_ml_model(data, target):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1) 
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Send alert to Telegram
def send_alert(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
    except TelegramError as e:
        print(f"Failed to send message: {e}")

# Main logic
def main():
    try:
        # Load historical data
        data = fetch_crypto_data("BTC-USD", "5y")
        features = np.array([calculate_indicators(data)])  # Ensure features are 2D
        features = features.reshape(-1, 2)  # Reshape for model compatibility
        targets = np.random.choice([0, 1], size=(features.shape[0],))  # Dummy target data for example
        
        # Train model
        model = train_ml_model(features, targets)

        # Predict with the model
        prediction = model.predict(features[-1].reshape(1, -1))[0]
        message = f"Prediction: {'Buy' if prediction == 1 else 'Sell'}"
        
        # Send prediction to Telegram
        send_alert(message)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    main()