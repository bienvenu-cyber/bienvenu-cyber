import requests
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
from flask import Flask
import os
import threading

# Initialisation des paramètres Telegram
TELEGRAM_TOKEN = os.getenv(« TELEGRAM_TOKEN »)
CHAT_ID = os.getenv(« CHAT_ID »)
bot = Bot(token=TELEGRAM_TOKEN)

# Liste des cryptomonnaies à surveiller
CRYPTO_LIST = [« BTC-USD », « ETH-USD », « ADA-USD »]  # Utiliser les tickers de yfinance

# Fichier de suivi des performances
PERFORMANCE_LOG = « trading_performance.csv »

# Création de l’application Flask
app = Flask(__name__)

@app.route(‘/‘)
def home():
    return « Service is running! »

# Fonction pour récupérer les données historiques avec yfinance
def fetch_crypto_data(crypto_id, period=« 1y »):
    data = yf.download(crypto_id, period=period)
    return data[‘Close’].values

# Fonction pour entraîner un modèle de machine learning (à améliorer)
def train_ml_model(data, target):
    # Division des données en ensemble d’entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Modèle de régression logistique (à remplacer par un modèle plus complexe si nécessaire)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

# Fonction pour calculer les indicateurs techniques
def calculate_indicators(prices):
    # Calculer des indicateurs plus complets (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
    # Exemple simple :
    sma_short = prices[-10:].mean()
    sma_long = prices[-20:].mean()
    return sma_short, sma_long

# Fonction pour analyser les signaux avec le modèle ML
def analyze_signals(prices, model):
    # Calculer les indicateurs
    indicators = calculate_indicators(prices)

    # Préparer les données pour le modèle
    features = np.array(indicators).reshape(1, -1)
    prediction = model.predict(features)

    # Signal basé sur le modèle ML
    buy_signal = prediction[0] == 1

    # Calculer stop-loss et take-profit dynamiques (basés sur des indicateurs techniques)
    stop_loss = 0.95  # Exemple (à améliorer)
    take_profit = 1.05  # Exemple (à améliorer)

    return buy_signal, stop_loss, take_profit

# Fonction pour analyser une crypto
def analyze_crypto(crypto, model):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal, stop_loss, take_profit = analyze_signals(prices, model)
        if buy_signal:
            bot.send_message(CHAT_ID, f »Buy signal detected for {crypto}!\nStop Loss: {stop_loss}\nTake Profit: {take_profit} »)
        else:
            bot.send_message(CHAT_ID, f »No signal detected for {crypto} »)

# Fonction de logique de trading
def trading_logic():
    while True:
        # Charger des données historiques pour l’entraînement (à remplacer par vos propres données)
        data = fetch_crypto_data(« BTC-USD », « 5y »)
        # Créer des features (indicateurs techniques)
        features = calculate_indicators(data)
        # Créer des targets (signaux d’achat/vente basés sur une stratégie manuelle ou un autre modèle)
        targets = np.random.randint(0, 2, len(features))  # Remplacer par des données réelles pour l’entraînement

        # Entraîner le modèle
        model = train_ml_model(features, targets)

        # Analyser les cryptos
        for crypto in CRYPTO_LIST:
            analyze_crypto(crypto, model)

        time.sleep(300)  # Attendre 5 minutes avant de vérifier à nouveau

# Lancer la logique de trading dans un thread séparé
def start_trading():
    trading_thread = threading.Thread(target=trading_logic)
    trading_thread.daemon = True
    trading_thread.start()

# Démarrer Flask et la logique de trading en parallèle
if __name__ == « __main__ »:
    start_trading()  # Lancer la logique de trading en parallèle
    app.run(host=« 0.0.0.0 », port=10000)  # Démarrer Flask