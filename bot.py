import requests
import numpy as np
import pandas as pd
import time
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

# Initialisation des paramètres Telegram
TELEGRAM_TOKEN = os.getenv(“ TELEGRAM_TOKEN “)
CHAT_ID = os.getenv(« CHAT_ID »)
bot = Bot(token=TELEGRAM_TOKEN)

# Liste des cryptomonnaies à surveiller
CRYPTO_LIST = [« BTC-USD », « ETH-USD », « ADA-USD »]  # Utiliser les tickers de yfinance

# Fichier de suivi des performances
PERFORMANCE_LOG = « trading_performance.csv »

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

    return buy_signal

# Fonction principale pour analyser une crypto
def analyze_crypto(crypto, model):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal = analyze_signals(prices, model)
        if buy_signal:
            message = f »Buy Signal for {crypto} »
            bot.send_message(chat_id=CHAT_ID, text=message)

# Fonction principale
def main():
    # Charger des données historiques pour l’entraînement (à remplacer par vos propres données)
    data = fetch_crypto_data(« BTC-USD », « 5y »)
    # Créer des features (indicateurs techniques)
    features = calculate_indicators(data)
    # Créer des targets (signaux d’achat/vente basés sur une stratégie manuelle ou un autre modèle)
    targets = np.random.choice([0, 1], size=len(features))  # Exemple aléatoire pour l’entraînement

    # Entraîner le modèle
    model = train_ml_model(features, targets)

    while True:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda crypto: analyze_crypto(crypto, model), CRYPTO_LIST)
        time.sleep(300)

if __name__ == « __main__ »:
    main()