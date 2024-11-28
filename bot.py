import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot

# Configuration des logs
logging.basicConfig(level=logging.INFO)

# Initialisation des variables d'environnement
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("Les variables TELEGRAM_TOKEN et CHAT_ID doivent être définies dans l'environnement.")

# Initialisation du bot Telegram
bot = Bot(token=TELEGRAM_TOKEN)

# Liste des cryptomonnaies à surveiller
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "ADA-USD"]  # Utiliser les tickers de yfinance

# Application Flask pour le service web
app = Flask(__name__)

# Fonction pour récupérer les données historiques avec yfinance
def fetch_crypto_data(crypto_id, period="1y"):
    logging.info(f"Téléchargement des données pour {crypto_id}")
    data = yf.download(crypto_id, period=period)
    if data.empty:
        logging.warning(f"Aucune donnée trouvée pour {crypto_id}")
        return None
    return data['Close'].values

# Fonction pour entraîner un modèle de machine learning
def train_ml_model(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Fonction pour calculer les indicateurs techniques
def calculate_indicators(prices):
    sma_short = prices[-10:].mean()
    sma_long = prices[-20:].mean()
    return [sma_short, sma_long]

# Fonction pour analyser les signaux avec le modèle ML
def analyze_signals(prices, model):
    indicators = calculate_indicators(prices)
    features = np.array(indicators).reshape(1, -1)
    prediction = model.predict(features)
    buy_signal = prediction[0] == 1
    stop_loss, take_profit = None, None  # À calculer dynamiquement
    return buy_signal, stop_loss, take_profit

# Fonction principale pour analyser une crypto
def analyze_crypto(crypto, model):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        try:
            buy_signal, stop_loss, take_profit = analyze_signals(prices, model)
            if buy_signal:
                message = f"Signal d'achat détecté pour {crypto} ! 🚀\nStop Loss : {stop_loss}\nTake Profit : {take_profit}"
                bot.send_message(chat_id=CHAT_ID, text=message)
                logging.info(f"Message envoyé : {message}")
        except Exception as e:
            logging.error(f"Erreur dans l'analyse pour {crypto} : {e}")

# Route Flask pour déclencher l'analyse
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Charger les données et entraîner le modèle
        data = fetch_crypto_data("BTC-USD", "5y")
        features = calculate_indicators(data)
        targets = np.random.randint(0, 2, len(features))  # Exemple de données cibles
        model = train_ml_model(features, targets)

        # Analyser les cryptomonnaies
        with ThreadPoolExecutor() as executor:
            executor.map(lambda crypto: analyze_crypto(crypto, model), CRYPTO_LIST)

        return {"status": "success", "message": "Analyse terminée"}, 200
    except Exception as e:
        logging.error(f"Erreur dans la route /analyze : {e}")
        return {"status": "error", "message": str(e)}, 500

# Démarrage de l'application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)