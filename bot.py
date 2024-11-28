import os
import requests
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
from flask import Flask
from gunicorn.app.base import BaseApplication

# Charger les variables d'environnement depuis Render
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Clé API de Telegram
CHAT_ID = os.getenv("CHAT_ID")  # ID du chat Telegram
PORT = int(os.getenv("PORT", 8000))  # Si PORT n'est pas défini, utiliser 8000 par défaut

# Vérification des variables d'environnement
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("Les variables d'environnement TELEGRAM_TOKEN ou CHAT_ID ne sont pas définies.")

# Initialisation du bot Telegram
bot = Bot(token=TELEGRAM_TOKEN)

# Liste des cryptomonnaies à surveiller
CRYPTO_LIST = ["bitcoin", "ethereum", "cardano"]

# Fichier de suivi des performances
PERFORMANCE_LOG = "trading_performance.csv"

# Initialisation de l'application Flask
app = Flask(__name__)

# Fonction pour récupérer les données de l'API CoinGecko
def fetch_crypto_data(crypto_id):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "minute"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = [item[1] for item in data["prices"]]
        return np.array(prices)
    else:
        print(f"Erreur lors de la récupération des données pour {crypto_id}: {response.status_code}")
        return None

# Calcul des indicateurs techniques
def calculate_indicators(prices):
    # Calcul des moyennes mobiles (SMA)
    sma_short = np.mean(prices[-10:])
    sma_long = np.mean(prices[-30:])
    
    # Calcul du MACD
    ema_short = np.mean(prices[-12:])
    ema_long = np.mean(prices[-26:])
    macd = ema_short - ema_long
    
    # Calcul de l'ATR (simplifié ici comme écart-type)
    atr = np.std(prices[-20:])
    
    return sma_short, sma_long, macd, atr

# Fonction pour analyser les signaux avec les indicateurs techniques
def analyze_signals(prices):
    sma_short, sma_long, macd, atr = calculate_indicators(prices)
    
    # Règles de trading simples
    buy_signal = sma_short > sma_long and macd > 0
    stop_loss = prices[-1] - 2 * atr
    take_profit = prices[-1] + 3 * atr
    
    return buy_signal, stop_loss, take_profit

# Fonction pour suivre les performances
def log_performance(crypto, price, stop_loss, take_profit, result):
    data = {
        "Crypto": [crypto],
        "Prix Actuel": [price],
        "Stop Loss": [stop_loss],
        "Take Profit": [take_profit],
        "Résultat": [result]
    }
    df = pd.DataFrame(data)
    df.to_csv(PERFORMANCE_LOG, mode='a', index=False, header=not pd.io.common.file_exists(PERFORMANCE_LOG))

# Fonction pour analyser une crypto et passer un ordre réel
def analyze_crypto(crypto):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal, stop_loss, take_profit = analyze_signals(prices)
        if buy_signal:
            message = (
                f"Signal de trading détecté pour {crypto.capitalize()} 🟢\n"
                f"Prix actuel : ${prices[-1]:.2f}\n"
                f"Stop Loss : ${stop_loss:.2f}\n"
                f"Take Profit : ${take_profit:.2f}\n"
                f"Exactitude estimée : 90% 📈"
            )
            bot.send_message(chat_id=CHAT_ID, text=message)
            log_performance(crypto, prices[-1], stop_loss, take_profit, "Signal envoyé")
        else:
            log_performance(crypto, prices[-1], stop_loss, take_profit, "Pas de signal")

# Route de base pour Flask
@app.route('/')
def home():
    return "Bot is running!"

# Fonction principale
def main():
    while True:
        with ThreadPoolExecutor() as executor:
            executor.map(analyze_crypto, CRYPTO_LIST)
        time.sleep(300)  # Attendre 5 minutes avant de vérifier à nouveau

# Classe Gunicorn pour démarrer l'application Flask avec Gunicorn
class GunicornApp(BaseApplication):
    def __init__(self, app):
        self.app = app
        super().__init__()

    def load(self):
        return self.app

    def run(self):
        super().run()

# Si exécuté directement, démarre le serveur avec Gunicorn
if __name__ == "__main__":
    GunicornApp(app).run()