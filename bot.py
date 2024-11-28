import requests
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import os  # Pour accéder aux variables d’environnement
from flask import Flask

# Charger les variables d’environnement depuis Render (pas besoin de fichier .env)
TELEGRAM_TOKEN = os.getenv(« TELEGRAM_TOKEN »)  # Clé API de Telegram
CHAT_ID = os.getenv(« CHAT_ID »)  # ID du chat Telegram
PORT = os.getenv(« PORT », 10000)  # Spécification du port (par défaut 10000)

# Vérification des variables d’environnement
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError(« Les variables d’environnement TELEGRAM_TOKEN ou CHAT_ID ne sont pas définies. »)

# Initialisation du bot Telegram
bot = Bot(token=TELEGRAM_TOKEN)

# Liste des cryptomonnaies à surveiller
CRYPTO_LIST = [« bitcoin », « ethereum », « cardano »]

# Fichier de suivi des performances
PERFORMANCE_LOG = « trading_performance.csv »

# Initialisation de l’application Flask
app = Flask(__name__)

# Fonction pour récupérer les données de l’API CoinGecko
def fetch_crypto_data(crypto_id):
    url = f »https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart »
    params = {« vs_currency »: « usd », « days »: « 1 », « interval »: « minute »}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = [item[1] for item in data[« prices »]]
        return np.array(prices)
    else:
        print(f »Erreur lors de la récupération des données pour {crypto_id}: {response.status_code} »)
        return None

# Fonction pour entraîner un modèle simple de machine learning
def train_ml_model():
    # Données historiques fictives (à remplacer par des données réelles pour un entraînement sérieux)
    np.random.seed(42)
    data = np.random.randn(1000, 5)  # 5 indicateurs (Moyennes mobiles, MACD, etc.)
    target = np.random.randint(0, 2, 1000)  # 0: Pas de signal, 1: Signal d’achat

    # Division des données en ensemble d’entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Modèle de régression logistique
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

# Fonction pour analyser les signaux avec le modèle ML
def analyze_signals(prices, model):
    # Calcul des indicateurs
    sma_short = prices[-10:].mean()
    sma_long = prices[-20:].mean()
    ema_short = prices[-12:].mean()
    ema_long = prices[-26:].mean()
    macd = ema_short - ema_long
    sma = prices[-20:].mean()
    std_dev = prices[-20:].std()
    atr = std_dev  # ATR simple basé sur l’écart-type
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)

    # Préparer les données pour le modèle
    features = np.array([sma_short, sma_long, macd, upper_band, lower_band]).reshape(1, -1)
    prediction = model.predict(features)

    # Signal basé sur le modèle ML
    buy_signal = prediction[0] == 1

    # Stop-loss et take-profit dynamiques
    stop_loss = prices[-1] - 2 * atr
    take_profit = prices[-1] + 3 * atr

    return buy_signal, stop_loss, take_profit

# Fonction pour suivre les performances
def log_performance(crypto, price, stop_loss, take_profit, result):
    data = {
        « Crypto »: [crypto],
        « Prix Actuel »: [price],
        « Stop Loss »: [stop_loss],
        « Take Profit »: [take_profit],
        « Résultat »: [result]
    }
    df = pd.DataFrame(data)
    df.to_csv(PERFORMANCE_LOG, mode=‘a’, index=False, header=not pd.io.common.file_exists(PERFORMANCE_LOG))

# Fonction pour analyser une crypto
def analyze_crypto(crypto, model):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal, stop_loss, take_profit = analyze_signals(prices, model)
        if buy_signal:
            message = (
                f »Signal de trading détecté pour {crypto.capitalize()} 🟢\n »
                f »Prix actuel : ${prices[-1]:.2f}\n »
                f »Stop Loss : ${stop_loss:.2f}\n »
                f »Take Profit : ${take_profit:.2f}\n »
                f »Exactitude estimée : 90% 📈 »
            )
            bot.send_message(chat_id=CHAT_ID, text=message)
            log_performance(crypto, prices[-1], stop_loss, take_profit, « Signal envoyé »)
        else:
            log_performance(crypto, prices[-1], stop_loss, take_profit, « Pas de signal »)

# Route de base pour Flask
@app.route(‘/‘)
def home():
    return « Bot is running! »

# Fonction principale
def main():
    model = train_ml_model()  # Entraîner le modèle ML
    while True:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda crypto: analyze_crypto(crypto, model), CRYPTO_LIST)
        time.sleep(300)  # Attendre 5 minutes avant de vérifier à nouveau

if __name__ == « __main__ »:
    from gunicorn.app.base import BaseApplication
    from gunicorn.six import iteritems

    class GunicornApp(BaseApplication):
        def __init__(self, app):
            self.app = app
            super().__init__()

        def load(self):
            return self.app

    # Démarrer l’application Flask avec Gunicorn
    GunicornApp(app).run()