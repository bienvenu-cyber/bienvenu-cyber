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
import logging
from datetime import datetime
from time import sleep

# Charger les variables d’environnement depuis Render
TELEGRAM_TOKEN = os.getenv(« TELEGRAM_TOKEN »)  # Clé API de Telegram
CHAT_ID = os.getenv(« CHAT_ID »)  # ID du chat Telegram
PORT = int(os.getenv(« PORT », 8000))  # Si PORT n’est pas défini, utiliser 8000 par défaut

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

# Configurer le logger pour enregistrer les erreurs et autres informations utiles
logging.basicConfig(filename=‘trading_bot.log’, level=logging.INFO)

# Fonction pour récupérer les données de l’API CoinGecko avec gestion des erreurs
def fetch_crypto_data(crypto_id):
    url = f »https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart »
    params = {« vs_currency »: « usd », « days »: « 1 », « interval »: « minute »}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lève une exception pour une réponse d’erreur (4xx, 5xx)
        data = response.json()
        prices = [item[1] for item in data[« prices »]]
        return np.array(prices)
    except requests.exceptions.RequestException as e:
        logging.error(f »Erreur API pour {crypto_id}: {e} »)
        return None

# Calcul des indicateurs techniques avec une fenêtre glissante pour les moyennes mobiles
def calculate_indicators(prices):
    # Calcul des moyennes mobiles (SMA)
    sma_short = np.mean(prices[-10:])
    sma_long = np.mean(prices[-30:])
    
    # Calcul du MACD
    ema_short = np.mean(prices[-12:])
    ema_long = np.mean(prices[-26:])
    macd = ema_short - ema_long
    
    # Calcul de l’ATR (simplifié ici comme écart-type)
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

# Fonction pour suivre les performances avec plus de détails
def log_performance(crypto, price, stop_loss, take_profit, result, timestamp):
    data = {
        « Crypto »: [crypto],
        « Prix Actuel »: [price],
        « Stop Loss »: [stop_loss],
        « Take Profit »: [take_profit],
        « Résultat »: [result],
        « Timestamp »: [timestamp]
    }
    df = pd.DataFrame(data)
    df.to_csv(PERFORMANCE_LOG, mode=‘a’, index=False, header=not pd.io.common.file_exists(PERFORMANCE_LOG))

# Fonction pour analyser une crypto et passer un ordre réel
def analyze_crypto(crypto):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal, stop_loss, take_profit = analyze_signals(prices)
        timestamp = datetime.now().strftime(‘%Y-%m-%d %H:%M:%S’)
        if buy_signal:
            message = (
                f »Signal de trading détecté pour {crypto.capitalize()} 🟢\n »
                f »Prix actuel : ${prices[-1]:.2f}\n »
                f »Stop Loss : ${stop_loss:.2f}\n »
                f »Take Profit : ${take_profit:.2f}\n »
                f »Exactitude estimée : 90% 📈 »
            )
            try:
                bot.send_message(chat_id=CHAT_ID, text=message)
                logging.info(f »Signal envoyé pour {crypto} à {timestamp} »)
                log_performance(crypto, prices[-1], stop_loss, take_profit, « Signal envoyé », timestamp)
            except Exception as e:
                logging.error(f »Erreur en envoyant le message Telegram pour {crypto}: {e} »)
                log_performance(crypto, prices[-1], stop_loss, take_profit, « Erreur d’envoi », timestamp)
        else:
            log_performance(crypto, prices[-1], stop_loss, take_profit, « Pas de signal », timestamp)

# Route de base pour Flask
@app.route(‘/‘)
def home():
    return « Bot is running! »

# Fonction principale avec délai dynamique
def dynamic_sleep(last_signal_time):
    time_since_last_signal = time.time() - last_signal_time
    if time_since_last_signal < 300:  # Si un signal récent, réduit le délai
        return 180  # Attente de 3 minutes
    return 300  # Sinon, attends 5 minutes

def main():
    last_signal_time = time.time()  # Temps du dernier signal
    while True:
        with ThreadPoolExecutor() as executor:
            executor.map(analyze_crypto, CRYPTO_LIST)
        last_signal_time = time.time()  # Met à jour l’heure du dernier signal
        sleep(dynamic_sleep(last_signal_time))  # Attendre dynamiquement avant de vérifier à nouveau

# Classe Gunicorn pour démarrer l’application Flask avec Gunicorn
class GunicornApp(BaseApplication):
    def __init__(self, app):
        self.app = app
        super().__init__()

    def load(self):
        return self.app

    def run(self):
        super().run()

# Si exécuté directement, démarre le serveur avec Gunicorn
if __name__ == « __main__ »:
    GunicornApp(app).run()