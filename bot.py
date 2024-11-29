import os
import logging
import requests
from flask import Flask, request, jsonify
import time

# Configuration de l'application Flask
app = Flask(__name__)

# Variables pour Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # Clé API de Telegram
CHAT_ID = os.getenv("CHAT_ID")  # ID du chat Telegram

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fonction pour envoyer un message sur Telegram
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {'chat_id': CHAT_ID, 'text': message}
        response = requests.get(url, params=payload)
        
        # Affiche la réponse de l'API pour vérifier l'envoi du message
        logger.info(f"Réponse de Telegram: {response.json()}")

        if response.status_code != 200:
            logger.error(f"Erreur en envoyant le message : {response.status_code}")
        else:
            logger.info(f"Message envoyé à {CHAT_ID}: {message}")
    except Exception as e:
        logger.error(f"Erreur dans l'envoi du message Telegram : {e}")

# Test de l'envoi d'un message à Telegram
def test_telegram_connection():
    send_telegram_message("Test message envoyé par le bot")

# Route principale de l'application
@app.route('/')
def home():
    # Tester la connexion et envoyer un message
    test_telegram_connection()
    
    # Exemple d'envoi d'un message quand une condition est remplie
    send_telegram_message("Le service est en ligne et fonctionne.")
    
    # Message de bienvenue dès que le bot commence à fonctionner
    send_telegram_message("Bot ready to make some cash!")

    return "Bot en ligne et prêt à envoyer des messages."

# Route pour traiter les requêtes
@app.route('/signal', methods=['POST'])
def handle_signal():
    try:
        data = request.json
        logger.info(f"Signal reçu : {data}")
        
        # Logique d'envoi du signal de trading
        if "signal" in data:
            signal = data["signal"]
            send_telegram_message(f"Signal de trading reçu: {signal}")
            logger.info("Signal envoyé à Telegram")
        else:
            logger.warning("Aucun signal trouvé dans les données reçues")
            return jsonify({"error": "Signal non trouvé"}), 400
    except Exception as e:
        logger.error(f"Erreur lors du traitement du signal: {e}")
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"status": "success"}), 200

# Démarrage de l'application
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)