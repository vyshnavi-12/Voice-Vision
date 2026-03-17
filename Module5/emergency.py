import json
import os
import requests
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

# Setup paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "secure_contacts.json")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")

# ---------- LOAD ENCRYPTION KEY ----------
if not os.path.exists(KEY_FILE):
    # Instead of crashing, we log this for the backend console
    print("❌ ERROR: Encryption key (secret.key) not found in Module5 folder.")
    cipher_suite = None
else:
    with open(KEY_FILE, "rb") as kf:
        cipher_suite = Fernet(kf.read())

# ---------- LOAD CONTACTS ----------
def load_contacts():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading contacts: {e}")
        return []

# ---------- DECRYPT PHONE ----------
def decrypt_phone(encrypted_phone):
    if cipher_suite is None:
        return encrypted_phone # Fallback if key is missing
    decrypted = cipher_suite.decrypt(encrypted_phone.encode()).decode()
    return decrypted

# ---------- GET ALL CARETAKERS ----------
def get_all_caretakers():
    contacts = load_contacts()
    numbers = []
    for entry in contacts:
        if entry.get("type") == "caretaker":
            numbers.append(decrypt_phone(entry["phone"]))
    return numbers

# ---------- SEND SMS ----------
def send_sms(phone_numbers, message):
    if not phone_numbers:
        print("⚠️ No phone numbers found to send SMS.")
        return

    numbers_string = ",".join(phone_numbers)
    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "sender_id": "TXTIND",
        "message": message,
        "language": "english",
        "route": "q",
        "numbers": numbers_string
    }

    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        print("SMS API response:")
        print(response.text)
    except Exception as e:
        print(f"Failed to connect to SMS API: {e}")

# ---------- EMERGENCY FUNCTION ----------

def trigger_emergency(location=None):
    """
    Called after the Frontend captures GPS.
    Fast path: Decrypts numbers and sends the map link immediately.
    """
    print(" [Emergency] 🆘 Emergency command received. Processing SMS alert.")
    
    numbers = get_all_caretakers()
    if not numbers:
        return "No caretaker contacts registered."

    # Standard Google Maps URL format for mobile accessibility
    # Accepts both 'latitude' (from app.py) and 'lat' (from direct calls)
    lat = location.get("latitude") or location.get("lat")
    lon = location.get("longitude") or location.get("lon")

    if lat and lon:
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
    else:
        maps_url = "Location not available"

    message = f"Emergency Alert!\n\nThe user needs assistance.\n\nLive Location:\n{maps_url}"
    
    send_sms(numbers, message)
    return "Emergency alert sent to your caretakers."