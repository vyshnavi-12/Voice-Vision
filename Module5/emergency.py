import json
import os
import requests
from cryptography.fernet import Fernet
from dotenv import load_dotenv

from location import get_current_location

load_dotenv()

DB_FILE = os.path.join(os.path.dirname(__file__), "secure_contacts.json")
KEY_FILE = os.path.join(os.path.dirname(__file__), "secret.key")

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")


# ---------- LOAD ENCRYPTION KEY ----------
if not os.path.exists(KEY_FILE):
    raise Exception("Encryption key not found.")

with open(KEY_FILE, "rb") as kf:
    cipher_suite = Fernet(kf.read())


# ---------- LOAD CONTACTS ----------
def load_contacts():

    if not os.path.exists(DB_FILE):
        return []

    with open(DB_FILE, "r") as f:
        return json.load(f)


# ---------- DECRYPT PHONE ----------
def decrypt_phone(encrypted_phone):

    decrypted = cipher_suite.decrypt(encrypted_phone.encode()).decode()
    return decrypted


# ---------- GET ALL CARETAKERS ----------
def get_all_caretakers():

    contacts = load_contacts()
    numbers = []

    for entry in contacts:

        if entry["type"] == "caretaker":
            numbers.append(decrypt_phone(entry["phone"]))

    return numbers


# ---------- SEND SMS ----------
def send_sms(phone_numbers, message):

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

    response = requests.post(url, data=payload, headers=headers)

    print("SMS API response:")
    print(response.text)


# ---------- EMERGENCY FUNCTION ----------
def trigger_emergency():

    print("   [Emergency] 🆘 Emergency command received")

    numbers = get_all_caretakers()

    if not numbers:
        print("No caretaker contacts registered.")
        return

    print("   [Emergency] 📍 Fetching GPS location")

    loc_data = get_current_location()

    if loc_data["status"] != "SUCCESS":
        print("Location error:", loc_data["message"])
        return

    maps_url = loc_data["maps_url"]

    message = f"""
Emergency Alert!

The user may need assistance.

Location:
{maps_url}
"""

    send_sms(numbers, message)

    print("Emergency alert sent successfully.")