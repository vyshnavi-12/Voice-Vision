import json
import os
import requests
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

# Set up file paths for contacts and encryption key
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "secure_contacts.json")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")

# Load encryption key to decrypt stored phone numbers
if not os.path.exists(KEY_FILE):
    print("❌ ERROR: Encryption key (secret.key) not found in Module5 folder.")
    cipher_suite = None
else:
    with open(KEY_FILE, "rb") as kf:
        cipher_suite = Fernet(kf.read())

# Read all saved emergency contacts from the database
def load_contacts():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading contacts: {e}")
        return []

# Decrypt an encrypted phone number
def decrypt_phone(encrypted_phone):
    if cipher_suite is None:
        return encrypted_phone
    decrypted = cipher_suite.decrypt(encrypted_phone.encode()).decode()
    return decrypted

# Get phone numbers of all emergency caretakers
def get_all_caretakers():
    contacts = load_contacts()
    numbers = []
    for entry in contacts:
        if entry.get("type") == "caretaker":
            # Decrypt the phone number
            numbers.append(decrypt_phone(entry["phone"]))
    return numbers

# Send SMS message to emergency contacts using Fast2SMS API
def send_sms(phone_numbers, message):
    if not phone_numbers:
        print("⚠️ No phone numbers found to send SMS.")
        return

    # Combine all phone numbers into a comma-separated list
    numbers_string = ",".join(phone_numbers)
    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "sender_id": "TXTIND",
        "message": message,
        "language": "english",
        "route": "q",
        "numbers": numbers_string
    }

    # Add API key to headers
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        # Send the SMS request
        response = requests.post(url, data=payload, headers=headers)
        print("SMS API response:")
        print(response.text)
    except Exception as e:
        print(f"Failed to connect to SMS API: {e}")


# Send emergency alert SMS with location to all caretakers
def trigger_emergency(location=None):
    print(" [Emergency] 🆘 Emergency command received. Processing SMS alert.")
    
    # Get all emergency contact phone numbers
    numbers = get_all_caretakers()
    if not numbers:
        return "No caretaker contacts registered."

    # Create Google Maps URL with user's location
    lat = location.get("latitude") or location.get("lat")
    lon = location.get("longitude") or location.get("lon")

    if lat and lon:
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
    else:
        maps_url = "Location not available"

    # Create emergency message with location
    message = f"Emergency Alert!\n\nThe user needs assistance.\n\nLive Location:\n{maps_url}"
    
    # Send SMS to all caretakers
    send_sms(numbers, message)
    return "Sucess"