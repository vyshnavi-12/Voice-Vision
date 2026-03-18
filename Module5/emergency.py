import json
import os
import re
import requests
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE  = os.path.join(BASE_DIR, "secure_contacts.json")
KEY_FILE = os.path.join(BASE_DIR, "secret.key")

FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")

if not os.path.exists(KEY_FILE):
    print("❌ ERROR: Encryption key (secret.key) not found in Module5 folder.")
    cipher_suite = None
else:
    with open(KEY_FILE, "rb") as kf:
        cipher_suite = Fernet(kf.read())

def load_contacts():
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading contacts: {e}")
        return []

def decrypt_phone(encrypted_phone):
    if cipher_suite is None:
        return encrypted_phone
    return cipher_suite.decrypt(encrypted_phone.encode()).decode()

def get_all_caretakers():
    contacts = load_contacts()
    return [decrypt_phone(e["phone"]) for e in contacts if e.get("type") == "caretaker"]

def get_contact_by_name(name):
    """Return phone number of a specific contact by name (case-insensitive partial match)."""
    contacts = load_contacts()
    name_lower = name.lower().strip()
    for entry in contacts:
        if name_lower in entry.get("name", "").lower():
            return decrypt_phone(entry["phone"])
    return None

def extract_target_from_command(user_text):
    """
    Parse user command to find who to send the emergency to.

    Examples handled:
      "help me emergency"                        → None  (send to all)
      "send emergency to vaishnavi"              → "vaishnavi"
      "emergency alert to all caretakers"        → "all"
      "help me send sos to mahesh"               → "mahesh"
      "emergency send to all contacts"           → "all"
      "help vaishnavi emergency"                 → "vaishnavi"  (name anywhere in sentence)

    Returns:
      "all"        → send to every caretaker
      "<name>"     → send only to that person
      None         → send to all (default)
    """
    if not user_text:
        return None

    text = user_text.lower().strip()

    # Check for explicit "all" keywords first
    all_keywords = [
        "all caretakers", "all contacts", "all emergency",
        "everyone", "all people", "everybody"
    ]
    if any(kw in text for kw in all_keywords):
        return "all"

    # Try to extract a name after "to", "alert to", "send to", "sos to"
    to_match = re.search(r'\bto\s+([a-zA-Z]+)', text)
    if to_match:
        candidate = to_match.group(1).strip()
        # Skip generic words that are not names
        skip_words = {"all", "my", "the", "a", "an", "caretaker",
                      "contact", "emergency", "everyone", "them"}
        if candidate not in skip_words:
            return candidate

    # Check if any registered contact name appears anywhere in the sentence
    contacts = load_contacts()
    for entry in contacts:
        name = entry.get("name", "").lower()
        if name and name in text:
            return name

    return None  # default: send to all

def send_sms(phone_numbers, message):
    if not phone_numbers:
        print("⚠️ No phone numbers found to send SMS.")
        return

    numbers_string = ",".join(phone_numbers)
    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "sender_id": "TXTIND",
        "message":   message,
        "language":  "english",
        "route":     "q",
        "numbers":   numbers_string,
    }
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type":  "application/x-www-form-urlencoded",
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        print("SMS API response:", response.text)
    except Exception as e:
        print(f"Failed to connect to SMS API: {e}")

def trigger_emergency(location=None, target=None):
    """
    Send emergency SMS.

    target:
      None or "all"  → send to all caretakers
      "<name>"       → send only to that named contact
    """
    print(" [Emergency] 🆘 Emergency command received. Processing SMS alert.")

    # Build location URL
    if location:
        lat = location.get("latitude") or location.get("lat")
        lon = location.get("longitude") or location.get("lon")
        maps_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon \
                   else "Location not available"
    else:
        maps_url = "Location not available"

    message = (f"Emergency Alert!\n\n"
               f"The user needs assistance.\n\n"
               f"Live Location:\n{maps_url}")

    # Decide who gets the SMS
    if target and target != "all":
        phone = get_contact_by_name(target)
        if not phone:
            print(f"⚠️ Contact '{target}' not found, sending to all.")
            numbers = get_all_caretakers()
        else:
            numbers = [phone]
            print(f"📱 Sending emergency to '{target}' only.")
    else:
        numbers = get_all_caretakers()
        print("📱 Sending emergency to all caretakers.")

    if not numbers:
        return "No caretaker contacts registered."

    send_sms(numbers, message)
    return "Sucess"