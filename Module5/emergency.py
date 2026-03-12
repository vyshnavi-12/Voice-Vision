import speech_recognition as sr
import pyttsx3
import json
import os
from cryptography.fernet import Fernet
from twilio.rest import Client
# Import your location function from the other file
# Assuming your location file is named 'location_service.py'
from location import get_current_location

# --- CONFIGURATION ---
DB_FILE = "secure_contacts.json"
KEY_FILE = "secret.key"
# Get these from twilio.com/console
TWILIO_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE = ''

if not os.path.exists(KEY_FILE):
    print("Encryption key not found.")
    exit()

with open(KEY_FILE, "rb") as kf:
    cipher_suite = Fernet(kf.read())


# --- Fetch & Decrypt Contact ---
def get_contact_number(target_name):
    if not os.path.exists(DB_FILE):
        print("No contact database found.")
        return None

    with open(DB_FILE, "r") as f:
        data = json.load(f)

    for entry in data:
        if entry["name"].lower() == target_name.lower():
            encrypted_num = entry["phone"].encode()
            decrypted_num = cipher_suite.decrypt(encrypted_num).decode()
            return f"+91{decrypted_num}"  # India country code

    return None


# --- Emergency Trigger ---
def trigger_action():
    name = input("Enter the name of the emergency contact: ").strip()

    if not name:
        print("No name entered. Aborting.")
        return

    phone_number = get_contact_number(name)

    if not phone_number:
        print(f"{name} not found in secure contact list.")
        return

    print(f"Contact found: {name}")
    print("Fetching GPS location...")

    # 1️⃣ Get Location
    loc_data = get_current_location()

    if loc_data["status"] != "SUCCESS":
        print(f"Location error: {loc_data['message']}")
        return

    maps_url = loc_data["maps_url"]
    print(f"Location acquired: {maps_url}")

    # 2️⃣ Twilio Action
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

        # Send SMS
        client.messages.create(
            body=f"🚨 Emergency Alert!\nMy current location:\n{maps_url}",
            from_=TWILIO_PHONE,
            to=phone_number
        )

        print("Emergency SMS sent.")

        # Initiate Call
        client.calls.create(
            twiml="""
                <Response>
                    <Say>This is an emergency alert. The location has been sent to your messages.</Say>
                </Response>
            """,
            from_=TWILIO_PHONE,
            to=phone_number
        )

        print("Emergency call initiated successfully.")

    except Exception as e:
        print("Twilio error occurred:")
        print(e)


# --- MAIN LOOP ---
print("🚨 Emergency Contact System Online")

while True:
    cmd = input("\nType 'call' to trigger emergency or 'exit' to quit: ").lower()

    if cmd == "call":
        trigger_action()
    elif cmd in ["exit", "quit"]:
        print("System shutting down.")
        break
    else:
        print("Invalid command.")
