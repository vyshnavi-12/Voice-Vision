import json
import os
import re
from cryptography.fernet import Fernet

DB_FILE = os.path.join(os.path.dirname(__file__), "secure_contacts.json")
KEY_FILE = os.path.join(os.path.dirname(__file__), "secret.key")


# Generate or load the encryption key for storing phone numbers securely
def load_key():
    if not os.path.exists(KEY_FILE):
        # Create a new key if it doesn't exist
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()

    return Fernet(key)


cipher_suite = load_key()


# Read all saved emergency contacts from the database file
def load_contacts():
    if not os.path.exists(DB_FILE):
        return []

    with open(DB_FILE, "r") as f:
        return json.load(f)


# Save emergency contact list to the database file
def save_contacts(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)


# Add a new emergency contact (name and phone number)
def register_contact(name, phone, contact_type="caretaker"):
    # Remove all non-digit characters from phone number
    phone = re.sub(r"\D", "", phone)

    if len(phone) != 10:
        return "Invalid phone number."

    # Encrypt the phone number for security
    encrypted_phone = cipher_suite.encrypt(phone.encode()).decode()

    contacts = load_contacts()

    # Add the new contact to the list
    contacts.append({
        "name": name,
        "phone": encrypted_phone,
        "type": contact_type
    })

    save_contacts(contacts)

    return "Success"


# Get a list of all saved contact names
def get_all_contacts():
    contacts = load_contacts()
    names = [c["name"] for c in contacts]

    return names