import json
import os
import re
from cryptography.fernet import Fernet

DB_FILE = os.path.join(os.path.dirname(__file__), "secure_contacts.json")
KEY_FILE = os.path.join(os.path.dirname(__file__), "secret.key")


# ---------- KEY MANAGEMENT ----------
def load_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()

    return Fernet(key)


cipher_suite = load_key()


# ---------- LOAD CONTACTS ----------
def load_contacts():

    if not os.path.exists(DB_FILE):
        return []

    with open(DB_FILE, "r") as f:
        return json.load(f)


# ---------- SAVE CONTACT ----------
def save_contacts(data):

    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------- REGISTER CONTACT ----------
def register_contact(name, phone, contact_type="caretaker"):

    phone = re.sub(r"\D", "", phone)

    if len(phone) != 10:
        return "Invalid phone number."

    encrypted_phone = cipher_suite.encrypt(phone.encode()).decode()

    contacts = load_contacts()

    contacts.append({
        "name": name,
        "phone": encrypted_phone,
        "type": contact_type
    })

    save_contacts(contacts)

    return f"{name} has been registered as an emergency contact."


# ---------- LIST CONTACTS ----------
def get_all_contacts():

    contacts = load_contacts()
    names = [c["name"] for c in contacts]

    return names