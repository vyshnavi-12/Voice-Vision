import json
import os
import re
from cryptography.fernet import Fernet

DB_FILE = "secure_contacts.json"
KEY_FILE = "secret.key"

# --- Encryption Key Management ---
if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(key)
else:
    with open(KEY_FILE, "rb") as key_file:
        key = key_file.read()

cipher_suite = Fernet(key)


def get_verified_name():
    while True:
        name = input("Enter contact name: ").strip()

        if not name:
            print("Name cannot be empty.")
            continue

        spelling = ", ".join(list(name))
        confirm = input(f"You entered '{name}'. Spelled: {spelling}. Is this correct? (yes/no): ").lower()

        if confirm in ["yes", "y"]:
            return name
        else:
            corrected = input("Please type the correct spelling: ").replace(" ", "")
            if corrected:
                return corrected
            else:
                print("Invalid spelling. Let's try again.")


def register_contact():
    # 1. Name Verification
    name = get_verified_name()

    # 2. Phone Validation (10-digit Indian mobile)
    while True:
        phone_input = input(f"Enter the 10-digit mobile number for {name}: ")
        clean_phone = re.sub(r'\D', '', phone_input)

        if len(clean_phone) == 10:
            encrypted_phone = cipher_suite.encrypt(clean_phone.encode()).decode()
            break
        else:
            print(f"Error: You entered {len(clean_phone)} digits. Please enter exactly 10 digits.")

    # 3. Secure Save
    new_entry = {"name": name, "phone": encrypted_phone}

    data = []
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            data = json.load(f)

    data.append(new_entry)

    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Registration complete. {name} stored securely.")


# --- Main Loop ---
print("Secure Terminal Contact Manager Online.")

while True:
    cmd = input("\nType 'register' to add contact or 'exit' to quit: ").lower()

    if cmd == "register":
        register_contact()
    elif cmd in ["exit", "quit"]:
        print("System offline. Goodbye.")
        break
    else:
        print("Unknown command.")
