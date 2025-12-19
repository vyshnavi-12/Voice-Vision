# import pyttsx3
# import time

# print("Testing Offline Voice...")
# try:
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 150)
#     engine.setProperty('volume', 1.0)
    
#     print("Speaking now...")
#     engine.say("This is a test of the offline voice system.")
#     engine.runAndWait()
#     print("Did you hear that?")
    
# except Exception as e:
#     print(f"❌ Critical Error: {e}")

import pyttsx3

print("--- VOICE DOCTOR STARTED ---")
engine = pyttsx3.init()
voices = engine.getProperty('voices')

print(f"Found {len(voices)} voices on this computer.")

for index, voice in enumerate(voices):
    print(f"\nTesting Voice #{index}: {voice.name}")
    try:
        engine.setProperty('voice', voice.id)
        engine.say(f"Hello, I am voice number {index}")
        engine.runAndWait()
    except Exception as e:
        print(f"  ❌ Failed to use this voice: {e}")

print("\n--- TEST COMPLETE ---")