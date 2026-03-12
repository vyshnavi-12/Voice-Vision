import time
import requests
import threading
import simpleaudio as sa
from stt_engine import WhisperSTT
from intent_parser import IntentParser
from tts_engine import TextToSpeech
from wakeword import WakeWordListener
import integrate_modules as modules

# Maximum idle time before assistant sleeps
SILENCE_TIMEOUT = 15  

# Controls background obstacle monitoring
obstacle_detection_enabled = True

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

def play_alert():
    try:
        wave = sa.WaveObject.from_wave_file("alert.wav")
        play = wave.play()
    except Exception as e:
        print(f"Sound error: {e}")

def obstacle_monitor():
    global obstacle_detection_enabled

    while True:

        if not obstacle_detection_enabled:
            time.sleep(0.2)
            continue

        obstacle = modules.run_obstacle_detection()

        if obstacle:
            print("⚠️ Obstacle detected!")
            play_alert()

        time.sleep(0.5)

def main():
    global obstacle_detection_enabled

    print("\n------------------------------------------------")
    print(" 🚀 INITIALIZING VOICE VISION...")

    try:
        wake_engine = WakeWordListener()

        # Default language = English
        stt = WhisperSTT(language="en")

        parser = IntentParser()
        tts = TextToSpeech()

    except Exception as e:
        print(f"\n❌ CRITICAL INIT ERROR: {e}")
        return

    print(" ✅ SYSTEM READY.")
    print("------------------------------------------------")

    # Start obstacle detection background thread
    obstacle_thread = threading.Thread(
        target=obstacle_monitor,
        daemon=True
    )

    obstacle_thread.start()

    print("🟢 Obstacle monitoring started.")

    # ---------------- OUTER LOOP (SLEEP MODE) ----------------
    while True:
        try:
            print("\n💤 SLEEPING: Waiting for 'Hello Vision'...")

            if wake_engine.listen():

                print("\n✨ WAKE WORD DETECTED!")
                tts.speak("I am listening.", stt.language)

                last_interaction_time = time.time()

                # ---------------- INNER LOOP (ACTIVE MODE) ----------------
                while True:

                    # Auto sleep after silence
                    if time.time() - last_interaction_time > SILENCE_TIMEOUT:
                        print(" ⏳ Silence detected. Sleeping.")
                        tts.speak("Sleeping.", stt.language)
                        break

                    # Listen
                    audio = stt.listen()
                    if not audio:
                        continue

                    # Transcribe
                    user_text = stt.transcribe(audio)
                    if not user_text:
                        continue

                    print(f" 🗣️ HEARD: '{user_text}'")
                    last_interaction_time = time.time()

                    # ---------------- INTENT PARSING ----------------
                    intent, target_lang = parser.parse(user_text)

                    print(f" 🎯 ACTION: {intent}")

                    # Pause obstacle detection while running commands
                    obstacle_detection_enabled = False

                    response_text = ""

                    # ---------------- LANGUAGE SWITCH ----------------
                    if intent == "SWITCH_LANGUAGE":

                        if target_lang:
                            stt.set_language(target_lang)

                            confirmations = {
                                "en": "Language switched to English.",
                                "te": "భాష తెలుగు కు మార్చబడింది.",
                                "hi": "भाषा हिंदी में बदल दी गई है।"
                            }

                            tts.speak(confirmations[target_lang], target_lang)

                        else:
                            tts.speak("Please specify the language.", stt.language)

                        continue

                    # ---------------- NORMAL INTENT DISPATCH ----------------
                    if intent == "UNKNOWN":
                        fallback_responses = {
                            "en": "I'm sorry, I didn't understand that. Please repeat.",
                            "te": "క్షమించండి, నేను అర్థం చేసుకోలేకపోయాను. మళ్లీ చెప్పండి.",
                            "hi": "माफ कीजिए, मैं समझ नहीं पाया। कृपया दोबारा कहें।"
                        }

                        response_text = fallback_responses.get(
                            stt.language,
                            fallback_responses["en"]
                        )

                    elif intent == "CURRENCY_DETECTION":
                        response_text = modules.run_currency_detection(stt.language)

                    elif intent == "FACE_RECOGNITION":
                        response_text = modules.run_face_recognition(stt.language)

                    elif intent == "REGISTER_FACE":
                        tts.speak("What is the person's name?", stt.language)
                        audio = stt.listen()
                        name = stt.transcribe(audio)
                        response_text = modules.run_face_registration(stt.language, name)

                    elif intent == "PEOPLE_COUNT":
                        response_text = modules.run_people_count(stt.language)

                    elif intent == "PERSON_DESCRIPTION":
                        response_text = modules.run_people_description(stt.language)

                    elif intent == "SCENE_DESCRIPTION":
                        if check_internet():
                            print("🌐 Internet detected → running scene description")
                            response_text = modules.run_realtime_scene_description(stt.language)

                        else:
                            print("📴 Offline mode → falling back to object detection")
                            response_text = modules.run_object_detection(stt.language)

                    elif intent == "OBJECT_DETECTION":
                        response_text = modules.run_object_detection(stt.language)

                    elif intent == "NAVIGATION":
                        tts.speak("What object should I look for?", stt.language)

                        audio = stt.listen()
                        user_text = stt.transcribe(audio)

                        target_object = user_text.lower().strip()

                        response_text = modules.run_navigation_assistance(
                            stt.language,
                            target_object
                        )

                    elif intent == "OCR":
                        response_text = modules.run_ocr_module(stt.language)

                    elif intent == "EMERGENCY":
                        response_text = modules.run_safety_emergency(stt.language)

                    elif intent == "REGISTER_FACE":
                        response_text = modules.run_registration_flow(stt, tts, user_text)

                    elif intent == "STOP":
                        tts.speak("Sleeping.", stt.language)
                        break

                    # ---------------- SPEAK RESPONSE ----------------
                    if response_text:
                        tts.speak(response_text, stt.language)

                    # Resume obstacle detection
                    obstacle_detection_enabled = True

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

        except Exception as e:
            print(f"❌ ERROR: {e}")
            break


if __name__ == "__main__":
    main()