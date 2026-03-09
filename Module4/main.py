import time
from stt_engine import WhisperSTT
from intent_parser import IntentParser
from tts_engine import TextToSpeech
from wakeword import WakeWordListener
import integrate_modules as modules

# Maximum idle time before assistant sleeps
SILENCE_TIMEOUT = 15  


def main():

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

                    elif intent == "PEOPLE_DETECTION":
                        response_text = modules.run_people_detection(stt.language)

                    elif intent == "SCENE_DESCRIPTION":
                        response_text = modules.run_realtime_scene_description(stt.language)

                    elif intent == "OBJECT_DETECTION":
                        response_text = modules.run_object_detection(stt.language)

                    elif intent == "OBSTACLE_DETECTION":
                        response_text = modules.run_obstacle_detection(stt.language)

                    elif intent == "NAVIGATION":
                        response_text = modules.run_navigation_assistance(stt.language)

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

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

        except Exception as e:
            print(f"❌ ERROR: {e}")
            break


if __name__ == "__main__":
    main()