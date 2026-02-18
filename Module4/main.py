import time
from stt_engine import HybridSTT
from intent_parser import IntentParser
from tts_engine import TextToSpeech
from wakeword import WakeWordListener
import integrate_modules as modules

# Maximum idle time (in seconds) before the assistant goes back to sleep
SILENCE_TIMEOUT = 15  


def main():
    """
    Main control loop for Voice Vision.
    Handles system initialization, wake-word detection,
    continuous listening, intent parsing, and module execution.
    """

    print("\n------------------------------------------------")
    print(" 🚀 INITIALIZING VOICE VISION...")

    # ---- SYSTEM INITIALIZATION ----
    # Initialize all core subsystems:
    # - Wake word detection
    # - Speech-to-text (Hybrid: Online + Offline)
    # - Intent understanding (semantic)
    # - Text-to-speech output
    try:
        wake_engine = WakeWordListener()
        stt = HybridSTT()
        parser = IntentParser(use_bert=True) 
        tts = TextToSpeech()
    except Exception as e:
        print(f"\n❌ CRITICAL INIT ERROR: {e}")
        return

    print(" ✅ SYSTEM READY.")
    print("------------------------------------------------")


    # ---- OUTER LOOP: SLEEP MODE ----
    # The system stays dormant until the wake word is detected.
    while True:
        try:
            print("\n💤 SLEEPING: Waiting for 'Hello Vision'...")

            # Block until wake word is detected
            if wake_engine.listen(): 
                print("\n✨ WAKE WORD DETECTED!")
                tts.speak("I am listening.", stt.current_lang_code)

                # Track last user interaction to detect silence
                last_interaction_time = time.time()


                # ---- INNER LOOP: ACTIVE LISTENING MODE ----
                # Continues until silence timeout or STOP intent
                while True:

                    # If no interaction for too long, return to sleep
                    if time.time() - last_interaction_time > SILENCE_TIMEOUT:
                        print(" ⏳ Silence detected. Sleeping.")
                        tts.speak("Sleeping.", stt.current_lang_code)
                        break 

                    # Capture audio input
                    audio = stt.listen()
                    if not audio:
                        continue 

                    # Convert speech to text
                    user_text = stt.transcribe(audio)
                    if not user_text:
                        continue


                    # ---- MANUAL LANGUAGE SWITCH GUARD ----
                    # Backup safeguard in case STT recognizes language words
                    # but internal switch logic fails.
                    # This ensures language consistency before intent parsing.
                    lower_text = user_text.lower()

                    if "telugu" in lower_text or "తెలుగు" in lower_text:
                        stt.current_lang_code = 'te-IN'
                        tts.speak("Switching to Telugu.", 'en-IN')
                        last_interaction_time = time.time()
                        continue

                    elif "hindi" in lower_text or "हिंदी" in lower_text:
                        stt.current_lang_code = 'hi-IN'
                        tts.speak("Switching to Hindi.", 'en-IN')
                        last_interaction_time = time.time()
                        continue

                    elif "english" in lower_text or "ఇంగ్లీష్" in lower_text:
                        stt.current_lang_code = 'en-IN'
                        tts.speak("Switching to English.", 'en-IN')
                        last_interaction_time = time.time()
                        continue


                    # Log recognized speech
                    print(f" 🗣️ HEARD: '{user_text}'")
                    last_interaction_time = time.time()


                    # ---- INTENT UNDERSTANDING ----
                    # Convert user text into a high-level action intent
                    intent = parser.parse(user_text, stt.current_lang_code)
                    print(f" 🎯 ACTION: {intent}")

                    response_text = ""


                    # ---- INTENT EXECUTION DISPATCHER ----
                    # Routes the detected intent to its corresponding module
                    if intent == "UNKNOWN":
                        response_text = (
                            "I'm sorry, I didn't quite catch that. Could you repeat?"
                        )

                    elif intent == "FACE_RECOGNITION":
                        response_text = modules.run_face_recognition(
                            stt.current_lang_code
                        )

                    elif intent == "DESCRIBE_SCENE":
                        response_text = modules.run_realtime_scene_description(
                            stt.current_lang_code
                        )

                    elif intent == "PEOPLE_DETECTION":
                        response_text = modules.run_people_detection(
                            stt.current_lang_code
                        )

                    elif intent == "OBJECT_DETECTION":
                        response_text = modules.run_object_detection(
                            stt.current_lang_code
                        )

                    elif intent == "OBSTACLE_DETECTION":
                        response_text = modules.run_obstacle_detection(
                            stt.current_lang_code
                        )

                    elif intent == "NAVIGATION":
                        response_text = modules.run_navigation_assistance(
                            stt.current_lang_code
                        )

                    elif intent == "READ_TEXT":
                        response_text = modules.run_ocr_module(
                            stt.current_lang_code
                        )

                    elif intent == "EMERGENCY":
                        response_text = modules.run_safety_emergency(
                            stt.current_lang_code
                        )

                    elif intent == "REGISTER_FACE":
                        response_text = modules.run_registration_flow(
                            stt, tts, user_text
                        )

                    elif intent == "STOP":
                        tts.speak("Sleeping.", stt.current_lang_code)
                        break 


                    # ---- VOICE RESPONSE OUTPUT ----
                    if response_text:
                        tts.speak(response_text, stt.current_lang_code)

        # Graceful shutdown
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break

        except Exception as e:
            print(f"❌ ERROR: {e}")
            break


if __name__ == "__main__":
    main()
