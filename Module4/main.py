import time
from stt_engine import HybridSTT
from intent_parser import IntentParser
from tts_engine import TextToSpeech
from wakeword import WakeWordListener
import integrate_modules as modules

SILENCE_TIMEOUT = 15  

def main():
    print("\n------------------------------------------------")
    print(" 🚀 INITIALIZING VOICE VISION...")
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

    while True:
        try:
            print("\n💤 SLEEPING: Waiting for 'Hello Vision'...")
            
            if wake_engine.listen(): 
                print("\n✨ WAKE WORD DETECTED!")
                tts.speak("I am listening.", stt.current_lang_code)
                last_interaction_time = time.time()
                
                while True:
                    if time.time() - last_interaction_time > SILENCE_TIMEOUT:
                        print(" ⏳ Silence detected. Sleeping.")
                        tts.speak("Sleeping.", stt.current_lang_code)
                        break 

                    audio = stt.listen()
                    if not audio: continue 
                    
                    user_text = stt.transcribe(audio)
                    if not user_text: continue

                    # --- NEW: MANUAL LANGUAGE SWITCH GUARD ---
                    # Sometimes STT transcribes "to Telugu" but doesn't trigger the internal switch.
                    # This hard check ensures the language actually changes.
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

                    print(f" 🗣️ HEARD: '{user_text}'")
                    last_interaction_time = time.time()

                    # --- INTENT PARSING ---
                    intent = parser.parse(user_text, stt.current_lang_code)
                    print(f" 🎯 ACTION: {intent}")
                    
                    response_text = ""
                    
                    # --- UPDATED EXECUTION BLOCKS ---
                    if intent == "UNKNOWN":
                        # If the Brain score was < 0.60, ask the user to repeat
                        response_text = "I'm sorry, I didn't quite catch that. Could you repeat?"
                    
                    elif intent == "FACE_RECOGNITION":
                        response_text = modules.run_face_recognition(stt.current_lang_code)

                    elif intent == "DESCRIBE_SCENE":
                        response_text = modules.run_realtime_scene_description(stt.current_lang_code)

                    elif intent == "PEOPLE_DETECTION":
                        response_text = modules.run_people_detection(stt.current_lang_code)

                    elif intent == "OBJECT_DETECTION":
                        response_text = modules.run_object_detection(stt.current_lang_code)

                    elif intent == "OBSTACLE_DETECTION":
                        response_text = modules.run_obstacle_detection(stt.current_lang_code)

                    elif intent == "NAVIGATION":
                        response_text = modules.run_navigation_assistance(stt.current_lang_code)

                    elif intent == "READ_TEXT":
                        response_text = modules.run_ocr_module(stt.current_lang_code)

                    elif intent == "EMERGENCY":
                        response_text = modules.run_safety_emergency(stt.current_lang_code)

                    elif intent == "REGISTER_FACE":
                        response_text = modules.run_registration_flow(stt, tts, user_text)

                    elif intent == "STOP":
                        tts.speak("Sleeping.", stt.current_lang_code)
                        break 

                    if response_text:
                        tts.speak(response_text, stt.current_lang_code)

        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ ERROR: {e}")
            break

if __name__ == "__main__":
    main()