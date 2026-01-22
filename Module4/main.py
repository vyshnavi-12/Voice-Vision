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
        # The new parser handles all the complex matching logic
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

                    print(f" 🗣️ HEARD: '{user_text}'")
                    last_interaction_time = time.time()

                    # --- CLEAN INTENT PARSING ---
                    # No manual lists. The AI decides everything.
                    intent = parser.parse(user_text, stt.current_lang_code)

                    print(f" 🎯 ACTION: {intent}")
                    response_text = ""
                    
                    # --- EXECUTION BLOCKS ---
                    if intent == "REGISTER_FACE":
                        # The brain knows "save cheyu" = REGISTER_FACE
                        response_text = modules.run_registration_flow(stt, tts, user_text)

                    elif intent == "PEOPLE_DETECTION":
                        response_text = modules.run_people_module(stt.current_lang_code)

                    elif intent == "DESCRIBE_SCENE":
                        response_text = modules.run_vision_module(stt.current_lang_code)
                    
                    elif intent == "READ_TEXT":
                        response_text = modules.run_ocr_module(stt.current_lang_code)

                    elif intent == "STOP":
                        tts.speak("Sleeping.", stt.current_lang_code)
                        break 

                    elif intent == "EMERGENCY":
                        response_text = "Emergency Mode! Sending SOS."

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