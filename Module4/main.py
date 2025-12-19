import time
from stt_engine import HybridSTT
from intent_parser import IntentParser
from tts_engine import TextToSpeech
from wakeword import WakeWordListener
import mock_modules as modules

# --- CONFIGURATION ---
# The app waits 15 seconds for you to speak. 
# If you say nothing, it naturally goes back to sleep.
SILENCE_TIMEOUT = 15  

def main():
    print("\n------------------------------------------------")
    print(" 🚀 INITIALIZING VOICE VISION (NATURAL MODE)...")
    
    try:
        # 1. THE EAR (Wake Word)
        print(" 👂 Loading Wake Word Engine...")
        wake_engine = WakeWordListener()
        
        # 2. THE EAR (Speech-to-Text)
        stt = HybridSTT()
        
        # 3. THE BRAIN (Intent Parser - BERT Powered)
        print(" 🧠 Loading Semantic Brain...")
        parser = IntentParser(use_bert=True) 
        
        # 4. THE MOUTH (Text-to-Speech)
        tts = TextToSpeech()
        
    except Exception as e:
        print(f"\n❌ CRITICAL INIT ERROR: {e}")
        return

    print(" ✅ SYSTEM READY.")
    print("------------------------------------------------")

    # --- MAIN PROGRAM LOOP ---
    while True:
        try:
            # =================================================
            # STATE 1: SLEEP MODE (Standby)
            # =================================================
            print("\n💤 SLEEPING: Waiting for 'Hello Vision'...")
            
            # This blocks execution until the wake word is detected
            if wake_engine.listen(): 
                print("\n✨ WAKE WORD DETECTED! (App is AWAKE)")
                tts.speak("I am listening.", stt.current_lang_code)

                # =================================================
                # STATE 2: CONVERSATION MODE
                # =================================================
                last_interaction_time = time.time()
                
                while True:
                    # 1. CHECK NATURAL TIMEOUT
                    # If 15 seconds pass with no valid command, we assume interaction is over.
                    if time.time() - last_interaction_time > SILENCE_TIMEOUT:
                        print(" ⏳ Natural silence detected (15s). Going to sleep.")
                        tts.speak("Sleeping.", stt.current_lang_code)
                        break # Break inner loop -> Back to State 1

                    # 2. LISTEN
                    print(f"\n[{stt.lang_names[stt.current_lang_code]}] Listening...")
                    audio = stt.listen()
                    
                    # If silence, just loop back and check timer again
                    if not audio:
                        continue 

                    # 3. TRANSCRIBE
                    user_text = stt.transcribe(audio)
                    if not user_text: continue

                    # 4. IDENTIFY INTENT (The Brain)
                    intent = parser.parse(user_text, stt.current_lang_code)
                    
                    # === HUMAN-LIKE FILTER ===
                    # If BERT says "UNKNOWN", we assume you are talking to someone else.
                    # We do NOT say "I didn't understand". We just stay silent (like a polite human).
                    if intent == "UNKNOWN":
                        print(" 🤷 Unknown intent (Ignoring background conversation).")
                        continue 

                    # 5. EXECUTE VALID COMMAND
                    print(f" 🎯 ACTION: {intent}")
                    last_interaction_time = time.time() # <--- RESET TIMER (Conversation continues)

                    response_text = ""
                    if intent == "READ_TEXT":
                        response_text = modules.run_ocr_module(stt.current_lang_code)
                    
                    elif intent == "DESCRIBE_SCENE":
                        response_text = modules.run_vision_module(stt.current_lang_code)
                        
                    elif intent == "PEOPLE_DETECTION":
                        response_text = modules.run_people_module(stt.current_lang_code)
                        
                    elif intent == "EMERGENCY":
                        response_text = "Emergency Mode Activated! Sending Alerts..." if "en" in stt.current_lang_code else "SOS Alert Sent!"

                    # 6. SPEAK RESPONSE
                    if response_text:
                        tts.speak(response_text, stt.current_lang_code)
                    
                    # Loop continues...

        except KeyboardInterrupt:
            print("\n👋 Exiting Voice Vision...")
            break
        except Exception as e:
            print(f"❌ ERROR IN MAIN LOOP: {e}")
            break

if __name__ == "__main__":
    main()