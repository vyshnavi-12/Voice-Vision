import time
from stt_engine import HybridSTT
from intent_parser import IntentParser
from tts_engine import TextToSpeech
from wakeword import WakeWordListener
from face_engine import FaceEngine  # <--- FIXED: Import your new engine
import mock_modules as modules

# --- CONFIGURATION ---
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
        
        # 3. THE BRAIN (Intent Parser & Face Engine)
        print(" 🧠 Loading Semantic Brain...")
        parser = IntentParser(use_bert=True) 
        face_sys = FaceEngine()  # <--- FIXED: Initialize face_sys here
        
        # 4. THE MOUTH (Text-to-Speech)
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
                print("\n✨ WAKE WORD DETECTED! (App is AWAKE)")
                tts.speak("I am listening.", stt.current_lang_code)

                last_interaction_time = time.time()
                
                while True:
                    if time.time() - last_interaction_time > SILENCE_TIMEOUT:
                        print(" ⏳ Natural silence detected (15s). Going to sleep.")
                        tts.speak("Sleeping.", stt.current_lang_code)
                        break 

                    print(f"\n[{stt.lang_names[stt.current_lang_code]}] Listening...")
                    audio = stt.listen()
                    # Inside the while True loop of main.py, after getting user_text:
                    if not audio:
                        continue 

                    user_text = stt.transcribe(audio)
                    if not user_text: continue
                    # --- Corrected Logic in main.py (After user_text = stt.transcribe(audio)) ---

                    # Define keywords that trigger the registration process
                    registration_keywords = ["register", "save this person", "remember this person"]

                    if any(word in user_text.lower() for word in registration_keywords):
                        # Try to find a name in the sentence
                        if "as" in user_text.lower():
                            new_person_name = user_text.lower().split("as")[-1].strip().replace(".", "")
                        else:
                            # Fallback if no name is provided in the sentence
                            tts.speak("What name should I use for this person?", stt.current_lang_code)
                            name_audio = stt.listen()
                            new_person_name = stt.transcribe(name_audio).strip().replace(".", "")

                        if new_person_name:
                            tts.speak(f"Starting registration for {new_person_name}. Look at the camera.", stt.current_lang_code)
                            
                            # Use the logic from your face_recog.py (averaging 8 samples)
                            success = face_sys.register_new_face(new_person_name)
                            
                            if success:
                                tts.speak(f"Success. I have registered {new_person_name}.", stt.current_lang_code)
                            else:
                                tts.speak("Registration failed. I couldn't see a face.", stt.current_lang_code)
                        
                        last_interaction_time = time.time()
                        continue # Go back to the top of the loop
                    

                    intent = parser.parse(user_text, stt.current_lang_code)
                    
                    if intent == "UNKNOWN":
                        print(" 🤷 Unknown intent (Ignoring background conversation).")
                        continue 

                    print(f" 🎯 ACTION: {intent}")
                    last_interaction_time = time.time() 

                    response_text = ""
                    
                    # --- INTENT EXECUTION ---
                    if intent == "READ_TEXT":
                        response_text = modules.run_ocr_module(stt.current_lang_code)
                    
                    elif intent == "DESCRIBE_SCENE":
                        response_text = modules.run_vision_module(stt.current_lang_code)
                    elif intent== "REGISTER_FACE":
                        name=''
                        if "as" in user_text.lower():
                            name = user_text.lower().split("as")[-1].strip().replace(".", "")
                        if not name:
                            tts.speak("What name should I use for this person?", stt.current_lang_code)
                            name_audio = stt.listen()
                            name = stt.transcribe(name_audio).strip().replace(".", "")
                        if name:
                            tts.speak(f"Starting registration for {name}. Look at the camera.", stt.current_lang_code)
                            success = face_sys.register_new_face(name)
                            if success:
                                response_text = f"Success. I have registered {name}."
                            else:
                                response_text = "Registration failed. I couldn't see a face."
                    elif intent == "PEOPLE_DETECTION":
                        # FIXED: Use the integrated face_sys data
                        data = face_sys.analyze_scene()

                        if data["status"] == "FOUND":
                            name, dist, desc = data["name"], data["distance"], data["description"]
                            
                            templates = {
                                "en-IN": f"I see {name}. They are about {dist} meters away and appear {desc}.",
                                "hi-IN": f"मैं {name} को देख रहा हूँ। वे लगभग {dist} मीटर दूर हैं और {desc} लग रहे हैं।",
                                "te-IN": f"నేను {name}ని చూస్తున్నాను. వారు సుమారు {dist} మీటర్ల దూరంలో ఉన్నారు."
                            }
                            response_text = templates.get(stt.current_lang_code, templates["en-IN"])
                        else:
                            # Handling NO_FACE or errors
                            err_templates = {
                                "en-IN": "I don't see anyone clearly right now.",
                                "hi-IN": "मुझे अभी कोई स्पष्ट रूप से नहीं दिख रहा है।",
                                "te-IN": "ప్రస్తుతం ఎవరూ స్పష్టంగా కనిపించడం లేదు."
                            }
                            response_text = err_templates.get(stt.current_lang_code, err_templates["en-IN"])
                        
                    elif intent == "EMERGENCY":
                        print("EMERGENCY ACTIVATED!")
                        import threading
                        record_thread=threading.Thread(target=face_sys.start_emergency_recording)
                        record_thread.start()
                        import safety_module
                        location_link = "https://www.google.com/maps?q=17.3850,78.4867" # Example coords
                        safety_module.send_emergency_sms(f"SOS! I am in danger. My location: {location_link}")
                        import pygame
                        pygame.mixer.init()
                        siren=pygame.mixer.Sound("siren.wav")
                        siren.play(loops=1)
                        tts.speak("Emergency activated. Contacting authorities.",stt.current_lang_code)
                        import numpy as np
                        import cv2
                        flash=np.full((1080,1920,3),255,dtype=np.uint8)
                        cv2.imshow("FLASH",flash)
                        cv2.waitKey(500)
                        cv2.destroyWindow("FLASH")

                    # 6. SPEAK RESPONSE
                    elif intent == "STOP":
                        tts.speak("Shutting down.", stt.current_lang_code)
                        import os
                        os._exit(0) # Force immediate shutdown of all threads and processes
                    if response_text:
                        tts.speak(response_text, stt.current_lang_code)

        except KeyboardInterrupt:
            print("\n👋 Exiting Voice Vision...")
            break
        except Exception as e:
            print(f"❌ ERROR IN MAIN LOOP: {e}")
            break

if __name__ == "__main__":
    main()
