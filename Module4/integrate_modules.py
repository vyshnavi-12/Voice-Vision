import time
import sys
import os

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

# --- 2. IMPORT FACE MODULE ---
try:
    # Uses the fixed face_recog.py (with the RGB fix)
    import Module2.face_recog as fr
    FACE_MODULE_AVAILABLE = True
    print(" [Module Integration] ✅ Face Recognition System Loaded")
except ImportError:
    fr = None
    FACE_MODULE_AVAILABLE = False
    print(" [Module Integration] ⚠️ Error: Could not load Module2.")

# --- 3. MOCK MODULES (Vision & OCR) ---

def run_vision_module(lang):
    """Simulates detecting objects (Chair/Table)"""
    print(f"   [Mock Vision] 📸 Capturing Image & Analyzing...")
    time.sleep(1) 
    if "te" in lang: return "మీ ముందు ఒక కుర్చీ మరియు టేబుల్ ఉన్నాయి." 
    elif "hi" in lang: return "आपके सामने एक कुर्सी और मेज़ है।" 
    else: return "There is a chair and a table in front of you."

def run_ocr_module(lang):
    """Simulates reading text (Bills/Signs)"""
    print(f"   [Mock OCR] 📄 Scanning Text...")
    time.sleep(1)
    if "te" in lang: return "బిల్లు మొత్తం 500 రూపాయలు."
    elif "hi" in lang: return "कुल बिल 500 रुपये है।"
    else: return "The total bill amount is 500 rupees."

# --- 4. PEOPLE DETECTION (FACE RECOGNITION) ---

def run_people_module(lang):
    """
    This IS the Face Recognition function.
    It tells you WHO is in front of you.
    """
    if not FACE_MODULE_AVAILABLE:
        return "Face module not active."

    print(f"   [Face System] 👤 Analyzing Scene...")
    try:
        # Load Database
        db = fr.load_database()
        if not db["encodings"]:
             if "te" in lang: return "దయచేసి మొదట రిజిస్టర్ చేయండి."
             if "hi" in lang: return "कृपया पहले चेहरा रजिस्टर करें।"
             return "Please register a face first."

        # Run Recognition (Single Frame)
        result = fr.recognize_single_frame(db)
        
        # Handle Results
        if result == "Camera Error":
            return "Camera error."
        elif result == "NO_DB":
            return "Database empty."
        elif result == "UNKNOWN":
            if "te" in lang: return "నాకు ఎవరూ కనిపించడం లేదు."
            if "hi" in lang: return "मुझे कोई नहीं दिख रहा।"
            return "I don't see anyone I know."
        else:
            # Result is the Name (e.g., "Vaishnavi")
            if "te" in lang: return f"నేను {result}ని చూస్తున్నాను."
            elif "hi" in lang: return f"मैं {result} को देख रहा हूँ।"
            else: return f"I see {result}."

    except Exception as e:
        print(f"Face Error: {e}")
        return "Error in vision system."

# --- 5. REGISTRATION CONVERSATION FLOW ---

def run_registration_flow(stt_engine, tts_engine, initial_text):
    """
    Handles: Ask Name -> Open Camera -> Confirm Success
    """
    if not FACE_MODULE_AVAILABLE:
        return "Error. Face module is missing."

    lang = stt_engine.current_lang_code
    print("   [Face System] 📝 Starting Registration Flow...")

    # 1. Extract Name (e.g., "Register as Rahul")
    new_name = ""
    text_lower = initial_text.lower()
    if "as" in text_lower:
        parts = text_lower.split("as")
        if len(parts) > 1:
            new_name = parts[-1].strip().replace(".", "")
    
    # 2. Ask for name if missing
    if not new_name:
        if "te" in lang: msg = "నేను ఏ పేరుతో సేవ్ చేయాలి?"
        elif "hi" in lang: msg = "मुझे किस नाम से सेव करना चाहिए?"
        else: msg = "What name should I save?"
        
        tts_engine.speak(msg, lang)
        
        name_audio = stt_engine.listen()
        if name_audio:
            new_name = stt_engine.transcribe(name_audio).strip().replace(".", "")

    if not new_name:
        return "I didn't hear a name."

    # 3. Prompt to look at camera
    if "te" in lang: msg = f"{new_name}ని రిజిస్టర్ చేస్తున్నాను. కెమెరా వైపు చూడండి."
    elif "hi" in lang: msg = f"{new_name} को रजिस्टर कर रहा हूँ। कैमरे की ओर देखें।"
    else: msg = f"Registering {new_name}. Look at the camera."
    
    tts_engine.speak(msg, lang)

    # 4. Open Camera & Register
    try:
        db = fr.load_database()
        success = fr.register_person(db, new_name, num_samples=8)
    except Exception as e:
        print(f"Reg Error: {e}")
        success = False

    # 5. Result
    if success:
        if "te" in lang: return f"విజయం. {new_name} రిజిస్టర్ అయ్యారు."
        elif "hi" in lang: return f"सफल। {new_name} रजिस्टर हो गए हैं।"
        else: return f"Success. Registered {new_name}."
    else:
        if "te" in lang: return "రిజిస్ట్రేషన్ విఫలమైంది."
        elif "hi" in lang: return "रजिस्ट्रेशन विफल रहा।"
        else: return "Registration failed. Camera error."