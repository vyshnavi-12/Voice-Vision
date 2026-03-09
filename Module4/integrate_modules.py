import time
import sys
import os

# --- 1. SYSTEM INITIALIZATION ---
print(" [System] 🚀 Assistive AI Hub: Multi-Module Integration Loaded (Mock Mode)")

def get_lang_msg(responses, lang_code):
    """Helper to select the correct language string."""
    if "te" in lang_code: return responses['te']
    elif "hi" in lang_code: return responses['hi']
    return responses['en']

# --- 2. CORE MODULES (MOCK) ---

def run_currency_detection(lang):
    """
    CURRENCY DETECTION: Identifies Indian currency notes and announces value.
    """
    res = {
        'en': "Detected Indian currency note of 500 rupees.",
        'te': "500 రూపాయల భారతీయ నోటు గుర్తించబడింది.",
        'hi': "500 रुपये का भारतीय नोट पाया गया है।"
    }
    print("   [Currency] 💰 Scanning for currency...")
    return get_lang_msg(res, lang)

def run_realtime_scene_description(lang):
    """General overview of the environment."""
    res = {
        'en': "You are in a spacious room. There is a window on your right and a bookshelf ahead.",
        'te': "మీరు ఒక విశాలమైన గదిలో ఉన్నారు. మీ కుడి వైపున ఒక కిటికీ మరియు ముందు పుస్తకాల షెల్ఫ్ ఉంది.",
        'hi': "आप एक बड़े कमरे में हैं। आपके दाईं ओर एक खिड़की है और आगे किताबों की अलमारी है।"
    }
    print("   [Scene] 🌎 Analyzing environment...")
    return get_lang_msg(res, lang)

def run_object_detection(lang):
    """Identifies specific household or street objects."""
    res = {
        'en': "I see a laptop, a spectacles case, and a cup on the desk.",
        'te': "డెస్క్‌పై ల్యాప్‌టాప్, కళ్ళద్దాల పెట్టె మరియు ఒక కప్పు ఉన్నాయి.",
        'hi': "मेज पर एक लैपटॉप, चश्मे का डिब्बा और एक कप है।"
    }
    print("   [Vision] 🔍 Scanning for objects...")
    return get_lang_msg(res, lang)

def run_obstacle_detection(lang):
    """Safety alerts for immediate path hazards."""
    res = {
        'en': "Caution! There is a footstool directly in your path.",
        'te': "జాగ్రత్త! మీ దారిలో ఒక చిన్న స్టూల్ ఉంది.",
        'hi': "सावधान! आपके रास्ते में एक छोटा स्टूल है।"
    }
    print("   [Safety] ⚠️ Checking for obstacles...")
    return get_lang_msg(res, lang)

def run_navigation_assistance(lang):
    """Directional guidance."""
    res = {
        'en': "Walk straight for five steps, then turn left to find the exit.",
        'te': "ఐదు అడుగులు నేరుగా నడవండి, ఆపై బయటకు వెళ్ళడానికి ఎడమవైపుకు తిరగండి.",
        'hi': "पाँच कदम सीधे चलें, फिर बाहर निकलने के लिए बाईं ओर मुड़ें।"
    }
    print("   [Nav] 📍 Calculating path...")
    return get_lang_msg(res, lang)

def run_people_detection(lang):
    """
    DESCRIBING PEOPLE: Focuses on count and appearance.
    """
    res = {
        'en': "There are two people standing in front of you. One is wearing a blue shirt.",
        'te': "మీ ముందు ఇద్దరు వ్యక్తులు నిలబడి ఉన్నారు. ఒకరు నీలం రంగు చొక్కా ధరించి ఉన్నారు.",
        'hi': "आपके सामने दो लोग खड़े हैं। एक ने नीली कमीज पहनी है।"
    }
    print("   [People Det] 👥 Describing physical presence...")
    return get_lang_msg(res, lang)

def run_face_recognition(lang):
    """
    RECOGNIZING FACES: Focuses on identity of registered users.
    """
    res = {
        'en': "I recognize 'Arjun' standing in front of you.",
        'te': "నేను మీ ముందు ఉన్న 'అర్జున్'ని గుర్తుపట్టాను.",
        'hi': "मैं आपके सामने खड़े 'अर्जुन' को पहचान पा रहा हूँ।"
    }
    print("   [Face Rec] 👤 Identifying known faces...")
    return get_lang_msg(res, lang)

def run_ocr_module(lang):
    """Reading text from signs or papers."""
    res = {
        'en': "The sign reads: 'Pharmacy - Open 24 Hours'.",
        'te': "సైన్ బోర్డు మీద ఇలా ఉంది: 'ఫార్మసీ - 24 గంటలు తెరిచి ఉంటుంది'.",
        'hi': "बोर्ड पर लिखा है: 'फार्मेसी - 24 घंटे खुला है'।"
    }
    print("   [OCR] 📄 Extracting text...")
    return get_lang_msg(res, lang)

def run_safety_emergency(lang):
    """
    SAFETY & EMERGENCY: Triggered when the user asks for help.
    Sends alert message + optional location to pre-registered emergency contacts.
    (Mock Implementation)
    """
    print("   [Emergency] 🆘 Emergency command detected...")
    time.sleep(1)

    print("   [Emergency] 📍 Fetching current location (mock GPS)...")
    time.sleep(1)

    print("   [Emergency] 📲 Sending alert to registered contacts via SMS/WhatsApp...")
    time.sleep(1)

    res = {
        'en': "Your emergency message with location has been sent to your registered contacts. Help is on the way.",
        'te': "మీ అత్యవసర సందేశం మరియు మీ ప్రస్తుత స్థానం నమోదు చేసిన సంప్రదింపులకు పంపబడింది. సహాయం త్వరలో చేరుతుంది.",
        'hi': "आपका आपातकालीन संदेश और स्थान पंजीकृत संपर्कों को भेज दिया गया है। सहायता रास्ते में है।"
    }

    return get_lang_msg(res, lang)

# --- 3. REGISTRATION FLOW (MOCK) ---

def run_registration_flow(stt_engine, tts_engine, initial_text):
    """Simulated face registration process."""
    lang = stt_engine.language
    
    msg_ask = {'en': "Please look at the camera for registration.", 
               'te': "రిజిస్ట్రేషన్ కోసం కెమెరా వైపు చూడండి.", 
               'hi': "पंजीकरण के लिए कैमरे की ओर देखें।"}
    
    tts_engine.speak(get_lang_msg(msg_ask, lang), lang)
    time.sleep(2) # Mocking capture
    
    res_done = {
        'en': "Registration complete. Face saved to database.",
        'te': "రిజిస్ట్రేషన్ పూర్తయింది. ముఖం డేటాబేస్‌లో సేవ్ చేయబడింది.",
        'hi': "पंजीकरण पूरा हुआ। चेहरा डेटाबेस में सहेज लिया गया है।"
    }
    return get_lang_msg(res_done, lang)