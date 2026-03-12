import time
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Module1.object_detection import detect_objects
from Module1.obstacle_detection import detect_obstacle
from Module1.scene_description import describe_scene
from Module1.navigation_assistance import navigate_to_object
from Module1.currency_recognition import detect_currency
from Module2.people_detection import count_people, describe_person

# --- 1. SYSTEM INITIALIZATION ---
print(" [System] 🚀 Assistive AI Hub: Multi-Module Integration Loaded (Mock Mode)")

def get_lang_msg(responses, lang_code):
    """Helper to select the correct language string."""
    if "te" in lang_code: return responses['te']
    elif "hi" in lang_code: return responses['hi']
    return responses['en']

# --- 2. CORE MODULES  ---

def run_currency_detection(lang):

    print("   [Currency] 💰 Running currency detection...")

    currency = detect_currency()

    if not currency:

        responses = {
            "en": "I cannot detect any currency note.",
            "te": "నాకు కరెన్సీ నోటు కనిపించలేదు.",
            "hi": "मुझे कोई मुद्रा नोट दिखाई नहीं दे रहा है।"
        }

        return responses.get(lang, responses["en"])

    responses = {
        "en": f"This appears to be {currency} rupees.",
        "te": f"ఇది {currency} రూపాయల నోటు.",
        "hi": f"यह {currency} रुपये का नोट है।"
    }

    return responses.get(lang, responses["en"])

def run_realtime_scene_description(lang):

    print("   [Vision] 🌎 Running scene description...")

    description = describe_scene()

    return description

def run_object_detection(lang):

    print("   [Vision] 🔍 Running object detection...")

    objects = detect_objects()

    if not objects:

        responses = {
            "en": "I do not see any objects nearby.",
            "te": "నా ముందు ఎలాంటి వస్తువులు కనిపించలేదు.",
            "hi": "मुझे कोई वस्तु दिखाई नहीं दे रही है।"
        }

        return responses.get(lang, responses["en"])

    elif len(objects) == 1:

        responses = {
            "en": f"There is a {objects[0]} in front of you.",
            "te": f"మీ ముందు ఒక {objects[0]} ఉంది.",
            "hi": f"आपके सामने एक {objects[0]} है।"
        }

        return responses.get(lang, responses["en"])

    else:

        object_list = ", ".join(objects[:-1]) + " and " + objects[-1]

        responses = {
            "en": f"There are {object_list} in front of you.",
            "te": f"మీ ముందు {object_list} ఉన్నాయి.",
            "hi": f"आपके सामने {object_list} हैं।"
        }

        return responses.get(lang, responses["en"])

def run_obstacle_detection():

    obstacle = detect_obstacle()

    return obstacle

def run_navigation_assistance(lang, target_object):

    print("   [Navigation] 🧭 Running navigation assistance...")

    guidance = navigate_to_object(target_object)

    return guidance

def run_people_count(lang):

    print("   [People] 👥 Counting people...")

    count = count_people()

    if count == 0:

        responses = {
            "en": "I do not see anyone.",
            "te": "నాకు ఎవరూ కనిపించలేదు.",
            "hi": "मुझे कोई व्यक्ति दिखाई नहीं दे रहा है।"
        }

    elif count == 1:

        responses = {
            "en": "There is one person in front of you.",
            "te": "మీ ముందు ఒక వ్యక్తి ఉన్నాడు.",
            "hi": "आपके सामने एक व्यक्ति है।"
        }

    else:

        responses = {
            "en": f"There are {count} people in front of you.",
            "te": f"మీ ముందు {count} మంది వ్యక్తులు ఉన్నారు.",
            "hi": f"आपके सामने {count} लोग हैं।"
        }

    return responses.get(lang, responses["en"])

def run_people_description(lang):

    print("   [People] 🧑 Describing person...")

    message = describe_person()

    return message

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