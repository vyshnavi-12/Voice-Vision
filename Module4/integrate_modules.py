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
from Module2.face_detection import register_face, recognize_face
from Module5.phone_register import register_contact
from Module5.emergency import trigger_emergency
from Module3.ocr import run_ocr

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

    print("   [Face] 👤 Running face recognition...")

    message = recognize_face()

    if "Unknown" in message or "do not" in message:

        responses = {
            "en": message,
            "te": "ఈ వ్యక్తిని నేను గుర్తించలేకపోతున్నాను.",
            "hi": "मैं इस व्यक्ति को पहचान नहीं पा रहा हूँ।"
        }

    else:

        name = message.split()[0]

        responses = {
            "en": f"{name} is standing in front of you.",
            "te": f"{name} మీ ముందు నిలబడి ఉన్నారు.",
            "hi": f"{name} आपके सामने खड़े हैं।"
        }

    return responses.get(lang, responses["en"])

def run_face_registration(lang, name):

    print("   [Face] 📸 Registering new face...")

    message = register_face(name)

    if "successfully" in message:

        responses = {
            "en": f"{name} has been registered successfully.",
            "te": f"{name} విజయవంతంగా నమోదు చేయబడింది.",
            "hi": f"{name} सफलतापूर्वक पंजीकृत किया गया है।"
        }

    else:

        responses = {
            "en": "Face registration failed.",
            "te": "ముఖం నమోదు విఫలమైంది.",
            "hi": "चेहरा पंजीकरण असफल रहा।"
        }

    return responses.get(lang, responses["en"])

def run_ocr_module(lang, speak_callback=None):

    print("   [OCR] 📄 Starting OCR — align the camera to the text...")

    # Tell user to hold document before camera opens
    if speak_callback:
        guidance_prompts = {
            "en": "Please hold the document in front of the camera.",
            "te": "దయచేసి డాక్యుమెంట్‌ని కెమెరా ముందు పట్టుకోండి.",
            "hi": "कृपया दस्तावेज़ को कैमरे के सामने रखें।"
        }
        speak_callback(guidance_prompts.get(lang, guidance_prompts["en"]))

    # Pass speak_callback so live guidance is spoken during alignment
    text = run_ocr(speak_callback=speak_callback)

    # Handle camera error
    if text.startswith("Camera error"):
        return text

    # Handle no text found
    if not text or text.strip() == "" or text == "I could not read any text.":
        no_text_responses = {
            "en": "I could not read any text.",
            "te": "నాకు ఏ టెక్స్ట్ చదవడం సాధ్యం కాలేదు.",
            "hi": "मैं कोई टेक्स्ट नहीं पढ़ पाया।"
        }
        return no_text_responses.get(lang, no_text_responses["en"])

    # Prefix result with spoken intro
    intros = {
        "en": "The text reads: ",
        "te": "టెక్స్ట్ ఇలా ఉంది: ",
        "hi": "टेक्स्ट इस प्रकार है: "
    }

    return intros.get(lang, intros["en"]) + text

def run_phone_registration(lang, name, phone):

    print("   [Safety] 📱 Registering emergency contact...")

    result = register_contact(name, phone)

    responses = {
        "en": result,
        "te": f"{name} అత్యవసర సంప్రదింపుగా నమోదు చేయబడింది.",
        "hi": f"{name} को आपातकालीन संपर्क के रूप में जोड़ा गया है।"
    }

    return responses.get(lang, responses["en"])

def run_safety_emergency(lang):

    print("   [Emergency] 🆘 Emergency command detected")

    trigger_emergency()

    res = {
        'en': "Emergency alert has been sent to your caretakers.",
        'te': "మీ అత్యవసర సందేశం మీ కేర్‌టేకర్లకు పంపబడింది.",
        'hi': "आपातकालीन संदेश आपके संपर्कों को भेज दिया गया है।"
    }

    return get_lang_msg(res, lang)

