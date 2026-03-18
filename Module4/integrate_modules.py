import time
import sys
import os
import re

# Making sure the system can see all my sub-folders (Module1, Module2, etc.)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Module1.object_detection import detect_objects
from Module1.obstacle_detection import detect_obstacle
from Module1.scene_description import describe_scene
from Module1.navigation_assistance import navigate_to_object
from Module1.currency_recognition import detect_currency

from Module2.people_detection import count_people, describe_person
from Module2.face_detection import register_face, recognize_face

from Module3.ocr import run_ocr, get_text_bounding_box
from backend_api.guidance import GuidanceSystem

from Module5.phone_register import register_contact
from Module5.emergency import trigger_emergency


# --- 1. SYSTEM INITIALIZATION ---
print(" [System] 🚀 Assistive AI Hub: Multi-Module Integration Loaded (Mock Mode)")

def get_lang_msg(responses, lang_code):
    # Quick helper to grab the right translation based on the user's setting
    if "te" in lang_code: return responses['te']
    elif "hi" in lang_code: return responses['hi']
    return responses['en']

# --- 2. CORE MODULES  ---

def run_currency_detection(lang, frame):
    # Logic for identifying rupee notes
    print("   [Currency] 💰 Running currency detection...")

    currency = detect_currency(frame)

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

def run_realtime_scene_description(lang, frame):
    # Generates a natural language summary of what the camera sees
    print("   [Vision] 🌎 Running scene description...")
    description = describe_scene(frame)
    return description

def run_object_detection(lang, frame):
    # Lists out specific objects found in the frame
    print("   [Vision] 🔍 Running object detection...")

    objects = detect_objects(frame)

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
        # Formatting a list like "A, B and C" for better speech flow
        object_list = ", ".join(objects[:-1]) + " and " + objects[-1]

        responses = {
            "en": f"There are {object_list} in front of you.",
            "te": f"మీ ముందు {object_list} ఉన్నాయి.",
            "hi": f"आपके सामने {object_list} हैं।"
        }
        return responses.get(lang, responses["en"])

def run_obstacle_detection(frame):
    # Safety first: check if the user is about to walk into something
    obstacle = detect_obstacle(frame)
    return obstacle

def run_navigation_assistance(lang, frame, target_object):
    # Provides directional guidance (Left, Right, Straight)
    print("   [Navigation] 🧭 Running navigation assistance...")
    guidance = navigate_to_object(frame, target_object)
    return guidance

def run_people_count(lang, frame):
    # Specifically for counting how many humans are present
    print("   [People] 👥 Counting people...")
    count = count_people(frame)

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

def run_people_description(lang, frame):
    # Describe clothing, gender, or age of the person
    print("   [People] 🧑 Describing person...")
    message = describe_person(frame)
    return message

def run_face_recognition(lang, frame):
    # Compare detected face against our saved database
    print("   [Face] 👤 Running face recognition...")
    message = recognize_face(frame)

    if "Unknown" in message or "do not" in message:
        responses = {
            "en": message,
            "te": "ఈ వ్యక్తిని నేను గుర్తించలేకపోతున్నాను.",
            "hi": "मैं इस व्यक्ति को पहचान नहीं पा रहा हूँ।"
        }
    else:
        # Extract name and build a friendly sentence
        name = message.split()[0]
        responses = {
            "en": f"{name} is standing in front of you.",
            "te": f"{name} మీ ముందు నిలబడి ఉన్నారు.",
            "hi": f"{name} आपके सामने खड़े हैं।"
        }

    return responses.get(lang, responses["en"])

def run_face_registration(lang, name, frame):
    # Saving a new face so we can recognize them next time
    print("   [Face] 📸 Registering new face...")

    if frame is None:
        responses = {
            "en": "I could not see your face. Please try again.",
            "te": "ముఖం కనిపించలేదు. మళ్లీ ప్రయత్నించండి.",
            "hi": "चेहरा नहीं दिखा। कृपया पुनः प्रयास करें।"
        }
        return responses.get(lang, responses["en"])

    try:
        message = register_face(name, [frame]) 
    except Exception as e:
        print(f"Face registration error: {e}")
        return "Face registration failed. Please make sure your face is clearly visible."

    if message == "Success":
        responses = {
            "en": f"{name} has been registered successfully.",
            "te": f"{name} విజయవంతంగా నమోదు చేయబడింది.",
            "hi": f"{name} सफलतापूर्वक पंजीकृत किया गया है।"
        }
    else:
        responses = {
            "en": "No face detected. Please make sure your face is clearly visible.",
            "te": "ముఖం కనిపించలేదు. కెమెరాకు స్పష్టంగా కనిపించేలా ఉంచండి.",
            "hi": "चेहरा नहीं मिला। कृपया कैमरे के सामने स्पष्ट रूप से दिखें।"
        }
    return responses.get(lang, responses["en"])

def run_ocr_module(lang, frame):
    # Reading text from books, bills, or signs
    print("   [OCR] Running OCR on mobile frame...")
 
    if frame is None:
        no_frame = {
            "en": "I could not access the camera. Please try again.",
            "te": "కెమెరా అందుబాటులో లేదు. మళ్లీ ప్రయత్నించండి.",
            "hi": "कैमरा उपलब्ध नहीं है। कृपया पुनः प्रयास करें।",
        }
        return no_frame.get(lang, no_frame["en"])
 
    # Check if the paper is centered before wasting CPU on OCR
    guidance_system = GuidanceSystem()
    guidance_system.update_frame_dims(frame)
    text_box  = get_text_bounding_box(frame)
    guide_msg = guidance_system.get_guidance(text_box)
 
    if guide_msg != "OK":
        print(f"   [OCR] Guidance needed: {guide_msg}")
        return guide_msg 
 
    # Frame is perfect, now extract the text
    print("   [OCR] Frame aligned — running full OCR...")
    text = run_ocr(frame)
 
    if not text or text.strip() == "" or text == "I could not read any text.":
        no_text = {
            "en": "I could not read any text from this frame. Please try again.",
            "te": "ఈ ఫ్రేమ్ నుండి టెక్స్ట్ చదవలేకపోయాను. మళ్లీ ప్రయత్నించండి.",
            "hi": "इस फ्रेम से कोई टेक्स्ट नहीं पढ़ सका। कृपया पुनः प्रयास करें।",
        }
        return no_text.get(lang, no_text["en"])
 
    intros = {
        "en": "Frame is aligned. I have read the text. It says: ",
        "te": "ఫ్రేమ్ అలైన్ అయింది. నేను టెక్స్ట్ చదివాను. అది ఇలా చెప్తుంది: ",
        "hi": "फ्रेम सही है। मैंने टेक्स्ट पढ़ लिया। यह कहता है: ",
    }
    return intros.get(lang, intros["en"]) + text

def run_phone_registration(lang, name, phone):
    print("   [Safety] Registering emergency contact...")

    result = register_contact(name, phone)

    if result == "Success":
        import re
        digits_only = re.sub(r"\D", "", phone)
        return {
            "en": f"Done! {name} has been saved as an emergency contact with number {digits_only}.",
            "te": f"అయింది! {name} ని {digits_only} నంబర్ తో అత్యవసర సంప్రదింపుగా సేవ్ చేసాను.",
            "hi": f"हो गया! {name} को {digits_only} नंबर के साथ आपातकालीन संपर्क के रूप में सहेजा गया।",
        }.get(lang, f"Done! {name} has been saved as an emergency contact with number {digits_only}.")

    else:
        # register_contact returned "Invalid phone number."
        return {
            "en": f"The phone number {phone} is not valid. It should be 10 digits. Please try again.",
            "te": f"ఫోన్ నంబర్ {phone} చెల్లదు. 10 అంకెలు ఉండాలి. మళ్లీ చెప్పండి.",
            "hi": f"फोन नंबर {phone} मान्य नहीं है। 10 अंक होने चाहिए। कृपया दोबारा बताएं।",
        }.get(lang, f"The phone number {phone} is not valid. It should be 10 digits. Please try again.")

def run_safety_emergency(lang, location=None, user_text=None):
    print("   [Emergency] 🆘 Emergency command detected")

    from Module5.emergency import extract_target_from_command

    if location is not None:
        # GPS received — now actually send the SMS
        target = extract_target_from_command(user_text) if user_text else None
        result = trigger_emergency(location, target=target)

        if result == "Sucess":
            if target and target != "all":
                return {
                    "en": f"Emergency alert has been sent to {target}.",
                    "te": f"అత్యవసర హెచ్చరిక {target} కి పంపబడింది.",
                    "hi": f"आपातकालीन संदेश {target} को भेज दिया गया है।",
                }.get(lang, f"Emergency alert has been sent to {target}.")
            else:
                return {
                    "en": "Emergency alert has been sent to all your caretakers.",
                    "te": "అత్యవసర హెచ్చరిక మీ అందరు కేర్‌టేకర్లకు పంపబడింది.",
                    "hi": "आपातकालीन संदेश सभी संपर्कों को भेज दिया गया।",
                }.get(lang, "Emergency alert has been sent to all your caretakers.")
        else:
            return {
                "en": "No emergency contacts found. Please register a caretaker first.",
                "te": "అత్యవసర సంప్రదింపులు లేవు. ముందు కేర్‌టేకర్ నంబర్ నమోదు చేయండి.",
                "hi": "कोई संपर्क नहीं मिला। पहले केयरटेकर नंबर दर्ज करें।",
            }.get(lang, "No emergency contacts found. Please register a caretaker first.")

    else:
        # No GPS yet — ask for confirmation BEFORE sending anything
        # Frontend will show this as a follow-up question
        return {
            "en": "Who should I alert? Say a name, or say all caretakers.",
            "te": "ఎవరికి హెచ్చరిక పంపాలి? ఒక పేరు చెప్పండి, లేదా అందరికి అని చెప్పండి.",
            "hi": "किसे सूचित करूं? कोई नाम बताएं, या सभी केयरटेकर कहें।",
        }.get(lang, "Who should I alert? Say a name, or say all caretakers.")