import sys
import os
import io
import requests
import numpy as np
import Module4.integrate_modules as modules
from Module4.intent_parser import IntentParser
from Module4.stt_engine import WhisperSTT
from Module4.tts_engine import TextToSpeech
from Module4.wakeword import WakeWordListener
from Module4.guidance import GuidanceSystem

parser      = IntentParser()
stt         = WhisperSTT()
tts         = TextToSpeech()
ww_listener = WakeWordListener()
guidance    = GuidanceSystem()

obstacle_detection_enabled = True

# ── Follow-up state ────────────────────────────────────────────────────────────
pending_intent     = None
pending_info_type  = None
pending_extra_data = {}   # accumulates answers across turns — must NOT be reset between turns
pending_frame      = None

GUIDANCE_KEYWORDS = [
    "move the camera", "closer to", "camera to the",
    "tilt the camera", "bring the camera",
]

def _is_guidance_response(text):
    if not text:
        return False
    return any(kw in text.lower() for kw in GUIDANCE_KEYWORDS)

def reset_pending_state():
    global pending_intent, pending_info_type, pending_extra_data, pending_frame
    pending_intent     = None
    pending_info_type  = None
    pending_extra_data = {}
    pending_frame      = None
    print("Pending state cleared.")

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

def process_obstacle_check(frame):
    global obstacle_detection_enabled
    if not obstacle_detection_enabled:
        return None
    obstacle = modules.run_obstacle_detection(frame)
    return "alert" if obstacle else None

def decode_to_pcm16(audio_bytes):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        return audio.raw_data
    except Exception as e:
        print(f"PCM decode error: {e}")
        return None

# ── handle_interaction_logic ───────────────────────────────────────────────────
# Returns: (audio_bytes, intent, needs_more_info, ocr_needs_guidance)
def handle_interaction_logic(audio_bytes, frame=None, current_lang="en", is_awake=False):
    global obstacle_detection_enabled
    global pending_intent, pending_info_type, pending_extra_data, pending_frame

    # ── SLEEPING: Porcupine checks for wake word ──────────────────────────────
    if not is_awake:
        pcm_bytes = decode_to_pcm16(audio_bytes)
        if pcm_bytes is None:
            return None, "IDLE", False, False
        if ww_listener.process_audio(pcm_bytes):
            obstacle_detection_enabled = False
            reset_pending_state()
            listening_msgs = {
                "en": "I am listening",
                "te": "నేను వింటున్నాను",
                "hi": "मैं सुन रहा हूँ",
            }
            msg = listening_msgs.get(current_lang, listening_msgs["en"])
            print("✅ Wake Word Detected by Porcupine — system AWAKE.")
            return tts.speak_to_bytes(msg), "WAKE_WORD_DETECTED", False, False
        return None, "IDLE", False, False

    # ── AWAKE: transcribe with Whisper ────────────────────────────────────────
    obstacle_detection_enabled = False

    user_text = stt.transcribe(audio_bytes)
    if not user_text or user_text.strip() == "":
        obstacle_detection_enabled = True
        return None, "UNKNOWN", False, False

    print(f"Transcribed: {user_text}")

    # ── FOLLOW-UP: pending state → treat transcription as the answer ──────────
    if pending_intent is not None:
        answer = user_text.strip().rstrip('!?.').strip()
        print(f"Follow-up answer: '{answer}' for '{pending_info_type}'")

        # ── KEY FIX: accumulate into existing extra_data, do NOT reset ────────
        pending_extra_data[pending_info_type] = answer

        intent_to_run = pending_intent
        extra         = dict(pending_extra_data)   # snapshot current accumulated data
        saved_frame   = pending_frame

        # Clear pending state BEFORE calling — module may set new pending
        pending_intent     = None
        pending_info_type  = None
        pending_extra_data = {}
        pending_frame      = None

        result = process_command("", current_lang, saved_frame,
                                 extra_data=extra, force_intent=intent_to_run)

        # Save new pending if module still needs more info
        # Pass existing extra so it continues accumulating
        _save_pending_if_needed(result, saved_frame, existing_extra=extra)

        ocr_guidance = (result["intent"] == "OCR" and
                        _is_guidance_response(result["response"]))
        return (tts.speak_to_bytes(result["response"]),
                result["intent"], result["needs_more_info"], ocr_guidance)

    # ── NORMAL COMMAND ────────────────────────────────────────────────────────
    intent, _ = parser.parse(user_text)
    result = process_command(user_text, current_lang, frame, intent_hint=intent)
    _save_pending_if_needed(result, frame)
    ocr_guidance = (result["intent"] == "OCR" and
                    _is_guidance_response(result["response"]))
    return (tts.speak_to_bytes(result["response"]),
            result["intent"], result["needs_more_info"], ocr_guidance)


def _save_pending_if_needed(result, frame, existing_extra=None):
    """
    Save follow-up state if module needs more info.
    Uses result["updated_extra"] first (cleaned by process_command),
    then falls back to existing_extra passed from caller.
    """
    global pending_intent, pending_info_type, pending_extra_data, pending_frame
    if result["needs_more_info"]:
        pending_intent    = result["intent"]
        pending_info_type = result["info_type"]
        # Prefer updated_extra from result (has bad data removed e.g. invalid phone)
        # Fall back to existing_extra if updated_extra is empty
        updated = result.get("updated_extra", {})
        pending_extra_data = updated if updated else (dict(existing_extra) if existing_extra else {})
        pending_frame     = frame
        print(f"Waiting for follow-up: {pending_intent} needs '{pending_info_type}' | have so far: {list(pending_extra_data.keys())}")


# ── process_command ────────────────────────────────────────────────────────────
def process_command(user_text, current_lang, frame=None,
                    extra_data=None, location=None,
                    force_intent=None, intent_hint=None):
    global obstacle_detection_enabled

    if force_intent:
        intent = force_intent
        target_lang = None
    elif intent_hint:
        intent = intent_hint
        target_lang = None
    else:
        intent, target_lang = parser.parse(user_text)

    response_text    = ""
    new_lang         = current_lang
    needs_more_info  = False
    info_needed_type = None

    obstacle_detection_enabled = False

    if intent == "SWITCH_LANGUAGE":
        if target_lang:
            new_lang = target_lang
            confirmations = {
                "en": "Language switched to English.",
                "te": "భాష తెలుగు కు మార్చబడింది.",
                "hi": "भाषा हिंदी में बदल दी गई है।",
            }
            response_text = confirmations.get(target_lang, "Language switched.")
        else:
            response_text = "Please specify the language."

    elif intent == "UNKNOWN":
        fallback = {
            "en": "I'm sorry, I didn't understand that. Please repeat.",
            "te": "క్షమించండి, నేను అర్థం చేసుకోలేకపోయాను. మళ్లీ చెప్పండి.",
            "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। कृपया दोबारा कहें।",
        }
        response_text = fallback.get(current_lang, fallback["en"])

    elif intent == "CURRENCY_DETECTION":
        response_text = modules.run_currency_detection(current_lang, frame)

    elif intent == "FACE_RECOGNITION":
        response_text = modules.run_face_recognition(current_lang, frame)

    elif intent == "REGISTER_FACE":
        if extra_data and "name" in extra_data:
            response_text = modules.run_face_registration(current_lang, extra_data["name"], frame)
        else:
            response_text = "What is the person's name?"
            needs_more_info, info_needed_type = True, "name"

    elif intent == "PEOPLE_COUNT":
        response_text = modules.run_people_count(current_lang, frame)

    elif intent == "PERSON_DESCRIPTION":
        response_text = modules.run_people_description(current_lang, frame)

    elif intent == "SCENE_DESCRIPTION":
        if check_internet():
            response_text = modules.run_realtime_scene_description(current_lang, frame)
        else:
            response_text = modules.run_object_detection(current_lang, frame)

    elif intent == "OBJECT_DETECTION":
        response_text = modules.run_object_detection(current_lang, frame)

    elif intent == "NAVIGATION":
        if extra_data and "target_object" in extra_data:
            response_text = modules.run_navigation_assistance(current_lang, frame, extra_data["target_object"])
        else:
            response_text = "What object should I look for?"
            needs_more_info, info_needed_type = True, "target_object"

    elif intent == "OCR":
        response_text = modules.run_ocr_module(current_lang, frame)

    elif intent == "EMERGENCY":
        response_text = modules.run_safety_emergency(current_lang, location)

    elif intent == "REGISTER_CONTACT":
        if extra_data and "name" in extra_data and "phone" in extra_data:
            # Both collected — attempt registration
            result_msg = modules.run_phone_registration(
                current_lang, extra_data["name"], extra_data["phone"])
            response_text = result_msg

            # If phone was invalid, stay awake and ask again
            invalid_keywords = ["not valid", "చెల్లదు", "मान्य नहीं"]
            if any(kw in result_msg for kw in invalid_keywords):
                # Keep name, clear phone — ask for phone again
                needs_more_info  = True
                info_needed_type = "phone"
                # Remove bad phone from extra_data so _save_pending carries only name
                extra_data.pop("phone", None)
        elif extra_data and "name" in extra_data:
            # Have name, need phone
            response_text = "Got it. Now tell me the phone number."
            needs_more_info, info_needed_type = True, "phone"
        else:
            # Need name first
            response_text = "Tell me the contact name."
            needs_more_info, info_needed_type = True, "name"

    elif intent == "STOP":
        stop_msgs = {
            "en": "Going to sleep.",
            "te": "నిద్రపోతున్నాను.",
            "hi": "सो रहा हूँ।",
        }
        response_text = stop_msgs.get(current_lang, stop_msgs["en"])

    obstacle_detection_enabled = True

    return {
        "response":        response_text,
        "language":        new_lang,
        "intent":          intent,
        "needs_more_info": needs_more_info,
        "info_type":       info_needed_type,
        "updated_extra":   extra_data or {},  # cleaned extra_data (bad phone removed etc.)
    }