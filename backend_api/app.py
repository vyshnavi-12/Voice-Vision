import sys
import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module4.main import handle_interaction_logic, process_obstacle_check
from Module4.tts_engine import TextToSpeech

app  = Flask(__name__)
_tts = TextToSpeech()

def base64_to_cv2(b64_string):
    if not b64_string:
        return None
    try:
        img_data = base64.b64decode(b64_string)
        nparr    = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Image conversion error: {e}")
        return None

# ── Obstacle detection ────────────────────────────────────────────────────────
@app.route('/check_obstacle', methods=['POST'])
def check_obstacle():
    try:
        data  = request.json
        frame = base64_to_cv2(data.get('image'))
        if frame is None:
            return jsonify({"alert": False})
        result = process_obstacle_check(frame)
        return jsonify({"alert": result == "alert"})
    except Exception:
        return jsonify({"alert": False})

# ── Voice command ─────────────────────────────────────────────────────────────
@app.route('/process_command', methods=['POST'])
def process_command():
    try:
        data      = request.json
        audio_b64 = data.get('audio')
        if not audio_b64:
            return jsonify({"error": "No audio"}), 400

        audio_bytes = base64.b64decode(audio_b64)
        frame       = base64_to_cv2(data.get('image'))
        is_awake    = bool(data.get('is_awake', False))

        audio_response_bytes, intent, needs_more_info, ocr_guidance = \
            handle_interaction_logic(
                audio_bytes, frame,
                current_lang=data.get('lang', 'en'),
                is_awake=is_awake,
            )

        print(f"DEBUG intent → {intent} | needs_more_info → {needs_more_info} | ocr_guidance → {ocr_guidance}")

        if audio_response_bytes:
            return jsonify({
                "intent":            "EMERGENCY_REQUESTED" if intent == "EMERGENCY" else intent,
                "audio":             base64.b64encode(audio_response_bytes).decode('utf-8'),
                "needs_more_info":   needs_more_info,
                "ocr_needs_guidance": ocr_guidance,
            })

        return jsonify({"status": "idle", "intent": intent}), 204

    except Exception as e:
        print(f"Command processing error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── TTS endpoint (for "Going to sleep" etc.) ──────────────────────────────────
@app.route('/tts', methods=['POST'])
def tts_speak():
    try:
        data  = request.json
        text  = data.get('text', 'Going to sleep.')
        lang  = data.get('lang', 'en')
        audio = _tts.speak_to_bytes(text, lang_code=lang)
        if audio:
            return jsonify({"audio": base64.b64encode(audio).decode('utf-8')})
        return jsonify({"error": "TTS failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Emergency GPS ─────────────────────────────────────────────────────────────
@app.route('/trigger_emergency', methods=['POST'])
def trigger_emergency():
    try:
        data     = request.json
        location = data.get('location')
        if not location:
            return jsonify({"status": "error", "message": "No location"}), 400
        from Module5.emergency import trigger_emergency_sms
        trigger_emergency_sms(location)
        return jsonify({"status": "emergency_sent"})
    except Exception as e:
        print(f"Emergency trigger error: {e}")
        return jsonify({"status": "error"}), 500


# ── OCR frame check (no audio needed — called after guidance) ─────────────────
@app.route('/check_ocr', methods=['POST'])
def check_ocr():
    try:
        data  = request.json
        frame = base64_to_cv2(data.get('image'))
        lang  = data.get('lang', 'en')

        if frame is None:
            return jsonify({"error": "No frame"}), 400

        # Quick alignment check before running heavy OCR
        from Module3.ocr import get_text_bounding_box
        from backend_api.guidance import GuidanceSystem

        guidance_system = GuidanceSystem()
        guidance_system.update_frame_dims(frame)
        text_box  = get_text_bounding_box(frame)
        guide_msg = guidance_system.get_guidance(text_box)

        GUIDANCE_KEYWORDS = ["move the camera", "closer to", "camera to the",
                             "tilt the camera", "bring the camera"]

        if guide_msg != "OK":
            # Camera not aligned yet — give guidance
            audio_bytes = _tts.speak_to_bytes(guide_msg)
            audio_b64   = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None
            return jsonify({
                "response":           guide_msg,
                "audio":              audio_b64,
                "ocr_needs_guidance": True,
            })

        # Frame aligned — tell user we are processing, then run OCR
        scanning_msgs = {
            "en": "Frame aligned. Scanning the text now, please hold steady.",
            "te": "ఫ్రేమ్ సరిగ్గా ఉంది. టెక్స్ట్ స్కాన్ చేస్తున్నాను, అలాగే ఉండండి.",
            "hi": "फ्रेम सही है। अभी टेक्स्ट स्कैन कर रहा हूँ, स्थिर रहें।",
        }
        scanning_msg   = scanning_msgs.get(lang, scanning_msgs["en"])
        scanning_audio = _tts.speak_to_bytes(scanning_msg)

        # Run full OCR
        from Module3.ocr import run_ocr
        text = run_ocr(frame)

        if not text or text.strip() == "" or text == "I could not read any text.":
            no_text = {
                "en": "I could not read any text. Please try again.",
                "te": "టెక్స్ట్ చదవలేకపోయాను. మళ్లీ ప్రయత్నించండి.",
                "hi": "कोई टेक्स्ट नहीं पढ़ सका। कृपया पुनः प्रयास करें।",
            }
            response_text = no_text.get(lang, no_text["en"])
        else:
            intros = {
                "en": "The text reads: ",
                "te": "టెక్స్ట్ ఇలా ఉంది: ",
                "hi": "टेक्स्ट इस प्रकार है: ",
            }
            response_text = intros.get(lang, intros["en"]) + text

        result_audio = _tts.speak_to_bytes(response_text)

        # Combine scanning announcement + result into one audio response
        # by concatenating the raw MP3 bytes
        combined_audio = b""
        if scanning_audio:
            combined_audio += scanning_audio
        if result_audio:
            combined_audio += result_audio

        audio_b64 = base64.b64encode(combined_audio).decode('utf-8') if combined_audio else None

        return jsonify({
            "response":           response_text,
            "audio":              audio_b64,
            "ocr_needs_guidance": False,
        })
    except Exception as e:
        print(f"check_ocr error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)