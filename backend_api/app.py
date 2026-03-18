import sys
import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify

# Adding the parent directory to path so I can import my custom modules from Module4 and Module5
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module4.main import handle_interaction_logic, process_obstacle_check
from Module4.tts_engine import TextToSpeech

app  = Flask(__name__)
_tts = TextToSpeech() # Initializing my TTS engine globally to use in different routes

# Helper to turn the base64 string from the app into an image OpenCV can actually read
def base64_to_cv2(b64_string):
    if not b64_string:
        return None
    try:
        img_data = base64.b64decode(b64_string)
        nparr    = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Converting buffer to BGR image
    except Exception as e:
        print(f"Image conversion error: {e}")
        return None

# Route for real-time obstacle alerts
@app.route('/check_obstacle', methods=['POST'])
def check_obstacle():
    try:
        data  = request.json
        frame = base64_to_cv2(data.get('image'))
        if frame is None:
            return jsonify({"alert": False})
        
        # Calling my obstacle detection logic from Module4
        result = process_obstacle_check(frame)
        return jsonify({"alert": result == "alert"})
    except Exception:
        return jsonify({"alert": False})

# Main route that handles user voice commands and camera frames together
@app.route('/process_command', methods=['POST'])
def process_command():
    try:
        data      = request.json
        audio_b64 = data.get('audio')
        if not audio_b64:
            return jsonify({"error": "No audio"}), 400

        # Decoding audio and image sent from the mobile side
        audio_bytes = base64.b64decode(audio_b64)
        frame       = base64_to_cv2(data.get('image'))
        is_awake    = bool(data.get('is_awake', False))

        # This is the "brain" call - handles voice, intent, and vision logic in one go
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
                "audio":             base64.b64encode(audio_response_bytes).decode('utf-8'), # Converting audio back to b64 for the app
                "needs_more_info":   needs_more_info,
                "ocr_needs_guidance": ocr_guidance,
            })

        return jsonify({"status": "idle", "intent": intent}), 204

    except Exception as e:
        print(f"Command processing error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Endpoint to convert plain text to audio bytes (used for status updates)
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

# Emergency route - sends SMS with the user's GPS location
@app.route('/trigger_emergency', methods=['POST'])
def trigger_emergency():
    try:
        data     = request.json
        location = data.get('location')
        if not location:
            return jsonify({"status": "error", "message": "No location"}), 400
 
        # Get the original command text so we know who to send to
        from Module4.main import last_emergency_text
        from Module5.emergency import extract_target_from_command, trigger_emergency as _trigger
 
        target = extract_target_from_command(last_emergency_text)
 
        # Build location URL message
        lat = location.get("latitude")
        lon = location.get("longitude")
        maps_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon \
                   else "Location not available"
 
        result = _trigger(location=location, target=target)
 
        if target and target != "all":
            msg = f"Emergency alert sent to {target}."
        else:
            msg = "Emergency alert sent to all caretakers."
 
        audio = _tts.speak_to_bytes(msg)
        audio_b64 = base64.b64encode(audio).decode('utf-8') if audio else None
 
        return jsonify({
            "status": "emergency_sent",
            "audio":  audio_b64,
            "target": target or "all",
        })
 
    except Exception as e:
        print(f"Emergency trigger error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "error"}), 500


# Keywords used to check if the OCR needs the user to adjust the camera position
GUIDANCE_KEYWORDS = [
    "move the camera", "closer to", "camera to the",
    "tilt the camera", "bring the camera",
]

# Dedicated route for OCR - tells the user how to align the camera for better reading
@app.route('/check_ocr', methods=['POST'])
def check_ocr():
    try:
        data  = request.json
        frame = base64_to_cv2(data.get('image'))
        lang  = data.get('lang', 'en')

        if frame is None:
            return jsonify({"error": "No frame"}), 400

        import Module4.integrate_modules as modules
        response_text = modules.run_ocr_module(lang, frame)
        
        # Checking if the response contains instructions like "move closer"
        ocr_guidance  = any(kw in response_text.lower() for kw in GUIDANCE_KEYWORDS)

        if ocr_guidance:
            # If not aligned, just speak the instruction (e.g., "Move closer")
            audio_b64 = base64.b64encode(
                _tts.speak_to_bytes(response_text)
            ).decode('utf-8')
        else:
            # If aligned, give a status update then read the actual text
            scanning = {
                "en": "Frame aligned. Scanning now, please hold steady.",
                "te": "ఫ్రేమ్ సరిగ్గా ఉంది. స్కాన్ చేస్తున్నాను, అలాగే ఉండండి.",
                "hi": "फ्रेम सही है। स्कैन कर रहा हूँ, स्थिर रहें।",
            }.get(lang, "Frame aligned. Scanning now, please hold steady.")

            # Combining the "Scanning now" audio with the actual OCR result audio
            combined  = (_tts.speak_to_bytes(scanning) or b"") + \
                        (_tts.speak_to_bytes(response_text) or b"")
            audio_b64 = base64.b64encode(combined).decode('utf-8')

        return jsonify({
            "response":           response_text,
            "audio":              audio_b64,
            "ocr_needs_guidance": ocr_guidance,
        })

    except Exception as e:
        print(f"check_ocr error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on 0.0.0.0 so it's accessible over the local network (for the mobile app)
    app.run(host='0.0.0.0', port=5000, debug=False)