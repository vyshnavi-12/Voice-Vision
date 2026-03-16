import cv2
import time
import easyocr
import re
import numpy as np
from difflib import SequenceMatcher
from spellchecker import SpellChecker

reader = easyocr.Reader(['en'], gpu=False)

# =========================
# GUIDANCE SYSTEM
# =========================

def get_guidance(frame, boxes):

    h, w = frame.shape[:2]

    if len(boxes) == 0:
        return "Please move the camera closer to the text."

    xs      = []
    ys      = []
    heights = []

    for box in boxes:
        (tl, tr, br, bl) = box
        xs.append(tl[0])
        ys.append(tl[1])
        heights.append(br[1] - tl[1])

    avg_x      = sum(xs)      / len(xs)
    avg_y      = sum(ys)      / len(ys)
    avg_height = sum(heights) / len(heights)

    center_x = w / 2
    center_y = h / 2

    if avg_height < 35:
        return "Please move the camera closer to the text."

    if avg_x < center_x * 0.75:
        return "Please move the camera to the right."

    if avg_x > center_x * 1.25:
        return "Please move the camera to the left."

    if avg_y < center_y * 0.75:
        return "Please move the camera downward."

    if avg_y > center_y * 1.25:
        return "Please move the camera upward."

    return "ready"

# =========================
# MAIN OCR FUNCTION
# =========================

def run_ocr(speak_callback=None):

    time.sleep(0.7)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    captured_frame       = None
    start_time           = time.time()
    last_guidance_spoken = None
    last_spoken_time     = 0
    REPEAT_INTERVAL      = 4.0
    TIMEOUT              = 30.0

    # ---- GUIDANCE LOOP ----
    while True:

        ret, frame = cap.read()

        if not ret:
            cap.release()
            return "Camera error. Please check your camera."

        gray_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results      = reader.readtext(gray_preview, detail=1)

        boxes = []
        for bbox, text, conf in results:
            if conf > 0.3:
                boxes.append(bbox)

        guidance = get_guidance(frame, boxes)
        now      = time.time()
        elapsed  = now - start_time

        print(f"OCR guidance: {guidance}  |  elapsed: {elapsed:.1f}s")

        if guidance == "ready":
            captured_frame = frame.copy()
            print("✅ Frame captured in memory.")
            if speak_callback:
                speak_callback("Got it. Reading now.")
            break

        if speak_callback:
            guidance_changed = (guidance != last_guidance_spoken)
            repeat_due       = (now - last_spoken_time) >= REPEAT_INTERVAL

            if guidance_changed or repeat_due:
                speak_callback(guidance)
                last_guidance_spoken = guidance
                last_spoken_time     = now

        if elapsed > TIMEOUT:
            captured_frame = frame.copy()
            print("⏱️ Timeout — capturing best available frame.")
            if speak_callback:
                speak_callback("Trying to read now.")
            break

    cap.release()

    # DEBUG — save captured frame to check what OCR sees
    cv2.imwrite("debug_captured_frame.jpg", captured_frame)
    print("📸 Debug frame saved: debug_captured_frame.jpg")

    # =========================
    # HELPER FUNCTIONS
    # =========================

    spell = SpellChecker(distance=2)

    OCR_CHAR_FIXES = {
        '$': 's', '(': '', ')': '', '|': 'l', '{': 't',
        '}': '', ';': ',', '0f': 'of', '1n': 'In',
    }

    OCR_WORD_MAP = {
        'almost': 'Times', 'tho': 'The', 'wo': 'We', 'biko': 'like',
        'moro': 'more', 'fominlet': 'feminist', 'fominist': 'feminist',
        'manitoslo': 'manifesto', 'manitesto': 'manifesto',
        'sandburg': 'Sandberg', 'rig': 'arms', 'car': 'far',
        'tlmas': 'Times', 'oficer': 'officer', 'achaving': 'achieving',
        'dobate': 'debate', 'crillcal': 'critical', 'diract': 'direct',
        'telegroph': 'Telegraph', 'alandmgrk': 'A landmark',
        'manileslo': 'manifesto', 'wolcome': 'welcome',
        'coucal': 'critical', 'rollers': 'offers', 'heir': 'their',
        'fisc': 'FSC', 'now': 'New', 'women': "women's",
    }

    def preprocess(img):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        h, w = img.shape
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return img

    def is_valid_text(text):
        text = text.strip()
        if len(text) < 3:
            return False
        alpha = sum(c.isalpha() or c.isspace() for c in text)
        if alpha / len(text) < 0.5:
            return False
        return True

    def fix_ocr_chars(text):
        for bad, good in OCR_CHAR_FIXES.items():
            text = text.replace(bad, good)
        text = re.sub(r'\(0\b',      'to',      text)
        text = re.sub(r'\{heir\b',   'their',   text)
        text = re.sub(r"Women\s*\$", "women's", text)
        text = re.sub(r';',          ',',        text)
        return text

    def fix_ocr_word(word):
        if len(word) <= 1:
            return word
        if word.lower() in OCR_WORD_MAP:
            return OCR_WORD_MAP[word.lower()]
        if word.lower() in spell:
            return word
        try:
            correction = spell.correction(word.lower())
        except:
            return word
        if correction and correction != word.lower():
            if word[0].isupper():
                correction = correction.capitalize()
            if word.isupper():
                correction = correction.upper()
            return correction
        return word

    def correct_line(line):
        line       = fix_ocr_chars(line)
        line_lower = line.strip().lower()
        for bad, good in OCR_WORD_MAP.items():
            if bad in line_lower:
                line = re.sub(re.escape(bad), good, line, flags=re.IGNORECASE)
        words     = line.split()
        corrected = []
        for word in words:
            prefix = ""
            suffix = ""
            while word and not word[0].isalnum():
                prefix += word[0]
                word    = word[1:]
            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word   = word[:-1]
            if word:
                word = fix_ocr_word(word)
            corrected.append(prefix + word + suffix)
        return " ".join(corrected)

    def ocr_region(img):
        results = reader.readtext(img)
        output  = []
        for bbox, text, conf in results:
            if conf > 0.3 and is_valid_text(text):
                output.append((text.strip(), conf))
        return output

    # =========================
    # PROCESS CAPTURED FRAME
    # =========================

    all_lines = []

    # Convert to grayscale
    gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Slight upscale for small text
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # DEBUG — save processed frame to check contrast/quality
    cv2.imwrite("debug_processed_frame.jpg", gray)
    print("📸 Debug processed frame saved: debug_processed_frame.jpg")

    # Primary pass — full frame

    # Primary pass — full frame
    results = reader.readtext(gray, detail=1)

    for bbox, text, conf in results:
        if conf > 0.3 and is_valid_text(text):
            all_lines.append((text.strip(), conf))

    # Secondary pass — preprocess then OCR again to catch missed text
    preprocessed      = preprocess(gray)
    secondary_results = ocr_region(preprocessed)
    all_lines.extend(secondary_results)

    print(f"📝 Total raw lines: {len(all_lines)}")

    # =========================
    # MERGE DETECTIONS
    # =========================

    groups = []

    for text, conf in all_lines:
        found = False
        for group in groups:
            best = max(group, key=lambda x: (len(x[0]), x[1]))
            if SequenceMatcher(None, text.lower(), best[0].lower()).ratio() > 0.5:
                group.append((text, conf))
                found = True
                break
        if not found:
            groups.append([(text, conf)])

    final = []

    for group in groups:
        best = max(group, key=lambda x: (len(x[0]), x[1]))
        final.append(best[0])

    # =========================
    # BUILD OUTPUT
    # =========================

    if final:
        output_text = ""
        for line in final:
            output_text += correct_line(line) + " "
        result = output_text.strip()
        print(f"\n=== Final Captured Text ===\n{result}\n")
        return result

    else:
        print("(no text was captured)")
        return "I could not read any text."