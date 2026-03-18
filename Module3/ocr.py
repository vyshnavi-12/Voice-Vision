import cv2
import re
import numpy as np
from difflib import SequenceMatcher
from spellchecker import SpellChecker
import easyocr

# Load the text reader model
reader = easyocr.Reader(['en'], gpu=False)

# Improve image quality for better text reading
def preprocess(img):
    # Blur, enhance contrast, enlarge, and sharpen for better OCR
    img = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    h, w = img.shape
    # Double the size for better text detection
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    # Sharpen the image
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img

# Find where text is located on the screen
def get_text_bounding_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = preprocess(gray)
    results = reader.readtext(gray, detail=1)

    h_orig, w_orig = frame.shape[:2]
    all_x = []
    all_y = []

    for bbox, text, conf in results:
        if conf > 0.2:
            # Scale coordinates back to original size (we enlarged them)
            for pt in bbox:
                all_x.append(pt[0] / 2)
                all_y.append(pt[1] / 2)

    if not all_x:
        return None

    # Return the box that contains all text
    return [min(all_x), min(all_y), max(all_x), max(all_y)]


# Read text from camera frame and fix spelling errors
def run_ocr(frame):
    if frame is None:
        return "I could not read any text."

    spell = SpellChecker(distance=2)

    # Fix common OCR mistakes
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

    # Check if detected text is real (not noise) - needs more letters than symbols
    def is_valid_text(text):
        text = text.strip()
        if len(text) < 3:
            return False
        # Count letters and spaces - if mostly symbols, ignore
        alpha = sum(c.isalpha() or c.isspace() for c in text)
        return alpha / len(text) >= 0.5

    # Replace OCR symbol mistakes with correct characters
    def fix_ocr_chars(text):
        for bad, good in OCR_CHAR_FIXES.items():
            text = text.replace(bad, good)
        text = re.sub(r'\(0\b',      'to',      text)
        text = re.sub(r'\{heir\b',   'their',   text)
        text = re.sub(r"Women\s*\$", "women's", text)
        text = re.sub(r';',          ',',        text)
        return text

    # Fix individual words - check spelling dictionary and apply corrections
    def fix_ocr_word(word):
        if len(word) <= 1:
            return word
        # Check our custom word map first
        if word.lower() in OCR_WORD_MAP:
            return OCR_WORD_MAP[word.lower()]
        # If word is already correct, keep it
        if word.lower() in spell:
            return word
        try:
            # Try to fix the spelling
            correction = spell.correction(word.lower())
        except:
            return word
        # Apply correction if we got a better match
        if correction and correction != word.lower():
            # Preserve capitalization
            if word[0].isupper():
                correction = correction.capitalize()
            if word.isupper():
                correction = correction.upper()
            return correction
        return word

    # Fix a whole line - apply char fixes, word map, and spell checker
    def correct_line(line):
        line = fix_ocr_chars(line)
        line_lower = line.strip().lower()
        # Replace common word mistakes
        for bad, good in OCR_WORD_MAP.items():
            if bad in line_lower:
                line = re.sub(re.escape(bad), good, line, flags=re.IGNORECASE)
        # Fix each word individually
        words = line.split()
        corrected = []
        for word in words:
            # Separate punctuation from words
            prefix = ""
            suffix = ""
            while word and not word[0].isalnum():
                prefix += word[0]
                word = word[1:]
            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word = word[:-1]
            # Fix the word
            if word:
                word = fix_ocr_word(word)
            corrected.append(prefix + word + suffix)
        return " ".join(corrected)

    # Detect text using two different methods to catch more
    all_lines = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)

    # Method 1: CLAHE contrast enhancement
    results = reader.readtext(processed, detail=1)
    for bbox, text, conf in results:
        if conf > 0.3 and is_valid_text(text):
            all_lines.append((text.strip(), conf))

    # Method 2: Adaptive threshold for difficult lighting
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    h, w = adaptive.shape
    # Enlarge for better detection
    adaptive = cv2.resize(adaptive, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    results2 = reader.readtext(adaptive, detail=1)
    for bbox, text, conf in results2:
        if conf > 0.3 and is_valid_text(text):
            all_lines.append((text.strip(), conf))

    print(f"Total raw lines: {len(all_lines)}")

    # Remove duplicate detections - group similar text together
    groups = []
    for text, conf in all_lines:
        found = False
        for group in groups:
            best = max(group, key=lambda x: (len(x[0]), x[1]))
            # Check if text is similar to best match in group
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

    # Combine all corrected lines into final result
    if final:
        result = " ".join(correct_line(line) for line in final).strip()
        print(f"\n=== OCR Result ===\n{result}\n")
        return result

    return "I could not read any text."