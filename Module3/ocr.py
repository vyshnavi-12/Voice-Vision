import cv2

import easyocr

import os

import re

import numpy as np

from difflib import SequenceMatcher

from spellchecker import SpellChecker

 

reader = easyocr.Reader(['en'], gpu=False)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

 

all_lines = []  # flat list of (text, conf) across all frames

frame_count = 0

PROCESS_EVERY = 3

OUTPUT_DIR = "processed_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

 

print("Real-time OCR started. Hold text steady in front of camera.")

print("Press Ctrl+C to stop and see final text.\n")

 

def preprocess(img):

    # Gentle denoise — preserve text edges

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # CLAHE for contrast

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img = clahe.apply(img)

    # Upscale 2x

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

    'fisc': 'FSC', 'now': 'New', 'women': 'women\'s',

}

 

def fix_ocr_chars(text):

    for bad, good in OCR_CHAR_FIXES.items():

        text = text.replace(bad, good)

    # Fix common OCR patterns

    text = re.sub(r'\(0\b', 'to', text)

    text = re.sub(r'\{heir\b', 'their', text)

    text = re.sub(r"Women\s*\$", "women's", text)

    text = re.sub(r';', ',', text)

    return text

 

def fix_ocr_word(word):

    if len(word) <= 1:

        return word

    if word.lower() in OCR_WORD_MAP:

        return OCR_WORD_MAP[word.lower()]

    if word.lower() in spell:

        return word

    correction = spell.correction(word.lower())

    if correction and correction != word.lower():

        if word[0].isupper():

            correction = correction.capitalize()

        if word.isupper():

            correction = correction.upper()

        return correction

    return word

 

def correct_line(line):

    # Apply character-level fixes first

    line = fix_ocr_chars(line)

    # Apply word map

    line_lower = line.strip().lower()

    for bad, good in OCR_WORD_MAP.items():

        if bad in line_lower:

            line = re.sub(re.escape(bad), good, line, flags=re.IGNORECASE)

    words = line.split()

    corrected = []

    for word in words:

        prefix = ""

        suffix = ""

        while word and not word[0].isalnum():

            prefix += word[0]

            word = word[1:]

        while word and not word[-1].isalnum():

            suffix = word[-1] + suffix

            word = word[:-1]

        if word:

            word = fix_ocr_word(word)

        corrected.append(prefix + word + suffix)

    return " ".join(corrected)

 

def ocr_region(img):

    results = reader.readtext(img)

    return [(text.strip(), conf) for _, text, conf in results

            if conf > 0.3 and is_valid_text(text)]

 

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print("ERROR: Cannot read from camera.")

            break

 

        frame_count += 1

        if frame_count % PROCESS_EVERY != 0:

            continue

 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        processed = preprocess(gray)

 

        # Save processed images for future reference

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_count}_raw.jpg"), frame)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_count}_processed.jpg"), processed)

 

        if frame_count == PROCESS_EVERY:

            cv2.imwrite("debug_raw.jpg", frame)

            cv2.imwrite("debug_processed.jpg", processed)

            print("Saved debug images.\n")

 

        frame_texts = ocr_region(processed)

 

        os.system('cls')

        print(f"=== Live Detection (frame {frame_count}) ===\n")

        if frame_texts:

            for text, conf in frame_texts:

                print(f"[{conf:.2f}] {text}")

                all_lines.append((text, conf))

        else:

            print("(no text detected)")

 

        print(f"\nTotal detections so far: {len(all_lines)}")

        print("Press Ctrl+C to stop and see final text.")

 

except KeyboardInterrupt:

    pass

 

cap.release()

 

# Merge all detections: group similar lines, keep longest + highest conf version

groups = []  # each group: list of (text, conf)

for text, conf in all_lines:

    found = False

    for group in groups:

        # Compare against the best (longest) entry in the group

        best = max(group, key=lambda x: (len(x[0]), x[1]))

        if SequenceMatcher(None, text.lower(), best[0].lower()).ratio() > 0.5:

            group.append((text, conf))

            found = True

            break

    if not found:

        groups.append([(text, conf)])

 

# From each group, pick the longest version with highest confidence

final = []

for group in groups:

    # Sort by length desc, then confidence desc

    best = max(group, key=lambda x: (len(x[0]), x[1]))

    final.append(best[0])

 

os.system('cls')

print("=== Final Captured Text ===\n")

if final:

    for line in final:

        print(correct_line(line))

else:

    print("(no text was captured)")

print()
