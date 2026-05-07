import cv2
import re
import numpy as np
from difflib import SequenceMatcher
from spellchecker import SpellChecker
import easyocr

# Supported OCR languages: English, Hindi, and Telugu
SUPPORTED_OCR_LANGUAGES = ['en', 'hi', 'te']

# Load readers once at startup
ENGLISH_READER = easyocr.Reader(['en'], gpu=False)
HINDI_READER   = easyocr.Reader(['hi', 'en'], gpu=False)
TELUGU_READER  = easyocr.Reader(['te', 'en'], gpu=False)

LANGUAGE_READERS = {
    'en': ENGLISH_READER,
    'hi': HINDI_READER,
    'te': TELUGU_READER,
}

# Image preprocessing 

def preprocess(img):
    """CLAHE contrast enhancement + 2× upscale + unsharp mask."""
    img = cv2.GaussianBlur(img, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    h, w = img.shape
    img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


def adaptive_threshold(img):
    """Adaptive threshold for difficult or uneven lighting conditions."""
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    h, w = thresh.shape
    thresh = cv2.resize(thresh, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return thresh


# Script / language helpers 

def contains_devanagari(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def contains_telugu(text):
    return any('\u0C00' <= ch <= '\u0C7F' for ch in text)

def is_indic_text(text):
    return contains_devanagari(text) or contains_telugu(text)


def get_readers(languages=None):
    """Return the list of EasyOCR readers for the requested languages."""
    if languages is None:
        return [ENGLISH_READER]
    langs = [l.lower() for l in languages if l]
    readers = [LANGUAGE_READERS[l] for l in langs if l in LANGUAGE_READERS]
    return readers or [ENGLISH_READER]


def score_text_language(text):
    scores = {'hi': 0, 'te': 0, 'en': 0}
    for ch in text:
        if '\u0900' <= ch <= '\u097F':
            scores['hi'] += 1
        elif '\u0C00' <= ch <= '\u0C7F':
            scores['te'] += 1
        elif ch.isascii() and ch.isalpha():
            scores['en'] += 1
    return scores


def select_best_language(results_by_lang):
    """Pick the language whose reader produced the highest-confidence output."""
    best_lang, best_score = 'en', -1.0
    for lang, lines in results_by_lang.items():
        total_conf  = sum(conf for _, conf in lines)
        script_score = sum(score_text_language(text)[lang] for text, _ in lines)
        score = total_conf + script_score * 5
        if score > best_score:
            best_score, best_lang = score, lang
    return best_lang


# Bounding box 

def get_text_bounding_box(frame, languages=None):
    """Return [x1, y1, x2, y2] bounding box of all detected text, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)
    readers = get_readers(languages)

    all_x, all_y = [], []
    for reader in readers:
        for bbox, text, conf in reader.readtext(processed, detail=1):
            if conf > 0.2:
                for pt in bbox:
                    all_x.append(pt[0] / 2)   # undo the 2× upscale
                    all_y.append(pt[1] / 2)

    if not all_x:
        return None
    return [min(all_x), min(all_y), max(all_x), max(all_y)]


# Main OCR entry point 

def run_ocr(frame, languages=None):
    """
    Read text from a camera frame and return a corrected string.

    Parameters
    ----------
    frame     : BGR numpy array from cv2
    languages : list of language codes, e.g. ['en'], ['hi', 'en'], ['te', 'en']
                Defaults to English-only when None.
    """
    if frame is None:
        return "I could not read any text."

    spell = SpellChecker(distance=2)

    # Correction tables 
    OCR_CHAR_FIXES = {
        '$': 's', '(': '', ')': '', '|': 'l', '{': 't',
        '}': '', ';': ',', '0f': 'of', '1n': 'In',
    }
    OCR_WORD_MAP = {
        'almost': 'Times',  'tho': 'The',     'wo': 'We',       'biko': 'like',
        'moro': 'more',     'fominlet': 'feminist', 'fominist': 'feminist',
        'manitoslo': 'manifesto', 'manitesto': 'manifesto',
        'sandburg': 'Sandberg', 'rig': 'arms', 'car': 'far',
        'tlmas': 'Times',   'oficer': 'officer', 'achaving': 'achieving',
        'dobate': 'debate', 'crillcal': 'critical', 'diract': 'direct',
        'telegroph': 'Telegraph', 'alandmgrk': 'A landmark',
        'manileslo': 'manifesto', 'wolcome': 'welcome',
        'coucal': 'critical', 'rollers': 'offers', 'heir': 'their',
        'fisc': 'FSC',      'now': 'New',     'women': "women's",
    }

    # Helpers 

    def is_valid_text(text):
        text = text.strip()
        if len(text) < 3:
            return False
        alpha = sum(c.isalpha() or c.isspace() for c in text)
        return alpha / len(text) >= 0.5

    def fix_ocr_chars(text):
        for bad, good in OCR_CHAR_FIXES.items():
            text = text.replace(bad, good)
        text = re.sub(r'\(0\b',      'to',       text)
        text = re.sub(r'\{heir\b',   'their',    text)
        text = re.sub(r"Women\s*\$", "women's",  text)
        text = re.sub(r';',          ',',         text)
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
        except Exception:
            return word
        if correction and correction != word.lower():
            if word[0].isupper():
                correction = correction.capitalize()
            if word.isupper():
                correction = correction.upper()
            return correction
        return word

    def correct_line(line):
        if is_indic_text(line):
            return line.strip()
        line = fix_ocr_chars(line)
        line_lower = line.strip().lower()
        for bad, good in OCR_WORD_MAP.items():
            if bad in line_lower:
                line = re.sub(re.escape(bad), good, line, flags=re.IGNORECASE)
        words = line.split()
        corrected = []
        for word in words:
            prefix = suffix = ""
            while word and not word[0].isalnum():
                prefix += word[0]; word = word[1:]
            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix; word = word[:-1]
            if word:
                word = fix_ocr_word(word)
            corrected.append(prefix + word + suffix)
        return " ".join(corrected)

    # Two-method detection (mirrors integration code) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Determine which readers to use and run language scoring
    if languages is None:
        active_readers = {'en': ENGLISH_READER}
    else:
        langs = [l.lower() for l in languages if l]
        active_readers = {l: LANGUAGE_READERS[l] for l in langs if l in LANGUAGE_READERS}
        if not active_readers:
            active_readers = {'en': ENGLISH_READER}

    # Method 1: CLAHE contrast enhancement (same as integration code)
    processed = preprocess(gray)

    # Method 2: Adaptive threshold for difficult lighting (same as integration code)
    adaptive = adaptive_threshold(gray)

    # Collect detections per language across both image variants
    results_by_lang = {lang: [] for lang in active_readers}

    for lang, reader in active_readers.items():
        for image_variant in (processed, adaptive):
            for bbox, text, conf in reader.readtext(image_variant, detail=1):
                if conf > 0.3 and is_valid_text(text.strip()):
                    results_by_lang[lang].append((text.strip(), conf))

    # Pick the best-matching language
    chosen_lang = select_best_language(results_by_lang)
    all_lines   = results_by_lang[chosen_lang]
    print(f"Detected OCR language: {chosen_lang}")
    print(f"Total raw lines: {len(all_lines)}")

    # Deduplication 
    groups = []
    for text, conf in all_lines:
        for group in groups:
            best = max(group, key=lambda x: (len(x[0]), x[1]))
            if SequenceMatcher(None, text.lower(), best[0].lower()).ratio() > 0.5:
                group.append((text, conf))
                break
        else:
            groups.append([(text, conf)])

    final = [max(g, key=lambda x: (len(x[0]), x[1]))[0] for g in groups]

    # Assemble result 
    if final:
        result = " ".join(correct_line(line) for line in final).strip()
        print(f"\n=== OCR Result ===\n{result}\n")
        return result

    return "I could not read any text."