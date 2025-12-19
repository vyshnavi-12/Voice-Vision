# Module4/mock_modules.py
import time

def run_vision_module(lang):
    """Simulates detecting objects"""
    print(f"   [Mock Vision] 📸 Capturing Image & Analyzing...")
    time.sleep(1) # Simulate processing time
    
    if "te" in lang:
        return "మీ ముందు ఒక కుర్చీ మరియు టేబుల్ ఉన్నాయి." 
    elif "hi" in lang:
        return "आपके सामने एक कुर्सी और मेज़ है।" 
    else:
        return "There is a chair and a table in front of you."

def run_ocr_module(lang):
    """Simulates reading text"""
    print(f"   [Mock OCR] 📄 Scanning Text...")
    time.sleep(1)
    
    if "te" in lang:
        return "బిల్లు మొత్తం 500 రూపాయలు."
    elif "hi" in lang:
        return "कुल बिल 500 रुपये है।"
    else:
        return "The total bill amount is 500 rupees."

def run_people_module(lang):
    """Simulates face recognition"""
    print(f"   [Mock Face] 👤 Scanning Faces...")
    time.sleep(1)
    
    if "te" in lang:
        return "నాకు ఎవరూ కనిపించడం లేదు." 
    elif "hi" in lang:
        return "मुझे कोई नहीं दिख रहा।" 
    else:
        return "I do not see anyone I know."