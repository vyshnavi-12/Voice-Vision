# Evaluation/intent_parser.py

import re
import torch
from sentence_transformers import SentenceTransformer, util


class IntentParser:
    """
    Evaluation-only Intent Parser.
    Pure semantic intent classification.
    No language switching.
    No runtime side-effects.
    """

    def __init__(self):
        print(" 🧠 Loading Multilingual Semantic Brain...")

        try:
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        except Exception as e:
            print(f" ❌ Model Load Error: {e}")
            self.model = None
            return

        # -------------------------------
        # Intent Bank (Same as Runtime)
        # -------------------------------
        self.intent_bank = {
            "CURRENCY_DETECTION": [
                # ENGLISH (10+ variations)
                "scan the currency", "identify this money", "tell me the value of this note",
                "check which rupee note this is", "detect the currency in front of me",
                "what note is this", "currency value", "money identification", "note scanner",
                "rupee detection", "scan note", "read currency", "note value", "money scan",
                "currency check", "scan money", "currency scanner", "note reader", "money value",
                
                # TELUGU (15+ variations)
                "dabbu choodu", "dabbu enta", "idi ye note", "currency choodu", "note scan cheyi",
                "rupee note enti", "paisa choodu", "dabbu value", "note value cheppu",
                "currency detect cheyi", "money scan cheyi", "note gurthu pettuko",
                "ఇప్పుడే నోటును స్కాన్ చేయి", "డబ్బు చూడు", "ఇది ఏ నోటు",
                "కరెన్సీ చూడు", "నోటు వాల్యూ చెప్పు", "పైసా ఎంత",
                
                # HINDI (12+ variations)
                "paise pehchano", "yeh kaunsa note hai", "currency dikhao", "note ka value",
                "paise kitne hain", "currency check karo", "note scanner", "money identify",
                "पैसे पहचानो", "ये कौन सा नोट है", "करेंसी चेक करो", "नोट का वैल्यू",
                "पैसे कितने हैं", "मनी स्कैन करो", "करेंसी वैल्यू बताओ"
            ],

            "FACE_RECOGNITION": [
                # ENGLISH
                "who is this person", "who is in front of me", "recognize this face",
                "do you know him", "do you know her", "name this person", "identify him",
                "recognize her", "tell me who this is", "face recognition", "who are you seeing",
                
                # TELUGU  
                "tana peru enti", "evaru ithanu", "mukham gurthu unda", "ithanu evaru",
                "ఈ మనిషి పేరు ఏమిటి", "ముఖం గుర్తుందా", "ఇతను ఎవరు", "ఆమె పేరు చెప్పు",
                "ముఖం గుర్తించు", "ఫేస్ రికగ్నైజ్ చేయి",
                
                # HINDI
                "kaun hai ye", "chehra pehchano", "iska naam kya hai", "yeh kisne hai",
                "मुकाबला पहचानो", "इसका नाम क्या है", "ये कौन है", "चेहरा पहचानो",
                "फेस रिकग्निशन", "उसका नाम बताओ"
            ],

            "PEOPLE_DETECTION": [
                # ENGLISH
                "how many people are here", "are there people around", "is anyone standing there",
                "count the people", "people count", "number of people", "people nearby",
                "any people around", "how many persons", "people detection", "crowd size",
                
                # TELUGU
                "evaru unnaru", "entha mandi unnaru", "ikkada evaru unnaru", "manishi entha mandi",
                "ఎవరు ఉన్నారు", "ఎంత మంది ఉన్నారు", "ఇక్కడ ఎవరు ఉన్నారు",
                "మనుషులు ఎంత మంది", "ప్రజలు ఎంత మంది", "పీపుల్ కౌంట్",
                
                # HINDI
                "kitne log hain", "koi hai kya", "yahan kitne log hain", "log gin lo",
                "कितने लोग हैं", "कोई है क्या", "यहाँ कितने लोग हैं", "लोग गिनो",
                "पर्सन्स काउंट", "क्राउड साइज"
            ],

            "SCENE_DESCRIPTION": [
                # ENGLISH
                "describe the scene", "what is in front of me", "what is here", "look around",
                "scene description", "surroundings", "environment overview", "what do you see",
                "full scene", "area description", "describe my surroundings", "scene overview",
                
                # TELUGU
                "na mundhu em undhi", "scene vivarinchu", "em kanipistundi", "chuttu choodu",
                "సీన్ వివరించు", "నా ముందు ఏముంది", "ఏమి కనిపిస్తుంది",
                "పరిసరాలు చెప్పు", "ఎన్విరాన్మెంట్ డిస్క్రైబ్", "సీన్ ఓవర్వ్యూ",
                
                # HINDI
                "mere samne kya hai", "nazara batao", "kya dikh raha hai", "scene describe karo",
                "मेरे सामने क्या है", "नजारा बताओ", "क्या दिख रहा है", "दृश्य वर्णन",
                "परीस्थिति बताओ", "सुराउंडिंग्स डिस्क्राइब"
            ],

            "OBJECT_DETECTION": [
                # ENGLISH
                "what objects are here", "what is on the desk", "identify objects", "what am I holding",
                "find items", "objects around", "spot objects", "detect things", "list objects",
                "what things", "items detection", "object list", "things on table", "detect items",
                
                # TELUGU
                "desk paina em unnay", "vastuvulu cheppu", "objects choodu", "items list cheyi",
                "వస్తువులు చెప్పు", "డెస్క్ మీద ఏమున్నాయి", "ఆబ్జెక్ట్స్ చూడు",
                "ఐటెమ్స్ డిటెక్ట్", "థింగ్స్ లిస్ట్", "వస్తువుల డిటెక్షన్",
                
                # HINDI
                "mez par kya rakha hai", "cheezon ki pehchaan", "objects dikhao", "items gin lo",
                "मेज पर क्या रखा है", "चीजों की पहचान", "ऑब्जेक्ट्स दिखाओ", "आइटम्स गिनो",
                "चीजें लिस्ट करो", "ऑब्जेक्ट डिटेक्शन"
            ],

            "OBSTACLE_DETECTION": [
                # ENGLISH
                "is my path clear", "any obstacles", "is there anything in my way", "am i going to hit something",
                "obstacles ahead", "path blocked", "clear path", "something in front", "danger ahead",
                "check obstacles", "blockage detection", "path clear check",
                
                # TELUGU
                "addankulu unnaya", "darilo emaina addu undha", "path clear aa", "obstacle unda",
                "అడ్డంకులు ఉన్నాయా", "దారిలో ఏమైనా అడ్డు ఉందా", "పాత్ క్లియర్ ఆ",
                "ఆబ్స్టాకల్ ఉందా", "బ్లాకేజ్ చెక్", "డేంజర్ ఎహెడ్",
                
                # HINDI
                "kya rasta saaf hai", "koi rukawat hai kya", "raasta block hai", "obstacle check",
                "क्या रास्ता साफ है", "कोई रुकावट है क्या", "रास्ता ब्लॉक है", "ऑब्स्टेकल चेक",
                "पाथ क्लियर", "खतरा आगे"
            ],

            "NAVIGATION": [
                # ENGLISH
                "how do i get out", "where is the exit", "directions please", "guide me to the door",
                "show the way", "path directions", "exit directions", "guide me out", "navigation help",
                "where to go", "route guidance", "lead me out", "escape route",
                
                # TELUGU
                "biyataki ela vellali", "dhaari chupinchu", "exit ekkada", "navigation cheyi",
                "బయటకి ఎలా వెళ్లాలి", "ధారి చూపించు", "ఎగ్జిట్ ఎక్కడ", "నావిగేషన్ చేయి",
                "రూట్ గైడ్", "పత్ షో చేయి", "ఎస్కేప్ రూట్",
                
                # HINDI
                "rasta batao", "exit kahan hai", "rasta dikhao", "navigation karo", "guide karo",
                "रास्ता बताओ", "एग्जिट कहाँ है", "रास्ता दिखाओ", "नेविगेशन करो", "गाइड करो",
                "मार्गदर्शन दो", "एस्केप रूट"
            ],

            "OCR": [
                # ENGLISH
                "read text", "scan document", "read the bill", "what is written", "read this",
                "text scanner", "document reader", "bill reading", "scan text", "read writing",
                "ocr scan", "text recognition", "read label",
                
                # TELUGU
                "chaduvu", "text chadu", "kagitham chadu", "aksharalu chadu", "bill chaduvu",
                "టెక్స్ట్ చదువు", "కాగితం చదువు", "అక్షరాలు చదువు", "బిల్ చదువు",
                "ocr చేయి", "డాక్యుమెంట్ రీడ్", "లేబుల్ చదువు",
                
                # HINDI
                "padho", "kya likha hai", "ispe kya likha hai", "bill padho", "text padho",
                "पढ़ो", "क्या लिखा है", "इसपे क्या लिखा है", "बिल पढ़ो", "टेक्स्ट पढ़ो",
                "ओसीआर स्कैन", "डॉक्यूमेंट पढ़ो"
            ],

            "REGISTER_FACE": [
                # ENGLISH
                "register this person", "save this face", "remember him", "save her face",
                "register face", "store this person", "memorize face", "add to contacts",
                "save identity", "register identity",
                
                # TELUGU
                "ee manishi ni gurthu pettuko", "face save cheyu", "register cheyi", "save cheyi",
                "ఈ మనిషి ని గుర్తు పెట్టుకో", "ఫేస్ సేవ్ చేయి", "రిజిస్టర్ చేయి",
                "కాంటాక్ట్స్ లో యాడ్ చేయి", "ఐడెంటిటీ సేవ్",
                
                # HINDI
                "is chehre ko yaad rakho", "face save karo", "register karo", "store karo",
                "इस चेहरे को याद रखो", "फेस सेव करो", "रजिस्टर करो", "स्टोर करो",
                "कॉन्टैक्ट्स में ऐड करो"
            ],

            "EMERGENCY": [
                # ENGLISH
                "help me", "i am in danger", "sos", "save my life", "emergency", "help now",
                "urgent help", "danger", "rescue me", "emergency sos", "mayday",
                
                # TELUGU
                "nannu kapadu", "apadha", "emergency", "sos", "sahaayam", "urgent",
                "నన్ను కాపాడు", "అపَد", "ఎమర్జెన్సీ", "SOS", "సహాయం", "అత్యవసరం",
                
                # HINDI
                "bachao", "madad karo", "emergency", "sos", "khatera", "help",
                "बचाओ", "मदद करो", "इमरजेंसी", "SOS", "खतरा", "हेल्प"
            ],

            "STOP": [
                # ENGLISH
                "stop", "exit", "sleep", "stop now", "end", "quit", "shut down", "go sleep",
                "stop talking", "finish", "cancel", "close", "power off",
                
                # TELUGU
                "aapu", "pani aipoyindi", "exit", "stop cheyi", "close cheyi", "band cheyi",
                "ఆపు", "పని అయిపోయింది", "ఎగ్జిట్", "స్టాప్ చేయి", "క్లోజ్ చేయి",
                "బంద్ చేయి", "షట్ డౌన్",
                
                # HINDI
                "chup", "band karo", "exit", "stop karo", "finish karo", "so jao",
                "चुप", "बंद करो", "एग्जिट", "स्टॉप करो", "फिनिश करो", "सो जाओ",
                "पावर ऑफ", "क्लोज करो"
            ]
        }

        # -------------------------------
        # Precompute embeddings
        # -------------------------------
        self.corpus_embeddings = {}

        for intent, phrases in self.intent_bank.items():
            self.corpus_embeddings[intent] = self.model.encode(
                phrases,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        print(" ✅ Intent Brain Ready.")

    def parse(self, text):
        if not text or self.model is None:
            return "UNKNOWN"

        text = text.lower().strip()
        user_embedding = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

        best_intent = "UNKNOWN"
        best_score = 0.0
        scores = {}

        for intent, intent_vectors in self.corpus_embeddings.items():
            cosine_scores = util.cos_sim(user_embedding, intent_vectors)
            score = torch.max(cosine_scores).item()
            scores[intent] = score
            
            if score > best_score:
                best_score = score
                best_intent = intent

        print(f"🧠 '{text}' → {best_intent} ({best_score:.3f})")

        # DYNAMIC THRESHOLD: Accept if top score > 0.45 AND 10% margin
        if best_score > 0.45 and (best_score - max(scores.values()) * 0.9) > 0.05:
            return best_intent

        return "UNKNOWN"
    
    def parse_noisy(self, text):
        """Specialized for noisy STT output"""
        if not text or self.model is None:
            return "UNKNOWN"
        
        # Clean STT garbage
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        if len(text.split()) < 2:  # Too short → UNKNOWN
            return "UNKNOWN"
        
        # Use same logic but LOWER threshold for noisy input
        user_embedding = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        
        best_intent = "UNKNOWN"
        best_score = 0.0
        
        for intent, intent_vectors in self.corpus_embeddings.items():
            cosine_scores = util.cos_sim(user_embedding, intent_vectors)
            score = torch.max(cosine_scores).item()
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        # Noisy threshold: Accept ANYTHING > 0.35
        if best_score > 0.35:
            return best_intent
            
        return "UNKNOWN"

