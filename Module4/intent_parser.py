# intent_parser.py

import os
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
        
        # Point to the folder you created with the download script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        local_model_path = os.path.join(
            PROJECT_ROOT,
            "models",
            "paraphrase-multilingual-mpnet-base-v2"
        )

        if not os.path.exists(local_model_path):
            raise FileNotFoundError(
                f"❌ Error: Could not find model at {local_model_path}.\n"
                "Make sure you ran download_models.py first!"
            )

        self.model = SentenceTransformer(local_model_path)

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

            # ---------------- ENGLISH ----------------
            "who is in front of me", "who is this person", "do you know this person",
            "recognize this face", "identify this person", "who am i looking at",
            "tell me who this is", "do you recognize him", "do you recognize her",
            "whose face is this",

            # ---------------- TELUGU ----------------
            "ఇతను ఎవరు", "ఆమె ఎవరు", "నా ముందు ఎవరు ఉన్నారు",
            "ఈ వ్యక్తి ఎవరు", "ఈ ముఖం ఎవరిది", "ముఖం గుర్తించు",
            "ఈ మనిషి పేరు చెప్పు", "ఈ వ్యక్తిని గుర్తించగలవా",

            # ---------------- HINDI ----------------
            "ये कौन है", "मेरे सामने कौन है", "इस व्यक्ति को पहचानो",
            "इसका नाम क्या है", "क्या तुम इसे पहचानते हो", "चेहरा पहचानो",
            "ये व्यक्ति कौन है", "बताओ यह कौन है"
            ],

            "PEOPLE_COUNT": [

            # ---------------- ENGLISH ----------------
            "how many people are here", "count the people", "how many persons are there",
            "number of people", "are there people around", "how many people are in front of me",
            "is anyone standing there", "people count",

            # ---------------- TELUGU ----------------
            "ఎంత మంది ఉన్నారు", "ఇక్కడ ఎంత మంది ఉన్నారు", "నా ముందు ఎంత మంది ఉన్నారు",
            "ఎవరు ఉన్నారు", "మనుషులు ఎంత మంది", "ప్రజలు ఎంత మంది",

            # ---------------- HINDI ----------------
            "कितने लोग हैं", "यहाँ कितने लोग हैं", "मेरे सामने कितने लोग हैं",
            "लोग गिनो", "लोग कितने हैं", "क्या यहाँ कोई है"
            ],

            "PERSON_DESCRIPTION": [

            # ---------------- ENGLISH ----------------
            "describe the person", "what does the person look like", "describe the person in front of me",
            "tell me about the person", "what is the person wearing", "describe him",
            "describe her", "how does the person look",

            # ---------------- TELUGU ----------------
            "ఈ వ్యక్తిని వివరించు", "ఆ వ్యక్తి ఎలా ఉన్నాడు", "ఆమె ఎలా కనిపిస్తోంది",
            "ఆ మనిషి గురించి చెప్పు", "వాడు ఎలా ఉన్నాడు", "వాళ్లు ఎలా ఉన్నారు",

            # ---------------- HINDI ----------------
            "इस व्यक्ति का वर्णन करो", "यह व्यक्ति कैसा दिखता है", "उस व्यक्ति के बारे में बताओ",
            "वह क्या पहन रहा है", "उसका वर्णन करो","यह आदमी कैसा दिखता है"
            ],

            "SCENE_DESCRIPTION": [
                # ENGLISH
                "describe the scene", "what is in front of me", "look around",
                "scene description", "surroundings", "environment overview", "what do you see",
                "full scene description", "area description", "describe my surroundings", "scene overview",
                
                # TELUGU
                "na mundhu em undhi", "scene vivarinchu", "em kanipistundi", "chuttu choodu",
                "సీన్ వివరించు", "నా ముందు ఏముంది", "ఏమి కనిపిస్తుంది",
                "పరిసరాలు చెప్పు", "ఎన్విరాన్మెంట్ డిస్క్రైబ్", "సీన్ ఓవర్వ్యూ",
                
                # HINDI
                "mere samne kya hai", "nazara batao", "kya dikh raha hai", "scene describe karo",
                "मेरे सामने क्या है", "नजारा बताओ", "क्या दिख रहा है", "दृश्य वर्णन",
                "परीस्थिति बताओ", "सुराउंडिंग्स डिस्क्राइब",

                # Obstacle Detection - ENGLISH
                "is my path clear", "any obstacles", "is there anything in my way", "am i going to hit something",
                "obstacles ahead", "path blocked", "clear path", "something in front", "danger ahead",
                "check obstacles", "blockage detection", "path clear check",
                
                # Obstacle Detection - TELUGU
                "addankulu unnaya", "darilo emaina addu undha", "path clear aa", "obstacle unda",
                "అడ్డంకులు ఉన్నాయా", "దారిలో ఏమైనా అడ్డు ఉందా", "పాత్ క్లియర్ ఆ",
                "ఆబ్స్టాకల్ ఉందా", "బ్లాకేజ్ చెక్", "డేంజర్ ఎహెడ్",
                
                # Obstacle Detection - HINDI
                "kya rasta saaf hai", "koi rukawat hai kya", "raasta block hai", "obstacle check",
                "क्या रास्ता साफ है", "कोई रुकावट है क्या", "रास्ता ब्लॉक है", "ऑब्स्टेकल चेक",
                "पाथ क्लियर", "खतरा आगे"
            ],

            "OBJECT_DETECTION": [

                # ENGLISH
                "what objects are here", "what objects are in front of me", "what objects do you see",
                "identify objects", "detect objects", "detect objects in front", "find objects",
                "list objects around me", "what items are here", "what things are here",
                "what object is this", "what objects are nearby", "tell me the objects around me",

                # TELUGU
                "desk paina em unnay", "vastuvulu cheppu", "objects choodu", "items list cheyi",
                "వస్తువులు చెప్పు", "డెస్క్ మీద ఏమున్నాయి", "ఆబ్జెక్ట్స్ చూడు", "ఐటెమ్స్ డిటెక్ట్",
                "వస్తువులు గుర్తించు",

                # HINDI
                "mez par kya rakha hai", "cheezon ki pehchaan", "objects dikhao",
                "items gin lo", "मेज पर क्या रखा है", "चीजों की पहचान", "ऑब्जेक्ट्स दिखाओ",
                "आसपास की चीजें बताओ", "कौन से ऑब्जेक्ट हैं"
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

            "FACE_RECOGNITION": [

            # ---------------- ENGLISH ----------------
            "who is in front of me", "who is this person", "do you know this person",
            "recognize this face", "identify this person", "who am i looking at",
            "tell me who this is", "do you recognize him", "do you recognize her",
            "whose face is this", "who is standing here",
            "who is standing in front of me", "who is this",

            # ---------------- TELUGU ----------------
            "ఇతను ఎవరు", "ఆమె ఎవరు", "నా ముందు ఎవరు ఉన్నారు",
            "ఈ వ్యక్తి ఎవరు", "ఈ ముఖం ఎవరిది", "ఈ మనిషి పేరు చెప్పు",
            "ఈ వ్యక్తిని గుర్తించగలవా", "ఈ వ్యక్తి ఎవరో చెప్పు", "నా ముందు నిలబడి ఉన్న వ్యక్తి ఎవరు",
            "ఈ ముఖం ఎవరిదో చెప్పు",

            # ---------------- HINDI ----------------
            "ये कौन है", "मेरे सामने कौन है", "इस व्यक्ति को पहचानो",
            "इसका नाम क्या है", "क्या तुम इसे पहचानते हो", "चेहरा पहचानो",
            "यह व्यक्ति कौन है", "बताओ यह कौन है", "मेरे सामने खड़ा व्यक्ति कौन है",
            "इसका चेहरा पहचानो"
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

            "REGISTER_CONTACT": [

                # ---------------- ENGLISH ----------------
                "register contact", "add emergency contact", "save a phone number",
                "add caretaker number", "save this number", "register caretaker",
                "store emergency contact", "add contact for help", "save emergency number",
                "add new contact",

                # ---------------- TELUGU ----------------
                "సంప్రదింపు నమోదు చేయి", "కాంటాక్ట్ సేవ్ చేయి", "అత్యవసర సంప్రదింపు జోడించు",
                "ఈ నంబర్ సేవ్ చేయి", "కేర్‌టేకర్ నంబర్ సేవ్ చేయి", "కాంటాక్ట్ యాడ్ చేయి",
                "సహాయం కోసం నంబర్ సేవ్ చేయి", "ఫోన్ నంబర్ నమోదు చేయి",

                # ---------------- HINDI ----------------
                "संपर्क जोड़ो", "आपातकालीन संपर्क जोड़ो", "फोन नंबर सेव करो",
                "केयरटेकर नंबर सेव करो", "नया संपर्क जोड़ो", "इस नंबर को सेव करो",
                "इमरजेंसी कॉन्टैक्ट जोड़ो", "संपर्क पंजीकृत करो"
            ],

            "EMERGENCY": [

                # ---------------- ENGLISH ----------------
                "help me", "i need help", "i am in danger",
                "emergency", "sos", "call for help",
                "send emergency alert", "please help me",
                "urgent help", "danger help",

                # ---------------- TELUGU ----------------
                "నన్ను కాపాడు", "సహాయం కావాలి", "ఎమర్జెన్సీ",
                "నాకు సహాయం చేయండి", "అపాయం ఉంది",
                "అత్యవసర సహాయం", "దయచేసి సహాయం చేయండి",

                # ---------------- HINDI ----------------
                "मदद करो", "मुझे मदद चाहिए", "मैं खतरे में हूँ",
                "इमरजेंसी", "तुरंत मदद करो", "आपातकालीन मदद",
                "कृपया मदद करो"
            ],

            "SWITCH_LANGUAGE": [

                # English
                "switch to telugu", "change to telugu", "speak in telugu",
                "switch to hindi", "change to hindi", "speak in hindi",
                "switch to english", "change to english", "speak in english",

                # Telugu
                "telugu ki marchu", "hindi ki marchu", "english ki marchu",
                "telugu lo matlaadu", "hindi lo matlaadu", "english lo matlaadu",

                # Hindi
                "telugu mein switch karo", "hindi mein switch karo",
                "english mein bolo", "language badlo", "bhasha badlo"
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
        """
        Unified semantic intent parser.
        Returns:
            (intent_name, target_language or None)
        """

        if not text or self.model is None:
            return ("UNKNOWN", None)

        text = text.lower().strip()

        # Encode user input
        user_embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        best_intent = "UNKNOWN"
        best_score = 0.0

        # Compute similarity against each intent bank
        for intent, intent_vectors in self.corpus_embeddings.items():
            cosine_scores = util.cos_sim(user_embedding, intent_vectors)
            score = torch.max(cosine_scores).item()

            if score > best_score:
                best_score = score
                best_intent = intent

        print(f"🧠 '{text}' → {best_intent} ({best_score:.3f})")

        # Prevent accidental STOP detection from short phrases
        if best_intent == "STOP" and len(text.split()) > 2:
            return ("UNKNOWN", None)

        # -------- Threshold Check --------
        if best_score < 0.45:
            return ("UNKNOWN", None)

        # -------- Language Switching Logic --------
        if best_intent == "SWITCH_LANGUAGE":

            if any(word in text for word in ["telugu", "తెలుగు"]):
                return ("SWITCH_LANGUAGE", "te")

            elif any(word in text for word in ["hindi", "हिंदी"]):
                return ("SWITCH_LANGUAGE", "hi")

            elif any(word in text for word in ["english", "ఇంగ్లీష్"]):
                return ("SWITCH_LANGUAGE", "en")

            else:
                # Switch intent detected but no clear target
                return ("SWITCH_LANGUAGE", None)

        # -------- Normal Intent --------
        return (best_intent, None)

