import torch
from sentence_transformers import SentenceTransformer, util

class IntentParser:
    def __init__(self, use_bert=True):
        print(" 🧠 Loading Multilingual Semantic Brain...")
        try:
            # This model supports 50+ languages including Hindi & Telugu
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.use_bert = True
        except Exception as e:
            print(f" ❌ Brain Error: {e}")
            self.use_bert = False
            return

        # --- THE TRAINING BANK (NO MORE LISTS IN MAIN.PY) ---
        # We teach the AI by giving it examples in mixed languages.
        # It learns the "Concept", not just the keyword.
        self.intent_bank = {
            "REGISTER_FACE": [
                # English
                "register this person", "save this face", "remember him", 
                "add to database", "save this person", "memorize this face",
                # Telugu (Context: Saving data)
                "ee person ni save cheyu", "face ni register cheyu", 
                "save cheyu", "gurthupettuko", "ee manishi evaru save cheyu",
                # Hindi
                "is chehre ko save karo", "yaad rakho", "register karo", 
                "iska naam save karo"
            ],
            "PEOPLE_DETECTION": [
                # English (Context: Identifying)
                "who is this", "who is in front of me", "do you know him", 
                "recognize this person", "do you see anyone",
                # Telugu
                "na mundhu evaru unnaru", "evaru unnaru", "tana peru enti", 
                "na mundhu evaraina unnara",
                # Hindi
                "mere samne kaun hai", "kya tum isse jante ho", 
                "kaun hai ye", "pehchano"
            ],
            "DESCRIBE_SCENE": [
                # English (Context: Objects/Surroundings)
                "describe the scene", "what is in front of me", 
                "what objects are here", "look around",
                # Telugu
                "na mundhu em undhi", "scene vivarinchu", "em kanipistundi",
                # Hindi
                "mere samne kya hai", "kya dikh raha hai", "scene describe karo"
            ],
            "READ_TEXT": [
                "read text", "scan document", "read the bill", "what is written",
                "padho", "chaduvu", "text chadu", "bill entha"
            ],
            "EMERGENCY": [
                # English (Context: Danger)
                "help me", "emergency", "i am in danger", "sos", "save my life",
                # Telugu
                "nannu kapadu", "apadha", "emergency", "help cheyu",
                # Hindi
                "bachao", "madad karo", "khatra hai"
            ],
            "STOP": [
                "stop", "exit", "sleep", "shut down", "chup raho", "aapu", "pani aipoyindi"
            ]
        }

        # --- PRE-CALCULATE VECTORS (Fast Performance) ---
        self.corpus_embeddings = {}
        for intent, phrases in self.intent_bank.items():
            # Encode all phrases for this intent into one block of math
            self.corpus_embeddings[intent] = self.model.encode(phrases, convert_to_tensor=True)
        
        print(" ✅ Brain Trained & Ready.")

    def parse(self, text, lang_code="en-IN"):
        if not text or not self.use_bert: return "UNKNOWN"
        text = text.lower().strip()

        # 1. Convert User Input to Vector
        user_embedding = self.model.encode(text, convert_to_tensor=True)

        best_intent = "UNKNOWN"
        best_score = 0.0

        # 2. Compare against every Intent Category
        for intent, intent_vectors in self.corpus_embeddings.items():
            # Find similarity with ALL examples in this category
            cosine_scores = util.cos_sim(user_embedding, intent_vectors)
            
            # Take the highest score from this category
            max_score_for_category = torch.max(cosine_scores).item()

            if max_score_for_category > best_score:
                best_score = max_score_for_category
                best_intent = intent

        # 3. Decision & Thresholding
        print(f" 🧠 Brain Analysis: '{text}' -> {best_intent} ({best_score:.2f})")

        # "Save cheyu" (Register) vs "Save me" (Emergency)
        # The vector math handles this distinction automatically now.
        
        if best_score > 0.40: # 40% Confidence Threshold
            return best_intent
        
        return "UNKNOWN"