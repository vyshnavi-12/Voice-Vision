from sentence_transformers import SentenceTransformer, util
import torch

class IntentParser:
    def __init__(self, use_bert=True):
        self.use_bert = use_bert
        self.model = None
        
        # --- 1. DEFINE INTENTS BY DESCRIPTION ---
        # Inside the __init__ of IntentParser in intent_parser.py
        self.intent_descriptions = {
            "READ_TEXT": "Read text, scan documents, bills, menus, or signboards.",
            "DESCRIBE_SCENE": "Describe the scene, environment, objects, or surroundings.",
            "PEOPLE_DETECTION": "Detect people, faces, find who is in front of me.",
            "REGISTER_FACE": "Register a new face, save a person, remember someone, learn a name.", # <--- NEW
            "EMERGENCY": "Activate emergency SOS, help, danger, save me, or the distinguished phrase 'it is very cold today'.",
            "STOP": "Stop speaking, go to sleep, exit, shut down."
        }
        
        # --- 2. LOAD BRAIN ---
        if self.use_bert:
            print(" 🧠 Loading Semantic Brain (MiniLM)...")
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                
                # Encode the DESCRIPTIONS once
                self.intent_names = list(self.intent_descriptions.keys())
                self.descriptions = list(self.intent_descriptions.values())
                
                # Create a vector map of the descriptions
                self.embedding_matrix = self.model.encode(self.descriptions, convert_to_tensor=True)
                print(" ✅ Brain Ready.")
            except Exception as e:
                print(f" ❌ BERT Load Failed (Offline?): {e}")
                print(" ⚠️ Switching to Keyword Backup Mode.")
                self.use_bert = False

    def parse(self, text, lang_code="en-IN"):
        if not text: return "UNKNOWN"
        
        # --- STRATEGY A: SEMANTIC MATCHING (Smart) ---
        if self.use_bert:
            try:
                # 1. Convert User Command to Vector
                user_embedding = self.model.encode(text, convert_to_tensor=True)
                
                # 2. Compare User Vector vs. Description Vectors
                cosine_scores = util.cos_sim(user_embedding, self.embedding_matrix)
                
                # 3. Find Best Match
                best_match_idx = torch.argmax(cosine_scores).item()
                best_score = cosine_scores[0][best_match_idx].item()
                best_intent = self.intent_names[best_match_idx]
                
                # Threshold
                if best_score > 0.30: 
                    print(f" 🧠 Matched: '{best_intent}' (Score: {best_score:.2f})")
                    return best_intent
                else:
                    print(f" 🧠 Unsure (Score: {best_score:.2f}). Ignoring.")
                    return "UNKNOWN"
            except:
                # If BERT crashes mid-process, fall back
                return self._keyword_fallback(text)

        # --- STRATEGY B: KEYWORD BACKUP (Offline/Robust) ---
        return self._keyword_fallback(text)

    def _keyword_fallback(self, text):
        """Simple backup logic if BERT fails to load"""
        text = text.lower()
        print(f" 🔍 Checking Keywords for: '{text}'")
        
        if any(w in text for w in ["read", "scan", "text", "bill", "menu", "padho", "chaduvu"]):
            return "READ_TEXT"
        
        if any(w in text for w in ["describe", "scene", "what is", "look", "varninchu", "dekho"]):
            return "DESCRIBE_SCENE"
            
        if any(w in text for w in ["who", "person", "people", "face", "evaru", "kaun"]):
            return "PEOPLE_DETECTION"
            
        if any(w in text for w in ["help", "sos", "save", "emergency", "bachao", "kapadandi"]):
            return "EMERGENCY"
            
        return "UNKNOWN"