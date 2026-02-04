import torch                                      # Tensor operations
from sentence_transformers import SentenceTransformer, util  # Semantic embeddings + similarity

class IntentParser:
    def __init__(self, use_bert=True):
        print(" 🧠 Loading Multilingual Semantic Brain...")
        try:
            # Multilingual model supports English, Telugu, and Hindi
            self.model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.use_bert = True
        except Exception as e:
            # Disable intent parsing if model fails to load
            print(f" ❌ Brain Error: {e}")
            self.use_bert = False
            return

        # --- INTENT TRAINING BANK ---
        # Each intent contains example phrases in multiple languages
        self.intent_bank = {
            "FACE_RECOGNITION": [
                "who is this person", "who is in front of me", "recognize this face",
                "do you know him", "tana peru enti", "evaru ithanu", "kaun hai ye"
            ],
            "PEOPLE_DETECTION": [
                "how many people are here", "are there people around",
                "is anyone standing there", "evaru unnaru", "entha mandi unnaru",
                "kitne log hain"
            ],
            "DESCRIBE_SCENE": [
                "describe the scene", "what is in front of me", "what is here",
                "look around", "na mundhu em undhi", "scene vivarinchu",
                "mere samne kya hai"
            ],
            "OBJECT_DETECTION": [
                "what objects are here", "what is on the desk", "identify objects",
                "what am I holding", "find items", "desk paina em unnay",
                "mez par kya hai", "vastuvulu cheppu"
            ],
            "OBSTACLE_DETECTION": [
                "is my path clear", "any obstacles", "is there anything in my way",
                "am i going to hit something", "addankulu unnaya", "darilo emaina undha",
                "kya rasta saaf hai"
            ],
            "NAVIGATION": [
                "how do i get out", "where is the exit", "directions please",
                "guide me to the door", "biyataki ela vellali", "dhaari chupinchu",
                "bahar nikalne ka rasta", "rasta batao"
            ],
            "READ_TEXT": [
                "read text", "scan document", "read the bill", "what is written",
                "chaduvu", "text chadu", "padho", "kagitham chadu"
            ],
            "REGISTER_FACE": [
                "register this person", "save this face", "remember him",
                "ee manishi ni gurthu pettuko", "face save cheyu", "is chehre ko yaad rakho"
            ],
            "EMERGENCY": [
                "help me", "i am in danger", "sos", "save my life",
                "nannu kapadu", "apadha", "emergency help", "bachao", "madad karo"
            ],
            "STOP": [
                "stop", "exit", "sleep", "aapu", "pani aipoyindi", "chup"
            ]
        }

        # Precompute embeddings for all intent phrases (performance optimization)
        self.corpus_embeddings = {}
        for intent, phrases in self.intent_bank.items():
            self.corpus_embeddings[intent] = self.model.encode(
                phrases,
                convert_to_tensor=True
            )

        print(" ✅ Brain Trained & Optimized.")

    def parse(self, text, lang_code="en-IN"):
        """
        Converts user speech text into a high-level intent.
        """
        if not text or not self.use_bert:
            return "UNKNOWN"

        text = text.lower().strip()

        # 1. Encode user input into semantic vector
        user_embedding = self.model.encode(
            text,
            convert_to_tensor=True
        )

        best_intent = "UNKNOWN"
        best_score = 0.0

        # 2. Compare input against each intent category
        for intent, intent_vectors in self.corpus_embeddings.items():
            cosine_scores = util.cos_sim(
                user_embedding,
                intent_vectors
            )

            # Take the best match within the intent category
            max_score_for_category = torch.max(cosine_scores).item()

            if max_score_for_category > best_score:
                best_score = max_score_for_category
                best_intent = intent

        print(
            f" 🧠 Brain Analysis: '{text}' -> "
            f"{best_intent} ({best_score:.2f})"
        )

        # 3. Thresholding to avoid false positives
        # Higher threshold improves robustness in noisy/multilingual speech
        if best_score > 0.60:
            return best_intent

        return "UNKNOWN"
