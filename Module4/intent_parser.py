import torch                                      # PyTorch for tensor operations
from sentence_transformers import SentenceTransformer, util
# SentenceTransformer → creates semantic embeddings
# util → provides cosine similarity for comparing embeddings


class IntentParser:
    def __init__(self, use_bert=True):
        """
        Initializes the intent parser.
        Loads a multilingual semantic model and prepares
        all intent phrase embeddings in advance for fast inference.
        """

        print(" 🧠 Loading Multilingual Semantic Brain...")

        # ---- MODEL LOADING ----
        # Try to load a multilingual sentence embedding model.
        # This model understands meaning (not keywords) across
        # English, Telugu, and Hindi.
        try:
            self.model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2'
            )
            self.use_bert = True
        except Exception as e:
            # If model loading fails, disable intent parsing safely
            print(f" ❌ Brain Error: {e}")
            self.use_bert = False
            return


        # ---- INTENT TRAINING BANK ----
        # Each intent has multiple "anchor phrases".
        # These phrases act as semantic representatives
        # for that intent across different languages.
        self.intent_bank = {
            "FACE_RECOGNITION": [
                "who is this person", "who is in front of me", "recognize this face",
                "do you know him", "tana peru enti", "evaru ithanu",
                "kaun hai ye", "chehra pehchano"
            ],
            "PEOPLE_DETECTION": [
                "how many people are here", "are there people around",
                "is anyone standing there", "evaru unnaru", "entha mandi unnaru",
                "ikkada entha mandi unnaru", "yahan kitne log hain", "kya koi khada hai"
            ],
            "DESCRIBE_SCENE": [
                "describe the scene", "what is in front of me", "what is here",
                "look around", "na mundhu em undhi", "scene vivarinchu",
                "em kanipistundi", "mere samne kya hai", "nazara batao", "kya dikh raha hai"
            ],
            "OBJECT_DETECTION": [
                "what objects are here", "what is on the desk", "identify objects",
                "what am I holding", "find items", "desk paina em unnay",
                "vastuvulu cheppu", "mez par kya rakha hai", "is chiz ka naam kya hai"
            ],
            "OBSTACLE_DETECTION": [
                "is my path clear", "any obstacles", "is there anything in my way",
                "am i going to hit something", "addankulu unnaya", "darilo emaina addu undha",
                "kya rasta saaf hai", "koi rukawat hai kya"
            ],
            "NAVIGATION": [
                "how do i get out", "where is the exit", "directions please",
                "guide me to the door", "biyataki ela vellali", "dhaari chupinchu",
                "rasta batao", "exit kahan hai", "rasta dikhao"
            ],
            "READ_TEXT": [
                "read text", "scan document", "read the bill", "what is written",
                "chaduvu", "text chadu", "kagitham chadu", "aksharalu chadu",
                "padho", "kya likha hai", "ispe kya likha hai"
            ],
            "REGISTER_FACE": [
                "register this person", "save this face", "remember him",
                "ee manishi ni gurthu pettuko", "face save cheyu", "kotha face register cheyu",
                "is chehre ko yaad rakho", "ise save karo"
            ],
            "EMERGENCY": [
                "help me", "i am in danger", "sos", "save my life",
                "nannu kapadu", "apadha", "emergency help", "bachao", "madad karo"
            ],
            "STOP": [
                "stop", "exit", "sleep", "aapu", "pani aipoyindi",
                "chup", "so jao", "band karo"
            ]
        }


        # ---- PRECOMPUTE EMBEDDINGS ----
        # Convert every phrase of every intent into vectors once.
        # This avoids recomputing embeddings during runtime,
        # making intent detection faster and more efficient.
        self.corpus_embeddings = {}
        for intent, phrases in self.intent_bank.items():
            self.corpus_embeddings[intent] = self.model.encode(
                phrases,
                convert_to_tensor=True
            )

        print(" ✅ Brain Trained & Optimized.")


    def parse(self, text, lang_code="en-IN"):
        """
        Takes user speech text and returns the best-matching intent
        using semantic similarity.
        """

        # If input is empty or model failed to load,
        # intent detection cannot be performed
        if not text or not self.use_bert:
            return "UNKNOWN"

        # Normalize text for better semantic matching
        text = text.lower().strip()


        # ---- USER INPUT ENCODING ----
        # Convert the user sentence into a semantic embedding
        user_embedding = self.model.encode(
            text,
            convert_to_tensor=True
        )

        best_intent = "UNKNOWN"
        best_score = 0.0


        # ---- SEMANTIC COMPARISON ----
        # Compare user embedding against each intent's phrase embeddings.
        # Cosine similarity measures how close the meanings are.
        for intent, intent_vectors in self.corpus_embeddings.items():
            cosine_scores = util.cos_sim(
                user_embedding,
                intent_vectors
            )

            # Select the highest similarity score within this intent
            max_score_for_category = torch.max(cosine_scores).item()

            # Track the best intent across all categories
            if max_score_for_category > best_score:
                best_score = max_score_for_category
                best_intent = intent


        # Debug output to understand model decisions
        print(
            f" 🧠 Brain Analysis: '{text}' -> "
            f"{best_intent} ({best_score:.2f})"
        )


        # ---- CONFIDENCE THRESHOLDING ----
        # If similarity is strong enough, accept the intent.
        # Otherwise, return UNKNOWN to avoid false activations.
        if best_score > 0.60:
            return best_intent

        return "UNKNOWN"