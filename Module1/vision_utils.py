from ultralytics import YOLOWorld

# --- CONSTANTS ---
FOCAL_LENGTH = 600
KNOWN_WIDTH = 0.5 
SAFE_DISTANCE = 1.5 

# --- THE REFINED "GOD LIST" ---

# 1. INDOOR ITEMS (Home & Office)
# I added specific adjectives to help it tell the difference.
INDOOR_ITEMS = [
    # Electronics
    "smartphone", "laptop", "computer mouse", "keyboard", "monitor screen", 
    "television", "remote control", "headphones", "ceiling fan", 
    "wall clock", "wifi router", "extension cord", "mosquito repellent machine",
    
    # Personal & Table Items
    "closed book", "open book", "notebook", "pen", "pencil", 
    "backpack", "handbag", "wallet", "keys", "reading glasses", 
    "water bottle", "glass bottle", "coffee mug", "cup", 
    "plate", "spoon", "fork", "knife", "medicine bottle", "medicine blister strip"
]

# 2. FURNITURE & OBSTACLES (Things to avoid)
OBSTACLES = [
    "person", "chair", "office chair", "sofa", "couch", "bed", 
    "dining table", "office desk", "wooden cabinet", "wardrobe", "shelf",
    "door", "closed door", "open door", "wall", "pillar",
    "refrigerator", "washing machine", "dustbin", "window curtain", 
    "staircase", "steps"
]

# 3. OUTDOOR & STREET (Only for outside)
OUTDOOR_ITEMS = [
    "car", "bus", "truck", "motorcycle", "scooter", "bicycle", "auto rickshaw",
    "pedestrian", "street pole", "tree", "fence", "metal gate", 
    "traffic light", "signboard", "park bench", "fire hydrant", 
    "pothole", "concrete curb", "dog", "cow", "umbrella"
]

# Combine for the model
ALL_CLASSES = INDOOR_ITEMS + OBSTACLES + OUTDOOR_ITEMS

def load_model():
    """Loads the model with the refined list."""
    print(f"Loading AI with {len(ALL_CLASSES)} optimized objects...")
    
    # Using the Medium version for better accuracy
    model = YOLOWorld('yolov8m-worldv2.pt') 
    
    model.set_classes(ALL_CLASSES)
    return model

def is_hazard(name):
    """Returns True if the object is a collision risk."""
    # Note: We treat outdoor items as hazards too
    return name in OBSTACLES or name in OUTDOOR_ITEMS