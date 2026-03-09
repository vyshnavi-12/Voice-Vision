# Evaluation/evaluate_intent_en.py

import pandas as pd
from config import COMMANDS_CSV, LANGUAGE
from metrics import compute_intent_metrics
from intent_parser import IntentParser

parser = IntentParser()

df = pd.read_csv(COMMANDS_CSV)

# Filter by language (EN)
df = df[df["Language"] == LANGUAGE]

y_true = []
y_pred = []

for _, row in df.iterrows():
    text = row["Text"]
    label = row["Intent"]

    pred = parser.parse(text)

    print(f"\nTEXT : {text}")
    print(f"TRUE : {label}")
    print(f"PRED : {pred}")

    y_true.append(label)
    y_pred.append(pred)

compute_intent_metrics(y_true, y_pred)