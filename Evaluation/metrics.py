# Evaluation/metrics.py

from jiwer import wer
from sklearn.metrics import accuracy_score, classification_report

def compute_wer(refs, hyps):
    value = wer(refs, hyps)
    print("\n==============================")
    print(f"WER: {value:.3f}")
    print(f"Approx STT Accuracy: {(1 - value)*100:.2f}%")
    print("==============================")
    return value


def compute_intent_metrics(y_true, y_pred):
    print("\n==============================")
    print("Intent Accuracy:", accuracy_score(y_true, y_pred))
    print("\nDetailed Report:\n")
    print(classification_report(y_true, y_pred))
    print("==============================")