"""
fer_benchmark/runner.py
Thin wrapper — kept for CLI use; Colab calls model.predict() directly.
"""
import time
from typing import List, Tuple

def run_model_on_dataset(model, dataset) -> List[dict]:
    predictions = []
    errors = 0
    for i, (img_path, true_label) in enumerate(dataset):
        t0 = time.perf_counter()
        try:
            pred_label, confidence = model.predict(img_path)
        except Exception as e:
            pred_label, confidence = "unknown", 0.0
            errors += 1
            if errors <= 3:
                print(f"  WARN [{errors}]: {e}")
        lat = (time.perf_counter() - t0) * 1000
        predictions.append({
            "img_path":   str(img_path),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": round(confidence, 4),
            "latency_ms": round(lat, 2),
        })
    return predictions
