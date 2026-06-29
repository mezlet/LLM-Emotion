"""DeepFace wrapper — identical interface to OllamaFERClient."""
import numpy as np
from pathlib import Path

LABEL_MAP = {
    "angry": "anger", "disgust": "disgust", "fear": "fear",
    "happy": "happiness", "sad": "sadness", "surprise": "surprise", "neutral": "neutral",
}
_df = None
def _get():
    global _df
    if _df is None:
        from deepface import DeepFace; _df = DeepFace
    return _df

class DeepFaceModel:
    name = "DeepFace"
    def __init__(self, backend="retinaface", device="cuda"):
        self.backend = backend
        if device == "cpu":
            import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print(f"  [DeepFace] Warming up (backend={backend}) ...")
        try:
            _get().analyze(img_path=np.zeros((64,64,3),dtype=np.uint8),
                           actions=["emotion"], detector_backend=backend,
                           enforce_detection=False, silent=True)
        except Exception: pass
        print("  [DeepFace] Ready.")

    def predict(self, img_path):
        result = _get().analyze(img_path=str(img_path), actions=["emotion"],
                                detector_backend=self.backend,
                                enforce_detection=False, silent=True)
        if isinstance(result, list): result = result[0]
        dominant = result.get("dominant_emotion", "unknown")
        scores   = result.get("emotion", {})
        return LABEL_MAP.get(dominant.lower(), dominant.lower()), scores.get(dominant, 0.0)/100.0
