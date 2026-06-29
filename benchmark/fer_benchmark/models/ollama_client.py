"""
Unified Ollama client for Qwen2.5-VL and LLaVA.
Uses /api/chat for qwen2.5vl (supports system prompt),
    /api/generate for llava.
"""
import base64, re, requests
from pathlib import Path

CANONICAL = {"anger","disgust","fear","happiness","sadness","surprise","neutral","contempt"}
SYNONYMS  = {"angry":"anger","happy":"happiness","sad":"sadness","fearful":"fear",
             "scared":"fear","disgusted":"disgust","surprised":"surprise","calm":"neutral"}

SYSTEM = ("You are an expert facial emotion recognition system. "
          "Analyse the facial expression and respond with EXACTLY ONE WORD from: "
          "anger, disgust, fear, happiness, sadness, surprise, neutral, contempt. "
          "Output only the single label.")
PROMPT = ("What facial emotion does this person show? "
          "Reply with exactly one word: anger, disgust, fear, happiness, sadness, surprise, neutral, or contempt.")

def _b64(p):
    with open(p, "rb") as f: return base64.b64encode(f.read()).decode()

def _parse(text):
    t = text.strip().lower()
    if t in CANONICAL: return t, 0.95
    if t in SYNONYMS:  return SYNONYMS[t], 0.90
    for w in re.split(r"\W+", t):
        if w in CANONICAL: return w, 0.80
        if w in SYNONYMS:  return SYNONYMS[w], 0.75
    return "unknown", 0.0

class OllamaFERClient:
    def __init__(self, host="http://localhost:11434", model_tag="qwen2.5vl:7b",
                 temperature=0.1, timeout=90):
        self.host = host.rstrip("/")
        self.model_tag = model_tag
        self.temperature = temperature
        self.timeout = timeout
        self.name = model_tag
        # Choose endpoint based on model family
        self._use_chat = "qwen" in model_tag.lower()
        print(f"  [{model_tag}] host={self.host}  endpoint={'chat' if self._use_chat else 'generate'}")

    def predict(self, img_path):
        img_b64 = _b64(img_path)
        if self._use_chat:
            payload = {
                "model": self.model_tag,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": PROMPT, "images": [img_b64]},
                ],
                "options": {"temperature": self.temperature, "num_predict": 10},
                "stream": False,
            }
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return _parse(r.json()["message"]["content"])
        else:
            payload = {
                "model": self.model_tag,
                "prompt": PROMPT,
                "images": [img_b64],
                "options": {"temperature": self.temperature, "num_predict": 15, "num_gpu": 99},
                "stream": False,
            }
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return _parse(r.json().get("response", ""))
