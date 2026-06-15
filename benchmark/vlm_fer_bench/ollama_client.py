"""Ollama VLM inference client."""

import time
import requests
from .config import OLLAMA_HOST, DEFAULT_TIMEOUT, DEFAULT_RETRIES
from .image_utils import image_to_base64


def query_ollama_vlm(model_tag: str, image_path: str, prompt: str,
                      timeout: int = DEFAULT_TIMEOUT,
                      retries: int = DEFAULT_RETRIES,
                      num_gpu: int = -1,
                      keep_alive: str = "30m") -> dict:
    """
    Send image + prompt to an Ollama-served VLM, return raw response + latency.

    num_gpu: number of layers to offload to GPU. -1 = let Ollama decide (default,
             usually means "as many as fit"). Set explicitly if you need to force
             full GPU residency or limit it due to shared VRAM.
    keep_alive: how long Ollama keeps the model loaded in memory between calls.
                Higher value avoids reload overhead across many images.
    """
    img_b64 = image_to_base64(image_path)

    payload = {
        "model": model_tag,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.0,
            "num_predict": 10,
            "num_gpu": num_gpu,
        },
    }

    last_err = None
    for _ in range(retries + 1):
        try:
            start = time.time()
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=timeout,
            )
            elapsed = time.time() - start
            resp.raise_for_status()
            data = resp.json()
            return {
                "raw_response": data.get("response", "").strip(),
                "latency_sec": elapsed,
                "error": None,
            }
        except Exception as e:
            last_err = str(e)
            time.sleep(2)

    return {"raw_response": "", "latency_sec": None, "error": last_err}