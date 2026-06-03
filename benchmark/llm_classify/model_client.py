"""
model_client.py
---------------
Thin wrapper around the Ollama REST API for text generation.
Handles retries and timeout gracefully.
"""

import json
import time
import logging
import urllib.request
import urllib.error
from config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT_S, OLLAMA_MAX_TOKENS, OLLAMA_TEMPERATURE

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_S = 5


class OllamaClient:
    """Synchronous client for Ollama /api/generate endpoint."""

    def __init__(self, model: str, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/api/generate"

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to the model and return the full text response.
        Retries up to MAX_RETRIES times on transient errors.
        Returns an empty string on persistent failure.
        """
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": OLLAMA_MAX_TOKENS,
            },
        }).encode("utf-8")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(
                    self.endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body.get("response", "")

            except urllib.error.URLError as e:
                log.warning(f"[{self.model}] Request failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)

            except json.JSONDecodeError as e:
                log.error(f"[{self.model}] JSON decode error: {e}")
                break

        log.error(f"[{self.model}] All {MAX_RETRIES} attempts failed. Returning empty response.")
        return ""