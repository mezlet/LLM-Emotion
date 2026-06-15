"""
model_client.py
Ollama HTTP client with retry logic and optional HuggingFace backend.

Backends:
  OllamaClient        — local Ollama server (default; matches your Ameca setup)
  HuggingFaceClient   — direct HF transformers (requires GPU)
"""

import time

import requests

from config import MODELS, GENERATION, OLLAMA_HOST, OLLAMA_TIMEOUT, MAX_RETRIES, RETRY_DELAY
from prompt_builder import build_full_prompt, build_chat_messages


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin wrapper around the Ollama /api/generate endpoint.
    Includes exponential-backoff retry on transient failures.
    """

    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host.rstrip("/")

    def generate(
        self,
        model_key:  str,
        emotion:    str,
        context:    str,
        utterance:  str,
        max_tokens: int  = GENERATION["max_tokens"],
    ) -> tuple[str, float]:
        """
        Generate a response for the given sample.
        Returns (generated_text, latency_seconds).
        Raises RuntimeError after MAX_RETRIES exhausted.
        """
        model_tag = MODELS[model_key]["ollama_tag"]
        prompt    = build_full_prompt(emotion, context, utterance)

        payload = {
            "model":  model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": GENERATION["temperature"],
                "top_p":       GENERATION["top_p"],
            },
        }

        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0   = time.perf_counter()
                resp = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=OLLAMA_TIMEOUT,
                )
                latency = time.perf_counter() - t0
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()
                return text, latency

            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * (2 ** (attempt - 1))   # exponential back-off
                    print(f"[model_client] Attempt {attempt} failed ({exc}). "
                          f"Retrying in {wait:.1f}s …")
                    time.sleep(wait)

        raise RuntimeError(
            f"[model_client] {model_key} failed after {MAX_RETRIES} attempts: {last_exc}"
        )

    def health_check(self) -> bool:
        """Return True if Ollama server is reachable."""
        try:
            requests.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return list of locally available Ollama model tags."""
        resp = requests.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]


# ── HuggingFace ───────────────────────────────────────────────────────────────

class HuggingFaceClient:
    """
    HF Transformers backend.
    Models are downloaded on first use; GPU strongly recommended.
    """

    def __init__(self, device: str = "auto"):
        self.device  = device
        self._loaded: dict = {}

    def _load(self, model_key: str):
        if model_key in self._loaded:
            return self._loaded[model_key]

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_id = MODELS[model_key]["hf_tag"]
        print(f"[model_client] Loading {model_id} …")
        tok   = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self._loaded[model_key] = (tok, model)
        return tok, model

    def generate(
        self,
        model_key:  str,
        emotion:    str,
        context:    str,
        utterance:  str,
        max_tokens: int = GENERATION["max_tokens"],
    ) -> tuple[str, float]:
        import torch

        tok, model = self._load(model_key)
        messages   = build_chat_messages(emotion, context, utterance)

        try:
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
        except Exception:
            full      = build_full_prompt(emotion, context, utterance)
            input_ids = tok(full, return_tensors="pt").input_ids.to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens = max_tokens,
                do_sample      = True,
                temperature    = GENERATION["temperature"],
                top_p          = GENERATION["top_p"],
                pad_token_id   = tok.eos_token_id,
            )
        latency = time.perf_counter() - t0

        new_tokens = out[0][input_ids.shape[-1]:]
        text       = tok.decode(new_tokens, skip_special_tokens=True).strip()
        return text, latency


# ── Factory ───────────────────────────────────────────────────────────────────

def get_client(backend: str = "ollama", **kwargs):
    """Return the appropriate client by backend name."""
    if backend == "ollama":
        return OllamaClient(**kwargs)
    if backend == "hf":
        return HuggingFaceClient(**kwargs)
    raise ValueError(f"Unknown backend '{backend}'. Choose 'ollama' or 'hf'.")