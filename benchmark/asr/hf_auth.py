"""
Hugging Face authentication helper.
Handles login from environment variable or interactive prompt,
and verifies access to a gated dataset before the benchmark starts.

Priority order for token:
  1. HF_TOKEN environment variable
  2. ~/.cache/huggingface/token  (written by `huggingface-cli login`)
  3. Interactive prompt (if running in a terminal)
"""

import os
import sys
from pathlib import Path


HF_TOKEN_ENV  = "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"
HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"


def get_token() -> str | None:
    """Return a cached/env token without prompting."""
    if token := os.environ.get(HF_TOKEN_ENV, "").strip():
        return token
    if HF_TOKEN_FILE.exists():
        token = HF_TOKEN_FILE.read_text().strip()
        if token:
            return token
    return None


def ensure_login(dataset_id: str | None = None, interactive: bool = True) -> str:
    """
    Ensure the user is logged in to Hugging Face.

    1. Checks HF_TOKEN env var.
    2. Checks cached token file (~/.cache/huggingface/token).
    3. If neither found and interactive=True, prompts for a token and caches it.
    4. Optionally verifies access to `dataset_id` (gated datasets need extra steps).

    Returns the token string.
    Raises RuntimeError if login cannot be established.
    """
    try:
        from huggingface_hub import login, whoami, HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is not installed.\n"
            "Install via: pip install huggingface_hub"
        )

    token = get_token()

    if token:
        try:
            info = whoami(token=token)
            print(f"  [HF] Logged in as: {info['name']}")
        except Exception:
            print("  [HF] Cached token appears invalid — re-authenticating ...")
            token = None

    if not token:
        if not interactive or not sys.stdin.isatty():
            raise RuntimeError(
                "No Hugging Face token found.\n"
                "Set the HF_TOKEN environment variable:\n"
                "    export HF_TOKEN=hf_your_token_here\n"
                "Or run `huggingface-cli login` once to cache your token.\n"
                "Get a token at: https://huggingface.co/settings/tokens"
            )

        print("\n  Hugging Face login required for Common Voice streaming.")
        print("  Get a token at: https://huggingface.co/settings/tokens\n")
        token = input("  Paste your HF token (input hidden): ").strip()
        if not token:
            raise RuntimeError("No token provided — aborting.")

        try:
            login(token=token, add_to_git_credential=False)
            info = whoami(token=token)
            print(f"  [HF] Logged in as: {info['name']}")
            # Cache for future runs
            HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            HF_TOKEN_FILE.write_text(token)
            print(f"  [HF] Token cached at {HF_TOKEN_FILE}")
        except Exception as e:
            raise RuntimeError(f"HF login failed: {e}")

    if dataset_id:
        _verify_dataset_access(dataset_id, token)

    return token


def _verify_dataset_access(dataset_id: str, token: str):
    """
    Check whether the user has accepted the dataset terms.
    Prints a clear message with the accept URL if not.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.dataset_info(dataset_id, token=token)
    except Exception as e:
        err = str(e).lower()
        if "403" in err or "gated" in err or "access" in err or "forbidden" in err:
            raise RuntimeError(
                f"\n  Access denied to '{dataset_id}'.\n"
                f"  You need to accept the dataset terms first:\n"
                f"  → https://huggingface.co/datasets/{dataset_id}\n"
                f"  (log in on the HF website and click 'Agree and access repository')\n"
            )
        # Non-auth errors (e.g. network) — warn but don't abort
        print(f"  [HF] Warning: could not verify dataset access: {e}")