"""
Google Cloud Speech-to-Text v1 inference runner.
Wraps the google-cloud-speech client for the benchmark.

Authentication:
    Set the GOOGLE_APPLICATION_CREDENTIALS env var to the path of a
    service-account JSON key file, or run in a GCP environment with
    Application Default Credentials.

    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

Install:
    pip install google-cloud-speech

Audio requirements:
    - WAV files at 16 kHz, mono (the loaders already produce this format)
    - Files ≤ 60 s are sent inline (synchronous recognize)
    - Files > 60 s use LongRunningRecognize via a GCS URI — set
      GOOGLE_STORAGE_BUCKET env var to a GCS bucket you own, or keep
      audio clips under 60 s.
"""

import gc
import io
import os
import re
import wave
from pathlib import Path
from typing import Optional


class GoogleASRRunner:
    """
    Transcribes audio files using Google Cloud Speech-to-Text v1.

    Short clips (≤ 60 s) are sent via synchronous RecognizeRequest.
    Longer clips require a GCS bucket (set GOOGLE_STORAGE_BUCKET).
    """

    def __init__(
        self,
        language_code: str = "en-US",
        model: str = "latest_long",
        sample_rate: int = 16000,
        gcs_bucket: Optional[str] = None,
        enable_automatic_punctuation: bool = False,
    ):
        """
        Parameters
        ----------
        language_code : str
            BCP-47 language tag, e.g. "en-US", "en-GB".
        model : str
            Google model name: "latest_long", "latest_short", "phone_call",
            "video", "command_and_search", "default".
        sample_rate : int
            Audio sample rate in Hz (must match the WAV files).
        gcs_bucket : str | None
            GCS bucket name for long-audio uploads (> 60 s clips).
            If None and a clip exceeds 60 s, that clip is skipped with a warning.
        enable_automatic_punctuation : bool
            Whether Google should add punctuation (disabled by default so
            normalized output stays comparable with other runners).
        """
        self.language_code = language_code
        self.model = model
        self.sample_rate = sample_rate
        self.gcs_bucket = gcs_bucket or os.environ.get("GOOGLE_STORAGE_BUCKET")
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self._client = None

    # ------------------------------------------------------------------
    def load_model(self):
        try:
            from google.cloud import speech
        except ImportError:
            raise ImportError(
                "google-cloud-speech is not installed. "
                "Install via: pip install google-cloud-speech"
            )

        print(f"  [Google ASR] Initialising Speech-to-Text client "
              f"(model={self.model}, lang={self.language_code}) ...")
        self._client = speech.SpeechClient()
        # Validate credentials by listing nothing — surfaces auth errors early.
        try:
            self._client.transport  # noqa: just access the transport to trigger auth
        except Exception as e:
            raise RuntimeError(
                "Failed to authenticate with Google Cloud Speech-to-Text.\n"
                "Set GOOGLE_APPLICATION_CREDENTIALS to a valid service-account key.\n"
                f"Original error: {e}"
            )
        print("  [Google ASR] Client ready.")

    def transcribe(self, audio_path: str) -> str:
        if self._client is None:
            raise RuntimeError("Client not initialised. Call load_model() first.")

        from google.cloud import speech

        duration_s = self._wav_duration(audio_path)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            model=self.model,
            enable_automatic_punctuation=self.enable_automatic_punctuation,
        )

        if duration_s <= 60.0:
            audio_bytes = Path(audio_path).read_bytes()
            audio = speech.RecognitionAudio(content=audio_bytes)
            response = self._client.recognize(config=config, audio=audio)
        else:
            gcs_uri = self._upload_to_gcs(audio_path)
            if gcs_uri is None:
                return ""
            audio = speech.RecognitionAudio(uri=gcs_uri)
            operation = self._client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=300)

        transcript = " ".join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        )
        return self._normalize(transcript)

    def unload_model(self):
        self._client = None
        gc.collect()
        print("  [Google ASR] Client released.")

    # ------------------------------------------------------------------
    def _wav_duration(self, audio_path: str) -> float:
        with wave.open(audio_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    def _upload_to_gcs(self, audio_path: str) -> Optional[str]:
        if not self.gcs_bucket:
            print(
                f"  [Google ASR] WARNING: clip '{Path(audio_path).name}' "
                f"exceeds 60 s but no GCS bucket is configured. "
                f"Set GOOGLE_STORAGE_BUCKET env var or pass gcs_bucket=. Skipping."
            )
            return None
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for clips > 60 s. "
                "Install via: pip install google-cloud-storage"
            )
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)
        blob_name = f"benchmark_asr_tmp/{Path(audio_path).name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(audio_path)
        return f"gs://{self.gcs_bucket}/{blob_name}"

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s'\-]", "", text)
        return re.sub(r"\s+", " ", text)
