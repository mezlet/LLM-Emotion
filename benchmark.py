import os
import time
import json
import base64
import yaml
import requests
import pandas as pd
import numpy as np

from tqdm import tqdm
from jiwer import wer
from sklearn.metrics import accuracy_score, f1_score
from deepface import DeepFace


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def measure_latency(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def save_results(df, filename):
    ensure_results_dir()
    path = os.path.join("results", filename)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ---------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------

def call_ollama(model, prompt, ollama_url):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def classify_emotion_with_llm(model, text, labels, ollama_url):
    labels_text = ", ".join(labels)

    prompt = f"""
You are an emotion classification system.

Classify the user's emotion into exactly one of these labels:

{labels_text}

User text:
{text}

Return only the emotion label.
"""

    output = call_ollama(model, prompt, ollama_url)
    output = output.lower().strip()

    for label in labels:
        if label.lower() in output:
            return label

    return "unknown"


def generate_emotional_response(model, user_text, emotion, ollama_url):
    prompt = f"""
You are a social humanoid robot assistant.

The user emotion is: {emotion}

Generate a short, natural, emotionally congruent response.
The response should be:
- empathetic
- contextually relevant
- suitable for a humanoid robot
- not overly long

User said:
{user_text}
"""

    return call_ollama(model, prompt, ollama_url)


# ---------------------------------------------------------
# ASR benchmarking
# ---------------------------------------------------------

def transcribe_with_fasterwhisper(audio_path):
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)

    text = " ".join(segment.text for segment in segments)
    return text.strip()


def transcribe_with_whisperx(audio_path):
    import whisperx

    device = "cpu"
    model = whisperx.load_model("base", device=device, compute_type="int8")

    result = model.transcribe(audio_path)
    return result["segments"][0]["text"].strip() if result["segments"] else ""


def benchmark_asr(csv_path="data/asr_eval.csv"):
    df = pd.read_csv(csv_path)

    backends = {
        "fasterwhisper": transcribe_with_fasterwhisper,
        "whisperx": transcribe_with_whisperx
    }

    rows = []

    for backend_name, transcribe_fn in backends.items():
        predictions = []
        references = []
        latencies = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ASR: {backend_name}"):
            audio_path = row["audio_path"]
            ground_truth = row["ground_truth"]

            try:
                prediction, latency = measure_latency(transcribe_fn, audio_path)
            except Exception as e:
                print(f"Error with {backend_name} on {audio_path}: {e}")
                prediction = ""
                latency = np.nan

            predictions.append(prediction)
            references.append(ground_truth)
            latencies.append(latency)

        rows.append({
            "component": "ASR",
            "candidate": backend_name,
            "wer": wer(references, predictions),
            "avg_latency_seconds": np.nanmean(latencies)
        })

    result_df = pd.DataFrame(rows)
    save_results(result_df, "asr_results.csv")
    return result_df


# ---------------------------------------------------------
# Facial expression benchmarking
# ---------------------------------------------------------

DEEPFACE_TO_PLUTCHIK = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "trust"
}


def classify_face_deepface(image_path):
    result = DeepFace.analyze(
        img_path=image_path,
        actions=["emotion"],
        enforce_detection=False
    )

    if isinstance(result, list):
        result = result[0]

    deepface_emotion = result["dominant_emotion"].lower()
    return DEEPFACE_TO_PLUTCHIK.get(deepface_emotion, "unknown")


def benchmark_deepface(csv_path="data/face_eval.csv"):
    df = pd.read_csv(csv_path)

    y_true = []
    y_pred = []
    latencies = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Face: DeepFace"):
        image_path = row["image_path"]
        label = row["label"]

        try:
            prediction, latency = measure_latency(classify_face_deepface, image_path)
        except Exception as e:
            print(f"DeepFace error on {image_path}: {e}")
            prediction = "unknown"
            latency = np.nan

        y_true.append(label)
        y_pred.append(prediction)
        latencies.append(latency)

    return {
        "component": "Facial Expression",
        "candidate": "DeepFace",
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "avg_latency_seconds": np.nanmean(latencies)
    }


# ---------------------------------------------------------
# Emotion LLM benchmarking
# ---------------------------------------------------------

def benchmark_emotion_llms(csv_path="data/emotion_text_eval.csv", config=None):
    df = pd.read_csv(csv_path)

    rows = []

    for model in config["models"]["emotion_llms"]:
        y_true = []
        y_pred = []
        latencies = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Emotion LLM: {model}"):
            text = row["text"]
            label = row["label"]

            prediction, latency = measure_latency(
                classify_emotion_with_llm,
                model,
                text,
                config["emotion_labels"],
                config["ollama_url"]
            )

            y_true.append(label)
            y_pred.append(prediction)
            latencies.append(latency)

        rows.append({
            "component": "Emotion LLM",
            "candidate": model,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "avg_latency_seconds": np.mean(latencies)
        })

    result_df = pd.DataFrame(rows)
    save_results(result_df, "emotion_llm_results.csv")
    return result_df


# ---------------------------------------------------------
# Response generation benchmarking
# ---------------------------------------------------------

def simple_congruence_score(response, emotion):
    """
    Lightweight automatic congruence heuristic.

    For thesis-quality evaluation, replace or complement this with:
    - human annotation
    - GPT-based judge
    - rubric-based evaluation
    """

    emotion_keywords = {
        "joy": ["happy", "glad", "wonderful", "great", "excited"],
        "sadness": ["sorry", "difficult", "sad", "understand", "support"],
        "anger": ["frustrating", "understand", "calm", "upsetting"],
        "fear": ["safe", "worry", "scary", "support", "careful"],
        "surprise": ["unexpected", "surprising", "wow"],
        "disgust": ["unpleasant", "sorry", "difficult"],
        "trust": ["trust", "support", "here for you"],
        "anticipation": ["looking forward", "soon", "exciting"]
    }

    response_lower = response.lower()
    keywords = emotion_keywords.get(emotion, [])

    if not keywords:
        return 0.0

    matches = sum(1 for word in keywords if word in response_lower)
    return matches / len(keywords)


def benchmark_response_llms(csv_path="data/response_eval.csv", config=None):
    df = pd.read_csv(csv_path)

    rows = []

    for model in config["models"]["response_llms"]:
        congruence_scores = []
        latencies = []
        responses = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Response LLM: {model}"):
            user_text = row["user_text"]
            emotion = row["emotion"]

            response, latency = measure_latency(
                generate_emotional_response,
                model,
                user_text,
                emotion,
                config["ollama_url"]
            )

            congruence = simple_congruence_score(response, emotion)

            responses.append(response)
            congruence_scores.append(congruence)
            latencies.append(latency)

        rows.append({
            "component": "Response LLM",
            "candidate": model,
            "avg_congruence_score": np.mean(congruence_scores),
            "avg_latency_seconds": np.mean(latencies)
        })

        response_df = df.copy()
        response_df["model"] = model
        response_df["generated_response"] = responses
        save_results(response_df, f"generated_responses_{model.replace(':', '_')}.csv")

    result_df = pd.DataFrame(rows)
    save_results(result_df, "response_llm_results.csv")
    return result_df


# ---------------------------------------------------------
# Component selection
# ---------------------------------------------------------

def normalize_latency(series):
    """
    Lower latency is better, so this converts latency into a score
    where higher is better.
    """

    max_latency = series.max()
    min_latency = series.min()

    if max_latency == min_latency:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1 - ((series - min_latency) / (max_latency - min_latency))


def select_best_candidate(df, component_name):
    subset = df[df["component"] == component_name].copy()

    if subset.empty:
        return None

    if "wer" in subset.columns and subset["wer"].notna().any():
        subset["score"] = normalize_latency(subset["avg_latency_seconds"]) * 0.4
        subset["score"] += (1 - subset["wer"]) * 0.6

    elif "accuracy" in subset.columns and "macro_f1" in subset.columns:
        subset["score"] = subset["accuracy"] * 0.4
        subset["score"] += subset["macro_f1"] * 0.4
        subset["score"] += normalize_latency(subset["avg_latency_seconds"]) * 0.2

    elif "avg_congruence_score" in subset.columns:
        subset["score"] = subset["avg_congruence_score"] * 0.7
        subset["score"] += normalize_latency(subset["avg_latency_seconds"]) * 0.3

    else:
        return None

    return subset.sort_values("score", ascending=False).iloc[0]


def generate_selection_report(result_files):
    dfs = []

    for file in result_files:
        if os.path.exists(file):
            dfs.append(pd.read_csv(file))

    all_results = pd.concat(dfs, ignore_index=True, sort=False)

    selected_rows = []

    for component in all_results["component"].unique():
        best = select_best_candidate(all_results, component)
        if best is not None:
            selected_rows.append(best)

    selection_df = pd.DataFrame(selected_rows)
    save_results(selection_df, "component_selection_report.csv")

    return selection_df


# ---------------------------------------------------------
# Main runner
# ---------------------------------------------------------

def main():
    config = load_config()

    all_component_results = []

    print("\nRunning ASR benchmark...")
    try:
        asr_results = benchmark_asr()
        all_component_results.append(asr_results)
    except Exception as e:
        print(f"ASR benchmark skipped because of error: {e}")

    print("\nRunning facial expression benchmark...")
    try:
        deepface_result = benchmark_deepface()
        face_results = pd.DataFrame([deepface_result])
        save_results(face_results, "face_results.csv")
        all_component_results.append(face_results)
    except Exception as e:
        print(f"Face benchmark skipped because of error: {e}")

    print("\nRunning emotion LLM benchmark...")
    try:
        emotion_results = benchmark_emotion_llms(config=config)
        all_component_results.append(emotion_results)
    except Exception as e:
        print(f"Emotion LLM benchmark skipped because of error: {e}")

    print("\nRunning response LLM benchmark...")
    try:
        response_results = benchmark_response_llms(config=config)
        all_component_results.append(response_results)
    except Exception as e:
        print(f"Response LLM benchmark skipped because of error: {e}")

    print("\nGenerating final component selection report...")

    result_files = [
        "results/asr_results.csv",
        "results/face_results.csv",
        "results/emotion_llm_results.csv",
        "results/response_llm_results.csv"
    ]

    selection_report = generate_selection_report(result_files)

    print("\nSelected components:")
    print(selection_report)


if __name__ == "__main__":
    main()