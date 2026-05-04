PLUTCHIK = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation"
]


def fuse_emotions(text, prosody, face):
    votes = {}

    for source in [text, prosody, face]:
        if not source:
            continue
        votes[source] = votes.get(source, 0) + 1

    # majority vote
    if votes:
        return max(votes.items(), key=lambda x: x[1])[0]

    return "trust"  # fallback