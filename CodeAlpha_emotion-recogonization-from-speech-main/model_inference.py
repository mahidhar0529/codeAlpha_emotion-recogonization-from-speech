"""
Inference module for EmotionSense deep learning model.
Loads the trained PyTorch model and provides emotion predictions.
"""

import os
import re
import torch
import torch.nn as nn

EMOTIONS = ["happy","sad","angry","fearful","loving","surprised","calm","neutral"]
EMOJI_MAP = {
    "happy": "\U0001f60a", "sad": "\U0001f622", "angry": "\U0001f621",
    "fearful": "\U0001f628", "loving": "\U0001f970", "surprised": "\U0001f632",
    "calm": "\U0001f60c", "neutral": "\U0001f610"
}


class EmotionInference:
    """Loads and runs the trained emotion classification model."""

    def __init__(self, model_path="model_data/emotion_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.max_len = 30
        self.loaded = False

        if os.path.exists(model_path):
            try:
                self._load_model(model_path)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")

    def _load_model(self, path):
        """Load the trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        vocab_size = checkpoint.get("vocab_size", 5000)
        self.max_len = checkpoint.get("max_len", 30)
        self.vocab = checkpoint.get("vocab_word2idx", {})

        # Recreate model architecture
        from emotion_model import EmotionBiLSTM
        self.model = EmotionBiLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            n_layers=2,
            n_classes=len(EMOTIONS),
            dropout=0
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.loaded = True
        print(f"Emotion model loaded! Vocab: {vocab_size}, Max len: {self.max_len}")

    def tokenize(self, text):
        """Convert text to token IDs."""
        words = re.sub(r"[^a-z\s]", "", text.lower()).split()
        ids = [self.vocab.get(w, 1) for w in words[:self.max_len]]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids.extend([0] * pad_len)
        return ids

    def predict(self, text):
        """
        Predict emotion probabilities for text.
        Returns dict of emotion -> score (percentage).
        """
        if not self.loaded or not text or not text.strip():
            return None

        tokens = self.tokenize(text)
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        scores = {}
        for i, emotion in enumerate(EMOTIONS):
            scores[emotion] = round(probs[i].item() * 100, 1)

        return scores

    def predict_batch(self, texts):
        """Predict emotions for a list of texts."""
        results = []
        for text in texts:
            scores = self.predict(text)
            if scores:
                dominant = max(scores, key=scores.get)
                results.append({
                    "scores": scores,
                    "dominant": {"emotion": dominant, "score": scores[dominant]}
                })
            else:
                results.append({
                    "scores": {e: 0 for e in EMOTIONS},
                    "dominant": {"emotion": "neutral", "score": 100}
                })
        return results


def analyze_with_dl(text, inference_engine):
    """
    Full emotion analysis using the deep learning model.
    Returns the same format as server.py's analyze_emotion.
    """
    if not text or not text.strip():
        base = {e: 0 for e in EMOTIONS}
        base["neutral"] = 100
        return {
            "scores": base,
            "dominant": {"emotion": "neutral", "score": 100},
            "sentiment": 0,
            "sentiment_label": "neutral",
            "method": "none",
            "word_count": 0
        }

    # Get DL model scores
    dl_scores = inference_engine.predict(text)

    if dl_scores is None:
        return None

    # Calculate simple sentiment from emotion scores
    pos_emotions = ["happy", "loving", "surprised", "calm"]
    neg_emotions = ["sad", "angry", "fearful"]
    pos_score = sum(dl_scores.get(e, 0) for e in pos_emotions)
    neg_score = sum(dl_scores.get(e, 0) for e in neg_emotions)
    sentiment = round((pos_score - neg_score) / 100, 4)

    sentiment_label = "positive" if sentiment > 0.1 else ("negative" if sentiment < -0.1 else "neutral")

    # Round to ints
    final = {}
    for e in EMOTIONS:
        final[e] = int(round(dl_scores.get(e, 0)))

    # Normalize to 100
    total = sum(final.values())
    if total != 100 and final:
        final[max(final, key=final.get)] += (100 - total)

    dominant = max(final, key=final.get)
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    top3 = [{"emotion": e, "score": s} for e, s in ranked[:3]]

    return {
        "scores": final,
        "dominant": {"emotion": dominant, "score": final[dominant]},
        "top_3": top3,
        "sentiment": sentiment,
        "sentiment_label": sentiment_label,
        "method": "deep_learning",
        "word_count": len(re.sub(r"[^a-z\s]", "", text.lower()).split())
    }
