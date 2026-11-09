"""
EmotionSense - Python Backend Server
Flask API for Speech Emotion Recognition with Deep Learning
"""

import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

# =============== DEEP LEARNING MODEL ===============
try:
    from model_inference import EmotionInference, analyze_with_dl
    DL_ENGINE = EmotionInference()
    DL_AVAILABLE = DL_ENGINE.loaded
    print(f"Deep Learning model: {'Available' if DL_AVAILABLE else 'Not loaded - run train_model.py first'}")
except Exception as e:
    DL_AVAILABLE = False
    DL_ENGINE = None
    print(f"Deep Learning model not available: {e}")


# =============== FALLBACK KEYWORD ANALYSIS ===============
EMOTION_KEYWORDS = {
    "happy": ["happy","joy","glad","cheerful","delighted","thrilled","excited","wonderful","fantastic","amazing","great","awesome","beautiful","lovely","elated","pleased","laughing","smile","fun","funny","hilarious","celebrate","superb","terrific","fabulous","upbeat","optimistic","positive","bright","radiant","joyful"],
    "sad": ["sad","unhappy","depressed","miserable","gloomy","sorrow","heartbroken","broken","lonely","alone","grief","cry","crying","tears","disappointed","hopeless","helpless","hurt","pain","suffering","regret","melancholy","blue","down","low","devastated","despair","desperate","somber","dreary","dismal","lament","anguish"],
    "angry": ["angry","mad","furious","rage","irritated","annoyed","frustrated","hate","hateful","hostile","aggressive","violent","fury","wrath","livid","outraged","enraged","bitter","resentful","fuming","infuriated","indignant","irate","grumpy","cranky","fierce"],
    "fearful": ["scared","afraid","fearful","frightened","terrified","terror","panic","anxious","anxiety","nervous","worried","worry","dread","horrified","alarmed","tense","uneasy","overwhelmed","threatened","intimidated","shaky","trembling","petrified","apprehensive","distressed","jittery","timid"],
    "loving": ["love","loved","loving","adore","affection","affectionate","caring","care","compassionate","tender","passion","passionate","romance","romantic","darling","dear","beloved","sweet","sweetheart","lovely","warm","warmth","gentle","kindness","kind","hug","embrace","cherish","devoted","fond"],
    "surprised": ["surprise","surprised","astonishing","astonished","astounding","shock","shocked","unexpected","sudden","wow","incredible","unbelievable","stunning","startled","amazed","wonder","miraculous","extraordinary","staggering","speechless","dumbfounded","bewildered","stunned"],
    "calm": ["calm","peaceful","peace","relaxed","serene","tranquil","composed","balanced","harmony","gentle","easy","chill","quiet","stillness","still","content","centered","grounded","mindful","meditation","zen","soothing","restful","placid","collected","poised","cool","mellow"],
}

POSITIVE = {"happy","love","great","awesome","wonderful","fantastic","amazing","beautiful","excellent","good","nice","best","brilliant","superb","perfect","lovely","joy","glad","pleased","delighted","thrilled","elated","euphoric","terrific","fabulous","marvelous","splendid","magnificent","glorious","grateful","thankful","blessed"}
NEGATIVE = {"sad","angry","hate","bad","terrible","awful","horrible","worst","ugly","evil","pain","hurt","sorrow","grief","misery","agony","anguish","despair","hopeless","lonely","depressed","miserable","heartbroken","devastated","furious","rage","violent","hateful","bitter","resentful","dreadful","tragic","dismal","bleak","grim","gloomy","somber","dreary"}
INTENSIFIERS = {"very","really","extremely","incredibly","absolutely","totally","completely","utterly","deeply","truly","highly","so","too","such","quite"}


def tokenize(text):
    text = re.sub(r"[^a-z\s\-']", "", text.lower())
    words = [w.strip("'-'") for w in text.split() if w.strip("'-'")]
    return [w for w in words if len(w) > 0]


def analyze_sentiment(words):
    if not words: return 0.0
    score = 0.0
    for i, w in enumerate(words):
        if w in POSITIVE:
            score += 1.5 if (i > 0 and words[i-1] in INTENSIFIERS) else 1.0
        elif w in NEGATIVE:
            score -= 1.5 if (i > 0 and words[i-1] in INTENSIFIERS) else 1.0
    return round(max(-1.0, min(1.0, score / len(words))), 4)


def analyze_keywords_fallback(words):
    """Fallback keyword-based analysis when DL model is unavailable."""
    scores = {}
    total = 0
    for emotion, kws in EMOTION_KEYWORDS.items():
        c = sum(1 for w in words for kw in kws if kw in w)
        scores[emotion] = c
        total += c
    if total == 0:
        r = {e: 0 for e in EMOTION_KEYWORDS}
        r["neutral"] = 100
        return r
    pcts = {e: round(c / total * 100, 1) for e, c in scores.items()}
    pcts["neutral"] = 0
    s = sum(pcts.values())
    if s != 100 and s > 0:
        f = 100.0 / s
        for e in pcts:
            pcts[e] = round(pcts[e] * f, 1)
    pcts = dict(sorted(pcts.items(), key=lambda x: x[1], reverse=True))
    return pcts


def apply_nuance(pcts, sentiment):
    p = dict(pcts)
    adj = abs(sentiment) * 12
    if sentiment > 0.15 and "happy" in p:
        p["happy"] += adj
        if "neutral" in p and p["neutral"] > adj:
            p["neutral"] -= adj
    elif sentiment < -0.15:
        for e in ["sad", "angry"]:
            if e in p:
                p[e] += adj * 0.5
        if "neutral" in p and p["neutral"] > adj * 0.5:
            p["neutral"] -= adj * 0.5
        if "happy" in p:
            p["happy"] = max(0, p["happy"] - adj * 0.3)
    t = sum(p.values())
    if t > 0:
        f = 100.0 / t
        for e in p:
            p[e] = round(p[e] * f, 1)
    return p


def normalize_scores(scores_dict):
    """Round scores to ints and ensure they sum to 100."""
    EMOTIONS = ["happy","sad","angry","fearful","loving","surprised","calm","neutral"]
    final = {}
    for e in EMOTIONS:
        final[e] = int(round(scores_dict.get(e, 0)))
    total = sum(final.values())
    if total != 100 and final:
        final[max(final, key=final.get)] += (100 - total)
    return final


def analyze_emotion(text):
    """
    Analyze emotion using Deep Learning model with keyword fallback.
    """
    if not text or not text.strip():
        base = {e: 0 for e in EMOTION_KEYWORDS}
        base["neutral"] = 100
        return {"scores": base, "dominant": {"emotion": "neutral", "score": 100},
                "sentiment": 0, "sentiment_label": "neutral", "method": "none", "word_count": 0}

    words = tokenize(text)
    if not words:
        base = {e: 0 for e in EMOTION_KEYWORDS}
        base["neutral"] = 100
        return {"scores": base, "dominant": {"emotion": "neutral", "score": 100},
                "sentiment": 0, "sentiment_label": "neutral", "method": "none", "word_count": 0}

    sentiment = analyze_sentiment(words)

    # Try deep learning model first
    if DL_AVAILABLE:
        try:
            dl_result = analyze_with_dl(text, DL_ENGINE)
            if dl_result:
                # Apply sentiment nuance
                combined = apply_nuance(dl_result["scores"], sentiment)
                final = normalize_scores(combined)
                dominant = max(final, key=final.get)
                ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
                sl = "positive" if sentiment > 0.1 else ("negative" if sentiment < -0.1 else "neutral")
                return {
                    "scores": final,
                    "dominant": {"emotion": dominant, "score": final[dominant]},
                    "top_3": [{"emotion": e, "score": s} for e, s in ranked[:3]],
                    "sentiment": sentiment,
                    "sentiment_label": sl,
                    "method": "deep_learning",
                    "word_count": len(words)
                }
        except Exception as e:
            print(f"DL inference error: {e}")

    # Fallback: keyword + sentiment analysis
    kw = analyze_keywords_fallback(words)
    combined = apply_nuance(kw, sentiment)
    final = normalize_scores(combined)
    dominant = max(final, key=final.get)
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    sl = "positive" if sentiment > 0.1 else ("negative" if sentiment < -0.1 else "neutral")
    top3 = [{"emotion": e, "score": s} for e, s in ranked[:3]]
    return {"scores": final, "dominant": {"emotion": dominant, "score": final[dominant]}, "top_3": top3, "sentiment": sentiment, "sentiment_label": sl, "method": method, "word_count": len(words)}


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "EmotionSense API", "dl_available": DL_AVAILABLE, "version": "2.0.0"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    return jsonify(analyze_emotion(text))


@app.route("/api/analyze-batch", methods=["POST"])
def analyze_batch():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "No texts provided"}), 400
    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "texts must be non-empty"}), 400
    return jsonify({"results": [analyze_emotion(t) for t in texts]})


@app.route("/api/emotions")
def emotions():
    return jsonify({"emotions": [
        {"id": "happy", "emoji": "😊", "name": "Happy", "color": "#ffd700"},
        {"id": "sad", "emoji": "😢", "name": "Sad", "color": "#4682b4"},
        {"id": "angry", "emoji": "😡", "name": "Angry", "color": "#ff4500"},
        {"id": "fearful", "emoji": "😨", "name": "Fearful", "color": "#800080"},
        {"id": "loving", "emoji": "🥰", "name": "Loving", "color": "#ff69b4"},
        {"id": "surprised", "emoji": "😲", "name": "Surprised", "color": "#ffa500"},
        {"id": "calm", "emoji": "😌", "name": "Calm", "color": "#008080"},
        {"id": "neutral", "emoji": "😐", "name": "Neutral", "color": "#808080"},
    ]})


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


if __name__ == "__main__":
    print("=" * 50)
    print("  EmotionSense API Server v2.0")
    print("  Deep Learning Emotion Recognition")
    print("=" * 50)
    print(f"  DL Model: {"Available" if DL_AVAILABLE else "Not loaded"}")
    print(f"  DL Engine: {engine_status}")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
