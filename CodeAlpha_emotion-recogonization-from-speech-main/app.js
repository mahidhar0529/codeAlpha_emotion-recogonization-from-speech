// =============== SPEECH RECOGNITION SETUP ===============
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;

if (!recognition) {
    document.getElementById('micStatus').textContent = 'Browser not supported';
    document.getElementById('micBtn').disabled = true;
    document.getElementById('micBtn').style.opacity = '0.4';
    document.getElementById('micBtn').style.cursor = 'not-allowed';
}

// =============== API CONFIG ===============
const API_BASE = '';
// Set to '' for same-origin, or 'http://localhost:5000' if frontend is served separately

// =============== EMOJI MAP ===============
const emotionEmojis = {
    happy: '\u{1F60A}',
    sad: '\u{1F622}',
    angry: '\u{1F621}',
    fearful: '\u{1F628}',
    loving: '\u{1F970}',
    surprised: '\u{1F632}',
    calm: '\u{1F60C}',
    neutral: '\u{1F610}'
};

const emotionColors = {
    happy: '#ffd700',
    sad: '#4682b4',
    angry: '#ff4500',
    fearful: '#800080',
    loving: '#ff69b4',
    surprised: '#ffa500',
    calm: '#008080',
    neutral: '#808080'
};

// =============== DOM ELEMENTS ===============
const micBtn = document.getElementById('micBtn');
const micStatus = document.getElementById('micStatus');
const soundWaves = document.getElementById('soundWaves');
const transcriptionBox = document.getElementById('transcriptionBox');
const clearBtn = document.getElementById('clearBtn');
const dominantEmoji = document.getElementById('dominantEmoji');
const dominantText = document.getElementById('dominantText');

// Score display elements
const scoreElements = {};
const fillElements = {};
['happy', 'sad', 'angry', 'fearful', 'loving', 'surprised', 'calm', 'neutral'].forEach(emotion => {
    scoreElements[emotion] = document.getElementById(`${emotion}Score`);
    fillElements[emotion] = document.getElementById(`${emotion}Fill`);
});

// =============== STATE ===============
let isListening = false;
let isSpeechActive = false;
let finalTranscript = '';
let interimTranscript = '';
let currentScores = {
    happy: 0, sad: 0, angry: 0, fearful: 0,
    loving: 0, surprised: 0, calm: 0, neutral: 0
};
let analysisTimeout = null;
let isProcessing = false;

// =============== UTILITY FUNCTIONS ===============
function showNotification(message, isError = false) {
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notif = document.createElement('div');
    notif.className = `notification${isError ? ' error' : ''}`;
    notif.textContent = message;
    document.body.appendChild(notif);

    requestAnimationFrame(() => {
        notif.classList.add('show');
    });

    setTimeout(() => {
        notif.classList.remove('show');
        setTimeout(() => notif.remove(), 500);
    }, 3000);
}

// =============== API CALLS ===============
async function analyzeWithAPI(text) {
    if (!text || !text.trim()) return null;
    
    try {
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text.trim() })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.warn('API call failed, falling back to client-side analysis:', error.message);
        return null;
    }
}

async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('API Server:', data);
            return true;
        }
    } catch (e) {
        console.log('API server not available, using client-side analysis');
    }
    return false;
}

// =============== CLIENT-SIDE FALLBACK ANALYSIS ===============
const emotionKeywords = {
    happy: [
        'happy', 'joy', 'joyful', 'glad', 'cheerful', 'delighted', 'thrilled',
        'excited', 'wonderful', 'fantastic', 'amazing', 'great', 'awesome',
        'beautiful', 'lovely', 'elated', 'merry', 'jolly', 'pleased',
        'satisfied', 'content', 'laughing', 'smile', 'laughter', 'fun',
        'funny', 'hilarious', 'celebrate', 'celebration', 'party', 'yay',
        'woohoo', 'hurray', 'superb', 'magnificent', 'splendid',
        'terrific', 'fabulous', 'marvelous', 'glorious',
        'upbeat', 'optimistic', 'hopeful', 'positive', 'bright',
    ],
    sad: [
        'sad', 'unhappy', 'depressed', 'miserable', 'gloomy', 'sorrow',
        'sorrowful', 'heartbroken', 'broken', 'lonely', 'alone', 'grief',
        'mourning', 'cry', 'crying', 'tears', 'weep', 'weeping',
        'disappointed', 'hopeless', 'helpless', 'hurt', 'pain', 'painful',
        'suffering', 'regret', 'regretful', 'melancholy', 'blue', 'down',
        'low', 'bummed', 'devastated', 'despair', 'desperate',
        'somber', 'dreary', 'dismal', 'heartache', 'lament',
    ],
    angry: [
        'angry', 'mad', 'furious', 'rage', 'raging', 'irritated',
        'irritation', 'annoyed', 'annoyance', 'frustrated', 'frustration',
        'hate', 'hateful', 'hostile', 'aggression', 'aggressive', 'violent',
        'fury', 'wrath', 'livid', 'outraged', 'enraged', 'bitter',
        'bitterness', 'resentful', 'resentment', 'fuming', 'seething',
        'infuriated', 'incensed', 'indignant', 'irate', 'cross',
        'grumpy', 'cranky', 'fierce',
    ],
    fearful: [
        'scared', 'afraid', 'fearful', 'frightened', 'terrified', 'terror',
        'panic', 'panicked', 'anxious', 'anxiety', 'nervous', 'worried',
        'worry', 'dread', 'dreadful', 'horrified', 'horror', 'alarmed',
        'alarm', 'tense', 'uneasy', 'restless', 'overwhelmed', 'threatened',
        'intimidated', 'shaky', 'trembling', 'petrified', 'spooked',
        'paranoid', 'suspicious', 'apprehensive', 'distressed', 'unease',
        'jittery', 'fidgety', 'timid', 'shy',
    ],
    loving: [
        'love', 'loved', 'loving', 'adore', 'adoration', 'affection',
        'affectionate', 'caring', 'care', 'compassionate', 'compassion',
        'tender', 'tenderness', 'passion', 'passionate', 'romance',
        'romantic', 'darling', 'dear', 'beloved', 'sweet', 'sweetheart',
        'lovely', 'warm', 'warmth', 'gentle', 'gentleness', 'kindness',
        'kind', 'hug', 'embrace', 'cherish', 'devoted', 'fond', 'attached',
        'protective', 'cuddle', 'snuggle', 'heart', 'darling',
    ],
    surprised: [
        'surprise', 'surprised', 'astonishing', 'astonished',
        'astounding', 'shock', 'shocked', 'shocking', 'unexpected', 'sudden',
        'wow', 'incredible', 'unbelievable', 'stunning', 'startled',
        'startle', 'amazed', 'wonder', 'miraculous',
        'extraordinary', 'staggering', 'mind-blowing', 'speechless',
        'dumbfounded', 'bewildered', 'stunned', 'flabbergasted',
        'gobsmacked', 'dumbstruck', 'stupefied',
    ],
    calm: [
        'calm', 'peaceful', 'peace', 'relaxed', 'relaxation', 'serene',
        'serenity', 'tranquil', 'tranquility', 'composed', 'composure',
        'balanced', 'balance', 'harmony', 'harmonious', 'gentle', 'easy',
        'chill', 'quiet', 'stillness', 'still', 'content', 'centered',
        'grounded', 'mindful', 'meditation', 'zen', 'soothing', 'restful',
        'placid', 'untroubled', 'unruffled', 'collected', 'poised', 'cool',
        'mellow', 'laid-back', 'undisturbed',
    ]
};

function analyzeEmotionFallback(text) {
    const words = text.toLowerCase()
        .replace(/[^a-z\s'-]/g, '')
        .split(/\s+/)
        .filter(w => w.length > 0);

    if (words.length === 0) {
        return { happy: 0, sad: 0, angry: 0, fearful: 0, loving: 0, surprised: 0, calm: 0, neutral: 100 };
    }

    const scores = {};
    let totalMatches = 0;

    for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
        let count = 0;
        for (const word of words) {
            for (const keyword of keywords) {
                if (word.includes(keyword)) {
                    count++;
                    break;
                }
            }
        }
        scores[emotion] = count;
        totalMatches += count;
    }

    if (totalMatches === 0) {
        scores.neutral = words.length * 0.3;
        totalMatches = scores.neutral;
    }

    const percentages = {};
    for (const emotion of Object.keys(emotionKeywords)) {
        percentages[emotion] = totalMatches > 0
            ? Math.round((scores[emotion] / totalMatches) * 100)
            : 0;
    }
    percentages.neutral = scores.neutral
        ? Math.round((scores.neutral / totalMatches) * 100)
        : (totalMatches === 0 ? 100 : 0);

    const sum = Object.values(percentages).reduce((a, b) => a + b, 0);
    if (sum > 0 && sum !== 100) {
        const factor = 100 / sum;
        for (const key of Object.keys(percentages)) {
            percentages[key] = Math.round(percentages[key] * factor);
        }
    }

    // Fix rounding to sum to 100
    const finalSum = Object.values(percentages).reduce((a, b) => a + b, 0);
    if (finalSum !== 100) {
        const maxKey = Object.keys(percentages).reduce((a, b) => percentages[a] > percentages[b] ? a : b);
        percentages[maxKey] += (100 - finalSum);
    }

    return percentages;
}

// =============== EMOTION ANALYSIS ===============
async function analyzeEmotion(text) {
    // Try API first
    const apiResult = await analyzeWithAPI(text);
    
    if (apiResult && apiResult.scores) {
        currentScores = apiResult.scores;
        return apiResult;
    }
    
    // Fallback to client-side
    const scores = analyzeEmotionFallback(text);
    currentScores = scores;
    
    const dominant = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
    const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    
    return {
        scores: scores,
        dominant: { emotion: dominant, score: scores[dominant] },
        top_3: ranked.slice(0, 3).map(([e, s]) => ({ emotion: e, score: s })),
        sentiment: 0,
        sentiment_label: 'neutral',
        method: 'client',
        word_count: text.trim().split(/\s+/).length
    };
}

function resetScores() {
    const reset = {
        happy: 0, sad: 0, angry: 0, fearful: 0,
        loving: 0, surprised: 0, calm: 0, neutral: 0
    };
    currentScores = reset;
    return reset;
}

// =============== UI UPDATE ===============
function updateEmotionUI(result) {
    if (!result || !result.scores) return;

    const scores = result.scores;
    const dominant = result.dominant;

    // Update each emotion card
    for (const [emotion, score] of Object.entries(scores)) {
        if (fillElements[emotion]) {
            fillElements[emotion].style.width = `${score}%`;
            fillElements[emotion].style.background = `linear-gradient(90deg, ${emotionColors[emotion]}, ${emotionColors[emotion]}dd)`;
        }
        if (scoreElements[emotion]) {
            scoreElements[emotion].textContent = `${score}%`;
        }
    }

    // Highlight dominant emotion card
    document.querySelectorAll('.emotion-card').forEach(card => {
        card.classList.remove('dominant');
        if (card.dataset.emotion === dominant.emotion) {
            card.classList.add('dominant');
        }
    });

    // Update dominant display
    if (dominantEmoji) {
        dominantEmoji.textContent = emotionEmojis[dominant.emotion] || '\u{1F3A4}';
    }
    if (dominantText) {
        const emotionName = dominant.emotion.charAt(0).toUpperCase() + dominant.emotion.slice(1);
        dominantText.textContent = `${emotionName} (${dominant.score}%)`;
    }

    // Update the dominant section background
    const dominantSection = document.querySelector('.dominant-section');
    if (dominantSection) {
        dominantSection.style.borderColor = emotionColors[dominant.emotion] || 'var(--card-border)';
        dominantSection.style.background = `rgba(255, 255, 255, 0.08)`;
    }
}

function updateTranscription(finalText, interimText) {
    transcriptionBox.innerHTML = '';
    
    if (!finalText && !interimText) {
        transcriptionBox.innerHTML = '<span class="placeholder-text">No speech detected yet. Tap the microphone to start...</span>';
        return;
    }

    if (finalText) {
        const finalSpan = document.createElement('span');
        finalSpan.className = 'final-text';
        finalSpan.textContent = finalText;
        transcriptionBox.appendChild(finalSpan);
    }

    if (interimText) {
        const interimSpan = document.createElement('span');
        interimSpan.className = 'interim-text';
        interimSpan.textContent = interimText;
        transcriptionBox.appendChild(interimSpan);
    }

    // Auto-scroll to bottom
    transcriptionBox.scrollTop = transcriptionBox.scrollHeight;
}

// =============== SPEECH RECOGNITION EVENTS ===============
if (recognition) {
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        isListening = true;
        isSpeechActive = false;
        micBtn.classList.add('active');
        micStatus.textContent = 'Listening...';
        soundWaves.classList.add('active');
    };

    recognition.onresult = (event) => {
        isSpeechActive = true;
        micBtn.classList.remove('active');
        micBtn.classList.add('listening');
        micStatus.textContent = 'Speaking...';

        interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
            } else {
                interimTranscript += transcript;
            }
        }

        updateTranscription(finalTranscript, interimTranscript);

        // Debounced analysis
        if (analysisTimeout) clearTimeout(analysisTimeout);
        analysisTimeout = setTimeout(async () => {
            const textToAnalyze = (finalTranscript + interimTranscript).trim();
            if (textToAnalyze) {
                isProcessing = true;
                const result = await analyzeEmotion(textToAnalyze);
                updateEmotionUI(result);
                isProcessing = false;
            }
        }, 800);
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        let message = 'Error occurred';
        switch (event.error) {
            case 'no-speech': message = 'No speech detected'; break;
            case 'audio-capture': message = 'Microphone not accessible'; break;
            case 'not-allowed': message = 'Microphone permission denied'; break;
            case 'network': message = 'Network error'; break;
            default: message = `Error: ${event.error}`;
        }
        micStatus.textContent = message;
        showNotification(message, true);
        stopListening();
    };

    recognition.onend = () => {
        if (isListening) {
            // Automatically restart if still listening
            try {
                recognition.start();
            } catch (e) {
                // Ignore if already started
            }
        } else {
            micBtn.classList.remove('active', 'listening');
            soundWaves.classList.remove('active');
            if (!finalTranscript && !interimTranscript) {
                micStatus.textContent = 'Tap to Speak';
            } else {
                micStatus.textContent = 'Tap to continue...';
            }
        }
    };
}

// =============== CONTROLS ===============
function startListening() {
    if (!recognition) {
        showNotification('Speech recognition not supported in this browser', true);
        return;
    }
    
    if (isListening) return;
    
    // Reset if needed
    if (recognition) {
        try {
            recognition.start();
        } catch (e) {
            console.warn('Recognition start error:', e);
        }
    }
}

function stopListening() {
    isListening = false;
    if (recognition) {
        try {
            recognition.stop();
        } catch (e) {
            // Ignore
        }
    }
    micBtn.classList.remove('active', 'listening');
    soundWaves.classList.remove('active');
}

function toggleListening() {
    if (isListening) {
        stopListening();
        micStatus.textContent = finalTranscript ? 'Tap to continue...' : 'Tap to Speak';
    } else {
        startListening();
    }
}

function clearTranscription() {
    finalTranscript = '';
    interimTranscript = '';
    updateTranscription('', '');
    resetScores();
    updateEmotionUI({
        scores: { happy: 0, sad: 0, angry: 0, fearful: 0, loving: 0, surprised: 0, calm: 0, neutral: 100 },
        dominant: { emotion: 'neutral', score: 100 }
    });
    dominantEmoji.textContent = '\u{1F3A4}';
    dominantText.textContent = 'Say something...';
    
    const dominantSection = document.querySelector('.dominant-section');
    if (dominantSection) {
        dominantSection.style.borderColor = 'var(--card-border)';
        dominantSection.style.background = 'var(--card-bg)';
    }
    
    micStatus.textContent = 'Tap to Speak';
    showNotification('Transcription cleared');
}

// =============== EVENT LISTENERS ===============
micBtn.addEventListener('click', toggleListening);

clearBtn.addEventListener('click', clearTranscription);

// Keyboard shortcut: Space to toggle
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        toggleListening();
    }
});

// =============== NOTIFICATION STYLES ===============
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
.notification {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%) translateY(100px);
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-sm);
    padding: 12px 24px;
    color: var(--text-primary);
    font-size: 0.9rem;
    font-weight: 500;
    z-index: 1000;
    opacity: 0;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    pointer-events: none;
}
.notification.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}
.notification.error {
    border-color: #ff4757;
    background: rgba(255, 71, 87, 0.15);
}
`;
document.head.appendChild(notificationStyles);

// =============== INIT ===============
(async function init() {
    const apiAvailable = await checkAPIHealth();
    if (apiAvailable) {
        console.log('Using server-side ML analysis');
        showNotification('Connected to EmotionSense API');
    } else {
        console.log('Using client-side analysis fallback');
    }
})();
