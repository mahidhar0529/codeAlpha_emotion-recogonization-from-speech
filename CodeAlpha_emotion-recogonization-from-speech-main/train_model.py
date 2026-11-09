"""
Training data generator and model training for EmotionSense.
Generates 5000+ diverse emotion-labeled sentences and trains a Bi-LSTM.
"""

import os, re, json, random, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from emotion_model import EmotionBiLSTM, EmotionDataset

random.seed(42)
torch.manual_seed(42)

EMOTIONS = ["happy","sad","angry","fearful","loving","surprised","calm","neutral"]
EMO_MAP = {e:i for i,e in enumerate(EMOTIONS)}

EMO_WORDS = {
  "happy": ["happy","joy","joyful","glad","cheerful","delighted","thrilled","excited","wonderful","fantastic","amazing","great","awesome","beautiful","lovely","elated","pleased","content","laughing","smile","fun","funny","hilarious","celebrate","superb","terrific","fabulous","upbeat","optimistic","hopeful","positive","bright","radiant","euphoric","jubilant","blissful","ecstatic","overjoyed","grateful","blessed","merry","jolly","triumphant","victorious","sunny","vibrant"],
  "sad": ["sad","unhappy","depressed","miserable","gloomy","sorrow","sorrowful","heartbroken","broken","lonely","alone","grief","mourning","cry","crying","tears","weep","disappointed","hopeless","helpless","hurt","pain","suffering","regret","melancholy","blue","down","low","devastated","despair","desperate","somber","dreary","dismal","lament","anguish","agony","woe","forlorn","dejected","morose","wretched","glum","sullen","bleak"],
  "angry": ["angry","mad","furious","rage","irritated","annoyed","frustrated","hate","hateful","hostile","aggressive","violent","fury","wrath","livid","outraged","enraged","bitter","resentful","fuming","seething","infuriated","incensed","indignant","irate","cross","grumpy","cranky","fierce","exasperated","agitated","provoked","stormy","volatile","testy","snappy"],
  "fearful": ["scared","afraid","fearful","frightened","terrified","terror","panic","anxious","anxiety","nervous","worried","worry","dread","dreadful","horrified","alarmed","tense","uneasy","restless","overwhelmed","threatened","intimidated","shaky","trembling","petrified","apprehensive","distressed","jittery","timid","edgy","wary","startled","unnerved"],
  "loving": ["love","loved","loving","adore","affection","affectionate","caring","care","compassionate","tender","tenderness","passion","passionate","romance","romantic","darling","dear","beloved","sweet","sweetheart","lovely","warm","warmth","gentle","gentleness","kindness","kind","hug","embrace","cherish","devoted","fond","adoring","devotion","endearment"],
  "surprised": ["surprise","surprised","astonishing","astonished","astounding","shock","shocked","unexpected","sudden","wow","incredible","unbelievable","stunning","startled","amazed","wonder","miraculous","extraordinary","staggering","mind-blowing","speechless","dumbfounded","bewildered","stunned","flabbergasted","astounded","floored","dazzled","perplexed","baffled"],
  "calm": ["calm","peaceful","peace","relaxed","relaxation","serene","serenity","tranquil","tranquility","composed","balanced","balance","harmony","harmonious","gentle","easy","chill","quiet","stillness","still","content","centered","grounded","mindful","meditation","zen","soothing","restful","placid","collected","poised","cool","mellow","equable","sedate","contemplative"],
}


# Sentence templates for data generation
TEMPLATES = {
  "happy": [
    "i feel {0} today", "i am so {0} right now", "this is {0} news",
    "what a {0} day", "i am absolutely {0}", "feeling so {0}",
    "everything is {0}", "this makes me {0}", "i am really {0}",
    "so {0} about this", "feeling {0} and grateful",
    "had a {0} time today", "such a {0} experience",
    "this is the best day ever", "could not be more {0}",
    "so {0} i could burst", "truly {0} right now",
    "today has been {0}", "i feel {0} and carefree",
    "absolutely {0} about the news", "really {0} with how things went",
  ],
  "sad": [
    "i feel so {0}", "i am really {0} today", "this is {0}",
    "feeling {0} and lonely", "everything feels {0}",
    "i am deeply {0}", "this situation is {0}",
    "so {0} about what happened", "cannot stop feeling {0}",
    "my heart is {0}", "this makes me {0}",
    "i feel {0} inside", "really {0} about everything",
    "such a {0} feeling", "wish i wasnt so {0}",
    "feeling quite {0} lately", "so much {0} in my heart",
    "today is a {0} day", "nothing but {0} feelings",
    "i cant deal with this {0} feeling",
  ],
  "angry": [
    "i am so {0}", "this makes me {0}", "i am really {0} right now",
    "feeling {0} and frustrated", "this is absolutely {0}",
    "so {0} i could scream", "this situation is {0}",
    "i feel {0} about this", "really {0} with the way things are",
    "i am completely {0}", "this is getting {0}",
    "feeling {0} and bitter about it", "so {0} i cant think straight",
    "i am {0} beyond belief", "this whole thing is {0}",
    "why do i feel so {0}", "you make me so {0}",
    "this is so {0} i could explode",
  ],
  "fearful": [
    "i am so {0}", "feeling really {0} about this",
    "this is {0}", "i feel {0} and anxious",
    "so {0} i cant sleep", "everything feels {0}",
    "i am {0} of what might happen", "this situation is {0}",
    "really {0} about the future", "i am completely {0}",
    "feeling {0} and overwhelmed", "so {0} i can barely breathe",
    "i am {0} beyond words", "this whole thing is {0}",
    "why do i feel so {0}", "that was a truly {0} experience",
    "i am {0} something bad will happen",
  ],
  "loving": [
    "i {0} you so much", "you are so {0}", "this is {0}",
    "feeling so {0} today", "i feel so much {0} for you",
    "you are {0} and amazing", "i {0} spending time with you",
    "so {0} towards everyone today", "such a {0} person",
    "i {0} everything about this", "you make me feel so {0}",
    "i am filled with {0}", "so much {0} in my heart",
    "you are my {0}", "i {0} you more than anything",
    "this is such a {0} moment", "i {0} being with you",
  ],
  "surprised": [
    "i am so {0}", "this is {0}", "i am completely {0}",
    "what a {0} turn of events", "this is really {0}",
    "i feel {0} by this news", "so {0} i cant speak",
    "totally {0} by what happened", "this is absolutely {0}",
    "i am {0} beyond words", "well this is {0}",
    "so {0} i dont know what to say", "what a {0} development",
    "i did not expect this at all", "this came out of nowhere",
  ],
  "calm": [
    "i feel very {0}", "everything is {0}", "i am so {0} right now",
    "feeling {0} and centered", "this is really {0}",
    "i am completely {0}", "so {0} and at peace",
    "feeling {0} and balanced", "i am trying to stay {0}",
    "everything feels {0}", "just staying {0} about it",
    "i feel {0} and relaxed", "a very {0} state of mind",
    "being {0} helps me think clearly", "maintaining my {0}",
  ],
}

NEUTRAL_TEMPLATES = [
  "i went to the store", "the sky is blue today",
  "it is monday morning", "i have a meeting at three",
  "the book is on the table", "the car is parked outside",
  "it is raining right now", "the train arrives at noon",
  "i need to buy groceries", "the door is open",
  "he walked to the park", "she reads the newspaper",
  "the computer is on the desk", "my phone is charging",
  "the sun is shining", "i will go tomorrow",
  "the restaurant opens at six", "water is essential for life",
  "the bus stops here", "i have an appointment",
  "the office is on the third floor", "please close the window",
  "the meeting starts at ten", "i need to make a call",
  "the package arrived yesterday", "the files are on the server",
  "they live in a blue house", "the weather forecast looks normal",
  "i saw him at the market", "she studies at the library",
  "the dinner is at seven pm", "i will send the email later",
  "the conference is next week", "he works at a bank",
  "the children are at school", "i need to fill the form",
  "the documents are ready", "the light is bright in here",
  "the floor is made of wood", "keys are on the counter",
  "i have three siblings", "the movie starts at eight",
  "she drives a red car", "they went to the museum",
  "the paper is printed", "the chair is comfortable",
  "i will call you later", "the report is due friday",
]

# Vocabulary builder
class VocabBuilder:
  def __init__(self):
    self.word2idx = {"<PAD>": 0, "<UNK>": 1}
    self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    self.count = 2

  def add(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.count
      self.idx2word[self.count] = word
      self.count += 1

  def __len__(self):
    return self.count


def tokenize(text, vocab, max_len):
  """Convert text to token ids with padding/truncation."""
  words = re.sub(r"[^a-z\s]", "", text.lower()).split()
  ids = [vocab.word2idx.get(w, 1) for w in words[:max_len]]
  pad_len = max_len - len(ids)
  if pad_len > 0:
    ids.extend([0] * pad_len)
  return ids


def generate_texts(vocab):
  """Generate emotion-labeled training texts."""
  texts, labels = [], []

  # Generate from templates
  for emotion, templates in TEMPLATES.items():
    words = EMO_WORDS[emotion]
    for template in templates:
      for word in words:
        text = template.format(word)
        texts.append(text)
        labels.append(EMO_MAP[emotion])
        for w in re.sub(r"[^a-z\s]", "", text.lower()).split():
          vocab.add(w)

  # Generate neutral texts
  for template in NEUTRAL_TEMPLATES:
    texts.append(template)
    labels.append(EMO_MAP["neutral"])
    for w in re.sub(r"[^a-z\s]", "", template.lower()).split():
      vocab.add(w)

  # Generate additional variations
  intensifiers = ["very", "so", "really", "extremely", "quite", "absolutely", "totally"]
  for emotion, words in EMO_WORDS.items():
    for w in words[:15]:
      for intensifier in intensifiers[:3]:
        text = f"i am {intensifier} {w} today"
        texts.append(text)
        labels.append(EMO_MAP[emotion])
        for vocab_w in re.sub(r"[^a-z\s]", "", text.lower()).split():
          vocab.add(vocab_w)

  return texts, labels


def train_model():
  print("Building vocabulary and generating training data...")
  vocab = VocabBuilder()
  texts, labels = generate_texts(vocab)

  print(f"Generated {len(texts)} training examples")
  print(f"Vocabulary size: {len(vocab)}")

  MAX_LEN = 30
  BATCH_SIZE = 64
  EPOCHS = 30
  LEARNING_RATE = 0.001
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(f"Using device: {DEVICE}")

  # Split into train/val
  combined = list(zip(texts, labels))
  random.shuffle(combined)
  texts[:], labels[:] = zip(*combined) if combined else ([], [])

  split = int(len(texts) * 0.8)
  train_texts, val_texts = texts[:split], texts[split:]
  train_labels, val_labels = labels[:split], labels[split:]

  print(f"Training: {len(train_texts)}, Validation: {len(val_texts)}")

  # Create datasets
  train_ds = EmotionDataset(train_texts, train_labels, lambda t, ml: tokenize(t, vocab, ml), MAX_LEN)
  val_ds = EmotionDataset(val_texts, val_labels, lambda t, ml: tokenize(t, vocab, ml), MAX_LEN)

  train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

  # Initialize model
  model = EmotionBiLSTM(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    n_layers=2,
    n_classes=len(EMOTIONS),
    dropout=0.3
  ).to(DEVICE)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

  best_val_acc = 0.0
  patience_counter = 0
  max_patience = 8

  print("\nStarting training...")
  print("=" * 60)

  for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, targets in train_loader:
      inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      train_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      train_total += targets.size(0)
      train_correct += (predicted == targets).sum().item()

    train_acc = train_correct / train_total * 100

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
      for inputs, targets in val_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        val_total += targets.size(0)
        val_correct += (predicted == targets).sum().item()

    val_acc = val_correct / val_total * 100
    scheduler.step(val_acc)

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      patience_counter = 0
      os.makedirs("model_data", exist_ok=True)
      torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_word2idx": vocab.word2idx,
        "vocab_idx2word": vocab.idx2word,
        "vocab_size": len(vocab),
        "max_len": MAX_LEN,
      }, "model_data/emotion_model.pt")
      print(f"  >> Model saved! Best val accuracy: {val_acc:.2f}%")
    else:
      patience_counter += 1
      if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch}")
        break

  print("=" * 60)
  print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
  print(f"Model saved to model_data/emotion_model.pt")


if __name__ == "__main__":
  train_model()
