"""
PyTorch Bi-LSTM Model for Emotion Classification
"""

import torch
import torch.nn as nn


class EmotionBiLSTM(nn.Module):
  """Bidirectional LSTM model for text emotion classification."""

  def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, n_layers=2, n_classes=8, dropout=0.3):
    super(EmotionBiLSTM, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(
      embedding_dim,
      hidden_dim,
      num_layers=n_layers,
      batch_first=True,
      bidirectional=True,
      dropout=dropout if n_layers > 1 else 0
    )
    self.dropout = nn.Dropout(dropout)
    self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
    self.fc1 = nn.Linear(hidden_dim * 2, 64)
    self.fc2 = nn.Linear(64, n_classes)

  def forward(self, x):
    # x: (batch, seq_len)
    embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
    lstm_out, (hidden, cell) = self.lstm(embedded)
    # Concatenate final forward and backward hidden states
    hidden_fwd = hidden[-2, :, :]  # forward direction, last layer
    hidden_bwd = hidden[-1, :, :]  # backward direction, last layer
    combined = torch.cat((hidden_fwd, hidden_bwd), dim=1)  # (batch, hidden_dim*2)
    combined = self.batch_norm(combined)
    combined = self.dropout(combined)
    out = torch.relu(self.fc1(combined))
    out = self.dropout(out)
    out = self.fc2(out)
    return out


class EmotionDataset(torch.utils.data.Dataset):
  """PyTorch Dataset for emotion classification."""

  def __init__(self, texts, labels, tokenizer, max_len=50):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = self.texts[idx]
    label = self.labels[idx]
    tokens = self.tokenizer(text, self.max_len)
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

