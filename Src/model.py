import torch
import torch.nn as nn

class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx = 0, dropout = 0.3):
        super(SentimentBiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim * 2, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedding = self.dropout(self.embedding(text))

        _, (hidden, _) = self.lstm(embedding)

        hidden_forward = hidden[-2, :, :]
        hidden_backword = hidden[-1, :, :]

        hidden_cat = torch.cat((hidden_forward, hidden_backword), dim=1)
        hidden_cat = self.dropout(hidden_cat)

        return self.fc(hidden_cat)