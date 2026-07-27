from collections import Counter
import unicodedata
import re
import torch
from torch.utils.data import Dataset

class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 2
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    @staticmethod
    def clean_text(text):

        text = unicodedata.normalize('NFKD', text)

        # 2. Obliterate invisible/zero-width characters (soft hyphens, zero-width joiners)
        text = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff\u00ad]', '', text)

        # 3. Strip HTML tags & Web URLs BEFORE removing punctuation
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # 4. Lowercase
        text = text.lower()

        # 5. Drop standalone decomposed accent marks (ASCII conversion)
        text = text.encode('ascii', 'ignore').decode('utf-8')

        # 6. Keep ONLY lowercase ASCII letters, numbers, and spaces (strips punctuation & underscores)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # 7. Collapse all extra whitespace into a single space
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text):
        cleaned_text = self.clean_text(text)
        return cleaned_text.split()

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2idx=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2idx = label2idx or {
            "Negative": 0,
            "Positive": 1,
            "Neutral": 2,
            "Irrelevant": 3
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        numericalized_text = self.vocab.numericalize(self.texts[index])
        raw_label = self.labels[index]
        label_idx = self.label2idx[raw_label] if isinstance(raw_label, str) else raw_label

        return torch.tensor(numericalized_text, dtype=torch.long), label_idx

class PadCollate:
    def __init__(self, pad_idx = 0, label2idx=None):
        self.pad_idx = pad_idx
        self.label2idx = label2idx or {
            "Negative": 0,
            "Positive": 1,
            "Neutral": 2,
            "Irrelevant": 3
        }

    def __call__(self, batch):
        texts = [item[0] for item in batch]

        encoded_labels = [
            self.label2idx[item[1]] if isinstance(item[1], str) else item[1]
            for item in batch
        ]
        labels = torch.tensor(encoded_labels, dtype=torch.long)

        texts_padded = torch.nn.utils.rnn.pad_sequence(
            texts, batch_first=True, padding_value=self.pad_idx
        )
        return texts_padded, labels