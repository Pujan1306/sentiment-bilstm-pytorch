import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Src.dataset import Vocabulary, TextDataset, PadCollate
from Src.model import SentimentBiLSTM
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch as mlp
import os

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment_BiLSTM_Pipeline")

df = pd.read_csv("Data/Processed/twitter_training_processed.csv")

df = df.dropna(subset=["texts"])
df = df[df["texts"].astype(str).str.strip().ne("")]

processed_texts = df["texts"].tolist()
processed_labels = df["labels"].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    processed_texts,
    processed_labels,
    test_size=0.2,
    random_state=42,
    stratify=processed_labels,
)

vocab = Vocabulary(freq_threshold=1)
vocab.build_vocab(processed_texts)

os.makedirs("models", exist_ok=True)
with open("models/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

train_dataset = TextDataset(texts=train_texts, labels=train_labels, vocab=vocab)
val_dataset = TextDataset(texts=val_texts, labels=val_labels, vocab=vocab)

pad_collate = PadCollate(pad_idx=vocab.stoi["<PAD>"])

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentBiLSTM(
    vocab_size=len(vocab.stoi),
    embed_dim=64,
    hidden_dim=32,
    pad_idx=vocab.stoi["<PAD>"]
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
best_val_loss = float("inf")
with mlflow.start_run() as run:
    mlflow.log_params({
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": epochs,
        "embed_dim": 64,
        "hidden_dim": 32,
        "data_source": "Kaggle_Twitter_Entity_Sentiment_Analysis_Dataset",
    })

    print(f"Starting training for {epochs} epochs on device: {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_texts, batch_labels in train_loader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            prediction = model(batch_texts)
            loss = criterion(prediction, batch_labels)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(prediction, dim=1)
            total_correct += torch.sum(pred == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            running_loss += loss.item()  * batch_labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = (total_correct / total_samples)

        model.eval()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)

                prediction = model(batch_texts)
                loss = criterion(prediction, batch_labels)

                pred = torch.argmax(prediction, dim=1)
                total_correct += torch.sum(pred == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

                running_loss += loss.item() * batch_labels.size(0)

            epoch_val_loss = running_loss / total_samples
            epoch_val_acc = (total_correct / total_samples)

        step = epoch + 1
        mlflow.log_metric("train_loss", epoch_loss, step=step)
        mlflow.log_metric("train_accuracy", epoch_acc, step=step)
        mlflow.log_metric("val_loss", epoch_val_loss, step=step)
        mlflow.log_metric("val_accuracy", epoch_val_acc, step=step)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_bilstm_model.pth")

    print(f"\nLoading best model checkpoint (Val Loss: {best_val_loss:.4f}) for MLflow...")
    model.load_state_dict(torch.load("models/best_bilstm_model.pth"))

    mlp.log_model(
        pytorch_model=model,
        artifact_path="sentiment_model",
        registered_model_name="Sentimental_BiLSTM_Pytorch",
        serialization_format="pickle",
    )

    mlflow.log_artifact("models/vocab.pkl", artifact_path="vocab")
    print(f"Training Complete! Model registered in MLflow Registry (Run ID: {run.info.run_id})")





