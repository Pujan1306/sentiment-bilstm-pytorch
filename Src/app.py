import pickle
import mlflow
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from Src.dataset import Vocabulary
from Src.model import SentimentBiLSTM

app = FastAPI(title="Sentiment BiLSTM Model", version="1.0")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

LABELS = {
    0: "Negative",
    1: "Positive",
    2: "Neutral",
    3: "Irrelevant"
}

MODEL_NAME = "Sentimental_BiLSTM_Pytorch"
STAGE_OR_VERSION = "latest"

model = None
vocab = None

try:
    try:
        vocab_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}/{STAGE_OR_VERSION}/vocab/vocab.pkl"
        )
        print("Loaded vocabulary from MLflow Registry.")
    except Exception as artifact_err:
        print(f"MLflow artifact not found. Falling back to local disk: 'models/vocab.pkl'")
        vocab_local_path = Path("models/vocab.pkl")

    with open(vocab_local_path, "rb") as f:
        vocab = pickle.load(f)

    model_uri = f"models:/{MODEL_NAME}/{STAGE_OR_VERSION}"
    print(f"Fetching registered model: {model_uri}...")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    print("Serving engine linked successfully to model assets.")

except Exception as e:
    print(f"Failed to load assets! Error detail:\n{e}")
    model = None
    vocab = None

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(payload: PredictRequest):
    if model is None or vocab is None:
        raise HTTPException(
            status_code=503,
            detail="Serving engine is uninitialized. Check terminal startup logs for exact error details."
        )

    sentence = payload.text

    if not sentence or not sentence.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    numericalized = vocab.numericalize(sentence)

    if len(numericalized) == 0:
        unk_idx = vocab.stoi.get("<UNK>", vocab.stoi.get("<unk>", 0))
        numericalized = [unk_idx]

    tensor = torch.tensor(numericalized, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)
        pred_idx = torch.argmax(prediction, dim=1).item()

    return {
        "text": sentence,
        "sentiment": LABELS.get(pred_idx, "Unknown")
    }