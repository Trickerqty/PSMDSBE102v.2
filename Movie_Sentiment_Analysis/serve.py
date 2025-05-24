import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.requests import Request
from ray import serve
from transformers import BertTokenizer
from models import BertClassifier
from config import index_to_class
from pathlib import Path

# === Define Pydantic model for input ===
class SentimentRequest(BaseModel):
    text: str

# === FastAPI App ===
app = FastAPI(
    title="Movie Sentiment Analysis",
    description="Predict movie review sentiment: positive, neutral, or negative.",
    version="1.0"
)

# === Ray Serve Deployment ===
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2})
@serve.ingress(app)
class MovieSentimentDeploymentFromCheckpoint:
    def __init__(self, model_path: str):
        self.model = BertClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.index_to_class = index_to_class
        print(f"Loaded model from: {model_path}")

    @app.post("/predict/")
    async def predict(self, request: SentimentRequest):
        text = request.text
        tokens = self.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(tokens["input_ids"], tokens["attention_mask"])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_index = probs.argmax(1)[0]
            pred_label = self.index_to_class[pred_index]

        return {
            "input_text": text,
            "prediction": pred_label,
            "probabilities": {
                self.index_to_class[i]: float(p) for i, p in enumerate(probs[0])
            }
        }

# === Manual Launch with Correct Route Prefix ===
def launch_service_from_checkpoint(checkpoint_dir: str):
    model_path = os.path.join(checkpoint_dir, "model.pt")
    print("Starting FastAPI with Ray Serve...")
    print("API available at: http://localhost:8000")
    print("UI at: http://localhost:8000/docs")
    serve.run(
        MovieSentimentDeploymentFromCheckpoint.bind(model_path=model_path),
        route_prefix="/"
    )
