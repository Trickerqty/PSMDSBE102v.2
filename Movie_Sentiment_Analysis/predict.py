import os
import torch
import pandas as pd
from transformers import BertTokenizer
from models import BertClassifier
from utils import format_prob

# === Torch Predictor Class ===
class TorchPredictor:
    def __init__(self, model, preprocessor=None, index_to_class=None):
        self.model = model
        self.model.eval()
        self.preprocessor = preprocessor
        self.index_to_class = index_to_class

    def __call__(self, batch):
        with torch.inference_mode():
            output = self.model(batch["input_ids"], batch["attention_mask"])
            predictions = torch.argmax(output, dim=1).cpu().numpy()
        return {"output": predictions}

    def predict_proba(self, batch):
        with torch.inference_mode():
            output = self.model(batch["input_ids"], batch["attention_mask"])
            probs = torch.softmax(output, dim=1).cpu().numpy()
        return {"output": probs}

    def get_preprocessor(self):
        return self.preprocessor

    def get_index_to_class(self):
        return self.index_to_class

    @classmethod
    def from_checkpoint(cls, checkpoint, preprocessor=None, index_to_class=None):
        checkpoint_dir = checkpoint.to_directory()
        model_path = os.path.join(checkpoint_dir, "model.pt")
        model = BertClassifier()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return cls(model=model, preprocessor=preprocessor, index_to_class=index_to_class)

# === Batch Prediction Function ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def custom_predict_proba(df, predictor, index_to_class):
    df["text"] = df["title"] + " " + df["description"]
    tokens = tokenizer(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = predictor.model(tokens["input_ids"], tokens["attention_mask"])
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    results = []
    for prob in probs:
        pred_index = int(prob.argmax())
        results.append({
            "prediction": index_to_class[pred_index],
            "probabilities": format_prob(prob, index_to_class)
        })
    return results
