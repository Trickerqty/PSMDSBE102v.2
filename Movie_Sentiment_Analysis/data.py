import ray
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from config import set_seeds

# === GLOBAL ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_encoder = LabelEncoder()

def load_clean_data(path="TextAnalytics_cleaned.csv", sample_size=250):
    df = pd.read_csv(path).head(sample_size)
    df["label"] = label_encoder.fit_transform(df["Target"])
    return df

def create_ray_datasets(df):
    train_df, val_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=1234)
    train_ds = ray.data.from_pandas(train_df).random_shuffle(seed=1234)
    val_ds = ray.data.from_pandas(val_df).random_shuffle(seed=1234)
    return train_ds, val_ds

def preprocess(batch):
    texts = [str(x) for x in batch["Text"]]
    labels = batch["label"].tolist()

    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )

    return {
        "results": [
            {
                "input_ids": tokens["input_ids"][i],
                "attention_mask": tokens["attention_mask"][i],
                "label": labels[i]
            }
            for i in range(len(texts))
        ]
    }
