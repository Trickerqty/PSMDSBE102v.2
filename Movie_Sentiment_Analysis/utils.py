import torch
from torch.nn.utils.rnn import pad_sequence

# === Collate Function ===
def custom_collate_fn(batch):
    if isinstance(batch, dict) and "results" in batch:
        batch = batch["results"]

    if not isinstance(batch[0], dict):
        raise TypeError(f"Expected batch of dicts but got: {type(batch[0])}, value: {batch[0]}")

    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attention_masks = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
        "attention_mask": pad_sequence(attention_masks, batch_first=True, padding_value=0),
        "labels": labels
    }

# === Format Probabilities ===
def format_prob(prob, index_to_class):
    return {index_to_class[i]: float(p) for i, p in enumerate(prob)}