import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, dropout_p=0.3, num_classes=3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        return self.fc(self.dropout(pooled_output))
