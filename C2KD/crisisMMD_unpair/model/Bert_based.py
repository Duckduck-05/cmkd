import torch.nn as nn
from transformers import BertModel



class BertClassifier(nn.Module):
    def __init__(
        self,
        pretrained="bert-base-uncased",
        num_classes=2,
        feat_dim=768,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = BertModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        features = outputs.last_hidden_state[:, 0]  # [CLS]
        features = self.dropout(features)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

