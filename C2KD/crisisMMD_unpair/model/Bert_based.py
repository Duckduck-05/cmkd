import torch.nn as nn
from transformers import BertModel, BertConfig



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


class BertTeacher(nn.Module):
    def __init__(self, num_labels=8, pretrained_name="bert-base-uncased"):
        super().__init__()

        # 🔹 Load pretrained BERT
        self.encoder = BertModel.from_pretrained(
            pretrained_name,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def encode(self, input_ids, attention_mask):
        """
        Latent representation for KD (CLS embedding)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0]  # CLS

    def forward(self, input_ids, attention_mask):
        features = self.encode(input_ids, attention_mask)
        logits = self.classifier(features)
        return logits 

class SmallBertStudent(nn.Module):
    def __init__(self, num_labels=8):
        super().__init__()

        # 🔹 Small BERT config (NO pretrained)
        config = BertConfig(
            vocab_size=30522,
            hidden_size=384,          # ↓ smaller than 768
            num_hidden_layers=6,      # ↓ fewer layers
            num_attention_heads=6,    # must divide hidden_size
            intermediate_size=1536,   # usually 4 * hidden_size
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        self.encoder = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def encode(self, input_ids, attention_mask):
        """
        Return latent representation (CLS token)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # CLS embedding
        return outputs.last_hidden_state[:, 0]  # [B, H]

    def forward(self, input_ids, attention_mask):
        features = self.encode(input_ids, attention_mask)
        logits = self.classifier(features)
        return logits
    

class FeatureProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)