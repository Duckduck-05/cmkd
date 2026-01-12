import torch.nn as nn
from transformers import BertModel



class BertTextEncoder(nn.Module):
    def __init__(self, pretrained="bert-base-uncased", out_dim=768):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.proj = nn.Identity()  # optional projection

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_feat = outputs.last_hidden_state[:, 0]  # [CLS]
        return self.proj(cls_feat)
