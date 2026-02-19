import torch
import torch.nn as nn
from transformers import AutoModel
from utils.config import Config

class TextEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(Config.TEXT_MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, Config.NUM_LABELS)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        
        logits = self.classifier(cls_embedding)
        return logits
