import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig

class MultitaskXLMR(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_categories=6):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        
        # Binary Hate/Non-Hate Head
        self.binary_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        
        # 6-Class Hate Category Head
        self.category_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_categories)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled output (CLS token representation)
        pooled_output = outputs.pooler_output
        
        binary_logits = self.binary_head(pooled_output)
        category_logits = self.category_head(pooled_output)
        
        return binary_logits, category_logits
