import json
import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast

class CSHSMDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        self.cat_to_id = {
            "Gender": 0, "Political": 1, "Religion": 2, 
            "Geo-political": 3, "Violence": 4, "Others": 5
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        record = self.data[idx]
        text = record["Comment"]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        binary_label = 1 if record["Label"] == "Hate" else 0
        
        cat_str = record.get("Hate Category", "")
        # Mask non-hate categories with -1
        if binary_label == 1 and cat_str in self.cat_to_id:
            cat_label = self.cat_to_id[cat_str]
        else:
            cat_label = -1
            
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "category_label": torch.tensor(cat_label, dtype=torch.long),
            "text": text
        }
