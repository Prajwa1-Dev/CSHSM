import pytest
import torch
from src.models.multitask_xlmr import MultitaskXLMR
from src.data.dataset import CSHSMDataset
from transformers import XLMRobertaTokenizerFast

def test_multitask_forward():
    model = MultitaskXLMR()
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones((2, 128))
    
    bin_logits, cat_logits = model(input_ids, attention_mask)
    
    assert bin_logits.shape == (2, 2)
    assert cat_logits.shape == (2, 6)
    
def test_dataset(tmp_path):
    f = tmp_path / "dummy.jsonl"
    with open(f, "a") as fw:
        fw.write('{"Comment": "Test comment", "Label": "Hate", "Hate Category": "Gender"}\n')
        fw.write('{"Comment": "Another", "Label": "Non-Hate", "Hate Category": ""}\n')
    
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    dataset = CSHSMDataset(str(f), tokenizer, max_length=16)
    
    assert len(dataset) == 2
    assert dataset[0]["binary_label"].item() == 1
    assert dataset[0]["category_label"].item() == 0
    
    assert dataset[1]["binary_label"].item() == 0
    assert dataset[1]["category_label"].item() == -1 # Masked for non-hate
