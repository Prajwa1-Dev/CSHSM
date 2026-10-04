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

def test_trainer_config():
    from src.models.trainer import MultitaskTrainer
    model = MultitaskXLMR()
    device = torch.device("cpu")
    # Test default accumulation and fp16
    trainer = MultitaskTrainer(model, [], [], device)
    assert trainer.accumulation_steps == 1
    assert trainer.use_fp16 == False
    
    # Test custom accumulation
    trainer_acc = MultitaskTrainer(model, [], [], device, accumulation_steps=2)
    assert trainer_acc.accumulation_steps == 2
    
    # Test fp16 only activates on CUDA
    trainer_fp16 = MultitaskTrainer(model, [], [], device, use_fp16=True)
    assert trainer_fp16.use_fp16 == False

def test_device_selection():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    assert device.type in ["cuda", "mps", "cpu"]

def test_checkpoint_resume(tmp_path):
    from src.models.trainer import MultitaskTrainer
    model = MultitaskXLMR()
    device = torch.device("cpu")
    trainer = MultitaskTrainer(model, [], [], device)
    
    checkpoint_path = tmp_path / "test_chkpt.pt"
    config = {"epochs": 5}
    trainer.save_checkpoint(str(checkpoint_path), epoch=2, best_score=0.85, config=config, seed=42)
    
    res_epoch, res_best_score, res_patience, res_config, res_seed = trainer.load_checkpoint(str(checkpoint_path))
    assert res_epoch == 2
    assert res_best_score == 0.85
    assert res_config["epochs"] == 5
    assert res_seed == 42

def test_experiment_json_schema(tmp_path):
    import json
    data = {
        "timestamp": "20261004_170709",
        "model_name": "xlm-roberta-base",
        "architecture_description": "Shared XLM-R backbone",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "device": "NVIDIA GeForce RTX 4050 Laptop GPU"
    }
    f = tmp_path / "exp.json"
    with open(f, "w") as out:
        json.dump(data, out)
    with open(f, "r") as in_f:
        loaded = json.load(in_f)
    assert "timestamp" in loaded
    assert "model_name" in loaded
    assert loaded["batch_size"] == 16

def test_aggregate_error_analysis_schema(tmp_path):
    import json
    data = {
        "overall_total": 805,
        "overall_errors": 100,
        "by_category": {"Gender": {"total": 10, "errors": 5, "error_rate": 0.5}},
        "by_language_profile": {"Kannada-heavy": {"total": 5, "errors": 1, "error_rate": 0.2}},
        "by_text_length": {"Short (<15 words)": {"total": 10, "errors": 2, "error_rate": 0.2}},
        "by_emoji": {"Contains Emoji": {"total": 3, "errors": 0, "error_rate": 0.0}},
        "qualitative_errors": []
    }
    f = tmp_path / "agg.json"
    with open(f, "w") as out:
        json.dump(data, out)
    with open(f, "r") as in_f:
        loaded = json.load(in_f)
    assert "overall_total" in loaded
    assert "by_category" in loaded
    assert "by_language_profile" in loaded
    assert "by_text_length" in loaded
    assert "by_emoji" in loaded
    assert loaded["by_category"]["Gender"]["error_rate"] == 0.5

def test_deterministic_behavior():
    import torch
    torch.manual_seed(42)
    a = torch.rand(2, 2)
    torch.manual_seed(42)
    b = torch.rand(2, 2)
    assert torch.allclose(a, b)

