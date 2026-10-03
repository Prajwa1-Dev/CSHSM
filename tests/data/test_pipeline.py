import pytest
import os
from src.data.loader import DatasetLoader
from src.data.validator import DatasetValidator
from src.data.cleaner import TextCleaner
from src.data.analyzer import DatasetAnalyzer
from src.data.splitter import DatasetSplitter

def test_loader_missing_file():
    loader = DatasetLoader("non_existent_file.json")
    with pytest.raises(FileNotFoundError):
        loader.load()

def test_loader_unsupported_format(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("a,b,c")
    loader = DatasetLoader(str(f))
    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load()

def test_validator_missing_fields():
    validator = DatasetValidator(text_field="text", label_field="label")
    data = [{"text": "hello"}] # Missing label
    with pytest.raises(ValueError, match="Schema validation failed"):
        validator.validate_schema(data)

def test_cleaner():
    cleaner = TextCleaner()
    # Test whitespace normalization
    assert cleaner.clean_text("Hello \n world  ") == "Hello world"
    # Test unicode normalization
    assert cleaner.clean_text("e\u0301") == "é"
    # Test dataset processing
    data = [{"text": "  dirty \t text "}, {"not_text": "ignore"}]
    cleaned_data, stats = cleaner.process_dataset(data, text_field="text")
    assert cleaned_data[0]["text"] == "dirty text"

def test_analyzer_empty():
    analyzer = DatasetAnalyzer()
    res = analyzer.analyze([])
    assert "error" in res
    assert "pending real HASTIKA dataset" in res["error"]

def test_analyzer_synthetic():
    analyzer = DatasetAnalyzer()
    data = [
        {"text": "A", "label": "hate"},
        {"text": "A", "label": "hate"},
        {"text": "B", "label": "non-hate"},
        {"label": "hate"} # Missing text
    ]
    res = analyzer.analyze(data)
    assert res["total_records"] == 4
    assert res["missing_text_values"] == 1
    assert res["duplicate_comments"] == 3 - 2 # 3 texts, 2 unique = 1 duplicate
    assert res["label_distribution"]["hate"] == 3
    assert res["text_length"]["mean"] == 1.0

def test_splitter(tmp_path):
    splitter = DatasetSplitter(seed=42, output_dir=str(tmp_path))
    data = [{"id": i} for i in range(100)]
    splits = splitter.split(data, 0.8, 0.1, 0.1)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10
    
    # Test reproducibility
    splits2 = splitter.split(data, 0.8, 0.1, 0.1)
    assert splits["train"] == splits2["train"]
    
    # Test save
    splitter.save_splits(splits)
    assert (tmp_path / "train.jsonl").exists()
