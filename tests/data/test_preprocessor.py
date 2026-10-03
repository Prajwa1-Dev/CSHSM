import pytest
import unicodedata
from src.data.cleaner import TextCleaner
from src.data.deduplicator import DuplicateHandler
from src.data.splitter import DatasetSplitter

def test_kannada_semantic_preservation():
    cleaner = TextCleaner()
    # Kannada text with spaces and a conjunct.
    text = " ನಮಸ್ಕಾರ   ಬೆಂಗಳೂರು "
    cleaned = cleaner.clean_text(text)
    # After normalization and stripping, it should be semantically intact
    expected = unicodedata.normalize('NFC', "ನಮಸ್ಕಾರ ಬೆಂಗಳೂರು")
    assert cleaned == expected
    assert "ನಮಸ್ಕಾರ" in cleaned

def test_url_mention_masking():
    cleaner = TextCleaner()
    text = "Check out http://example.com and @user123!"
    cleaned = cleaner.clean_text(text)
    assert "[URL]" in cleaned
    assert "[USER]" in cleaned
    assert "http://example.com" not in cleaned
    assert "@user123" not in cleaned
    
def test_conflict_flagging():
    deduper = DuplicateHandler()
    data = [
        {"Comment": "Same text", "Label": "Hate", "Hate Category": "Gender"},
        {"Comment": "Same text", "Label": "Non-Hate", "Hate Category": ""}
    ]
    deduped, conflicts, dups_rem, conf_flag = deduper.deduplicate(data)
    assert len(deduped) == 0
    assert len(conflicts) == 2
    assert conf_flag == 2
    
def test_composite_stratification_counts():
    splitter = DatasetSplitter(seed=42)
    # Generate 10 items for a class to pass the >3 threshold
    data = [{"Label": "Hate", "Hate Category": "Gender", "Comment": str(i)} for i in range(10)]
    splits = splitter.split(data, label_col="Label", cat_col="Hate Category")
    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
