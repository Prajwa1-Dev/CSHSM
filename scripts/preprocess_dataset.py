import json
from pathlib import Path
from src.data.loader import DatasetLoader
from src.data.cleaner import TextCleaner
from src.data.deduplicator import DuplicateHandler
from src.data.splitter import DatasetSplitter

def run_pipeline():
    print("Starting M3 Preprocessing Pipeline...")
    
    config = {
        "text_field": "Comment",
        "label_field": "Label",
        "category_field": "Hate Category"
    }
    
    report = {
        "raw_record_count": 0,
        "processed_record_count": 0,
        "duplicates_removed": 0,
        "conflicts_flagged": 0,
        "empty_records_removed": 0,
        "urls_replaced": 0,
        "mentions_replaced": 0,
        "encoding_repairs": [],
        "labels_normalized": 0,
        "categories_normalized": 0,
        "split_statistics": {}
    }
    
    # 1. Load Data
    print("Loading raw data...")
    loader = DatasetLoader("data/raw/HASTIKA Corpus.csv")
    raw_data, repairs = loader.load()
    report["raw_record_count"] = len(raw_data)
    report["encoding_repairs"] = repairs
    
    # 2. Clean Text, Normalize Labels/Categories, Mask URL/Mentions
    print("Cleaning text and normalizing schemas...")
    cleaner = TextCleaner()
    cleaned_data, clean_stats = cleaner.process_dataset(
        raw_data, 
        text_field=config["text_field"],
        label_field=config["label_field"],
        category_field=config["category_field"]
    )
    report["urls_replaced"] = clean_stats["urls_replaced"]
    report["mentions_replaced"] = clean_stats["mentions_replaced"]
    report["labels_normalized"] = clean_stats["labels_normalized"]
    report["categories_normalized"] = clean_stats["categories_normalized"]
    
    # 3. Filter Empty
    print("Filtering empty records...")
    filtered_data = []
    for r in cleaned_data:
        text = r.get(config["text_field"], "")
        label = r.get(config["label_field"], "")
        if text.strip() and label.strip():
            filtered_data.append(r)
        else:
            report["empty_records_removed"] += 1
            
    # 4. Deduplicate & Resolve Conflicts
    print("Deduplicating and handling conflicts...")
    deduper = DuplicateHandler(
        text_field=config["text_field"],
        label_field=config["label_field"],
        category_field=config["category_field"]
    )
    deduped_data, conflicts, dups_removed, confs_flagged = deduper.deduplicate(filtered_data)
    
    report["duplicates_removed"] = dups_removed
    report["conflicts_flagged"] = confs_flagged
    report["processed_record_count"] = len(deduped_data)
    
    # 5. Save Processed & Conflicts
    print("Saving processed data and conflicts...")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Sort for deterministic output
    deduped_data.sort(key=lambda x: x.get(config["text_field"], ""))
    with open("data/processed/hastika_cleaned.jsonl", "w", encoding='utf-8') as f:
        for record in deduped_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    if conflicts:
        conflicts.sort(key=lambda x: x.get(config["text_field"], ""))
        with open("data/processed/annotation_conflicts.jsonl", "w", encoding='utf-8') as f:
            for record in conflicts:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
    # 6. Stratified Split
    print("Performing stratified split...")
    splitter = DatasetSplitter(seed=42)
    splits = splitter.split(
        deduped_data, 
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        label_col=config["label_field"],
        cat_col=config["category_field"]
    )
    splitter.save_splits(splits)
    
    # Verify leakage
    train_texts = set(r[config["text_field"]] for r in splits["train"])
    val_texts = set(r[config["text_field"]] for r in splits["val"])
    test_texts = set(r[config["text_field"]] for r in splits["test"])
    
    assert len(train_texts.intersection(val_texts)) == 0, "Leakage between train and val"
    assert len(train_texts.intersection(test_texts)) == 0, "Leakage between train and test"
    assert len(val_texts.intersection(test_texts)) == 0, "Leakage between val and test"
    
    report["split_statistics"] = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"])
    }
    
    # 7. Save Report
    with open("data/processed/preprocessing_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print("M3 Preprocessing Pipeline completed successfully.")
    
if __name__ == "__main__":
    run_pipeline()
