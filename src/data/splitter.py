import random
import json
from pathlib import Path
from collections import defaultdict

class DatasetSplitter:
    def __init__(self, seed: int = 42, output_dir: str = "data/splits"):
        self.seed = seed
        self.output_dir = Path(output_dir)

    def split(self, data: list, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, label_col="Label", cat_col="Hate Category"):
        if not data:
            raise ValueError("Empty dataset cannot be split.")
            
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Split ratios must sum to 1.0")

        # Create composite keys and check counts
        groups = defaultdict(list)
        for record in data:
            # Deterministic ordering for reproducibility: sort by ID or Text before grouping
            groups[f"{record.get(label_col, '')}_{record.get(cat_col, '')}"].append(record)
            
        # Verify sufficient samples
        for k, v in groups.items():
            if len(v) < 3: # Need at least 3 for train, val, test
                # If a class has < 3 samples, we can't stratify cleanly.
                # In real scenario, we might merge or raise error. 
                # Given HASTIKA minimum class is 232, this should easily pass.
                raise ValueError(f"Composite class {k} has too few samples ({len(v)}) for an 80/10/10 split.")

        random.seed(self.seed)
        
        splits = {"train": [], "val": [], "test": []}
        
        for k in sorted(groups.keys()):
            class_records = groups[k]
            # Ensure deterministic order inside each class before shuffling
            class_records.sort(key=lambda r: r.get('Comment', ''))
            random.shuffle(class_records)
            
            total = len(class_records)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            
            splits["train"].extend(class_records[:train_end])
            splits["val"].extend(class_records[train_end:val_end])
            splits["test"].extend(class_records[val_end:])
            
        # Final shuffle of the splits to mix classes
        for s in splits:
            random.shuffle(splits[s])
            
        return splits

    def save_splits(self, splits: dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, split_data in splits.items():
            filepath = self.output_dir / f"{name}.jsonl"
            with open(filepath, 'w', encoding='utf-8') as f:
                for record in split_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return True
