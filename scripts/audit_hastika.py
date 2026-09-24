import csv
import chardet
import collections
import json
import statistics
import re
from pathlib import Path

# Paths
FILE_PATH = "data/raw/HASTIKA Corpus.csv"

def get_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(100000)
    result = chardet.detect(raw_data)
    return result

def audit_dataset():
    encoding_info = get_encoding(FILE_PATH)
    file_size = Path(FILE_PATH).stat().st_size
    
    with open(FILE_PATH, "r", encoding='utf-8', errors='replace') as f:
        # Detect delimiter and structure
        sample = f.read(10000)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except:
            delimiter = ',' # fallback
            
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            header = []
            
        # Analysis structures
        total_rows = 0
        empty_rows = 0
        malformed_rows = 0
        missing_values = collections.defaultdict(int)
        
        labels = collections.Counter()
        categories = collections.Counter()
        label_x_category = collections.defaultdict(collections.Counter)
        
        comments = []
        duplicate_comments = collections.Counter()
        
        # Heuristics
        kannada_pattern = re.compile(r'[\u0C80-\u0CFF]')
        latin_pattern = re.compile(r'[A-Za-z]')
        emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]')
        
        # Stats
        kannada_count = 0
        latin_count = 0
        mixed_count = 0
        emoji_count = 0
        lengths = []
        
        mojibake_found = False
        mojibake_examples = []
        
        first_few = []
        last_few = []

        for row in reader:
            total_rows += 1
            if not row or all(not cell.strip() for cell in row):
                empty_rows += 1
                continue
                
            if len(row) != len(header):
                malformed_rows += 1
                
            # Pad row if necessary to match header for analysis
            while len(row) < len(header):
                row.append("")
                
            if total_rows <= 3:
                first_few.append(row)
            last_few = row # Will end up being the last row
            
            # Record missing values
            for i, col in enumerate(header):
                if not row[i].strip():
                    missing_values[col] += 1
                    
            # Identify columns (heuristics)
            label = row[2].strip() if len(row) > 2 else ""
            category = row[3].strip() if len(row) > 3 else ""
            comment = row[1].strip() if len(row) > 1 else ""
            
            labels[label] += 1
            if label.lower() == "hate":
                categories[category] += 1
            elif label.lower() == "non-hate" and category:
                categories[category] += 1 # unexpected, track it
                
            label_x_category[label][category] += 1
            
            duplicate_comments[comment] += 1
            comments.append(comment)
            lengths.append(len(comment))
            
            has_kannada = bool(kannada_pattern.search(comment))
            has_latin = bool(latin_pattern.search(comment))
            
            if has_kannada: kannada_count += 1
            if has_latin: latin_count += 1
            if has_kannada and has_latin: mixed_count += 1
            if emoji_pattern.search(comment): emoji_count += 1
            
            if 'ð' in comment:
                mojibake_found = True
                if len(mojibake_examples) < 5:
                    mojibake_examples.append(comment)
                    
    # Process duplicates
    exact_duplicates = sum(1 for v in duplicate_comments.values() if v > 1)
    
    res = {
        "file_info": {
            "size_bytes": file_size,
            "detected_encoding": encoding_info,
            "delimiter": delimiter,
            "columns": header,
            "total_records": total_rows
        },
        "structure": {
            "empty_rows": empty_rows,
            "malformed_rows": malformed_rows,
            "missing_values": dict(missing_values),
            "first_few": first_few,
            "last_row": last_few
        },
        "labels": dict(labels),
        "categories": dict(categories),
        "label_x_category": {k: dict(v) for k,v in label_x_category.items()},
        "text_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "std": statistics.stdev(lengths) if len(lengths) > 1 else 0
        },
        "language_stats": {
            "has_kannada": kannada_count,
            "has_latin": latin_count,
            "mixed_script": mixed_count,
            "has_emoji": emoji_count
        },
        "duplicates": {
            "unique_comments": len(duplicate_comments),
            "duplicate_comments_count": exact_duplicates,
        },
        "mojibake": {
            "found": mojibake_found,
            "examples": mojibake_examples
        }
    }
    
    with open("data/raw/audit_results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    audit_dataset()
