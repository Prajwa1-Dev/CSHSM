import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self):
        """Loads the dataset and handles missing files/unsupported formats."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.filepath}")
        
        if self.filepath.suffix not in ['.json', '.jsonl', '.csv']:
            raise ValueError(f"Unsupported file format: {self.filepath.suffix}. Expected .json, .jsonl, or .csv")
            
        data = []
        repairs = []
        
        if self.filepath.suffix == '.csv':
            import csv
            # Diagnose encoding first
            with open(self.filepath, 'rb') as f:
                raw_bytes = f.read()
            
            decoded_text = ""
            for i, b in enumerate(raw_bytes):
                try:
                    # try decoding just this byte/sequence - actually this is slow.
                    pass
                except:
                    pass
            
            # Efficient safe decoding
            decoded_text = raw_bytes.decode('utf-8', errors='replace')
            invalid_chars_count = decoded_text.count('\ufffd')
            if invalid_chars_count > 0:
                repairs.append(f"Replaced {invalid_chars_count} invalid UTF-8 bytes with U+FFFD.")
                
            # Process as CSV
            import io
            reader = csv.DictReader(io.StringIO(decoded_text))
            for row in reader:
                data.append(dict(row))
            return data, repairs

        try:
            if self.filepath.suffix == '.jsonl':
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            elif self.filepath.suffix == '.json':
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data = content
                    else:
                        raise ValueError("JSON file must contain a list of records")
            return data, repairs
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON/JSONL file: {e}")
