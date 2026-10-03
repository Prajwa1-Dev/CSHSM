class DatasetAnalyzer:
    def __init__(self, text_field: str = "text", label_field: str = "label", category_field: str = "category"):
        self.text_field = text_field
        self.label_field = label_field
        self.category_field = category_field

    def analyze(self, data: list):
        if not data:
            return {"error": "Dataset is empty or unavailable. Analysis pending real HASTIKA dataset."}

        total_records = len(data)
        missing_text = sum(1 for r in data if not r.get(self.text_field))
        
        texts = [r.get(self.text_field, "") for r in data if r.get(self.text_field)]
        unique_texts = set(texts)
        duplicates = len(texts) - len(unique_texts)
        
        labels = [r.get(self.label_field) for r in data if self.label_field in r]
        label_dist = {label: labels.count(label) for label in set(labels)}
        
        lengths = [len(t) for t in texts]
        
        return {
            "total_records": total_records,
            "missing_text_values": missing_text,
            "duplicate_comments": duplicates,
            "label_distribution": label_dist,
            "text_length": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": sum(lengths) / len(lengths) if lengths else 0
            }
        }
