import re
import unicodedata

class TextCleaner:
    def __init__(self, config=None):
        self.config = config or {}
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')

    def clean_text(self, text: str) -> str:
        """
        Conservative text cleaning preserving semantics and emojis.
        - Whitespace normalization
        - Unicode normalization (NFC)
        - Mask URLs and mentions
        """
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode to compose characters (NFC for Indic scripts)
        text = unicodedata.normalize('NFC', text)
        
        # Mask URLs and mentions
        text = self.url_pattern.sub('[URL]', text)
        text = self.mention_pattern.sub('[USER]', text)
        
        # Replace newlines and tabs with spaces
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def normalize_label(self, label: str) -> str:
        if not label: return ""
        label = label.strip().capitalize()
        if label in ["Hate", "Non-hate"]:
            return "Hate" if label == "Hate" else "Non-Hate"
        return label

    def normalize_category(self, category: str, label: str) -> str:
        if label == "Non-Hate":
            return ""
        if not category: return ""
        category = category.strip().capitalize()
        if category.lower() == "geo-political":
            return "Geo-political"
        return category

    def process_dataset(self, data: list, text_field: str = "text", label_field: str = "label", category_field: str = "category") -> list:
        cleaned_data = []
        stats = {"urls_replaced": 0, "mentions_replaced": 0, "labels_normalized": 0, "categories_normalized": 0}
        
        for record in data:
            cleaned_record = record.copy()
            
            orig_text = cleaned_record.get(text_field, "")
            if orig_text and isinstance(orig_text, str):
                if self.url_pattern.search(orig_text): stats["urls_replaced"] += 1
                if self.mention_pattern.search(orig_text): stats["mentions_replaced"] += 1
                cleaned_record[text_field] = self.clean_text(orig_text)
                
            orig_label = cleaned_record.get(label_field, "")
            cleaned_label = self.normalize_label(orig_label)
            if orig_label != cleaned_label: stats["labels_normalized"] += 1
            cleaned_record[label_field] = cleaned_label
            
            orig_cat = cleaned_record.get(category_field, "")
            cleaned_cat = self.normalize_category(orig_cat, cleaned_label)
            if orig_cat != cleaned_cat: stats["categories_normalized"] += 1
            cleaned_record[category_field] = cleaned_cat
                
            cleaned_data.append(cleaned_record)
            
        return cleaned_data, stats
