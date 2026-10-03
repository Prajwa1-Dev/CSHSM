class DatasetValidator:
    def __init__(self, text_field: str = "text", label_field: str = "label", category_field: str = "category"):
        self.text_field = text_field
        self.label_field = label_field
        self.category_field = category_field

    def validate_schema(self, data: list):
        """Validates that all required fields are present in the dataset."""
        if not data:
            raise ValueError("Dataset is empty.")
            
        errors = []
        for idx, record in enumerate(data):
            if self.text_field not in record:
                errors.append(f"Record {idx}: Missing text field '{self.text_field}'")
            if self.label_field not in record:
                errors.append(f"Record {idx}: Missing label field '{self.label_field}'")
            # Category might be optional if non-hate, but we log if we expect it in hate examples.
            # We don't enforce category_field globally unless specified.
        
        if errors:
            raise ValueError(f"Schema validation failed with {len(errors)} errors. First 5: {errors[:5]}")
        return True
