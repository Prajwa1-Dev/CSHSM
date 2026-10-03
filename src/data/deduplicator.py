import json
from collections import defaultdict

class DuplicateHandler:
    def __init__(self, text_field="Comment", label_field="Label", category_field="Hate Category"):
        self.text_field = text_field
        self.label_field = label_field
        self.category_field = category_field

    def deduplicate(self, data: list):
        """
        Deduplicates data based on text. 
        - Identical texts with identical labels/categories are kept as a single record.
        - Identical texts with conflicting annotations are flagged and excluded.
        """
        groups = defaultdict(list)
        for record in data:
            text = record.get(self.text_field, "")
            groups[text].append(record)
            
        deduplicated = []
        conflicts = []
        
        duplicates_removed = 0
        conflicts_flagged = 0
        
        # Sort keys to ensure deterministic ordering
        for text in sorted(groups.keys()):
            records = groups[text]
            if len(records) == 1:
                deduplicated.append(records[0])
            else:
                # Check for conflicts
                first_label = records[0].get(self.label_field)
                first_cat = records[0].get(self.category_field)
                
                is_conflict = False
                for r in records[1:]:
                    if r.get(self.label_field) != first_label or r.get(self.category_field) != first_cat:
                        is_conflict = True
                        break
                        
                if is_conflict:
                    conflicts.extend(records)
                    conflicts_flagged += len(records)
                else:
                    deduplicated.append(records[0])
                    duplicates_removed += (len(records) - 1)
                    
        return deduplicated, conflicts, duplicates_removed, conflicts_flagged
