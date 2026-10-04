import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass
class KBEntry:
    kb_id: str
    hate_text: str
    counter_speech: str
    category: str
    target: Optional[str]
    strategy: Optional[str]
    language: str
    source: str
    source_id: str
    provenance: str
    generation_method: str
    quality_status: str
    review_status: str
    review_score: Optional[Dict[str, int]]
    human_audited: bool
    created_at: str
    version: str
    notes: str

    def to_dict(self):
        return asdict(self)

def validate_entry(entry: KBEntry) -> bool:
    if not entry.kb_id or not entry.hate_text or not entry.counter_speech or not entry.category or not entry.language or not entry.source or not entry.provenance:
        return False
    if entry.category not in ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others", "UNMAPPED"]:
        return False
    return True
