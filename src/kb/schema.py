import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class KBEntry:
    kb_id: str
    hate_text: str
    counter_speech: str
    category: str
    target: Optional[str] = None
    strategy: Optional[str] = None
    language: str = "kn-en"
    source: str = "synthetic_v2"
    source_id: str = ""
    provenance: str = "synthetic_ai_generated"
    generation_method: str = "v2_revision"
    quality_status: str = "pending_review"
    review_status: str = "unreviewed"
    review_score: Optional[Dict[str, int]] = None
    human_audited: bool = False
    created_at: str = "2026-11-02T00:00:00Z"
    version: str = "2.0"
    notes: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KBEntry":
        kb_id = str(data.get("kb_id") or data.get("id") or "")
        hate_text = str(data.get("hate_text") or data.get("hate_comment") or "")
        counter_speech = str(data.get("counter_speech") or "")
        category = str(data.get("category") or "Others")
        target = data.get("target")
        strategy = data.get("strategy")
        language = str(data.get("language") or "kn-en")
        source = str(data.get("source") or data.get("source_split") or "synthetic_v2")
        source_id = str(data.get("source_id") or data.get("id") or "")
        provenance = str(data.get("provenance") or "synthetic_ai_generated")
        generation_method = str(data.get("generation_method") or "v2_revision")
        quality_status = str(data.get("quality_status") or "pending_review")
        review_status = str(data.get("review_status") or "unreviewed")
        review_score = data.get("review_score")
        human_audited = bool(data.get("human_audited", False))
        created_at = str(data.get("created_at") or "2026-11-02T00:00:00Z")
        version = str(data.get("version") or "2.0")

        notes = str(data.get("notes") or "")
        if not notes:
            meta_parts = []
            if "english_translation" in data and data["english_translation"]:
                meta_parts.append(f"translation: {data['english_translation']}")
            if "script_type" in data and data["script_type"]:
                meta_parts.append(f"script: {data['script_type']}")
            notes = " | ".join(meta_parts)

        return cls(
            kb_id=kb_id,
            hate_text=hate_text,
            counter_speech=counter_speech,
            category=category,
            target=target,
            strategy=strategy,
            language=language,
            source=source,
            source_id=source_id,
            provenance=provenance,
            generation_method=generation_method,
            quality_status=quality_status,
            review_status=review_status,
            review_score=review_score,
            human_audited=human_audited,
            created_at=created_at,
            version=version,
            notes=notes,
        )

def validate_entry(entry: KBEntry) -> bool:
    if not entry.kb_id or not entry.hate_text or not entry.counter_speech or not entry.category or not entry.language or not entry.source or not entry.provenance:
        return False
    if entry.category not in ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others", "UNMAPPED"]:
        return False
    return True
