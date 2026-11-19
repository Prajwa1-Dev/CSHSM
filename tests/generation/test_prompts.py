"""
Unit Tests for CSHSM Milestone G2: Centralized Kanglish Prompt Template.
"""

import json
import urllib.request
import pytest

from src.generation.prompts import (
    CANONICAL_CATEGORIES,
    CATEGORY_CONTEXT,
    normalize_category,
    get_category_guidance,
    build_system_prompt,
    build_user_prompt,
    build_chat_messages,
    KanglishPromptBuilder
)
from src.generation.adapters import CloudflareWorkersAIAdapter, GroqAdapter


# =====================================================================
# 1. Category Normalization & Guidance Tests
# =====================================================================

def test_canonical_categories_defined():
    expected = ("Gender", "Political", "Religion", "Geo-political", "Violence", "Others")
    assert CANONICAL_CATEGORIES == expected
    for cat in expected:
        assert cat in CATEGORY_CONTEXT
        assert len(CATEGORY_CONTEXT[cat]) > 20


def test_normalize_category_canonical():
    for cat in CANONICAL_CATEGORIES:
        assert normalize_category(cat) == cat


def test_normalize_category_lowercase_and_whitespace():
    assert normalize_category("gender") == "Gender"
    assert normalize_category("  POLITICAL  ") == "Political"
    assert normalize_category("religion") == "Religion"
    assert normalize_category("geo-political") == "Geo-political"
    assert normalize_category("geopolitical") == "Geo-political"
    assert normalize_category("violence") == "Violence"
    assert normalize_category("others") == "Others"


def test_normalize_category_aliases():
    assert normalize_category("women") == "Gender"
    assert normalize_category("lgbtq") == "Gender"
    assert normalize_category("politics") == "Political"
    assert normalize_category("religious") == "Religion"
    assert normalize_category("regional") == "Geo-political"
    assert normalize_category("migrants") == "Geo-political"
    assert normalize_category("violent") == "Violence"
    assert normalize_category("threats") == "Violence"
    assert normalize_category("general") == "Others"


def test_normalize_category_unknown_or_empty_falls_back_to_others():
    assert normalize_category("") == "Others"
    assert normalize_category(None) == "Others"
    assert normalize_category("completely_unknown_category") == "Others"
    assert normalize_category(12345) == "Others"  # type: ignore


def test_category_guidance_all_categories_distinct():
    guidances = set()
    for cat in CANONICAL_CATEGORIES:
        g = get_category_guidance(cat)
        assert isinstance(g, str)
        assert len(g) > 20
        guidances.add(g)
    assert len(guidances) == len(CANONICAL_CATEGORIES)


# =====================================================================
# 2. Language & Tone Constraints in System Prompt
# =====================================================================

def test_system_prompt_kanglish_requirements():
    sys_prompt = build_system_prompt("Gender")

    # Kanglish / Romanized Kannada specification
    assert "Romanized Kannada" in sys_prompt or "Kanglish" in sys_prompt
    # Latin script only rule
    assert "Kannada script" in sys_prompt
    # 70-80% Kannada, 20-30% English vocabulary rule
    assert "70–80%" in sys_prompt or "70-80%" in sys_prompt
    assert "20–30%" in sys_prompt or "20-30%" in sys_prompt
    # 2 to 4 sentences constraint
    assert "2 to 4" in sys_prompt or "2-4" in sys_prompt
    # Tone: empathetic, de-escalating, non-aggressive
    assert "empathetic" in sys_prompt.lower()
    assert "respectful" in sys_prompt.lower()
    assert "de-escalate" in sys_prompt.lower() or "de-escalating" in sys_prompt.lower()
    # No attacks/threats
    assert "without attacking" in sys_prompt.lower() or "do not attack" in sys_prompt.lower()


def test_system_prompt_category_injection():
    for cat in CANONICAL_CATEGORIES:
        prompt = build_system_prompt(cat)
        assert f"Topic Category: {cat}" in prompt
        assert CATEGORY_CONTEXT[cat] in prompt


# =====================================================================
# 3. User Prompt Construction
# =====================================================================

def test_user_prompt_contains_comment_and_category():
    comment = "Hudugir ge driving baralla, road alli accident guarantee."
    user_prompt = build_user_prompt(hateful_comment=comment, category="Gender")

    assert "Category: Gender" in user_prompt
    assert comment in user_prompt
    assert "Romanized Kannada-English (Kanglish)" in user_prompt
    assert "2-4 sentences" in user_prompt


def test_user_prompt_handles_whitespace_and_none():
    user_prompt = build_user_prompt("   ", category=None)
    assert "Category: Others" in user_prompt
    assert 'Hateful comment: ""' in user_prompt


# =====================================================================
# 4. Chat Messages Structure
# =====================================================================

def test_build_chat_messages_structure():
    messages = build_chat_messages(
        hateful_comment="Sample hate speech text",
        category="Religion"
    )

    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Romanized Kannada" in messages[0]["content"]
    assert "Sample hate speech text" in messages[1]["content"]
    assert "Religion" in messages[0]["content"]
    assert "Religion" in messages[1]["content"]


def test_build_chat_messages_system_override():
    custom_sys = "Custom system instruction for testing."
    messages = build_chat_messages(
        hateful_comment="Test comment",
        category="Political",
        system_instruction_override=custom_sys
    )

    assert messages[0]["content"] == custom_sys
    assert "Test comment" in messages[1]["content"]


# =====================================================================
# 5. KanglishPromptBuilder Class Interface
# =====================================================================

def test_kanglish_prompt_builder_class():
    builder = KanglishPromptBuilder()
    sys_p = builder.get_system_prompt("Violence")
    user_p = builder.get_user_prompt("Attack those people!", "Violence")
    messages = builder.build_messages("Attack those people!", "Violence")

    assert "Violence" in sys_p
    assert "Violence" in user_p
    assert len(messages) == 2
    assert messages[0]["content"] == sys_p
    assert messages[1]["content"] == user_p


# =====================================================================
# 6. Adapter Integration Verification
# =====================================================================

def test_cloudflare_adapter_uses_centralized_prompt(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "dummy-account-id")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "dummy-api-token")

    captured_payload = {}

    class MockHTTPResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({
                "success": True,
                "result": {"response": "Driving skill gender mele depend aagalla, practice maadidre safe driver aagbahudu."}
            }).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        return MockHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    adapter = CloudflareWorkersAIAdapter(model_id="@cf/google/gemma-4-26b-a4b-it")
    out = adapter.generate(
        prompt="Hudugir ge driving baralla",
        category="Gender"
    )

    assert out.status == "SUCCESS"
    assert "Driving skill" in (out.raw_text or "")

    # Verify messages in payload came from centralized prompt builder
    assert "messages" in captured_payload
    messages = captured_payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Romanized Kannada (Kanglish)" in messages[0]["content"]
    assert "Topic Category: Gender" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Category: Gender" in messages[1]["content"]
    assert "Hudugir ge driving baralla" in messages[1]["content"]


def test_groq_adapter_uses_centralized_prompt(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")

    captured_payload = {}

    class MockHTTPResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {"content": "Navu ella community jothe mutual respect inda irbeku."},
                    "finish_reason": "stop"
                }],
                "usage": {"total_tokens": 45}
            }).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        return MockHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    adapter = GroqAdapter(model_id="openai/gpt-oss-20b")
    out = adapter.generate(
        prompt="Those people from that community are criminals",
        category="Religion"
    )

    assert out.status == "SUCCESS"
    assert "mutual respect" in (out.raw_text or "")

    # Verify messages in payload came from centralized prompt builder
    assert "messages" in captured_payload
    messages = captured_payload["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Romanized Kannada (Kanglish)" in messages[0]["content"]
    assert "Topic Category: Religion" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Category: Religion" in messages[1]["content"]
    assert "Those people from that community are criminals" in messages[1]["content"]
