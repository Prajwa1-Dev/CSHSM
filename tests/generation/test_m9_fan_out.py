import os
import sys
import time
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.routing.router import RoutingDecision
from src.generation.adapters import (
    BaseModelAdapter,
    AdapterOutput,
    MockModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleGeminiAdapter,
    MistralAdapter,
    CohereAdapter,
    CloudflareWorkersAIAdapter,
    GroqAdapter
)
from src.generation.fan_out import (
    GenerationRequest,
    GeneratedCandidate,
    AdapterExecutionRecord,
    FanOutResult,
    GenerationFanOut,
    create_default_fan_out,
    is_cloudflare_configured,
    is_groq_configured
)


# =====================================================================
# 1. Adapter Slot Count Tests (1, 3, and 5 Configured Slots)
# =====================================================================

def test_one_configured_adapter_success():
    adapter = MockModelAdapter(
        model_id="mock-model-1",
        canned_response="We should treat each other with mutual respect and empathy."
    )
    fan_out = GenerationFanOut(adapters=[adapter])

    req = GenerationRequest(query_text="Sample hate speech text", category="Religion")
    result = fan_out.execute(req)

    assert result.total_slots == 1
    assert result.enabled_slots == 1
    assert result.successful_slots == 1
    assert result.failed_slots == 0
    assert result.status == "SUCCESS"
    assert result.safe_fallback_triggered is False
    assert result.fallback_response is None
    assert len(result.safe_candidates) == 1
    assert len(result.execution_records) == 1

    cand = result.safe_candidates[0]
    assert cand.model_id == "mock-model-1"
    assert cand.text == "We should treat each other with mutual respect and empathy."
    assert cand.quality_status == "generated_candidate"
    assert cand.review_status == "unreviewed"
    assert cand.human_audited is False


def test_three_configured_adapters_success():
    adapters = [
        MockModelAdapter(model_id="mock-slot-1", canned_response="Counter response 1: Empathy builds bridges."),
        MockModelAdapter(model_id="mock-slot-2", canned_response="Counter response 2: Stereotypes mislead our understanding."),
        MockModelAdapter(model_id="mock-slot-3", canned_response="Counter response 3: Respect is essential for any community.")
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    req = GenerationRequest(query_text="Sample hate speech context", category="Gender")
    result = fan_out.execute(req)

    assert result.total_slots == 3
    assert result.successful_slots == 3
    assert result.failed_slots == 0
    assert result.status == "SUCCESS"
    assert len(result.safe_candidates) == 3
    assert len(result.execution_records) == 3

    # Check distinct candidate IDs
    cand_ids = [c.candidate_id for c in result.safe_candidates]
    assert len(set(cand_ids)) == 3


def test_five_configured_adapters_success():
    fan_out = create_default_fan_out(use_mock=True)

    assert len(fan_out.adapters) == 5
    req = GenerationRequest(query_text="Hateful generalization about a minority group", category="Religion")
    result = fan_out.execute(req)

    assert result.total_slots == 5
    assert result.successful_slots == 5
    assert result.failed_slots == 0
    assert result.status == "SUCCESS"
    assert len(result.safe_candidates) == 5
    assert len(result.execution_records) == 5

    for idx, rec in enumerate(result.execution_records):
        assert rec.slot_index == idx
        assert rec.status == "SUCCESS"
        assert rec.is_mock is True
        assert rec.candidate_id is not None


# =====================================================================
# 2. Per-Model Timeouts, Exceptions, and Isolation
# =====================================================================

def test_adapter_per_model_timeout_isolation():
    # 3 fast adapters, 1 slow adapter that exceeds timeout
    adapters = [
        MockModelAdapter(model_id="fast-1", canned_response="Fast constructive reply 1.", delay_sec=0.0),
        MockModelAdapter(model_id="fast-2", canned_response="Fast constructive reply 2.", delay_sec=0.0),
        MockModelAdapter(model_id="slow-timeout", canned_response="Slow reply.", delay_sec=1.0, timeout=0.1),
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    start = time.perf_counter()
    result = fan_out.execute(GenerationRequest(query_text="Test prompt"))
    elapsed = time.perf_counter() - start

    # Elapsed should be well under 1.5s
    assert elapsed < 1.0

    assert result.total_slots == 3
    assert result.successful_slots == 2
    assert result.failed_slots == 1
    assert result.status == "PARTIAL_SUCCESS"
    assert len(result.safe_candidates) == 2

    # Check records
    slow_rec = [r for r in result.execution_records if r.model_id == "slow-timeout"][0]
    assert slow_rec.status == "TIMEOUT"
    assert slow_rec.error_category == "TIMEOUT"
    assert "exceeded timeout" in slow_rec.error_message


def test_adapter_isolated_exception():
    adapters = [
        MockModelAdapter(model_id="healthy-1", canned_response="Healthy response 1."),
        MockModelAdapter(model_id="faulty", exception_to_raise=RuntimeError("Connection reset by peer")),
        MockModelAdapter(model_id="healthy-2", canned_response="Healthy response 2.")
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    result = fan_out.execute(GenerationRequest(query_text="Test prompt"))

    assert result.total_slots == 3
    assert result.successful_slots == 2
    assert result.failed_slots == 1
    assert result.status == "PARTIAL_SUCCESS"
    assert len(result.safe_candidates) == 2

    faulty_rec = [r for r in result.execution_records if r.model_id == "faulty"][0]
    assert faulty_rec.status == "ADAPTER_ERROR"
    assert faulty_rec.error_category == "UNKNOWN_ERROR"
    assert "Connection reset by peer" in faulty_rec.error_message


# =====================================================================
# 3. Basic Structural Validation: Empty, Malformed, Length, Duplicates
# =====================================================================

def test_malformed_empty_and_whitespace_outputs():
    adapters = [
        MockModelAdapter(model_id="empty-string", canned_response=""),
        MockModelAdapter(model_id="whitespace", canned_response="     \n\t  "),
        MockModelAdapter(model_id="valid", canned_response="Constructive valid reply to the hate text.")
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    result = fan_out.execute(GenerationRequest(query_text="Test hate comment"))

    # Empty outputs are captured as EMPTY_OUTPUT in adapter or EMPTY_TEXT in validation
    assert len(result.valid_candidates) == 1
    assert result.valid_candidates[0].model_id == "valid"


def test_excessively_long_output():
    # Max chars set to 200
    long_text = "This is a sentence about respectful communication. " * 10
    assert len(long_text) > 200

    adapters = [
        MockModelAdapter(model_id="long-model", canned_response=long_text),
        MockModelAdapter(model_id="normal-model", canned_response="Normal length response that is constructive.")
    ]
    fan_out = GenerationFanOut(adapters=adapters, max_chars=200)

    result = fan_out.execute(GenerationRequest(query_text="Test hate comment"))

    assert len(result.raw_candidates) == 2
    assert len(result.valid_candidates) == 1
    assert result.valid_candidates[0].model_id == "normal-model"

    rejected = [c for c in result.rejected_candidates if c.model_id == "long-model"][0]
    assert rejected.structural_validation["is_excessively_long"] is True
    assert rejected.structural_validation["validation_reason"] == "EXCESSIVE_LENGTH"


def test_insufficient_length_output():
    adapters = [
        MockModelAdapter(model_id="too-short", canned_response="Stop."),
        MockModelAdapter(model_id="normal", canned_response="Constructive counter narrative that addresses the topic.")
    ]
    fan_out = GenerationFanOut(adapters=adapters, min_chars=10)

    result = fan_out.execute(GenerationRequest(query_text="Test hate comment"))

    assert len(result.valid_candidates) == 1
    assert result.valid_candidates[0].model_id == "normal"

    rejected = [c for c in result.rejected_candidates if c.model_id == "too-short"][0]
    assert rejected.structural_validation["is_valid"] is False
    assert rejected.structural_validation["validation_reason"] == "INSUFFICIENT_LENGTH"


def test_duplicate_candidate_detection():
    # Two models produce verbatim identical text
    identical_text = "Empathy and understanding are key to resolving cultural tensions."
    adapters = [
        MockModelAdapter(model_id="model-alpha", canned_response=identical_text),
        MockModelAdapter(model_id="model-beta", canned_response=identical_text),
        MockModelAdapter(model_id="model-gamma", canned_response="A completely different, distinct counter narrative.")
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    result = fan_out.execute(GenerationRequest(query_text="Test hate comment"))

    assert len(result.raw_candidates) == 3
    assert len(result.valid_candidates) == 2  # model-alpha (first) and model-gamma
    assert len(result.safe_candidates) == 2

    # Second identical candidate was marked as duplicate
    dup_cand = [c for c in result.rejected_candidates if c.model_id == "model-beta"][0]
    assert dup_cand.structural_validation["is_duplicate"] is True
    assert dup_cand.structural_validation["validation_reason"] == "DUPLICATE_CANDIDATE"


# =====================================================================
# 4. Safety Filtering Integration
# =====================================================================

def test_safety_filtering_rejection():
    adapters = [
        MockModelAdapter(
            model_id="safe-model",
            canned_response="Diverse opinions make us stronger; attacking people personally doesn't solve any issue."
        ),
        MockModelAdapter(
            model_id="hate-reinforcement",
            canned_response="I agree you are right that those people are terrible."  # triggers hate reinforcement rule
        ),
        MockModelAdapter(
            model_id="violent-escalation",
            canned_response="We should kill all who disagree with our viewpoints."  # triggers threat/violence rule
        )
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    result = fan_out.execute(GenerationRequest(query_text="Hate comment about politics"))

    # All 3 were structurally valid, but only 1 passed safety filtering
    assert len(result.raw_candidates) == 3
    assert len(result.valid_candidates) == 3
    assert len(result.safe_candidates) == 1
    assert result.safe_candidates[0].model_id == "safe-model"

    # Verify rejection reasons
    reinforce_cand = [c for c in result.rejected_candidates if c.model_id == "hate-reinforcement"][0]
    assert reinforce_cand.safety_validation["is_safe"] is False
    assert reinforce_cand.safety_validation["rejection_reason"] == "hate reinforcement"

    violence_cand = [c for c in result.rejected_candidates if c.model_id == "violent-escalation"][0]
    assert violence_cand.safety_validation["is_safe"] is False
    assert violence_cand.safety_validation["rejection_reason"] == "threats or violent escalation"


# =====================================================================
# 5. Outcome States: Partial Success, Total Failure, All Filtered
# =====================================================================

def test_partial_success_scenario():
    adapters = [
        MockModelAdapter(model_id="slot-1", canned_response="Good counter response 1."),
        MockModelAdapter(model_id="slot-2", canned_response="Good counter response 2."),
        MockModelAdapter(model_id="slot-3", exception_to_raise=ValueError("API error")),
        MockModelAdapter(model_id="slot-4", delay_sec=1.0, timeout=0.1),
        MockModelAdapter(model_id="slot-5", canned_response="You are an idiot shut up.")  # Counter-hate rule violation
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    result = fan_out.execute(GenerationRequest(query_text="Test prompt"))

    assert result.total_slots == 5
    assert result.successful_slots == 3
    assert result.failed_slots == 2
    assert result.status == "PARTIAL_SUCCESS"
    assert result.safe_fallback_triggered is False
    assert len(result.safe_candidates) == 2
    assert len(result.rejected_candidates) == 1  # slot-5 rejected by safety


def test_total_failure_scenario():
    # All 3 adapters fail
    adapters = [
        MockModelAdapter(model_id="fail-1", exception_to_raise=RuntimeError("Fail 1")),
        MockModelAdapter(model_id="fail-2", delay_sec=1.0, timeout=0.1),
        MockModelAdapter(model_id="fail-3", exception_to_raise=ConnectionError("Fail 3"))
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    result = fan_out.execute(GenerationRequest(query_text="Test prompt"))

    assert result.status == "TOTAL_FAILURE"
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert len(result.safe_candidates) == 0


def test_all_filtered_scenario():
    # Adapters succeed, but all outputs violate safety rules
    adapters = [
        MockModelAdapter(model_id="toxic-1", canned_response="You are an idiot shut up."),
        MockModelAdapter(model_id="toxic-2", canned_response="I agree you are right."),
    ]
    fan_out = GenerationFanOut(adapters=adapters)

    result = fan_out.execute(GenerationRequest(query_text="Test prompt"))

    assert result.status == "ALL_FILTERED"
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == "I cannot respond to this respectfully."
    assert len(result.safe_candidates) == 0
    assert len(result.rejected_candidates) == 2


# =====================================================================
# 6. Ingesting M8 RoutingDecision Contract
# =====================================================================

def test_m8_routing_decision_consumption():
    m8_decision = RoutingDecision(
        query="Refugees are destroying our neighborhoods",
        action="FALLBACK_GENERATION",
        threshold=0.85,
        similarity_score=0.62,
        retrieved_candidate=None,
        fallback_reason="LOW_SIMILARITY",
        fallback_payload={
            "query": "Refugees are destroying our neighborhoods",
            "top_score": 0.62,
            "threshold": 0.85,
            "top_candidate": {
                "kb_id": "KB-00123",
                "counter_speech": "Communities thrive when we welcome newcomers with compassion.",
                "category": "Geo-political"
            }
        },
        provenance={"run_id": "M6-RUN-001", "router": "FallbackRouter"},
        disclaimer="SAFE FALLBACK — Top candidate similarity below threshold; defaulting to generation."
    )

    fan_out = create_default_fan_out(use_mock=True)
    result = fan_out.execute(m8_decision)

    assert result.request.query_text == "Refugees are destroying our neighborhoods"
    assert result.request.category == "Geo-political"
    assert result.request.reference_context == "Communities thrive when we welcome newcomers with compassion."
    assert result.status == "SUCCESS"
    assert len(result.safe_candidates) == 5


# =====================================================================
# 7. Candidate Review and Quality Invariants
# =====================================================================

def test_candidate_review_and_quality_invariants():
    fan_out = create_default_fan_out(use_mock=True)
    result = fan_out.execute(GenerationRequest(query_text="Hate comment context"))

    assert len(result.safe_candidates) > 0
    for cand in result.safe_candidates:
        assert cand.quality_status == "generated_candidate"
        assert cand.review_status == "unreviewed"
        assert cand.human_audited is False
        assert "UNREVIEWED GENERATED CANDIDATE" in cand.disclaimer
        assert cand.candidate_id.startswith("GEN-")
        assert cand.latency_ms >= 0.0


# =====================================================================
# 8. Real Provider Adapters: Unconfigured Credentials Fail-Closed
# =====================================================================

def test_real_providers_unconfigured_credentials_fail_closed(monkeypatch):
    # Ensure all API key env vars are unset
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    real_adapters = [
        OpenAIAdapter(model_id="gpt-4o-mini"),
        AnthropicAdapter(model_id="claude-3-haiku"),
        GoogleGeminiAdapter(model_id="gemini-3.8-flash"),
        MistralAdapter(model_id="mistral-small"),
        CohereAdapter(model_id="command-r")
    ]

    for adapter in real_adapters:
        assert adapter.is_mock is False
        out = adapter.generate(prompt="Test prompt")
        assert out.status == "UNCONFIGURED_CREDENTIALS"
        assert out.error_category == "AUTH_ERROR"
        assert out.raw_text is None
        assert "Missing credentials" in out.error_message


# =====================================================================
# 9. Serialization & Configuration Validations
# =====================================================================

def test_serialization_to_dict():
    fan_out = create_default_fan_out(use_mock=True)
    result = fan_out.execute(GenerationRequest(query_text="Sample hate speech text"))

    d = result.to_dict()
    assert d["total_slots"] == 5
    assert d["status"] == "SUCCESS"
    assert len(d["safe_candidates"]) == 5
    assert len(d["execution_records"]) == 5

    # Check JSON serializability
    json_str = json.dumps(d)
    assert "GEN-" in json_str


def test_fan_out_initialization_validations():
    with pytest.raises(ValueError, match="requires at least 1"):
        GenerationFanOut(adapters=[])

    with pytest.raises(ValueError, match="at most 5 adapter slots"):
        too_many = [MockModelAdapter(model_id=f"m-{i}") for i in range(6)]
        GenerationFanOut(adapters=too_many)


# =====================================================================
# 10. Hybrid Gemini-Live Fan-Out Tests
# =====================================================================

def test_default_fan_out_has_five_mocks():
    fan_out = create_default_fan_out()
    assert len(fan_out.adapters) == 5
    for adapter in fan_out.adapters:
        assert adapter.is_mock is True
        assert adapter.provider == "mock"


def test_existing_all_live_mode_unchanged():
    fan_out = create_default_fan_out(use_mock=False, use_legacy_models=True)
    assert len(fan_out.adapters) == 5
    expected_providers = ["openai", "anthropic", "google", "mistral", "cohere"]
    for idx, adapter in enumerate(fan_out.adapters):
        assert adapter.is_mock is False
        assert adapter.provider == expected_providers[idx]


def test_hybrid_fan_out_one_live_gemini_four_mocks():
    fan_out = create_default_fan_out(use_live_gemini=True)
    assert len(fan_out.adapters) == 5

    # Slot 1: Mock OpenAI
    assert fan_out.adapters[0].is_mock is True
    assert fan_out.adapters[0].provider == "mock"
    assert fan_out.adapters[0].model_id == "mock-openai-gpt4o-mini"

    # Slot 2: Mock Anthropic
    assert fan_out.adapters[1].is_mock is True
    assert fan_out.adapters[1].provider == "mock"
    assert fan_out.adapters[1].model_id == "mock-anthropic-claude3-haiku"

    # Slot 3: Live GoogleGeminiAdapter (real API)
    assert fan_out.adapters[2].is_mock is False
    assert fan_out.adapters[2].provider == "google"
    assert fan_out.adapters[2].model_id == "gemini-3.8-flash"

    # Slot 4: Mock Mistral
    assert fan_out.adapters[3].is_mock is True
    assert fan_out.adapters[3].provider == "mock"
    assert fan_out.adapters[3].model_id == "mock-mistral-small"

    # Slot 5: Mock Cohere
    assert fan_out.adapters[4].is_mock is True
    assert fan_out.adapters[4].provider == "mock"
    assert fan_out.adapters[4].model_id == "mock-cohere-command-r"

    # Exact counts: exactly 1 live, 4 mocks
    live_count = sum(1 for a in fan_out.adapters if not a.is_mock)
    mock_count = sum(1 for a in fan_out.adapters if a.is_mock)
    assert live_count == 1
    assert mock_count == 4


def test_hybrid_fan_out_missing_gemini_credentials_fails_closed(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    fan_out = create_default_fan_out(use_live_gemini=True)
    req = GenerationRequest(query_text="Sample hate context", category="Religion")
    result = fan_out.execute(req)

    # 4 mocks succeed, 1 live fails closed due to missing credentials
    assert result.total_slots == 5
    assert result.successful_slots == 4
    assert result.failed_slots == 1
    assert result.status == "PARTIAL_SUCCESS"
    assert len(result.safe_candidates) == 4

    # Slot 3 record must report UNCONFIGURED_CREDENTIALS / AUTH_ERROR
    gemini_rec = result.execution_records[2]
    assert gemini_rec.slot_index == 2
    assert gemini_rec.model_id == "gemini-3.8-flash"
    assert gemini_rec.provider == "google"
    assert gemini_rec.is_mock is False
    assert gemini_rec.status == "UNCONFIGURED_CREDENTIALS"
    assert gemini_rec.error_category == "AUTH_ERROR"
    assert "Missing credentials" in gemini_rec.error_message


# =====================================================================
# 11. Finalized 5-Generator Roster & Timeout Tests
# =====================================================================

def test_finalized_five_generator_mock_slots():
    fan_out = create_default_fan_out(use_mock=True)
    assert len(fan_out.adapters) == 5
    expected_models = [
        "mock-gemma4-26b-it",
        "mock-glm-4.7-flash",
        "mock-gpt-oss-20b",
        "mock-qwen-3.8-27b",
        "mock-gpt-oss-120b"
    ]
    for idx, expected in enumerate(expected_models):
        assert fan_out.adapters[idx].model_id == expected
        assert fan_out.adapters[idx].is_mock is True
        assert fan_out.adapters[idx].provider == "mock"


def test_finalized_five_generator_live_slots():
    fan_out = create_default_fan_out(use_mock=False)
    assert len(fan_out.adapters) == 5
    assert fan_out.adapters[0].provider == "cloudflare"
    assert fan_out.adapters[0].model_id == "@cf/google/gemma-4-26b-a4b-it"
    assert fan_out.adapters[1].provider == "cloudflare"
    assert fan_out.adapters[1].model_id == "@cf/zai-org/glm-4.7-flash"
    assert fan_out.adapters[2].provider == "groq"
    assert fan_out.adapters[2].model_id == "openai/gpt-oss-20b"
    assert fan_out.adapters[3].provider == "groq"
    assert fan_out.adapters[3].model_id == "qwen/qwen3.8-27b"
    assert fan_out.adapters[4].provider == "groq"
    assert fan_out.adapters[4].model_id == "openai/gpt-oss-120b"


def test_cloudflare_adapter_unconfigured_credentials_fails_closed(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    adapter = CloudflareWorkersAIAdapter(model_id="@cf/google/gemma-4-26b-a4b-it")
    out = adapter.generate(prompt="Sample hate text")
    assert out.status == "UNCONFIGURED_CREDENTIALS"
    assert out.error_category == "AUTH_ERROR"
    assert out.raw_text is None
    assert "Missing credentials" in out.error_message


def test_groq_adapter_unconfigured_credentials_fails_closed(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    adapter = GroqAdapter(model_id="openai/gpt-oss-20b")
    out = adapter.generate(prompt="Sample hate text")
    assert out.status == "UNCONFIGURED_CREDENTIALS"
    assert out.error_category == "AUTH_ERROR"
    assert out.raw_text is None
    assert "Missing credentials" in out.error_message


def test_fan_out_hard_timeout_shutdown_isolation():
    # Verify ThreadPoolExecutor shutdown does not hang waiting on slow/unresponsive threads
    slow_adapter = MockModelAdapter(
        model_id="mock-slow-slot",
        delay_sec=10.0,
        timeout=10.0
    )
    fast_adapter = MockModelAdapter(
        model_id="mock-fast-slot",
        canned_response="Fast counter-speech response.",
        delay_sec=0.0
    )
    fan_out = GenerationFanOut(adapters=[slow_adapter, fast_adapter], overall_timeout=0.4)
    t0 = time.perf_counter()
    res = fan_out.execute(GenerationRequest(query_text="Sample query"))
    elapsed = time.perf_counter() - t0

    # Must terminate promptly without waiting for the 10.0s thread
    assert elapsed < 2.0, f"Shutdown hung! Elapsed: {elapsed:.2f}s"
    assert res.total_slots == 2
    assert res.failed_slots >= 1
    slow_records = [r for r in res.execution_records if r.model_id == "mock-slow-slot"]
    assert len(slow_records) == 1
    assert slow_records[0].status == "TIMEOUT"


# =====================================================================
# Milestone G4: Resilient Per-Provider Live Execution Tests
# =====================================================================

def test_is_cloudflare_configured(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    assert is_cloudflare_configured() is False

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "dummy_token")
    assert is_cloudflare_configured() is False

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "dummy_account")
    assert is_cloudflare_configured() is True


def test_is_groq_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert is_groq_configured() is False

    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")
    assert is_groq_configured() is True


def test_fan_out_all_mock_mode():
    fan_out = create_default_fan_out(use_mock=True)
    assert len(fan_out.adapters) == 5
    for a in fan_out.adapters:
        assert isinstance(a, MockModelAdapter)
        assert a.is_mock is True
        assert a.provider == "mock"

    expected_models = [
        "mock-gemma4-26b-it",
        "mock-glm-4.7-flash",
        "mock-gpt-oss-20b",
        "mock-qwen-3.8-27b",
        "mock-gpt-oss-120b"
    ]
    assert [a.model_id for a in fan_out.adapters] == expected_models


def test_fan_out_all_live_configured(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "dummy_token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "dummy_account")
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")

    fan_out = create_default_fan_out(use_mock=False)
    assert len(fan_out.adapters) == 5
    for a in fan_out.adapters:
        assert a.is_mock is False

    assert isinstance(fan_out.adapters[0], CloudflareWorkersAIAdapter)
    assert fan_out.adapters[0].model_id == "@cf/google/gemma-4-26b-a4b-it"

    assert isinstance(fan_out.adapters[1], CloudflareWorkersAIAdapter)
    assert fan_out.adapters[1].model_id == "@cf/zai-org/glm-4.7-flash"

    assert isinstance(fan_out.adapters[2], GroqAdapter)
    assert fan_out.adapters[2].model_id == "openai/gpt-oss-20b"

    assert isinstance(fan_out.adapters[3], GroqAdapter)
    assert fan_out.adapters[3].model_id == "qwen/qwen3.8-27b"

    assert isinstance(fan_out.adapters[4], GroqAdapter)
    assert fan_out.adapters[4].model_id == "openai/gpt-oss-120b"


def test_fan_out_partial_live_groq_only(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")

    fan_out = create_default_fan_out(use_mock=False, allow_mock_fallback=True)
    assert len(fan_out.adapters) == 5

    # Cloudflare slots fell back to mock because credentials were missing
    assert isinstance(fan_out.adapters[0], MockModelAdapter)
    assert fan_out.adapters[0].is_mock is True
    assert fan_out.adapters[0].model_id == "mock-gemma4-26b-it"

    assert isinstance(fan_out.adapters[1], MockModelAdapter)
    assert fan_out.adapters[1].is_mock is True
    assert fan_out.adapters[1].model_id == "mock-glm-4.7-flash"

    # Groq slots are live
    assert isinstance(fan_out.adapters[2], GroqAdapter)
    assert fan_out.adapters[2].is_mock is False
    assert fan_out.adapters[2].model_id == "openai/gpt-oss-20b"

    assert isinstance(fan_out.adapters[3], GroqAdapter)
    assert fan_out.adapters[3].is_mock is False
    assert fan_out.adapters[3].model_id == "qwen/qwen3.8-27b"

    assert isinstance(fan_out.adapters[4], GroqAdapter)
    assert fan_out.adapters[4].is_mock is False
    assert fan_out.adapters[4].model_id == "openai/gpt-oss-120b"


def test_fan_out_partial_live_cloudflare_only(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "dummy_token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "dummy_account")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    fan_out = create_default_fan_out(use_mock=False, allow_mock_fallback=True)
    assert len(fan_out.adapters) == 5

    # Cloudflare slots are live
    assert isinstance(fan_out.adapters[0], CloudflareWorkersAIAdapter)
    assert fan_out.adapters[0].is_mock is False
    assert fan_out.adapters[0].model_id == "@cf/google/gemma-4-26b-a4b-it"

    assert isinstance(fan_out.adapters[1], CloudflareWorkersAIAdapter)
    assert fan_out.adapters[1].is_mock is False
    assert fan_out.adapters[1].model_id == "@cf/zai-org/glm-4.7-flash"

    # Groq slots fell back to mock
    for idx in (2, 3, 4):
        assert isinstance(fan_out.adapters[idx], MockModelAdapter)
        assert fan_out.adapters[idx].is_mock is True


def test_fan_out_unconfigured_fails_closed_without_mock_fallback(monkeypatch):
    # Missing credentials with allow_mock_fallback=False must NOT silently use mock
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    fan_out = create_default_fan_out(use_mock=False, allow_mock_fallback=False)
    for a in fan_out.adapters:
        assert a.is_mock is False

    req = GenerationRequest(query_text="Sample hate speech text", category="Religion")
    result = fan_out.execute(req)

    assert result.total_slots == 5
    assert result.successful_slots == 0
    assert result.failed_slots == 5
    assert result.status == "TOTAL_FAILURE"
    assert result.safe_fallback_triggered is True
    assert result.fallback_response == fan_out.fallback_message

    for rec in result.execution_records:
        assert rec.is_mock is False
        assert rec.status == "UNCONFIGURED_CREDENTIALS"
        assert rec.error_category == "AUTH_ERROR"


def test_fan_out_explicit_provider_toggles(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "dummy_token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "dummy_account")
    monkeypatch.setenv("GROQ_API_KEY", "dummy_groq_key")

    # Explicit toggle: Cloudflare live, Groq mock
    fo1 = create_default_fan_out(use_live_cloudflare=True, use_live_groq=False)
    assert fo1.adapters[0].is_mock is False
    assert fo1.adapters[1].is_mock is False
    assert fo1.adapters[2].is_mock is True
    assert fo1.adapters[3].is_mock is True
    assert fo1.adapters[4].is_mock is True

    # Explicit toggle: Cloudflare mock, Groq live
    fo2 = create_default_fan_out(use_live_cloudflare=False, use_live_groq=True)
    assert fo2.adapters[0].is_mock is True
    assert fo2.adapters[1].is_mock is True
    assert fo2.adapters[2].is_mock is False
    assert fo2.adapters[3].is_mock is False
    assert fo2.adapters[4].is_mock is False


def test_fan_out_resilient_partial_provider_failure():
    # Slot 0: TimeoutError
    # Slot 1: Runtime error (e.g. 500 error)
    # Slots 2, 3, 4: Successful responses
    adapters = [
        MockModelAdapter(
            model_id="mock-failing-1",
            provider="cloudflare",
            exception_to_raise=TimeoutError("Connection timed out after 25s")
        ),
        MockModelAdapter(
            model_id="mock-failing-2",
            provider="cloudflare",
            exception_to_raise=RuntimeError("Provider HTTP 500 internal error")
        ),
        MockModelAdapter(
            model_id="mock-working-3",
            provider="groq",
            canned_response="Constructive counter-speech 1: Empathy reduces hostility."
        ),
        MockModelAdapter(
            model_id="mock-working-4",
            provider="groq",
            canned_response="Constructive counter-speech 2: Facts and respect bridge divides."
        ),
        MockModelAdapter(
            model_id="mock-working-5",
            provider="groq",
            canned_response="Constructive counter-speech 3: Mutual human dignity is paramount."
        )
    ]
    fan_out = GenerationFanOut(adapters=adapters, overall_timeout=70.0)
    req = GenerationRequest(query_text="Hate speech context for resiliency testing", category="Gender")
    res = fan_out.execute(req)

    assert res.total_slots == 5
    assert res.successful_slots == 3
    assert res.failed_slots == 2
    assert res.status == "PARTIAL_SUCCESS"
    assert res.safe_fallback_triggered is False
    assert len(res.safe_candidates) == 3
    assert len(res.execution_records) == 5

    rec0 = res.execution_records[0]
    assert rec0.status == "TIMEOUT" or rec0.status == "ADAPTER_ERROR"
    assert rec0.candidate_id is None

    rec1 = res.execution_records[1]
    assert rec1.status == "ADAPTER_ERROR"
    assert rec1.candidate_id is None

    for idx in (2, 3, 4):
        rec = res.execution_records[idx]
        assert rec.status == "SUCCESS"
        assert rec.candidate_id is not None


def test_fan_out_immutable_provenance_and_truthful_mock_flag():
    adapters = [
        MockModelAdapter(model_id="mock-slot-0", provider="mock", canned_response="Response 1 for testing."),
        MockModelAdapter(model_id="mock-slot-1", provider="mock", canned_response="Response 2 for testing.")
    ]
    fan_out = GenerationFanOut(adapters=adapters)
    res = fan_out.execute(GenerationRequest(query_text="Test query", category="Political"))

    # Assert provenance records exactly match adapter properties
    prov_adapters = res.provenance["adapters"]
    assert len(prov_adapters) == 2
    for idx, p in enumerate(prov_adapters):
        assert p["slot"] == idx
        assert p["is_mock"] is True
        assert p["provider"] == "mock"
        assert p["model_id"] == f"mock-slot-{idx}"

    for cand in res.safe_candidates:
        assert cand.is_mock is True
