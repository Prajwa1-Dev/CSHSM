"""
CSHSM Milestone 9: Generation Branch & Five-Model Fan-Out Architecture.
"""

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
    create_default_fan_out
)

__all__ = [
    "BaseModelAdapter",
    "AdapterOutput",
    "MockModelAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GoogleGeminiAdapter",
    "MistralAdapter",
    "CohereAdapter",
    "CloudflareWorkersAIAdapter",
    "GroqAdapter",
    "GenerationRequest",
    "GeneratedCandidate",
    "AdapterExecutionRecord",
    "FanOutResult",
    "GenerationFanOut",
    "create_default_fan_out"
]
