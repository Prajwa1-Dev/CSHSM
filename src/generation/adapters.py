"""
CSHSM Milestone 9: Generation Model Adapters.

Defines:
- BaseModelAdapter: Shared abstract interface for LLM counter-speech generation.
- AdapterOutput: Standardized result container for a single adapter call.
- MockModelAdapter: Deterministic mock adapter for development and testing.
- Real Provider Adapters:
    1. OpenAIAdapter (Slot 1)
    2. AnthropicAdapter (Slot 2)
    3. GoogleGeminiAdapter (Slot 3)
    4. MistralAdapter (Slot 4)
    5. CohereAdapter (Slot 5)

All real adapters:
- Read credentials strictly from environment variables (os.environ).
- Never log, print, or leak secrets.
- Fail closed with UNCONFIGURED_CREDENTIALS if keys are missing.
- Do not invent fake API outputs or simulate network calls when unconfigured.
"""

import os
import time
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict

from src.generation.prompts import build_chat_messages


# =====================================================================
# Adapter Output Model
# =====================================================================

@dataclass
class AdapterOutput:
    """
    Normalized result emitted by any model adapter call.
    """
    model_id: str
    provider: str
    is_mock: bool
    raw_text: Optional[str]
    latency_ms: float
    status: str  # "SUCCESS" | "TIMEOUT" | "ADAPTER_ERROR" | "UNCONFIGURED_CREDENTIALS" | "EMPTY_OUTPUT"
    error_category: Optional[str]  # "TIMEOUT" | "AUTH_ERROR" | "RATE_LIMIT" | "NETWORK_ERROR" | "OUTPUT_PARSING" | "UNKNOWN_ERROR" | None
    error_message: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Base Adapter Interface
# =====================================================================

class BaseModelAdapter(ABC):
    """
    Abstract interface for counter-speech generation model adapters.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique model identifier (e.g., 'gpt-4o-mini', 'claude-3-haiku', etc.)."""
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name ('openai', 'anthropic', 'google', 'mistral', 'cohere', 'mock')."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Whether this adapter is a mock/development adapter."""
        pass

    @property
    def timeout(self) -> float:
        """Per-model default timeout in seconds."""
        return 10.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        """
        Generates a counter-speech candidate.
        Must handle its own exceptions and return AdapterOutput.
        """
        pass


# =====================================================================
# Deterministic Mock Adapter (Development & Testing)
# =====================================================================

class MockModelAdapter(BaseModelAdapter):
    """
    Deterministic mock model adapter for development and testing.
    Allows simulating:
      - Normal successful generation
      - Delays / timeouts
      - Isolated exceptions
      - Malformed or empty outputs
      - Explicit error categories
    """

    def __init__(
        self,
        model_id: str = "mock-model-v1",
        provider: str = "mock",
        canned_response: Optional[str] = "We can disagree on ideas without attacking each other's dignity.",
        delay_sec: float = 0.0,
        exception_to_raise: Optional[Exception] = None,
        force_status: Optional[str] = None,
        force_error_category: Optional[str] = None,
        timeout: float = 5.0
    ):
        self._model_id = model_id
        self._provider = provider
        self.canned_response = canned_response
        self.delay_sec = delay_sec
        self.exception_to_raise = exception_to_raise
        self.force_status = force_status
        self.force_error_category = force_error_category
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def is_mock(self) -> bool:
        return True

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        # Simulate delay
        if self.delay_sec > 0:
            if self.delay_sec > eff_timeout:
                time.sleep(eff_timeout)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return AdapterOutput(
                    model_id=self._model_id,
                    provider=self._provider,
                    is_mock=True,
                    raw_text=None,
                    latency_ms=latency_ms,
                    status="TIMEOUT",
                    error_category="TIMEOUT",
                    error_message=f"Mock adapter exceeded timeout of {eff_timeout}s (simulated delay: {self.delay_sec}s)",
                    metadata={"delay_sec": self.delay_sec}
                )
            time.sleep(self.delay_sec)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Simulate exception
        if self.exception_to_raise is not None:
            return AdapterOutput(
                model_id=self._model_id,
                provider=self._provider,
                is_mock=True,
                raw_text=None,
                latency_ms=latency_ms,
                status=self.force_status or "ADAPTER_ERROR",
                error_category=self.force_error_category or "UNKNOWN_ERROR",
                error_message=str(self.exception_to_raise),
                metadata={"exception_type": type(self.exception_to_raise).__name__}
            )

        # Forced status override
        if self.force_status and self.force_status != "SUCCESS":
            return AdapterOutput(
                model_id=self._model_id,
                provider=self._provider,
                is_mock=True,
                raw_text=self.canned_response,
                latency_ms=latency_ms,
                status=self.force_status,
                error_category=self.force_error_category,
                error_message="Forced status override in mock adapter.",
                metadata={}
            )

        # Check empty output
        if self.canned_response is None or not self.canned_response.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider=self._provider,
                is_mock=True,
                raw_text=self.canned_response or "",
                latency_ms=latency_ms,
                status="EMPTY_OUTPUT",
                error_category="OUTPUT_PARSING",
                error_message="Adapter emitted empty or whitespace-only response.",
                metadata={"prompt_preview": prompt[:50]}
            )

        return AdapterOutput(
            model_id=self._model_id,
            provider=self._provider,
            is_mock=True,
            raw_text=self.canned_response,
            latency_ms=latency_ms,
            status="SUCCESS",
            error_category=None,
            error_message=None,
            metadata={"category": category}
        )


# =====================================================================
# Real Cloud Provider Adapters (Slots 1 to 5)
# =====================================================================

class OpenAIAdapter(BaseModelAdapter):
    """
    Slot 1: OpenAI API Adapter (e.g., gpt-4o-mini).
    Reads OPENAI_API_KEY from environment. Fails closed if unconfigured.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key_env_var: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 10.0
    ):
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key or not api_key.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider="openai",
                is_mock=False,
                raw_text=None,
                latency_ms=0.0,
                status="UNCONFIGURED_CREDENTIALS",
                error_category="AUTH_ERROR",
                error_message=f"Missing credentials: '{self._api_key_env_var}' is not set in environment.",
                metadata={"api_key_configured": False}
            )

        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        system_msg = (
            "You are an expert counter-speech generator. Generate a constructive, "
            "non-hateful, de-escalating counter-narrative for the provided context. "
            "Do not attack the speaker or use toxic slurs."
        )
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Category: {category or 'General'}\nHate Speech Context: {prompt}"}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 250)
        }

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "CSHSM-Client/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                text = data["choices"][0]["message"]["content"].strip()
                return AdapterOutput(
                    model_id=self._model_id,
                    provider="openai",
                    is_mock=False,
                    raw_text=text,
                    latency_ms=latency_ms,
                    status="SUCCESS" if text else "EMPTY_OUTPUT",
                    error_category=None if text else "OUTPUT_PARSING",
                    error_message=None if text else "Empty response content from OpenAI API.",
                    metadata={"finish_reason": data["choices"][0].get("finish_reason")}
                )
        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            category_err = "AUTH_ERROR" if he.code in (401, 403) else ("RATE_LIMIT" if he.code == 429 else "NETWORK_ERROR")
            return AdapterOutput(
                model_id=self._model_id,
                provider="openai",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category=category_err,
                error_message=f"HTTP {he.code}: {he.reason}",
                metadata={"status_code": he.code}
            )
        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_timeout = "timed out" in str(ue.reason).lower()
            return AdapterOutput(
                model_id=self._model_id,
                provider="openai",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="TIMEOUT" if is_timeout else "ADAPTER_ERROR",
                error_category="TIMEOUT" if is_timeout else "NETWORK_ERROR",
                error_message=str(ue.reason),
                metadata={}
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AdapterOutput(
                model_id=self._model_id,
                provider="openai",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category="UNKNOWN_ERROR",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__}
            )


class AnthropicAdapter(BaseModelAdapter):
    """
    Slot 2: Anthropic API Adapter (e.g., claude-3-haiku-20240307).
    Reads ANTHROPIC_API_KEY from environment. Fails closed if unconfigured.
    """

    def __init__(
        self,
        model_id: str = "claude-3-haiku-20240307",
        api_key_env_var: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 10.0
    ):
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key or not api_key.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider="anthropic",
                is_mock=False,
                raw_text=None,
                latency_ms=0.0,
                status="UNCONFIGURED_CREDENTIALS",
                error_category="AUTH_ERROR",
                error_message=f"Missing credentials: '{self._api_key_env_var}' is not set in environment.",
                metadata={"api_key_configured": False}
            )

        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        system_msg = (
            "You are an expert counter-speech generator. Generate a constructive, "
            "non-hateful, de-escalating counter-narrative for the provided context. "
            "Do not attack the speaker or use toxic slurs."
        )
        payload = {
            "model": self._model_id,
            "system": system_msg,
            "messages": [
                {"role": "user", "content": f"Category: {category or 'General'}\nHate Speech Context: {prompt}"}
            ],
            "max_tokens": kwargs.get("max_tokens", 250)
        }

        req = urllib.request.Request(
            f"{self._base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                content = data.get("content", [])
                text = content[0]["text"].strip() if content else ""
                return AdapterOutput(
                    model_id=self._model_id,
                    provider="anthropic",
                    is_mock=False,
                    raw_text=text,
                    latency_ms=latency_ms,
                    status="SUCCESS" if text else "EMPTY_OUTPUT",
                    error_category=None if text else "OUTPUT_PARSING",
                    error_message=None if text else "Empty content from Anthropic API.",
                    metadata={"stop_reason": data.get("stop_reason")}
                )
        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            category_err = "AUTH_ERROR" if he.code in (401, 403) else ("RATE_LIMIT" if he.code == 429 else "NETWORK_ERROR")
            return AdapterOutput(
                model_id=self._model_id,
                provider="anthropic",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category=category_err,
                error_message=f"HTTP {he.code}: {he.reason}",
                metadata={"status_code": he.code}
            )
        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_timeout = "timed out" in str(ue.reason).lower()
            return AdapterOutput(
                model_id=self._model_id,
                provider="anthropic",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="TIMEOUT" if is_timeout else "ADAPTER_ERROR",
                error_category="TIMEOUT" if is_timeout else "NETWORK_ERROR",
                error_message=str(ue.reason),
                metadata={}
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AdapterOutput(
                model_id=self._model_id,
                provider="anthropic",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category="UNKNOWN_ERROR",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__}
            )


class GoogleGeminiAdapter(BaseModelAdapter):
    """
    Slot 3: Google Gemini API Adapter (e.g., gemini-3.8-flash).
    Reads GOOGLE_API_KEY from environment. Fails closed if unconfigured.
    """

    def __init__(
        self,
        model_id: str = "gemini-3.8-flash",
        api_key_env_var: str = "GOOGLE_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 60.0
    ):
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "google"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key or not api_key.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider="google",
                is_mock=False,
                raw_text=None,
                latency_ms=0.0,
                status="UNCONFIGURED_CREDENTIALS",
                error_category="AUTH_ERROR",
                error_message=f"Missing credentials: '{self._api_key_env_var}' is not set in environment.",
                metadata={"api_key_configured": False}
            )

        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        system_instruction = (
            "You are an expert counter-speech generator. Generate a constructive, "
            "non-hateful, de-escalating counter-narrative for the provided context. "
            "Do not attack the speaker or use toxic slurs."
        )
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "parts": [
                        {"text": f"Category: {category or 'General'}\nHate Speech Context: {prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 250),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }

        url = f"{self._base_url}/models/{self._model_id}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                candidates = data.get("candidates", [])
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "").strip()
                return AdapterOutput(
                    model_id=self._model_id,
                    provider="google",
                    is_mock=False,
                    raw_text=text,
                    latency_ms=latency_ms,
                    status="SUCCESS" if text else "EMPTY_OUTPUT",
                    error_category=None if text else "OUTPUT_PARSING",
                    error_message=None if text else "Empty text from Gemini response.",
                    metadata={}
                )
        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            category_err = "AUTH_ERROR" if he.code in (401, 403) else ("RATE_LIMIT" if he.code == 429 else "NETWORK_ERROR")
            return AdapterOutput(
                model_id=self._model_id,
                provider="google",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category=category_err,
                error_message=f"HTTP {he.code}: {he.reason}",
                metadata={"status_code": he.code}
            )
        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_timeout = "timed out" in str(ue.reason).lower()
            return AdapterOutput(
                model_id=self._model_id,
                provider="google",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="TIMEOUT" if is_timeout else "ADAPTER_ERROR",
                error_category="TIMEOUT" if is_timeout else "NETWORK_ERROR",
                error_message=str(ue.reason),
                metadata={}
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AdapterOutput(
                model_id=self._model_id,
                provider="google",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category="UNKNOWN_ERROR",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__}
            )


class MistralAdapter(BaseModelAdapter):
    """
    Slot 4: Mistral AI API Adapter (e.g., mistral-small-latest).
    Reads MISTRAL_API_KEY from environment. Fails closed if unconfigured.
    """

    def __init__(
        self,
        model_id: str = "mistral-small-latest",
        api_key_env_var: str = "MISTRAL_API_KEY",
        base_url: str = "https://api.mistral.ai/v1",
        timeout: float = 10.0
    ):
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "mistral"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key or not api_key.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider="mistral",
                is_mock=False,
                raw_text=None,
                latency_ms=0.0,
                status="UNCONFIGURED_CREDENTIALS",
                error_category="AUTH_ERROR",
                error_message=f"Missing credentials: '{self._api_key_env_var}' is not set in environment.",
                metadata={"api_key_configured": False}
            )

        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        system_msg = (
            "You are an expert counter-speech generator. Generate a constructive, "
            "non-hateful, de-escalating counter-narrative for the provided context. "
            "Do not attack the speaker or use toxic slurs."
        )
        payload = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Category: {category or 'General'}\nHate Speech Context: {prompt}"}
            ],
            "max_tokens": kwargs.get("max_tokens", 250)
        }

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "CSHSM-Client/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                text = data["choices"][0]["message"]["content"].strip()
                return AdapterOutput(
                    model_id=self._model_id,
                    provider="mistral",
                    is_mock=False,
                    raw_text=text,
                    latency_ms=latency_ms,
                    status="SUCCESS" if text else "EMPTY_OUTPUT",
                    error_category=None if text else "OUTPUT_PARSING",
                    error_message=None if text else "Empty content from Mistral API.",
                    metadata={}
                )
        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            category_err = "AUTH_ERROR" if he.code in (401, 403) else ("RATE_LIMIT" if he.code == 429 else "NETWORK_ERROR")
            return AdapterOutput(
                model_id=self._model_id,
                provider="mistral",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category=category_err,
                error_message=f"HTTP {he.code}: {he.reason}",
                metadata={"status_code": he.code}
            )
        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_timeout = "timed out" in str(ue.reason).lower()
            return AdapterOutput(
                model_id=self._model_id,
                provider="mistral",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="TIMEOUT" if is_timeout else "ADAPTER_ERROR",
                error_category="TIMEOUT" if is_timeout else "NETWORK_ERROR",
                error_message=str(ue.reason),
                metadata={}
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AdapterOutput(
                model_id=self._model_id,
                provider="mistral",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category="UNKNOWN_ERROR",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__}
            )


class CohereAdapter(BaseModelAdapter):
    """
    Slot 5: Cohere API Adapter (e.g., command-r).
    Reads COHERE_API_KEY from environment. Fails closed if unconfigured.
    """

    def __init__(
        self,
        model_id: str = "command-r",
        api_key_env_var: str = "COHERE_API_KEY",
        base_url: str = "https://api.cohere.com/v1",
        timeout: float = 10.0
    ):
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "cohere"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key or not api_key.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider="cohere",
                is_mock=False,
                raw_text=None,
                latency_ms=0.0,
                status="UNCONFIGURED_CREDENTIALS",
                error_category="AUTH_ERROR",
                error_message=f"Missing credentials: '{self._api_key_env_var}' is not set in environment.",
                metadata={"api_key_configured": False}
            )

        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        preamble = (
            "You are an expert counter-speech generator. Generate a constructive, "
            "non-hateful, de-escalating counter-narrative for the provided context. "
            "Do not attack the speaker or use toxic slurs."
        )
        payload = {
            "model": self._model_id,
            "message": f"Category: {category or 'General'}\nHate Speech Context: {prompt}",
            "preamble": preamble,
            "max_tokens": kwargs.get("max_tokens", 250)
        }

        req = urllib.request.Request(
            f"{self._base_url}/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "CSHSM-Client/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                text = data.get("text", "").strip()
                return AdapterOutput(
                    model_id=self._model_id,
                    provider="cohere",
                    is_mock=False,
                    raw_text=text,
                    latency_ms=latency_ms,
                    status="SUCCESS" if text else "EMPTY_OUTPUT",
                    error_category=None if text else "OUTPUT_PARSING",
                    error_message=None if text else "Empty text from Cohere API.",
                    metadata={}
                )
        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            category_err = "AUTH_ERROR" if he.code in (401, 403) else ("RATE_LIMIT" if he.code == 429 else "NETWORK_ERROR")
            return AdapterOutput(
                model_id=self._model_id,
                provider="cohere",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category=category_err,
                error_message=f"HTTP {he.code}: {he.reason}",
                metadata={"status_code": he.code}
            )
        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_timeout = "timed out" in str(ue.reason).lower()
            return AdapterOutput(
                model_id=self._model_id,
                provider="cohere",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="TIMEOUT" if is_timeout else "ADAPTER_ERROR",
                error_category="TIMEOUT" if is_timeout else "NETWORK_ERROR",
                error_message=str(ue.reason),
                metadata={}
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AdapterOutput(
                model_id=self._model_id,
                provider="cohere",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category="UNKNOWN_ERROR",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__}
            )


# =====================================================================
# Cloudflare Workers AI Adapter (Slots 1 & 2)
# =====================================================================

class CloudflareWorkersAIAdapter(BaseModelAdapter):
    """
    Cloudflare Workers AI REST API Adapter.
    Supports:
      - Gemma 4 26B IT (@cf/google/gemma-4-26b-a4b-it)
      - GLM-4.7-Flash (@cf/zai-org/glm-4.7-flash)

    Reads CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN from environment.
    Fails closed if unconfigured. Never logs secrets.
    """

    def __init__(
        self,
        model_id: str = "@cf/google/gemma-4-26b-a4b-it",
        account_id_env_var: str = "CLOUDFLARE_ACCOUNT_ID",
        api_token_env_var: str = "CLOUDFLARE_API_TOKEN",
        base_url: str = "https://api.cloudflare.com/client/v4/accounts",
        timeout: float = 25.0
    ):
        self._model_id = model_id
        self._account_id_env_var = account_id_env_var
        self._api_token_env_var = api_token_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "cloudflare"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        account_id = os.getenv(self._account_id_env_var)
        api_token = os.getenv(self._api_token_env_var)

        if not account_id or not account_id.strip() or not api_token or not api_token.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider="cloudflare",
                is_mock=False,
                raw_text=None,
                latency_ms=0.0,
                status="UNCONFIGURED_CREDENTIALS",
                error_category="AUTH_ERROR",
                error_message=(
                    f"Missing credentials: '{self._account_id_env_var}' or "
                    f"'{self._api_token_env_var}' is not set in environment."
                ),
                metadata={"credentials_configured": False}
            )

        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        messages = build_chat_messages(hateful_comment=prompt, category=category)

        payload = {
            "messages": messages,
            "max_completion_tokens": kwargs.get("max_tokens", 250),
            "temperature": kwargs.get("temperature", 0.7),
            "chat_template_kwargs": {"enable_thinking": False}
        }

        url = f"{self._base_url}/{account_id}/ai/run/{self._model_id}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}",
                "User-Agent": "CSHSM-Client/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

                text = ""
                result = data.get("result", {})
                if isinstance(result, dict):
                    if "response" in result and result["response"]:
                        text = str(result["response"]).strip()
                    elif "choices" in result and result["choices"]:
                        text = result["choices"][0].get("message", {}).get("content", "").strip()
                elif isinstance(result, str):
                    text = result.strip()

                return AdapterOutput(
                    model_id=self._model_id,
                    provider="cloudflare",
                    is_mock=False,
                    raw_text=text,
                    latency_ms=latency_ms,
                    status="SUCCESS" if text else "EMPTY_OUTPUT",
                    error_category=None if text else "OUTPUT_PARSING",
                    error_message=None if text else "Empty text emitted by Cloudflare Workers AI.",
                    metadata={"success": data.get("success", True)}
                )
        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            category_err = "AUTH_ERROR" if he.code in (401, 403) else ("RATE_LIMIT" if he.code == 429 else "NETWORK_ERROR")
            return AdapterOutput(
                model_id=self._model_id,
                provider="cloudflare",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category=category_err,
                error_message=f"HTTP {he.code}: {he.reason}",
                metadata={"status_code": he.code}
            )
        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_timeout = "timed out" in str(ue.reason).lower()
            return AdapterOutput(
                model_id=self._model_id,
                provider="cloudflare",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="TIMEOUT" if is_timeout else "ADAPTER_ERROR",
                error_category="TIMEOUT" if is_timeout else "NETWORK_ERROR",
                error_message=str(ue.reason),
                metadata={}
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AdapterOutput(
                model_id=self._model_id,
                provider="cloudflare",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category="UNKNOWN_ERROR",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__}
            )


# =====================================================================
# Groq API Adapter (Slots 3, 4, 5)
# =====================================================================

class GroqAdapter(BaseModelAdapter):
    """
    Groq OpenAI-Compatible Chat Completions Adapter.
    Supports:
      - GPT-OSS-20B (openai/gpt-oss-20b)
      - Qwen 3.8 27B (qwen/qwen3.8-27b)
      - GPT-OSS-120B (openai/gpt-oss-120b)

    Reads GROQ_API_KEY from environment. Fails closed if unconfigured.
    Never logs secrets.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        api_key_env_var: str = "GROQ_API_KEY",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 25.0
    ):
        self._model_id = model_id
        self._api_key_env_var = api_key_env_var
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "groq"

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return self._timeout

    def generate(
        self,
        prompt: str,
        category: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AdapterOutput:
        api_key = os.getenv(self._api_key_env_var)
        if not api_key or not api_key.strip():
            return AdapterOutput(
                model_id=self._model_id,
                provider="groq",
                is_mock=False,
                raw_text=None,
                latency_ms=0.0,
                status="UNCONFIGURED_CREDENTIALS",
                error_category="AUTH_ERROR",
                error_message=f"Missing credentials: '{self._api_key_env_var}' is not set in environment.",
                metadata={"api_key_configured": False}
            )

        eff_timeout = timeout if timeout is not None else self._timeout
        start_time = time.perf_counter()

        messages = build_chat_messages(hateful_comment=prompt, category=category)

        payload = {
            "model": self._model_id,
            "messages": messages,
            "max_completion_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7),
            "reasoning_effort": "low"
        }

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "CSHSM-Client/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                choices = data.get("choices", [])
                text = ""
                finish_reason = None
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    finish_reason = choices[0].get("finish_reason")

                return AdapterOutput(
                    model_id=self._model_id,
                    provider="groq",
                    is_mock=False,
                    raw_text=text,
                    latency_ms=latency_ms,
                    status="SUCCESS" if text else "EMPTY_OUTPUT",
                    error_category=None if text else "OUTPUT_PARSING",
                    error_message=None if text else "Empty content from Groq API.",
                    metadata={
                        "finish_reason": finish_reason,
                        "usage": data.get("usage", {})
                    }
                )
        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            category_err = "AUTH_ERROR" if he.code in (401, 403) else ("RATE_LIMIT" if he.code == 429 else "NETWORK_ERROR")
            return AdapterOutput(
                model_id=self._model_id,
                provider="groq",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category=category_err,
                error_message=f"HTTP {he.code}: {he.reason}",
                metadata={"status_code": he.code}
            )
        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_timeout = "timed out" in str(ue.reason).lower()
            return AdapterOutput(
                model_id=self._model_id,
                provider="groq",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="TIMEOUT" if is_timeout else "ADAPTER_ERROR",
                error_category="TIMEOUT" if is_timeout else "NETWORK_ERROR",
                error_message=str(ue.reason),
                metadata={}
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return AdapterOutput(
                model_id=self._model_id,
                provider="groq",
                is_mock=False,
                raw_text=None,
                latency_ms=latency_ms,
                status="ADAPTER_ERROR",
                error_category="UNKNOWN_ERROR",
                error_message=str(exc),
                metadata={"error_type": type(exc).__name__}
            )
