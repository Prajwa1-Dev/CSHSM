"""
CSHSM Milestone 12: Pipeline Classifier Interface & Adapters.

Provides:
- ClassifierOutput: Typed container for hate classification results.
- BaseClassifier: Abstract base class for classifiers.
- MockClassifier: Deterministic, configurable classifier for testing with explicit is_mock flag.
- XLMROracleClassifier: Real Multitask XLM-R classifier adapter that fails closed
  if model dependencies (torch, transformers) or model weights are missing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import os


@dataclass
class ClassifierOutput:
    """
    Standardized classification output for pipeline input screening.
    """
    is_hate: bool
    category: Optional[str]
    confidence: float
    model_id: str
    is_mock: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseClassifier(ABC):
    """Abstract base class for hate speech classifiers."""

    @abstractmethod
    def predict(self, text: str) -> ClassifierOutput:
        """
        Classifies input text.

        Returns ClassifierOutput containing hate prediction, category, confidence,
        model ID, and simulation flag.
        """
        pass


class MockClassifier(BaseClassifier):
    """
    Deterministic mock classifier for testing and offline integration.

    Guarantees:
      - Always sets is_mock=True.
      - Never performs network or GPU operations.
      - Configurable hate verdict, category, confidence, and simulated exceptions.
    """

    def __init__(
        self,
        default_is_hate: bool = True,
        default_category: Optional[str] = "religion",
        default_confidence: float = 0.95,
        model_id: str = "mock-xlmr-multitask",
        canned_predictions: Optional[Dict[str, ClassifierOutput]] = None,
        exception_to_raise: Optional[Exception] = None
    ):
        self.default_is_hate = default_is_hate
        self.default_category = default_category
        self.default_confidence = default_confidence
        self.model_id = model_id
        self.canned_predictions = canned_predictions or {}
        self.exception_to_raise = exception_to_raise

    def predict(self, text: str) -> ClassifierOutput:
        if self.exception_to_raise is not None:
            raise self.exception_to_raise

        if not text or not text.strip():
            # Fail closed on empty/blank input: treated as non-hate or classified as invalid
            return ClassifierOutput(
                is_hate=False,
                category=None,
                confidence=1.0,
                model_id=self.model_id,
                is_mock=True,
                metadata={"reason": "empty_input"}
            )

        stripped = text.strip()
        if stripped in self.canned_predictions:
            return self.canned_predictions[stripped]

        # Case-insensitive canned predictions match
        for k, v in self.canned_predictions.items():
            if stripped.lower() == k.strip().lower():
                return v

        # Common non-hate friendly patterns for prototype demonstration
        benign_starters = (
            "have a wonderful",
            "have a great",
            "good morning",
            "good afternoon",
            "good evening",
            "blessed day",
            "peaceful day",
            "hello friend",
            "thank you"
        )
        if any(stripped.lower().startswith(b) for b in benign_starters):
            return ClassifierOutput(
                is_hate=False,
                category=None,
                confidence=0.98,
                model_id=self.model_id,
                is_mock=True,
                metadata={"reason": "heuristic_mock_benign"}
            )

        return ClassifierOutput(
            is_hate=self.default_is_hate,
            category=self.default_category if self.default_is_hate else None,
            confidence=self.default_confidence,
            model_id=self.model_id,
            is_mock=True,
            metadata={"rule": "default_mock_policy"}
        )


class XLMROracleClassifier(BaseClassifier):
    """
    Adapter for trained Multitask XLM-R classifier.

    Fail-closed policy:
      If PyTorch or transformers are missing from the runtime environment, or
      if checkpoint weights cannot be loaded, this classifier raises a RuntimeError
      rather than silently classifying inputs as non-hate.
    """

    CATEGORIES = ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"]
    DEFAULT_CHECKPOINT = "models/xlmr_best.pt"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "xlm-roberta-base",
        device: Optional[str] = None
    ):
        if model_path is None:
            if os.path.isfile(self.DEFAULT_CHECKPOINT):
                self.model_path = self.DEFAULT_CHECKPOINT
            else:
                proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                candidate = os.path.join(proj_root, self.DEFAULT_CHECKPOINT)
                self.model_path = candidate if os.path.isfile(candidate) else self.DEFAULT_CHECKPOINT
        else:
            self.model_path = model_path

        self.model_name = model_name

        if device is None or device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None

        # Check environment requirements
        self._check_and_initialize()

    def _check_and_initialize(self) -> None:
        """Verifies environment dependencies and loads model if available."""
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            # PyTorch or transformers missing
            raise RuntimeError(
                f"Cannot initialize XLMROracleClassifier: Required library is missing ({exc}). "
                "The pipeline will fail closed rather than silently classifying inputs as non-hate."
            ) from exc

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"XLM-R weights checkpoint not found at: '{self.model_path}'. "
                "Cannot classify without authoritative model weights."
            )

        # If we reach here, attempt to load model
        try:
            from src.models.multitask_xlmr import MultitaskXLMR
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = MultitaskXLMR(model_name=self.model_name, num_categories=len(self.CATEGORIES))
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self._model.load_state_dict(state_dict, strict=True)
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load XLM-R model from '{self.model_path}': {exc}"
            ) from exc

    def predict(self, text: str) -> ClassifierOutput:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "XLMROracleClassifier is not loaded. Cannot run inference."
            )

        if not text or not text.strip():
            return ClassifierOutput(
                is_hate=False,
                category=None,
                confidence=1.0,
                model_id=f"xlmr-{self.model_name}",
                is_mock=False,
                metadata={"reason": "empty_input"}
            )

        import torch
        import torch.nn.functional as F

        encoded = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            binary_logits, category_logits = self._model(input_ids, attention_mask)
            binary_probs = F.softmax(binary_logits, dim=-1)[0]
            cat_probs = F.softmax(category_logits, dim=-1)[0]

        hate_prob = float(binary_probs[1].item())
        is_hate = hate_prob >= 0.5

        top_cat_idx = int(torch.argmax(cat_probs).item())
        cat_conf = float(cat_probs[top_cat_idx].item())
        category = self.CATEGORIES[top_cat_idx] if (is_hate and top_cat_idx < len(self.CATEGORIES)) else None

        return ClassifierOutput(
            is_hate=is_hate,
            category=category,
            confidence=round(hate_prob if is_hate else (1.0 - hate_prob), 4),
            model_id=f"xlmr-{self.model_name}",
            is_mock=False,
            metadata={
                "binary_probabilities": [round(float(p.item()), 4) for p in binary_probs],
                "category_confidence": round(cat_conf, 4) if is_hate else None
            }
        )
