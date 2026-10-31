"""
CSHSM Milestone 12: Main Pipeline Package.

Integrates data cleaning, hate classification, dense retrieval, threshold routing,
multi-model generation fan-out, candidate evaluation & ranking, and final safety gating.
"""

from src.pipeline.classifier import (
    ClassifierOutput,
    BaseClassifier,
    MockClassifier,
    XLMROracleClassifier
)
from src.pipeline.orchestrator import (
    PipelineResult,
    PipelineOrchestrator,
    create_default_pipeline,
    create_mock_retriever
)

__all__ = [
    "ClassifierOutput",
    "BaseClassifier",
    "MockClassifier",
    "XLMROracleClassifier",
    "PipelineResult",
    "PipelineOrchestrator",
    "create_default_pipeline",
    "create_mock_retriever"
]
