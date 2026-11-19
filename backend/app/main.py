"""
CSHSM Milestone 13: Flask REST API Backend.

Provides HTTP REST endpoints for the CSHSM Counter-Speech Generation system:
- GET  /health and /api/health: Liveness, resource diagnostics, and mock status.
- POST /api/process: End-to-end pipeline processing for hate speech queries.
- GET  /api/kb/stats: Knowledge base statistics, review queue count, and approval status.
- GET  /api/kb/review-queue: Candidate pairs in review queue with AI assessment recommendations.

Architecture & Invariants:
- Uses Flask 3.1.3 (already installed; zero new package dependencies).
- Integrates PipelineOrchestrator from src.pipeline.orchestrator.
- Built-in CORS headers for cross-origin browser requests from React/Vite dashboard.
- Fails closed on any unexpected error; never leaks unformatted stack traces.
- Transparently reports mock mode and pending human approval status.
"""

import os
import sys
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, Response, render_template

# Add project root to sys.path and load environment variables from root .env
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(_PROJECT_ROOT)

_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
if os.path.isfile(_ENV_PATH):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_ENV_PATH, override=False)
    except ImportError:
        pass

from src.pipeline.orchestrator import (
    PipelineOrchestrator,
    create_default_pipeline,
    PipelineResult
)


def create_app(
    orchestrator: Optional[PipelineOrchestrator] = None,
    test_config: Optional[Dict[str, Any]] = None,
    use_live_gemini: bool = False
) -> Flask:
    """
    Application factory for the CSHSM REST API backend.
    """
    app = Flask(__name__)
    app.config.setdefault("MAX_CONTENT_LENGTH", 1 * 1024 * 1024)  # 1 MB max request payload size
    if test_config:
        app.config.update(test_config)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": "Payload Too Large",
            "message": "Request payload exceeds maximum allowed size of 1 MB."
        }), 413

    # Initialize or inject pipeline orchestrator
    if orchestrator is None:
        pipeline_mode = os.getenv("CSHSM_PIPELINE_MODE", "").strip().lower()
        if pipeline_mode == "m8":
            from scripts.demonstrate_m8_pipeline import build_m8_pipeline
            is_mock_env = os.getenv("CSHSM_M8_MOCK", "").strip().lower() in ("1", "true", "yes")
            is_live_env = os.getenv("CSHSM_M8_LIVE", "").strip().lower() in ("1", "true", "yes")
            if is_mock_env and is_live_env:
                raise ValueError(
                    "Conflicting CSHSM M8 configuration: both CSHSM_M8_LIVE and CSHSM_M8_MOCK are set. "
                    "Choose exactly one mode."
                )
            # Default to completely offline mock execution unless CSHSM_M8_LIVE=1 is explicitly set
            use_mock = not is_live_env
            use_real_classifier = os.getenv("CSHSM_M8_USE_REAL_CLASSIFIER", "").strip().lower() in ("1", "true", "yes")
            pipeline = build_m8_pipeline(use_mock=use_mock, use_real_classifier=use_real_classifier)
        elif os.getenv("CSHSM_DEMO_PIPELINE", "").strip().lower() in ("1", "true", "yes"):
            from scripts.demonstrate_phase3 import build_demonstration_pipeline
            pipeline = build_demonstration_pipeline()
        else:
            enable_gemini = use_live_gemini or (
                os.getenv("CSHSM_USE_LIVE_GEMINI", "").strip().lower() in ("1", "true", "yes")
            )
            pipeline = create_default_pipeline(use_mock=True, threshold=0.80, use_live_gemini=enable_gemini)
    else:
        pipeline = orchestrator

    # -----------------------------------------------------------------
    # CORS Headers Hook
    # -----------------------------------------------------------------
    @app.after_request
    def apply_cors_headers(response: Response) -> Response:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    # -----------------------------------------------------------------
    # Research Dashboard UI Endpoint
    # -----------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def dashboard():
        return render_template("index.html")

    @app.route("/favicon.ico", methods=["GET"])
    def favicon():
        return app.send_static_file("favicon.ico")

    # -----------------------------------------------------------------
    # Health & Diagnostics Endpoints
    # -----------------------------------------------------------------
    @app.route("/health", methods=["GET"])
    @app.route("/api/health", methods=["GET"])
    def health_check():
        now_iso = datetime.now(timezone.utc).isoformat()
        return jsonify({
            "status": "ok",
            "service": "CSHSM REST API",
            "version": "1.0",
            "framework": "Flask 3.1.3",
            "mock_mode": bool(pipeline.allow_mock),
            "classifier_type": type(pipeline.classifier).__name__,
            "threshold": pipeline.threshold or pipeline.router.default_threshold,
            "timestamp": now_iso
        }), 200

    # -----------------------------------------------------------------
    # Main Pipeline Processing Endpoint
    # -----------------------------------------------------------------
    @app.route("/api/process", methods=["POST", "OPTIONS"])
    def process_query():
        if request.method == "OPTIONS":
            return Response("", status=204)

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({
                "error": "Bad Request",
                "message": "Request body must be a valid JSON object."
            }), 400

        if "text" not in payload:
            return jsonify({
                "error": "Bad Request",
                "message": "Field 'text' is required in request body."
            }), 400

        MAX_INPUT_LENGTH = 2000

        text = payload.get("text")
        if text is None or not isinstance(text, str):
            return jsonify({
                "error": "Bad Request",
                "message": "Field 'text' must be a string."
            }), 400

        if len(text) > MAX_INPUT_LENGTH:
            return jsonify({
                "error": "Bad Request",
                "message": f"Field 'text' exceeds maximum allowed length of {MAX_INPUT_LENGTH} characters."
            }), 400

        try:
            result: PipelineResult = pipeline.process(text)
            return jsonify(result.to_dict()), 200
        except Exception as exc:
            # Server-side diagnostic logging with full traceback
            app.logger.error("Unhandled exception in /api/process: %s", exc, exc_info=True)
            # Top-level fail-closed safety net returning sanitized client-facing response
            now_iso = datetime.now(timezone.utc).isoformat()
            return jsonify({
                "outcome": "SAFE_FALLBACK",
                "input_text": text,
                "response_text": None,
                "candidate_id": None,
                "source_origin": "NONE",
                "is_hate": False,
                "category": None,
                "confidence": 0.0,
                "model_id": None,
                "provider": None,
                "quality_status": "fallback",
                "review_status": "unreviewed",
                "human_audited": False,
                "safe_fallback_triggered": True,
                "fallback_reason": "INTERNAL_SERVER_ERROR: An unexpected error occurred while processing the request.",
                "fallback_response": pipeline.fallback_message,
                "component_statuses": {"api": "ERROR"},
                "provenance": {
                    "error": "InternalServerError",
                    "error_type": type(exc).__name__,
                    "timestamp": now_iso
                },
                "disclaimer": pipeline.DISCLAIMER_FALLBACK
            }), 200

    # -----------------------------------------------------------------
    # Knowledge Base Statistics Endpoint
    # -----------------------------------------------------------------
    @app.route("/api/kb/stats", methods=["GET"])
    def kb_statistics():
        filtered_path = os.path.join(_PROJECT_ROOT, "data", "counter_speech_kb", "filtered_candidates.jsonl")
        queue_path = os.path.join(_PROJECT_ROOT, "data", "counter_speech_kb", "review_queue.jsonl")
        approved_path = os.path.join(_PROJECT_ROOT, "data", "counter_speech_kb", "approved_kb.jsonl")

        filtered_count = 19436
        if os.path.isfile(filtered_path):
            try:
                with open(filtered_path, "r", encoding="utf-8") as f:
                    filtered_count = sum(1 for _ in f)
            except Exception:
                pass

        queue_count = 0
        if os.path.isfile(queue_path):
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    queue_count = sum(1 for _ in f)
            except Exception:
                pass

        approved_count = 0
        if os.path.isfile(approved_path):
            try:
                with open(approved_path, "r", encoding="utf-8") as f:
                    approved_count = sum(1 for _ in f)
            except Exception:
                pass

        categories = {
            "Religion": 16616,
            "Gender": 1267,
            "Geo-political": 945,
            "Others": 608
        }

        # Dynamically inspect active KB V2 source and metadata relative to project root
        active_kb_v2_jsonl = os.path.join(_PROJECT_ROOT, "data", "counter_speech_kb", "cshsm_kannada_english_kb_v2.jsonl")
        active_kb_v2_latest = os.path.join(_PROJECT_ROOT, "data", "counter_speech_kb", "embeddings_v2", "LATEST.json")

        active_kb_v2: Dict[str, Any] = {
            "status": "unavailable",
            "record_count": 0,
            "categories": {},
            "embedding_model": None,
            "vector_dimension": None,
            "index_backend": "FAISS IndexFlatIP",
            "scope": "development_provisional",
            "review_status": "unreviewed",
            "human_approved": False,
            "disclaimer": "Authoritative KB V2 code-mixed Kannada counter-speech candidates. Unapproved pending human review."
        }

        if os.path.isfile(active_kb_v2_jsonl):
            try:
                cat_distribution: Dict[str, int] = {}
                v2_total = 0
                with open(active_kb_v2_jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            v2_total += 1
                            item = json.loads(line_str)
                            cat = item.get("category", "Others")
                            cat_distribution[cat] = cat_distribution.get(cat, 0) + 1
                active_kb_v2["status"] = "available"
                active_kb_v2["record_count"] = v2_total
                active_kb_v2["categories"] = cat_distribution
            except Exception as exc:
                app.logger.warning("Error reading active KB V2 jsonl: %s", exc)

        if os.path.isfile(active_kb_v2_latest):
            try:
                with open(active_kb_v2_latest, "r", encoding="utf-8") as f:
                    latest_meta = json.load(f)
                    active_kb_v2["embedding_model"] = latest_meta.get("model_identifier")
                    active_kb_v2["scope"] = latest_meta.get("scope", "development_provisional")
                    shape = latest_meta.get("matrix_shape")
                    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                        active_kb_v2["vector_dimension"] = shape[1]
            except Exception as exc:
                app.logger.warning("Error reading active KB V2 LATEST.json: %s", exc)

        return jsonify({
            "total_filtered_candidates": filtered_count,
            "review_queue_count": queue_count,
            "approved_kb_count": approved_count,
            "human_approved": approved_count > 0,
            "approval_status": "approved" if approved_count > 0 else "pending_human_review",
            "categories": categories,
            "active_kb_v2": active_kb_v2,
            "disclaimer": "All source candidates and review queue items are unapproved. No automated approval.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    # -----------------------------------------------------------------
    # Review Queue & AI Assessment Endpoint
    # -----------------------------------------------------------------
    @app.route("/api/kb/review-queue", methods=["GET"])
    def get_review_queue():
        queue_path = os.path.join(_PROJECT_ROOT, "data", "counter_speech_kb", "review_queue.jsonl")
        ai_review_path = os.path.join(_PROJECT_ROOT, "reports", "m12_1_ai_quality_review.json")

        candidates = []
        if os.path.exists(queue_path):
            try:
                with open(queue_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            candidates.append(json.loads(line))
            except Exception:
                pass

        assessments = {}
        summary_counts = {}
        if os.path.exists(ai_review_path):
            try:
                with open(ai_review_path, "r", encoding="utf-8") as f:
                    review_doc = json.load(f)
                    summary_counts = review_doc.get("summary", {})
                    for item in review_doc.get("candidate_assessments", []):
                        assessments[item.get("kb_id")] = item
            except Exception:
                pass

        merged = []
        for c in candidates:
            item = dict(c)
            item["ai_assessment"] = assessments.get(c.get("kb_id"))
            merged.append(item)

        return jsonify({
            "total_candidates": len(merged),
            "summary": summary_counts,
            "candidates": merged,
            "review_method": "ai_assisted",
            "human_approved": False,
            "disclaimer": "AI-assisted quality review for engineering triage only. Non-human-approved."
        }), 200

    return app


# Default application instance for development and WSGI runners
app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
