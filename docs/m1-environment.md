# Environment Setup Documentation (M1)

## Environment Overview
- **Operating System**: macOS 12.7.6
- **Python Version**: 3.13.2
- **pip Version**: 24.3.1
- **Git Version**: 2.37.1 (Apple Git-137.1)
- **Node.js Version**: v24.20.0
- **npm Version**: 11.19.0
- **Virtual Environment**: `.venv` using Python `venv` module.

## Storage
- **Available Storage at Setup**: 9.6 GiB
- **Note**: Heavy ML models (XLM-R, BGE-M3) and datasets must have their sizes explicitly evaluated before downloading in future milestones to adhere to the strict storage constraint (< 7.75 GB initially). 

## Project Structure
The repository is structured to support the eventual pipeline without over-engineering:
```text
CSHSM/
├── backend/            # FastAPI application logic
├── frontend/           # Streamlit or lightweight UI (future)
├── data/               # Git-ignored datasets directory
│   ├── raw/
│   ├── processed/
│   └── knowledge_base/
├── models/             # Git-ignored local model weights directory
├── scripts/            # Setup and utility scripts
├── tests/              # Pytest suite
├── experiments/        # Scripts for ablation studies and baselines
├── logs/               # Git-ignored runtime logs
├── docs/               # Architecture and planning documentation
├── .venv/              # Git-ignored local Python environment
├── .env.example        # Environment variables template
├── .gitignore          # Repository ignore rules
├── README.md           # Project identity and instructions
├── requirements.txt    # Project dependencies
└── ROADMAP.md          # 22-Milestone implementation plan
```

## Dependencies (M1 Minimal)
The following minimal dependencies have been installed to support the backend health check and testing framework:
- `fastapi>=0.100.0`
- `uvicorn>=0.23.0`
- `pytest>=7.4.0`
- `httpx>=0.24.1`

Heavy ML packages (Torch, Transformers, Sentence-Transformers, FAISS) are explicitly omitted in M1.

## Verification
- **Python Environment**: Activated using `source .venv/bin/activate`.
- **Backend**: Verified functioning with a lightweight FastAPI `/health` endpoint.
- **Tests**: `pytest tests/test_health.py` runs and passes successfully.
- **Git**: `.gitignore` accurately ignores `.venv`, `.env`, `__pycache__`, `data/`, and `models/`.

## Issues Encountered
None. The environment setup proceeded smoothly. The backend framework is primed for future integration.
