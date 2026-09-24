# CSHSM — Counter-Speech Generation for Hate Content on Social Media

## System Objective
CSHSM is a research-oriented hybrid AI system designed to detect hate content in social media comments and generate or retrieve safe, relevant, and constructive counter-speech responses.

## Development Principles
1. **Research before implementation.**
2. **Implement one milestone at a time.**
3. **Every milestone must produce a verifiable result.**
4. **Test components independently before integration.**
5. **Do not blindly trust model outputs.**
6. **Do not use arbitrary thresholds.**
7. **Measure before optimizing.**
8. **Keep experiments reproducible.**
9. **Track model/dataset versions.**
10. **Keep secrets out of Git.**
11. **Prefer simple architecture over unnecessary complexity.**
12. **Do not download large resources without checking storage first.**

## Hardware and Storage Constraint
The development environment has a strict constraint of approximately **7.75 GB of free storage**.
Future milestones must estimate and document the size of:
- model sizes
- dataset sizes
- FAISS index sizes
- cache sizes
- dependency sizes
- temporary disk requirements
before any download or installation occurs.

## Documentation
- [Architecture](docs/architecture.md)
- [Project Scope](docs/project-scope.md)
- [Technical Decisions](docs/technical-decisions.md)
- [Research Plan](docs/research-plan.md)
- [Full Implementation Roadmap](ROADMAP.md)

## Note on Development Tools vs System Models
The models used to build this project (e.g., Antigravity, Claude, Gemini assistants) are **development tools**. They are not components of the final CSHSM system. The final architecture operates independently using its own configured ML pipelines and API clients as defined in the technical decisions.
