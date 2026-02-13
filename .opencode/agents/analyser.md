---
description: Content Analyser agent — health checks, integration testing, and service management
mode: subagent
tools:
  write: false
  edit: false
  bash: true
permission:
  bash:
    "python3 tools/check_service.py*": allow
    "PYTHONPATH=. python3 -m unittest*": allow
    "cat *": allow
    "grep *": allow
---

# Content Analyser Agent

Manages the Content Analyser — a Dockerised FastAPI service for duplication detection and contradiction analysis.

## Quick Commands

| Action | Command |
|--------|---------|
| Health check | `python3 tools/check_service.py` |
| Integration tests | `python3 tools/check_service.py --test` |
| Full test suite | `python3 tools/check_service.py --all` |
| Unit tests | `PYTHONPATH=. python3 -m unittest discover tests -v` |

## Service Dependencies

- LLM endpoint (vLLM) — for topic/assertion extraction and verification
- Embedding endpoint (vLLM) — for chunk and assertion embedding
- Normalised Semantic Chunker — for document chunking
- Reranker endpoint (optional) — for contradiction candidate filtering

## Deployment

See the project README for Docker Compose setup. Service endpoints and credentials are configured via environment variables (see `.env.example`).
