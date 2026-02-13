# content-analyser — GitHub Copilot Instructions

## Project Overview

content-analyser is a Dockerised FastAPI service that performs duplication detection and contradiction analysis on document sets. It composes external chunking, embedding, and LLM services to identify duplicate content and contradictory statements.

This service was extracted from the doc-sanitiser monolith's `source/duplication/` and `source/contradictions/` modules to create a reusable, independently testable component.

## Key Rules

- **Use `python3`** not `python`.
- **British spelling** — `sanitise`, `analyse`, `colour`, etc.
- **No real infrastructure in source** — never put real IPs, hostnames, LXC IDs, or API keys in committed code. All loaded from environment variables.
- **TDD approach** — write tests first when implementing new features.
- **Stdlib HTTP clients** — all HTTP client code uses `urllib.request` only. No `requests`, no `httpx`.
- **FastAPI + pydantic** — web framework. Use `pydantic.BaseModel` for request/response models.
- **Run tests with unittest** — `PYTHONPATH=. python3 -m unittest discover tests -v`.

## Project Structure

```
content-analyser/
├── app.py                    # FastAPI endpoints: /, /health, /analyse, /duplicates, /contradictions
├── source/
│   ├── duplication.py        # Topic analysis + file grouping + chunk-level comparison
│   ├── contradictions.py     # Group-based + assertion-based contradiction detection
│   ├── assertions.py         # LLM-based structured assertion extraction
│   ├── chunker_client.py     # HTTP client for Normalised Semantic Chunker
│   ├── embedding_client.py   # HTTP client for vLLM embedding endpoint
│   ├── llm_client.py         # HTTP client for vLLM chat endpoint
│   └── models.py             # Pydantic models for requests/responses
├── tests/
│   ├── __init__.py
│   ├── test_api.py           # FastAPI endpoint tests (mocked backends)
│   ├── test_duplication.py   # Duplication detection unit tests
│   └── test_contradictions.py # Contradiction detection unit tests
├── tools/
│   └── check_service.py      # Health check + integration tests
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

## Running Tests

```bash
# Unit tests (fast, no external services needed)
PYTHONPATH=. python3 -m unittest discover tests -v

# Integration tests against live service (reads endpoint from .env)
python3 tools/check_service.py --test

# Health check only
python3 tools/check_service.py

# Full integration test suite
python3 tools/check_service.py --all
```

## Upstream Services

| Service | Purpose | Required |
|---------|---------|----------|
| LLM endpoint (vLLM) | Topic extraction, assertion extraction, contradiction verification | Required |
| Embedding endpoint (vLLM) | Chunk and assertion embedding for similarity comparison | Required |
| Normalised Semantic Chunker | Split documents into semantic chunks for comparison | Required |
| Reranker endpoint (vLLM) | Pre-filter contradiction candidates | Optional |

## Relationship to doc-sanitiser

This service was extracted from `doc-sanitiser/source/duplication/` (detector.py, chunker.py) and `doc-sanitiser/source/contradictions/` (detector.py, assertions.py). The doc-sanitiser orchestrator calls this service's HTTP API instead of importing those modules directly. See `_plans/BUILD-PLAN.md` for extraction details.

## API Design

### POST /analyse

Full analysis: runs both duplication detection and contradiction detection on the submitted files. Returns combined results.

### POST /duplicates

Duplication detection only: topic extraction → file grouping → chunk-level embedding comparison.

### POST /contradictions

Contradiction detection only: requires file groups (from duplication) as input. Assertion extraction → embedding → reranker pre-filter → LLM verification.
