# content-analyser

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Dockerised FastAPI service that performs duplication detection and contradiction analysis on document sets. It composes external chunking, embedding, and LLM services to identify duplicate content and contradictory statements across multiple files.

## ⚠️ AI-Generated Content Notice

This project was **generated with AI assistance** and should be treated accordingly:

- **Not production-ready**: Created for a specific homelab environment.
- **May contain bugs**: AI-generated code can have subtle issues.
- **Author's Python experience**: The author is not an experienced Python programmer.

### AI Tools Used

- GitHub Copilot (Claude models)
- Local vLLM instances for validation

### Licensing Note

Released under the **MIT License**. Given the AI-generated nature:
- The author makes no claims about originality
- Use at your own risk
- If you discover any copyright concerns, please open an issue

---

## How It Works

The Content Analyser receives a set of documents and performs two major analysis tasks:

### Duplication Detection

1. **Topic extraction** — LLM analyses each file, extracts topics and entities
2. **File grouping** — clusters files sharing common topics into groups
3. **Chunk-level comparison** — chunks files via an external semantic chunker, embeds chunks, computes pairwise cosine similarity to find duplicate content
4. Returns topic groups, file groups, and chunk-level duplicate matches with similarity scores

### Contradiction Detection

1. **Assertion extraction** — LLM extracts factual assertions from each file
2. **Assertion embedding** — embeds assertions for clustering
3. **Candidate pairing** — uses reranker to pre-filter likely contradictions
4. **LLM verification** — sends candidate pairs to LLM for contradiction confirmation
5. Returns structured contradiction findings with file references, excerpts, and confidence scores

## Prerequisites

- **Docker** on the host
- **LLM endpoint** — OpenAI-compatible chat API (e.g., vLLM) for topic/assertion extraction and verification
- **Embedding endpoint** — OpenAI-compatible embedding API (e.g., vLLM) for chunk and assertion embedding
- **Normalised Semantic Chunker** — chunking service accessible via HTTP
- **(Optional) Reranker endpoint** — for improved contradiction candidate filtering

## Quick Start

### 1. Build the image

```bash
docker build -t content-analyser:latest .
```

### 2. Start the service

```bash
docker compose up -d
```

### 3. Check health

```bash
curl http://localhost:8209/health
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info and version |
| `GET` | `/health` | Health check (tests LLM + embedding connectivity) |
| `POST` | `/analyse` | Full analysis: duplication + contradiction detection |
| `POST` | `/duplicates` | Duplication detection only |
| `POST` | `/contradictions` | Contradiction detection only (requires file groups) |

## Project Structure

```
content-analyser/
├── app.py                    # FastAPI endpoints
├── source/
│   ├── duplication.py        # Topic analysis + file grouping + chunk comparison
│   ├── contradictions.py     # Assertion extraction + embedding + LLM verification
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
# Unit tests (no external services needed)
PYTHONPATH=. python3 -m unittest discover tests -v

# Health check against live service (reads endpoint from .env)
python3 tools/check_service.py

# Integration tests
python3 tools/check_service.py --test

# Full test suite
python3 tools/check_service.py --all
```

## Environment Variables

All service endpoints and credentials are loaded from environment variables.
See `.env.example` for client-side configuration.
The server-side config (vLLM endpoints, embedding URL, chunker URL, API keys) is injected via `secrets.env` at deploy time — never committed to source.

## Ecosystem

This service is part of the [xarta](https://github.com/xarta) document analysis ecosystem — a set of Dockerised microservices built for a nested Proxmox homelab running local AI inference on an RTX 5090 and RTX 4000 Blackwell.

The project grew out of a practical need: AI-generated infrastructure code (Proxmox, LXC, Docker configs) is full of secrets and environment-specific details that make it unsafe to share and hard to reuse. These services clean, chunk, embed, and ingest that code into a vector database so AI agents can query it efficiently — and so sanitised versions can be published to GitHub.

All services delegate compute-heavy work (LLM chat, embeddings, reranking) to shared vLLM endpoints via OpenAI-compatible APIs, taking advantage of batched and parallel GPU operations rather than bundling models locally.

This is a work in progress, decomposing the original project that became too monolithic and difficult to develop into more manageable components.  The original project had many incomplete features that were proving difficult to implement with generative AI without impacting other features and so even when all components are migrated there will still be some development to do before the features originally envisaged are complete.

| Repository | Description |
|---|---|
| [Normalized-Semantic-Chunker](https://github.com/xarta/Normalized-Semantic-Chunker) | Embedding-based semantic text chunking with statistical token-size control. Lightweight fork — delegates embeddings to a remote vLLM endpoint. |
| [Agentic-Chunker](https://github.com/xarta/Agentic-Chunker) | LLM-driven proposition chunking — uses chat completions to semantically group content. Fork replacing Google Gemini with local vLLM. |
| [gitleaks-validator](https://github.com/xarta/gitleaks-validator) | Dockerised gitleaks wrapper — pattern-driven secret scanning and replacement via REST API. |
| [knowledge-service](https://github.com/xarta/knowledge-service) | Document ingestion into SeekDB (vector database) with RAG query interface. Composes chunking + embedding services. |
| [content-analyser](https://github.com/xarta/content-analyser) | Duplication detection and contradiction analysis across document sets. Composes chunking, embedding, and LLM services. |
| [file-service](https://github.com/xarta/file-service) | File operations REST API — read, write, archive, copy, move. Safety-constrained path access for containerised pipeline services. |
| [sanitiser-dashboard](https://github.com/xarta/sanitiser-dashboard) | Pipeline dashboard UI and logging API — receives structured events, timing, and request logs from pipeline services. |

## License

MIT — see [LICENSE](LICENSE).
