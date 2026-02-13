"""
Content Analyser — FastAPI application.

Provides HTTP endpoints for duplication detection and contradiction
analysis across documentation files.

Endpoints:
  GET   /               — service info
  GET   /health         — health check (LLM, embedding, chunker, reranker)
  POST  /analyse        — full analysis (duplication + contradictions)
  POST  /duplicates     — duplication detection only
  POST  /contradictions — contradiction detection only

Environment variables:
  VLLM_BASE_URL       — vLLM LLM endpoint (required)
  VLLM_API_KEY        — LLM Bearer token (required)
  EMBEDDING_BASE_URL  — vLLM embedding endpoint (required)
  EMBEDDING_API_KEY   — Embedding Bearer token (required)
  CHUNKER_URL         — Normalised Semantic Chunker endpoint (required)
  RERANKER_BASE_URL   — Reranker endpoint (optional)
  RERANKER_API_KEY    — Reranker Bearer token (optional)
  LOG_LEVEL           — Logging level (default: INFO)
  MAX_WORKERS         — Default thread pool size (default: 3)
"""

import logging
import os
import sys
import time
from typing import Dict, Set

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from source.chunker_client import ChunkerClient, ChunkerConfig
from source.contradictions import ContradictionDetector
from source.duplication import DuplicationDetector
from source.embedding_client import EmbeddingClient, EmbeddingConfig
from source.llm_client import LLMClient, LLMConfig
from source.models import (
    AnalyseRequest,
    AnalyseResponse,
    ContradictionInfo,
    ContradictionResponse,
    ContradictionSummary,
    ContradictionsRequest,
    DuplicateMatchInfo,
    DuplicatesRequest,
    DuplicationResponse,
    DuplicationSummary,
    FileGroupInfo,
    HealthResponse,
    ServiceStatus,
    TopicInfo,
)

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

# Initialise FastAPI
app = FastAPI(
    title="Content Analyser",
    description="Duplication detection and contradiction analysis service",
    version="1.0.0",
)

# Global clients (initialised on startup)
llm_client: LLMClient = None  # type: ignore[assignment]
embedding_client: EmbeddingClient = None  # type: ignore[assignment]
chunker_client: ChunkerClient = None  # type: ignore[assignment]


@app.on_event("startup")
async def startup_event():
    """Initialise clients on startup."""
    global llm_client, embedding_client, chunker_client

    logger.info("Initialising content-analyser...")

    # Required: LLM client
    try:
        llm_config = LLMConfig.from_env()
        llm_client = LLMClient(llm_config)
        logger.info("LLM client configured: %s", llm_config.base_url)
    except Exception as exc:
        logger.error("LLM configuration failed: %s", exc)
        sys.exit(1)

    # Required: Embedding client
    try:
        embedding_config = EmbeddingConfig.from_env()
        embedding_client = EmbeddingClient(embedding_config)
        logger.info(
            "Embedding client configured: %s",
            embedding_config.embedding_base_url,
        )
    except Exception as exc:
        logger.error("Embedding configuration failed: %s", exc)
        sys.exit(1)

    # Required: Chunker client
    try:
        chunker_config = ChunkerConfig.from_env()
        chunker_client = ChunkerClient(chunker_config)
        logger.info("Chunker client configured: %s", chunker_config.base_url)
    except Exception as exc:
        logger.error("Chunker configuration failed: %s", exc)
        sys.exit(1)

    logger.info("Content-analyser ready (max_workers=%d)", MAX_WORKERS)


# ===================================================================
# Endpoints
# ===================================================================


@app.get("/")
async def root():
    """Service information."""
    return {
        "service": "content-analyser",
        "version": "1.0.0",
        "status": "running",
        "description": "Duplication detection and contradiction analysis service",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — test upstream service connectivity."""
    services: Dict[str, ServiceStatus] = {}

    # LLM
    try:
        start = time.monotonic()
        llm_client.test_connection()
        latency = (time.monotonic() - start) * 1000
        services["llm"] = ServiceStatus(status="up", latency_ms=round(latency, 1))
    except Exception as exc:
        services["llm"] = ServiceStatus(status="down", error=str(exc))

    # Embedding
    try:
        start = time.monotonic()
        embedding_client.test_connection()
        latency = (time.monotonic() - start) * 1000
        services["embedding"] = ServiceStatus(
            status="up", latency_ms=round(latency, 1),
        )
    except Exception as exc:
        services["embedding"] = ServiceStatus(status="down", error=str(exc))

    # Chunker
    try:
        start = time.monotonic()
        chunker_client.test_connection()
        latency = (time.monotonic() - start) * 1000
        services["chunker"] = ServiceStatus(
            status="up", latency_ms=round(latency, 1),
        )
    except Exception as exc:
        services["chunker"] = ServiceStatus(status="down", error=str(exc))

    # Reranker (optional)
    if embedding_client.config.reranker_base_url:
        try:
            start = time.monotonic()
            embedding_client.test_reranker()
            latency = (time.monotonic() - start) * 1000
            services["reranker"] = ServiceStatus(
                status="up", latency_ms=round(latency, 1),
            )
        except Exception as exc:
            services["reranker"] = ServiceStatus(
                status="down", error=str(exc),
            )

    overall = "healthy" if all(
        s.status == "up" for k, s in services.items() if k != "reranker"
    ) else "degraded"

    return HealthResponse(status=overall, services=services)


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse(request: AnalyseRequest):
    """Full analysis — duplication then contradiction detection."""
    if len(request.files) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 files are required for analysis",
        )

    start = time.monotonic()
    config = request.config or _default_config()
    workers = config.max_workers

    # Step 1: Duplication detection
    dup_result = _run_duplication(request.files, config)

    # Step 2: Contradiction detection using file groups from step 1
    groups_for_contra = [
        {
            "shared_topics": g.shared_topics,
            "files": g.files,
        }
        for g in dup_result.groups
    ]
    duplicate_pairs = _extract_duplicate_pairs(dup_result)
    contra_result = _run_contradictions(
        request.files, groups_for_contra, duplicate_pairs, workers,
    )

    duration = time.monotonic() - start

    return AnalyseResponse(
        duplication=dup_result,
        contradictions=contra_result,
        duration_seconds=round(duration, 2),
    )


@app.post("/duplicates", response_model=DuplicationResponse)
async def duplicates(request: DuplicatesRequest):
    """Duplication detection only."""
    if len(request.files) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 files are required for duplication detection",
        )

    config = request.config or _default_config()
    return _run_duplication(request.files, config)


@app.post("/contradictions", response_model=ContradictionResponse)
async def contradictions(request: ContradictionsRequest):
    """Contradiction detection only (requires file groups)."""
    if len(request.files) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 files are required for contradiction detection",
        )

    config = request.config or _default_config()
    workers = config.max_workers
    groups = [
        {
            "shared_topics": g.shared_topics,
            "files": g.files,
        }
        for g in request.groups
    ]

    return _run_contradictions(request.files, groups, set(), workers)


# ===================================================================
# Internal helpers
# ===================================================================

def _default_config():
    """Return default AnalysisConfig."""
    from source.models import AnalysisConfig
    return AnalysisConfig(max_workers=MAX_WORKERS)


def _run_duplication(
    files: Dict[str, str],
    config,
) -> DuplicationResponse:
    """Run the duplication detection pipeline and convert to response model."""
    detector = DuplicationDetector(
        llm_client=llm_client,
        chunker_client=chunker_client,
        embedding_client=embedding_client,
        max_workers=config.max_workers,
        threshold_high=config.similarity_threshold_high,
        threshold_medium=config.similarity_threshold_medium,
    )

    result = detector.analyse(files)

    # Convert to response models
    topics = [
        TopicInfo(
            file=t.file_path,
            topics=t.topics,
            entities=t.entities,
        )
        for t in result.topics
    ]

    groups = [
        FileGroupInfo(
            group_id=g.group_id,
            shared_topics=g.shared_topics,
            files=g.files,
        )
        for g in result.groups
    ]

    matches = [
        DuplicateMatchInfo(
            chunk_a={
                "file": m.chunk_a_file,
                "content": m.chunk_a_content,
            },
            chunk_b={
                "file": m.chunk_b_file,
                "content": m.chunk_b_content,
            },
            similarity=round(m.similarity, 4),
            severity=m.severity,
        )
        for m in result.matches
    ]

    summary = DuplicationSummary(
        files_analysed=result.files_analysed,
        chunks_compared=result.chunks_compared,
        duplicate_matches=len(result.matches),
        duplicate_groups=len(result.groups),
        chunks_skipped=result.chunks_skipped,
    )

    return DuplicationResponse(
        topics=topics,
        groups=groups,
        matches=matches,
        summary=summary,
    )


def _run_contradictions(
    files: Dict[str, str],
    groups: list,
    duplicate_pairs: Set[frozenset],
    workers: int,
) -> ContradictionResponse:
    """Run contradiction detection and convert to response model."""
    detector = ContradictionDetector(
        llm_client=llm_client,
        embedding_client=embedding_client,
        max_workers=workers,
    )

    result = detector.analyse_with_assertions(
        file_contents=files,
        file_groups=groups,
        duplicate_file_pairs=duplicate_pairs if duplicate_pairs else None,
    )

    findings = [
        ContradictionInfo(
            topic=f.topic,
            file_a=f.file1,
            excerpt_a=f.excerpt1,
            file_b=f.file2,
            excerpt_b=f.excerpt2,
            confidence=round(f.confidence, 2),
            severity=f.severity,
            explanation=f.explanation,
            type=f.type,
        )
        for f in result.findings
    ]

    summary = ContradictionSummary(
        groups_analysed=result.groups_analysed,
        assertions_extracted=result.assertions_extracted,
        contradictions_found=len(result.findings),
    )

    return ContradictionResponse(findings=findings, summary=summary)


def _extract_duplicate_pairs(dup_result: DuplicationResponse) -> Set[frozenset]:
    """Extract high-similarity file pairs from duplication results."""
    pairs: Set[frozenset] = set()
    for match in dup_result.matches:
        if match.severity == "high":
            file_a = match.chunk_a.get("file", "")
            file_b = match.chunk_b.get("file", "")
            if file_a and file_b:
                pairs.add(frozenset([file_a, file_b]))
    return pairs
