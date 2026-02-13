"""
Pydantic models for content-analyser API.

Defines request and response schemas for all endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ===================================================================
# Shared Config
# ===================================================================

class AnalysisConfig(BaseModel):
    """Optional analysis tuning parameters."""
    similarity_threshold_high: float = Field(0.95, description="High-similarity threshold")
    similarity_threshold_medium: float = Field(0.85, description="Medium-similarity threshold")
    max_workers: int = Field(3, description="Thread pool size for parallel LLM calls")
    chunker_max_tokens: Optional[int] = Field(None, description="Max tokens per chunk (chunker param)")


# ===================================================================
# Duplication Models
# ===================================================================

class TopicInfo(BaseModel):
    """Topic analysis result for a single file."""
    file: str
    topics: List[str]
    entities: List[str] = Field(default_factory=list)


class FileGroupInfo(BaseModel):
    """A group of files sharing topics."""
    group_id: int
    shared_topics: List[str]
    files: List[str]


class DuplicateMatchInfo(BaseModel):
    """A single duplicate match between two chunks."""
    chunk_a: Dict[str, Any]
    chunk_b: Dict[str, Any]
    similarity: float
    severity: str


class DuplicationSummary(BaseModel):
    """Summary statistics for duplication analysis."""
    files_analysed: int = 0
    chunks_compared: int = 0
    duplicate_matches: int = 0
    duplicate_groups: int = 0
    chunks_skipped: int = 0


class DuplicationResponse(BaseModel):
    """Duplication analysis portion of the response."""
    topics: List[TopicInfo] = Field(default_factory=list)
    groups: List[FileGroupInfo] = Field(default_factory=list)
    matches: List[DuplicateMatchInfo] = Field(default_factory=list)
    summary: DuplicationSummary = Field(default_factory=DuplicationSummary)


# ===================================================================
# Contradiction Models
# ===================================================================

class ContradictionInfo(BaseModel):
    """A single contradiction finding."""
    topic: str
    file_a: str
    excerpt_a: str
    file_b: str
    excerpt_b: str
    confidence: float = 0.0
    severity: str = "medium"
    explanation: str = ""
    type: str = "value"


class ContradictionSummary(BaseModel):
    """Summary statistics for contradiction analysis."""
    groups_analysed: int = 0
    assertions_extracted: int = 0
    contradictions_found: int = 0


class ContradictionResponse(BaseModel):
    """Contradiction analysis portion of the response."""
    findings: List[ContradictionInfo] = Field(default_factory=list)
    summary: ContradictionSummary = Field(default_factory=ContradictionSummary)


# ===================================================================
# Request Models
# ===================================================================

class AnalyseRequest(BaseModel):
    """Request for full analysis (duplication + contradictions)."""
    files: Dict[str, str] = Field(..., description="Map of relative path → content")
    config: Optional[AnalysisConfig] = Field(None, description="Analysis tuning parameters")


class DuplicatesRequest(BaseModel):
    """Request for duplication detection only."""
    files: Dict[str, str] = Field(..., description="Map of relative path → content")
    config: Optional[AnalysisConfig] = Field(None, description="Analysis tuning parameters")


class ContradictionsRequest(BaseModel):
    """Request for contradiction detection only."""
    files: Dict[str, str] = Field(..., description="Map of relative path → content")
    groups: List[FileGroupInfo] = Field(..., description="File groups (from prior /duplicates call or caller-provided)")
    config: Optional[AnalysisConfig] = Field(None, description="Analysis tuning parameters")


# ===================================================================
# Response Models
# ===================================================================

class AnalyseResponse(BaseModel):
    """Full analysis response."""
    duplication: DuplicationResponse
    contradictions: ContradictionResponse
    duration_seconds: float


class ServiceStatus(BaseModel):
    """Service dependency status."""
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    services: Dict[str, ServiceStatus]
