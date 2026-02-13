"""
Duplication detector for content-analyser.

Three-layer approach:

  1. **Topic analysis** — LLM extracts topics and entities from each file.
     ``TopicAnalyser`` calls the LLM and returns structured results.
  2. **File grouping** — ``FileGrouper`` clusters files that share topics
     so downstream stages know *which files are about the same thing*.
  3. **Content duplication** — ``DuplicationDetector`` chunks files via
     the external Normalised Semantic Chunker, embeds chunks via the
     embedding endpoint, runs pairwise cosine comparison, and
     BFS-groups related matches.

The combined output (``DuplicationResult``) carries topic groups, chunk-
level duplicate matches, and summary statistics.
"""

import hashlib
import json
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from source.chunker_client import ChunkerClient, ChunkerError
from source.embedding_client import EmbeddingClient, EmbeddingError, cosine_similarity
from source.llm_client import LLMClient, strip_think_tags, strip_markdown_fences


logger = logging.getLogger(__name__)


# ===================================================================
# Severity thresholds
# ===================================================================

SEVERITY_HIGH = 0.95
SEVERITY_MEDIUM = 0.85


def _severity_label(score: float) -> str:
    """Map a similarity score to a severity label."""
    if score >= SEVERITY_HIGH:
        return "high"
    elif score >= SEVERITY_MEDIUM:
        return "medium"
    return "low"


def _make_chunk_id(file_path: str, start_index: int, content: str) -> str:
    """Generate a unique chunk ID from path, position, and content prefix."""
    hash_input = f"{file_path}:{start_index}:{content[:100]}"
    return hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:12]


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class Chunk:
    """A chunk of content from a document.

    Attributes:
        content: The text content of the chunk.
        file_path: Source file path.
        start_index: Start character index in the original document.
        end_index: End character index in the original document.
        chunk_id: Unique identifier for caching.
    """
    content: str
    file_path: str
    start_index: int = 0
    end_index: int = 0
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = _make_chunk_id(
                self.file_path, self.start_index, self.content,
            )


@dataclass
class TopicResult:
    """Topics and entities extracted from a single file by the LLM.

    Attributes:
        file_path: Relative path to the file.
        topics: High-level topics (e.g. "installation", "database").
        entities: Specific named entities (services, tools, etc.).
    """
    file_path: str
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "topics": self.topics,
            "entities": self.entities,
        }


@dataclass
class FileGroup:
    """A group of files that share a topic or entity.

    Attributes:
        group_id: Numeric group ID.
        shared_topics: The shared topic(s).
        files: Paths of files in this group.
    """
    group_id: int
    shared_topics: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "shared_topics": self.shared_topics,
            "files": self.files,
        }


@dataclass
class DuplicateMatch:
    """A pair of chunks that are semantically similar.

    Attributes:
        chunk_a_file: File path of chunk A.
        chunk_a_content: Text content of chunk A.
        chunk_b_file: File path of chunk B.
        chunk_b_content: Text content of chunk B.
        similarity: Cosine similarity score (0–1).
        severity: Category: ``high`` (≥0.95), ``medium`` (≥0.85), ``low``.
    """
    chunk_a_file: str
    chunk_a_content: str
    chunk_b_file: str
    chunk_b_content: str
    similarity: float
    severity: str = ""

    def __post_init__(self) -> None:
        if not self.severity:
            self.severity = _severity_label(self.similarity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_a": {"file": self.chunk_a_file, "content": self.chunk_a_content},
            "chunk_b": {"file": self.chunk_b_file, "content": self.chunk_b_content},
            "similarity": round(self.similarity, 4),
            "severity": self.severity,
        }


@dataclass
class DuplicationResult:
    """Complete duplication analysis result.

    Carries all three layers of analysis output.
    """
    # Layer 1: topic analysis
    topics: List[TopicResult] = field(default_factory=list)

    # Layer 2: file grouping
    groups: List[FileGroup] = field(default_factory=list)

    # Layer 3: chunk-level duplication
    matches: List[DuplicateMatch] = field(default_factory=list)

    # Statistics
    files_analysed: int = 0
    total_chunks: int = 0
    chunks_compared: int = 0
    chunks_skipped: int = 0

    @property
    def duplicate_matches(self) -> int:
        return len(self.matches)

    @property
    def duplicate_groups(self) -> int:
        return len(self.groups)

    def get_duplicate_file_pairs(
        self,
        threshold: float = 0.85,
    ) -> Set[frozenset]:
        """Get file pairs with chunk-level similarity above threshold."""
        pairs: Set[frozenset] = set()
        for match in self.matches:
            if match.similarity >= threshold:
                pair = frozenset([match.chunk_a_file, match.chunk_b_file])
                if len(pair) == 2:
                    pairs.add(pair)
        return pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topics": [t.to_dict() for t in self.topics],
            "groups": [g.to_dict() for g in self.groups],
            "matches": [m.to_dict() for m in self.matches],
            "summary": {
                "files_analysed": self.files_analysed,
                "chunks_compared": self.chunks_compared,
                "duplicate_matches": self.duplicate_matches,
                "duplicate_groups": self.duplicate_groups,
            },
        }


# ===================================================================
# TopicAnalyser — Layer 1
# ===================================================================

_TOPIC_SYSTEM_PROMPT = """\
You are a documentation analyst. Analyse the provided file content and \
return a JSON object with:

{
  "topics": ["list", "of", "key", "topics"],
  "entities": ["specific", "named", "services", "tools", "mentioned"]
}

Rules:
- topics: high-level subjects (e.g. "installation", "database", \
"networking", "monitoring", "configuration")
- entities: specific named things (e.g. "PostgreSQL", "Caddy", "Docker", \
"systemd")
- Use lowercase for all values
- Return between 1 and 10 items per list
- If the file has very little content, return fewer items
- Return ONLY the JSON object, no explanation"""


class TopicAnalyser:
    """Extract topics and entities from file content using an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_workers: int = 1,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        self._llm = llm_client
        self._max_workers = max(1, max_workers)
        self._on_progress = on_progress

    def analyse_file(self, file_path: str, content: str) -> TopicResult:
        """Extract topics and entities from a single file."""
        truncated = content[:4000]
        user_message = f"File: {file_path}\n\n{truncated}"

        try:
            response = self._llm.ask(
                content=user_message,
                prompt=_TOPIC_SYSTEM_PROMPT,
                temperature=0.1,
            )

            parsed = self._parse_response(response)
            return TopicResult(
                file_path=file_path,
                topics=[t.lower().strip() for t in parsed.get("topics", []) if t],
                entities=[e.lower().strip() for e in parsed.get("entities", []) if e],
            )
        except Exception:
            return TopicResult(file_path=file_path)

    def analyse_files(
        self, files: List[tuple],
    ) -> List[TopicResult]:
        """Extract topics from multiple files (optionally parallel)."""
        if not files:
            return []

        total = len(files)

        # Sequential path
        if self._max_workers <= 1 or total == 1:
            results = []
            for idx, (file_path, content) in enumerate(files):
                result = self.analyse_file(file_path, content)
                results.append(result)
                if self._on_progress:
                    self._on_progress(file_path, idx + 1, total)
            return results

        # Parallel path — preserve input order
        results: List[Optional[TopicResult]] = [None] * total
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self.analyse_file, fp, content): i
                for i, (fp, content) in enumerate(files)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                fp = files[idx][0]
                completed += 1
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.warning("Topic analysis failed for %s", fp)
                    results[idx] = TopicResult(file_path=fp)

                if self._on_progress:
                    self._on_progress(fp, completed, total)

        return results  # type: ignore[return-value]

    @staticmethod
    def _parse_response(response: str) -> Dict[str, Any]:
        """Parse JSON from an LLM response, handling common quirks."""
        cleaned = strip_think_tags(response)
        cleaned = strip_markdown_fences(cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {}


# ===================================================================
# FileGrouper — Layer 2
# ===================================================================

class FileGrouper:
    """Group files by shared topics and entities."""

    @staticmethod
    def group(topic_results: List[TopicResult]) -> List[FileGroup]:
        """Build file groups from topic analysis results.

        Args:
            topic_results: Output from ``TopicAnalyser.analyse_files()``.

        Returns:
            List of ``FileGroup`` objects, each with ≥2 files.
        """
        groupings: Dict[str, List[str]] = {}

        for result in topic_results:
            for topic in result.topics:
                key = topic.lower()
                if key not in groupings:
                    groupings[key] = []
                if result.file_path not in groupings[key]:
                    groupings[key].append(result.file_path)

            for entity in result.entities:
                key = entity.lower()
                if key not in groupings:
                    groupings[key] = []
                if result.file_path not in groupings[key]:
                    groupings[key].append(result.file_path)

        # Only keep groups with 2+ files
        groups = []
        for group_id, (name, paths) in enumerate(
            sorted(groupings.items()), start=1,
        ):
            if len(paths) >= 2:
                groups.append(FileGroup(
                    group_id=group_id,
                    shared_topics=[name],
                    files=sorted(paths),
                ))

        return groups


# ===================================================================
# DuplicationDetector — Layer 3
# ===================================================================

class DuplicationDetector:
    """Full duplication detection pipeline.

    Integrates all three layers:
      1. LLM topic analysis (via ``TopicAnalyser``)
      2. File grouping (via ``FileGrouper``)
      3. Embedding-based content duplication (chunk → embed → compare)

    Uses the external Normalised Semantic Chunker for document chunking,
    replacing the built-in paragraph chunker.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        chunker_client: ChunkerClient,
        threshold_high: float = SEVERITY_HIGH,
        threshold_medium: float = SEVERITY_MEDIUM,
        cross_file_only: bool = True,
        max_workers: int = 1,
    ):
        """Initialise the detector.

        Args:
            llm_client: LLM client for topic extraction.
            embedding_client: Embedding client for chunk comparison.
            chunker_client: Chunker client for document chunking.
            threshold_high: High severity similarity threshold.
            threshold_medium: Medium severity threshold (also minimum match threshold).
            cross_file_only: If True, skip within-file pairs.
            max_workers: Maximum concurrent LLM calls.
        """
        self._llm = llm_client
        self._embedding_client = embedding_client
        self._chunker_client = chunker_client
        self._threshold_high = threshold_high
        self._threshold_medium = threshold_medium
        self._cross_file_only = cross_file_only
        self._max_workers = max(1, max_workers)

    def analyse(
        self,
        files: Dict[str, str],
        chunker_max_tokens: Optional[int] = None,
    ) -> DuplicationResult:
        """Run the full three-layer analysis pipeline.

        Args:
            files: Map of relative path → content.
            chunker_max_tokens: Optional max tokens for chunker config.

        Returns:
            ``DuplicationResult`` with all layers populated.
        """
        if len(files) < 2:
            return DuplicationResult(files_analysed=len(files))

        file_tuples = [(fp, content) for fp, content in files.items()]

        # Layer 1: Topic analysis
        analyser = TopicAnalyser(self._llm, max_workers=self._max_workers)
        topic_results = analyser.analyse_files(file_tuples)

        # Layer 2: File grouping
        file_groups = FileGrouper.group(topic_results)

        # Layer 3: Chunk-level duplication
        all_chunks = self._chunk_all_files(files, chunker_max_tokens)

        matches: List[DuplicateMatch] = []
        chunks_compared = 0
        chunks_skipped = 0

        if len(all_chunks) >= 2:
            # Embed all chunks
            chunk_texts = [c.content for c in all_chunks]
            try:
                embed_result = self._embedding_client.embed(chunk_texts)
                embeddings = embed_result.embeddings

                # Separate valid from invalid
                valid_indices = [
                    i for i, emb in enumerate(embeddings)
                    if emb and len(emb) > 0
                ]
                chunks_skipped = len(all_chunks) - len(valid_indices)
                chunks_compared = len(valid_indices)

                # Pairwise comparison
                if len(valid_indices) >= 2:
                    matches = self._find_matches(
                        all_chunks, embeddings, valid_indices,
                    )

            except EmbeddingError as exc:
                logger.error("Embedding failed during duplication detection: %s", exc)
                chunks_skipped = len(all_chunks)

        return DuplicationResult(
            topics=topic_results,
            groups=file_groups,
            matches=matches,
            files_analysed=len(files),
            total_chunks=len(all_chunks),
            chunks_compared=chunks_compared,
            chunks_skipped=chunks_skipped,
        )

    def _chunk_all_files(
        self,
        files: Dict[str, str],
        max_tokens: Optional[int] = None,
    ) -> List[Chunk]:
        """Chunk all files using the external semantic chunker.

        Args:
            files: Map of relative path → content.
            max_tokens: Optional max tokens to pass to the chunker.

        Returns:
            List of ``Chunk`` objects from all files.
        """
        all_chunks: List[Chunk] = []

        for file_path, content in files.items():
            if not content or not content.strip():
                continue

            try:
                kwargs = {}
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                raw_chunks = self._chunker_client.chunk_document(
                    filename=file_path,
                    content=content,
                    **kwargs,
                )

                for raw in raw_chunks:
                    text = raw.get("text", "")
                    if text and len(text.split()) >= 5:  # Skip tiny chunks
                        all_chunks.append(Chunk(
                            content=text,
                            file_path=file_path,
                            start_index=raw.get("start_index", 0),
                            end_index=raw.get("end_index", 0),
                        ))

            except ChunkerError as exc:
                logger.error(
                    "Chunker failed for %s: %s — skipping file",
                    file_path, exc,
                )

        return all_chunks

    def _find_matches(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        valid_indices: List[int],
    ) -> List[DuplicateMatch]:
        """Find all chunk pairs above the similarity threshold."""
        matches: List[DuplicateMatch] = []

        for i_idx in range(len(valid_indices)):
            i = valid_indices[i_idx]
            for j_idx in range(i_idx + 1, len(valid_indices)):
                j = valid_indices[j_idx]

                if self._cross_file_only and chunks[i].file_path == chunks[j].file_path:
                    continue

                sim = cosine_similarity(embeddings[i], embeddings[j])

                if sim >= self._threshold_medium:
                    matches.append(DuplicateMatch(
                        chunk_a_file=chunks[i].file_path,
                        chunk_a_content=chunks[i].content,
                        chunk_b_file=chunks[j].file_path,
                        chunk_b_content=chunks[j].content,
                        similarity=sim,
                    ))

        return matches
