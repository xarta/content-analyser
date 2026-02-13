"""
Contradiction detector for content-analyser.

Analyses groups of related files for *factual contradictions* using an
LLM. Provides two detection paths:

  1. **Group-based** — send full file contents per group to the LLM
     for contradiction analysis. Works without embeddings.

  2. **Assertion-based** — extract structured assertions, embed them,
     cluster by similarity, pre-filter with reranker, verify with LLM.
     Requires an embedding client. Supports within-file contradictions.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from source.assertions import Assertion, AssertionExtractor
from source.embedding_client import EmbeddingClient, cosine_similarity
from source.llm_client import LLMClient, strip_markdown_fences, strip_think_tags


logger = logging.getLogger(__name__)


# ===================================================================
# System prompts
# ===================================================================

_CONTRADICTION_SYSTEM_PROMPT = """\
You are a technical documentation analyst. Find CONTRADICTIONS \
between the provided documents.

A CONTRADICTION exists when two documents make INCOMPATIBLE claims \
about the SAME thing — they cannot both be true simultaneously.

VALUE CONTRADICTIONS — different specific values for the same fact:
- "runs on port 5432" vs "runs on port 5433" (same service, different port)
- "container 104" vs "container 102" (same service, different ID)
- "version 2.1.0" vs "version 3.0.0" (same software, different version)

SEMANTIC CONTRADICTIONS — opposing instructions, states, or requirements:
- "always restart the service after config changes" vs "never restart"
- "component X is required and must be installed" vs "component X is optional"
- "hot-reload is enabled by default" vs "hot-reload must be manually configured"

DO NOT flag ANY of these as contradictions:
- Similar or identical content across files (that is duplication)
- Complementary information (both can be true)
- Different contexts (production vs development)
- Old info vs new info (that is obsolescence)
- Different levels of detail about the same topic
- One document having information the other lacks
- Different phrasing of the same fact
- Different services or tools using different ports
- Different scripts or files that serve different purposes
- Different environments with different dependency versions by design
- Client-side vs server-side descriptions of the same technology
- Problem statements vs their workarounds or solutions
- Running a service manually (for testing) vs running it via systemd (for production)

Focus on these categories:
1. Port numbers (e.g., "runs on port 8000" vs "use port 8001")
2. IP addresses (e.g., different IPs for same host)
3. Version requirements (e.g., "requires Python 3.10" vs "requires Python 3.11+")
4. File paths (e.g., "/data/projects/" vs "/opt/newservice/")
5. Configuration values (specific different numbers or strings)
6. Service management (e.g., "restart after changes" vs "never restart")
7. Component status (e.g., "required" vs "optional")
8. Feature state (e.g., "enabled by default" vs "disabled by default")

CRITICAL REQUIREMENT: Both assertions must be about the SAME entity, \
SAME service, SAME file, SAME environment, and SAME time period.

IMPORTANT: Only include clear, definite contradictions where two \
documents make incompatible claims about the SAME thing. If unsure, do NOT include it.

Return a JSON array of contradictions (empty array [] if none found):
[
  {
    "topic": "specific topic of contradiction",
    "file1": "path to first file",
    "excerpt1": "exact relevant excerpt from file1",
    "file2": "path to second file",
    "excerpt2": "exact relevant excerpt from file2",
    "severity": "high|medium|low",
    "explanation": "what specifically conflicts",
    "type": "value|semantic"
  }
]

Return an EMPTY ARRAY [] if no clear contradictions exist."""


_VERIFICATION_SYSTEM_PROMPT = """\
You are a strict technical fact-checker. You will be shown a CANDIDATE \
contradiction found between two documentation files. Your job is to \
decide whether it is a GENUINE contradiction or a false alarm.

A GENUINE contradiction means the two excerpts make INCOMPATIBLE \
claims about the SAME thing.

Reject the finding if:
- The excerpts say the same thing in different words
- They describe different things that happen to look similar
- One is more detailed than the other but not conflicting
- They refer to different environments, stages, or contexts
- The difference is about old vs new (obsolescence, not contradiction)
- They describe different services that intentionally use different ports
- They compare different environments with different dependency versions

Respond ONLY with a JSON object — no markdown fences, no extra text:
{
  "is_contradiction": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of your verdict"
}

If uncertain, err on the side of REJECTING the finding (return false)."""


_ASSERTION_PAIR_PROMPT = """\
You are a technical documentation analyst. You will be shown two \
assertions extracted from documentation files. Determine whether they \
CONTRADICT each other.

Two assertions contradict when they make INCOMPATIBLE claims about the \
SAME thing — they cannot both be true simultaneously.

Types of contradiction:
- "value": Different specific values for the same fact
- "semantic": Opposing instructions, recommendations, or states

NOT contradictions:
- Complementary information (both can be true)
- Different contexts (production vs development)
- Different levels of detail about the same topic
- Same information phrased differently
- Different services or tools using different ports by design
- Different environments with intentionally different dependency versions

Respond ONLY with a JSON object — no markdown fences, no extra text:
{
  "is_contradiction": true or false,
  "type": "value" or "semantic",
  "confidence": 0.0 to 1.0,
  "explanation": "Brief explanation of your verdict"
}

If uncertain, err on the side of REJECTING the finding (return false)."""


# ===================================================================
# Quality helpers
# ===================================================================

def _severity_from_confidence(confidence: float) -> str:
    """Map a confidence score (0.0–1.0) to a severity level."""
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _word_set(text: str) -> Set[str]:
    """Extract a set of lowercase content words (≥3 chars) from text."""
    return {w.lower() for w in text.split() if len(w) >= 3}


def _assertion_overlap(c1: "Contradiction", c2: "Contradiction") -> bool:
    """Check whether two contradictions share assertion text overlap."""
    texts1 = [c1.assertion1 or c1.excerpt1, c1.assertion2 or c1.excerpt2]
    texts2 = [c2.assertion1 or c2.excerpt1, c2.assertion2 or c2.excerpt2]
    for t1 in texts1:
        w1 = _word_set(t1)
        if not w1:
            continue
        for t2 in texts2:
            w2 = _word_set(t2)
            if not w2:
                continue
            overlap = len(w1 & w2)
            smaller = min(len(w1), len(w2))
            if smaller > 0 and overlap / smaller >= 0.5:
                return True
    return False


def _merge_overlapping_findings(
    findings: List["Contradiction"],
) -> List["Contradiction"]:
    """Merge findings within a file pair that describe the same contradiction."""
    if len(findings) <= 1:
        return list(findings)

    clusters: List["Contradiction"] = []
    for f in findings:
        merged = False
        for i, rep in enumerate(clusters):
            if _assertion_overlap(f, rep):
                if len(f.explanation) > len(rep.explanation):
                    clusters[i] = f
                merged = True
                break
        if not merged:
            clusters.append(f)

    return clusters


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class Contradiction:
    """A single contradiction found between two files."""
    topic: str
    file1: str
    excerpt1: str
    file2: str
    excerpt2: str
    severity: str = "medium"
    explanation: str = ""
    type: str = "value"
    confidence: float = 0.0
    assertion1: str = ""
    assertion2: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "topic": self.topic,
            "file_a": self.file1,
            "excerpt_a": self.excerpt1,
            "file_b": self.file2,
            "excerpt_b": self.excerpt2,
            "confidence": round(self.confidence, 2),
            "severity": self.severity,
            "explanation": self.explanation,
            "type": self.type,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contradiction":
        return cls(
            topic=data.get("topic", ""),
            file1=data.get("file1", data.get("file_a", "")),
            excerpt1=data.get("excerpt1", data.get("excerpt_a", "")),
            file2=data.get("file2", data.get("file_b", "")),
            excerpt2=data.get("excerpt2", data.get("excerpt_b", "")),
            severity=data.get("severity", "medium"),
            explanation=data.get("explanation", ""),
            type=data.get("type", "value"),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class ContradictionResult:
    """Complete contradiction analysis result."""
    findings: List[Contradiction] = field(default_factory=list)
    groups_analysed: int = 0
    assertions_extracted: int = 0
    contradictions_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "summary": {
                "groups_analysed": self.groups_analysed,
                "assertions_extracted": self.assertions_extracted,
                "contradictions_found": len(self.findings),
            },
        }


# ===================================================================
# ContradictionDetector
# ===================================================================

class ContradictionDetector:
    """Detect contradictions across groups of related files.

    Provides two detection paths:

    1. **``analyse()``** — group-based: send file contents to LLM per group.
    2. **``analyse_with_assertions()``** — assertion-based pipeline:
       extract → embed → cluster → rerank → verify.
    """

    MAX_FILES_PER_GROUP = 5
    MAX_CONTENT_LENGTH = 12000
    SIMILARITY_THRESHOLD = 0.7
    RERANKER_THRESHOLD = 0.3

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: Optional[EmbeddingClient] = None,
        max_workers: int = 1,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        self._llm = llm_client
        self._embedding_client = embedding_client
        self._max_workers = max(1, max_workers)
        self._progress_callback = progress_callback

    def analyse(
        self,
        file_groups: List[Dict[str, Any]],
        file_contents: Dict[str, str],
        duplicate_file_pairs: Optional[Set[frozenset]] = None,
    ) -> ContradictionResult:
        """Run group-based contradiction detection.

        Args:
            file_groups: List of group dicts with ``shared_topics`` and
                ``files`` keys.
            file_contents: Mapping of file_path → full text content.
            duplicate_file_pairs: Known duplicate pairs to filter.

        Returns:
            ``ContradictionResult`` with deduplicated findings.
        """
        work_items: List[Tuple[str, Dict[str, str]]] = []
        groups_skipped = 0

        for group in file_groups:
            name = group.get("shared_topics", ["unknown"])[0] if isinstance(
                group.get("shared_topics"), list,
            ) else str(group.get("shared_topics", "unknown"))
            paths = group.get("files", [])

            if len(paths) < 2:
                groups_skipped += 1
                continue

            group_contents: Dict[str, str] = {}
            for path in paths[:self.MAX_FILES_PER_GROUP]:
                content = file_contents.get(path, "")
                if content:
                    group_contents[path] = content[:self.MAX_CONTENT_LENGTH]

            if len(group_contents) < 2:
                groups_skipped += 1
                continue

            work_items.append((name, group_contents))

        # Analyse groups
        all_contradictions = self._analyse_groups_parallel(work_items)

        # Deduplicate
        seen: Dict[tuple, Contradiction] = {}
        severity_rank = {"high": 3, "medium": 2, "low": 1}
        for c in all_contradictions:
            files = tuple(sorted([c.file1, c.file2]))
            topic = c.topic.lower().strip()
            key = (files[0], files[1], topic)
            if key in seen:
                if severity_rank.get(c.severity, 0) > severity_rank.get(
                    seen[key].severity, 0,
                ):
                    seen[key] = c
            else:
                seen[key] = c

        # Filter contradictions between known duplicate pairs
        filtered = list(seen.values())
        if duplicate_file_pairs:
            filtered = [
                c for c in filtered
                if frozenset([c.file1, c.file2]) not in duplicate_file_pairs
            ]

        # Verify each finding
        verified = self._verify_findings(filtered, file_contents)

        return ContradictionResult(
            findings=verified,
            groups_analysed=len(work_items),
            contradictions_found=len(verified),
        )

    def analyse_with_assertions(
        self,
        file_contents: Dict[str, str],
        file_groups: Optional[List[Dict[str, Any]]] = None,
        duplicate_file_pairs: Optional[Set[frozenset]] = None,
    ) -> ContradictionResult:
        """Assertion-based contradiction detection pipeline.

        Steps:
          1. Extract assertions from all files
          2. Embed assertions (batch)
          3. Cluster by similarity
          4. Pre-filter with reranker (if available)
          5. Verify candidate pairs with LLM
          6. Deduplicate and filter

        Args:
            file_contents: Mapping of file_path → content.
            file_groups: Optional file groups (metadata only).
            duplicate_file_pairs: Known duplicate pairs to skip.

        Returns:
            ``ContradictionResult`` with verified findings.
        """
        # Step 1: Extract assertions
        extractor = AssertionExtractor(
            self._llm, max_workers=self._max_workers,
        )
        file_tuples = [(path, content) for path, content in file_contents.items()]
        all_assertions = extractor.extract_from_files(file_tuples)

        # Flatten
        flat: List[Assertion] = []
        for path, assertions in all_assertions.items():
            flat.extend(assertions)

        total_assertions = len(flat)

        if len(flat) < 2:
            return ContradictionResult(
                groups_analysed=len(file_contents),
                assertions_extracted=total_assertions,
            )

        # Steps 2–4: Find candidate pairs
        if self._embedding_client:
            candidate_pairs = self._find_candidate_pairs_embedded(
                flat, duplicate_file_pairs,
            )
        else:
            candidate_pairs = self._find_candidate_pairs_brute(
                flat, duplicate_file_pairs,
            )

        # Step 5: Verify each candidate pair
        contradictions = self._verify_assertion_pairs_parallel(
            candidate_pairs, file_contents,
        )

        # Step 6: Deduplicate by assertion overlap
        if len(contradictions) > 1:
            by_pair: Dict[frozenset, List[Contradiction]] = {}
            for c in contradictions:
                pair = frozenset([c.file1, c.file2])
                by_pair.setdefault(pair, []).append(c)

            merged: List[Contradiction] = []
            for pair, findings in by_pair.items():
                merged.extend(_merge_overlapping_findings(findings))
            contradictions = merged

        # Filter duplicate pairs
        if duplicate_file_pairs:
            contradictions = [
                c for c in contradictions
                if frozenset([c.file1, c.file2]) not in duplicate_file_pairs
            ]

        return ContradictionResult(
            findings=contradictions,
            groups_analysed=len(file_contents),
            assertions_extracted=total_assertions,
            contradictions_found=len(contradictions),
        )

    # -----------------------------------------------------------------
    # Internal: group-based detection
    # -----------------------------------------------------------------

    def _analyse_groups_parallel(
        self,
        work_items: List[Tuple[str, Dict[str, str]]],
    ) -> List[Contradiction]:
        """Analyse file groups for contradictions, optionally in parallel."""
        if not work_items:
            return []

        # Sequential path
        if self._max_workers <= 1 or len(work_items) == 1:
            all_contradictions: List[Contradiction] = []
            for name, group_contents in work_items:
                all_contradictions.extend(
                    self._analyse_group(name, group_contents),
                )
            return all_contradictions

        # Parallel path
        results_by_idx: List[List[Contradiction]] = [[] for _ in work_items]
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._analyse_group, name, group_contents,
                ): i
                for i, (name, group_contents) in enumerate(work_items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_by_idx[idx] = future.result()
                except Exception:
                    logger.warning("Group analysis failed for item %d", idx)
                    results_by_idx[idx] = []

        all_contradictions = []
        for group_results in results_by_idx:
            all_contradictions.extend(group_results)
        return all_contradictions

    def _analyse_group(
        self,
        group_name: str,
        file_contents: Dict[str, str],
    ) -> List[Contradiction]:
        """Analyse a single file group for contradictions."""
        user_content = f"Document Group: {group_name}\n\n"
        for path, content in file_contents.items():
            user_content += f"=== FILE: {path} ===\n{content}\n\n"

        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": user_content}],
                system=_CONTRADICTION_SYSTEM_PROMPT,
                temperature=0.1,
            )
            return self._parse_response(response)
        except Exception:
            return []

    @staticmethod
    def _parse_response(response: str) -> List[Contradiction]:
        """Parse LLM response into Contradiction objects."""
        cleaned = strip_think_tags(response)
        cleaned = strip_markdown_fences(cleaned)
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [Contradiction.from_dict(item) for item in parsed]
        except json.JSONDecodeError:
            pass

        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, list):
                    return [Contradiction.from_dict(item) for item in parsed]
            except json.JSONDecodeError:
                pass

        return []

    # -----------------------------------------------------------------
    # Internal: verification
    # -----------------------------------------------------------------

    def _verify_findings(
        self,
        findings: List[Contradiction],
        file_contents: Dict[str, str],
    ) -> List[Contradiction]:
        """Verify findings with focused LLM calls."""
        if not findings:
            return []

        # Sequential path
        if self._max_workers <= 1 or len(findings) == 1:
            verified: List[Contradiction] = []
            for c in findings:
                if self._verify_contradiction(c, file_contents):
                    verified.append(c)
            return verified

        # Parallel path
        verified = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._verify_contradiction, c, file_contents,
                ): i
                for i, c in enumerate(findings)
            }
            results: List[Optional[bool]] = [None] * len(findings)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = True  # conservative: keep

        for i, keep in enumerate(results):
            if keep:
                verified.append(findings[i])

        return verified

    def _verify_contradiction(
        self,
        contradiction: Contradiction,
        file_contents: Dict[str, str],
    ) -> bool:
        """Verify a single contradiction with a focused LLM call."""
        parts = [
            f"Topic: {contradiction.topic}",
            f"File 1: {contradiction.file1}",
            f"Excerpt 1: {contradiction.excerpt1}",
            f"File 2: {contradiction.file2}",
            f"Excerpt 2: {contradiction.excerpt2}",
            f"Claimed severity: {contradiction.severity}",
            f"Explanation: {contradiction.explanation}",
        ]

        for label, path in [
            ("File 1", contradiction.file1),
            ("File 2", contradiction.file2),
        ]:
            content = file_contents.get(path, "")
            if content:
                truncated = content[:self.MAX_CONTENT_LENGTH]
                parts.append(
                    f"\n=== {label} full content ({path}) ===\n{truncated}"
                )

        user_content = "\n".join(parts)

        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": user_content}],
                system=_VERIFICATION_SYSTEM_PROMPT,
                temperature=0.1,
            )

            cleaned = strip_think_tags(response)
            cleaned = strip_markdown_fences(cleaned)
            cleaned = cleaned.strip()

            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(cleaned[start:end + 1])
                    except json.JSONDecodeError:
                        pass

            if parsed and isinstance(parsed, dict):
                result = parsed.get("is_contradiction")
                if isinstance(result, bool):
                    return result
                if isinstance(result, str):
                    return result.lower().strip() in ("true", "yes", "1")

            return True  # conservative: keep

        except Exception:
            return True  # conservative: keep on error

    # -----------------------------------------------------------------
    # Internal: assertion-based pipeline
    # -----------------------------------------------------------------

    def _find_candidate_pairs_embedded(
        self,
        assertions: List[Assertion],
        duplicate_file_pairs: Optional[Set[frozenset]] = None,
    ) -> List[Tuple[Assertion, Assertion]]:
        """Find candidate contradiction pairs using embeddings + reranker."""
        texts = [a.text for a in assertions]
        result = self._embedding_client.embed(texts)
        embeddings = result.embeddings

        candidates: List[Tuple[int, int, float]] = []
        for i in range(len(assertions)):
            for j in range(i + 1, len(assertions)):
                a1, a2 = assertions[i], assertions[j]
                if a1.file_path != a2.file_path:
                    if self._is_duplicate_pair(
                        a1.file_path, a2.file_path, duplicate_file_pairs,
                    ):
                        continue

                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self.SIMILARITY_THRESHOLD:
                    candidates.append((i, j, sim))

        candidates.sort(key=lambda x: x[2], reverse=True)

        # Refine with reranker if available
        if (
            self._embedding_client
            and self._embedding_client.config.reranker_base_url
        ):
            reranked: List[Tuple[int, int, float]] = []
            for i, j, sim in candidates:
                try:
                    rr = self._embedding_client.rerank(
                        assertions[i].text, [assertions[j].text],
                    )
                    score = rr.scores[0] if rr.scores else 0.0
                    if score >= self.RERANKER_THRESHOLD:
                        reranked.append((i, j, score))
                except Exception:
                    reranked.append((i, j, sim))
            reranked.sort(key=lambda x: x[2], reverse=True)
            return [(assertions[i], assertions[j]) for i, j, _ in reranked]

        return [(assertions[i], assertions[j]) for i, j, _ in candidates]

    @staticmethod
    def _find_candidate_pairs_brute(
        assertions: List[Assertion],
        duplicate_file_pairs: Optional[Set[frozenset]] = None,
    ) -> List[Tuple[Assertion, Assertion]]:
        """Generate candidate pairs without embeddings (brute-force)."""
        max_pairs = 50
        pairs: List[Tuple[Assertion, Assertion]] = []

        for i in range(len(assertions)):
            if len(pairs) >= max_pairs:
                break
            for j in range(i + 1, len(assertions)):
                if len(pairs) >= max_pairs:
                    break
                a1, a2 = assertions[i], assertions[j]
                if a1.file_path != a2.file_path:
                    if ContradictionDetector._is_duplicate_pair(
                        a1.file_path, a2.file_path, duplicate_file_pairs,
                    ):
                        continue
                pairs.append((a1, a2))

        return pairs

    def _verify_assertion_pair(
        self,
        a1: Assertion,
        a2: Assertion,
        file_contents: Dict[str, str],
    ) -> Optional[Contradiction]:
        """Verify whether two assertions contradict each other."""
        parts = [f"Assertion 1 (from {a1.file_path}"]
        if a1.section:
            parts[0] += f", section: {a1.section}"
        parts[0] += "):"
        parts.append(f'  "{a1.text}"')
        if a1.context:
            parts.append(f"  Context: {a1.context}")

        parts.append("")
        parts.append(f"Assertion 2 (from {a2.file_path}")
        if a2.section:
            parts[-1] += f", section: {a2.section}"
        parts[-1] += "):"
        parts.append(f'  "{a2.text}"')
        if a2.context:
            parts.append(f"  Context: {a2.context}")

        user_content = "\n".join(parts) + "\n/no_think"

        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": user_content}],
                system=_ASSERTION_PAIR_PROMPT,
                temperature=0.1,
            )

            cleaned = strip_think_tags(response)
            cleaned = strip_markdown_fences(cleaned)
            cleaned = cleaned.strip()

            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(cleaned[start:end + 1])
                    except json.JSONDecodeError:
                        pass

            if not parsed or not isinstance(parsed, dict):
                return None

            is_contradiction = parsed.get("is_contradiction")
            if isinstance(is_contradiction, str):
                is_contradiction = is_contradiction.lower().strip() in (
                    "true", "yes", "1",
                )

            if not is_contradiction:
                return None

            confidence = float(parsed.get("confidence", 0.5))
            ctype = parsed.get("type", "value")
            if ctype not in ("value", "semantic"):
                ctype = "value"

            return Contradiction(
                topic=parsed.get("explanation", a1.text[:60]),
                file1=a1.file_path,
                excerpt1=a1.text,
                file2=a2.file_path,
                excerpt2=a2.text,
                severity=_severity_from_confidence(confidence),
                explanation=parsed.get("explanation", ""),
                type=ctype,
                confidence=confidence,
                assertion1=a1.text,
                assertion2=a2.text,
            )

        except Exception:
            return None

    def _verify_assertion_pairs_parallel(
        self,
        candidate_pairs: List[Tuple[Assertion, Assertion]],
        file_contents: Dict[str, str],
    ) -> List[Contradiction]:
        """Verify assertion pairs with the LLM, optionally in parallel."""
        if not candidate_pairs:
            return []

        # Sequential path
        if self._max_workers <= 1 or len(candidate_pairs) <= 1:
            contradictions: List[Contradiction] = []
            for a1, a2 in candidate_pairs:
                result = self._verify_assertion_pair(a1, a2, file_contents)
                if result is not None:
                    contradictions.append(result)
            return contradictions

        # Parallel path
        contradictions = []
        results: List[Optional[Contradiction]] = [None] * len(candidate_pairs)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._verify_assertion_pair, a1, a2, file_contents,
                ): i
                for i, (a1, a2) in enumerate(candidate_pairs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.warning("Assertion pair verification failed for pair %d", idx)
                    results[idx] = None

        for r in results:
            if r is not None:
                contradictions.append(r)

        return contradictions

    @staticmethod
    def _is_duplicate_pair(
        file1: str,
        file2: str,
        duplicate_file_pairs: Optional[Set[frozenset]],
    ) -> bool:
        """Check whether two files are a known duplicate pair."""
        if not duplicate_file_pairs:
            return False
        return frozenset([file1, file2]) in duplicate_file_pairs
