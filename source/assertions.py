"""
Assertion extractor for content-analyser.

Extracts structured logical assertions from documentation files using an
LLM. Each assertion is a factual claim, instruction, recommendation,
requirement, or state declaration that can be compared with assertions
from other files to find contradictions.

The assertion extraction approach gives systematic coverage — every claim
in every document is identified and structured, rather than relying on
the LLM to spot contradictions in a single pass over raw text.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from source.llm_client import LLMClient, strip_markdown_fences, strip_think_tags


# ===================================================================
# Assertion types
# ===================================================================

ASSERTION_TYPES = ("value", "instruction", "recommendation", "requirement", "state")


# ===================================================================
# Extraction prompt
# ===================================================================

_EXTRACTION_SYSTEM_PROMPT = """\
You are a technical documentation analyst. Extract ALL logical \
assertions from the provided document.

An assertion is any statement that makes a factual claim, gives an \
instruction, offers a recommendation, states a requirement, or \
declares a state.

ASSERTION TYPES (use exactly these labels):
- "value": A factual claim about a specific value \
  (e.g. "runs on port 5432", "version 2.1.0", "container 104", \
  "installed at /data/projects/")
- "instruction": A directive telling the reader to do something \
  (e.g. "restart the service after config changes", \
  "run install.sh first", "use systemctl restart")
- "recommendation": A suggestion or preference \
  (e.g. "prefer TCP over UDP", "use the latest version")
- "requirement": Something that must be present or true \
  (e.g. "requires root access", "Python 3.10+ is needed", \
  "the monitoring agent must be installed")
- "state": A declaration about the current state of something \
  (e.g. "hot-reload is enabled by default", \
  "this component is optional", "feature X is disabled")

RULES:
1. Extract EVERY assertion, even if it seems obvious
2. Use the EXACT text from the document for the assertion text
3. Include the section heading where the assertion appears
4. Estimate start and end line numbers
5. Include surrounding context (the full paragraph)
6. Rate your confidence (0.0 to 1.0) in the extraction
7. Do NOT invent assertions that are not in the text
8. Extract assertions from ALL sections of the document

Return a JSON array of assertion objects:
[
  {
    "text": "the assertion as a clear statement",
    "assertion_type": "value|instruction|recommendation|requirement|state",
    "section": "section heading or empty string",
    "start_line": 1,
    "end_line": 3,
    "context": "the surrounding paragraph text",
    "confidence": 0.9
  }
]

Return an EMPTY ARRAY [] if no assertions can be extracted."""


# ===================================================================
# Data class
# ===================================================================

@dataclass
class Assertion:
    """A single logical assertion extracted from a documentation file.

    Attributes:
        text: The claim/instruction/recommendation as a clear statement.
        assertion_type: One of ``value``, ``instruction``,
            ``recommendation``, ``requirement``, ``state``.
        file_path: Source file path.
        section: Section heading where the assertion appears (empty if none).
        start_line: Approximate start line in the source file.
        end_line: Approximate end line in the source file.
        context: Surrounding paragraph text for downstream use.
        confidence: LLM's confidence in the extraction (0.0 to 1.0).
    """
    text: str
    assertion_type: str = "value"
    file_path: str = ""
    section: str = ""
    start_line: int = 0
    end_line: int = 0
    context: str = ""
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "text": self.text,
            "assertion_type": self.assertion_type,
            "file_path": self.file_path,
            "section": self.section,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "context": self.context,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Assertion":
        """Deserialise from a dict."""
        return cls(
            text=data.get("text", ""),
            assertion_type=data.get("assertion_type", "value"),
            file_path=data.get("file_path", ""),
            section=data.get("section", ""),
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            context=data.get("context", ""),
            confidence=float(data.get("confidence", 0.9)),
        )


# ===================================================================
# AssertionExtractor
# ===================================================================

# Maximum characters of file content to send to the LLM
MAX_EXTRACTION_LENGTH = 12000

logger = logging.getLogger(__name__)


class AssertionExtractor:
    """Extract structured assertions from documentation files via LLM.

    Usage::

        extractor = AssertionExtractor(llm_client, max_workers=4)
        assertions = extractor.extract_from_file("setup-guide.md", content)
        all_assertions = extractor.extract_from_files(files)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_workers: int = 1,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        """Initialise with an LLM client.

        Args:
            llm_client: Configured ``LLMClient`` for the LLM.
            max_workers: Maximum concurrent extraction threads.
            on_progress: Optional callback ``(file_path, index, total)``
                fired after each file completes.
        """
        self._llm = llm_client
        self._max_workers = max(1, max_workers)
        self._on_progress = on_progress

    def extract_from_file(
        self,
        file_path: str,
        content: str,
    ) -> List[Assertion]:
        """Extract all assertions from a single file.

        Sends file content (truncated to ``MAX_EXTRACTION_LENGTH``) to
        the LLM and parses the structured response.

        Args:
            file_path: Relative path to the file.
            content: Full text content of the file.

        Returns:
            List of ``Assertion`` objects extracted from the file.
        """
        if not content or not content.strip():
            return []

        truncated = content[:MAX_EXTRACTION_LENGTH]

        user_content = (
            f"File: {file_path}\n\n"
            f"{truncated}\n\n"
            f"Extract ALL assertions from this document. /no_think"
        )

        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": user_content}],
                system=_EXTRACTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=8192,
            )

            assertions = self._parse_response(response, file_path)
            return assertions

        except Exception:
            logger.warning("Assertion extraction failed for %s", file_path)
            return []

    def extract_from_files(
        self,
        files: List[Tuple[str, str]],
    ) -> Dict[str, List[Assertion]]:
        """Extract assertions from multiple files (optionally in parallel).

        When ``max_workers > 1``, files are processed concurrently using
        ``ThreadPoolExecutor``.  Individual file failures return an empty
        assertion list for that file — they do not crash the batch.

        Args:
            files: List of ``(file_path, content)`` tuples.

        Returns:
            Dict mapping file_path to list of assertions.
        """
        if not files:
            return {}

        total = len(files)

        # Sequential path (max_workers == 1 or single file)
        if self._max_workers <= 1 or total == 1:
            result: Dict[str, List[Assertion]] = {}
            for idx, (file_path, content) in enumerate(files):
                result[file_path] = self.extract_from_file(file_path, content)
                if self._on_progress:
                    self._on_progress(file_path, idx + 1, total)
            return result

        # Parallel path
        result = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_path = {
                executor.submit(self.extract_from_file, fp, content): fp
                for fp, content in files
            }

            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                completed += 1
                try:
                    result[fp] = future.result()
                except Exception:
                    logger.warning(
                        "Assertion extraction failed for %s — returning "
                        "empty list", fp,
                    )
                    result[fp] = []

                if self._on_progress:
                    self._on_progress(fp, completed, total)

        return result

    @staticmethod
    def _parse_response(
        response: str,
        file_path: str,
    ) -> List[Assertion]:
        """Parse LLM response into Assertion objects.

        Handles common LLM quirks: think tags, markdown fences,
        extra text before/after JSON.

        Args:
            response: Raw LLM response text.
            file_path: File path to set on each assertion.

        Returns:
            List of ``Assertion`` objects.
        """
        cleaned = strip_think_tags(response)
        cleaned = strip_markdown_fences(cleaned)
        cleaned = cleaned.strip()

        parsed = None

        # Try direct parse
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array
        if parsed is None:
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    pass

        # Handle truncated JSON arrays (LLM hit token limit)
        if parsed is None:
            start = cleaned.find("[")
            if start >= 0:
                fragment = cleaned[start:]
                last_obj_end = fragment.rfind("}")
                if last_obj_end > 0:
                    candidate = fragment[:last_obj_end + 1].rstrip().rstrip(",") + "]"
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        pass

        if not isinstance(parsed, list):
            return []

        assertions: List[Assertion] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            assertion = Assertion.from_dict(item)
            assertion.file_path = file_path
            # Validate assertion_type
            if assertion.assertion_type not in ASSERTION_TYPES:
                assertion.assertion_type = "value"
            assertions.append(assertion)

        return assertions
