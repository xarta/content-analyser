"""Unit tests for contradiction detection logic."""

import json
import unittest
from unittest.mock import MagicMock, patch

from source.contradictions import (
    Contradiction,
    ContradictionDetector,
    ContradictionResult,
    _assertion_overlap,
    _merge_overlapping_findings,
    _severity_from_confidence,
)


class TestSeverityFromConfidence(unittest.TestCase):
    """Tests for confidence → severity mapping."""

    def test_high_confidence(self):
        self.assertEqual(_severity_from_confidence(0.9), "high")
        self.assertEqual(_severity_from_confidence(0.8), "high")

    def test_medium_confidence(self):
        self.assertEqual(_severity_from_confidence(0.6), "medium")
        self.assertEqual(_severity_from_confidence(0.5), "medium")

    def test_low_confidence(self):
        self.assertEqual(_severity_from_confidence(0.3), "low")
        self.assertEqual(_severity_from_confidence(0.1), "low")


class TestContradiction(unittest.TestCase):
    """Tests for Contradiction dataclass."""

    def test_to_dict(self):
        c = Contradiction(
            topic="Python version",
            file1="a.md",
            excerpt1="Python 3.11",
            file2="b.md",
            excerpt2="Python 3.9",
            severity="high",
            explanation="Different Python versions",
            type="value",
            confidence=0.92,
        )
        d = c.to_dict()
        self.assertEqual(d["topic"], "Python version")
        self.assertEqual(d["file_a"], "a.md")
        self.assertEqual(d["file_b"], "b.md")
        self.assertEqual(d["confidence"], 0.92)

    def test_from_dict(self):
        data = {
            "topic": "Port conflict",
            "file1": "x.md",
            "excerpt1": "port 8080",
            "file2": "y.md",
            "excerpt2": "port 9090",
            "severity": "medium",
        }
        c = Contradiction.from_dict(data)
        self.assertEqual(c.topic, "Port conflict")
        self.assertEqual(c.file1, "x.md")

    def test_from_dict_with_api_keys(self):
        """Test from_dict with file_a/file_b API-style keys."""
        data = {
            "topic": "Port",
            "file_a": "x.md",
            "excerpt_a": "8080",
            "file_b": "y.md",
            "excerpt_b": "9090",
        }
        c = Contradiction.from_dict(data)
        self.assertEqual(c.file1, "x.md")
        self.assertEqual(c.excerpt1, "8080")


class TestContradictionResult(unittest.TestCase):
    """Tests for ContradictionResult."""

    def test_empty_result(self):
        r = ContradictionResult()
        d = r.to_dict()
        self.assertEqual(d["findings"], [])
        self.assertEqual(d["summary"]["contradictions_found"], 0)

    def test_with_findings(self):
        r = ContradictionResult(
            findings=[
                Contradiction(
                    topic="test",
                    file1="a.md",
                    excerpt1="x",
                    file2="b.md",
                    excerpt2="y",
                ),
            ],
            groups_analysed=3,
            assertions_extracted=20,
        )
        d = r.to_dict()
        self.assertEqual(len(d["findings"]), 1)
        self.assertEqual(d["summary"]["groups_analysed"], 3)
        self.assertEqual(d["summary"]["assertions_extracted"], 20)


class TestAssertionOverlap(unittest.TestCase):
    """Tests for _assertion_overlap helper."""

    def test_overlapping_findings(self):
        c1 = Contradiction(
            topic="Python version",
            file1="a.md",
            excerpt1="Install Python 3.11 for the service",
            file2="b.md",
            excerpt2="Install Python 3.9 for the service",
        )
        c2 = Contradiction(
            topic="Python requirement",
            file1="a.md",
            excerpt1="Install Python 3.11 for the application",
            file2="b.md",
            excerpt2="Requires Python 3.9",
        )
        self.assertTrue(_assertion_overlap(c1, c2))

    def test_non_overlapping_findings(self):
        c1 = Contradiction(
            topic="Port",
            file1="a.md",
            excerpt1="runs on port 8080",
            file2="b.md",
            excerpt2="runs on port 9090",
        )
        c2 = Contradiction(
            topic="Docker version",
            file1="a.md",
            excerpt1="Docker 24.0 required",
            file2="b.md",
            excerpt2="Docker 25.0 required",
        )
        self.assertFalse(_assertion_overlap(c1, c2))


class TestMergeOverlappingFindings(unittest.TestCase):
    """Tests for _merge_overlapping_findings."""

    def test_empty_list(self):
        self.assertEqual(_merge_overlapping_findings([]), [])

    def test_single_finding(self):
        c = Contradiction(
            topic="test", file1="a.md", excerpt1="x",
            file2="b.md", excerpt2="y",
        )
        result = _merge_overlapping_findings([c])
        self.assertEqual(len(result), 1)

    def test_merges_overlapping(self):
        c1 = Contradiction(
            topic="Python", file1="a.md",
            excerpt1="Python 3.11 required for the service",
            file2="b.md",
            excerpt2="Python 3.9 required for the service",
            explanation="short",
        )
        c2 = Contradiction(
            topic="Python version", file1="a.md",
            excerpt1="Python 3.11 required for the application",
            file2="b.md",
            excerpt2="Python 3.9 required for the application",
            explanation="longer explanation wins in merge",
        )
        result = _merge_overlapping_findings([c1, c2])
        self.assertEqual(len(result), 1)


class TestContradictionDetectorParseResponse(unittest.TestCase):
    """Tests for ContradictionDetector._parse_response."""

    def test_parse_valid_json_array(self):
        raw = json.dumps([{
            "topic": "Port",
            "file1": "a.md",
            "excerpt1": "8080",
            "file2": "b.md",
            "excerpt2": "9090",
            "severity": "high",
            "explanation": "Different ports",
            "type": "value",
        }])
        result = ContradictionDetector._parse_response(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].topic, "Port")

    def test_parse_empty_array(self):
        result = ContradictionDetector._parse_response("[]")
        self.assertEqual(len(result), 0)

    def test_parse_with_markdown_fences(self):
        raw = "```json\n[]\n```"
        result = ContradictionDetector._parse_response(raw)
        self.assertEqual(len(result), 0)

    def test_parse_with_think_tags(self):
        raw = "<think>reasoning</think>\n[]"
        result = ContradictionDetector._parse_response(raw)
        self.assertEqual(len(result), 0)

    def test_parse_embedded_array(self):
        raw = "Here are the results: [{\"topic\":\"test\",\"file1\":\"a\",\"excerpt1\":\"x\",\"file2\":\"b\",\"excerpt2\":\"y\"}] end"
        result = ContradictionDetector._parse_response(raw)
        self.assertEqual(len(result), 1)

    def test_parse_garbage(self):
        result = ContradictionDetector._parse_response("not json at all")
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
