"""Unit tests for duplication detection logic."""

import json
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from source.duplication import (
    Chunk,
    DuplicateMatch,
    DuplicationDetector,
    DuplicationResult,
    FileGroup,
    FileGrouper,
    TopicResult,
    _severity_label,
)


class TestSeverityLabel(unittest.TestCase):
    """Tests for similarity → severity mapping."""

    def test_high_severity(self):
        self.assertEqual(_severity_label(0.96), "high")
        self.assertEqual(_severity_label(0.95), "high")
        self.assertEqual(_severity_label(1.0), "high")

    def test_medium_severity(self):
        self.assertEqual(_severity_label(0.90), "medium")
        self.assertEqual(_severity_label(0.85), "medium")

    def test_low_severity(self):
        self.assertEqual(_severity_label(0.80), "low")
        self.assertEqual(_severity_label(0.50), "low")


class TestChunk(unittest.TestCase):
    """Tests for the Chunk dataclass."""

    def test_chunk_id_generation(self):
        c1 = Chunk(file_path="a.md", content="hello", start_index=0)
        c2 = Chunk(file_path="a.md", content="hello", start_index=0)
        self.assertEqual(c1.chunk_id, c2.chunk_id)

    def test_different_chunks_different_ids(self):
        c1 = Chunk(file_path="a.md", content="hello", start_index=0)
        c2 = Chunk(file_path="b.md", content="hello", start_index=0)
        self.assertNotEqual(c1.chunk_id, c2.chunk_id)


class TestDuplicateMatch(unittest.TestCase):
    """Tests for DuplicateMatch auto-severity."""

    def test_auto_severity_high(self):
        m = DuplicateMatch(
            chunk_a_file="a.md",
            chunk_a_content="x",
            chunk_b_file="b.md",
            chunk_b_content="y",
            similarity=0.97,
        )
        self.assertEqual(m.severity, "high")

    def test_auto_severity_medium(self):
        m = DuplicateMatch(
            chunk_a_file="a.md",
            chunk_a_content="x",
            chunk_b_file="b.md",
            chunk_b_content="y",
            similarity=0.90,
        )
        self.assertEqual(m.severity, "medium")

    def test_to_dict(self):
        m = DuplicateMatch(
            chunk_a_file="a.md",
            chunk_a_content="x",
            chunk_b_file="b.md",
            chunk_b_content="y",
            similarity=0.95,
        )
        d = m.to_dict()
        self.assertIn("chunk_a", d)
        self.assertIn("chunk_b", d)
        self.assertEqual(d["similarity"], 0.95)


class TestDuplicationResult(unittest.TestCase):
    """Tests for DuplicationResult."""

    def test_empty_result(self):
        r = DuplicationResult()
        self.assertEqual(r.duplicate_matches, 0)
        self.assertEqual(r.duplicate_groups, 0)
        self.assertEqual(r.files_analysed, 0)

    def test_get_duplicate_file_pairs(self):
        r = DuplicationResult(
            matches=[
                DuplicateMatch("a.md", "x", "b.md", "y", 0.96),
                DuplicateMatch("c.md", "x", "d.md", "y", 0.80),
            ],
        )
        pairs = r.get_duplicate_file_pairs(threshold=0.85)
        self.assertEqual(len(pairs), 1)
        self.assertIn(frozenset(["a.md", "b.md"]), pairs)


class TestFileGrouper(unittest.TestCase):
    """Tests for FileGrouper topic → group logic."""

    def test_groups_files_by_shared_topics(self):
        topics = [
            TopicResult(file_path="a.md", topics=["setup", "docker"]),
            TopicResult(file_path="b.md", topics=["setup", "install"]),
            TopicResult(file_path="c.md", topics=["api"]),
        ]
        grouper = FileGrouper()
        groups = grouper.group(topics)

        # a.md and b.md share "setup" — should be in a group
        setup_groups = [
            g for g in groups if "setup" in g.shared_topics
        ]
        self.assertTrue(len(setup_groups) >= 1)
        setup_files = setup_groups[0].files
        self.assertIn("a.md", setup_files)
        self.assertIn("b.md", setup_files)


class TestTopicResult(unittest.TestCase):
    """Tests for TopicResult."""

    def test_to_dict(self):
        t = TopicResult(
            file_path="readme.md",
            topics=["overview"],
            entities=["Docker"],
        )
        d = t.to_dict()
        self.assertEqual(d["file"], "readme.md")
        self.assertEqual(d["topics"], ["overview"])
        self.assertEqual(d["entities"], ["Docker"])


if __name__ == "__main__":
    unittest.main()
