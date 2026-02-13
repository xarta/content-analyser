"""Unit tests for content-analyser FastAPI endpoints."""

import json
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestRootEndpoint(unittest.TestCase):
    """Tests for GET /."""

    def setUp(self):
        # Patch clients before importing app
        with patch("app.LLMConfig"), \
             patch("app.LLMClient"), \
             patch("app.EmbeddingConfig"), \
             patch("app.EmbeddingClient"), \
             patch("app.ChunkerConfig"), \
             patch("app.ChunkerClient"):
            from app import app
            self.client = TestClient(app)

    def test_root_returns_service_info(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["service"], "content-analyser")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["status"], "running")


class TestDuplicatesEndpoint(unittest.TestCase):
    """Tests for POST /duplicates."""

    def setUp(self):
        with patch("app.LLMConfig"), \
             patch("app.LLMClient"), \
             patch("app.EmbeddingConfig"), \
             patch("app.EmbeddingClient"), \
             patch("app.ChunkerConfig"), \
             patch("app.ChunkerClient"):
            from app import app
            self.client = TestClient(app)

    def test_rejects_single_file(self):
        resp = self.client.post(
            "/duplicates",
            json={"files": {"one.md": "content"}},
        )
        self.assertEqual(resp.status_code, 400)

    def test_rejects_empty_files(self):
        resp = self.client.post(
            "/duplicates",
            json={"files": {}},
        )
        # Pydantic might accept empty dict but endpoint checks len < 2
        self.assertIn(resp.status_code, [400, 422])


class TestContradictionsEndpoint(unittest.TestCase):
    """Tests for POST /contradictions."""

    def setUp(self):
        with patch("app.LLMConfig"), \
             patch("app.LLMClient"), \
             patch("app.EmbeddingConfig"), \
             patch("app.EmbeddingClient"), \
             patch("app.ChunkerConfig"), \
             patch("app.ChunkerClient"):
            from app import app
            self.client = TestClient(app)

    def test_rejects_single_file(self):
        resp = self.client.post(
            "/contradictions",
            json={
                "files": {"one.md": "content"},
                "groups": [],
            },
        )
        self.assertEqual(resp.status_code, 400)

    def test_rejects_missing_groups(self):
        resp = self.client.post(
            "/contradictions",
            json={"files": {"a.md": "x", "b.md": "y"}},
        )
        self.assertEqual(resp.status_code, 422)


class TestAnalyseEndpoint(unittest.TestCase):
    """Tests for POST /analyse."""

    def setUp(self):
        with patch("app.LLMConfig"), \
             patch("app.LLMClient"), \
             patch("app.EmbeddingConfig"), \
             patch("app.EmbeddingClient"), \
             patch("app.ChunkerConfig"), \
             patch("app.ChunkerClient"):
            from app import app
            self.client = TestClient(app)

    def test_rejects_single_file(self):
        resp = self.client.post(
            "/analyse",
            json={"files": {"one.md": "content"}},
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
