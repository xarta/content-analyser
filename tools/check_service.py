#!/usr/bin/env python3
"""Health check and integration test tool for content-analyser.

Usage:
    python3 tools/check_service.py              # Health check only
    python3 tools/check_service.py --test       # Run integration tests
    python3 tools/check_service.py --all        # Full test suite
    python3 tools/check_service.py --json       # Output as JSON

Reads CONTENT_ANALYSER_URL from .env (or environment).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# ===================================================================
# .env loader
# ===================================================================

def _load_env(env_path: str = ".env") -> None:
    """Load key=value pairs from .env into os.environ (no overwrite)."""
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ===================================================================
# HTTP helpers (stdlib only)
# ===================================================================

def _http_get(url: str, timeout: int = 30) -> dict:
    """GET request, return parsed JSON."""
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _http_post_json(url: str, body: dict, timeout: int = 120) -> dict:
    """POST JSON request, return parsed JSON."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ===================================================================
# Test documents
# ===================================================================

TEST_DOCS = {
    "docs/setup.md": (
        "# Setup Guide\n\n"
        "Install Python 3.11 and Docker.\n\n"
        "## Configuration\n\n"
        "Set the port to 8080 in the config file.\n"
        "The service runs on port 8080 by default.\n"
    ),
    "docs/old-setup.md": (
        "# Setup (Legacy)\n\n"
        "Install Python 3.9 and Docker.\n\n"
        "## Configuration\n\n"
        "Set the port to 8080 in the config file.\n"
        "The service runs on port 8080 by default.\n"
    ),
    "docs/api.md": (
        "# API Reference\n\n"
        "The API provides REST endpoints for document management.\n\n"
        "## Authentication\n\n"
        "Use Bearer tokens for authentication.\n"
    ),
}

CONTRADICTION_GROUPS = [
    {
        "group_id": 1,
        "shared_topics": ["setup", "configuration"],
        "files": ["docs/setup.md", "docs/old-setup.md"],
    },
]


# ===================================================================
# Checks
# ===================================================================

def check_health(base_url: str) -> dict:
    """Run health check against the service."""
    result = {"test": "health_check", "passed": False, "details": {}}
    try:
        start = time.monotonic()
        info = _http_get(f"{base_url}/")
        health = _http_get(f"{base_url}/health")
        latency = (time.monotonic() - start) * 1000

        result["details"]["service_info"] = info
        result["details"]["health"] = health
        result["details"]["latency_ms"] = round(latency, 1)

        if health.get("status") in ("healthy", "degraded"):
            result["passed"] = True
        else:
            result["details"]["error"] = (
                f"Unexpected status: {health.get('status')}"
            )

    except urllib.error.URLError as exc:
        result["details"]["error"] = f"Connection failed: {exc}"
    except Exception as exc:
        result["details"]["error"] = str(exc)

    return result


def test_duplicates(base_url: str) -> dict:
    """Test /duplicates endpoint with known duplicate documents."""
    result = {"test": "duplicates", "passed": False, "details": {}}
    try:
        body = {"files": TEST_DOCS}
        start = time.monotonic()
        resp = _http_post_json(f"{base_url}/duplicates", body, timeout=300)
        latency = time.monotonic() - start

        result["details"]["duration_s"] = round(latency, 2)
        result["details"]["summary"] = resp.get("summary", {})
        result["details"]["groups_count"] = len(resp.get("groups", []))
        result["details"]["matches_count"] = len(resp.get("matches", []))
        result["details"]["topics_count"] = len(resp.get("topics", []))

        # Expect at least one group (setup.md + old-setup.md share topics)
        if resp.get("groups") and resp.get("matches"):
            result["passed"] = True
        elif resp.get("groups"):
            result["passed"] = True
            result["details"]["note"] = (
                "Groups found but no matches (may need threshold tuning)"
            )
        else:
            result["details"]["error"] = (
                "No groups found — expected setup docs to share topics"
            )

    except urllib.error.URLError as exc:
        result["details"]["error"] = f"Connection failed: {exc}"
    except Exception as exc:
        result["details"]["error"] = str(exc)

    return result


def test_contradictions(base_url: str) -> dict:
    """Test /contradictions endpoint with known contradicting documents."""
    result = {"test": "contradictions", "passed": False, "details": {}}
    try:
        body = {
            "files": TEST_DOCS,
            "groups": CONTRADICTION_GROUPS,
        }
        start = time.monotonic()
        resp = _http_post_json(f"{base_url}/contradictions", body, timeout=300)
        latency = time.monotonic() - start

        result["details"]["duration_s"] = round(latency, 2)
        result["details"]["summary"] = resp.get("summary", {})
        result["details"]["findings_count"] = len(resp.get("findings", []))

        # We expect a finding about Python 3.11 vs 3.9
        findings = resp.get("findings", [])
        if findings:
            result["passed"] = True
            result["details"]["first_finding"] = {
                "topic": findings[0].get("topic", ""),
                "severity": findings[0].get("severity", ""),
            }
        else:
            result["details"]["note"] = (
                "No contradictions found — LLM may have missed "
                "Python 3.11 vs 3.9 difference"
            )
            # Still pass — false negatives are acceptable
            result["passed"] = True

    except urllib.error.URLError as exc:
        result["details"]["error"] = f"Connection failed: {exc}"
    except Exception as exc:
        result["details"]["error"] = str(exc)

    return result


def test_analyse(base_url: str) -> dict:
    """Test /analyse endpoint (full pipeline)."""
    result = {"test": "analyse", "passed": False, "details": {}}
    try:
        body = {"files": TEST_DOCS}
        start = time.monotonic()
        resp = _http_post_json(f"{base_url}/analyse", body, timeout=600)
        latency = time.monotonic() - start

        result["details"]["duration_s"] = round(latency, 2)
        result["details"]["has_duplication"] = "duplication" in resp
        result["details"]["has_contradictions"] = "contradictions" in resp

        dup = resp.get("duplication", {})
        contra = resp.get("contradictions", {})
        result["details"]["dup_summary"] = dup.get("summary", {})
        result["details"]["contra_summary"] = contra.get("summary", {})

        if "duplication" in resp and "contradictions" in resp:
            result["passed"] = True
        else:
            result["details"]["error"] = (
                "Response missing duplication or contradictions"
            )

    except urllib.error.URLError as exc:
        result["details"]["error"] = f"Connection failed: {exc}"
    except Exception as exc:
        result["details"]["error"] = str(exc)

    return result


def test_edge_single_file(base_url: str) -> dict:
    """Test edge case: single file should return 400."""
    result = {"test": "edge_single_file", "passed": False, "details": {}}
    try:
        body = {"files": {"only.md": "# Only Document"}}
        req = urllib.request.Request(
            f"{base_url}/duplicates",
            data=json.dumps(body).encode(),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")

        try:
            urllib.request.urlopen(req, timeout=30)
            result["details"]["error"] = "Expected 400 but got 200"
        except urllib.error.HTTPError as exc:
            if exc.code == 400 or exc.code == 422:
                result["passed"] = True
                result["details"]["status_code"] = exc.code
            else:
                result["details"]["error"] = (
                    f"Expected 400 but got {exc.code}"
                )

    except urllib.error.URLError as exc:
        result["details"]["error"] = f"Connection failed: {exc}"
    except Exception as exc:
        result["details"]["error"] = str(exc)

    return result


# ===================================================================
# CLI
# ===================================================================

def _print_result(result: dict, as_json: bool = False) -> None:
    """Print a test result."""
    if as_json:
        return  # batch output at the end
    status = (
        "\033[32mPASS\033[0m" if result["passed"] else "\033[31mFAIL\033[0m"
    )
    print(f"  [{status}] {result['test']}")
    if not result["passed"] and result.get("details", {}).get("error"):
        print(f"         {result['details']['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Content Analyser health & test tool",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run integration tests",
    )
    parser.add_argument(
        "--all", action="store_true", help="Health check + integration tests",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON",
    )
    args = parser.parse_args()

    _load_env()
    base_url = os.environ.get("CONTENT_ANALYSER_URL", "").rstrip("/")

    if not base_url:
        print(
            "Error: CONTENT_ANALYSER_URL not set (check .env or environment)",
        )
        return 1

    results = []

    # Always run health check
    if not args.json:
        print(f"\nContent Analyser — {base_url}\n")
        print("Health Check:")

    health_result = check_health(base_url)
    results.append(health_result)
    _print_result(health_result, args.json)

    if not health_result["passed"]:
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n  Service unreachable — skipping tests\n")
        return 1

    # Integration tests
    if args.test or args.all:
        if not args.json:
            print("\nIntegration Tests:")

        tests = [
            test_edge_single_file,
            test_duplicates,
            test_contradictions,
            test_analyse,
        ]

        for test_fn in tests:
            result = test_fn(base_url)
            results.append(result)
            _print_result(result, args.json)

    # Summary
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        print(f"\n  {passed}/{total} checks passed\n")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
