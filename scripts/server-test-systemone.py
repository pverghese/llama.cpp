#!/usr/bin/env python3
"""
Server test script for native /v1/systemone endpoint.

Tests parallel typed questions (choice, noul, score) against a running llama-server.
Uses only standard library (urllib.request, json) so it runs out-of-the-box anywhere.

Usage:
  python scripts/server-test-systemone.py
  python scripts/server-test-systemone.py --port 8096
  python scripts/server-test-systemone.py --host 127.0.0.1 --port 8080
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
WHITE = "\x1b[97m"


def _print(text="", end="\n"):
    sys.stdout.write(text + end)
    sys.stdout.flush()


def print_header(title: str):
    bar = "─" * 65
    _print(f"\n{BOLD}{CYAN}┌{bar}┐{RESET}")
    _print(f"{BOLD}{CYAN}│  {WHITE}{title}{CYAN}{' ' * max(0, 63 - len(title))}│{RESET}")
    _print(f"{BOLD}{CYAN}└{bar}┘{RESET}")


def http_post_json(url: str, payload: dict, timeout: float = 60.0) -> Tuple[int, dict, dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            headers = dict(resp.headers)
            body = json.loads(resp.read().decode("utf-8"))
            return status, body, headers
    except urllib.error.HTTPError as e:
        status = e.code
        headers = dict(e.headers)
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"error": str(e)}
        return status, body, headers


def http_get_json(url: str, timeout: float = 5.0) -> Tuple[int, dict]:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return 0, {"error": str(e)}


def run_test(name: str, fn) -> bool:
    _print(f"\n{BOLD}[RUNNING]{RESET} {name} ...")
    t0 = time.perf_counter()
    try:
        ok, msg = fn()
        dt = (time.perf_counter() - t0) * 1000
        if ok:
            _print(f"{GREEN}✓ PASS{RESET} {name} ({dt:.1f} ms) {DIM}{msg}{RESET}")
            return True
        else:
            _print(f"{RED}✗ FAIL{RESET} {name} ({dt:.1f} ms): {msg}")
            return False
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        _print(f"{RED}✗ ERROR{RESET} {name} ({dt:.1f} ms): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test llama-server /v1/systemone endpoint.")
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8096, fallback: 8080)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    args = parser.parse_args()

    # Discover port if not explicitly set
    ports_to_try = [args.port] if args.port else [8096, 8080]
    base_url = None
    for p in ports_to_try:
        url = f"http://{args.host}:{p}"
        status, data = http_get_json(f"{url}/health", timeout=2.0)
        if status == 200:
            base_url = url
            break

    if not base_url:
        target = f"http://{args.host}:{args.port or 8096}"
        print(f"{RED}Error:{RESET} Could not connect to llama-server at {target}/health")
        print(f"Make sure llama-server is running with --decision-seqs (e.g. --decision-seqs 16).")
        sys.exit(1)

    print_header(f"llama-server SystemOne (/v1/systemone) Test Suite: {base_url}")

    endpoint = f"{base_url}/v1/systemone"
    results = []

    # Test 1: Unambiguous facts
    def test_unambiguous() -> Tuple[bool, str]:
        payload = {
            "model": "jev-latest",
            "state": "The sky is blue. Paris is the capital of France. The number seven is odd.",
            "questions": {
                "paris": {
                    "type": "choice",
                    "instructions": "What is the capital of France?",
                    "criteria": {"berlin": "Berlin", "paris": "Paris", "rome": "Rome"}
                },
                "odd": {
                    "type": "noul",
                    "instructions": "Is seven an odd number?"
                },
                "sky": {
                    "type": "choice",
                    "instructions": "What colour is the sky, per the text?",
                    "criteria": {"red": None, "blue": None, "green": None}
                }
            }
        }
        status, data, headers = http_post_json(endpoint, payload, timeout=args.timeout)
        if status != 200:
            return False, f"HTTP {status}: {data}"
        ans = data.get("answers", {})
        if ans.get("paris", {}).get("choice") != "paris":
            return False, f"Expected paris, got {ans.get('paris')}"
        if ans.get("odd", {}).get("noul", 0) <= 0.5:
            return False, f"Expected odd.noul > 0.5, got {ans.get('odd')}"
        if ans.get("sky", {}).get("choice") != "blue":
            return False, f"Expected blue, got {ans.get('sky')}"
        cached = headers.get("x-llamajev-cached-tokens", "?")
        return True, f"cached_tokens={cached}"

    results.append(run_test("Unambiguous facts (choice, noul)", test_unambiguous))

    # Test 2: Score monotonicity
    def test_score() -> Tuple[bool, str]:
        rubric = ["Cosmetic", "Degraded but workaround exists", "Blocking, no workaround"]
        payload_low = {
            "model": "jev-latest",
            "state": "The export button is misaligned by a few pixels.",
            "questions": {
                "s": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": rubric}
            }
        }
        payload_high = {
            "model": "jev-latest",
            "state": "Every export fails and nobody can get their data out. Production is down.",
            "questions": {
                "s": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": rubric}
            }
        }
        status_low, data_low, _ = http_post_json(endpoint, payload_low, timeout=args.timeout)
        status_high, data_high, _ = http_post_json(endpoint, payload_high, timeout=args.timeout)
        if status_low != 200 or status_high != 200:
            return False, f"HTTP low={status_low}, high={status_high}"
        score_low = data_low["answers"]["s"]["score"]
        score_high = data_high["answers"]["s"]["score"]
        if score_low >= score_high:
            return False, f"Score monotonicity failed: low={score_low:.3f} >= high={score_high:.3f}"
        return True, f"low={score_low:.2f} < high={score_high:.2f}"

    results.append(run_test("Score monotonicity (rubric evaluation)", test_score))

    # Test 3: 26-way choice
    def test_twenty_six_way() -> Tuple[bool, str]:
        criteria = {f"opt{i}": f"The number {i}" for i in range(26)}
        criteria["opt20"] = "The answer to life, the universe and everything"
        payload = {
            "model": "jev-latest",
            "state": "Douglas Adams said the answer is forty-two.",
            "questions": {
                "q": {"type": "choice", "instructions": "Which option matches the text?", "criteria": criteria, "candidates": "keys"}
            }
        }
        status, data, _ = http_post_json(endpoint, payload, timeout=args.timeout)
        if status != 200:
            return False, f"HTTP {status}: {data}"
        ans = data["answers"]["q"]
        if ans["choice"] != "opt20":
            return False, f"Expected opt20, got {ans['choice']}"
        return True, f"choice={ans['choice']} conf={ans.get('confidence', 0):.2f}"

    results.append(run_test("26-way choice evaluation", test_twenty_six_way))

    # Test 4: 64-way choice well-formedness
    def test_sixty_four_way() -> Tuple[bool, str]:
        criteria = {f"opt{i}": f"The number {i}" for i in range(64)}
        criteria["opt42"] = "The answer to life, the universe and everything"
        payload = {
            "model": "jev-latest",
            "state": "Douglas Adams said the answer is forty-two.",
            "questions": {
                "q": {"type": "choice", "instructions": "Which option matches the text?", "criteria": criteria}
            }
        }
        status, data, _ = http_post_json(endpoint, payload, timeout=args.timeout)
        if status != 200:
            return False, f"HTTP {status}: {data}"
        ans = data["answers"]["q"]
        probs = ans["probabilities"]
        if len(probs) != 64:
            return False, f"Expected 64 probabilities, got {len(probs)}"
        prob_sum = sum(probs.values())
        if abs(prob_sum - 1.0) > 1e-4:
            return False, f"Probabilities do not sum to 1.0 (sum={prob_sum})"
        return True, f"64 options parsed, sum={prob_sum:.4f}"

    results.append(run_test("64-way choice well-formedness", test_sixty_four_way))

    # Test 5: Vision shapes test (if vision available)
    def test_vision() -> Tuple[bool, str]:
        status, props = http_get_json(f"{base_url}/props")
        has_vision = False
        if status == 200:
            has_vision = props.get("modalities", {}).get("vision", False) or "mmproj" in props.get("model_path", "")

        if not has_vision:
            return True, "[SKIPPED: server has no mmproj / vision modality]"

        img_path = Path(__file__).resolve().parent.parent / "tools" / "server" / "tests" / "assets" / "shapes-448.png"
        if not img_path.exists():
            return True, "[SKIPPED: test asset shapes-448.png not found]"

        b64 = base64.b64encode(img_path.read_bytes()).decode()
        payload = {
            "model": "jev-latest",
            "state": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Look at the image carefully."},
                ]
            }],
            "questions": {
                "red_shape": {
                    "type": "choice",
                    "instructions": "What shape is the red object?",
                    "criteria": {"circle": None, "square": None, "triangle": None, "none": "There is no red object"}
                },
                "count": {
                    "type": "choice",
                    "instructions": "How many distinct shapes are in the image?",
                    "criteria": {"1": None, "2": None, "3": None, "4": None, "5": None}
                },
                "blue_square": {"type": "noul", "instructions": "Is there a blue square in the image?"},
                "circle_quadrant": {
                    "type": "choice",
                    "instructions": "In which quadrant of the image is the circle?",
                    "criteria": {"top_left": None, "top_right": None, "bottom_left": None, "bottom_right": None}
                },
            }
        }
        status, data, _ = http_post_json(endpoint, payload, timeout=args.timeout)
        if status != 200:
            return False, f"HTTP {status}: {data}"
        ans = data["answers"]
        if ans["red_shape"]["choice"] != "circle":
            return False, f"Expected circle, got {ans['red_shape']['choice']}"
        if ans["count"]["choice"] != "3":
            return False, f"Expected 3, got {ans['count']['choice']}"
        if ans["blue_square"]["noul"] <= 0.5:
            return False, f"Expected blue_square.noul > 0.5, got {ans['blue_square']['noul']}"
        if ans["circle_quadrant"]["choice"] != "top_left":
            return False, f"Expected top_left, got {ans['circle_quadrant']['choice']}"
        return True, "4 vision questions answered correctly"

    results.append(run_test("Multimodal vision (/v1/systemone)", test_vision))

    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    print_header(f"Results: {passed}/{total} Passed")
    if passed == total:
        print(f"\n{GREEN}{BOLD}ALL TESTS PASSED!{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{RED}{BOLD}{total - passed} TESTS FAILED!{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
