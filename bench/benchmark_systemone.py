#!/usr/bin/env python3
"""
Benchmark runner for native llama.cpp /v1/systemone endpoint.
Evaluates accuracy, probability calibration, and parallel scoring latency
against decision datasets (e.g. bench/clinc150.jsonl, bench/easy.jsonl).
"""

import argparse
import collections
import datetime
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Ensure UTF-8 output and line buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server /v1/systemone endpoint on decision datasets."
    )
    parser.add_argument(
        "--input", "-i",
        default="bench/easy.jsonl",
        help="Path to JSONL dataset (default: bench/easy.jsonl)."
    )
    parser.add_argument(
        "--url", "-u",
        default="http://127.0.0.1:8096",
        help="Base URL of llama-server (default: http://127.0.0.1:8096)."
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/systemone",
        help="SystemOne endpoint path (default: /v1/systemone)."
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of items to evaluate (default: all)."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of items to skip from start (default: 0)."
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="Randomly sample N items from dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)."
    )
    parser.add_argument(
        "--family", "-f",
        default=None,
        help="Filter items by family name."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save detailed JSON results."
    )
    parser.add_argument(
        "--candidates", "-c",
        default="symbols",
        choices=["symbols", "keys"],
        help="Candidate readout mode: 'symbols' (default: pure alpha letters/symbols) or 'keys' (direct enum strings)."
    )
    parser.add_argument(
        "--prompt-mode", "-p",
        default="raw",
        choices=["raw", "primed_letters"],
        help="Prompt formatting mode: 'raw' (default for intent classification) or 'primed_letters' (recency options in state for MMLU/choice benchmarks)."
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Prompt suffix override (server default: 'Answer: ')."
    )
    parser.add_argument(
        "--mode", "-m",
        default="auto",
        choices=["auto", "tree", "greedy"],
        help="Decision scoring mode: 'auto' (default), 'tree' (full trie expansion), or 'greedy' (fast top-1 path)."
    )
    parser.add_argument(
        "--resume", "--continue",
        dest="resume",
        default=None,
        help="Path to results JSON file to resume/continue after a stoppage or crash."
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=25,
        help="Interval of evaluated items to auto-save partial results to disk (default: 25)."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose details for every item."
    )
    return parser.parse_args()


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    s = round(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m:02d}m {sec:02d}s"


def normalize_dataset_item(item: dict[str, Any]) -> dict[str, Any]:
    norm = dict(item)
    state = norm.get("state")
    if isinstance(state, str) and len(state) >= 2 and state[0] == '"' and state[-1] == '"':
        try:
            unquoted = json.loads(state)
            if isinstance(unquoted, str):
                state = unquoted
        except Exception:
            state = state[1:-1]
        norm["state"] = state

    if "question" not in norm or not isinstance(norm["question"], dict):
        q_type = "choice"
        if "labels" in norm:
            labels = norm["labels"]
            if labels == ["no", "yes"] or labels == ["false", "true"]:
                q_type = "noul"
        criteria = {}
        if "choices" in norm and isinstance(norm["choices"], list):
            criteria = {c: c for c in norm["choices"]}
        elif "labels" in norm and isinstance(norm["labels"], list):
            criteria = {c: c for c in norm["labels"]}
        norm["question"] = {
            "type": q_type,
            "instructions": norm.get("instructions", "Select the best matching decision."),
            "criteria": criteria
        }
    return norm


def load_dataset(file_path: str, family_filter: str | None = None,
                 offset: int = 0, limit: int | None = None,
                 sample: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    items: list[dict[str, Any]] = []

    # Check if loading from previous benchmark results JSON
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "benchmark_items" in data and data["benchmark_items"]:
                items = [normalize_dataset_item(it) for it in data["benchmark_items"]]
                print(f"Loaded {len(items)} saved benchmark items from: {file_path}", flush=True)
            elif "items" in data and data["items"]:
                items = [normalize_dataset_item(it) for it in data["items"]]
                print(f"Loaded {len(items)} items from results file: {file_path}", flush=True)
            elif "results" in data and data.get("input_file"):
                cand_input = data["input_file"]
                sub_ids = set(r["id"] for r in data["results"] if "id" in r)
                if os.path.exists(cand_input):
                    items = load_dataset(cand_input)
                    items = [it for it in items if it.get("id") in sub_ids]
                else:
                    raise ValueError(f"Results file {file_path} has no embedded 'benchmark_items' and input {cand_input} was not found.")
            else:
                raise ValueError(f"Results file {file_path} does not contain 'benchmark_items'.")
        elif isinstance(data, list):
            items = [normalize_dataset_item(it) for it in data]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(normalize_dataset_item(item))
                except Exception as e:
                    print(f"[WARN] Failed to parse line {line_num}: {e}", file=sys.stderr)

    if family_filter:
        items = [item for item in items if item.get("family") == family_filter]

    if offset > 0:
        items = items[offset:]

    if sample and sample < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, sample)

    if limit is not None and limit > 0:
        items = items[:limit]

    return items


def send_systemone_request(server_url: str, endpoint: str, payload: dict[str, Any],
                            timeout: float = 120.0) -> tuple[dict[str, Any], dict[str, str], float]:
    url = f"{server_url.rstrip('/')}/{endpoint.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        resp_data = json.loads(response.read().decode("utf-8"))
        headers = {k.lower(): v for k, v in response.headers.items()}
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return resp_data, headers, wall_ms


def check_server_health(server_url: str) -> bool:
    health_url = f"{server_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def build_systemone_payload(
    item: dict[str, Any],
    candidates_mode: str = "symbols",
    suffix: str | None = None,
    mode: str = "auto",
    prompt_mode: str = "raw"
) -> tuple[dict[str, Any], str]:
    state = item.get("state", "")
    question = item.get("question", {})
    qid = "intent" if "intent" in item.get("family", "") else "decision"

    if prompt_mode == "primed_letters":
        instructions = question.get("instructions", "Which option is the correct answer?")
        criteria = question.get("criteria", {})
        labels = item.get("labels", [])
        if isinstance(criteria, dict):
            keys = list(criteria.keys())
            for lb in labels:
                if lb not in keys:
                    keys.append(lb)
        else:
            keys = list(labels)

        criteria_lines = []
        for i, k in enumerate(keys):
            sym = chr(ord('A') + i) if i < 26 else str(i)
            desc = ""
            if isinstance(criteria, dict) and k in criteria:
                v = criteria[k]
                desc = f": {v}" if v else ""
            criteria_lines.append(f"{sym}. {k}{desc}")

        ask_text = "Respond with only the letter of the best option."
        full_context = (
            f"# Question\n{instructions}\n\n"
            + (f"# Evidence\n{state}\n\n" if state else "")
            + f"# Criterion\n{instructions}\n\n"
            f"# Options\n" + "\n".join(criteria_lines) + f"\n\n{ask_text}"
        )

        payload = {
            "model": "jev-latest",
            "state": full_context,
            "preamble": "You are a System One decision model. Select the single option letter that best matches.",
            "candidates": "keys",
            "questions": {
                qid: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": {k: k for k in keys}
                }
            }
        }
    else:
        payload = {
            "model": "jev-latest",
            "state": state,
            "candidates": candidates_mode,
            "questions": {
                qid: question
            }
        }

    if suffix is not None:
        payload["suffix"] = suffix
    if mode != "auto":
        payload["mode"] = mode
    return payload, qid


def compute_summary_data(
    results: list[dict[str, Any]],
    total_items: int,
    total_elapsed: float,
    server_url: str,
    endpoint: str,
    input_file: str,
    candidates: str,
    prompt_mode: str,
    mode: str,
    timestamp: str | None = None,
    benchmark_items: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    valid_res = [r for r in results if "wall_ms" in r]
    latencies = [r["wall_ms"] for r in valid_res]
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    evaluated_count = len(results)
    overall_acc = (correct_count / evaluated_count) * 100.0 if evaluated_count > 0 else 0.0

    families = sorted({r.get("family", "default") for r in results})
    family_summary = {}
    for fam in families:
        fam_items = [r for r in results if r.get("family", "default") == fam]
        fam_total = len(fam_items)
        fam_correct = sum(1 for r in fam_items if r.get("is_correct", False))
        fam_acc = (fam_correct / fam_total) * 100.0 if fam_total > 0 else 0.0
        fam_probs = [r["probability"] for r in fam_items if "probability" in r and r["probability"] is not None]
        fam_latencies = [r["wall_ms"] for r in fam_items if "wall_ms" in r and r["wall_ms"] is not None]
        family_summary[fam] = {
            "total": fam_total,
            "correct": fam_correct,
            "accuracy_pct": fam_acc,
            "mean_prob": statistics.mean(fam_probs) if fam_probs else 0.0,
            "mean_latency_ms": statistics.mean(fam_latencies) if fam_latencies else 0.0
        }

    summary_data: dict[str, Any] = {
        "timestamp": timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "server_url": server_url,
        "endpoint": endpoint,
        "input_file": input_file,
        "candidates": candidates,
        "prompt_mode": prompt_mode,
        "mode": mode,
        "total_items": total_items,
        "evaluated_count": evaluated_count,
        "correct_count": correct_count,
        "overall_accuracy_pct": overall_acc,
        "total_elapsed_sec": total_elapsed,
        "family_breakdown": family_summary,
        "latency": {
            "mean_ms": statistics.mean(latencies) if latencies else 0.0,
            "median_ms": statistics.median(latencies) if latencies else 0.0,
            "min_ms": min(latencies) if latencies else 0.0,
            "max_ms": max(latencies) if latencies else 0.0
        },
        "results": results
    }
    if benchmark_items:
        summary_data["benchmark_items"] = benchmark_items
    return summary_data


def write_results_json(out_file: str, summary_data: dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.prompt_mode == "primed_letters" and args.candidates == "symbols":
        args.candidates = "keys"

    resumed_map: dict[str, dict[str, Any]] = {}
    prev_elapsed_sec = 0.0
    resume_data: dict[str, Any] | None = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume file not found: {args.resume}")
        with open(args.resume, "r", encoding="utf-8") as f:
            resume_data = json.load(f)

        # Infer input dataset if default was kept
        if args.input in ("bench/clinc150.jsonl", "bench\\clinc150.jsonl") and "input_file" in resume_data:
            cand = resume_data["input_file"]
            if os.path.exists(cand):
                args.input = cand
            else:
                base = os.path.basename(cand)
                in_bench = os.path.join("bench", base)
                in_res_dir = os.path.join(os.path.dirname(args.resume) or ".", base)
                if os.path.exists(in_bench):
                    args.input = in_bench
                elif os.path.exists(in_res_dir):
                    args.input = in_res_dir

        if "prompt_mode" in resume_data and args.prompt_mode == "raw":
            args.prompt_mode = resume_data["prompt_mode"]
        if "candidates" in resume_data and args.candidates == "symbols":
            args.candidates = resume_data["candidates"]
        if "mode" in resume_data and args.mode == "auto":
            args.mode = resume_data["mode"]
        if "endpoint" in resume_data and args.endpoint == "/v1/systemone":
            args.endpoint = resume_data["endpoint"]

        prev_elapsed_sec = float(resume_data.get("total_elapsed_sec", 0.0))
        for r in resume_data.get("results", []):
            if "id" in r and r.get("predicted") is not None:
                resumed_map[r["id"]] = r

    dataset_base = os.path.splitext(os.path.basename(args.input))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.dirname(args.input) if (os.path.dirname(args.input) and not any(ch in args.input for ch in "*?")) else "bench"
    if args.output:
        out_file = args.output
    elif args.resume:
        out_file = args.resume
    else:
        out_file = os.path.join(out_dir, f"results_systemone_{dataset_base}_{timestamp}.json")

    print("=" * 70)
    print("LLAMA.CPP /v1/systemone ENDPOINT BENCHMARK")
    print("=" * 70)
    print(f"Target Server : {args.url}")
    print(f"Endpoint      : {args.endpoint}")
    print(f"Dataset File  : {args.input}")
    if args.sample:
        print(f"Sample Size   : {args.sample} (seed={args.seed})")
    if args.limit:
        print(f"Item Limit    : {args.limit}")
    if args.offset:
        print(f"Offset        : {args.offset}")
    if args.family:
        print(f"Family Filter : {args.family}")
    print(f"Candidates    : {args.candidates}")
    print(f"Prompt Mode   : {args.prompt_mode}")
    print(f"Mode          : {args.mode}")
    if args.resume:
        print(f"Resume Source : {args.resume}")
    print(f"Output File   : {out_file}")

    print(f"Checking server health at {args.url}...", end=" ", flush=True)
    if check_server_health(args.url):
        print("[HEALTHY]", flush=True)
    else:
        print("[WARNING: /health endpoint did not respond with 200]", flush=True)

    if args.resume and resume_data and "benchmark_items" in resume_data and resume_data["benchmark_items"]:
        items = [normalize_dataset_item(it) for it in resume_data["benchmark_items"]]
        print(f"Loaded {len(items)} saved benchmark items from resume file: {args.resume}", flush=True)
    else:
        items = load_dataset(
            file_path=args.input,
            family_filter=args.family,
            offset=args.offset,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed
        )
    total_items = len(items)
    print(f"Loaded {total_items} benchmark items to evaluate.")

    resumed_count = sum(1 for it in items if it.get("id") in resumed_map)
    if resumed_count > 0:
        print(f"Resume State  : Found {resumed_count}/{total_items} items already evaluated in resume file.", flush=True)
        if resumed_count < total_items:
            print(f"Resuming live evaluation from item {resumed_count + 1}...", flush=True)
        else:
            print("All items in dataset are already evaluated in resume file.", flush=True)
    print("-" * 70)

    if total_items == 0:
        print("No items to evaluate.")
        return

    results: list[dict[str, Any]] = []
    correct_count = 0
    total_wall_ms = 0.0
    total_input_tokens = 0
    cached_tokens_list: list[int] = []
    family_stats = collections.defaultdict(lambda: {"total": 0, "correct": 0, "probs": [], "latencies": []})

    session_start_time = time.perf_counter()

    try:
        for idx, item in enumerate(items, 1):
            item_id = item.get("id", f"item_{idx}")
            family = item.get("family", "default")
            expected = item.get("expected")

            # Fast-forward already evaluated items
            if item_id in resumed_map:
                rec = resumed_map[item_id]
                results.append(rec)
                is_correct = rec.get("is_correct", False)
                if is_correct:
                    correct_count += 1
                fam_stat = family_stats[family]
                fam_stat["total"] += 1
                if is_correct:
                    fam_stat["correct"] += 1
                if "probability" in rec and rec["probability"] is not None:
                    fam_stat["probs"].append(rec["probability"])
                if "wall_ms" in rec and rec["wall_ms"] is not None:
                    fam_stat["latencies"].append(rec["wall_ms"])
                    total_wall_ms += rec["wall_ms"]
                if "input_tokens" in rec and rec["input_tokens"] is not None:
                    total_input_tokens += rec["input_tokens"]
                if "cached_tokens" in rec and rec["cached_tokens"] is not None:
                    cached_tokens_list.append(rec["cached_tokens"])
                continue

            payload, qid = build_systemone_payload(item, args.candidates, args.suffix, args.mode, args.prompt_mode)
            t_item_start = time.perf_counter()

            try:
                resp_data, headers, wall_ms = send_systemone_request(args.url, args.endpoint, payload)
                item_elapsed_ms = (time.perf_counter() - t_item_start) * 1000.0

                ans_obj = resp_data.get("answers", {}).get(qid, {})
                predicted = ans_obj.get("choice")
                if predicted is None:
                    if "noul" in ans_obj:
                        predicted = "yes" if ans_obj["noul"] >= 0.5 else "no"
                    elif "score" in ans_obj:
                        winner = ans_obj.get("winner")
                        if winner is not None and str(winner).strip() != "":
                            predicted = str(winner)
                        elif "probabilities" in ans_obj and isinstance(ans_obj["probabilities"], dict) and ans_obj["probabilities"]:
                            predicted = str(max(ans_obj["probabilities"].items(), key=lambda kv: kv[1])[0])
                        elif "score" in ans_obj and ans_obj["score"] is not None:
                            predicted = str(round(float(ans_obj["score"])))

                confidence = ans_obj.get("confidence", 0.0)
                probs = ans_obj.get("probabilities", {})
                pred_prob = probs.get(predicted, confidence) if isinstance(probs, dict) else confidence

                is_correct = (str(predicted).strip().lower() == str(expected).strip().lower())
                if is_correct:
                    correct_count += 1

                usage = resp_data.get("usage", {})
                in_tokens = usage.get("input_tokens", 0)
                total_input_tokens += in_tokens
                total_wall_ms += wall_ms

                cached_tok = int(headers.get("x-llamajev-cached-tokens", 0))
                cached_tokens_list.append(cached_tok)

                fam_stat = family_stats[family]
                fam_stat["total"] += 1
                if is_correct:
                    fam_stat["correct"] += 1
                fam_stat["probs"].append(pred_prob)
                fam_stat["latencies"].append(wall_ms)

                session_now = time.perf_counter()
                session_elapsed = session_now - session_start_time
                total_elapsed_now = prev_elapsed_sec + session_elapsed
                session_items_done = idx - resumed_count

                if session_items_done > 0 and idx < total_items:
                    rate = session_elapsed / session_items_done
                    items_left = total_items - idx
                    eta_str = format_duration(rate * items_left)
                elif idx == total_items:
                    eta_str = "00m 00s"
                else:
                    eta_str = "--:--"

                running_acc = (correct_count / idx) * 100.0
                status_str = "[PASS]" if is_correct else "[FAIL]"
                print(f"[{idx:03d}/{total_items:03d}] {status_str} {item_id} ({family}) | Acc: {running_acc:5.1f}% ({correct_count}/{idx}) | Elapsed: {format_duration(total_elapsed_now)} | ETA: {eta_str}")
                print(f"         Exp: {expected} | Pred: {predicted} | Conf: {confidence:.1%} (Prob: {pred_prob:.1%})")
                print(f"         Tokens: {in_tokens} | Cached: {cached_tok} | Wall: {wall_ms:.1f}ms")

                if args.verbose and isinstance(probs, dict) and probs:
                    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    print(f"         Top 3: {', '.join(f'{k}: {v:.3f}' for k, v in top3)}")

                results.append({
                    "id": item_id,
                    "family": family,
                    "expected": expected,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "probability": pred_prob,
                    "wall_ms": wall_ms,
                    "input_tokens": in_tokens,
                    "cached_tokens": cached_tok,
                    "probabilities": probs
                })

                # Periodic auto-save
                if out_file and args.save_interval > 0 and session_items_done > 0 and session_items_done % args.save_interval == 0:
                    partial_summary = compute_summary_data(
                        results=results,
                        total_items=total_items,
                        total_elapsed=total_elapsed_now,
                        server_url=args.url,
                        endpoint=args.endpoint,
                        input_file=args.input,
                        candidates=args.candidates,
                        prompt_mode=args.prompt_mode,
                        mode=args.mode,
                        timestamp=timestamp,
                        benchmark_items=items
                    )
                    write_results_json(out_file, partial_summary)

            except Exception as e:
                session_now = time.perf_counter()
                session_elapsed = session_now - session_start_time
                total_elapsed_now = prev_elapsed_sec + session_elapsed
                print(f"[{idx:03d}/{total_items:03d}] [ERR ] {item_id} ({family}) | Elapsed: {format_duration(total_elapsed_now)}")
                print(f"         Error: {e}")
                results.append({
                    "id": item_id,
                    "family": family,
                    "expected": expected,
                    "error": str(e),
                    "is_correct": False
                })

    except KeyboardInterrupt:
        session_now = time.perf_counter()
        total_elapsed_now = prev_elapsed_sec + (session_now - session_start_time)
        print("\n\n" + "=" * 70, flush=True)
        print(f"[STOPPED] Benchmark interrupted by user (Ctrl+C) at item {len(results)}/{total_items}.", flush=True)
        if results and out_file:
            partial_summary = compute_summary_data(
                results=results,
                total_items=total_items,
                total_elapsed=total_elapsed_now,
                server_url=args.url,
                endpoint=args.endpoint,
                input_file=args.input,
                candidates=args.candidates,
                prompt_mode=args.prompt_mode,
                mode=args.mode,
                timestamp=timestamp,
                benchmark_items=items
            )
            write_results_json(out_file, partial_summary)
            print(f"Partial results ({len(results)} items) saved to: {out_file}", flush=True)
            print("-" * 70, flush=True)
            print("To continue this benchmark run:")
            print(f"  python bench/benchmark_systemone.py --resume \"{out_file}\"", flush=True)
        print("=" * 70, flush=True)
        return

    session_now = time.perf_counter()
    total_bench_time = prev_elapsed_sec + (session_now - session_start_time)
    overall_acc = (correct_count / total_items) * 100.0 if total_items > 0 else 0.0

    valid_res = [r for r in results if "wall_ms" in r]
    latencies = [r["wall_ms"] for r in valid_res]
    correct_probs = [r["probability"] for r in valid_res if r.get("is_correct") and "probability" in r]
    incorrect_probs = [r["probability"] for r in valid_res if not r.get("is_correct") and "probability" in r]

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated       : {len(results)}/{total_items}")
    print(f"Overall Accuracy      : {correct_count}/{len(results)} ({overall_acc:.2f}%)" if len(results) > 0 else "Overall Accuracy      : 0.00%")
    print(f"Total Elapsed Time    : {total_bench_time:.1f} s")
    if total_wall_ms > 0:
        tok_sec = (total_input_tokens / (total_wall_ms / 1000.0))
        print(f"Aggregate Token Rate  : {tok_sec:.1f} tokens/second")
    if correct_probs:
        print(f"Mean Conf. (Correct)  : {statistics.mean(correct_probs):.1%}")
    if incorrect_probs:
        print(f"Mean Conf. (Incorrect): {statistics.mean(incorrect_probs):.1%}")

    print("\n--- Accuracy by Task Family ---")
    print(f"{'Family':<22} | {'Count':<5} | {'Correct':<7} | {'Accuracy':<10} | {'Mean Conf':<10} | {'Mean Latency':<12}")
    print("-" * 75)
    family_summary = {}
    for fam, s in family_stats.items():
        acc = (s["correct"] / s["total"]) * 100.0 if s["total"] > 0 else 0.0
        m_prob = statistics.mean(s["probs"]) if s["probs"] else 0.0
        m_lat = statistics.mean(s["latencies"]) if s["latencies"] else 0.0
        print(f"{fam:<22} | {s['total']:<5} | {s['correct']:<7} | {acc:6.1f}%    | {m_prob:8.1%}  | {m_lat:10.1f} ms")
        family_summary[fam] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy_pct": acc,
            "mean_prob": m_prob,
            "mean_latency_ms": m_lat
        }

    if latencies:
        print("\n--- Latency Breakdown (Wall Timings) ---")
        print(f"{'Metric':<18} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'Min (ms)':<9} | {'Max (ms)':<9}")
        print("-" * 65)
        print(f"{'Wall Latency':<18} | {statistics.mean(latencies):10.1f} | {statistics.median(latencies):12.1f} | {min(latencies):9.1f} | {max(latencies):9.1f}")

    failures = [r for r in results if not r.get("is_correct", False)]
    if failures:
        print(f"\n--- Incorrect Items ({len(failures)} total) ---")
        for fail_item in failures[:15]:
            pred = fail_item.get("predicted", "N/A")
            prob = fail_item.get("probability", 0.0)
            print(f"- [{fail_item['id']}] ({fail_item['family']}): Expected '{fail_item['expected']}', got '{pred}' (prob: {prob:.1%})")
        if len(failures) > 15:
            print(f"    ... and {len(failures) - 15} more failures saved to result file.")

    summary_data = compute_summary_data(
        results=results,
        total_items=total_items,
        total_elapsed=total_bench_time,
        server_url=args.url,
        endpoint=args.endpoint,
        input_file=args.input,
        candidates=args.candidates,
        prompt_mode=args.prompt_mode,
        mode=args.mode,
        timestamp=timestamp,
        benchmark_items=items
    )

    write_results_json(out_file, summary_data)
    print(f"\nDetailed benchmark results saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
