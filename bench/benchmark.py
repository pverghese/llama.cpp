#!/usr/bin/env python3
"""
Benchmark runner for llama.cpp /v1/systemone endpoint.
Evaluates accuracy, probability calibration, and prefill/scoring latency
against decision benchmark datasets (e.g., bench/easy.jsonl, bench/hard.jsonl).
"""

import argparse
import collections
import datetime
import glob
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Ensure UTF-8 output and immediate line buffering on consoles/pipes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llama-server /v1/systemone on decision datasets."
    )
    parser.add_argument(
        "--input", "-i",
        default="bench/easy.jsonl",
        help="Path to JSONL dataset, glob pattern (e.g. 'bench/*.jsonl'), 'all', or previous results JSON (default: bench/easy.jsonl)."
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="Randomly sample N items from the dataset(s)."
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["random", "balanced"],
        default="random",
        help="Sampling strategy: 'random' (uniform across items) or 'balanced' (stratified across datasets)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling."
    )
    parser.add_argument(
        "--save-items",
        default=None,
        help="Optional path to save selected benchmark items to a JSONL file."
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
        "--family", "-f",
        default=None,
        help="Filter items by family name (e.g. 'intent', 'fact', 'long_policy')."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save detailed JSON results (default: bench/results_<dataset>_<timestamp>.json)."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print details for every single item."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-item progress lines; only print final summary."
    )
    parser.add_argument(
        "--prompt-mode", "-p",
        choices=["recency_letters", "primed_letters", "debiased_hybrid", "recency_enums", "legacy"],
        default="debiased_hybrid",
        help="Prompt mode: 'debiased_hybrid' (System 2 analytical prompt + primed question + recency options), 'primed_letters' (question primed before evidence, full options at end), 'recency_letters' (Reflex-style single letter + inlined criteria with recency), 'recency_enums' (raw enums + inlined criteria), or 'legacy' (system prompt)"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "tree", "greedy"],
        default="auto",
        help="Decision scoring mode: 'auto' (tree up to tree_max), 'tree' (exact trie scoring), or 'greedy' (fast single-path walking)."
    )
    parser.add_argument(
        "--compare-with",
        default=None,
        help="Path to previous results JSON to reuse items, compare in real-time, and print comparison report."
    )
    parser.add_argument(
        "--compare-url",
        default=None,
        help="Optional URL of second server to benchmark on same items immediately after first."
    )
    parser.add_argument(
        "--failed-from",
        default=None,
        help="Path to previous results JSON; benchmark only items that failed in that run."
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=1,
        help="Interval of items to print progress updates (default: 1)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunk long state text into N tokens with overlap (default: 0 = disabled, e.g. 512)."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Token overlap between consecutive chunks (default: 40)."
    )
    parser.add_argument(
        "--chunk-aggregation",
        choices=["max_confidence", "early_exit", "accumulated_prob"],
        default="max_confidence",
        help="Method to aggregate decisions across chunks (default: max_confidence)."
    )
    parser.add_argument(
        "--early-exit-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for early exit (default: 0.85)."
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


def compute_summary_data(
    results: list[dict[str, Any]],
    total_items: int,
    total_elapsed: float,
    server_url: str,
    input_file: str,
    prompt_mode: str,
    decision_mode: str = "auto",
    timestamp: str | None = None,
    benchmark_items: list[dict[str, Any]] | None = None,
    comparison_data: dict[str, Any] | None = None,
    compare_with: str | None = None
) -> dict[str, Any]:
    valid_results = [r for r in results if r.get("predicted") is not None]
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    evaluated_count = len(results)
    overall_acc = (correct_count / evaluated_count) * 100.0 if evaluated_count > 0 else 0.0

    prefill_times = [r["prefill_ms"] for r in valid_results if "prefill_ms" in r]
    scoring_times = [r["scoring_ms"] for r in valid_results if "scoring_ms" in r]
    total_times = [r["total_server_ms"] for r in valid_results if "total_server_ms" in r]
    tokens_list = [r["context_tokens"] for r in valid_results if "context_tokens" in r]
    cached_list = [r.get("cached_tokens", 0) for r in valid_results]
    prompt_list = [r.get("prompt_tokens", 0) for r in valid_results]

    total_tokens_prefilled = sum(tokens_list)
    total_cached_tokens = sum(cached_list)
    total_prompt_tokens = sum(prompt_list)
    cache_hit_pct = (total_cached_tokens / total_prompt_tokens * 100.0) if total_prompt_tokens > 0 else 0.0
    sum_prefill_sec = sum(prefill_times) / 1000.0 if prefill_times else 0.0
    prefill_tok_per_sec = total_tokens_prefilled / sum_prefill_sec if sum_prefill_sec > 0 else 0.0

    families = sorted({r["family"] for r in results})
    family_stats = {}
    for fam in families:
        fam_items = [r for r in results if r["family"] == fam]
        fam_correct = sum(1 for r in fam_items if r.get("is_correct", False))
        fam_total = len(fam_items)
        fam_acc = (fam_correct / fam_total) * 100.0 if fam_total > 0 else 0.0
        fam_probs = [r["probability"] for r in fam_items if "probability" in r]
        fam_mean_prob = statistics.mean(fam_probs) if fam_probs else 0.0
        fam_prefill = [r["prefill_ms"] for r in fam_items if "prefill_ms" in r]
        fam_mean_prefill = statistics.mean(fam_prefill) if fam_prefill else 0.0
        family_stats[fam] = {
            "total": fam_total,
            "correct": fam_correct,
            "accuracy_pct": fam_acc,
            "mean_prob": fam_mean_prob,
            "mean_prefill_ms": fam_mean_prefill
        }

    summary: dict[str, Any] = {
        "timestamp": timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "server_url": server_url,
        "input_file": input_file,
        "prompt_mode": prompt_mode,
        "decision_mode": decision_mode,
        "total_items": total_items,
        "evaluated_count": evaluated_count,
        "correct_count": correct_count,
        "overall_accuracy_pct": overall_acc,
        "total_elapsed_sec": total_elapsed,
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "total_cached_tokens": total_cached_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "cache_hit_pct": cache_hit_pct,
        "family_breakdown": family_stats,
        "latency": {
            "mean_prefill_ms": statistics.mean(prefill_times) if prefill_times else 0,
            "median_prefill_ms": statistics.median(prefill_times) if prefill_times else 0,
            "mean_scoring_ms": statistics.mean(scoring_times) if scoring_times else 0,
            "median_scoring_ms": statistics.median(scoring_times) if scoring_times else 0,
            "mean_total_ms": statistics.mean(total_times) if total_times else 0
        },
        "results": results
    }
    if benchmark_items:
        summary["benchmark_items"] = benchmark_items
    if comparison_data:
        summary["comparison"] = comparison_data
    if compare_with:
        summary["compare_with"] = compare_with
    return summary


def write_results_json(out_file: str, summary_data: dict[str, Any]) -> None:
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)


def chunk_text_approx(text: str, chunk_tokens: int = 512, overlap_tokens: int = 40) -> list[str]:
    if not text or chunk_tokens <= 0:
        return [text]

    chunk_chars = chunk_tokens * 4
    overlap_chars = overlap_tokens * 4
    step_chars = max(1, chunk_chars - overlap_chars)

    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_chars, text_len)
        if end < text_len:
            nl_pos = text.rfind("\n", start + step_chars, end)
            if nl_pos != -1 and nl_pos > start:
                end = nl_pos + 1
            else:
                dot_pos = text.rfind(". ", start + step_chars, end)
                if dot_pos != -1 and dot_pos > start:
                    end = dot_pos + 2

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(start + 1, end - overlap_chars)

    return chunks if chunks else [text]


def normalize_dataset_item(item: dict[str, Any]) -> dict[str, Any]:
    raw_q = item.get("question")
    if isinstance(raw_q, str):
        if raw_q.strip().startswith("{") and raw_q.strip().endswith("}"):
            try:
                parsed_q = json.loads(raw_q)
                q_obj = parsed_q if isinstance(parsed_q, dict) else {"instructions": raw_q.strip(), "type": "choice"}
            except Exception:
                q_obj = {"instructions": raw_q.strip(), "type": "choice"}
        else:
            q_obj = {"instructions": raw_q.strip(), "type": "choice"}
    elif isinstance(raw_q, dict):
        q_obj = dict(raw_q)
    else:
        q_obj = {"instructions": "Select the best matching decision.", "type": "choice"}

    options = item.get("options")
    labels = item.get("labels")
    if labels is None and isinstance(options, list):
        labels = []
        criteria = q_obj.get("criteria", {}) if isinstance(q_obj.get("criteria"), dict) else {}
        for i, opt in enumerate(options):
            if isinstance(opt, dict):
                opt_id = str(opt.get("id", i))
                labels.append(opt_id)
                if "description" in opt:
                    criteria[opt_id] = str(opt["description"])
                elif "text" in opt:
                    criteria[opt_id] = str(opt["text"])
            else:
                sym = str(opt)
                labels.append(sym)
                criteria[sym] = sym
        q_obj["criteria"] = criteria
        item["labels"] = labels

    if item.get("expected") is None:
        raw_label = item.get("label") if item.get("label") is not None else item.get("answer")
        if raw_label is None:
            raw_label = item.get("answerKey")

        if isinstance(raw_label, int):
            if isinstance(options, list) and 0 <= raw_label < len(options):
                opt = options[raw_label]
                item["expected"] = str(opt.get("id", raw_label)) if isinstance(opt, dict) else str(opt)
            elif labels and 0 <= raw_label < len(labels):
                item["expected"] = str(labels[raw_label])
            else:
                item["expected"] = str(raw_label)
        elif raw_label is not None:
            raw_str = str(raw_label).strip()
            if raw_str.isdigit() and labels and not str(labels[0]).isdigit() and int(raw_str) < len(labels):
                item["expected"] = str(labels[int(raw_str)])
            else:
                item["expected"] = raw_str

    item["question"] = q_obj
    return item


def save_items_jsonl(file_path: str, items: list[dict[str, Any]]) -> None:
    out_dir = os.path.dirname(file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_dataset(
    file_path: str,
    family_filter: str | None = None,
    offset: int = 0,
    limit: int | None = None,
    filter_ids: set[str] | list[str] | None = None,
    sample: int | None = None,
    sample_strategy: str = "random",
    seed: int | None = None
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # Check if loading from previous benchmark results JSON
    if file_path.endswith(".json") and os.path.exists(file_path):
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
                    items = load_dataset(cand_input, filter_ids=sub_ids)
                else:
                    raise ValueError(f"Results file {file_path} has no embedded 'benchmark_items' and input {cand_input} was not found.")
            else:
                raise ValueError(f"Results file {file_path} does not contain 'benchmark_items'.")
        elif isinstance(data, list):
            items = [normalize_dataset_item(it) for it in data]

    # Check if loading all datasets or wildcard glob
    elif file_path.lower() in ("all", "bench", "bench/*.jsonl") or any(ch in file_path for ch in "*?"):
        bench_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.exists(os.path.dirname(os.path.abspath(__file__))) else "bench"
        pattern = os.path.join(bench_dir, "*.jsonl") if file_path.lower() in ("all", "bench") else file_path
        matched_files = sorted(glob.glob(pattern))
        loaded_files = 0
        for fpath in matched_files:
            if os.path.basename(fpath).startswith("results") or os.path.getsize(fpath) == 0:
                continue
            f_items: list[dict[str, Any]] = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        norm = normalize_dataset_item(obj)
                        if norm.get("labels") and norm.get("expected") is not None and (norm.get("state") or norm.get("question")):
                            f_items.append(norm)
                    except Exception:
                        pass
            if f_items:
                items.extend(f_items)
                loaded_files += 1
        print(f"Loaded {len(items)} total items across {loaded_files} benchmark files ({pattern}).", flush=True)

    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(normalize_dataset_item(obj))
                except Exception as e:
                    print(f"[WARN] Failed to parse line {line_num}: {e}", file=sys.stderr)

    if filter_ids is not None:
        items = [item for item in items if item.get("id") in filter_ids]

    if family_filter:
        items = [item for item in items if item.get("family") == family_filter]

    if offset > 0:
        items = items[offset:]

    # Random sampling across items
    if sample is not None and sample > 0 and len(items) > sample:
        rng = random.Random(seed) if seed is not None else random.Random()
        if sample_strategy == "balanced":
            by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for it in items:
                grp = it.get("family") or it.get("provenance", {}).get("dataset") or "default"
                by_group[grp].append(it)
            sampled: list[dict[str, Any]] = []
            groups = list(by_group.keys())
            rng.shuffle(groups)
            per_group = max(1, sample // len(groups))
            for g in groups:
                grp_items = by_group[g]
                take = min(len(grp_items), per_group)
                sampled.extend(rng.sample(grp_items, take))
            if len(sampled) < sample and len(items) > len(sampled):
                sampled_ids = {id(x) for x in sampled}
                remaining = [x for x in items if id(x) not in sampled_ids]
                sampled.extend(rng.sample(remaining, min(sample - len(sampled), len(remaining))))
            elif len(sampled) > sample:
                sampled = rng.sample(sampled, sample)
            items = sampled
        else:
            items = rng.sample(items, sample)

    if limit is not None and limit > 0:
        items = items[:limit]

    return items


def get_choice_symbols(max_count: int = 255) -> list[str]:
    symbols = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    symbols += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    symbols += [chr(c) for c in range(ord("0"), ord("9") + 1)]
    for c in range(0x21, 0x7F):
        ch = chr(c)
        if ch not in symbols:
            symbols.append(ch)
    for c in range(0xA1, 0x100):
        symbols.append(chr(c))
    c = 0x0100
    while len(symbols) < max_count:
        symbols.append(chr(c))
        c += 1
    return symbols[:max_count]


def build_decision_payload(
    item: dict[str, Any],
    prompt_mode: str = "recency_letters",
    state_override: str | None = None,
    mode: str = "auto"
) -> tuple[dict[str, Any], dict[str, str] | None]:
    item = normalize_dataset_item(item)
    question_obj = item.get("question", {})
    raw_instructions = question_obj.get("instructions", "Select the best matching decision.")
    criteria = question_obj.get("criteria", {})
    q_type = question_obj.get("type", "choice")
    labels = item.get("labels", [])
    if not labels:
        if q_type == "noul":
            labels = ["no", "yes"]
        elif q_type == "score" and isinstance(criteria, list):
            labels = [str(i) for i in range(len(criteria))]
        elif isinstance(criteria, dict) and criteria:
            labels = list(criteria.keys())
        else:
            raise ValueError(f"Item {item.get('id')} has no labels/choices.")

    if len(labels) > 255:
        raise ValueError(
            f"Item {item.get('id', 'unknown')} has {len(labels)} choices. "
            "The llama.cpp decision engine supports at most 255 choices."
        )

    if state_override is not None:
        state_text = state_override
    else:
        raw_state = item.get("state", "")
        if isinstance(raw_state, (dict, list)):
            state_text = json.dumps(raw_state, ensure_ascii=False, indent=2)
        elif raw_state is None:
            state_text = ""
        else:
            state_text = str(raw_state)

    symbols = get_choice_symbols(255)

    if prompt_mode == "recency_letters":
        letter_map = {}
        opts_lines = []

        if q_type == "noul":
            t = criteria.get("true", "The statement is true.") if isinstance(criteria, dict) else "The statement is true."
            f = criteria.get("false", "The statement is false.") if isinstance(criteria, dict) else "The statement is false."
            opts_lines = [
                f"A. yes: {t}",
                f"B. no: {f}"
            ]
            letter_map = {"A": "yes", "B": "no"}
            ask_text = "Respond with only the letter of the best option."

        elif q_type == "score":
            if isinstance(criteria, list):
                n_levels = len(criteria) - 1
                for i, desc in enumerate(criteria):
                    sym = symbols[i]
                    key = str(labels[i]) if i < len(labels) else str(i)
                    letter_map[sym] = key
                    opts_lines.append(f"{sym}. (level {i} of {n_levels}) {desc}")
            else:
                for i, lb in enumerate(labels):
                    sym = symbols[i]
                    letter_map[sym] = str(lb)
                    desc = criteria.get(lb, "") if isinstance(criteria, dict) else ""
                    full_desc = f"{lb}: {desc}" if desc else str(lb)
                    opts_lines.append(f"{sym}. {full_desc}")
            ask_text = (
                "Respond with only the symbol of the level that best matches."
                if len(letter_map) > 26 else
                "Respond with only the letter of the level that best matches."
            )

        else:
            # Canonical ordering: use criteria.keys() if available, matching Reflex branch layout
            if isinstance(criteria, dict):
                keys = list(criteria.keys())
                for lb in labels:
                    if lb not in keys:
                        keys.append(lb)
            else:
                keys = list(labels)

            if len(keys) > 255:
                raise ValueError(
                    f"Item {item.get('id', 'unknown')} has {len(keys)} choices. "
                    "The llama.cpp decision engine supports at most 255 choices."
                )

            for i, k in enumerate(keys):
                sym = symbols[i]
                letter_map[sym] = str(k)
                desc = ""
                if isinstance(criteria, dict) and k in criteria:
                    v = criteria[k]
                    desc = f": {v}" if v else ""
                elif isinstance(criteria, list) and i < len(criteria):
                    desc = f": {criteria[i]}"
                opts_lines.append(f"{sym}. {k}{desc}")
            ask_text = (
                "Respond with only the symbol of the best option."
                if len(keys) > 26 else
                "Respond with only the letter of the best option."
            )

        evidence_header = "# Evidence\n" if state_text else ""
        full_context = (
            f"{evidence_header}{state_text}\n\n"
            f"# Criterion\n{raw_instructions}\n\n"
            f"# Options\n" + "\n".join(opts_lines) + f"\n\n{ask_text}"
        )
        is_large = len(letter_map) > 26
        target_name = "symbol" if is_large else "letter"
        payload: dict[str, Any] = {
            "contexts": [full_context],
            "instructions": f"You are a System One decision model. Select the single option {target_name} that best matches.",
            "schema": {
                "decision": {
                    "type": "enum",
                    "description": f"The {target_name} of the best option",
                    "choices": list(letter_map.keys())
                }
            }
        }
        if mode != "auto":
            payload["mode"] = mode
        if mode != "greedy" and len(letter_map) > 128:
            payload["tree_max"] = 255
        return payload, letter_map

    elif prompt_mode in ("primed_letters", "debiased_hybrid"):
        letter_map = {}
        criteria_lines = []
        options_lines = []

        if q_type == "noul":
            t = criteria.get("true", "The statement is true.") if isinstance(criteria, dict) else "The statement is true."
            f = criteria.get("false", "The statement is false.") if isinstance(criteria, dict) else "The statement is false."
            criteria_lines = [
                f"A. yes: {t}",
                f"B. no: {f}"
            ]
            options_lines = [
                "A. yes",
                "B. no"
            ]
            letter_map = {"A": "yes", "B": "no"}
            ask_text = "Respond with only the letter of the best option."

        elif q_type == "score":
            if isinstance(criteria, list):
                n_levels = len(criteria) - 1
                for i, desc in enumerate(criteria):
                    sym = symbols[i]
                    key = str(labels[i]) if i < len(labels) else str(i)
                    letter_map[sym] = key
                    criteria_lines.append(f"{sym}. (level {i} of {n_levels}) {desc}")
                    options_lines.append(f"{sym}. (level {i})")
            else:
                for i, lb in enumerate(labels):
                    sym = symbols[i]
                    letter_map[sym] = str(lb)
                    desc = criteria.get(lb, "") if isinstance(criteria, dict) else ""
                    full_desc = f"{lb}: {desc}" if desc else str(lb)
                    criteria_lines.append(f"{sym}. {full_desc}")
                    options_lines.append(f"{sym}. {lb}")
            ask_text = (
                "Respond with only the symbol of the level that best matches."
                if len(letter_map) > 26 else
                "Respond with only the letter of the level that best matches."
            )

        else:
            # Canonical ordering: use criteria.keys() if available, matching Reflex branch layout
            if isinstance(criteria, dict):
                keys = list(criteria.keys())
                for lb in labels:
                    if lb not in keys:
                        keys.append(lb)
            else:
                keys = list(labels)

            if len(keys) > 255:
                raise ValueError(
                    f"Item {item.get('id', 'unknown')} has {len(keys)} choices. "
                    "The llama.cpp decision engine supports at most 255 choices."
                )

            for i, k in enumerate(keys):
                sym = symbols[i]
                letter_map[sym] = str(k)
                desc = ""
                if isinstance(criteria, dict) and k in criteria:
                    v = criteria[k]
                    desc = f": {v}" if v else ""
                elif isinstance(criteria, list) and i < len(criteria):
                    desc = f": {criteria[i]}"
                criteria_lines.append(f"{sym}. {k}{desc}")
                options_lines.append(f"{sym}. {k}")
            ask_text = (
                "Respond with only the symbol of the best option."
                if len(keys) > 26 else
                "Respond with only the letter of the best option."
            )

        question_section = f"# Question\n{raw_instructions}\n\n"
        evidence_section = f"# Evidence\n{state_text}\n\n" if state_text else ""
        options_section = (
            f"# Criterion\n{raw_instructions}\n\n"
            f"# Options\n" + "\n".join(criteria_lines) + f"\n\n{ask_text}"
        )

        full_context = question_section + evidence_section + options_section

        is_large = len(letter_map) > 26
        target_name = "symbol" if is_large else "letter"
        if prompt_mode == "debiased_hybrid":
            instructions_text = (
                "You are an analytical evaluator. Carefully verify exceptions, dates, amendments, definitions, and exact arithmetic. "
                "Reject informal notes, trainee impressions, or surface answers. "
                f"Select the single option {target_name} that strictly satisfies all requirements."
            )
            desc_text = f"The {target_name} of the strictly verified best option"
        else:
            instructions_text = f"You are a System One decision model. Select the single option {target_name} that best matches."
            desc_text = f"The {target_name} of the best option"

        payload = {
            "contexts": [full_context],
            "instructions": instructions_text,
            "schema": {
                "decision": {
                    "type": "enum",
                    "description": desc_text,
                    "choices": list(letter_map.keys())
                }
            }
        }
        if mode != "auto":
            payload["mode"] = mode
        if mode != "greedy" and len(letter_map) > 128:
            payload["tree_max"] = 255
        return payload, letter_map

    elif prompt_mode == "recency_enums":
        opts_lines = []
        for i, lb in enumerate(labels):
            desc = ""
            if isinstance(criteria, dict) and lb in criteria:
                desc = f": {criteria[lb]}"
            elif isinstance(criteria, list) and i < len(criteria):
                desc = f": {criteria[i]}"
            opts_lines.append(f"- {lb}{desc}")

        evidence_header = "# Evidence\n" if state_text else ""
        full_context = (
            f"{evidence_header}{state_text}\n\n"
            f"# Criterion\n{raw_instructions}\n\n"
            f"# Options\n" + "\n".join(opts_lines) + "\n\n"
            "Select the best matching decision from the options."
        )
        payload = {
            "contexts": [full_context],
            "instructions": "You are a System One decision model. Select the requested field value from the allowed options.",
            "schema": {
                "decision": {
                    "type": "enum",
                    "description": raw_instructions,
                    "choices": labels
                }
            }
        }
        if mode != "auto":
            payload["mode"] = mode
        if mode != "greedy" and len(labels) > 128:
            payload["tree_max"] = 255
        return payload, None

    else: # legacy
        instructions_text = raw_instructions
        if criteria:
            if isinstance(criteria, dict):
                crit_lines = [f"- {k}: {v}" for k, v in criteria.items()]
                instructions_text += "\n\nCriteria and Definitions:\n" + "\n".join(crit_lines)
            elif isinstance(criteria, list):
                crit_lines = [f"- {v}" for v in criteria]
                instructions_text += "\n\nCriteria and Definitions:\n" + "\n".join(crit_lines)
            elif isinstance(criteria, str):
                instructions_text += "\n\nCriteria and Definitions:\n" + criteria

        payload = {
            "contexts": [state_text],
            "instructions": instructions_text,
            "schema": {
                "decision": {
                    "type": "enum",
                    "description": raw_instructions,
                    "choices": labels
                }
            }
        }
        if mode != "auto":
            payload["mode"] = mode
        if mode != "greedy" and len(labels) > 128:
            payload["tree_max"] = 255
        return payload, None


def send_decision_request(server_url: str, endpoint: str, payload: dict[str, Any], timeout: float = 180.0) -> tuple[dict[str, Any], float]:
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
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return resp_data, wall_ms


def check_server_health(server_url: str) -> bool:
    health_url = f"{server_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return bool(isinstance(data, dict) and data.get("status") == "ok")
    except Exception:
        return False


def generate_comparison_report(
    curr_summary: dict[str, Any],
    prev_summary: dict[str, Any],
    curr_label: str = "Candidate (Run 2)",
    prev_label: str = "Baseline (Run 1)"
) -> dict[str, Any]:
    curr_results = curr_summary.get("results", [])
    prev_results = prev_summary.get("results", [])

    prev_map = {r["id"]: r for r in prev_results if "id" in r}
    curr_map = {r["id"]: r for r in curr_results if "id" in r}

    common_ids = [r["id"] for r in curr_results if r["id"] in prev_map]
    total_common = len(common_ids)

    # Accuracy
    prev_correct = sum(1 for cid in common_ids if prev_map[cid].get("is_correct"))
    curr_correct = sum(1 for cid in common_ids if curr_map[cid].get("is_correct"))
    prev_acc = (prev_correct / total_common * 100.0) if total_common > 0 else 0.0
    curr_acc = (curr_correct / total_common * 100.0) if total_common > 0 else 0.0
    acc_diff = curr_acc - prev_acc

    # Agreement and discordance
    both_correct = [cid for cid in common_ids if prev_map[cid].get("is_correct") and curr_map[cid].get("is_correct")]
    both_incorrect = [cid for cid in common_ids if not prev_map[cid].get("is_correct") and not curr_map[cid].get("is_correct")]
    recovered = [cid for cid in common_ids if not prev_map[cid].get("is_correct") and curr_map[cid].get("is_correct")]
    regressed = [cid for cid in common_ids if prev_map[cid].get("is_correct") and not curr_map[cid].get("is_correct")]
    agreed = [
        cid for cid in common_ids
        if str(prev_map[cid].get("predicted")).strip().lower() == str(curr_map[cid].get("predicted")).strip().lower()
    ]

    def calc_lat(res_list: list[dict[str, Any]], key: str) -> dict[str, float]:
        vals = [r[key] for r in res_list if key in r and r[key] is not None]
        return {
            "mean": statistics.mean(vals) if vals else 0.0,
            "median": statistics.median(vals) if vals else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }

    curr_common_res = [curr_map[cid] for cid in common_ids]
    prev_common_res = [prev_map[cid] for cid in common_ids]

    curr_prefill = calc_lat(curr_common_res, "prefill_ms")
    prev_prefill = calc_lat(prev_common_res, "prefill_ms")
    curr_scoring = calc_lat(curr_common_res, "scoring_ms")
    prev_scoring = calc_lat(prev_common_res, "scoring_ms")
    curr_total = calc_lat(curr_common_res, "total_server_ms")
    prev_total = calc_lat(prev_common_res, "total_server_ms")

    def speedup_str(base: float, cand: float) -> str:
        if cand <= 0:
            return "N/A"
        ratio = base / cand
        diff = cand - base
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:6.1f} ms ({ratio:5.2f}x)"

    prev_tok_s = prev_summary.get("prefill_tok_per_sec", 0.0)
    curr_tok_s = curr_summary.get("prefill_tok_per_sec", 0.0)
    tok_s_ratio = (curr_tok_s / prev_tok_s) if prev_tok_s > 0 else 0.0

    all_families = sorted(list(set(curr_summary.get("family_breakdown", {}).keys()) | set(prev_summary.get("family_breakdown", {}).keys())))
    fam_comp = {}
    for fam in all_families:
        cf = curr_summary.get("family_breakdown", {}).get(fam, {})
        pf = prev_summary.get("family_breakdown", {}).get(fam, {})
        fam_comp[fam] = {
            "prev_total": pf.get("total", 0),
            "prev_acc": pf.get("accuracy_pct", 0.0),
            "prev_prefill": pf.get("mean_prefill_ms", 0.0),
            "curr_total": cf.get("total", 0),
            "curr_acc": cf.get("accuracy_pct", 0.0),
            "curr_prefill": cf.get("mean_prefill_ms", 0.0),
        }

    report_obj: dict[str, Any] = {
        "prev_server": prev_summary.get("server_url", prev_label),
        "curr_server": curr_summary.get("server_url", curr_label),
        "common_items_count": total_common,
        "prev_accuracy_pct": prev_acc,
        "curr_accuracy_pct": curr_acc,
        "accuracy_diff_pct": acc_diff,
        "agreement_count": len(agreed),
        "agreement_pct": (len(agreed) / total_common * 100.0) if total_common > 0 else 0.0,
        "both_correct_count": len(both_correct),
        "both_incorrect_count": len(both_incorrect),
        "recovered_count": len(recovered),
        "regressed_count": len(regressed),
        "timing": {
            "prefill_mean_ms": {"prev": prev_prefill["mean"], "curr": curr_prefill["mean"], "speedup": (prev_prefill["mean"] / curr_prefill["mean"]) if curr_prefill["mean"] > 0 else 0.0},
            "scoring_mean_ms": {"prev": prev_scoring["mean"], "curr": curr_scoring["mean"], "speedup": (prev_scoring["mean"] / curr_scoring["mean"]) if curr_scoring["mean"] > 0 else 0.0},
            "total_mean_ms": {"prev": prev_total["mean"], "curr": curr_total["mean"], "speedup": (prev_total["mean"] / curr_total["mean"]) if curr_total["mean"] > 0 else 0.0},
            "prefill_tok_per_sec": {"prev": prev_tok_s, "curr": curr_tok_s, "speedup": tok_s_ratio}
        },
        "recovered_items": [
            {
                "id": cid,
                "family": curr_map[cid].get("family"),
                "expected": curr_map[cid].get("expected"),
                "prev_predicted": prev_map[cid].get("predicted"),
                "curr_predicted": curr_map[cid].get("predicted"),
                "prev_prefill_ms": prev_map[cid].get("prefill_ms"),
                "curr_prefill_ms": curr_map[cid].get("prefill_ms")
            }
            for cid in recovered
        ],
        "regressed_items": [
            {
                "id": cid,
                "family": curr_map[cid].get("family"),
                "expected": curr_map[cid].get("expected"),
                "prev_predicted": prev_map[cid].get("predicted"),
                "curr_predicted": curr_map[cid].get("predicted"),
                "prev_prefill_ms": prev_map[cid].get("prefill_ms"),
                "curr_prefill_ms": curr_map[cid].get("prefill_ms")
            }
            for cid in regressed
        ],
        "family_comparison": fam_comp
    }

    # Print comparison report
    print("\n" + "=" * 78)
    print("SERVER BENCHMARK COMPARISON REPORT")
    print("=" * 78)
    print(f"{'Metric':<26} | {prev_label[:22]:<22} | {curr_label[:22]:<22} | {'Delta / Speedup'}")
    print("-" * 78)
    print(f"{'Server URL':<26} | {report_obj['prev_server'][:22]:<22} | {report_obj['curr_server'][:22]:<22} | -")
    print(f"{'Items Compared':<26} | {total_common:<22} | {total_common:<22} | -")
    acc_sign = "+" if acc_diff >= 0 else ""
    acc_item_diff = curr_correct - prev_correct
    item_diff_sign = "+" if acc_item_diff >= 0 else ""
    print(f"{'Overall Accuracy':<26} | {prev_correct:>3}/{total_common:<3} ({prev_acc:5.2f}%)       | {curr_correct:>3}/{total_common:<3} ({curr_acc:5.2f}%)       | {acc_sign}{acc_diff:5.2f}% ({item_diff_sign}{acc_item_diff} items)")
    prev_el = prev_summary.get("total_elapsed_sec", 0.0)
    curr_el = curr_summary.get("total_elapsed_sec", 0.0)
    el_diff = curr_el - prev_el
    el_sign = "+" if el_diff >= 0 else ""
    el_speedup = (prev_el / curr_el) if curr_el > 0 else 0.0
    print(f"{'Total Elapsed Time':<26} | {prev_el:6.1f} s                | {curr_el:6.1f} s                | {el_sign}{el_diff:5.1f} s ({el_speedup:.2f}x)")
    tok_diff = curr_tok_s - prev_tok_s
    tok_sign = "+" if tok_diff >= 0 else ""
    print(f"{'Prefill Throughput':<26} | {prev_tok_s:6.1f} tok/s            | {curr_tok_s:6.1f} tok/s            | {tok_sign}{tok_diff:5.1f} tok/s ({tok_s_ratio:.2f}x)")
    prev_hit = prev_summary.get("cache_hit_pct", 0.0)
    curr_hit = curr_summary.get("cache_hit_pct", 0.0)
    hit_diff = curr_hit - prev_hit
    hit_sign = "+" if hit_diff >= 0 else ""
    print(f"{'Prefix Cache Hit Rate':<26} | {prev_hit:5.1f}%                 | {curr_hit:5.1f}%                 | {hit_sign}{hit_diff:5.1f}%")

    print("\n--- Latency Breakdown (Mean / Median Server Timings) ---")
    print(f"{'Metric':<20} | {prev_label[:20]:<20} | {curr_label[:20]:<20} | {'Delta (Mean) / Speedup'}")
    print("-" * 78)
    print(f"{'Prefill Latency':<20} | {prev_prefill['mean']:6.1f} ms / {prev_prefill['median']:6.1f} ms  | {curr_prefill['mean']:6.1f} ms / {curr_prefill['median']:6.1f} ms  | {speedup_str(prev_prefill['mean'], curr_prefill['mean'])}")
    print(f"{'Scoring Latency':<20} | {prev_scoring['mean']:6.1f} ms / {prev_scoring['median']:6.1f} ms  | {curr_scoring['mean']:6.1f} ms / {curr_scoring['median']:6.1f} ms  | {speedup_str(prev_scoring['mean'], curr_scoring['mean'])}")
    print(f"{'Total Server Time':<20} | {prev_total['mean']:6.1f} ms / {prev_total['median']:6.1f} ms  | {curr_total['mean']:6.1f} ms / {curr_total['median']:6.1f} ms  | {speedup_str(prev_total['mean'], curr_total['mean'])}")

    if fam_comp:
        print("\n--- Accuracy & Prefill Timing by Task Family ---")
        print(f"{'Family':<20} | {'Base Acc':<9} | {'Cand Acc':<9} | {'Acc Diff':<9} | {'Base Prefill':<13} | {'Cand Prefill':<13} | {'Speedup'}")
        print("-" * 88)
        for fam, fc in fam_comp.items():
            f_diff = fc["curr_acc"] - fc["prev_acc"]
            f_dsign = "+" if f_diff >= 0 else ""
            f_spd = (fc["prev_prefill"] / fc["curr_prefill"]) if fc["curr_prefill"] > 0 else 0.0
            print(f"{fam:<20} | {fc['prev_acc']:6.1f}%   | {fc['curr_acc']:6.1f}%   | {f_dsign}{f_diff:5.1f}%   | {fc['prev_prefill']:8.1f} ms   | {fc['curr_prefill']:8.1f} ms   | {f_spd:5.2f}x")

    print("\n--- Item Agreement & Discordance Matrix ---")
    print(f"Agreement Rate      : {len(agreed)}/{total_common} ({report_obj['agreement_pct']:.1f}%)")
    print(f"Both Correct (++)   : {len(both_correct)} items")
    print(f"Both Incorrect (--) : {len(both_incorrect)} items")
    print(f"Recovered (+1 OK)   : {len(recovered)} items (Failed in Run 1 -> Passed in Run 2)")
    print(f"Regressed (-1 FAIL) : {len(regressed)} items (Passed in Run 1 -> Failed in Run 2)")

    if recovered:
        print(f"\nRecovered Items ({len(recovered)} total):")
        for item in report_obj["recovered_items"][:10]:
            print(f"  + [{item['id']}] ({item['family']}): was '{item['prev_predicted']}' -> now '{item['curr_predicted']}' (Expected: '{item['expected']}') | Prefill: {item['prev_prefill_ms']:.1f}ms -> {item['curr_prefill_ms']:.1f}ms")
        if len(recovered) > 10:
            print(f"    ... and {len(recovered) - 10} more recovered items.")

    if regressed:
        print(f"\nRegressed Items ({len(regressed)} total):")
        for item in report_obj["regressed_items"][:10]:
            print(f"  - [{item['id']}] ({item['family']}): was '{item['prev_predicted']}' -> now '{item['curr_predicted']}' (Expected: '{item['expected']}') | Prefill: {item['prev_prefill_ms']:.1f}ms -> {item['curr_prefill_ms']:.1f}ms")
        if len(regressed) > 10:
            print(f"    ... and {len(regressed) - 10} more regressed items.")

    print("=" * 78)
    return report_obj


def print_single_server_report(
    summary_data: dict[str, Any],
    results: list[dict[str, Any]],
    args: argparse.Namespace
) -> None:
    valid_results = [r for r in results if r.get("predicted") is not None]
    total_items = summary_data["total_items"]
    correct_count = summary_data["correct_count"]
    overall_acc = summary_data["overall_accuracy_pct"]
    prefill_tok_per_sec = summary_data["prefill_tok_per_sec"]
    family_stats = summary_data["family_breakdown"]
    latency_stats = summary_data["latency"]

    prefill_times = [r["prefill_ms"] for r in valid_results if "prefill_ms" in r]
    scoring_times = [r["scoring_ms"] for r in valid_results if "scoring_ms" in r]
    total_times = [r["total_server_ms"] for r in valid_results if "total_server_ms" in r]
    correct_probs = [r["probability"] for r in valid_results if r["is_correct"]]
    incorrect_probs = [r["probability"] for r in valid_results if not r["is_correct"]]

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated       : {len(results)}/{total_items}")
    print(f"Overall Accuracy      : {correct_count}/{len(results)} ({overall_acc:.2f}%)")
    print(f"Total Elapsed Time    : {summary_data['total_elapsed_sec']:.1f} s")
    print(f"Aggregate Prefill Speed: {prefill_tok_per_sec:.1f} tokens/second")
    if summary_data.get("total_cached_tokens", 0) > 0:
        tot_c = summary_data["total_cached_tokens"]
        tot_p = summary_data["total_prompt_tokens"]
        hit_pct = summary_data["cache_hit_pct"]
        print(f"Prefix Cache Hit Rate : {hit_pct:.1f}% ({tot_c:,}/{tot_p:,} prompt tokens)")
    if correct_probs:
        print(f"Mean Conf. (Correct)  : {statistics.mean(correct_probs):.1%}")
    if incorrect_probs:
        print(f"Mean Conf. (Incorrect): {statistics.mean(incorrect_probs):.1%}")

    print("\n--- Accuracy by Task Family ---")
    print(f"{'Family':<22} | {'Count':<5} | {'Correct':<7} | {'Accuracy':<10} | {'Mean Conf':<10} | {'Prefill (ms)':<12}")
    print("-" * 75)
    for fam, s in family_stats.items():
        print(f"{fam:<22} | {s['total']:<5} | {s['correct']:<7} | {s['accuracy_pct']:6.1f}%    | {s['mean_prob']:8.1%}  | {s['mean_prefill_ms']:10.1f}")

    if prefill_times and scoring_times:
        print("\n--- Latency Breakdown (Server Timings) ---")
        print(f"{'Metric':<18} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'Min (ms)':<9} | {'Max (ms)':<9}")
        print("-" * 65)
        print(f"{'Prefill Latency':<18} | {latency_stats['mean_prefill_ms']:10.1f} | {latency_stats['median_prefill_ms']:12.1f} | {min(prefill_times):9.1f} | {max(prefill_times):9.1f}")
        print(f"{'Scoring Latency':<18} | {latency_stats['mean_scoring_ms']:10.1f} | {latency_stats['median_scoring_ms']:12.1f} | {min(scoring_times):9.1f} | {max(scoring_times):9.1f}")
        print(f"{'Total Server Time':<18} | {latency_stats['mean_total_ms']:10.1f} | {statistics.median(total_times):12.1f} | {min(total_times):9.1f} | {max(total_times):9.1f}")

    if getattr(args, "chunk_size", 0) > 0:
        print("\n--- Chunk Aggregation Comparison Matrix ---")
        print(f"{'Strategy':<20} | {'Correct':<8} | {'Accuracy':<10} | {'Mean Prefill':<14} | {'Total Prefill (s)':<17}")
        print("-" * 75)
        for strat in ["max_confidence", "early_exit", "accumulated_prob"]:
            s_correct = sum(1 for r in results if r.get("strategies", {}).get(strat, {}).get("is_correct", False))
            s_acc = (s_correct / total_items) * 100.0 if total_items > 0 else 0.0
            s_prefills = [r["strategies"][strat]["prefill_ms"] for r in results if "strategies" in r and strat in r["strategies"]]
            s_mean_prefill = statistics.mean(s_prefills) if s_prefills else 0.0
            s_total_prefill_sec = sum(s_prefills) / 1000.0 if s_prefills else 0.0
            print(f"{strat:<20} | {s_correct:>3}/{total_items:<3}  | {s_acc:>6.2f}%    | {s_mean_prefill:>8.1f} ms    | {s_total_prefill_sec:>8.1f} s")

    # List Failures
    failures = [r for r in results if not r.get("is_correct", False)]
    if failures:
        print(f"\n--- Incorrect Items ({len(failures)} total) ---")
        for fail_item in failures[:15]:
            print(f"- [{fail_item['id']}] ({fail_item['family']}): Expected '{fail_item['expected']}', got '{fail_item.get('predicted')}' (prob: {fail_item.get('probability', 0.0):.1%})")
            rationale = fail_item.get("provenance", {}).get("rationale")
            if rationale:
                safe_rat = str(rationale[:120]).encode("ascii", errors="replace").decode("ascii")
                print(f"    Rationale: {safe_rat}...")
        if len(failures) > 15:
            print(f"    ... and {len(failures) - 15} more failures saved to result file.")


def evaluate_target_server(
    server_url: str,
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    compare_map: dict[str, dict[str, Any]] | None = None,
    resumed_map: dict[str, dict[str, Any]] | None = None,
    out_file: str | None = None,
    prev_elapsed_sec: float = 0.0,
    server_label: str = ""
) -> dict[str, Any]:
    compare_map = compare_map or {}
    resumed_map = resumed_map or {}
    total_items = len(items)

    resumed_count = sum(1 for it in items if it.get("id") in resumed_map)
    if resumed_count > 0:
        print(f"Resume State  : Found {resumed_count}/{total_items} items already evaluated in resume file.", flush=True)
        if resumed_count < total_items:
            print(f"Resuming live evaluation from item {resumed_count + 1}...", flush=True)
        else:
            print("All items in dataset are already evaluated in resume file.", flush=True)
    print("-" * 70, flush=True)

    results: list[dict[str, Any]] = []
    correct_count = 0
    session_start_time = time.perf_counter()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        for idx, item in enumerate(items, 1):
            item_id = item.get("id", f"item-{idx}")
            family = item.get("family", "unknown")
            expected = item.get("expected")

            # Fast-forward already evaluated items
            if item_id in resumed_map:
                rec = resumed_map[item_id]
                results.append(rec)
                if rec.get("is_correct", False):
                    correct_count += 1
                continue

            try:
                raw_state = item.get("state", "")
                if isinstance(raw_state, (dict, list)):
                    state_text = json.dumps(raw_state, ensure_ascii=False, indent=2)
                elif raw_state is None:
                    state_text = ""
                else:
                    state_text = str(raw_state)

                if args.chunk_size > 0:
                    chunks = chunk_text_approx(state_text, chunk_tokens=args.chunk_size, overlap_tokens=args.chunk_overlap)
                else:
                    chunks = [state_text]

                chunk_evals: list[dict[str, Any]] = []
                for c_idx, chunk in enumerate(chunks):
                    payload, letter_map = build_decision_payload(item, args.prompt_mode, state_override=chunk, mode=args.mode)
                    resp_data, c_wall_ms = send_decision_request(server_url, args.endpoint, payload)

                    result_item = resp_data["results"][0]
                    raw_predicted = result_item["decision"].get("decision")
                    predicted_c = letter_map[raw_predicted] if (letter_map and raw_predicted in letter_map) else raw_predicted
                    field_info = result_item["fields"].get("decision", {})
                    prob_c = field_info.get("probability", 0.0)
                    timings_c = resp_data.get("timings", {})
                    usage_c = resp_data.get("usage", {})

                    chunk_evals.append({
                        "chunk_idx": c_idx,
                        "predicted": predicted_c,
                        "probability": prob_c,
                        "prefill_ms": timings_c.get("prefill_ms", 0.0),
                        "scoring_ms": timings_c.get("scoring_ms", 0.0),
                        "total_ms": timings_c.get("total_ms", 0.0),
                        "wall_ms": c_wall_ms,
                        "context_tokens": usage_c.get("context_tokens", 0),
                        "cached_tokens": usage_c.get("cached_tokens", 0),
                        "prompt_tokens": usage_c.get("prompt_tokens", 0)
                    })

                    if args.chunk_aggregation == "early_exit" and prob_c >= args.early_exit_threshold:
                        break

                # Strategy 1: max_confidence
                best_chunk = max(chunk_evals, key=lambda x: x["probability"])
                pred_max = best_chunk["predicted"]
                prob_max = best_chunk["probability"]
                prefill_max = sum(c["prefill_ms"] for c in chunk_evals)
                scoring_max = sum(c["scoring_ms"] for c in chunk_evals)
                total_max = sum(c["total_ms"] for c in chunk_evals)
                wall_max = sum(c["wall_ms"] for c in chunk_evals)
                tokens_max = sum(c["context_tokens"] for c in chunk_evals)
                cached_max = sum(c.get("cached_tokens", 0) for c in chunk_evals)
                prompt_max = sum(c.get("prompt_tokens", 0) for c in chunk_evals)

                # Strategy 2: early_exit
                exit_evals = []
                exit_chunk = None
                for c in chunk_evals:
                    exit_evals.append(c)
                    if c["probability"] >= args.early_exit_threshold:
                        exit_chunk = c
                        break
                if exit_chunk is None:
                    exit_chunk = best_chunk
                pred_ee = exit_chunk["predicted"]
                prob_ee = exit_chunk["probability"]
                prefill_ee = sum(c["prefill_ms"] for c in exit_evals)
                scoring_ee = sum(c["scoring_ms"] for c in exit_evals)
                total_ee = sum(c["total_ms"] for c in exit_evals)
                wall_ee = sum(c["wall_ms"] for c in exit_evals)
                tokens_ee = sum(c["context_tokens"] for c in exit_evals)
                cached_ee = sum(c.get("cached_tokens", 0) for c in exit_evals)
                prompt_ee = sum(c.get("prompt_tokens", 0) for c in exit_evals)

                # Strategy 3: accumulated_prob
                label_scores: dict[str, float] = {}
                for c in chunk_evals:
                    lb = c["predicted"]
                    label_scores[lb] = label_scores.get(lb, 0.0) + c["probability"]
                pred_accum = max(label_scores.items(), key=lambda x: x[1])[0]
                prob_accum = label_scores[pred_accum] / len(chunk_evals)

                # Primary decision
                if args.chunk_aggregation == "early_exit":
                    predicted = pred_ee
                    prob = prob_ee
                    prefill_ms = prefill_ee
                    scoring_ms = scoring_ee
                    total_server_ms = total_ee
                    wall_ms = wall_ee
                    context_tokens = tokens_ee
                    cached_tokens = cached_ee
                    prompt_tokens = prompt_ee
                elif args.chunk_aggregation == "accumulated_prob":
                    predicted = pred_accum
                    prob = prob_accum
                    prefill_ms = prefill_max
                    scoring_ms = scoring_max
                    total_server_ms = total_max
                    wall_ms = wall_max
                    context_tokens = tokens_max
                    cached_tokens = cached_max
                    prompt_tokens = prompt_max
                else:
                    predicted = pred_max
                    prob = prob_max
                    prefill_ms = prefill_max
                    scoring_ms = scoring_max
                    total_server_ms = total_max
                    wall_ms = wall_max
                    context_tokens = tokens_max
                    cached_tokens = cached_max
                    prompt_tokens = prompt_max

                pred_str = str(predicted).strip().lower()
                exp_str = str(expected).strip().lower()
                q_val = item.get("question")
                q_type = q_val.get("type", "choice") if isinstance(q_val, dict) else "choice"
                if q_type == "noul":
                    noul_map = {"0": "no", "1": "yes", "false": "no", "true": "yes"}
                    pred_str = noul_map.get(pred_str, pred_str)
                    exp_str = noul_map.get(exp_str, exp_str)

                is_correct = (pred_str == exp_str)
                if is_correct:
                    correct_count += 1

                record = {
                    "id": item_id,
                    "family": family,
                    "expected": expected,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "probability": prob,
                    "context_tokens": context_tokens,
                    "cached_tokens": cached_tokens,
                    "prompt_tokens": prompt_tokens,
                    "prefill_ms": prefill_ms,
                    "scoring_ms": scoring_ms,
                    "total_server_ms": total_server_ms,
                    "wall_ms": wall_ms,
                    "chunk_count": len(chunks),
                    "chunks_evaluated": len(chunk_evals),
                    "strategies": {
                        "max_confidence": {
                            "predicted": pred_max,
                            "is_correct": (str(pred_max).strip().lower() == str(expected).strip().lower()),
                            "probability": prob_max,
                            "prefill_ms": prefill_max
                        },
                        "early_exit": {
                            "predicted": pred_ee,
                            "is_correct": (str(pred_ee).strip().lower() == str(expected).strip().lower()),
                            "probability": prob_ee,
                            "prefill_ms": prefill_ee
                        },
                        "accumulated_prob": {
                            "predicted": pred_accum,
                            "is_correct": (pred_accum.strip().lower() == str(expected).strip().lower()),
                            "probability": prob_accum,
                            "prefill_ms": prefill_max
                        }
                    },
                    "provenance": item.get("provenance", {})
                }
                results.append(record)

                session_now = time.perf_counter()
                session_elapsed = session_now - session_start_time
                total_elapsed_now = prev_elapsed_sec + session_elapsed
                elapsed_str = format_duration(total_elapsed_now)

                session_items_done = idx - resumed_count
                if session_items_done > 0 and idx < total_items:
                    rate = session_elapsed / session_items_done
                    items_left = total_items - idx
                    eta_str = format_duration(rate * items_left)
                elif idx == total_items:
                    eta_str = "00m 00s"
                else:
                    eta_str = "--:--"

                status_str = "[OK]  " if is_correct else "[FAIL]"
                running_acc = (correct_count / idx) * 100.0

                prev_item = compare_map.get(item_id)
                if prev_item and not args.quiet:
                    prev_ok = prev_item.get("is_correct", False)
                    if not prev_ok and is_correct:
                        print(f"  --> [RECOVERED +1] {item_id}: was {prev_item.get('predicted')} -> now {predicted} [OK]", flush=True)
                    elif prev_ok and not is_correct:
                        print(f"  --> [REGRESSED -1] {item_id}: was {prev_item.get('predicted')} -> now {predicted} [FAIL]", flush=True)

                if not args.quiet:
                    if args.verbose:
                        print(f"[{idx:03d}/{total_items:03d}] {status_str} {item_id} ({family}) | Elapsed: {elapsed_str} | ETA: {eta_str}", flush=True)
                        print(f"         Exp: {expected} | Pred: {predicted} | Prob: {prob:.1%}", flush=True)
                        cached_info = f" (cached: {cached_tokens})" if cached_tokens > 0 else ""
                        print(f"         Tokens: {context_tokens}{cached_info} | Prefill: {prefill_ms:.1f}ms | Score: {scoring_ms:.1f}ms | Wall: {wall_ms:.1f}ms", flush=True)
                    elif args.update_interval > 1:
                        if idx % args.update_interval == 0 or idx == total_items:
                            chunk_info = f" ({len(chunks)} chunks)" if len(chunks) > 1 else ""
                            comp_str = ""
                            if compare_map:
                                best_corr = sum(1 for r in results if compare_map.get(r["id"], {}).get("is_correct", False))
                                best_acc = (best_corr / idx) * 100.0
                                delta = running_acc - best_acc
                                sign = "+" if delta >= 0 else ""
                                comp_str = f" [vs Baseline: {best_acc:5.1f}% ({best_corr}/{idx}), diff: {sign}{delta:4.1f}%]"
                            print(f"[{idx:03d}/{total_items:03d}] Acc: {running_acc:5.1f}% ({correct_count}/{idx}){comp_str} | Elapsed: {elapsed_str} | ETA: {eta_str} | Last: {item_id[:22]} | {status_str} | Prefill: {prefill_ms:6.1f}ms{chunk_info}", flush=True)
                    else:
                        sys.stdout.write(
                            f"\r[{idx:03d}/{total_items:03d}] Acc: {running_acc:5.1f}% ({correct_count}/{idx}) | "
                            f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                            f"{item_id[:20]:20s} | {status_str} | Prefill: {prefill_ms:6.1f}ms | Score: {scoring_ms:5.1f}ms"
                        )
                        sys.stdout.flush()

                # Periodic auto-save
                if out_file and args.save_interval > 0 and session_items_done > 0 and session_items_done % args.save_interval == 0:
                    partial_summary = compute_summary_data(
                        results=results,
                        total_items=total_items,
                        total_elapsed=total_elapsed_now,
                        server_url=server_url,
                        input_file=args.input,
                        prompt_mode=args.prompt_mode,
                        decision_mode=args.mode,
                        timestamp=timestamp,
                        benchmark_items=items,
                        compare_with=getattr(args, "compare_with", None)
                    )
                    write_results_json(out_file, partial_summary)

            except Exception as e:
                print(f"\n[ERROR] Failed evaluating {item_id}: {e}", file=sys.stderr)
                results.append({
                    "id": item_id,
                    "family": family,
                    "expected": expected,
                    "predicted": None,
                    "is_correct": False,
                    "probability": 0.0,
                    "error": str(e)
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
                server_url=server_url,
                input_file=args.input,
                prompt_mode=args.prompt_mode,
                decision_mode=args.mode,
                timestamp=timestamp,
                benchmark_items=items,
                compare_with=getattr(args, "compare_with", None)
            )
            write_results_json(out_file, partial_summary)
            print(f"Partial results ({len(results)} items) saved to: {out_file}", flush=True)
            print("-" * 70, flush=True)
            print("To continue this benchmark run:")
            print(f"  python bench/benchmark.py --resume \"{out_file}\"", flush=True)
        print("=" * 70, flush=True)
        return compute_summary_data(
            results=results,
            total_items=total_items,
            total_elapsed=total_elapsed_now,
            server_url=server_url,
            input_file=args.input,
            prompt_mode=args.prompt_mode,
            decision_mode=args.mode,
            timestamp=timestamp,
            benchmark_items=items,
            compare_with=getattr(args, "compare_with", None)
        )

    session_now = time.perf_counter()
    total_elapsed = prev_elapsed_sec + (session_now - session_start_time)
    if not args.quiet and not args.verbose:
        sys.stdout.write("\n")

    summary_data = compute_summary_data(
        results=results,
        total_items=total_items,
        total_elapsed=total_elapsed,
        server_url=server_url,
        input_file=args.input,
        prompt_mode=args.prompt_mode,
        decision_mode=args.mode,
        timestamp=timestamp,
        benchmark_items=items,
        compare_with=getattr(args, "compare_with", None)
    )
    if out_file:
        write_results_json(out_file, summary_data)

    print_single_server_report(summary_data, results, args)
    return summary_data


def run_benchmark() -> None:
    args = parse_args()

    if args.endpoint == "/v1/systemone":
        try:
            import benchmark_systemone
        except ImportError:
            from bench import benchmark_systemone
        benchmark_systemone.main()
        return

    # Automatically set input to compare_with file if compare_with is provided and input was left as default
    if args.compare_with and args.input in ("bench/easy.jsonl", "bench\\easy.jsonl"):
        if os.path.exists(args.compare_with):
            args.input = args.compare_with

    resumed_map: dict[str, dict[str, Any]] = {}
    prev_elapsed_sec = 0.0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume file not found: {args.resume}")
        with open(args.resume, "r", encoding="utf-8") as f:
            resume_data = json.load(f)

        # Infer input dataset if default was kept
        if args.input in ("bench/easy.jsonl", "bench\\easy.jsonl") and "input_file" in resume_data:
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

        # Infer compare_with if it was a comparison run and not explicitly specified
        if not args.compare_with:
            cand_comp = resume_data.get("compare_with")
            if not cand_comp and "input_file" in resume_data and str(resume_data["input_file"]).endswith(".json"):
                cand_comp = resume_data["input_file"]
            if cand_comp:
                if os.path.exists(cand_comp):
                    args.compare_with = cand_comp
                else:
                    base = os.path.basename(cand_comp)
                    in_bench = os.path.join("bench", base)
                    in_res_dir = os.path.join(os.path.dirname(args.resume) or ".", base)
                    if os.path.exists(in_bench):
                        args.compare_with = in_bench
                    elif os.path.exists(in_res_dir):
                        args.compare_with = in_res_dir

        if "prompt_mode" in resume_data and args.prompt_mode == "debiased_hybrid":
            args.prompt_mode = resume_data["prompt_mode"]
        if "decision_mode" in resume_data and args.mode == "auto":
            args.mode = resume_data["decision_mode"]

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
        out_file = os.path.join(out_dir, f"results_{dataset_base}_{timestamp}.json")

    print("=" * 70)
    print("LLAMA.CPP DECISION ENDPOINT BENCHMARK")
    print("=" * 70)
    print(f"Target Server : {args.url}")
    if args.compare_url:
        print(f"Compare Server: {args.compare_url}")
    print(f"Dataset File  : {args.input}")
    if args.sample:
        print(f"Sample Size   : {args.sample} ({args.sample_strategy} strategy, seed={args.seed})")
    print(f"Prompt Mode   : {args.prompt_mode}")
    print(f"Decision Mode : {args.mode}")
    if args.resume:
        print(f"Resume Source : {args.resume}")
    print(f"Output File   : {out_file}")
    if args.family:
        print(f"Family Filter : {args.family}")
    if args.limit:
        print(f"Item Limit    : {args.limit}")

    # Check server connectivity
    print(f"Checking server health at {args.url}...", end=" ", flush=True)
    if check_server_health(args.url):
        print("[HEALTHY]", flush=True)
    else:
        print("[WARNING: /health endpoint did not respond with ok]", flush=True)

    if args.compare_url:
        print(f"Checking compare server health at {args.compare_url}...", end=" ", flush=True)
        if check_server_health(args.compare_url):
            print("[HEALTHY]", flush=True)
        else:
            print("[WARNING: /health endpoint did not respond with ok]", flush=True)

    filter_ids = None
    if args.failed_from:
        if not os.path.exists(args.failed_from):
            raise FileNotFoundError(f"Previous results file not found: {args.failed_from}")
        with open(args.failed_from, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        prev_res = prev_data.get("results", [])
        filter_ids = set(r["id"] for r in prev_res if not r.get("is_correct"))
        print(f"Filter Failed : Only evaluating {len(filter_ids)} items that failed in {args.failed_from}", flush=True)

    compare_summary: dict[str, Any] | None = None
    compare_map: dict[str, dict[str, Any]] = {}
    if args.compare_with:
        if os.path.exists(args.compare_with):
            with open(args.compare_with, "r", encoding="utf-8") as f:
                compare_summary = json.load(f)
            for r in compare_summary.get("results", []):
                if "id" in r:
                    compare_map[r["id"]] = r
            print(f"Loaded comparison baseline ({len(compare_map)} results) from: {args.compare_with}", flush=True)
        else:
            print(f"[WARNING] compare-with file not found: {args.compare_with}", flush=True)

    # Load dataset
    items = load_dataset(
        file_path=args.input,
        family_filter=args.family,
        offset=args.offset,
        limit=args.limit,
        filter_ids=filter_ids,
        sample=args.sample,
        sample_strategy=args.sample_strategy,
        seed=args.seed
    )
    total_items = len(items)
    print(f"Loaded {total_items} benchmark items to evaluate.", flush=True)

    if total_items == 0:
        print("No items to evaluate. Exiting.")
        return

    if args.save_items:
        save_items_jsonl(args.save_items, items)

    s1_summary = evaluate_target_server(
        server_url=args.url,
        items=items,
        args=args,
        compare_map=compare_map,
        resumed_map=resumed_map,
        out_file=out_file,
        prev_elapsed_sec=prev_elapsed_sec,
        server_label=f"Server 1 ({args.url})" if args.compare_url else args.url
    )

    if compare_summary:
        comp_report = generate_comparison_report(
            curr_summary=s1_summary,
            prev_summary=compare_summary,
            curr_label=f"Candidate ({args.url})",
            prev_label=f"Baseline ({compare_summary.get('server_url', args.compare_with)})"
        )
        s1_summary["comparison"] = comp_report
        write_results_json(out_file, s1_summary)

    if args.compare_url:
        s2_out_file = None
        if out_file:
            root, ext = os.path.splitext(out_file)
            s2_out_file = f"{root}_compare{ext}"

        s1_map = {r["id"]: r for r in s1_summary.get("results", []) if "id" in r}
        print("\n" + "=" * 70)
        print(f"BENCHMARKING SECOND SERVER: {args.compare_url}")
        print("=" * 70)
        s2_summary = evaluate_target_server(
            server_url=args.compare_url,
            items=items,
            args=args,
            compare_map=s1_map,
            resumed_map=None,
            out_file=s2_out_file,
            prev_elapsed_sec=0.0,
            server_label=f"Server 2 ({args.compare_url})"
        )

        comp_report = generate_comparison_report(
            curr_summary=s2_summary,
            prev_summary=s1_summary,
            curr_label=f"Server 2 ({args.compare_url})",
            prev_label=f"Server 1 ({args.url})"
        )
        s2_summary["comparison"] = comp_report
        if s2_out_file:
            write_results_json(s2_out_file, s2_summary)
            print(f"\nServer 2 results and comparison saved to: {s2_out_file}")

    print(f"\nDetailed benchmark results saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
