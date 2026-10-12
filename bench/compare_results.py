import json
import os

f_liquid = "bench/rs1_hard.json"
f_jev = "c:/Users/pverg/coding/llama.cpp/bench/results_hard_20260929_070024.json"

d1 = json.load(open(f_liquid, encoding="utf-8"))
d2 = json.load(open(f_jev, encoding="utf-8"))

print(f"| Model | Total Items | Correct | Accuracy | Total Time | Mean / Item |")
print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
t1_mean = (d1['total_elapsed_sec'] / d1['total_items']) * 1000.0
t2_mean = (d2['total_elapsed_sec'] / d2['total_items']) * 1000.0
print(f"| **LiquidAI d1-3B (HTP0)** | {d1['total_items']} | {d1['correct_count']} | **{d1['overall_accuracy_pct']:.2f}%** | {d1['total_elapsed_sec']:.1f} s | {t1_mean:.1f} ms |")
print(f"| **JevK5 4B (Qwen3.5)** | {d2['total_items']} | {d2['correct_count']} | **{d2['overall_accuracy_pct']:.2f}%** | {d2['total_elapsed_sec']:.1f} s | {t2_mean:.1f} ms |")

print("\n### Family Breakdown Comparison\n")
print(f"| Family | Count | LiquidAI d1-3B | JevK5 4B | Delta Acc | LiquidAI Mean Latency |")
print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")

fams = sorted(set(list(d1.get("family_breakdown", {}).keys()) + list(d2.get("family_breakdown", {}).keys())))
for f in fams:
    f1 = d1.get("family_breakdown", {}).get(f, {})
    f2 = d2.get("family_breakdown", {}).get(f, {})
    cnt = f1.get("total", f2.get("total", 0))
    a1 = f1.get("accuracy_pct", 0.0)
    a2 = f2.get("accuracy_pct", 0.0)
    c1 = f1.get("correct", 0)
    c2 = f2.get("correct", 0)
    lat1 = f1.get("mean_latency_ms", 0.0)
    delta = a2 - a1
    print(f"| `{f}` | {cnt} | {c1}/{cnt} ({a1:.1f}%) | {c2}/{cnt} ({a2:.1f}%) | **{delta:+.1f}%** | {lat1:.1f} ms |")
