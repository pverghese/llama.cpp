# SystemOne (/v1/systemone) Endpoint and Qualcomm Hexagon HTP Guide

## Overview

The native `POST /v1/systemone` endpoint provides TypeSafe Jev-compatible structured decision capability directly within `llama-server`.
Instead of generating free-form text or running multiple sequential inference passes, the server prefills shared context once and evaluates a batch of typed questions (choice, noul, score) concurrently across parallel sequence branches.

All branch answers are scored against single-token labels and returned as exact probability distributions, along with normalized entropy confidence scores.

---

## Endpoint Specification: `POST /v1/systemone`

### Request Format

```json
{
  "model": "jev-latest",
  "state": "The sky is blue. Paris is the capital of France. The number seven is odd.",
  "preamble": "Optional instructions prepended to questions.",
  "questions": {
    "paris": {
      "type": "choice",
      "instructions": "What is the capital of France?",
      "criteria": {
        "berlin": "Berlin",
        "paris": "Paris",
        "rome": "Rome"
      }
    },
    "odd": {
      "type": "noul",
      "instructions": "Is seven an odd number?"
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate urgency level.",
      "criteria": [
        "Routine",
        "Urgent",
        "Emergency"
      ]
    }
  }
}
```

#### Request Fields

- `model` (string, required): Model name or alias (e.g. `jev-latest`).
- `state` (string | array | object, required):
  - Text string: Raw context evaluated directly.
  - Object / Array: Serialized as JSON into user message context.
  - Chat Messages array: Formatted using the model chat template (e.g. `[{"role": "user", "content": "..."}]`).
  - Multimodal parts: Supports `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` when server has an active multimodal projector (`--mmproj`).
- `preamble` (string, optional): Text prepended to questions in the system prompt.
- `suffix` (string, optional, default `"Answer: "`): Prompt suffix appended immediately before candidate token paths. Defaults to `"Answer: "` with a trailing space to ensure proper leading-space BPE token alignment.
- `mode` (string, optional, default `"auto"`): Scoring algorithm: `"auto"` (default), `"tree"` (full trie expansion), or `"greedy"` (fast top-1 path decoding). Using `"mode": "greedy"` with `"candidates": "keys"` delivers ~650 ms sub-second inference over 75+ multi-token semantic categories while achieving top accuracy.
- `candidates` (string, optional, default `"symbols"`): Target readout token mode (can also be set per-question):
  - `"symbols"` (default): Pure alphabetic letter candidates (`A`..`Z`, `a`..`z`, `AA`..`ZZ`). Provides fast sub-second decision scoring without punctuation bias.
  - `"keys"`: Direct multi-token semantic strings matching criteria keys (e.g. `change_ai_name`). Recommended when evaluating fine-grained near-duplicate classes where direct semantic token probabilities provide maximum discrimination.
- `questions` (object, required): Map of 1 to 64 question definitions keyed by unique question IDs:
  - `choice`: Single selection among 2 to 255 options.
    - `instructions` (string | object): Guidance for selection.
    - `criteria` (object): Map of option keys to text descriptions (or null).
    - `candidates` (string, optional): Override candidate mode for this question (`"keys"` or `"symbols"`).
  - `noul`: Boolean evaluation (yes/no).
    - `instructions` (string | object): Yes/no statement or query.
    - `criteria` (object, optional): Definitions of `true` (default "Yes") and `false` (default "No").
  - `score`: Continuous rating over an ordered rubric.
    - `instructions` (string | object): Scoring instructions.
    - `criteria` (array | object): 2 to 64 ordered rubric levels (index 0, 1, 2, ...).

#### Example: High-Accuracy Intent Classification with Direct Keys (`"candidates": "keys"`)

When evaluating fine-grained or near-duplicate categories (e.g. banking intents or legal provisions), opt into `"candidates": "keys"` with `"suffix": "Answer: "` to calculate log probabilities directly over the semantic tokens:

```json
{
  "model": "jev-latest",
  "state": "I need to get a new virtual debit card for online shopping.",
  "candidates": "keys",
  "suffix": "Answer: ",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Select the single intent that best matches the user's request.",
      "criteria": {
        "getting_virtual_card": "User wants a standard virtual debit card",
        "get_disposable_virtual_card": "User wants a disposable virtual card that regenerates after use",
        "order_physical_card": "User wants a physical plastic card delivered"
      }
    }
  }
}
```

Candidate mode can also be overridden on a per-question basis within the same request:

```json
{
  "model": "jev-latest",
  "state": "The app crashed when I tried to order a physical card.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Select matching intent",
      "criteria": {
        "order_physical_card": "Card order flow",
        "app_crash": "Application stability issue"
      },
      "candidates": "keys"
    },
    "urgent": {
      "type": "noul",
      "instructions": "Is the user blocked?"
    }
  }
}
```

### Response Format

```json
{
  "model": "jev-latest",
  "answers": {
    "paris": {
      "type": "choice",
      "choice": "paris",
      "probabilities": {
        "berlin": 0.012,
        "paris": 0.985,
        "rome": 0.003
      },
      "confidence": 0.96
    },
    "odd": {
      "type": "noul",
      "noul": 0.992
    },
    "urgency": {
      "type": "score",
      "score": 0.15,
      "legend": {
        "0": "Routine",
        "1": "Urgent",
        "2": "Emergency"
      },
      "probabilities": {
        "0": 0.85,
        "1": 0.15,
        "2": 0.00
      },
      "confidence": 0.81
    }
  },
  "usage": {
    "input_tokens": 128,
    "output_tokens": 4
  }
}
```

#### Response Headers

Every successful 200 response sets the following diagnostic headers:
- `x-llamajev-cached-tokens`: Tokens reused from cached prefix / prefill.
- `x-llamajev-prefix-tokens`: Length of shared prefix before branching.
- `x-llamajev-readout-retries`: Retry count for label readout (0 under native parallel-decision).
- `x-llamajev-model`: Model identifier serving the request.

---

## Building llama-server with Qualcomm Hexagon NPU and Adreno GPU Support

On Windows 11 on ARM64 (Snapdragon X Elite), `llama-server` runs with hardware acceleration on both:
- **Qualcomm Hexagon Tensor Processor (HTP0 / NPU)** via the WHQL-signed DSP kernels and dynamic entrypoint shim.
- **Qualcomm Adreno GPU** via native OpenCL with embedded Adreno matrix/vector kernels (`ggml-opencl.dll`).

### 1. Build Configuration

Configure CMake with dynamic backend plugin support (`BUILD_SHARED_LIBS=ON` and `GGML_BACKEND_DL=ON`) and OpenCL GPU acceleration pointing to Qualcomm's OpenCL SDK:

```powershell
# Configure build with Clang toolchain and Adreno OpenCL GPU support
cmake -B build -G Ninja `
  -DCMAKE_MAKE_PROGRAM="C:/Users/pverg/AppData/Local/Microsoft/WinGet/Links/ninja.exe" `
  -DCMAKE_TOOLCHAIN_FILE="cmake/arm64-windows-llvm.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_SHARED_LIBS=ON `
  -DGGML_BACKEND_DL=ON `
  -DGGML_OPENCL=ON `
  -DCMAKE_PREFIX_PATH="C:/Qualcomm/OpenCL_SDK/2.3.2"

# Compile llama-server, llama-bench, and llama-cli
cmake --build build --config Release --target llama-server llama-bench llama-cli
```

### 2. Runtime Driver and Backend Deployment

Place the following 6 files into `build\bin\` alongside `llama-server.exe`:

| File | Source / Origin | Description |
| :--- | :--- | :--- |
| `libcdsprpc.dll` | Windows System / Qualcomm Driver Store | Qualcomm FastRPC user-mode driver interface |
| `libomp140.aarch64.dll` | Visual C++ Redistributable (ARM64) | OpenMP runtime |
| `ggml-hexagon-core.dll` | Qualcomm GenieX / Snapdragon package | Qualcomm Hexagon WHQL core runtime (API v2) |
| `libggml-htp-v73.so` | Qualcomm GenieX / Snapdragon package | Microsoft WHQL-signed DSP kernel (v73) |
| `libggml-htp.cat` | Qualcomm GenieX / Snapdragon package | FastRPC driver cryptographic catalog |
| `ggml-hexagon.dll` | Compiled from source shim | Dynamic entrypoint bridge (adapts API v2 -> v3) |

#### Compiling the Entrypoint Shim (`ggml-hexagon.dll`)

Upstream `llama.cpp` uses `GGML_BACKEND_API_VERSION 3`, while Qualcomm's WHQL-signed core library exports `ggml_backend_hexagon_reg` with API version 2. The shim in `scripts/snapdragon/ggml_hexagon_shim.c` dynamically bridges the symbol and adapts the buffer type interfaces to API version 3:

```powershell
clang -target aarch64-pc-windows-msvc -shared -O2 `
  -I ggml/include -I ggml/src `
  scripts/snapdragon/ggml_hexagon_shim.c `
  -lkernel32 `
  -o build\bin\ggml-hexagon.dll
```

### 3. Environment Configuration

Set FastRPC driver path and DSP performance tuning before starting the server:

```powershell
# Set path to directory containing libggml-htp-v73.so and libggml-htp.cat
$env:ADSP_LIBRARY_PATH = (Resolve-Path "build\bin").Path

# Performance profile
$env:GGML_HEXAGON_POWER_MODE = "burst"  # Peak DSP clock frequency
$env:GGML_HEXAGON_OPFUSION   = "1"        # Fuse RMSNorm + GEMM on DSP
$env:GGML_HEXAGON_OPPOLL     = "0"        # Passive OS wait
```

### 4. Running the Server

Run `llama-server` with `--decision-seqs` (required for `/v1/systemone` and `/v1/decision`):

```powershell
build\bin\llama-server.exe `
  -m "C:/Qualcomm/models/qwen-2b/Qwen3.5-2B-Q4_0.gguf" `
  -dev HTP0 -ngl 99 `
  -c 4096 -b 512 -ub 512 `
  --decision-seqs 16 `
  --port 8096
```

> Note: `--decision-seqs N` (minimum 3) enables the unified KV cache (`kv_unified = true`) and reserves parallel sequence slots so all questions decode in a single forward pass on the NPU.

### 5. Prefix Caching and Large Catalogues

When a single decision question contains a large options catalogue (e.g. 150+ intents), `/v1/systemone` automatically renders the instructions and catalogue into the System message (`system_text`).

- **Shared Context (`head`)**: The system prompt containing the static catalogue is cached on `seq_snap`. Across requests evaluating against the same question/catalogue, `prepare_prefix` finds a 100% prefix cache hit.
- **Dynamic Branching**: Only the dynamic user state is evaluated in the trunk context, leaving a compact suffix (`"Answer:\n"`) on each parallel sequence branch.
- **Performance**: On Qualcomm Hexagon HTP0, prefix caching reduces prefill latency on 150-choice catalogues from ~4,500 ms to <200 ms.

### 6. Architecture Decisions: Adaptive Layout and Scoring

The server implements two automatic structural optimizations based on candidate cardinality:

#### A. Automatic Greedy Mode for Large Catalogues (`choices > 12`)
- **Motivation**: For large taxonomies (e.g. Banking77 with 77 intents, CLINC150 with 151 intents), multi-token trie expansion in tree search adds redundant token overhead while top-1 greedy prefix discrimination achieves identical or superior intent accuracy.
- **Rule**: When `mode == "auto"` (server default), the server inspects candidate set size:
  - `choices > 12`: Automatically activates `"greedy"` mode (`tree_max = 12`).
  - `choices <= 12`: Evaluates via full branch tree exploration (`"tree"`).
  - Explicit client overrides (`"mode": "tree"` or `"mode": "greedy"`) are always honored.
- **Impact**: Reduces Banking77 latency from ~2,500 ms to ~650 ms per item on Qualcomm Hexagon NPU while maintaining 69% accuracy.

#### B. Primed Recency Format for Reasoning Tasks (`choices <= 12`)
- **Motivation**: For multiple-choice reasoning benchmarks (e.g. MMLU, ARC Challenge with 4 choices A-D), placing options in the system prompt separates them from the problem context, causing value-vs-key generation bias (model attempts to output the answer value rather than the choice letter, dropping accuracy to ~28%).
- **Rule**: When a single choice question has `choices <= 12` (or when explicitly requested via `"prompt_mode": "primed_letters"`):
  - **User Turn (`state`)**: Dynamically assembled with the problem statement followed by labeled choices:
    ```markdown
    # Question
    <instructions>

    # Evidence
    <problem statement / context>

    # Criterion
    <instructions>

    # Options
    A. A: 0
    B. B: 4
    C. C: 2
    D. D: 6

    Respond with only the letter of the best option.
    ```
  - **System Turn**: Sets the evaluator persona (`"You are a System One decision model. Select the single option letter that best matches."`).
  - **Assistant Turn**: Prompts with `"Answer: "`, scoring choice letter tokens (`A`, `B`, `C`, `D`).
- **Impact**: Restores MMLU abstract algebra accuracy from 28.0% to 49.0%-58.0%, exceeding the reference `/v1/decision` endpoint (46.0%).

---

## Benchmarking

A dedicated benchmark script `bench/benchmark_systemone.py` evaluates accuracy and latency against JSONL datasets:

```powershell
# Benchmark clinc150 dataset (151 intent classes) with single-token symbols (default)
python bench/benchmark_systemone.py --input bench/clinc150.jsonl --url http://127.0.0.1:8096 -n 100 --candidates symbols

# Benchmark fine-grained taxonomies (e.g. banking77, ledgar) with direct keys and space suffix
python bench/benchmark_systemone.py --input bench/banking77.jsonl --url http://127.0.0.1:8096 -n 100 --candidates keys --suffix "Answer: "

# Fast sub-second evaluation of large taxonomies using greedy keys (~650 ms per item)
python bench/benchmark_systemone.py --input bench/banking77.jsonl --url http://127.0.0.1:8096 -n 100 --candidates keys --mode greedy

# Benchmark multiple-choice reasoning benchmarks (e.g. MMLU, ARC) using primed recency formatting
python bench/benchmark_systemone.py --input bench/mmlu.jsonl --url http://127.0.0.1:8096 -n 100 --prompt-mode primed_letters --mode greedy
```

---

## Running the Tests

### 1. Standalone CLI Test Suite (Zero Dependencies)

The standalone runner uses only standard Python libraries (`urllib.request`, `json`) and runs against any live server:

```powershell
# Run against default port (8096, with fallback to 8080)
python scripts/server-test-systemone.py

# Custom host / port
python scripts/server-test-systemone.py --host 127.0.0.1 --port 8096
```

This validates:
- Unambiguous facts (`choice` and `noul`)
- Score monotonicity on a severity rubric
- 26-way choice classification
- 64-way choice well-formedness and probability distributions
- Multimodal vision questions with `tools/server/tests/assets/shapes-448.png` (auto-detected if `--mmproj` is loaded)

### 2. Pytest Live Integration Tests

Run the end-to-end live test suite against a running server:

```powershell
$env:LLAMAJEV_LIVE_URL = "http://127.0.0.1:8096"
pytest tools/server/tests/unit/test_systemone_live.py
```

### 3. Pytest Offline Scoring Math Tests

Run verification of softmax normalization, shift invariance, and normalized entropy confidence calculations:

```powershell
pytest tools/server/tests/unit/test_systemone_scoring.py
```

### 4. Pytest Native Server Unit Tests

Run automated lifecycle unit tests:

```powershell
pytest tools/server/tests/unit/test_systemone.py
```
