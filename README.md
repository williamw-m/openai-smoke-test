# OpenAI Smoke Test Benchmark

A smoke testing tool for OpenAI-compatible endpoints.

## Features

- Concurrent user simulation with customizable query load
- Summary report with:
  - Average time to first token (TTFT)
  - Tokens per second (TPS)
  - Percentiles (p50, p90)
- Customizable prompt size (min/max word count)
- Optional single-run mode for quick testing

## Setup

```bash
make setup
```

This creates a virtual environment in .venv/, installs dependencies, and
installs the tool locally in editable mode.

## Example Usage

Below is a light run against a local Ollama instance.

```bash
 openai-smoke-test git:(main) ✗ .venv/bin/openai-smoketest --api-key ollama --api-base http://localhost:11434/v1 --model granite3.3:8b --num-users 2 --queries-per-user 1
Running queries: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:41<00:00, 20.67s/it]

--- SUMMARY REPORT ---
Total Queries: 2
Successful Queries: 2
Failed Queries: 0
+-------------------------+--------+-------+-------+
| Metric                  |   Mean |   P50 |   P90 |
+=========================+========+=======+=======+
| Time to First Token (s) |  18.54 | 18.54 | 28.07 |
+-------------------------+--------+-------+-------+
| Tokens per Second       |   6.47 |  6.47 | 10.25 |
+-------------------------+--------+-------+-------+
Successful
```




