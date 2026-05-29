# vLLM offline load benchmark

Part of the [end-to-end workload overview](../README.md) — **step 3 of 3**.

Report-grade offline benchmark for the Llama 3.1 70B FP8 vLLM deployment. Runs on the **bare-metal GPU node** (needs `rocm-smi` for power metrics) and sends async completion requests to the lmstack-router service.

## Prerequisites

- vLLM deployment and lmstack-router running in namespace `vllm` (see [step 2 — Deploy vLLM](../02-vllm-llama3-70b-fp8/README.md))
- **Single bare-metal node with 8 AMD GPUs** — run the benchmark on the same node that serves inference so `rocm-smi` power samples match the workload
- Python 3.10+ with `aiohttp`: `pip install aiohttp`
- `rocm-smi` available on the GPU node
- A `prompts.json` file (see [Prompts file](#prompts-file-promptsjson); generate with `download_prompts.py`)

## 1. Get the lmstack-router service IP

The benchmark targets the **lmstack-router** service (`llm-router`), not the vLLM pod directly. Router service port is **80** (forwards to router port 8001).

```bash
kubectl -n vllm get svc llm-router
```

Note the **CLUSTER-IP** column, or fetch it explicitly:

```bash
ROUTER_IP=$(kubectl -n vllm get svc llm-router -o jsonpath='{.spec.clusterIP}')
echo "Router ClusterIP: ${ROUTER_IP}"
```

Verify the router can reach vLLM backends:

```bash
curl -s "http://${ROUTER_IP}/health"
curl -s "http://${ROUTER_IP}/v1/models" | jq .
```

## 2. Set `VLLM_URL`

Export the router ClusterIP as the completions endpoint:

```bash
export VLLM_URL="http://${ROUTER_IP}/v1/completions"
echo "VLLM_URL=${VLLM_URL}"
```

One-liner:

```bash
export VLLM_URL="http://$(kubectl -n vllm get svc llm-router -o jsonpath='{.spec.clusterIP}')/v1/completions"
```

### If the GPU node cannot reach ClusterIP

Use port-forward from a machine with `kubectl` access, then point `VLLM_URL` at localhost:

```bash
kubectl -n vllm port-forward svc/llm-router 8080:80
export VLLM_URL="http://127.0.0.1:8080/v1/completions"
```

## 3. Run the benchmark

```bash
pip install aiohttp

export MODEL="/models/Llama-3.1-70B-Instruct-FP8"
export GPU_INDEX="0,1,2,3,4,5,6,7"
export DURATION=600          # total trial duration (seconds)
export WARMUP_S=60           # excluded from throughput/power stats
export CONCURRENCIES=512
export MAX_TOKENS=1024
export TRIALS=3
export LABEL="exp_inference_only_report_grade"

python3 offline_bench_4.py
```

Results are written to `./${LABEL}_<timestamp>_summary.json` in the current directory.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | *(required)* | lmstack-router completions URL, e.g. `http://10.43.x.x/v1/completions` |
| `MODEL` | `/models/Llama-3.1-70B-Instruct-FP8` | Model name passed in API requests |
| `PROMPTS_FILE` | `prompts.json` | Path to prompt dataset |
| `GPU_INDEX` | `0,1,2,3,4,5,6,7` | Comma-separated GPU indices for `rocm-smi` power sampling |
| `DURATION` | `600` | Trial duration in seconds |
| `WARMUP_S` | `60` | Warmup period excluded from metrics |
| `CONCURRENCIES` | `512` | Max in-flight requests |
| `MAX_TOKENS` | `1024` | Max completion tokens per request |
| `TRIALS` | `3` | Number of trials (aggregate reports median / p10 / p90) |
| `SEED` | `42` | Base random seed |
| `POWER_SAMPLE_S` | `1.0` | `rocm-smi` sampling interval |
| `REQUEST_TIMEOUT_S` | `300` | Per-request HTTP timeout |
| `LABEL` | `exp_inference_only_report_grade` | Prefix for output JSON filename |

## Prompts file (`prompts.json`)

`offline_bench_4.py` reads prompts from `prompts.json` by default (`PROMPTS_FILE` env var overrides the path). The file is **not checked into this repo** (~20k entries, large on disk); generate it locally or copy an existing dataset into this directory before running the benchmark.

### Format

JSON array. Each entry is either:

- a plain string, or
- an object with:
  - `prompt` (required) — text sent to `/v1/completions`
  - `length_class` (optional) — one of `short`, `medium`, or `long`

```json
[
  {
    "prompt": "Can you explain contrastive learning in simple terms?",
    "length_class": "short"
  },
  {
    "prompt": "Summarize the following article:\n\n...",
    "length_class": "long"
  }
]
```

### How the benchmark uses prompts

`PromptStream` in `offline_bench_4.py`:

1. Groups prompts by `length_class` when present.
2. **Round-robins** across available classes (`short` → `medium` → `long` → …) so prefill lengths stay balanced during a run.
3. Picks a **random prompt within the selected class** using a per-trial seed (`SEED` + trial index) for reproducibility.
4. If no `length_class` fields exist, falls back to random selection from the flat prompt list.

At startup the script prints pool sizes, e.g. `short=6000 medium=8000 long=6000 total=20000`.

### Reference dataset (used in MiTAC experiments)

The production `prompts.json` contains **20,000 unique prompts**:

| `length_class` | Count | Word-count bucket | Role in benchmark |
|----------------|-------|-------------------|-------------------|
| `short` | 6,000 | &lt; 50 words | Quick prefill; stresses decode throughput |
| `medium` | 8,000 | 50–200 words | Balanced prefill/decode mix |
| `long` | 6,000 | 200+ words | Heavy prefill; exercises chunked prefill |

Sources (merged and deduplicated by the first 200 characters):

| Hugging Face dataset | Typical length | Content |
|----------------------|----------------|---------|
| `OpenAssistant/oasst1` | short–medium | Top-level prompter turns |
| `HuggingFaceH4/ultrachat_200k` | medium–long | First user message from SFT split |
| `cnn_dailymail` (`3.0.0`) | long | `Summarize the following article:\n\n{article}` |

### Generate `prompts.json`

Use the included helper script (requires `pip install datasets` and a Hugging Face token):

```bash
pip install datasets

export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
python3 download_prompts.py
```

This writes `prompts.json` in the current directory. Re-runs overwrite the file.

Alternatively, point at an existing file:

```bash
export PROMPTS_FILE=/path/to/prompts.json
python3 offline_bench_4.py
```

## Output

Each run produces a JSON summary with:

- Per-trial throughput, latency, energy, and power (from `rocm-smi`)
- Aggregate median / p10 / p90 for tokens/sec, tokens/watt, and average power
- Warmup period excluded from throughput and power integrals

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `VLLM_URL is not set` | Export `VLLM_URL` using the lmstack-router ClusterIP (step 2) |
| Connection refused / timeout | Confirm `llm-router` is running: `kubectl -n vllm get pods,svc` |
| All requests fail | Check vLLM pod is ready and router sees backends: `curl http://${ROUTER_IP}/v1/models` |
| Power metrics are zero | Run on the GPU node; confirm `rocm-smi --showpower --json` works |
| `no valid prompts found` | Generate or copy `prompts.json`; see [Prompts file](#prompts-file-promptsjson) |

## Other steps

- **Step 1 — Model download**: [guide](../01-vllm-model-download/README.md)
- **Step 2 — Deploy vLLM**: [guide](../02-vllm-llama3-70b-fp8/README.md) (includes `06-lmstack-router.yaml`)
