# vLLM model cache

Part of the [end-to-end workload overview](../README.md) — **step 1 of 3**.

Download Hugging Face models to `/mnt/data/vllm-model-cache/` for use with vLLM deployments on the cluster.

This workflow targets a **single bare-metal node with 8 AMD GPUs**. Weights are stored on that node's local disk via `hostPath` (`/mnt/data/vllm-model-cache/`) so the vLLM pod can mount them offline without re-fetching from Hugging Face on every restart. Run the download **on the GPU node** that will serve the model (or on the node pinned by `nodeSelector` in the deployment).

## Models

| Model | Hugging Face ID | Local path |
|-------|-----------------|------------|
| Llama 3.1 70B Instruct FP8 | `meta-llama/Llama-3.1-70B-Instruct-FP8` | `/mnt/data/vllm-model-cache/meta-llama/Llama-3.1-70B-Instruct-FP8` |
| Qwen3.5 122B A10B FP8 | `Qwen/Qwen3.5-122B-A10B-FP8` | `/mnt/data/vllm-model-cache/Qwen/Qwen3.5-122B-A10B-FP8` |

---

## Download: Llama 3.1 70B Instruct FP8

### Prerequisites

1. **Single bare-metal node with 8 AMD GPUs** — the Llama 3.1 70B FP8 vLLM deployment uses tensor parallel size 8 and mounts model weights from a host path on one GPU node. Download the model on that same node so the files land under `/mnt/data/vllm-model-cache/` where the pod expects them.

2. **Accept the Llama license** on Hugging Face (gated model):  
   https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct-FP8

3. **Create a Hugging Face token** with read access:  
   https://huggingface.co/settings/tokens

4. **Ensure sufficient disk space** on the GPU node — FP8 70B weights are roughly 70–80 GB. Plan for ~100 GB free on the volume.

5. **Create the target directory** on the GPU node (if it does not exist):

```bash
sudo mkdir -p /mnt/data/vllm-model-cache/meta-llama/Llama-3.1-70B-Instruct-FP8
sudo chown -R "$USER:$USER" /mnt/data/vllm-model-cache
```

### Option A: Hugging Face CLI (recommended)

```bash
pip install "huggingface_hub[cli]" hf_transfer

export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
huggingface-cli login --token "$HF_TOKEN"

export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_ID="meta-llama/Llama-3.1-70B-Instruct-FP8"
MODEL_DIR="/mnt/data/vllm-model-cache/meta-llama/Llama-3.1-70B-Instruct-FP8"

hf download "$MODEL_ID" \
  --local-dir "$MODEL_DIR" \
  --token "$HF_TOKEN"
```

Verify the download:

```bash
ls -lh "$MODEL_DIR"
test -f "$MODEL_DIR/config.json" && echo "Download OK"
```

### Option B: Python `snapshot_download`

```bash
pip install huggingface_hub hf_transfer

export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
export HF_HUB_ENABLE_HF_TRANSFER=1

python3 <<'EOF'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.1-70B-Instruct-FP8",
    local_dir="/mnt/data/vllm-model-cache/meta-llama/Llama-3.1-70B-Instruct-FP8",
    token=os.environ["HF_TOKEN"],
)
print("Done.")
EOF
```

### Idempotent re-run

Skip the download if the model is already present:

```bash
MODEL_DIR="/mnt/data/vllm-model-cache/meta-llama/Llama-3.1-70B-Instruct-FP8"

if [ -f "$MODEL_DIR/config.json" ]; then
  echo "Model already exists — skipping."
else
  hf download meta-llama/Llama-3.1-70B-Instruct-FP8 \
    --local-dir "$MODEL_DIR" \
    --token "$HF_TOKEN"
fi
```

### Use with vLLM

Point vLLM at the local path:

```bash
vllm serve /mnt/data/vllm-model-cache/meta-llama/Llama-3.1-70B-Instruct-FP8
```

In Kubernetes, mount the host path in your pod spec:

```yaml
volumes:
  - name: model-dir
    hostPath:
      path: /mnt/data/vllm-model-cache/meta-llama/Llama-3.1-70B-Instruct-FP8
      type: Directory
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `403 Forbidden` / gated repo | Accept the model license on Hugging Face and retry with a valid token |
| `401 Unauthorized` | Regenerate your token; confirm `HF_TOKEN` is exported in the shell |
| Slow download | Install `hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1` |
| Permission denied on `/mnt/data/...` | Create the directory with `sudo` and fix ownership: `sudo chown -R $USER:$USER /mnt/data/vllm-model-cache` |

---

## Next step

Continue to **step 2 — Deploy vLLM**: [guide](../02-vllm-llama3-70b-fp8/README.md)
