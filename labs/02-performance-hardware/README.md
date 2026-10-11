# Lab 02 — Hardware Sizing & Performance

## 1. Core Concept & DevOps Analogy

LLM performance is a capacity-planning problem with three separate costs:

| Cost | When | Bottleneck | Analogy |
|---|---|---|---|
| **Load** | Once per runner start | Disk, security scanning, process + CUDA init | Container cold start (image pull + init) |
| **Prompt processing** | Per request, all prompt tokens in parallel | GPU compute | Batch job |
| **Generation** | Per output token, sequential | GPU memory bandwidth | Latency-bound request loop |

VRAM is the hard limit: weights + KV cache (grows with context) + recurrent state + buffers + CUDA overhead. When it doesn't fit, layers spill to CPU/RAM and generation slows sharply — like a pod swapping to disk.

## 2. Prerequisites & Hardware Budget

* Lab 01 complete; `scripts/check-vram.ps1`.
* Models: `qwen3.5:4b` (3.3 GB), `qwen3.5:9b` (6.6 GB: `model.gguf` 5,629,109,120 B + `mmproj.gguf` 921,704,832 B).
* Expected `qwen3.5:9b` VRAM at 4,096 context: ≈6.5–7.0 GB (estimate from Lab 01's breakdown — to be measured). Idle baseline ≈2.4 GB, so total use ≈9 GB of 10.24 GB.

## 3. Architecture

See [`docs/architecture.md`](../../docs/architecture.md) for the memory breakdown measured in Lab 01.

## 4. Step-by-Step Guide

### Step 1 — The first-load mystery (`qwen3.5:9b`)

Lab 01 saw a 40.7 s first-ever load vs 3–4 s reloads on a ~7 GB/s NVMe. Hypothesis: Windows Defender real-time scanning of a newly written blob on first open.

Terminal A: `.\scripts\check-vram.ps1`

Terminal C — Defender CPU (per-core %, so 100 = one full core; 16 threads = 1600 max):

```powershell
while ($true) {
    $v = (Get-Counter '\Process(MsMpEng)\% Processor Time').CounterSamples.CookedValue
    '{0:HH:mm:ss}  Defender CPU {1,6:N1}%' -f (Get-Date), $v
}
```

Wait until Defender CPU is near 0 (it may still be scanning the download). Then in Terminal B:

```powershell
ollama run qwen3.5:9b --verbose --think=false "Say OK."   # first-ever load
ollama stop qwen3.5:9b
ollama run qwen3.5:9b --verbose --think=false "Say OK."   # second load
ollama ps
ollama stop qwen3.5:9b
```

### Step 2 — Fair speed benchmark: 4b vs 9b (`scripts/bench.ps1`)

Calls `POST /api/generate` directly so every run does identical work: thinking off, `temperature 0`, `seed 42`, fixed `num_predict`, fixed `num_ctx`. Each run prefixes a unique marker so the prompt is never served from the prefix cache.

```powershell
.\scripts\bench.ps1 -Model qwen3.5:4b
.\scripts\bench.ps1 -Model qwen3.5:9b
```

## 5. Hypotheses → Observed Metrics

### Step 1 results (2026-10-10)

| Check | Hypothesis | Observed |
|---|---|---|
| Defender before test | busy scanning the fresh download, then ≈0 | 23% right after download, 0–5% at test time ✅ |
| First load (9b) | much slower than second; Defender CPU high | ❌ **5.07 s** — no penalty; Defender peaked at 48% of one core |
| Second load (9b) | ≈4–6 s | 5.03 s ✅ (server log: 4.0 s) |
| VRAM (9b, ctx 4,096) | ≈6.5–7.0 GB | ≈6,450 MiB ≈ 6.8 GB (log breakdown; monitor lines not captured) ✅ |
| `ollama ps` | `100% GPU` | `5.6 GB`, `100% GPU`, `4096` ✅ — `SIZE` underestimates again |
| Generation rate | below 4b's ≈117 tok/s | not measurable: "Say OK." generates 2 tokens → Step 2 |

**Root cause of Lab 01's 40.7 s first load: NVIDIA's CUDA compute cache.** `%APPDATA%\NVIDIA\ComputeCache` gained 21 files (70 MB) at 19:10–19:11 — exactly during that load — and none during the 9b loads. The driver compiled GPU kernels on llama-server's first CUDA run and cached them; every later load of *any* model reuses them. It's a one-time cost per installation/driver version, like a cold CI build cache. Expect it again after an Ollama or driver upgrade, and on every fresh Kubernetes node (Phase 8).

**Memory comparison (MiB, from `server.log`):**

| Consumer | 4b | 9b |
|---|---|---|
| Weights on GPU | 2,515 | 4,812 |
| Weights in system RAM | 497 | 546 |
| Vision projector (inferred from file) | ≈640 | ≈880 |
| Compute buffers | 291 | 340 |
| KV cache (ctx 4,096) | 128 | **128** |
| Recurrent state | 50 | 50 |
| GPU total | ≈3,860 (measured) | ≈6,450 (estimated) |

KV cache is identical: its size depends on attention-layer count × KV heads × head dimension, not total parameters.

## 6. Troubleshooting & Failure Modes

| Symptom | Likely cause | Check |
|---|---|---|
| Load takes tens of seconds | Security scan of model blobs | Defender CPU during load |
| `ollama ps` shows CPU/GPU split | Not enough free VRAM | `check-vram.ps1` before loading |

## 7. Cleanup

`ollama stop qwen3.5:9b`. Keep both models for later steps.

## 8. Check Your Understanding

**1. The first-ever load took 40 s, later loads of any model 3–5 s. Why would a brand-new Kubernetes GPU node show the same pattern?**

<details>
<summary>Answer</summary>

The 40 s was the NVIDIA driver compiling CUDA kernels on llama-server's first run and caching them in `%APPDATA%\NVIDIA\ComputeCache` (Linux: `~/.nv/ComputeCache`). A fresh node — or a container whose filesystem is discarded — has no cache, so the first pod pays the compile cost. Mitigations: a warm-up request in a startup probe window, persisting the cache on a volume, or using images whose binaries already contain code for your GPU architecture.

</details>

**2. The 9b has 2× the parameters of the 4b, yet the same 128 MiB KV cache. What does KV cache size actually depend on?**

<details>
<summary>Answer</summary>

`2 × attention_layers × kv_heads × head_dim × context × bytes_per_element`. It scales with **context length** and the attention geometry, not total parameter count. Larger models often add width in the feed-forward layers (more weights) without changing KV heads. In hybrid models like qwen3.5, only the full-attention layers count; the rest keep a fixed-size recurrent state.

</details>

**3. Why can't a "Say OK." prompt measure generation speed?**

<details>
<summary>Answer</summary>

It generates 2 tokens in ≈15 ms — timer resolution, scheduling jitter, and per-request overhead dominate. Rates need hundreds of tokens per sample, several runs, and fixed settings (`num_predict`, `temperature`, `seed`, thinking off) so models are compared on equal work.

</details>
