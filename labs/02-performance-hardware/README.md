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

### Step 3 — Context size vs VRAM and speed

Same ≈3,000-token prompt every run; only `num_ctx` changes. **Run the two sweeps one after the other, never concurrently.**

```powershell
ollama stop qwen3.5:4b; ollama stop qwen3.5:9b
.\scripts\bench.ps1 -Model qwen3.5:4b -LogLines 120 -NumCtx 4096,8192,16384,32768,65536
ollama stop qwen3.5:4b
.\scripts\bench.ps1 -Model qwen3.5:9b -LogLines 120 -NumCtx 4096,8192,16384,32768,65536
ollama stop qwen3.5:9b
```

**Accidental experiment — concurrent sweeps (19:58):** both sweeps were started at the same time in separate terminals. The scheduler logged `llama-server model predicted to exceed available memory, evicting` before nearly every request and reloaded the other model each time; every request took ≈18.8 s. The log still confirmed the KV cache sizes: **1,024 MiB at 32K and 2,048 MiB at 64K** (exactly 128 MiB × 8 and × 16). Ollama's predicted need for the 9b at 64K was "13.2 GiB" vs ≈8.2 GiB from its own buffer breakdown — unexplained; check in the isolated run.

### Step 3 results (isolated sweeps, 2026-10-10)

Raw (2 runs per context; prompt = 120 synthetic log lines ≈ 5,064 tokens):

| Model | Ctx | LoadS (run 1) | PromptTok | PromptTokS | GenTokS | VramMiB | GPU |
|---|---|---|---|---|---|---|---|
| 4b | 4,096 | 3.99 | **2,050** | 4,036 / *47,626* | 120.1 / 120.3 | 5,162 | 100% |
| 4b | 8,192 | 5.04 | 5,063 | 4,908 / 5,186 | 117.0 / 118.6 | 5,372 | 100% |
| 4b | 16,384 | 5.12 | 5,064 | 4,914 / 5,093 | 117.2 / 114.9 | 5,644 | 100% |
| 4b | 32,768 | 5.13 | 5,064 | 4,814 / 5,059 | 115.4 / 113.5 | 6,195 | 100% |
| 4b | 65,536 | 5.34 | 5,064 | 4,569 / 4,664 | 113.5 / 113.9 | 7,170 | 100% |
| 9b | 4,096 | 6.18 | **2,050** | 3,090 / *36,996* | 84.8 / 84.0 | 7,805 | 100% |
| 9b | 8,192 | 6.14 | 5,068 | 3,444 / 3,513 | 85.5 / 85.5 | 8,054 | 100% |
| 9b | 16,384 | 6.09 | 5,069 | 3,323 / 3,422 | 85.2 / 85.2 | 8,207 | 100% |
| 9b | 32,768 | 6.11 | 5,068 | 3,345 / 3,430 | 85.6 / 85.5 | 8,718 | 100% |
| 9b | 65,536 | 6.07 | 5,069 | 3,327 / 3,431 | 85.7 / 85.7 | 9,765 | 100% |

| Check | Hypothesis | Observed |
|---|---|---|
| Prompt size | ≈3,000 tokens | ❌ ≈5,064 tokens (≈42 tokens per log line) |
| Truncation at default 4,096 ctx | — | ⚠️ **Silent**: only 2,050 of 5,064 tokens kept, no error. Cut from the **front** — the cache-busting marker was removed, so run 2 hit the prefix cache (impossible 47,626 tok/s) |
| VRAM growth 4K → 64K | ≈ +1.9 GB (KV 128 → 2,048 MiB) | ✅ 4b +2,008 MiB, 9b +1,960 MiB → **≈32 KiB of KV cache per token** (a standard 8B like Llama 3 8B: 128 KiB/token) |
| 9b − 4b at 4K | ≈2,590 MiB (Lab 01 estimates) | ✅ 2,643 MiB |
| 9b fits at 64K | ❌ predicted spill (≈10.8 GB needed) | ✅ fits: 9,765 / 10,240 MiB, 100% GPU, no slowdown. Prediction used a stale 2.4 GB baseline; idle was ≈1.3 GB this time. Only 475 MiB spare — not a safe default |
| Prompt processing | thousands tok/s; 9b ≈ half of 4b | ✅ 4b ≈5,000 tok/s (TTFT ≈1.0 s for 5K tokens), 9b ≈3,400 tok/s (≈1.5 s); ratio 1.47×, not 1.91× |
| Generation vs ctx | flat unless spilling | ✅ 9b flat ≈85.5; 4b 117 → 114 (−3%). Both slightly below Step 2 because 5K tokens are now in the context |
| `num_ctx` change reloads runner | yes | ✅ `LoadS` 4–6 s on run 1 of every context size → variants share a runner only if load-time settings match |

**Takeaways**

* **Set `num_ctx` deliberately.** The default 4,096 silently drops long inputs from the front.
* On this card, long context is cheap for qwen3.5 (≈2 GB for 64K) thanks to the hybrid architecture.
* Whether the 9b fits at 64K depends on what else is using the GPU at that moment.

### Step 4 — CPU vs GPU: partial offload curve

Documented way to force CPU: start the server with `CUDA_VISIBLE_DEVICES=-1` (all-or-nothing, requires a server restart). Used instead: the **request option `num_gpu`** (not in the Modelfile docs). Verified on 0.40.2: `num_gpu: 0` → `/api/ps` `size_vram = 0`, and `server.log` shows llama-server started with `-ngl 0` (llama.cpp's "number of GPU layers"), so partial values work too. qwen3.5:4b has 33 offloadable layers.

Same `server.log` command line also showed: llama-server listens on `127.0.0.1:<random port>` behind Ollama (proxy → internal service), `--no-mmproj-offload` in CPU mode, and `--context-shift --keep 4` (relevant to truncation — Phase 3).

Hardware for CPU inference: i7-10700K (8C/16T), 4 × 16 GB DDR4 rated 3200 but **running at 2400 MT/s** (dual channel → 38.4 GB/s theoretical, vs ≈760 GB/s on the GPU).

```powershell
ollama stop qwen3.5:4b
.\scripts\bench.ps1 -Model qwen3.5:4b -LogLines 30 -NumPredict 128 -NumCtx 8192 -NumGpu 33,24,16,8,0
ollama stop qwen3.5:4b
```

### Step 4 results (qwen3.5:4b, ctx 8,192, ≈1,297-token prompt, 128 generated)

| GPU layers | GPU share | Gen predicted | **Gen measured** (tok/s) | % of full | Prompt (tok/s) | TTFT (prompt only) | VRAM MiB |
|---|---|---|---|---|---|---|---|
| 33 | 100% | ≈117 | **121.7 / 124.1** | 100% | 3,773 / 4,641 | ≈0.3 s | 5,405 |
| 24 | 73% | ≈25 | **24.0 / 22.8** | 19% | 2,373 / 2,219 | ≈0.6 s | 3,920 |
| 16 | 48% | ≈15 | **13.4 / 12.5** | 11% | 1,796 / 1,503 | ≈0.8 s | 3,321 |
| 8 | 24% | ≈10 | **8.1 / 7.7** | 6% | 1,334 / 1,169 | ≈1.0 s | 2,770 |
| 0 | 0% | 6–10 | **6.8 / 7.0** | 6% | **76 / 76** | **≈17 s** | 1,300 (= idle baseline) |

**Generation follows a serial model:** time per token ≈ (CPU layer share × 145 ms) + (GPU layer share × 8.1 ms). Predicted 22.0 / 12.7 / 8.9 tok/s for 24 / 16 / 8 layers vs measured 23.4 / 13.0 / 7.9. Moving 27% of layers to the CPU cost 81% of the speed. CPU-only ≈ 3 GB per token at ≈21 GB/s effective (≈55% of DDR4-2400 dual-channel theoretical).

**Prompt processing degrades gently** (8/33 layers on GPU is still 16× faster than CPU-only). Likely cause, not yet verified: for large batches llama.cpp copies CPU-resident weights over PCIe and runs the matrix math on the GPU ("op offload", disabled by `--no-op-offload`); for one-token generation the copy isn't worth it. Check in Phase 9.

**VRAM:** the first 9 layers removed freed 1,485 MiB, later groups of 8 only ≈570 MiB — likely the large output (vocabulary) layer leaves first. Unconfirmed.

**Takeaways:** a smaller model at 100% GPU beats a larger partially-offloaded one; CPU-only is usable for short prompts but not for long inputs (RAG, logs); a CPU-only inference pod is ≈18× slower generating and ≈55× slower on prompts than a GPU pod.

### Step 5 — Quantization: Q4_K_M vs Q8_0

**Concept.** Quantization stores each weight in fewer bits. Q8_0 uses 8 bits per weight (plus one scale per block of 32 weights). Q4_K_M uses about 4.5–5 bits, mixing 4- and 6-bit blocks ("K-quants"; the M means "medium": a few sensitive tensors keep more bits). Same model and parameter count, different precision. It's like compressing an image: smaller and faster to move around, with some loss of detail. Generation is memory-bandwidth bound (Step 2), so fewer bytes means faster generation. The question is how much quality that costs.

**Verified on this machine (manifests):**

| Tag | Weights blob | Projector blob | Note |
|---|---|---|---|
| `qwen3.5:4b` (= `4b-q4_K_M`) | 2,526 MiB | 644 MiB | `ollama show`: Q4_K_M |
| `qwen3.5:4b-q8_0` | 4,275 MiB (1.69×) | 644 MiB, **same digest** | `ollama show`: Q8_0 |

The vision projector is one shared blob, so the download added ≈4.2 GiB to disk, not the 5.2 GB `ollama list` shows. That's content-addressed storage, like shared container image layers.

**Predictions** (Step 2 model: ≈4.8 ms fixed + GPU weight bytes ÷ ≈775 GB/s):

| Metric | Q4_K_M | Q8_0 predicted |
|---|---|---|
| Weights on GPU | 2,515 MiB | ≈4,250 MiB (×1.69) |
| VRAM total (nvidia-smi) | measured | ≈ +1,750 MiB |
| Generation | ≈122 tok/s | **≈95 tok/s** (8.2 ms → 10.6 ms per token) |
| Prompt processing | measured | similar (compute bound, not bandwidth bound) |
| Quality on short factual questions | — | about the same score; some wording differs |

**Run** (one model at a time, sequentially):

```powershell
ollama stop qwen3.5:4b
.\scripts\bench.ps1 -Model qwen3.5:4b      -LogLines 30 -NumPredict 128
ollama stop qwen3.5:4b
.\scripts\bench.ps1 -Model qwen3.5:4b-q8_0 -LogLines 30 -NumPredict 128
.\scripts\quality.ps1 -Models qwen3.5:4b,qwen3.5:4b-q8_0
ollama stop qwen3.5:4b-q8_0
```

`scripts/quality.ps1` asks six fixed questions (temperature 0, seed 42, thinking off). It scores each answer with keyword regexes and saves the full answers to `results/raw/quality-<time>.md` (gitignored). `SameText` shows whether Q8_0's answer is word-for-word identical to Q4_K_M's. The `fake-flag` question asks about a kubectl flag that doesn't exist, so it is a hallucination trap. Regex scoring is crude: read the answers file before trusting a score.

`bench.ps1` fix: it now matches the model name exactly. The old wildcard `qwen3.5:4b*` would also have matched `qwen3.5:4b-q8_0`.

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
| Every request takes ≈19 s; constant `starting llama-server` | Two models that don't fit together are requested concurrently → scheduler evicts and reloads on each request (thrashing) | `server.log`: `predicted to exceed available memory, evicting`. Run one model at a time, or use one model for all clients |
| `ParserError: An empty pipe element is not allowed` | A `foreach (...) {}` statement piped directly | Assign to a variable first: `$r = foreach (...) {...}; $r \| Format-Table` |

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

**4. The 9b has 1.91× the GPU weights of the 4b but generates only 1.38× slower. What does that tell you?**

<details>
<summary>Answer</summary>

Time per token = fixed overhead + weights ÷ bandwidth. Solving with both models: the *extra* 2.41 GB is read at ≈775 GB/s (the 3080's rated 760 GB/s), and ≈4.8 ms per token is fixed overhead (kernel launches, CPU-side sampling and embeddings, Windows GPU scheduling). For small models the overhead dominates, so shrinking the model gives diminishing speed returns.

</details>

**5. At `num_ctx` 4096 the 5,064-token prompt became 2,050 tokens. How did the "impossible" 47,626 tok/s prove which end was cut?**

<details>
<summary>Answer</summary>

Each run starts with a unique marker to defeat the prefix cache. Run 2 was served from cache, so the marker must have been gone — i.e. truncation removed the **beginning** of the prompt and both runs kept an identical tail. Lesson: with the default context, long inputs lose their start (often the instructions or the earliest log lines) silently. Set `num_ctx` large enough, and check `prompt_eval_count` against what you sent.

</details>

**6. Why does qwen3.5 need only ≈32 KiB of KV cache per token while Llama 3 8B needs 128 KiB?**

<details>
<summary>Answer</summary>

Llama 3 8B: every one of 32 layers is full attention → 2 (K,V) × 32 layers × 8 KV heads × 128 dims × 2 bytes = 128 KiB/token. qwen3.5 is hybrid: most layers keep a fixed-size recurrent state (the constant 50 MiB) and only a few full-attention layers store K/V per token. Same context, a quarter of the memory — architecture matters as much as parameter count when sizing VRAM.

</details>

**7. The 9b fit at 64K today, with 475 MiB spare. Should 64K be its default on this machine?**

<details>
<summary>Answer</summary>

No. Idle VRAM drifted between ≈1.3 and ≈2.4 GB during this lab; on a busier desktop the same config spills to CPU or fails. It's like setting a pod's memory limit to the node's free memory measured at one moment. A safe default leaves margin (e.g. 9b at 32K ≈ 7.4 GB of model allocation, or 4b at 64K), and you raise it deliberately when needed.

</details>

**8. With 73% of layers on the GPU, why do you get only 19% of full generation speed?**

<details>
<summary>Answer</summary>

Each token passes through all layers in sequence, so per-token time is the **sum** of GPU-layer time and CPU-layer time. CPU layers are ≈18× slower (RAM ≈38 GB/s vs VRAM ≈760 GB/s), so even a small CPU share dominates: 0.27 × 145 ms + 0.73 × 8.1 ms ≈ 45 ms per token ≈ 22 tok/s. It's like a request path through several services — the slowest hop sets the latency, not the average.

</details>

**9. A 14B model would fit 70% on this GPU; the 9b fits 100%. Which would you choose for an interactive assistant, and when might the other choice win?**

<details>
<summary>Answer</summary>

Interactive: the 9b at 100% GPU (≈85 tok/s) — the partially-offloaded 14B would generate around 20 tok/s. The 14B could win for offline/batch jobs where answer quality matters more than latency, or for prompt-heavy tasks with short outputs, since prompt processing degrades much more gently under partial offload. Measure quality before deciding (Phase 3).

</details>

**10. Q8_0 has 1.69× the weight bytes of Q4_K_M. Why should generation be less than 1.69× slower?**

<details>
<summary>Answer</summary>

Per-token time has a fixed part (≈4.8 ms here: kernel launches, sampling, the non-weight work) plus a part that scales with weight bytes read. Only the second part grows 1.69×: 4.8 + 3.4 ms → 4.8 + 5.8 ms, i.e. ≈1.3× slower. Same as a request with fixed connection overhead: doubling the payload doesn't double the latency.

</details>

**11. When would you pick Q8_0 over Q4_K_M on this GPU, and when would you pick a 9b at Q4 over a 4b at Q8?**

<details>
<summary>Answer</summary>

Q8_0 when it fits with your needed context and you measured a quality gain on your own tasks (e.g. precise code or numbers) worth the speed and VRAM. In general, more parameters at 4-bit beat fewer parameters at 8-bit for similar memory: the 9b Q4 (≈4.8 GiB of weights) is likely a better use of VRAM than the 4b Q8 (≈4.2 GiB). Verify on your own question set rather than trusting the rule of thumb.

</details>
