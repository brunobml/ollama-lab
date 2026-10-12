# VRAM & Speed Calculator (RTX 3080 10 GB, Ollama 0.40.2)

How to predict whether a model fits on this GPU and how fast it will run, **before** loading it. Every constant here was measured in Lab 01 and Lab 02 (2026-10-10 → 2026-10-12); sources are linked per section. Values marked *unverified* are theory, not measurements.

Like sizing a pod: add up the requests, compare with what the node has free *right now*, and keep headroom.

## 1. Does it fit?

```text
GPU total (10,240 MiB)
 ├─ Desktop / other apps (idle)     1,300 – 2,400 MiB   ← varies; measure it now
 └─ Ollama model allocation
     ├─ LLM weights on GPU          from the GGUF file, minus host-side embeddings
     ├─ Vision projector            the mmproj blob (loaded even for text-only prompts)
     ├─ KV cache                    32 KiB × num_ctx        (f16, qwen3.5)
     ├─ Recurrent state             50 MiB fixed            (qwen3.5 hybrid layers)
     ├─ Compute buffers             ≈290 – 340 MiB (+ a little with larger context)
     └─ CUDA runtime overhead       ≈240 MiB
```

**Model allocation (MiB) ≈ weights_gpu + projector + num_ctx / 32 + 50 + compute + 240**

**Fits if:** model allocation ≤ 10,240 − idle_now − margin.

* `idle_now`: `nvidia-smi` (or `scripts/check-vram.ps1`) with no model loaded. It has ranged from ≈1,300 to ≈2,400 MiB.
* Margin: keep ≥ 500 MiB for desktop spikes. The lab's working ceiling is **8,500 MiB** for Ollama allocations.

### Measured constants

| Component | qwen3.5:4b Q4_K_M | qwen3.5:9b Q4_K_M | qwen3.5:4b Q8_0 | Source |
|---|---|---|---|---|
| LLM weights on GPU | 2,515 | 4,812 | ≈4,250 | `server.log` (Q8_0: file ratio) |
| Weights kept in system RAM | 497 | 546 | — | `server.log` (likely token embeddings) |
| Vision projector | ≈640 | ≈880 | ≈640 (same blob) | inferred from file size |
| Compute buffers | 291 | 340 | — | `server.log` |
| KV cache at 4,096 | 128 | 128 | 128 | `server.log` |
| Recurrent state | 50 | 50 | 50 | `server.log` |
| CUDA overhead | ≈240 | ≈240 | ≈240 | remainder vs `nvidia-smi` |
| **Allocation at 4,096** | **3,860** measured | **≈6,450** | **5,608** measured | Lab 01, Lab 02 Step 1/5 |

### KV cache

KV cache = 2 (K and V) × attention layers × KV heads × head dim × num_ctx × bytes per element.

* **Measured for qwen3.5 (both sizes): 32 KiB per token** → 128 MiB at 4K, 1,024 MiB at 32K, 2,048 MiB at 64K. The 4b and 9b have the same attention layout, so the same KV cache.
* A model where every layer uses full attention costs much more. Llama 3 8B: 2 × 32 × 8 × 128 × 2 bytes = **128 KiB per token** (4× qwen3.5). *Calculated from published architecture, not measured here.*
* It is allocated **at load for the full `num_ctx`**, not as the conversation grows. Changing `num_ctx` reloads the runner.
* *Unverified:* `OLLAMA_KV_CACHE_TYPE=q8_0` should halve it (q4_0 quarter), and needs flash attention. Requires a server environment change, so not tested yet.

### Worked examples (prediction vs measurement)

| Config | Predicted allocation | Measured | Error |
|---|---|---|---|
| 4b Q4, ctx 4,096 | 2,515 + 640 + 128 + 50 + 291 + 240 = **3,864** | 3,860 | ≈0% |
| 4b Q4, ctx 65,536 | 3,864 + 1,920 (extra KV) = **5,784** | ≈5,868 (3,860 + 2,008) | −1.4% |
| 9b Q4, ctx 65,536 | 6,450 + 1,920 = **8,370** | ≈8,410 (9,765 total − ≈1,355 idle) | −0.5% |
| 4b Q8, ctx 4,096 | 3,864 + 1,735 (extra weights) = **5,599** | 5,608 (7,160 − 1,552 idle) | ≈0% |

The formula slightly under-predicts at long context: compute buffers also grow a little (≈+88 MiB from 4K to 64K on the 4b).

### Caveats

* **Ollama decides placement from its own estimate, not from this formula.** For the 9b at 64K it logged a predicted need of "13.2 GiB" while its buffers totalled ≈8.2 GiB (unexplained). If its estimate exceeds free VRAM, it offloads layers to the CPU or evicts other models even when the real allocation would fit. Check `ollama ps` (`100% GPU`?) after every load.
* **`ollama ps` SIZE underestimates** (4b: "3.1 GB" vs 3,860 MiB measured). Use `nvidia-smi` deltas.
* **Two loaded models add up.** Two models that don't fit together, requested concurrently, cause evict-and-reload on every request (≈19 s each). Lab 02 Step 3.

## 2. How fast will it be?

### Generation (100% on GPU)

Each token reads all GPU weights once, so generation is limited by memory bandwidth:

**time per token ≈ 4.7 ms + weights_gpu bytes ÷ 770 GB/s**

* ≈770 GB/s: measured twice (4b vs 9b; Q4 vs Q8). The RTX 3080 is rated at 760 GB/s.
* ≈4.7 ms: fixed per-token cost (kernel launches, sampling, CPU-side work). For small models it is more than half the total, so shrinking a small model further gains little speed.

| Config | Predicted | Measured |
|---|---|---|
| 4b Q4 (2,515 MiB) | 8.1 ms → **123 tok/s** | 117–125 |
| 4b Q8 (≈4,250 MiB) | 10.5 ms → **95 tok/s** | 95–97 |
| 9b Q4 (4,812 MiB) | 11.2 ms → **89 tok/s** | ≈85 (73–88 in one run) |

Context length barely matters for qwen3.5 (≈−3% at 5K tokens in context).

### Partial CPU offload

Layers run in sequence, so per-token times **add up**:

**time per token ≈ 4.7 ms + weights_gpu ÷ 770 GB/s + weights_cpu ÷ ≈21 GB/s**

≈21 GB/s is the measured effective CPU rate (DDR4-2400 dual channel, 38.4 GB/s theoretical). Check on the 4b with 24 of 33 layers on the GPU: 4.7 + 2.5 + 34 ≈ 41 ms → **24 tok/s** predicted, 23.4 measured. One-quarter of the weights on the CPU takes ≈80% of the time.

### Prompt processing and time to first token

Prompt tokens are processed in parallel, so this is compute bound, not bandwidth bound.

| Config | Prompt rate | 5,000-token prompt |
|---|---|---|
| 4b Q4, 100% GPU | ≈4,500 – 5,000 tok/s | ≈1.0 s |
| 9b Q4, 100% GPU | ≈3,400 tok/s | ≈1.5 s |
| 4b Q4, 8 of 33 layers on GPU | ≈1,200 tok/s | ≈4 s |
| 4b Q4, CPU only | 76 tok/s | ≈66 s |

**TTFT ≈ load time (if cold, ≈4–6 s) + prompt tokens ÷ prompt rate.** The first-ever load after an Ollama or driver upgrade is much slower (≈40 s: CUDA kernel compilation, cached afterwards).

## 3. Checklist before choosing a config

1. Measure `idle_now` with `nvidia-smi`.
2. Add up the allocation with the formula above for your `num_ctx`. Set `num_ctx` deliberately: the default 4,096 silently cuts long prompts from the front.
3. Leave ≥ 500 MiB margin and stay under the 8,500 MiB ceiling.
4. Load, then check `ollama ps` shows `100% GPU` and `nvidia-smi` matches the prediction.
5. Choose by quality on your own questions, not by size alone. On this card a larger model at Q4 was a better use of VRAM than a smaller one at Q8 (Lab 02 Step 5b).
