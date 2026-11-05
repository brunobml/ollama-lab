# Lab 09 — Runtime Comparison: Ollama, llama.cpp and LM Studio

## 1. Core Concept & DevOps Analogy

Ollama, llama.cpp's `llama-server` and LM Studio all run GGUF models. The question is what each one adds or takes away around the same inference engine.

* **llama.cpp** is the engine: model loading, the GPU kernels, the KV cache, sampling and an HTTP server. It's the equivalent of `containerd` or `runc`.
* **Ollama** is the platform around it: a model registry and store, an API, chat templates, a scheduler that loads and unloads models, and defaults. It's the equivalent of Docker, or a small kubelet. **On this machine, Ollama 0.40.13 literally runs llama.cpp's `llama-server.exe`** as its runner.
* **LM Studio** is a desktop application with its own model catalog, GUI and API server, also built on llama.cpp (among other engines).

So a fair comparison holds the engine constant where it can, then asks what changes: speed, memory, defaults, concurrency, APIs, and what's left for the operator to do.

## 2. Prerequisites & Hardware Budget

* Windows Ollama 0.40.13 running, with **nothing loaded** (each script checks). The same `qwen3.5:4b` GGUF from the Ollama store is used everywhere: no model download.
* Only one engine holds the model at a time: ≈3.9 GB of VRAM, within the 8,500 MiB ceiling.
* No lab containers or clusters running (each script checks).
* Downloads, each approved in advance (2026-11-05):
  * Step 1: none.
  * Step 2: the official llama.cpp b11963 CUDA 13.4 build (165 MB + 424 MB cudart, checked against GitHub's sha256).
  * LM Studio: shown and approved at its own step.

## 3. Architecture

```mermaid
flowchart LR
    C[client] -->|/api/generate, /api/chat| O[Ollama server<br/>Go: API, templates,<br/>scheduler, defaults]
    O -->|starts, proxies to| L1[llama-server.exe<br/>bundled, b11963]
    C -->|/completion, /v1/chat/completions| L2[llama-server.exe<br/>same binary, started by the script]
    L1 --> G[(GGUF blob<br/>Ollama store)]
    L2 --> G
```

**Verified before Step 1 (2026-11-05):**
* **Nothing else is installed:**
  * `llama-server`, `llama-cli` and `lms` are not on `PATH`;
  * LM Studio is not installed (no `%LOCALAPPDATA%\Programs\LM Studio`, no `~/.lmstudio`);
  * winget lists only Ollama 0.40.13 (0.40.14 is available, not applied).
* **Ollama ships llama.cpp:**
  * `lib\ollama\llama-server.exe`, with `libllama.dll`, `libmtmd.dll`, `ggml-*.dll`, and GPU variants in `cuda_v12`, `cuda_v13`, `rocm_v7_1` and `vulkan`. `cuda_v13` (903 MB) is used on this machine.
  * `llama-server --version` gives `0.8.0-dev (build 1, commit 39febd2ad)`, which is upstream commit `39febd2adb94`, 2026-10-28, "sycl: work-group argmax for short TOP_K rows". That's exactly release **b11963**.
* **Ollama's command line** (server.log): `llama-server.exe --model <blob> --port <random> --host 127.0.0.1 --no-webui --offline -c 4096 -np 1 --log-verbosity 4 --no-log-prefix --no-log-timestamps --no-jinja --chat-template chatml --mmproj <projector blob> --load-mode none --flash-attn auto -b 512 -ub 512 --context-shift --keep 4`.
* **Its environment** (`llm/llama_server.go`): `PATH` = the exe's folder, then the GPU variant's folder. It also sets **`GGML_BACKEND_PATH`** = that variant's `ggml-cuda.dll`.
  * Without `GGML_BACKEND_PATH`, the same binary loaded the model **on the CPU without any error** (`n_threads = 8`, VRAM unchanged).
* **The model is a plain GGUF:**
  * the 4b's model layer `sha256-a9d0a3e1…` (2.65 GB) and projector layer `sha256-e56a899b…` (676 MB) both start with `GGUF`;
  * the template layer is just `{{ .Prompt }}`;
  * **the params layer is `{"presence_penalty":1.5,"temperature":1,"top_k":20,"top_p":0.95}`**. Ollama applies these to every request that doesn't override them.
* **llama-server standalone** (127.0.0.1:11440): healthy 5.2 s after process start, +≈3,900 MiB.
  * It warns `security: no API key is set and CORS allows all origins`. Ollama restricts `OLLAMA_ORIGINS` to localhost-style origins by default.
* **The probe: the same raw ChatML prompt to both engines, temperature 0, median of 5:**

  | sampler settings | llama-server | Ollama | text |
  |---|---:|---:|---|
  | none given: each engine's own defaults | 144.7 tok/s, 61 tokens | 122.7 tok/s, 57 tokens | different from character 94 |
  | Ollama's (`presence_penalty 1.5, top_k 20, top_p 0.95`) | 118.0, 57 | 120.8, 57 | **byte-identical** |
  | llama-server's (`presence_penalty 0, top_k 40, top_p 0.95`) | 143.0, 61 | 145.1, 61 | **byte-identical** |

  The 18% gap and the different text both come from **sampler defaults, not the runtime**: about 1.2 ms per token. Every Ollama tok/s in Labs 02–08 was measured with `presence_penalty 1.5` on, and every "temperature 0" answer still had the penalty applied.
* **LM Studio:** 0.4.28-2 for Windows x64, a 626 MB installer (`installers.lmstudio.ai`, built 2026-10-31). It has a headless daemon (`llmster`) and a CLI (`lms`). Not installed yet.
* **Official llama.cpp b11963 Windows assets:**
  * `llama-b11963-bin-win-cuda-13.4-x64.zip`, 165.0 MB, `sha256:36e5f512…`;
  * `cudart-llama-bin-win-cuda-13.4-x64.zip`, 423.5 MB, `sha256:738f8c25…`.

## 4. Step-by-Step Guide

### Step 1 — Ollama vs its own engine

**What we're learning:** what Ollama costs and adds when the engine, the binary and the weights are identical. If the numbers differ, the cause is in Ollama's layer: defaults, templates, API, scheduler. It isn't in llama.cpp.

**What runs** (`scripts/engine_compare.py`, ≈8 min, no downloads):
* **The engine:** the bundled `llama-server.exe` started by the script on `127.0.0.1:11440`, with Ollama's flags and environment and the same GGUF blobs. It's stopped through its own process handle. One engine at a time holds the model.
* **A. Cold start:** an Ollama load, against llama-server from process start to healthy; VRAM for each.
* **B. The same raw prompt** under four sampler settings, on both engines: Ollama's defaults, llama-server's defaults, the penalty alone, and `top_k 20` alone. Speed and byte-identity are compared, plus prompt-eval speed on a ≈2.5K-token prompt.
* **C. Chat:** Ollama `/api/chat` (`think: false`) against llama-server `/v1/chat/completions` (chatml template): prompt tokens and text.
* **D. 4 concurrent requests:** llama-server with `-np 1`, `2` and `4`, against Ollama (which runs qwen3.5 with `-np 1`, Lab 04). Aggregate and per-request tok/s, and VRAM.
* **E. Per-request overhead:** 1-token requests, median of 20.

```powershell
ollama stop qwen3.5:4b                                   # nothing loaded (the script checks)
.\.venv\Scripts\python scripts\engine_compare.py         # ≈8 min
```

**Predictions** (A–C partly from the probe; D and E written before any run):

| Measure | Prediction | Why |
|---|---|---|
| A | Ollama first request 5–8 s; llama-server healthy in 5–6 s; +≈3,900 MiB for both | Lab 08 Step 1: Windows reload 4–5 s; probe 5.2 s |
| B: Ollama's / llama-server's defaults | ≈120 / ≈144 tok/s on **both** engines; byte-identical text within each row | Probe |
| B: penalty alone / `top_k 20` alone | the penalty carries the cost (≈120 tok/s); `top_k 20` alone ≈144 (low confidence) | A presence penalty touches every token in the 248K vocabulary; `top_k` should be cheaper |
| B: long prompt | prompt eval equal within ±5% | The same kernels; Lab 08 Step 1 (prompt eval equal across builds) |
| C | prompt tokens differ; text not byte-identical (low confidence) | Ollama renders qwen3.5 with its own renderer (`renderer=qwen3.5`, Lab 08 logs), llama-server with the generic chatml template; with `think: false` Ollama may add an empty think block |
| D: llama-server `-np 1` and Ollama | serialized: aggregate ≈ one stream (≈140 tok/s), per-request walls staggered | One slot |
| D: `-np 2` / `-np 4` | aggregate ≈1.6× / ≈2.5× of one stream; per-request tok/s lower; VRAM +100–300 MiB per extra slot (low confidence: whether qwen3.5's recurrent layers batch well at all) | Batched decoding shares each weight read; the hybrid model's per-slot state is small (Lab 01: 50 MiB recurrent + 32 KiB/token KV) |
| E | Ollama adds 5–20 ms per request | Go HTTP, template render, scheduler lookup, proxying to the runner |

### Step 2 — Two builds of the same commit: Ollama's bundled llama-server vs the official b11963

**What we're learning:** whether "llama.cpp" is one thing. The same source can be built with different compilers, CUDA toolkits and GPU-architecture lists. This step holds the commit constant and varies only the build, the way you'd compare two images built from one Dockerfile on different base images.

**Download** (approved 2026-11-05, D-17): into `tools\` (gitignored), from `github.com/ggml-org/llama.cpp/releases/tag/b11963`:
* `llama-b11963-bin-win-cuda-13.4-x64.zip`, 164,987,385 bytes, sha256 `36e5f512b29879031eb6693a79a9caa4bf9c82fb50523edd0899273359627817`;
* `cudart-llama-bin-win-cuda-13.4-x64.zip`, 423,535,356 bytes, sha256 `738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668`.

Both hashes match the release's published digests. They were extracted together into `tools\llama.cpp-b11963\`: 731 MB, including `llama-bench`, `llama-batched-bench`, `llama-cli` and `llama-perplexity`, none of which Ollama ships.

```powershell
# what was run (Git Bash); repeat only if tools\ is gone
curl -sL -o tools/downloads/<zip> https://github.com/ggml-org/llama.cpp/releases/download/b11963/<zip>
sha256sum tools/downloads/*.zip        # compare with the hashes above
unzip -q <zip> -d tools/llama.cpp-b11963
```

**Verified before the run:**

| | bundled with Ollama | official b11963 |
|---|---|---|
| `--version` | `0.8.0-dev (build 1, commit 39febd2ad)` | `0.8.0-dev (build 11963, commit 39febd2ad)` |
| compiler | Clang 18.1.8 | Clang 20.1.8 |
| CUDA runtime (`cudart64_13.dll`) | 6.14.11.**13000** (13.0) | 6.14.11.**13040** (13.4) |
| cuBLAS / cuBLASLt | 13.0 (50.3 / 477.9 MB) | 13.7 (54.9 / 492.8 MB) |
| `ggml-cuda.dll` | 141.0 MB | 155.0 MB |
| backend loading | needs `GGML_BACKEND_PATH` (Step 1) | self-contained: backends next to the exe |

* **The official build runs on the GPU** with the driver's CUDA 13.1, so CUDA's minor-version compatibility covers the 13.4 runtime. It was healthy in 6.2 s on its first start, used +≈3,900 MiB, and generated 140.9–143.1 tok/s with llama-server's sampler defaults. The bundled build gave 142.8 in Step 1.
* **A mistake of mine in the probe:** PowerShell variable names aren't case-sensitive, so `$P = '<prompt>'` overwrote `$p`, the process handle. The server kept running until I stopped it by its executable path, `tools\llama.cpp-b11963\llama-server.exe`, which only the lab's copy has. The script uses Python and its own process handles.

**What runs** (`scripts/build_compare.py`, ≈8 min):
* **A + B, 4 server processes in turn** (bundled, official, bundled, official), with Step 1's flags and the same GGUF blobs. For each:
  * process → healthy, and VRAM;
  * generation with Ollama's and with llama-server's sampler options (median of 5);
  * prompt processing on unique 1,640-token prompts (median of 3);
  * whether the text is byte-identical across all 4 processes.
* **C.** Each build with `-np 4` and 4 concurrent requests.
* **D.** `llama-bench -p 512 -n 128 -r 5 -fa auto` on the official build, the figure llama.cpp's own tooling reports.

```powershell
ollama stop qwen3.5:4b
.\.venv\Scripts\python scripts\build_compare.py          # ≈8 min
```

**Predictions** (written before the first full run):

| Measure | Prediction | Why |
|---|---|---|
| A | healthy in 4.5–6.5 s for both; VRAM within ±50 MiB | Same model, same buffers; the cuBLAS workspace may differ slightly |
| B: generation | equal within ±3% for both sampler settings | Decoding one token uses llama.cpp's own CUDA kernels (quantized matrix-vector), compiled from the same source |
| B: prompt processing | may differ by up to ±10% (low confidence) | Batched prefill can go through cuBLAS, and the versions differ (13.0 vs 13.7) |
| B: text | byte-identical across all 4 processes (moderate confidence) | Same kernels; Lab 08 Step 1 found even the Linux container build byte-identical. A different cuBLAS could change rounding in prefill |
| C | aggregate within ±5% (≈290 tok/s) | Step 1: 288–294 for the bundled build |
| D | tg128 ≈ 145–155 tok/s (a little above the server: no HTTP, no sampler); pp512 ≈ 5–7K tok/s | Step 1: server 143 and ≈5.6K at 1,640 tokens |

### Step 3 — LM Studio: what a desktop runtime installs, and what it runs

**What we're learning:** the third kind of runtime, a desktop application, from an operator's point of view. What does it put on the machine? What does it listen on? What engine does it actually run, and with which defaults? We find out by measuring, not from its docs.

**Approved 2026-11-05.** I downloaded and checked the installer; you run it.
* **The installer:** `LM-Studio-0.4.28-2-x64.exe` from `installers.lmstudio.ai/win32/x64/0.4.28-2/`.
  * 625,735,320 bytes, which matches the server's `Content-Length`; sha256 `442aa5cdb5b7d0a6693ba6d11e1663838afb2cdd8f5bfcc538cad23f00f0cf68`. LM Studio publishes no hash to compare against.
  * **Authenticode `Valid`:** signer *Element Labs Inc.* (Brooklyn, NY), issuer *DigiCert Trusted G4 Code Signing RSA4096 SHA384 2021 CA1*, timestamped by DigiCert, certificate valid until 2027-08-31.
  * ProductVersion `0.4.28+2`.
* **The docs disagree on where it installs:** one guide says `%LOCALAPPDATA%\LM-Studio`, another says the wizard asks. The `lms` CLI lives in `~\.lmstudio\bin`, and it needs the app to have run once. So `scripts/system_snapshot.ps1` diffs the machine instead. It takes a read-only snapshot of:
  * user and machine `PATH`;
  * the HKCU and HKLM Run keys and the Startup folder;
  * Start-menu entries, HKCU uninstall entries and URL protocol handlers;
  * scheduled tasks, services and matching firewall rules;
  * AppData, LocalAppData and the Programs folder;
  * the home folder's dot-directories, Program Files, and listening ports.

  The "before" snapshot was taken 2026-11-05 12:11.
* **The model:** `lms import -L` hard-links the existing 4b blob into LM Studio's model folder. That's the same C: volume, so no copy and no download.
* **Settings to keep off:** "run on login" and "serve on local network". The API's exposure is checked by connecting from the LAN address. Lab 08 found Windows' listener tables unreliable on this build.

**To install** (you):
1. Run `tools\downloads\LM-Studio-0.4.28-2-x64.exe`, accept the defaults, and note any choice the wizard offers (install location, all users or just you).
2. Start LM Studio once. If it offers to download a model, **skip** it. If it asks about a runtime or engine download, accept: that's its llama.cpp engine, and Step 3 measures it.
3. Tell me it's installed. I'll diff the snapshot, report every change, and set up the measurements.

**What the install changed** (`scripts/system_snapshot.ps1 -Compare`, saved in `results/raw/lmstudio-install/`):

| Where | Change |
|---|---|
| user `PATH` | + `C:\Users\bruno\.lmstudio\bin` (`lms.exe`) |
| `HKCU\…\CurrentVersion\Run` | + `electron.app.LM Studio = …\LM Studio.exe --run-as-service`: **starts at every login** (`settings.json`: `enableLocalService: true`) |
| Start menu, HKCU uninstall | + `LM Studio.lnk`; + `LM Studio 0.4.28+2` (per-user) |
| URL protocol handler | + `lmstudio://` |
| folders | + `%LOCALAPPDATA%\Programs\LM Studio` (the app, 771 MB), `%LOCALAPPDATA%\lm-studio-updater`, `%APPDATA%\LM Studio` (3 MB), `~\.lmstudio` (**2.5 GB**), `~\.lmstudio-home-pointer` |
| unchanged | machine `PATH`, HKLM Run, services, scheduled tasks, firewall rules, Program Files |

The listening-port diff was noise: Windows' listener tables are unreliable on this build (Lab 08). Its one new `0.0.0.0` listener belonged to `XrRuntime`, not LM Studio.

**What `~\.lmstudio` holds:**
* **Four llama.cpp engines**, each read from its `display-data.json`:
  * CPU, CUDA and Vulkan engines **2.67.0** = llama.cpp **b11999** (`0893ec98f`), installed with the app;
  * a CUDA 12 engine **2.70.0** = llama.cpp **b12116** (`5851a8e50`), **downloaded by itself at 12:18, 2 minutes after the install** (`autoUpdateExtensionPacks: true`).
* **The engine actually used for GGUF is 2.70.0** (`backend-preferences-v2.json`). It's a `llama-server.exe` (`0.9.0-dev (build 1, commit 5851a8e)`, Clang 19.1.1) with a CUDA 12.8 runtime (cudart 6.14.11.12080) from `vendor\win-llama-cuda12-vendor-v2`.
* An 84 MB embedding model (`nomic-embed-text-v1.5`), plus `node.exe`, `deno.exe` and `esbuild.exe` for plugins.
* `settings.json`: `jitModelTTL 3600 s`, `apiPredictionHistoryEviction.ttlDays: 30`, `appUpdateChannel: stable`.
* **The app's internal API is on `127.0.0.1:41343`** (`http-server.json`); it's what the `lms` CLI uses. Connecting from the LAN address `10.0.0.202` was **refused**.

**The model:** `lms import`'s default is to **move** the file. That would have taken the blob out of Ollama's store and broken Ollama's 4b. `-L` makes a hard link, but `lms` warns about the blob's missing `.gguf` extension. So I made the two hard links myself:
* `~\.lmstudio\models\ollama-lab\qwen3.5-4b\qwen3.5-4b-Q4_K_M.gguf` → Ollama's model blob;
* `mmproj-qwen3.5-4b.gguf` → Ollama's projector blob.

Each file now has 2 links and no extra disk. Deleting it in LM Studio removes only LM Studio's name. `lms ls` lists it as `qwen3.5-4b`, 4.2B, `qwen35`.

**Probe** (the same raw prompt as Steps 1–2):
* **Loading:** `lms load qwen3.5-4b --gpu max -c 4096 --parallel 1` took 5.96–5.98 s, +3,894 MiB.
* **The API server** answers on `127.0.0.1:1234`, and **is refused from `10.0.0.202`**.
* **The raw prompt went through unchanged** (23 prompt tokens).
* **LM Studio's API changed the sampling.** With llama-server's options, `/api/v0/completions` reported ≈125 tok/s, and the text differed from llama-server's ("…an automated health check…").
  * With Ollama's options it was ≈125 tok/s again, with yet another text.
* **LM Studio's own `llama-server.exe` run directly** with Step 1's flags gave 142.4 tok/s with llama-server's options and 126.9 with Ollama's. Its text was **byte-identical to Steps 1–2** for both. So b11963 and b12116, three different builds, wrote the same bytes.
* **Adding `repeat_penalty: 1, min_p: 0`** to the API request gave **byte-identical text to llama-server**, and the client-timed stream rose from 122 to 137 tok/s.
  * So the LM Studio layer adds its own sampler defaults, probably `repeat_penalty 1.1` and `min_p 0.05`. That isn't verified: the script tests each one separately.

**What runs** (`scripts/lmstudio_compare.py`, ≈6 min): LM Studio and its server must be running; the script loads and unloads the model with `lms`.
* **A.** Load time, VRAM, the engine in use, and TCP connects to 1234 and the internal port, on 127.0.0.1 and the LAN address.
* **B.** The raw prompt with llama-server's options, as sent, then with `repeat_penalty 1`, with `min_p 0`, and with both; then Ollama's options with and without both. For each: LM Studio's stats tok/s, client-timed stream rate, tokens, and byte-identity with Step 1's llama-server text.
* **C.** `/v1/chat/completions` with the GGUF's own template: prompt tokens (Ollama 23, llama-server chatml 19), and whether thinking comes back as a separate field.
* **D.** `--parallel 4`, 4 concurrent requests.
* **E.** 1-token requests, median of 20.
* **F.** A request with a marker in the prompt: new files in `api-prediction-history`, and any file under `~\.lmstudio` that holds the marker.

```powershell
ollama stop qwen3.5:4b
.\.venv\Scripts\python scripts\lmstudio_compare.py      # ≈6 min; LM Studio must be running
```

**Predictions** (written before the first full run):

| Measure | Prediction | Why |
|---|---|---|
| A | 5–7 s, +≈3,900 MiB; engine 2.70.0; 1234 and the internal port refused from the LAN | Probe |
| B: as sent | ≈125 (LM Studio's stats) / ≈122 (client); text ≠ llama-server | Probe |
| B: `repeat_penalty 1` alone | speed back to ≈137–142; text = llama-server (low confidence) | A repeat penalty touches every vocabulary entry, as the presence penalty did; `min_p` filters cheaply |
| B: `min_p 0` alone | speed unchanged (≈125); text ≠ llama-server | As above |
| B: both pinned | = llama-server, byte for byte; ≈137–142 | Probe |
| B: Ollama's options + both pinned | = Step 1's llama-server text with Ollama's options; ≈120–127 | Same engine family, same options |
| C | prompt tokens ≠ 19; thinking either in a separate field or absent (low confidence) | LM Studio uses the GGUF's Jinja template, which knows qwen3.5's think tags |
| D | aggregate ≈280–300 tok/s | Steps 1–2: 288–297 at 4 slots |
| E | 10–30 ms (an Electron/Node layer in front of llama-server) | Ollama's Go layer added ≈14 ms |
| F | no prompt stored (`api-prediction-history` stayed empty through ≈30 probe requests) | Probe; the 30-day TTL may apply to UI history only |

## 5. Hypotheses → Observed Metrics

### Step 1 — runs 1–3 (2026-11-05, `qwen3.5:4b` from the Ollama store, temperature 0, seed 42)

Runs: `engine-compare-20261105-111624`, `…-111808` and `…-111947`, all mine.
* **Runs 1–2 repeated the long prompt verbatim.** Ollama answered the repeats from its prompt cache, which showed as ≈39K prompt "tok/s" against 5.6–5.7K for llama-server, which ran with `cache_prompt: false`. That's a flaw in the script, not an engine difference.
* **Run 3 gives each long prompt a unique first line** and records `prompt_eval_cached_count`.
* Everything else agreed across the three runs within ≈2%. Values below are from run 3, with runs 1–2 in brackets where they add something.

**1. Same engine, same memory, similar cold start.**
* **Ollama:** the first request took 3.66 s (load 3.39 s), +3,902 MiB.
* **llama-server:** 4.82 s from process start to healthy, then 0.25 s for the first request, +3,899 MiB.
* Both read the 2.65 GB blob from Windows' page cache and hold the same buffers. Ollama's process was already running, so its time is load only.

**2. The sampler settings, not the runtime, decide the speed.** Medians of 5; the text was byte-identical between the engines in every row:

| sampler settings | llama-server | Ollama | tokens |
|---|---:|---:|---:|
| Ollama's defaults: `presence_penalty 1.5, top_k 20, top_p 0.95` | 119.3 | 120.7 | 57 |
| llama-server's defaults: `presence_penalty 0, top_k 40, top_p 0.95` | 142.8 | 144.8 | 61 |
| the penalty only (`top_k 40`) | 118.9 | 120.5 | 57 |
| `top_k 20` only (penalty 0) | 142.6 | 142.7 | 61 |

* **The presence penalty costs ≈17% (≈1.3 ms per token). `top_k 20` against 40 costs nothing measurable.** The penalty also changes the text: 57 tokens instead of 61, diverging at character 94.
* **With identical options Ollama was 0–1.7% faster** (runs 1–2: 0.3–3.5%). That's small but consistent; the cause isn't known.
* **Prompt processing, 1,640 tokens, nothing cached:** 5,553 vs 5,596 tok/s (+0.8%).
* So "Ollama is slower than llama.cpp" on this machine is entirely Ollama's default `presence_penalty: 1.5`, which comes from the model's params layer. It's not overhead.
* That also re-labels earlier labs: every Ollama tok/s from Lab 02 to Lab 08 includes the penalty, and every "temperature 0" answer had it applied.

**3. Chat templates: Ollama's renderer and llama-server's generic template differ.**
* With `think: false`, Ollama's `/api/chat` built **23 prompt tokens**. That's the same token count as B's raw prompt, which ends in an empty `<think>\n\n</think>\n\n`. Its answer matched B's text.
* llama-server's `/v1/chat/completions` with `--chat-template chatml` built **19 tokens**: no think block. The model then **wrote `<think>\n\n</think>\n\n` into the answer itself**, 61 completion tokens, and the text differed.
* So the same "chat" request isn't the same prompt across runtimes. Ollama ships a model-specific renderer, `renderer=qwen3.5` in its logs. With a generic template, the thinking control leaks into the content, and a client has to strip it.

**4. Concurrency: llama-server can batch qwen3.5; Ollama, as configured, can't.** 4 requests at once, up to 128 tokens each, llama-server's sampler defaults:

| engine | wall for all 4 | aggregate tok/s | per request tok/s | GPU MiB |
|---|---:|---:|---:|---:|
| llama-server `-np 1` | 1.88 s | 129.8 | 142 | 5,441 |
| llama-server `-np 2` | 1.18 s | **203.0** | 120–121 | 5,623 |
| llama-server `-np 4` | 0.83 s | **293.1** | 87 | 5,966 |
| Ollama (`-np 1`) | 1.90 s | 128.3 | 142–143 | 5,435 |

* **The hybrid model batches well.** Two slots gave 1.56× the aggregate (runs 1–2: 1.57×) and four gave 2.26× (2.25×, 2.21×). The text was identical in every slot.
* **Each extra slot cost ≈175–180 MiB** (+182 for 2, +525 for 4): a 4,096-token KV cache plus recurrent state per slot.
* **Per-request speed fell to 61% at 4 slots,** so batching trades single-user latency for total throughput.
* **Ollama serialized the 4,** exactly like `-np 1`. Lab 04 found `OLLAMA_NUM_PARALLEL=2` ignored for qwen3.5 (`-np 1`). Standalone llama-server shows the engine itself can do it.

**5. Ollama adds ≈14 ms per request.** For 1-token requests, median of 20, it was 8.4 ms on llama-server `/completion` against 22.8 ms on Ollama `/api/generate` (raw) (runs 1–2: 14.9, 14.3). That's Go HTTP, the scheduler lookup and the proxy hop to the runner. Next to a 0.5 s answer it's ≈3%; for an embeddings or classification service sending 1-token requests, it's most of the latency.

**Predictions vs observed:**

| Prediction | Observed |
|---|---|
| A: Ollama 5–8 s; llama-server healthy 5–6 s; +≈3,900 MiB both | ❌ Ollama 3.7 s (faster); ≈ llama-server 4.8 s; ✅ +3,902 / +3,899 |
| B: ≈120 / ≈144 on both; byte-identical | ✅ 119–121 / 143–145; identical in all 4 rows |
| B: the penalty carries the cost, `top_k 20` ≈ free | ✅ |
| B: long-prompt prompt eval within ±5% | ✅ +0.8% (run 3; runs 1–2 invalid: prompt cache) |
| C: prompt tokens differ, text not identical | ✅ 19 vs 23; the model wrote the think tags itself |
| D: `-np 1` and Ollama serialized ≈140 tok/s | ✅ serialized; ≈130 aggregate (gaps between requests) |
| D: `-np 2` / `-np 4` ≈1.6× / ≈2.5×; +100–300 MiB per slot | ✅ 1.56×; ❌ 2.26× (a little under); ✅ ≈175 MiB per slot |
| E: Ollama adds 5–20 ms | ✅ 14.3–14.9 ms |

**What Step 1 teaches:**
1. **Ollama is llama.cpp plus defaults, templates and a scheduler.** With identical options the two engines produced the same bytes at the same speed, within 2%. Benchmarks that compare their defaults compare settings, not engines.
2. **Read a model's params layer before any benchmark.** `presence_penalty 1.5` costs 17% of generation speed and changes the text, at temperature 0 too. Pin every sampler option explicitly in comparisons.
3. **Templates are part of the model.** Ollama's qwen3.5 renderer and llama-server's generic `chatml` build different prompts for the same chat. Where Ollama's renderer adds an empty think block, the generic template leaves the model to write its own into the answer.
4. **Batching is where llama-server pulls ahead.** 4 slots gave 2.26× the throughput for ≈525 MiB more, a configuration Ollama doesn't offer for this model. For one user that's irrelevant; for a shared service it's the main difference.
5. **Check your cache.** A repeated prompt measured Ollama's prompt cache, not its prompt processing: 39K against 5.6K tok/s.

### Step 2 — run 1 (2026-11-05, `qwen3.5:4b` GGUF, temperature 0, seed 42)

Run: `build-compare-20261105-114941`, mine. 4 server processes in turn, then `-np 4` once per build, then `llama-bench`.

**1. Same commit, same output, same memory.**

| build | round | process → healthy | VRAM Δ | gen tok/s, Ollama's options | gen tok/s, llama-server's | prompt tok/s, 1,640 tokens |
|---|---:|---:|---:|---:|---:|---:|
| bundled | 1 | 5.59 s | 3,897 MiB | 120.0 | 143.6 | 5,755 |
| official | 1 | 5.04 s | 3,897 MiB | **127.8** | 142.6 | 5,691 |
| bundled | 2 | 4.56 s | 3,899 MiB | 119.8 | 142.7 | 5,706 |
| official | 2 | 4.81 s | 3,899 MiB | **127.4** | 142.2 | 5,694 |

* **The text was byte-identical across all 4 processes,** for both sampler settings (57 and 61 tokens). Different compilers and CUDA 13.0 against 13.4 changed no output bytes.
* **Memory was identical** (3,897–3,899 MiB). Prompt processing was within 1.1%, and generation with llama-server's options within 1%.

**2. The difference is in the presence-penalty sampler, on the CPU.**
* With Ollama's options (`presence_penalty 1.5`), the official build was **6.5% faster** in both rounds: 127.4–127.8 against 119.8–120.0.
* Per token, the penalty costs ≈1.35 ms in the bundled build and ≈0.81 ms in the official one (1/120 − 1/143 against 1/127.6 − 1/142.4).
* The GPU work is the same in both builds: the same generation speed without the penalty, and the same bytes with it. So the difference is the CPU code that applies the penalty to all 248K token scores. The compiler (Clang 20.1.8 against 18.1.8) or its flags are plausible causes, but **neither is verified**.
* **Lesson:** "the same llama.cpp" can differ by 6% because of how it was built, and only in some configurations. Benchmark the binary you'll actually run, with the options you'll actually use.

**3. Batching and llama-bench.**
* **`-np 4` with 4 concurrent requests:**
  * bundled: 296.5 tok/s aggregate (88 per request);
  * official: 289.9 (86 per request), 2.2% lower;
  * both +4,441 MiB, and the same text in every slot.
* **`llama-bench`** (official only): **pp512 6,784 ± 89 tok/s, tg128 154.5 ± 0.7 tok/s.**
  * That's 8% above the server's generation speed (142–143), with no HTTP, no sampler and no detokenizing.
  * It's also above the server's 5.7K prompt rate, measured at 1,640 tokens and per request.
  * Figures from llama.cpp's own tooling are therefore an upper bound for what a served API delivers.

**Predictions vs observed:**

| Prediction | Observed |
|---|---|
| A: healthy 4.5–6.5 s; VRAM ±50 MiB | ✅ 4.56–5.59 s; identical (3,897–3,899) |
| B: generation equal ±3% | ✅ with llama-server's options (≤1%); ❌ with Ollama's: official +6.5% |
| B: prompt processing ±10% (low confidence) | ✅ within 1.1% |
| B: byte-identical across all 4 | ✅ |
| C: aggregate ±5% (≈290) | ✅ 296.5 / 289.9 |
| D: tg128 145–155; pp512 5–7K | ✅ 154.5; 6,784 |

**What Step 2 teaches:**
1. **A llama.cpp version doesn't determine its speed; the build does.** The same commit with a different compiler and CUDA toolkit gave the same bytes and the same GPU speed, but a 6.5% difference in the CPU-side sampler.
2. **The difference only shows in some configurations.** Without a presence penalty the builds were equal. A benchmark with one sampler setting would have called them identical, and one with Ollama's defaults would have called the official build faster.
3. **Ollama's 18% gap from Step 1 had two parts:** its default penalty (≈17%), and on top of that a build whose penalty path is slower than the official one.
4. **Synthetic benchmarks set the ceiling.** `llama-bench`'s tg128 was 8% above a served request. Quote them as the engine's limit, not as an API's throughput.

### Step 3 — run 1 (2026-11-05, LM Studio 0.4.28+2, engine 2.70.0 = llama.cpp b12116, `qwen3.5:4b` hard-linked)

Run: `lmstudio-compare-20261105-122711`, mine. The install diff and the probe are in §4 Step 3.

**1. Load and exposure.**
* `lms load`: 6.13 s, +3,906 MiB, runtime `llama.cpp-win-x86_64-nvidia-cuda12-avx2` 2.70.0.
* **TCP connects from `10.0.0.202` were refused** on both 1234 (the API) and 41343 (the internal API); both answer on 127.0.0.1.

**2. LM Studio's API adds a repeat penalty.** Raw prompt (23 tokens, passed unchanged), median of 5:

| options sent | LM Studio's stats | client-timed stream | tokens | = llama-server, byte for byte |
|---|---:|---:|---:|---|
| llama-server's | 125.1 | 125.5 | 58 | no |
| + `repeat_penalty 1` | **141.6** | 140.5 | 61 | **yes** |
| + `min_p 0` | 124.5 | 125.4 | 58 | no |
| + both | 140.7 | 141.1 | 61 | yes |
| Ollama's | 125.4 | – | 60 | no |
| Ollama's + both | 125.7 | – | 57 | yes |

* **The whole difference is `repeat_penalty`.** Pinned to 1, LM Studio wrote llama-server's exact bytes at its speed. `min_p 0` changed nothing.
* So LM Studio's default isn't 1, and it costs the same ≈1.2 ms per token as a presence penalty. Its exact value isn't in the server log at the default verbosity.
* LM Studio's `tokens_per_second` agreed with the client-timed stream within 1%.
* **With all options pinned, the three runtimes wrote the same bytes:** Ollama, llama-server (two builds of b11963) and LM Studio (b12116), with either sampler setting.

**3. Chat uses the GGUF's own template, with thinking on.**
* `/v1/chat/completions` built **21 prompt tokens** (Ollama 23, llama-server chatml 19).
* The model spent **all 256 tokens thinking** (`Thinking Process: 1. **Analyze the Request:** …`). That went into a separate `reasoning_content` field, so **`content` was empty**.
* Ollama with `think: false` answered directly, and llama-server's generic template leaked the think tags into the content.
* So three runtimes give three behaviours for one chat request. A client that reads only `content` with a 256-token budget would see an empty answer from LM Studio.

**4. Parallel 4 isn't the same configuration.**
* `--parallel 4` gave 4 slots sharing **one unified KV cache** (`kv_unified = 'true'`, 4,096 tokens), where llama-server's `-np 4` gave each slot its own.
* Aggregate throughput was **259.1 tok/s** (per request 75.6–80.5), against 288–297 for llama-server. VRAM was +4,061 MiB against +4,441.
* **The 4 texts weren't all identical,** where llama-server's 4 were. A unified cache changes how requests are batched together, which is a plausible reason, but it isn't verified.

**5. Overhead and retention.**
* **Per-request overhead: 27.4 ms** for a 1-token request. That compares with 22.8 for Ollama and 8.4 for llama-server, so LM Studio's layer adds ≈19 ms.
* **The prompts are kept in plain text** in LM Studio's server log, `~\.lmstudio\server-logs6-116-11-05.1.log`, at the default `runtimeLogVerbosityLevel: 3`:
  * each request body, with the marker prompt;
  * `Running completion on text: LAB09-LMSTUDIO-MARKER: Ex…`;
  * the chat messages, and the model's `reasoning_content`.
* The file is one per day, 283 KB after this session. I found no setting that limits how long it's kept.
* `api-prediction-history` stayed empty.

**Predictions vs observed:**

| Prediction | Observed |
|---|---|
| A: 5–7 s, +≈3,900; 2.70.0; refused from the LAN | ✅ 6.13 s, +3,906, 2.70.0, refused on both ports |
| B: as sent ≈125 / ≈122, text differs | ✅ 125.1 / 125.5 |
| B: `repeat_penalty 1` alone → speed and text restored (low confidence) | ✅ 141.6, identical |
| B: `min_p 0` alone → no change | ✅ |
| B: both pinned → identical, ≈137–142 | ✅ 140.7 |
| B: Ollama's options + both → identical, ≈120–127 | ✅ 125.7 |
| C: prompt tokens ≠ 19; thinking separate or absent | ✅ 21; separate, and it used the whole budget |
| D: ≈280–300 tok/s | ❌ 259.1, with a unified KV cache, so not the same configuration |
| E: 10–30 ms | ✅ 27.4 ms |
| F: no prompt stored | ❌ every prompt, response and reasoning in the daily server log |

**What Step 3 teaches:**
1. **A desktop runtime is a set of moving parts.** LM Studio:
   * changed the user `PATH`;
   * registered a login item and a URL handler;
   * installed four llama.cpp engines (2.5 GB);
   * **updated its engine by itself within 2 minutes;**
   * runs a second, internal API.

   Nothing was machine-wide and everything stayed on localhost, but pin the engine version (`autoUpdateExtensionPacks: false`) before any benchmark you want to repeat.
2. **Every runtime has its own sampler defaults:** Ollama a presence penalty from the model, LM Studio a repeat penalty from the app, llama-server none. Each default costs ≈17% of generation speed and changes the text. Pin `presence_penalty`, `repeat_penalty`, `top_k`, `top_p` and `min_p` explicitly in every comparison, and then the engines agree to the byte.
3. **Chat templates decide what "the same request" means.** The three runtimes gave three answers to one chat request: a direct answer, think tags leaking into the content, and an empty `content` with all the thinking in a separate field.
4. **"Parallel" can mean different memory layouts.** A unified KV cache uses less VRAM and gave less throughput here, and texts that weren't identical. Compare slot counts and KV layout, not just the flag.
5. **Know where each runtime keeps prompts:**
   * Ollama only with `OLLAMA_DEBUG_LOG_REQUESTS` (Lab 08);
   * llama-server not at all, in these runs;
   * LM Studio by default, in a daily log under your home folder.

## 6. Troubleshooting & Failure Modes

| Symptom | Cause | Check / fix |
|---|---|---|
| Ollama's bundled `llama-server.exe` runs, but on the CPU (VRAM unchanged, `n_threads = 8`, no CUDA lines) | ggml didn't find its CUDA backend; Ollama passes `GGML_BACKEND_PATH` | Set `GGML_BACKEND_PATH=<lib\ollama\cuda_v13\ggml-cuda.dll>` and put that folder on `PATH` |
| "llama.cpp is 18% faster than Ollama" on the same model | Different sampler defaults: Ollama applies the model's params layer (`presence_penalty 1.5`, `top_k 20`) | Compare with explicit, identical sampling options |
| The same model and prompt give different text on two engines at temperature 0 | Different penalties or templates, not randomness | Send identical raw prompts and options; compare byte by byte |
| llama-server: `no API key is set and CORS allows all origins` | llama-server's defaults | Keep `--host 127.0.0.1`; set `--api-key` if anything else can reach it |
| Ollama's prompt eval is 7× faster than llama-server's on the same prompt | The prompt was repeated: Ollama reused its prompt cache (`prompt_eval_cached_count`), and the other side ran with `cache_prompt: false` | Make every benchmark prompt unique; check `prompt_eval_cached_count` |
| Chat answers through llama-server start with `<think>\n\n</think>` | The generic `chatml` template has no thinking control, so the model writes the empty block itself | Use the model's own template (`--jinja` with the GGUF's template, or Ollama's renderer), or strip it in the client |
| Two builds of one llama.cpp commit differ in tok/s for some settings only | CPU-side code (e.g. the presence-penalty sampler) built by different compilers or flags; GPU kernels and output are the same | Benchmark the binary and options you'll run; compare with and without penalties |
| A PowerShell script loses its process handle (`Stop-Process: … 'Id' because it is null`) | PowerShell variable names aren't case-sensitive: `$P` and `$p` are the same variable | Use distinct names; stop a lab process by its full executable path |
| LM Studio ≈17% slower than llama-server with the same options; different text | LM Studio's API adds a repeat penalty (not 1) unless `repeat_penalty` is sent | Send `repeat_penalty: 1` (and the other sampler options) explicitly |
| LM Studio chat answer has empty `content` | The GGUF's template thinks by default; the reasoning went to `reasoning_content` and used the token budget | Read `reasoning_content`, raise `max_tokens`, or turn thinking off in the request or preset |
| `lms import` removed the source file | Its default is **move**; `-L` (hard link), `-l` (symlink) or `-c` (copy) keep it | Use `-L` for a file in another runtime's store, or make the hard link yourself (needs a `.gguf` name) |
| LM Studio's engine version changed between runs | `autoUpdateExtensionPacks: true` downloads new engines (2.70.0 arrived 2 minutes after the install) | Pin the engine; record `runtime.version` from `/api/v0` responses |
| Prompts found in `~\.lmstudio\server-logs\` | LM Studio logs request bodies, responses and reasoning at its default verbosity | Lower the log verbosity, delete the logs, or keep sensitive prompts out of it |
| LM Studio starts at login | `enableLocalService: true` adds `HKCU\…\Run: LM Studio.exe --run-as-service` | Turn the local service off in LM Studio's settings, or remove the Run value |
| `OLLAMA_NUM_PARALLEL` has no effect for qwen3.5 | Ollama starts its runner with `-np 1` for this model (Lab 04) | Run llama-server with `-np N` directly if you need batching |

## 7. Cleanup

```powershell
ollama stop qwen3.5:4b
Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object Id, Path   # check before stopping: Ollama's own runner has the bundled path
Remove-Item -Recurse tools\downloads, tools\llama.cpp-b11963   # optional: the official build (589 MB of zips + 731 MB) and the LM Studio installer (626 MB)
& "$HOME\.lmstudio\bin\lms.exe" unload --all                  # Step 3: the scripts already do this
Remove-Item "$HOME\.lmstudio\models\ollama-lab" -Recurse      # Step 3: removes only LM Studio's hard-link names; Ollama's blobs stay
# Uninstalling LM Studio (optional): Settings → Apps → LM Studio. Then delete ~\.lmstudio (2.5 GB), ~\.lmstudio-home-pointer,
# %APPDATA%\LM Studio and %LOCALAPPDATA%\lm-studio-updater, and remove ~\.lmstudio\bin from the user PATH if it remains.
```

The scripts stop the llama-server they started. Ollama's own runner uses the bundled executable, so check before stopping one by hand. Nothing in Steps 1–2 is installed: the official build is a folder of files under `tools\`, with no installer, registry entries or PATH change. Step 3 installed LM Studio; §4 Step 3 lists every change it made.

## 8. Check Your Understanding

1. Ollama runs llama.cpp's `llama-server.exe`. What, then, is Ollama, in the container-runtime analogy, and what does it add that you'd otherwise build yourself?
2. Same binary, same GGUF, same prompt, temperature 0: one engine was 18% faster and wrote different text. What was different, and how did the probe prove it?
3. Why did the bundled binary silently run on the CPU, and which environment variable fixed it?
4. Which single sampler option cost 17% of the generation speed, where did Ollama get it from, and why does it change the text even at temperature 0?
5. The same chat request was 23 prompt tokens in Ollama and 19 in llama-server. What were the 4 tokens, and what did the model do without them?
6. llama-server with `-np 4` gave 2.26× the throughput but 61% of the per-request speed. When would you choose it over Ollama's single slot, and what does each extra slot cost?
7. Runs 1–2 showed Ollama processing prompts at 39K tok/s. What was really measured?
8. The official and the bundled build produced the same bytes, yet one was 6.5% faster with Ollama's options and equal without them. Where in the pipeline can that difference sit, and why not on the GPU?
9. `llama-bench` said 154.5 tok/s and the server 142–143 tok/s with the same binary. What does the benchmark leave out?
10. Ollama, llama-server and LM Studio wrote three different texts for one raw prompt at temperature 0, then the same bytes. Which options did you have to pin, and where did each runtime's default come from?
11. LM Studio's chat answer came back with empty `content`. Where did the tokens go, and what would a client have to do differently from Ollama's API?
12. What did installing LM Studio change outside its own folders, and which of those changes would you undo on a shared machine?
13. Where does each of the three runtimes store prompts by default, and how would you check a fourth?
