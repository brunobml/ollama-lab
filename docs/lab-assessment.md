# Ollama Learning Lab — Technical Assessment & Recommendations

> **Assessment Date:** 2026-10-27  
> **Scope:** Phases 0–3 (Complete), Phase 4 (In Progress, Steps 1–3 complete, Step 4 staged), Repository Docs, Scripts, and Phases 5–9 Roadmap  
> **Hardware / Runtime Context:** RTX 3080 (10 GB VRAM) | i7-10700K | 64 GB DDR4-2400 | Windows 11 + PowerShell 7 | Ollama 0.40.2 → 0.40.10 | NVIDIA Driver 610.88 → 617.42 → 591.86

---

## 0. Status Update (2026-10-31, after Labs 05–06)

This assessment was written on 2026-10-27, during Lab 04. The text below is kept as written; this table records what happened to each recommendation.

| Recommendation (section) | Status |
| :--- | :--- |
| Create `docs/glossary.md` (§4, Rec. 1) | **Done**, extended with Labs 05–06 |
| Update the root `README.md` (§4, Rec. 3) | **Done** (links all docs and Labs 00–06) |
| Clarify the `results/` policy (§4, Rec. 3) | **Done**: `RoadMap.md` §4 now says curated results live in each lab's section 5; `results/raw/` is gitignored |
| Failure-mode cheat sheet (§4, Rec. 2) | **Open.** Labs 05–06 added more "HTTP 200 but wrong" cases (silent embedding truncation, the 16K → 2,050 agent prompt cut, invented verification). Each lab's section 6 table is the raw material |
| Move shared question sets to JSON (§4, Rec. 4) | Open (minor) |
| Version headers in `architecture.md` / `vram-calculator.md` (§3) | **Done**: kept "measured on 0.40.2" and added the current version (0.40.13, not re-measured) |
| Phase 4: concurrency, `n_slots = 1` (§5.1) | Done in Lab 04 Step 6 |
| Phase 4: `think=True` combined with `format` (§5.1) | Not checked |
| Phase 5: stay on Windows (§5.2) | **Done** (decision D-08) |
| Phase 5: embedding model + LLM coexisting in VRAM (§5.2) | Not measured as a separate experiment |
| Phase 6: a large-context Modelfile before `ollama launch` (§5.3) | **Done** (`qwen3.5:4b-ctx64k`). The prediction that tool definitions would exceed 2,050 tokens and be cut silently was **confirmed**: Claude Code sent 16,137 tokens, Ollama kept 2,050 (Lab 06 Step 5) |
| Phase 6: injection through tool output vs the approval gate (§5.3) | **Done** (Lab 06 Step 2): followed 1/24, blocked by the gate every time |
| Phases 7–8 (§5.4–5.5) | Still ahead |

---

## 1. Executive Summary

The `ollama-lab` repository is a well-structured engineering lab. Rather than treating local LLMs as black-box chat toys, the completed work treats Ollama and `llama-server` as stateful infrastructure components with measurable compute, memory-bandwidth, and I/O constraints.

Across Phases 0–3 and the first half of Phase 4, the lab has adhered strictly to the principles in [`ollama-roadmap-claude.md`](../ollama-roadmap-claude.md) and [`RoadMap.md`](../RoadMap.md):
* **Hypothesis-driven experimentation:** Every experiment states explicit predictions before running and records actual metrics afterward.
* **No fabricated or unexamined numbers:** Automated regex scoring bugs were caught by reading raw outputs (`3500\text{m}` LaTeX formatting, inverted Argo CD `selfHeal` direction, inherited Modelfile `presence_penalty`), and unexpected hardware/runtime behaviors were traced to root causes in logs and OS telemetry.
* **Strong DevOps mental models:** Concepts are consistently mapped to Kubernetes, OCI registries, admission control, and microservice reliability patterns.

The recommendations below focus on **(1) resolving minor documentation and repository drift**, **(2) tightening script maintainability without over-engineering**, and **(3) applying empirical findings from Labs 00–04 to de-risk Phases 4–9**.

---

## 2. What Is Working Well

### 2.1 High-Value Empirical Discoveries
Several findings from Labs 00–04 are non-obvious operational traps that directly impact real-world local AI engineering:

1. **The `localhost` IPv6 Penalty (Lab 00):** Measuring `http://localhost:11434` (2,103 ms) vs `http://127.0.0.1:11434` (16 ms) exposed Windows IPv6 (`::1`) fallback against Ollama's IPv4-only listener—a 2-second tax per API call that is easy to misdiagnose as slow inference.
2. **Cold-Start Root Cause Analysis (Labs 01–02):** Instead of accepting the 40.7 s first-ever load as "disk slowness", Lab 02 tested the Windows Defender hypothesis with `Get-Counter`, refuted it, and traced the delay to one-time CUDA JIT kernel compilation in `%APPDATA%\NVIDIA\ComputeCache` (re-confirmed after the driver update to `617.42`).
3. **Accurate VRAM & Bandwidth Models ([`docs/vram-calculator.md`](vram-calculator.md)):**
   * Proved `ollama ps` `SIZE` underestimates actual VRAM delta (`3.1 GB` reported vs `3,860 MiB` measured for `qwen3.5:4b` at 4K context).
   * Confirmed `qwen3.5`'s hybrid attention architecture costs only **32 KiB/token** of KV cache (+50 MiB fixed recurrent state), 4× cheaper than a standard full-attention 8B model (128 KiB/token), while also discovering why `llama-server` disables `--context-shift` for hybrid models.
   * Derived a predictive generation latency model ($\approx 4.7\text{ ms} + \text{weights}_{\text{GPU}} \div 770\text{ GB/s}$) and proved why partial CPU offload is a cliff: 27% of layers on DDR4-2400 CPU RAM ($\approx 21\text{ GB/s}$ effective) costs **81% of generation speed**.
4. **Silent Context Truncation Mechanics (Lab 03 Step 5):**
   * Proved `/api/generate` silently truncates oversized prompts from the front using $\text{kept} = 4 + (\text{num\_ctx} - 4) / 2$ (logged only as a `WARN` in `server.log`), whereas `/api/chat` silently drops whole oldest messages while preserving the `system` message.
   * Verified the undocumented `"truncate": false` request field on both `/api/generate` and `/api/chat` to force an HTTP 400 error instead of hallucinating from partial inputs.
5. **Client Failure Modes & Load Cancellation (Lab 04 Step 3):**
   * Verified from `.venv\Lib\site-packages\ollama\_client.py` that `ollama-python` defaults to `timeout=None`, omits the `truncate` parameter, raises lazy errors on `stream=True`, and leaves `httpx` timeouts unwrapped.
   * Proved that when a client times out during a cold load, Ollama aborts the `llama-server` load (`client connection closed before llama-server finished loading, aborting load`)—preventing GPU retry storms, but guaranteeing that a client with `timeout < load_time` will fail infinitely on retries.

### 2.2 Pedagogical Structure
The "Check Your Understanding" sections in [`labs/00-environment-baseline/README.md`](../labs/00-environment-baseline/README.md) through [`labs/04-apis-structured-output/README.md`](../labs/04-apis-structured-output/README.md) work well. In Lab 03 and Lab 04, framing each question around a concrete experimental anomaly (e.g., *"Why did the model say '30 and 5' when it knew '10 and 3'?"* or *"Which of these errors would you retry?"*) reinforces production engineering judgment rather than rote memorization.

---

## 3. Identified Gaps & Repository Drift

A review of the current repository state against [`RoadMap.md`](../RoadMap.md) and [`ollama-roadmap-claude.md`](../ollama-roadmap-claude.md) highlights a few areas of drift:

| Area | Current State | Target / Expectation | Impact |
| :--- | :--- | :--- | :--- |
| **Root [`README.md`](../README.md)** | Line 10 still reads: `Current lab: Lab 00 — Environment Baseline` and only links `docs/environment.md`. | Phases 0–3 are complete; Phase 4 is in progress. `docs/architecture.md` and `docs/vram-calculator.md` exist. | Entry-point navigation is out of date. |
| **Missing `docs/glossary.md`** | Listed in [`RoadMap.md`](../RoadMap.md) §4 (`(Phase 1+) LLM terms mapped to DevOps concepts`), but does not exist in `docs/`. | A single reference mapping LLM terms (GGUF, KV cache, hybrid recurrent state, prefill, sampler chain, NDJSON, grammar constraints) to DevOps equivalents. | High-value analogies are currently scattered across five lab READMEs. |
| **Committed summaries in `results/`** | [`RoadMap.md`](../RoadMap.md) §4 and `.gitignore` state: `Summaries committed; results/raw/ gitignored`. Currently `results/` contains *only* `results/raw/` (0 tracked files in `results/`). | Either commit concise benchmark summary artifacts in `results/` or update `RoadMap.md` §4 to reflect that summaries live inline in `labs/*/README.md`. | Minor mismatch between documented repo structure and actual Git state. |
| **Version & Driver Drift** | Ollama upgraded `0.40.2` → `0.40.5` → `0.40.8` → `0.40.10`; NVIDIA driver changed `610.88` → `617.42` → `591.86`. | Tracked well in `docs/environment.md` and `RoadMap.md`, but `docs/vram-calculator.md` and `docs/architecture.md` headers still say `0.40.2`. Also, Lab 03 Step 6 noted a new `llama-server GPU discovery watchdog timed out` warning on driver `591.86`. | Watchdog timeout warning on `591.86` could cause silent fallback to CPU if not monitored. |
| **Cross-script AST parsing in `scripts/`** | [`scripts/grounding.ps1`](../scripts/grounding.ps1) (lines 19–21) parses [`scripts/quality.ps1`](../scripts/quality.ps1)'s PowerShell AST + `Invoke-Expression` to reuse `$questions`. | Shared test data stored in a plain `.json` file (like [`labs/04-apis-structured-output/incidents.json`](../labs/04-apis-structured-output/incidents.json)). | AST + `Invoke-Expression` works today, but a JSON file is simpler, safer, and usable from both PowerShell and Python. |

---

## 4. Recommendations Under `docs/` and Repository Organization

### Recommendation 1: Create `docs/glossary.md` (Planned in RoadMap §4)
Consolidate the terminology and DevOps analogies established in Phases 0–4 into `docs/glossary.md`, grouped by layer:
* **Storage & Packaging:** GGUF, `mmproj` (vision projector), OCI-style manifests (`manifests` vs `manifests-v2`), content-addressed blobs, `Modelfile` parameter inheritance.
* **Memory & Compute:** Weights buffer, KV cache, GQA (Grouped-Query Attention), hybrid recurrent state (`qwen3.5`), compute buffers, prefill (compute-bound) vs autoregressive decode (memory-bandwidth-bound), partial CPU offload (`num_gpu` / `-ngl`).
* **Prompting & Context:** ChatML (`<|im_start|>`), Go `RENDERER` / `PARSER`, thinking prefill (`<think>\n\n</think>`), prefix cache (`prompt_eval_cached_count`), front truncation (`4 + (num_ctx - 4) / 2`) vs chat message trimming, context shift.
* **Sampling & Constraints:** Logits, penalties (`presence`, `frequency`, `repeat`), filters (`top_k`, `top_p`, `min_p`), `temperature`, `seed`, grammar/schema-constrained decoding (`format`).

### Recommendation 2: Add an Operational Runbook / Failure-Mode Cheat Sheet (`docs/failure-modes.md` or in `docs/architecture.md`)
Across Labs 00–04, you have uncovered a rich set of failure modes where **HTTP 200 is returned with a broken or degraded result**. Centralizing a diagnostic checklist in `docs/` will serve as a quick reference for Phases 5–9:
* **Symptom: Slow every call (+2s)** → Check `127.0.0.1` vs `localhost` IPv6 fallback.
* **Symptom: Slow generation (<25 tok/s)** → Check `ollama ps` for CPU/GPU split or `server.log` for `llama-server GPU discovery watchdog timed out`.
* **Symptom: 19s per request in multi-model workflows** → Check `server.log` for `predicted to exceed available memory, evicting` (VRAM thrashing).
* **Symptom: Confident hallucination on long inputs** → Check `prompt_eval_count == 4 + (num_ctx - 4) / 2` (silent `/api/generate` truncation) or silent message drops in `/api/chat`; enforce `truncate: false`.
* **Symptom: Empty response after 30–45s** → Check `done_reason == "length"` with empty `content` and full `thinking` (thinking loop exhausted `num_predict`).
* **Symptom: Sudden truncation before `num_predict`** → Check `prompt_eval_count + eval_count == num_ctx` (`done_reason == "length"` due to full context window with context-shift disabled).

### Recommendation 3: Update Root `README.md` and Clarify `results/` Policy
* Update [`README.md`](../README.md) to link [`docs/architecture.md`](architecture.md), [`docs/vram-calculator.md`](vram-calculator.md), and all active labs (`00` through `04`).
* Clarify in [`RoadMap.md`](../RoadMap.md) §4 that curated result tables are committed directly inside each `labs/XX-*/README.md` under `## 5. Hypotheses → Observed Metrics`, while `results/raw/` holds gitignored run artifacts.

### Recommendation 4: Move Shared Question Sets to JSON
[`labs/04-apis-structured-output/incidents.json`](../labs/04-apis-structured-output/incidents.json) is a great pattern: clean separation of test cases, ground truth, and scoring logic. Consider extracting the 6-question Kubernetes benchmark bank from [`scripts/quality.ps1`](../scripts/quality.ps1) into a JSON file so [`scripts/quality.ps1`](../scripts/quality.ps1), [`scripts/rescore.ps1`](../scripts/rescore.ps1), and [`scripts/grounding.ps1`](../scripts/grounding.ps1) (and future Python evaluation scripts) can read it without PowerShell AST parsing.

---

## 5. Technical Recommendations for Upcoming Phases (Phases 4–9)

### 5.1 Phase 4 (Remaining Steps 4–6): Structured Output, OpenAI API, and Concurrency
1. **Step 4 (`scripts/structured.py` & `incidents.json`):**
   * The untracked `scripts/structured.py` and `labs/04-apis-structured-output/incidents.json` are well-designed—especially testing `IncidentReportAnalysisFirst` (putting `analysis: str` first in the Pydantic model to give the model a "scratchpad" inside the JSON when `think=False`) and including `injection` and `no-incident` cases.
   * **Recommendation:** Also observe what happens when `think=True` is combined with `format=SCHEMA` on `qwen3.5:4b` in Ollama `0.40.10`. In some Ollama/llama.cpp versions, grammar constraints apply to the whole output stream unless the parser separates `<think>` before grammar enforcement—verifying whether `think=True` + `format` works or breaks on `qwen3.5` is a critical finding for agent workflows.
2. **Step 5 (OpenAI-Compatible `/v1/chat/completions`):**
   * Test how Ollama handles `num_ctx` and `think` when called via the `/v1` endpoint. Standard OpenAI parameters don't include `num_ctx`, which means tools using `/v1` silently fall back to the 4,096 default unless overridden via a `Modelfile` or `extra_body={"options": {"num_ctx": ...}}` (verify whether Ollama's `/v1` accepts `options` in the request body).
3. **Step 6 (Concurrency & `n_slots = 1`):**
   * Before changing `OLLAMA_NUM_PARALLEL`, measure 4 concurrent requests against the default single-slot runner (`n_slots = 1`). Verify two effects:
     1. Head-of-line blocking (request 4 waits for requests 1–3).
     2. **Prefix cache thrashing:** If two concurrent clients send different system prompts to a single slot, they evict each other's KV prefix cache on every turn.

### 5.2 Phase 5: Embeddings, Vector Search & Local RAG
1. **Keep Python on Windows for Phase 5 (Revise Decision D-03):**
   * Decision D-03 originally planned to move to WSL2 before Phase 5, while D-07 established a clean Windows Python 3.12 `.venv`.
   * **Recommendation:** Run Phase 5 in the Windows `.venv` first, and defer the WSL2 mirrored-networking change (`.wslconfig`) to Phase 8 (Operations & Containers). That keeps Phase 5 focused purely on embeddings, chunking, and retrieval quality without mixing in WSL2 networking changes.
2. **Test Dual-Model Coexistence in VRAM (Embedding Model + LLM):**
   * A real RAG pipeline calls `/api/embed` (with e.g. `nomic-embed-text`) and then `/api/chat` (with `qwen3.5:4b`) on every query.
   * **Crucial experiment:** Check `OLLAMA_MAX_LOADED_MODELS` behavior and VRAM usage when alternating `/api/embed` and `/api/chat`. Because `nomic-embed-text` is tiny (~275 MB), both models easily fit within the 8,500 MiB ceiling—**verify in `ollama ps` that both stay loaded simultaneously** rather than evicting each other.
3. **Carry Forward Lab 03 Step 6 Grounding Lessons:**
   * Lab 03 Step 6 proved that an overly literal system rule (`"Answer only from the reference..."`) caused a 50% false-abstention rate when terminology differed slightly (`Probe` vs `readinessProbe`). Use Phase 5's labeled Q&A set to tune the RAG prompt for both **answerable** queries (recall) and **unanswerable/outdated** queries (precision + abstention).

### 5.3 Phase 6: Tool Calling, MCP & Coding Agents
1. **Measure Tool-Schema Token Overhead & Truncation Risk:**
   * Coding agents (Claude Code, Codex) inject large system prompts and dozens of JSON tool schemas. On a 4,096 default context, the tool definitions alone can exceed `4 + (4096 - 4)/2 = 2050` tokens and trigger silent front-truncation of the system prompt!
   * **Recommendation:** In Phase 6, create a dedicated `Modelfile` variant (e.g. `qwen3.5-agent:4b` with `PARAMETER num_ctx 32768` or `65536`) before testing `ollama launch` or Codex, since external agents often use `/v1` endpoints that cannot set `num_ctx` per request.
2. **Test Indirect Prompt Injection Through Tool Output:**
   * Build directly on Lab 03 Step 3b and Lab 04's `injection` incident: have `get_pod_logs()` return a log line containing an instruction to call a mutating tool, and verify that your human-in-the-loop approval gate and argument allow-list block it regardless of model compliance.

### 5.4 Phase 7: Multimodal Models
1. **Audit the "Already Paid For" Vision Projector:**
   * Lab 01 discovered that `qwen3.5:4b` loads its 676 MB `mmproj.gguf` CLIP projector (~640 MiB VRAM + 223 MiB vision compute buffer) even for text-only requests.
   * **Recommendation:** In Phase 7, measure how image resolution scales `prompt_eval_count` (image tokens) and `prompt_eval_duration`, and test whether dense text in Grafana/Kubernetes dashboard screenshots requires cropping vs full-screen capture for a 4B/9B model.

### 5.5 Phase 8: Operations, WSL2, Docker & Kubernetes
1. **GPU Contention Guardrail:**
   * Running Ollama in Docker/Kubernetes while Windows tray Ollama is loaded will split the 10 GB RTX 3080 across two independent schedulers that do not coordinate VRAM budgets (each sees `nvidia-smi` free memory only at startup or allocation time). Make stopping the Windows Ollama tray process an explicit prerequisite step in Lab 08's runbook.
2. **Persist the CUDA Compute Cache in Containers:**
   * Apply Lab 02's discovery directly to Lab 08: a stateless container without a persistent volume for `~/.nv/ComputeCache` will pay the ~40 s CUDA kernel compilation penalty on every pod restart.

---

## 6. Suggested Priority Order for Immediate Next Actions

1. **Finish Lab 04 Step 4 (`scripts/structured.py`):** Run the staged structured-output experiment and record the shape vs content accuracy in [`labs/04-apis-structured-output/README.md`](../labs/04-apis-structured-output/README.md).
2. **Update [`README.md`](../README.md):** Point to the current lab and link all documents in `docs/`.
3. **Create `docs/glossary.md`:** Capture the DevOps ↔ LLM mental models and empirical rules of thumb from Phases 0–4 before starting Phase 5.
