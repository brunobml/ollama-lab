# How to Use Google NotebookLM with `ollama-lab`

Instead of maintaining duplicate summary files for every lab, feed the **actual source-of-truth files** (`labs/*/README.md` and `docs/*.md`) directly into NotebookLM and use the steering prompts below to control how it teaches and quizzes you.

---

## 1. Recommended Setup: One Notebook + Source Checkboxes

Create **one** notebook in Google NotebookLM (e.g., `Ollama Learning Lab`) and upload:

1. **Context & Role (Keep Checked Always):**
   * `ollama-roadmap-claude.md` — tells NotebookLM your background (Senior DevOps / Kubernetes Engineer), your hardware (RTX 3080 10 GB VRAM, i7-10700K, 64 GB RAM), and the teaching rules.
2. **Reference Docs (Check as Needed):**
   * `docs/environment.md`
   * `docs/architecture.md`
   * `docs/vram-calculator.md`
   * `docs/glossary.md`
3. **Lab Sources (Check Only the Lab You Are Studying):**
   * `labs/00-environment-baseline/README.md`
   * `labs/01-foundations/README.md`
   * `labs/02-performance-hardware/README.md`
   * `labs/03-prompting-behavior/README.md`
   * `labs/04-apis-structured-output/README.md`
   * `labs/05-embeddings-rag/README.md`
   * `labs/06-tools-mcp-agents/README.md` (long: six steps; ask about one step at a time)

> **Privacy note:** NotebookLM is a Google cloud service, so uploading sends these files to Google. That's a deliberate exception to this lab's "local by default" rule: it's for studying, not inference. The lab READMEs contain local paths, the Windows user name and tool configuration details; don't upload `results/raw/` (full agent prompts and transcripts) or anything from your real infrastructure.

> **Tip — Rename `README.md` inside NotebookLM:** Because every lab file is named `README.md`, click the three dots (`⋮`) next to each uploaded source in NotebookLM's left sidebar and rename it (`Lab 00 - Baseline`, `Lab 01 - Foundations`, `Lab 02 - Performance`, etc.). Then use the **checkboxes** on the left to isolate a single lab or combine multiple labs.

---

## 2. Three High-Leverage Study Modes in NotebookLM

### Mode 1: Customized Audio Overview ("Deep Dive")
Never click **Generate** with default settings—default Audio Overviews stay too high-level and skip the math and failure modes. Always click **Customize** and paste the lab-specific prompt from Section 3 below.

### Mode 2: Blind Oral Exam in Chat (Active Recall)
Each lab `README.md` already has `<details><summary>Answer</summary>` blocks in Section 8. Paste this prompt into Chat to turn NotebookLM into an interactive interviewer:

> **Copy-Paste Prompt (Works for Any Checked Lab):**  
> *"Act as a Staff Infrastructure Engineer interviewing me on the checked Lab README. Ask me one question at a time from Section 8 ('Check Your Understanding')—or invent a realistic production incident based on Section 5 ('Hypotheses → Observed Metrics'). Do NOT reveal the answer yet. Wait for my reply, then grade my answer strictly against the empirical findings in the source and point out any detail or trade-off I missed."*

### Mode 3: Cross-Lab Synthesis ("Connect the Dots")
Check **all** completed labs (`Lab 00` through `Lab 06` + `docs/architecture.md`) and ask cross-cutting architectural questions:
* *"Trace every place across Labs 01–04 where `num_ctx` caused a surprise, a reload, or a silent failure mode."*
* *"Walk through every HTTP 200 response across Labs 02–04 that actually contained a broken, truncated, or hallucinated result, and how client code should detect each one."*
* *"Explain how content-addressed weights blobs in Lab 01 led to runner sharing in Lab 01 Step 5, eviction thrashing in Lab 02 Step 3, and `num_ctx` reload thrashing in Lab 04 Step 6."*
* *"Trace silent truncation across the labs: `/api/generate` and `/api/chat` in Lab 03, `/api/embed` in Lab 05 (which end is dropped), and Claude Code's 16,137-token prompt cut to 2,050 in Lab 06 Step 5. For each, how would a client detect it?"*
* *"Find every place in Lab 06 where the safety gate turned out to be the model itself (Step 2's denial reports, Step 5's auto-mode classifier, Step 6's stored trust), and what a deterministic control would look like for each."*
* *"Lab 05 measured hybrid search as the best retriever, and Lab 06 Step 4 found it ranked the right runbook first in only 6 of 14 agent queries. Explain why both results are true."*

---

## 3. Lab-by-Lab Prompts (Audio Overview & Chat)

### Lab 00 — Environment Baseline
**Sources to check:** `ollama-roadmap-claude.md`, `docs/environment.md`, `labs/00-environment-baseline/README.md`

* **Customize Audio Overview Prompt:**
  > *"The listener is a Senior Kubernetes and DevOps Engineer learning local LLM infrastructure on an RTX 3080 (10 GB VRAM). Frame the discussion around treating Ollama as an unauthenticated local HTTP control plane (`127.0.0.1:11434`) rather than a desktop app. Focus on the three empirical findings in Lab 00: (1) why calling `localhost` costs 2.1 seconds on Windows due to IPv6 `::1` fallback while `127.0.0.1` takes 16 ms, (2) why Ollama's `server.log` available VRAM figure (`8.9 GiB`) is a stale startup snapshot compared to live `nvidia-smi` desktop drift (`1.6 GB` to `2.3 GB`), and (3) why WSL2 can see the GPU via `nvidia-smi` yet cannot reach the Windows Ollama server over NAT loopback."*
* **Chat Drill Prompt:**
  > *"Give me 3 troubleshooting scenarios from Lab 00 (slow 2s API calls, unexpected VRAM shortage vs `server.log`, and WSL2 connection failure) one at a time, and ask me which PowerShell or CLI commands I would run to prove the root cause."*

---

### Lab 01 — Foundations: Models on Disk and in Memory
**Sources to check:** `ollama-roadmap-claude.md`, `docs/architecture.md`, `labs/01-foundations/README.md`

* **Customize Audio Overview Prompt:**
  > *"The listener is a Senior DevOps Engineer learning how Ollama manages models on disk and in GPU memory on an RTX 3080. Focus on four concrete findings from Lab 01: (1) how `~/.ollama/models` works like an OCI image cache with SHA-256 content-addressed blobs, and why `qwen3.5:4b` includes a 676 MB `mmproj` vision projector blob that loads into VRAM even on text prompts; (2) why a 3.3 GB model file actually consumes 3,863 MiB of VRAM while keeping 497 MiB in host RAM, and why `qwen3.5` uses a hybrid KV cache (128 MiB) + 50 MiB recurrent state; (3) why turning thinking off made a prompt 15x faster (0.9s vs 13.2s) even though generation speed stayed at ~117 tok/s; and (4) why a custom `Modelfile` variant (`devops-assistant`) cost only 1.5 KB on disk and shared the exact same `llama-server.exe` runner in VRAM—plus why its system prompt caused it to misattribute EndpointSlice updates to the kubelet."*
* **Chat Drill Prompt:**
  > *"Quiz me on the memory and disk layout of `qwen3.5:4b` in Lab 01: ask me to account for the difference between `ollama list` (3.3 GB), `ollama ps` (3.1 GB), and `nvidia-smi` (+3,863 MiB), and ask me when two model names can share a single `llama-server` process."*

---

### Lab 02 — Hardware Sizing, Quantization & Performance
**Sources to check:** `ollama-roadmap-claude.md`, `docs/vram-calculator.md`, `labs/02-performance-hardware/README.md`

* **Customize Audio Overview Prompt:**
  > *"The listener is a Senior Platform Engineer studying GPU capacity planning and LLM performance modeling on an RTX 3080 (10 GB VRAM). Walk through the math and empirical discoveries in Lab 02 and `vram-calculator.md`: (1) how we solved the 40-second first-load mystery by ruling out Windows Defender and catching 70 MB of CUDA JIT kernel compilation in `NVIDIA\ComputeCache`; (2) the linear memory-bandwidth model (`4.7 ms + weights_gpu / 770 GB/s`) and why offloading just 27% of layers to DDR4-2400 system RAM (~21 GB/s) destroys 81% of generation speed; (3) why `qwen3.5`'s hybrid architecture costs only 32 KiB/token of KV cache (1/4th of Llama 3 8B) and why `4b` and `9b` have identical KV cache sizes; (4) how an impossible 47,626 tok/s prompt rate exposed silent front-truncation at 4,096 context; and (5) why spending VRAM on more parameters (`9b Q4`) beat spending it on more precision (`4b Q8`)."*
* **Chat Drill Prompt:**
  > *"Using the formulas in `vram-calculator.md` and Lab 02, give me 3 capacity-planning problems (predicting VRAM, tok/s, and CPU offload penalties for different context sizes and models on this RTX 3080) one at a time and check my calculations."*

---

### Lab 03 — Prompting, Sampling & Reproducibility
**Sources to check:** `ollama-roadmap-claude.md`, `docs/architecture.md`, `labs/03-prompting-behavior/README.md`

* **Customize Audio Overview Prompt:**
  > *"The listener is a Senior Platform Engineer studying Lab 03 (Prompting, Sampling, Context Overflow, and Grounding). Connect all 6 experiments to autoregressive token prediction: (1) why `qwen3.5:4b` contradicted itself ('Yes... cannot') with thinking off and how Ollama implements `think: false` by prefilling an empty `<think></think>` block; (2) why thinking and sampling penalties broke word-for-word token repetition (`MaxRepeat` 172 down to 1) on the Helm 3 question but still burned 4,096 tokens looping on ideas because a sampler cannot invent a missing fact; (3) why `temperature: 0` gives reproducibility, not truth (answering port 443 10/10 times when the model is split 60/40 against 6443); (4) how conflicting user text injected fake flags into `kubectl` commands even when the system prompt's 1-line format held; (5) the exact difference between silent truncation on `/api/generate` (`4 + (num_ctx - 4)/2 = 2050` tokens) vs `/api/chat` (dropping oldest messages while keeping `system`), why context shift is disabled for `qwen3.5`, and how undocumented `truncate: false` forces an HTTP 400; and (6) why grounding turned 1/4 into 4/4 but made the model 100% vulnerable to poisoned references."*
* **Chat Drill Prompt:**
  > *"Give me 5 failure modes observed in Lab 03 (empty answer with `done_reason: length`, generation stopping at 452 tokens when `num_predict` was 3000, invented cluster `prod`, invented ticket `INC-2026-10-27-13`, and `NOT IN REFERENCE` on `readinessProbe`) one by one, and ask me to explain the exact internal mechanism and fix."*

---

### Lab 04 — APIs, Streaming, Structured Output & Concurrency
**Sources to check:** `ollama-roadmap-claude.md`, `labs/04-apis-structured-output/README.md`

* **Customize Audio Overview Prompt:**
  > *"The listener is a Senior Platform Engineer studying Lab 04 (APIs, Streaming, Structured Output, OpenAI Compatibility, and Concurrency). Walk through the 6 engineering lessons: (1) why NDJSON streaming cuts warm TTFT 12x (0.10s vs 1.18s) without changing throughput, and why cold loads (4.1s) dominate both; (2) the failure modes of `ollama-python` (`timeout=None` default, lazy streaming exceptions raised on first `next()`, HTTP 500 on option type errors, and how a 1s client timeout aborts the server's model load—creating a liveness-probe trap if retries have short timeouts); (3) why JSON Schema `format` guarantees syntactic shape (35/35 valid vs `json-mode`'s string `'null'`), fails completely on `length` truncation (0/35), and still requires deterministic source checks against distractor pods and prompt injection; (4) why plain `/v1` OpenAI requests trigger 1,500 tokens of default thinking (returning an empty answer after 12.4s) and silently strip `num_ctx` and `truncate: false`, requiring `reasoning_effort: 'none'` and a `Modelfile` variant (`qwen3.5:4b-ctx8k`); and (5) why `n_slots=1` hides queue wait inside `total_duration`, and why alternating between `qwen3.5:4b` and `qwen3.5:4b-ctx8k` thrashes and reloads the runner on every switch."*
* **Chat Drill Prompt:**
  > *"Quiz me on designing a production Python incident-extraction pipeline using `ollama-python` based on Lab 04: press me on timeout configuration, which HTTP errors to retry vs fail fast, how to detect truncation and queue saturation, and why `/v1` behaves differently from `/api/chat`."*

---

### Lab 05 — Embeddings, Vector Search & Local RAG
**Sources to check:** `ollama-roadmap-claude.md`, `docs/glossary.md`, `labs/05-embeddings-rag/README.md`

* **Customize Audio Overview Prompt:**
  > *"The listener is a Senior Platform Engineer building a local runbook search and RAG system on an RTX 3080. Walk through the measured findings of Lab 05: (1) why embeddings measure 'about the same thing' rather than 'means the same' ('ready' vs 'not ready' scored 0.83–0.89, above a real paraphrase), and why similarity thresholds don't transfer between models; (2) why BM25 and dense retrieval fail on different questions (paraphrases vs identifiers like `toomanyrequests`) and how Reciprocal Rank Fusion got 33/33 without tuning; (3) why finding the right document is easy but the right section is hard, and why 'search small, return big' won end to end (47/51 vs 44/51); (4) how one example id in the prompt (`[oomkilled]`) became the most-cited runbook, and why fixing the prompt mattered more than any retrieval change; (5) why a 2× bigger reader (`9b`) didn't score better; and (6) what sqlite-vec added operationally (persistence, dimension checks, metadata filters, bit quantization with re-ranking) and what it didn't change (quality)."*
* **Chat Drill Prompt:**
  > *"Act as a reviewer of my RAG design for on-call runbooks. Ask me one question at a time about Lab 05: chunk size and boundaries, BM25 vs dense vs hybrid, how to build an eval set without author bias, how to detect a retrieval miss the reader won't notice (o07), and when a vector database is worth adding."*

---

### Lab 06 — Tool Calling, MCP & AI Agents
**Sources to check:** `ollama-roadmap-claude.md`, `docs/glossary.md`, `labs/06-tools-mcp-agents/README.md`

* **Customize Audio Overview Prompt (pick the steps you are studying):**
  > *"The listener is a Senior DevOps Engineer learning to run AI agents safely against local models. Walk through Lab 06: (1) Step 1: why `qwen3.5:4b` made zero invalid tool calls yet still failed by stopping too early or reading the wrong logs; (2) Step 2: the approval gate, prompt injection through pod logs (followed 1/24, blocked every time), and why the model's report of a denial can't be trusted; (3) Step 3: why `qwen2.5-coder:7b` executed nothing despite the `tools` capability, and why the bigger 9b was the most injectable with thinking off; (4) Step 4: what MCP standardizes (discovery, transport) and what it doesn't (trust, approval), shown by a lying server whose read-only hint let deletes run; (5) Step 5: Claude Code on a local model, a 16,137-token prompt silently cut to 2,050 at the default context, the prefix cache, and an auto-mode classifier running on the same 4B model; (6) Step 6: Codex via `ollama launch codex`, ChatGPT-connected apps offered to a local model, a sandbox that failed closed, stored trust that removed every approval prompt, and an agent that reported test results it never saw."*
* **Chat Drill Prompt:**
  > *"Give me Lab 06 incidents one at a time (a delete approved for a pod that doesn't exist, an MCP server that marks delete_pod read-only, a coding agent that answers a greeting instead of the question, an agent that says 'all tests pass' with no test output) and ask me for the root cause and the deterministic control that prevents it."*

---

## 4. Reusable Template for Future Labs (Phases 7–9)

When you finish a future lab (`labs/07-*` through `labs/09-*`), upload its `README.md` to the same NotebookLM notebook, check only `ollama-roadmap-claude.md` + that lab's `README.md`, and use this Audio Overview prompt:

> *"The listener is a Senior Kubernetes and DevOps Engineer running hands-on local LLM experiments on an RTX 3080 (10 GB VRAM). Do not give a generic overview. Walk step-by-step through the hypotheses, exact measured numbers, refuted predictions, and troubleshooting failure modes in the checked Lab README. Explain every concept using the DevOps analogies in the document and highlight what matters for production engineering."*
