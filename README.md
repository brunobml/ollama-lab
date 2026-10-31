# Ollama LAB

A hands-on learning lab for local LLM inference with Ollama on an RTX 3080 (10 GB VRAM).

* [`RoadMap.md`](RoadMap.md) — the roadmap, model choices, guardrails, decision log, and progress.
* [`ollama-roadmap-claude.md`](ollama-roadmap-claude.md) — the original learning brief.

## Documentation (`docs/`)

* [`docs/environment.md`](docs/environment.md) — verified environment baseline and Windows/WSL2/GPU layout.
* [`docs/architecture.md`](docs/architecture.md) — Ollama runtime, model store, request path, memory, and truncation rules.
* [`docs/vram-calculator.md`](docs/vram-calculator.md) — empirical VRAM sizing and token-speed calculator for this RTX 3080.
* [`docs/glossary.md`](docs/glossary.md) — LLM and Ollama terms mapped to DevOps/Kubernetes concepts.
* [`docs/lab-assessment.md`](docs/lab-assessment.md) — technical assessment from 2026-10-27, with a status update.
* [`docs/notebooklm-guide.md`](docs/notebooklm-guide.md) — how to study the labs with Google NotebookLM (workflow, per-lab Audio Overview and quiz prompts).

## Labs (`labs/`)

* [Lab 00 — Environment Baseline](labs/00-environment-baseline/README.md) ✅
* [Lab 01 — Foundations: Models on Disk and in Memory](labs/01-foundations/README.md) ✅
* [Lab 02 — Hardware Sizing & Performance](labs/02-performance-hardware/README.md) ✅
* [Lab 03 — Prompting, Sampling & Reproducibility](labs/03-prompting-behavior/README.md) ✅
* [Lab 04 — APIs, Streaming & Structured Output](labs/04-apis-structured-output/README.md) ✅
* [Lab 05 — Embeddings, Vector Search & Local RAG](labs/05-embeddings-rag/README.md) ✅
* [Lab 06 — Tool Calling, MCP & AI Agents](labs/06-tools-mcp-agents/README.md) ✅
* **Next lab:** Phase 7 — Multimodal Models (`labs/07-multimodal/`)

Each lab README follows the same template; measured results are in its section 5 ("Hypotheses → Observed Metrics").

## Code and outputs

* [`scripts/`](scripts/) — the experiment scripts (PowerShell and Python, run from the repo root with `.venv`).
* `results/raw/` — raw run outputs (gitignored). The curated numbers live in each lab's section 5.
