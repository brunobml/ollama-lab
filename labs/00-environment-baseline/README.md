# Lab 00 — Environment Baseline

## 1. Core Concept & DevOps Analogy

Ollama is a **local HTTP API server**. The `ollama` CLI, Python libraries, and coding agents are all clients of that API — the same way `kubectl`, Lens, and Argo CD are all clients of the kube-apiserver.

Most "Ollama problems" are service problems: wrong host, wrong port, nothing listening, model not loaded, not enough VRAM. You already know how to debug those once you see Ollama as a service.

## 2. Prerequisites & Hardware Budget

* Ollama 0.40.2 running on Windows (tray icon visible).
* PowerShell 7.
* No model is loaded; this lab uses **0 MiB** of model VRAM.

## 3. Architecture

```mermaid
flowchart LR
    CLI["ollama CLI"] -->|HTTP| S["Ollama server<br/>127.0.0.1:11434"]
    PS["PowerShell<br/>Invoke-RestMethod"] -->|HTTP| S
    S --> STORE["Model store<br/>~/.ollama/models"]
    S -->|CUDA| GPU["RTX 3080<br/>10 GB VRAM"]
    WSL["WSL2 (NAT)"] -.->|blocked| S
```

## 4. Step-by-Step Guide

### Step 1 — Ollama as an HTTP service (read-only)

```powershell
# 1. Health check — like a readiness probe
Invoke-RestMethod http://localhost:11434/api/version

# 2. "ollama list" is the same call as GET /api/tags
ollama list
Invoke-RestMethod http://localhost:11434/api/tags | ConvertTo-Json

# 3. Loaded models — "kubectl get pods" vs. "list images"
ollama ps
Invoke-RestMethod http://localhost:11434/api/ps | ConvertTo-Json

# 4. Who is listening, and on which interface?
Get-NetTCPConnection -LocalPort 11434 -State Listen |
  Select-Object LocalAddress, LocalPort, @{n='Process';e={(Get-Process -Id $_.OwningProcess).Name}}

# 5. VRAM baseline before any model is loaded
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# 6. How the server sees your hardware
Select-String "$env:LOCALAPPDATA\Ollama\server.log" -Pattern 'inference compute'
```

### Step 2 — VRAM monitor script

Build `scripts/check-vram.ps1`, a small poller around `nvidia-smi` and `/api/ps` that later labs use to watch model load and unload. (Guided in session.)

### Step 3 — Record the baseline

Write your observed values in the table below.

## 5. Hypotheses → Observed Metrics

| Check | Hypothesis | Observed |
|---|---|---|
| `/api/version` | `0.40.2` | |
| `/api/tags` | empty `models` list | |
| `/api/ps` | empty `models` list | |
| Listener | `127.0.0.1:11434` only, process `ollama` | |
| VRAM used (idle) | ≈1.6 GB of 10 GB | |
| Ollama "available" | ≈8.9 GiB, `library=CUDA` | |

## 6. Troubleshooting & Failure Modes

| Symptom | Likely cause | Check |
|---|---|---|
| `Unable to connect` on 11434 | Server not running | Tray icon; `Get-Process ollama*` |
| `library=cpu` in server.log | GPU not detected | Driver version; `nvidia-smi` |
| Port in use by another process | Second Ollama (e.g. Docker) | `Get-NetTCPConnection -LocalPort 11434` |
| WSL2 cannot connect | NAT + localhost bind (expected) | See `docs/environment.md` |

## 7. Cleanup

Nothing to clean up — this lab is read-only.

## 8. Check Your Understanding

1. `ollama list` and `ollama ps` are both empty. What does each tell you, and what is the Kubernetes equivalent of each?
2. Why can't WSL2 reach this API, and why is that a safe default?
3. Ollama reports 8.9 GiB available while `nvidia-smi` shows ≈1.6 GB used of 10 GB. Where does the difference come from?
4. If a teammate says "Ollama is slow", which two commands from this lab would you run first, and why?
