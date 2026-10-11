# Environment Assessment

Read-only assessment performed on 2026-10-10. Nothing was installed or modified.

## Summary

| Area | Observed |
|---|---|
| OS | Windows 11 Home |
| CPU | Intel Core i7-10700K @ 3.80GHz, 8 cores / 16 threads |
| RAM | 64 GB total, ≈44.5 GB free at idle |
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB, compute capability 8.6 |
| NVIDIA driver | 610.88 (Ollama requires ≥ 550 for this GPU — OK) |
| VRAM in use at idle | ≈1,647 MiB (desktop + virtual display adapters) |
| Ollama | 0.40.2, `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` |
| Ollama processes | `ollama app` (tray app, supervises) + `ollama` (server) |
| Ollama listener | `127.0.0.1:11434` only |
| `OLLAMA_*` env vars | none set (all defaults) |
| Model store | `%USERPROFILE%\.ollama\models` — empty |
| Ollama logs | `%LOCALAPPDATA%\Ollama\server.log`, `app.log` |
| Ollama GPU detection | `library=CUDA compute=8.6 ... total="10.0 GiB" available="8.9 GiB"` |
| WSL2 | Ubuntu 24.04.5 LTS, kernel 6.18.40.1-microsoft-standard-WSL2, NAT networking (no `.wslconfig`) |
| GPU in WSL2 | Visible (`nvidia-smi` reports the RTX 3080) |
| WSL2 → Windows Ollama | Unreachable via `localhost` and via the host gateway (`172.25.96.1`) |
| Python | Windows: 3.9.13 and 3.12.1 (`py` launcher); WSL2: 3.12.3 |
| Docker Desktop | CLI 29.8.1; engine error: "Docker Desktop is unable to start" |

## How the pieces fit

```text
┌──────────────────────── Windows 11 ─────────────────────────┐
│  ollama CLI ───HTTP──┐                                       │
│  PowerShell/Python ──┼──▶ Ollama server :11434 (127.0.0.1)   │
│  Codex (later) ──────┘     │  model store ~/.ollama/models    │
│                            ▼                                  │
│                 llama.cpp-based runner ──CUDA──▶ RTX 3080     │
│                            └── overflow layers ──▶ CPU + RAM  │
│                                                               │
│  WSL2 Ubuntu (NAT) ──✗──▶ :11434                              │
│  Docker Desktop ──✗ engine not starting                       │
└───────────────────────────────────────────────────────────────┘
```

## Implications

* **WSL2 cannot reach Ollama** because Ollama binds `127.0.0.1` and WSL2 uses NAT (its own network namespace). Options, all requiring approval:
  * WSL2 mirrored networking (`networkingMode=mirrored` in `%USERPROFILE%\.wslconfig`) — recommended; Ollama stays private.
  * `OLLAMA_HOST=0.0.0.0` — exposes Ollama on all interfaces, including the LAN; avoid.
  * A second Ollama inside WSL2 — two servers competing for 10 GB VRAM.
* **Docker** is needed only from Phase 8; it will be diagnosed then.
* **Phases 0–4** run from PowerShell 7 on Windows against `localhost`.

## Commands used

```powershell
(Get-Command ollama).Source
Get-Process ollama*
ollama list; ollama ps
Invoke-RestMethod http://127.0.0.1:11434/api/version
Get-NetTCPConnection -LocalPort 11434 -State Listen
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version,compute_cap --format=csv
Select-String "$env:LOCALAPPDATA\Ollama\server.log" -Pattern 'inference compute'
wsl -l -v
wsl -d Ubuntu -- curl -s -m 3 http://127.0.0.1:11434/api/version
docker version; docker info
```
