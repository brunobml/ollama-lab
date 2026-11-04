# Lab 08 — Operations: Ollama in Containers and Kubernetes

## 1. Core Concept & DevOps Analogy

So far Ollama has been a desktop process: a tray app on Windows, listening on `127.0.0.1:11434`, with its models in `%USERPROFILE%\.ollama\models`. To run it like any other service, it goes into a container and then into a cluster. That raises the usual platform questions, plus one GPU-specific one:

| Question | Container world | What's special for inference |
|---|---|---|
| Where does the state live? | Volumes, bind mounts, PVCs | The "state" is 3–7 GB of weights per model; how fast you can read it decides cold-start time |
| Who can reach it? | Published ports, Services, NetworkPolicies | Ollama has **no authentication**; the port *is* the access control |
| How does it get hardware? | `--gpus`, `nvidia.com/gpu` limits, device plugins | On Windows the GPU reaches Linux through a paravirtualized device (`/dev/dxg`), not `/dev/nvidia0` |
| What if two services share a node? | Requests and limits | VRAM isn't managed by the kernel or by Kubernetes; two Ollama instances only see each other as "less free memory" |

The analogy from Lab 01 still holds: weights are the image layers, Ollama is the runtime. This lab puts that runtime inside another runtime.

## 2. Prerequisites & Hardware Budget

* Docker Desktop 4.97.0 (engine 29.9.2), WSL2 backend, started by hand (AutoStart is off). The RoadMap's "Docker Desktop is unable to start" (2026-10-10) is no longer true: the engine reached `running` on 2026-10-14 and 2026-10-27.
* The Docker VM: 16 CPUs, 31.3 GiB RAM (half the host), kernel 6.18.54.1-microsoft-standard-WSL2, an `nvidia` runtime registered by Docker Desktop.
* Image `ollama/ollama:0.40.13`, the same version as the Windows install (pulled 2026-11-04, 3.39 GB compressed, 5.06 GB on disk, digest `sha256:b2f0122c476b…`, matches Docker Hub).
* No model downloads: the container reads the Windows store read-only (D-14).
* VRAM: each step runs **one** model on the GPU at a time, except Step 1 Part D (the 4b + a 0.6 GB embedding model, ≈6.8 GB in total). Loading the 4b in both instances is opt-in (`--contend`) and goes past the 8,500 MiB ceiling.
* **Your kubeconfig has a real AKS admin context** (`c8-aks-demo-admin`). No current context is set. Every `kubectl` and `helm` command in this lab passes `--context` explicitly, for a local cluster only.

**Existing Docker state, left alone:** 79 images (30 GB), 48 volumes (50 GB), 9 stopped containers (a k3d cluster `argolab`, localstack, moto, …), an older `ollama/ollama:latest` (0.23.1, May 2026) and a volume named `ollama` holding `qwen2.5-coder`. This lab uses its own names (`ollama-lab-*`) and label `ollama-lab=step1`, and refuses to run if those names exist without its label.

## 3. Architecture

```mermaid
flowchart LR
    subgraph Windows
        C[Python script] -->|127.0.0.1:11434| W[Ollama.exe 0.40.13]
        C -->|127.0.0.1:11435 / 11436| P[Docker Desktop port proxy]
        W --> S[(C:\Users\bruno\.ollama\models)]
        W --> G[(RTX 3080, WDDM driver)]
    end
    subgraph "Docker VM (WSL2)"
        P --> O[ollama serve in container :11434]
        O -->|9p bind mount, read-only| S
        O -->|ext4 named volume| V[(ollama-lab-models)]
        O -->|/dev/dxg + driver libs over 9p| G
    end
```

**Verified before Step 1 (2026-11-04):**
* **GPU passthrough works with no extra setup.** `docker run --gpus=all debian:bookworm-slim nvidia-smi` shows the RTX 3080 and driver 591.86. With `--gpus=all` the container gets `/dev/dxg` and the Windows driver files (`libcuda.so.1.1`, `nvidia-smi`) mounted read-only from the host over 9p. Without the flag it gets neither. There's no `/dev/nvidia0`: that matters later for Kubernetes device plugins.
* **The container uses the same GPU stack as Windows:** `library=CUDA compute=8.6 … libdirs=ollama,cuda_v13 driver=13.1`, 8.9 GiB available, default context 4,096, and the same runner flags (`--mmproj …`, `--flash-attn auto`). The llama.cpp builds differ: the Windows one has OpenMP and a different list of CUDA architectures.
* **Ollama in the container listens on `[::]:11434`** (`OLLAMA_HOST=0.0.0.0:11434` is baked into the image). Who can reach it is decided by `-p`, not by Ollama. Ollama's Docker docs use `-p 11434:11434`, which publishes on every host interface. This lab uses `-p 127.0.0.1:11435:11434` (D-06).
* **On start, Ollama prunes "unused blobs"** (`total unused blobs removed: 0`). With a read-write mount of the Windows store, a container could delete files the Windows install uses. Read-only removes that risk.
* **The container generates its own identity key** (`Couldn't find '/root/.ollama/id_ed25519'. Generating new private key.`) because only `models/` is mounted. It's a different identity from the Windows install, and it's lost when the container is removed.
* **Probe timings** (qwen3.5:4b, 23-token prompt, 57 tokens out, temperature 0):

| | load | wall | generation |
|---|---:|---:|---:|
| container, first load ever, bind mount | 77.5 s | 119.9 s | 74.1 tok/s |
| container, reload, bind mount | 26.1 s | 27.3 s | 97.2 tok/s |
| container, warm | 0.15 s | 1.3 s | 97.4 tok/s |
| Windows native, reload | 3.7–6.4 s | 4.7–7.5 s | 115.6–117.6 tok/s |
| Windows native, warm | 0.01 s | 1.0 s | 120.3 tok/s |

Two surprises that Step 1 is built to measure properly: **loading over the bind mount is ≈5× slower**, and **generation in the container is ≈18% slower** even with the model fully on the GPU.

## 4. Step-by-Step Guide

### Step 1 — Ollama in a container next to the Windows install

**What we're learning:**
* how a container gets the GPU on Docker Desktop;
* how the storage choice decides cold-start time;
* what a published port really exposes;
* what happens when two Ollama instances share one GPU.

**Three backends,** all running Ollama 0.40.13 with `qwen3.5:4b`, the same prompts, temperature 0, seed 42 and thinking off, measured one at a time:

| backend | where the weights come from | port |
|---|---|---|
| `windows` | native install, NTFS | 127.0.0.1:11434 |
| `bind` | container; Windows store bind-mounted **read-only** (9p across the VM boundary) | 127.0.0.1:11435 |
| `volume` | container; the 4b **copied locally** (no download) into the named volume `ollama-lab-models` (ext4 inside the Docker VM's disk) | 127.0.0.1:11436 |

**Parts:**
* **A. Exposure:** the container's `PortBindings`, `OLLAMA_HOST` inside it, and which address Windows is actually LISTENING on.
* **B. Load:**
  * reload time ×3 per backend (unload with `keep_alive: 0`, then the same request);
  * `docker run` → API answering;
  * `docker restart` → API answering, with the models still listed;
  * the copy into the volume, in MB/s.
* **C. Speed:**
  * warm generation (median of 5, 256 tokens);
  * prompt evaluation (median of 3, ≈1.5K-token prompt; the run number up front defeats the prefix cache);
  * whether the containers give the same text as Windows.
* **D. Two instances on one GPU:** Windows holds the 4b while the bind container loads `embeddinggemma:300m`. The measures are VRAM after each load, each instance's `/api/ps`, and whether Windows' tok/s changes.

```powershell
# Docker Desktop running; the Windows Ollama running with no model loaded (the script checks)
.\.venv\Scripts\python scripts\container_ollama.py              # ≈5–8 min
.\.venv\Scripts\python scripts\container_ollama.py --contend    # optional: both instances load the 4b (overflow past 8,500 MiB)
```

Output: `results\raw\container-ollama-<stamp>\` with `report.md` and `data.json` (every request, including the text). The script removes its containers at the end and keeps the 3.3 GB volume for later steps (§7 removes it).

**Predictions:**

| Measure | Prediction | Why |
|---|---|---|
| A: bindings | `HostIp 127.0.0.1` for 11435/11436; Windows LISTENING on `127.0.0.1:` only, owned by Docker's backend process, not by a container | Docker Desktop publishes ports through a Windows-side proxy; `OLLAMA_HOST=0.0.0.0` only means "all interfaces *inside* the container's network namespace" |
| B: reload, Windows | 3–7 s | Probe |
| B: reload, bind | 20–30 s | Probe: 26 s. Every load reads 3.3 GB through 9p (≈130 MB/s) |
| B: reload, volume | 2–6 s (similar to Windows) | ext4 inside the VM, and the VM's page cache keeps the file after the first read |
| B: copy into the volume | 20–40 s (100–170 MB/s) | The same 9p read, once |
| B: `docker run` → API | 3–6 s | GPU discovery took ≈3 s after listen in the probe |
| B: restart | models still listed (6), nothing downloaded | The bind mount is outside the container's writable layer |
| VRAM Δ per load | ≈3,850 MiB on all three, within ±50 | Same weights, same context, same CUDA library |
| C: generation | containers 15–20% below Windows; **bind = volume** | Probe 97 vs 116–120. Storage can't matter once the weights are on the GPU; the cause (the `/dev/dxg` path, or the different build) isn't known |
| C: prompt eval | containers also lower; gap size unknown | Same open cause |
| C: same text | identical to Windows (low confidence) | Same version and CUDA library; but the builds differ, and Lab 03 showed byte-identity depends on the exact kernels |
| D: VRAM | +≈3,850 (Windows 4b), then +600–900 (embedding in the container); both `size == size_vram` (100% GPU) | Each instance sees only what's free when it loads |
| D: Windows tok/s with both loaded | unchanged (±3%) | The embedding model sits idle; memory, not compute, is shared |

## 5. Hypotheses → Observed Metrics

### Step 1 — runs 1 and 2 (2026-11-04, `qwen3.5:4b`, temperature 0, seed 42, think off; run 2 with `--contend`)

Runs: `container-ollama-20261104-124536` and `…-133051`. Afterwards I ran three follow-up probes by hand: the CUDA kernel cache, reachability from the LAN address, and Windows tok/s in each Part D state (5 samples each).

**1. The port is the access control, and it held.**
* `PortBindings` was `{'11434/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '11435'}]}`, while inside the container `OLLAMA_HOST=0.0.0.0:11434`.
* The Windows-side listener is `127.0.0.1:11436`, owned by `com.docker.backend`, not by a container process.
* Connecting to the machine's Wi-Fi address (`10.0.0.202`) was **refused** on both 11434 (Windows Ollama) and 11436 (the container). Connecting to `127.0.0.1` answered.
* **Tooling caveat:** `netstat -ano` didn't list Docker's listener at all, so run 1's "listening" list was empty. Later in the session it stopped listing Ollama's own listener too, and `Get-NetTCPConnection` reported that one with state `0`. The cause is not known; this is a Windows 11 build 26300 machine. The script now checks reachability by connecting to the LAN address instead of reading listener tables.

**2. Storage decides the cold start.** Both runs:

| backend | first request in a new container (load / wall) | reloads 2–3 (load) | copy / start |
|---|---|---:|---|
| windows | 5.4–8.3 s / 6.3–9.3 s | 4.05–4.47 s | – |
| bind (9p, read-only) | 65.7–68.0 s / 108–109 s | 25.4–27.1 s | `docker run` → API 5.2–7.6 s; `docker restart` → API 4.1–4.4 s, 6 models listed |
| volume (ext4) | 42.6–48.9 s / 85–94 s | **2.85–3.16 s** | copy 3.32 GB in 21.1–21.4 s (155–158 MB/s); `docker run` → API 3.6–3.9 s |

* **Every reload from the bind mount re-reads 3.3 GB across the VM boundary** (≈130 MB/s). From the volume it's faster than native Windows. The VM's page cache keeps the file, and the VM has 31 GiB of RAM to do it.
* Copying into the volume took one 21 s read, and it pays back after the first reload.

**3. A new container pays ≈80–90 s once, whatever the storage.** That's the first-request wall time minus a normal reload. The first long prompt also ran at 2.0–2.3K tok/s instead of ≈6K. The follow-up probe showed why:
* the container's `/root/.nv/ComputeCache` is empty at start (4 KB);
* it holds **78 MB after the first request**: the CUDA driver compiles GPU kernels on first use;
* after a `docker restart`, which keeps the container's filesystem, the first request took **2.7 s load / 3.9 s wall**.

So the cost is tied to the container's writable layer. Every `docker rm` + `run`, every new image version, and every rescheduled pod pays it again. Windows keeps its own cache across restarts, so it doesn't pay this.

**4. Speed: generation 18–20% slower in a container; prompt processing the same.**

| backend | gen tok/s, run 1 / run 2 (median of 5, 256 tokens) | prompt tok/s, run 1 / run 2 (median of 3, ≈1.5K tokens) | VRAM Δ per load |
|---|---:|---:|---:|
| windows | 118.4 / 120.9 | 5,859 / 6,115 | 3,818 / 3,901 MiB |
| bind | 97.3 / 96.8 | 5,951 / 5,727 | 3,901 / 3,899 MiB |
| volume | 96.2 / 97.0 | 6,035 / 5,969 | 3,901 / 3,900 MiB |

* **bind = volume:** storage doesn't matter once the weights are on the GPU.
* **Prompt processing is within ±5% of Windows, and generation isn't.** Processing a prompt is a few large batched GPU operations. Generating is one small step per token, so a fixed overhead per GPU call shows up there and not in prefill. That pattern fits a per-call cost on the `/dev/dxg` path into the Windows driver. It's a hypothesis: the two llama.cpp builds also differ, and I haven't separated the two causes.
* **All three backends gave byte-identical text** on the short prompt, and the same answers to the long one (`3, 3, 13`). Those long-prompt answers are also wrong: the true count is 6. That's Lab 07 Step 3's counting failure again, in text this time.
* `/api/ps` said 2,995 MiB on all three, against ≈3,900 measured. It's the same gap as in Lab 07, and the container didn't change it.
* **Container RAM:** bind 2.7–3.2 GiB, volume 770 MiB (`docker stats`), with the weights in VRAM in both cases. The likely reason is that the bind-mounted file's pages are charged to the container. That's not verified.

**5. Two instances on one GPU: memory adds up, and neither instance knows about the other.**
* GPU memory went idle 2,203 → Windows 4b **6,104** → plus the container's embedding model **7,017 MiB** (run 2: 2,061 → 5,964 → 6,875). That's +3,901 and +913. Both instances reported 100% GPU (`size == size_vram`).
* **Windows tok/s didn't change.** The script's single sample said 94.9 and 104.8 tok/s, which looked like a 12–20% drop. The follow-up probe, 5 samples per state, gave medians of **119.0** (no container), **118.8** (container idle), **119.4** (container + embedding model), 118.3 (unloaded) and 120.3 (removed). The first request after a change was sometimes 101–106. The script now takes the median of 5.
* **`--contend`** (the 4b in both instances) put **9,730 MiB in use, 510 MiB free**. Both instances still said 100% GPU, and nothing failed or warned. The container planned its load using whatever was free after Windows' allocation. The container's 4b generated 74.9 tok/s and Windows 106.1, but those are single samples, not a measured slowdown.
* On a 10 GB card that's a working configuration with no headroom: one longer context in either instance would spill. **No scheduler, kernel or Kubernetes object accounts for this.**

**Predictions vs observed:**

| Prediction | Observed |
|---|---|
| A: `127.0.0.1` only, owned by Docker's backend | ✅ (and refused on the LAN address); ❌ `netstat` couldn't show it |
| B: reload windows 3–7 s / bind 20–30 s / volume 2–6 s | ✅ 4.1–4.5 / 25.4–27.1 / 2.9–3.2 s |
| B: copy 20–40 s | ✅ 21 s (156 MB/s) |
| B: `docker run` → API 3–6 s | ✅ 3.6–5.2 s, one 7.6 s |
| B: restart keeps the models | ✅ 6 listed, nothing downloaded |
| (not predicted) | ❌ first request in a new container: +80–90 s, the CUDA kernel cache |
| VRAM Δ ≈3,850 ± 50 on all three | ✅ 3,818–3,901 |
| C: containers 15–20% slower; bind = volume | ✅ 18–20%; bind = volume |
| C: prompt eval lower in containers | ❌ the same within ±5% |
| C: identical text (low confidence) | ✅ byte-identical |
| D: +3,850 then +600–900, both 100% GPU | ✅ +3,901, +911–913 |
| D: Windows tok/s unchanged | ✅ medians 118–120 (the script's single sample said otherwise) |

**What Step 1 teaches:**
1. **On Docker Desktop, the GPU needs only `--gpus=all`.** The container gets `/dev/dxg` and the host's driver files, and the CUDA stack and VRAM look the same as on Windows.
2. **Put the weights inside the VM.** A named volume reloaded in ≈3 s, against ≈26 s through a bind mount. In Kubernetes terms: a node-local PVC or a pre-pulled cache beats a network share on every reschedule.
3. **Cold start = reading the weights + compiling GPU kernels.** The kernel cache lives in the container's writable layer. Persist `/root/.nv` (a volume), or warm up before taking traffic. A readiness probe on `/api/version` would say "ready" ≈90 s too early.
4. **The port, not Ollama, is the access control.** Ollama listens on every interface inside the container, so `-p 127.0.0.1:…` is what kept it off the LAN. Verify by connecting from the LAN address; listener tables can be wrong.
5. **VRAM is shared without any accounting.** Two instances filled the card to 510 MiB free without a warning. Budget VRAM per node by hand, and expect the same in Kubernetes: `nvidia.com/gpu: 1` hands out whole devices and doesn't track memory.
6. **Measure speed with several samples.** One request after a state change was up to 15% slow, and a single sample turned that into a false finding.

## 6. Troubleshooting & Failure Modes

| Symptom | Cause | Check / fix |
|---|---|---|
| `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` | Docker Desktop isn't running (AutoStart is off) | Start Docker Desktop; wait for the engine (`docker version` shows a Server section) |
| `nvidia-smi: not found` in a container | Started without `--gpus=all` | Add `--gpus=all`; check `/dev/dxg` exists inside |
| The first load in a container takes minutes | The weights are read across the VM boundary (9p bind mount) | Copy them into a named volume, or accept a one-time cost |
| A container can reach Ollama but the LAN can too | `-p 11434:11434` publishes on every interface | Always `-p 127.0.0.1:<port>:11434`; check with `netstat -ano` for `LISTENING` |
| Blobs disappear from the Windows store | A container with a **read-write** mount pruned them as "unused" on start | Mount the shared store `:ro`; one writer per store |
| The first request in a new container takes ≈90 s; restarts are fast | The CUDA driver compiles kernels into `/root/.nv/ComputeCache` (empty → 78 MB) inside the container's writable layer | Mount a volume at `/root/.nv`, or send a warm-up request before marking the pod ready |
| Generation 18–20% slower in a container than native, prompt eval the same | Per-token overhead on the container's GPU path (hypothesis: `/dev/dxg` GPU-PV, or the different build) | Expect it on Docker Desktop; compare on a native Linux host before blaming Ollama |
| `netstat -ano` shows nothing on a published port that answers | Listener tables mis-reported on this Windows build (Docker's backend missing; Ollama's state `0`) | Test reachability: connect from the LAN address, not only `127.0.0.1` |
| Two instances both say 100% GPU and the card is nearly full | Each instance plans with the free VRAM it sees; nothing coordinates them | Budget per GPU by hand; leave headroom for context growth |
| A tok/s figure 10–20% off from one request | The first request after a load or state change is sometimes slow | Median of ≥5 requests |
| `docker run --entrypoint /bin/ollama …` fails from Git Bash | Git Bash rewrites `/bin/ollama` to `C:/Program Files/Git/bin/ollama` | `MSYS_NO_PATHCONV=1`, or use PowerShell |

## 7. Cleanup

```powershell
docker rm -f ollama-lab-bind ollama-lab-volume 2>$null   # the script already does this
docker volume rm ollama-lab-models                          # the 3.3 GB copy of the 4b
docker image rm ollama/ollama:0.40.13                       # optional: 5 GB; keep it for later steps
ollama stop qwen3.5:4b
```

Docker Desktop can stay running or be quit from the tray. Nothing in this lab changes its settings.

## 8. Check Your Understanding

1. The image sets `OLLAMA_HOST=0.0.0.0:11434`, yet the API in Step 1 isn't reachable from the LAN. Which setting decides that, and what would `-p 11434:11434` change?
2. Why can't the container see the GPU without `--gpus=all`, and what does it get when you add the flag?
3. Why does a reload take ≈26 s from the bind mount, but only seconds from a named volume or on Windows? Which of those numbers matters for a Kubernetes pod that's rescheduled?
4. Why must a model store shared by two Ollama instances be mounted read-only in at least one of them?
5. A new container's first request took ≈90 s, and after `docker restart` it took 4 s. Where did the time go, and how would you make a rescheduled Kubernetes pod avoid it?
6. Generation was 18–20% slower in the container but prompt processing wasn't. What does that pattern suggest about where the overhead is?
7. With both instances holding the 4b, the GPU had 510 MiB free and both still said 100% GPU. What would happen if one of them got a 32K-context request, and what in Kubernetes would have prevented this?
