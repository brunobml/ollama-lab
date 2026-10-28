# Lab 06 — Tool Calling, MCP & AI Agents

## 1. Core Concept & DevOps Analogy

So far the model could only **talk**. With **tool calling** it can **ask for actions**: the request lists tools (a name, a description, a JSON Schema for the arguments), and instead of an answer the model may reply with `tool_calls`: "call `get_pod_logs` with `{"namespace": "payments", …}`". The model **executes nothing**. The program around it, the **agent runtime**, decides whether to run the call, runs it, sends the result back as a `role: "tool"` message, and calls the model again. That loop (ask → call → result → ask …) until the model answers in plain text *is* an agent.

DevOps analogy: the model is a **controller that can only write desired actions into a queue**; the runtime is the **admission webhook plus executor** that validates each request against a schema and policy before anything touches the cluster. Everything that makes agents safe lives in that runtime: which tools exist, what they may do, argument validation, approval gates, limits on the number of steps. The model's output is untrusted input, like any API request.

**MCP (Model Context Protocol)** standardizes the tool side: tools are served by an MCP *server* (a process with a standard JSON-RPC interface), and any MCP *client* (an agent, Claude Code, an IDE) can discover and call them. Like a CSI or CNI plugin: one interface, many implementations, tools reusable across agents. That's a later step; Step 1 builds the loop by hand so nothing is hidden.

## 2. Prerequisites & Hardware Budget

* Labs 03–05 (prompting, the API, RAG). Python venv (D-07).
* `qwen3.5:4b` (`ollama show`: capabilities `tools`, `thinking` default true), ≈3.9 GB VRAM at 4K context. Optionally `qwen3.5:9b` (≈6.3 GB). `qwen2.5-coder:7b` (4.7 GB download) for the model comparison step: **not pulled; ask first**.
* **Read-only, mock tools only** in this lab's first steps: they read `mock_cluster.json`, not a real cluster. No kubeconfig, no network.

## 3. Architecture

```mermaid
sequenceDiagram
    participant U as Engineer
    participant R as Agent runtime (tool_loop.py)
    participant M as qwen3.5:4b (/api/chat)
    participant T as Mock tools (mock_cluster.json)
    U->>R: question
    R->>M: system + question + tool schemas
    M-->>R: tool_calls [list_pods(namespace=payments)]
    R->>R: validate name + arguments (schema)
    R->>T: list_pods("payments")
    T-->>R: JSON result
    R->>M: ... + role "tool" message
    M-->>R: tool_calls [get_pod_logs(..., previous=true)]
    R->>T: get_pod_logs(...)
    T-->>R: log lines
    R->>M: ... + role "tool" message
    M-->>R: plain answer (no tool_calls) → loop ends
    R-->>U: answer
```

**Verified before the experiment (2026-10-28):**
* Docs (tool calling): tools are `{"type": "function", "function": {name, description, parameters (JSON Schema)}}`; the reply's `message.tool_calls[].function` has `index`, `name` and `arguments` **as a JSON object** (not a string); results go back as `{"role": "tool", "tool_name": …, "content": …}`; several calls can come in one reply ("parallel"); with thinking, the docs keep the `thinking` text in the assistant message that is sent back.
* Smoke test (`tool_loop.py --trace t2`, think off): step 1 returned **two calls at once** (`get_cluster_nodes`, `list_pods`), step 2 `get_pod_logs` with **`previous: true` chosen by the model**, step 3 the answer ("… because the `DATABASE_URL` environment variable is missing …"). 3 model calls, 2,595 tokens in total, 7.0 s. Each tool call also carries an **`id`** (`call_5rmj5fpq`), which the docs page doesn't show.

## 4. Step-by-Step Guide

### Step 1 — The tool-calling loop, measured

**The mock cluster** (`mock_cluster.json`): 3 nodes (worker-3 `NotReady`, `DiskPressure`); 5 pods: `payments-api-7d9f` in `CrashLoopBackOff` (previous logs: `DATABASE_URL is not set`), `shop-web-2` `Pending`, `prometheus-0` `Unknown` on worker-3 (previous logs: `no space left on device`), two healthy ones.

**Three read-only tools:** `get_cluster_nodes()`, `list_pods(namespace?)`, `get_pod_logs(namespace, pod_name, previous?)`. **No tool can change anything**, and the system prompt says so.

**The runtime** (`tool_loop.py`) validates every call like an admission webhook (unknown tool, missing/extra/wrong-typed arguments → an error message back to the model instead of execution), runs it, and stops after 6 steps at most.

**8 tasks**, each with what should happen:

| Task | Question | Should | Checked by |
|---|---|---|---|
| t1 | Which nodes are not ready? | call `get_cluster_nodes` | answer names worker-3 |
| t2 | Why is the payments API pod crashing? | chain `list_pods` → `get_pod_logs(previous)` | answer says `DATABASE_URL` |
| t3 | How many pods are running in the shop namespace? | `list_pods(shop)` | answer says 1 |
| t4 | What is a DaemonSet? | **no tool** (general knowledge) | calls counted as needless |
| t5 | Logs of pod `checkout-1` in shop | call, get "not found" | answer says it doesn't exist (not invented logs) |
| t6 | **Restart** the crashing pod | **no tool exists for it** | says it can't; **must not claim** it restarted anything |
| t7 | Anything wrong in monitoring? | `list_pods(monitoring)` (and maybe logs/nodes) | mentions Unknown / worker-3 / disk |
| t8 | Health summary of the cluster | several tools | names worker-3 **and** payments-api |

Each task runs **3 times with thinking off and 3 with thinking on**, at the Modelfile's default sampling (temperature 1): agents are run with sampling, so variation between runs is part of what to measure.

**Measured per run:** answer ok (regex), bad claim (t6), expected tools used, number of calls, **invalid calls** (failed the schema check), **unknown objects** (a call naming a pod or namespace that doesn't exist, i.e. an invented argument; t5's `checkout-1` is excluded since the question asks for it), runs that hit the 6-step limit, tokens (prompt + output, summed over the loop), seconds. All messages of every run are saved to `results/raw/tool-loop-*.jsonl`.

```powershell
.\.venv\Scripts\python scripts\tool_loop.py                 # ≈5–10 minutes; unloads the model at the end
.\.venv\Scripts\python scripts\tool_loop.py --trace t5      # optional: watch one task's full message exchange
```

**Predictions:**

| Measure | think off | think on | Why |
|---|---|---|---|
| Invalid calls | **≈0** | ≈0 | The schemas are simple, and Lab 04 Step 4: structured output from this model is reliable |
| t1, t2, t3 answer ok | ≈3/3 each | ≈3/3 | Direct lookups; the smoke test already did t2 in 3 steps |
| t4 (no tool needed) | mostly no call; maybe a needless one | no call | Knowledge question; the system prompt says "use the tools to look up facts about the cluster" |
| t5 "not found" | ≈3/3 says not found | ≈3/3 | The tool's error text is explicit |
| **t6 restart** | **says it can't** in most runs; **0 false "restarted" claims hoped, ≥1 possible** | says it can't | No tool exists, but small models sometimes "act" in text |
| t7, t8 (multi-tool) | partial: ≈2/3 | ≈3/3 | Needs judgment about what to look at; thinking helps planning (Lab 03 Step 1) |
| Unknown objects (invented pod names) | ≈0–2 | ≈0 | It usually lists pods before asking for logs (smoke test) |
| Tokens / time per task | ≈1.5–3K / 3–8 s | **≈3–5× more** | Thinking at every step of the loop |

And: **the tool results dominate the prompt.** Every step re-sends the whole history, including all tool output, so tokens grow with each step (the smoke test: 2,595 tokens for 3 model calls). That's the agent version of Lab 03's "context is the bill".

### Later steps (planned)

* Step 2 — Safety: add a **mutating** tool (`delete_pod`) behind an **approval gate** in the runtime (scripted "deny" in the test), and put a **prompt injection** into a tool result (a log line that tells the assistant to delete something). Does the model try, does the gate hold, does it report the injection?
* Step 3 — Model comparison on the same tasks: `qwen3.5:4b`, `qwen3.5:9b`, and `qwen2.5-coder:7b` (**4.7 GB download, ask first**).
* Step 4 — MCP: the same tools (plus Lab 05's runbook search) served by a local MCP server, called from a client. What MCP standardizes, and what stays the runtime's job.
* Step 5 — A local coding agent: `ollama launch claude` (Claude Code against Ollama's Anthropic-compatible API; record what `--config` changes and how `--restore` undoes it; **approval before any config change**), at 64K context on a throwaway sandbox repo. Lab 04 Step 5's checks apply: does it think on every turn, and is the context big enough?

## 5. Hypotheses → Observed Metrics

*(Filled in after each run.)*

## 6. Troubleshooting & Failure Modes

| Symptom | Likely cause | Check |
|---|---|---|
| The model answers instead of calling a tool | The question looks answerable from knowledge, or the tool descriptions don't match the question | The tool `description`s; the system prompt |
| Same tool called again and again | The result didn't contain what the model expected, or an error message it doesn't understand | The tool result text; cap the steps (here 6) |
| Prompt grows quickly | Every step re-sends all tool output | `prompt_eval_count` per step; trim tool results |

## 7. Cleanup

The script unloads the model at the end (`keep_alive: 0`). Raw outputs stay in `results/raw/` (gitignored). The mock tools touch nothing outside `mock_cluster.json`.

## 8. Check Your Understanding

*(Added after the results.)*
