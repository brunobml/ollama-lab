"""Lab 04 Step 6. Concurrency, queueing and cancellation, with the server as it is (OLLAMA_NUM_PARALLEL 1).

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\concurrency.py      (≈2 minutes)

Part A: 4 requests at the same moment to one model with one slot (n_slots = 1), cold and warm.
        Queued or parallel? Does the server's total_duration include the time spent waiting?
Part B: 2 models at the same moment (qwen3.5:4b and qwen3.5:4b-ctx8k, the same weights).
        Two runners, two copies in VRAM? Do they really run in parallel, and at what tok/s?
Part C: a client that disconnects mid-generation. Does the server stop generating?
        Measured by how long the next request has to wait.
"""
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

HOST = 'http://127.0.0.1:11434'
MODEL = 'qwen3.5:4b'
VARIANT = 'qwen3.5:4b-ctx8k'
PROMPT = 'Explain in detail how a Kubernetes Deployment performs a rolling update.'
report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


def vram_used():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def unload_all():
    for m in httpx.get(f'{HOST}/api/ps').json()['models']:
        httpx.post(f'{HOST}/api/generate', json={'model': m['name'], 'keep_alive': 0}, timeout=60)
    time.sleep(2)


def stream_request(model, prompt, num_predict, t0, out, barrier=None, cancel_after=None):
    """Streams one /api/chat request; records times relative to t0 (seconds) and the final stats."""
    rec = {'model': model}
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'stream': True, 'think': False,
            'options': {'temperature': 0, 'seed': 42, 'num_predict': num_predict}}
    with httpx.Client(timeout=300) as c:
        if barrier:
            barrier.wait()
        rec['sent'] = time.perf_counter() - t0
        with c.stream('POST', f'{HOST}/api/chat', json=body) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if 'first' not in rec and d.get('message', {}).get('content'):
                    rec['first'] = time.perf_counter() - t0
                    if cancel_after is not None:
                        time.sleep(cancel_after)
                        rec['cancelled'] = time.perf_counter() - t0
                        break          # leaving the with-block closes the connection
                if d.get('done'):
                    rec.update({k: d.get(k) for k in ('total_duration', 'load_duration', 'prompt_eval_duration',
                                                      'eval_count', 'eval_duration', 'done_reason')})
    rec['end'] = time.perf_counter() - t0
    out.append(rec)


def run_together(jobs, cancel_after=None):
    """Starts all jobs (model, prompt, num_predict) at the same moment; returns records sorted by first token."""
    t0 = time.perf_counter()
    barrier = threading.Barrier(len(jobs))
    out = []
    threads = [threading.Thread(target=stream_request, args=(m, p, n, t0, out, barrier, cancel_after))
               for m, p, n in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(out, key=lambda r: r.get('first', 1e9))


def table(recs):
    say(f'{"#":>2} {"Model":<17} {"Sent":>5} {"1st tok":>8} {"End":>6} {"Wall":>6} {"Server":>7} {"Wait":>6} '
        f'{"Load":>5} {"Tokens":>6} {"tok/s":>6}')
    for i, r in enumerate(recs, 1):
        server = (r.get('total_duration') or 0) / 1e9
        wall = r['end'] - r['sent']
        tps = r['eval_count'] / (r['eval_duration'] / 1e9) if r.get('eval_duration') else 0
        # Wait = time the client waited that the server doesn't count in total_duration
        say(f'{i:>2} {r["model"]:<17} {r["sent"]:>5.2f} {r.get("first", 0):>8.2f} {r["end"]:>6.2f} {wall:>6.2f} '
            f'{server:>7.2f} {wall - server:>6.2f} {(r.get("load_duration") or 0) / 1e9:>5.2f} '
            f'{r.get("eval_count") or 0:>6} {tps:>6.0f}')


say(f'# concurrency ({MODEL}, OLLAMA_NUM_PARALLEL 1)\n')
say('Times in seconds from the moment all requests were released. Wall = End - Sent (client view), '
    'Server = total_duration, Wait = Wall - Server.\n')

# --- Part A: 4 requests, one model, one slot ---
jobs = [(MODEL, f'{PROMPT} (request {k})', 200) for k in range(1, 5)]
say('## Part A1: 4 simultaneous requests, cold model\n')
unload_all()
table(run_together(jobs))
say('\n## Part A2: the same, warm\n')
table(run_together(jobs))
say('\n## Part A3: the same 4, one after another (baseline)\n')
seq, t0 = [], time.perf_counter()
for m, p, n in jobs:
    stream_request(m, p, n, t0, seq)
table(seq)
say(f'\nsequential total {seq[-1]["end"]:.2f} s')

# --- Part B: 2 models at once ---
have_variant = any(m['name'] == VARIANT for m in httpx.get(f'{HOST}/api/tags').json()['models'])
if have_variant:
    say(f'\n## Part B: {MODEL} and {VARIANT} at the same moment (same weights, two names)\n')
    unload_all()
    base = vram_used()
    recs = run_together([(MODEL, PROMPT, 400), (VARIANT, PROMPT, 400)])
    table(recs)
    peak = vram_used()
    ps = httpx.get(f'{HOST}/api/ps').json()['models']
    say(f'\nloaded: {[(m["name"], m.get("context_length"), round(m["size_vram"] / 2**20)) for m in ps]} (MiB)')
    say(f'nvidia-smi memory.used: {base} MiB before, {peak} MiB with both loaded (+{peak - base})')
    overlap = min(r['end'] for r in recs) - max(r['first'] for r in recs)
    say(f'generation overlap: {overlap:.2f} s (> 0 means both were generating at the same time)')
else:
    say(f'\n({VARIANT} not found: Part B skipped)')

# --- Part C: client disconnects mid-generation ---
say('\n## Part C: request 1 (num_predict 3000) disconnects 1 s after its first token; request 2 is sent right after\n')
unload_all()
httpx.post(f'{HOST}/api/generate', json={'model': MODEL, 'prompt': 'hi', 'stream': False,
                                         'options': {'num_predict': 1}}, timeout=120)   # warm, so only generation is measured
t0 = time.perf_counter()
first = []
stream_request(MODEL, PROMPT, 3000, t0, first, cancel_after=1.0)
second = []
stream_request(MODEL, 'What is a Kubernetes Pod? One sentence.', 50, t0, second)
c1, c2 = first[0], second[0]
say(f'request 1: first token {c1["first"]:.2f} s, disconnected at {c1["cancelled"]:.2f} s')
say(f'request 2: sent {c2["sent"]:.2f} s, first token {c2["first"]:.2f} s '
    f'(waited {c2["first"] - c2["sent"]:.2f} s), server total {c2["total_duration"] / 1e9:.2f} s')
say('If the server kept generating request 1 (≈3000 tokens at ≈115 tok/s ≈ 26 s), request 2 would wait ≈25 s.')

unload_all()
out = Path('results/raw') / f'concurrency-{datetime.now():%Y%m%d-%H%M%S}.md'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(report) + '\n', encoding='utf-8')
say(f'\nSaved to {out}')
