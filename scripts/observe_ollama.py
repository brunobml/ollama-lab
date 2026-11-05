"""Lab 08 Step 4. Observability: what Ollama reports about itself, and what a sidecar has to add.

Run with the venv's interpreter, Docker Desktop running, from the repo root (≈6 min):
    .\\.venv\\Scripts\\python scripts\\observe_ollama.py

One lab container, ollama-lab-obs (ollama/ollama:0.40.13, --gpus=all, 127.0.0.1:11438, label ollama-lab=step4), on
Step 1's model volume, with OLLAMA_NUM_PARALLEL=1 and OLLAMA_MAX_QUEUE=2. A lab volume ollama-lab-obs-nv keeps the
CUDA kernel cache between the two container configurations. scripts/metrics_proxy.py runs in front of it on
127.0.0.1:11439. Everything is removed at the end, including that volume.
Parts:
  A  what Ollama exposes: endpoints, the access log line, llama-server's timing lines, log bytes per request
  B  8 concurrent streaming requests on a warm model: queue wait from the proxy vs Ollama's own numbers
  C  overload: a burst on a cold model vs on a warm one, with OLLAMA_MAX_QUEUE=2
  D  a client that disconnects mid-stream: what the access log, llama-server and the proxy record
  E  OLLAMA_DEBUG=1 + OLLAMA_DEBUG_LOG_REQUESTS=1: log volume, and where the prompts go
  F  the proxy's /metrics after all of it
"""
import json
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import metrics_proxy  # noqa: E402

NAME, LABEL, IMAGE = 'ollama-lab-obs', 'ollama-lab=step4', 'ollama/ollama:0.40.13'
VOLUME, NV_VOLUME, MODEL = 'ollama-lab-models', 'ollama-lab-obs-nv', 'qwen3.5:4b'
PORT, PROXY_PORT = 11438, 11439
DIRECT, PROXY = f'http://127.0.0.1:{PORT}', f'http://127.0.0.1:{PROXY_PORT}'
PROMPT = 'Explain a Kubernetes liveness probe in two sentences.'
MARKER = 'LAB08-PROMPT-MARKER'


def run(cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if check and r.returncode:
        raise SystemExit(f'{" ".join(cmd)} failed:\n{r.stdout}\n{r.stderr}')
    return r.stdout.strip()


def docker(*args, **kw):
    return run(['docker', *args], **kw)


def vram():
    return int(run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']))


def logs():
    r = subprocess.run(['docker', 'logs', NAME], capture_output=True)
    return (r.stdout + r.stderr).decode('utf-8', 'replace')


def mark():
    """Byte offsets of the container's stdout ([GIN] access log) and stderr (server and llama-server), kept apart:
    diffing the two streams concatenated mixes them up."""
    r = subprocess.run(['docker', 'logs', NAME], capture_output=True)
    return len(r.stdout), len(r.stderr)


def since(m):
    """What each stream gained since mark() m: (text, bytes)."""
    r = subprocess.run(['docker', 'logs', NAME], capture_output=True)
    new = r.stdout[m[0]:] + r.stderr[m[1]:]
    return new.decode('utf-8', 'replace'), len(new)


def start_container(*env):
    docker('rm', '-f', NAME, check=False)
    args = ['run', '-d', '--name', NAME, '--label', LABEL, '--gpus=all',
            '-v', f'{VOLUME}:/root/.ollama/models', '-v', f'{NV_VOLUME}:/root/.nv',
            '-e', 'OLLAMA_NUM_PARALLEL=1', '-e', 'OLLAMA_MAX_QUEUE=2']
    for e in env:
        args += ['-e', e]
    docker(*args, '-p', f'127.0.0.1:{PORT}:11434', IMAGE)
    for _ in range(120):
        try:
            client.get(DIRECT + '/api/version', timeout=2)
            return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise SystemExit(f'{NAME} never answered:\n{logs()[-2000:]}')


def body(n=128, stream=False, prompt=PROMPT):
    return {'model': MODEL, 'prompt': prompt, 'stream': stream, 'think': False,
            'options': {'temperature': 0, 'seed': 42, 'num_predict': n}}


def burst(base, count, n, stream):
    """`count` requests 50 ms apart from one shared client; returns per-request client-side results."""
    res = [None] * count

    def one(i):
        t = time.perf_counter()
        if stream:
            first, final = None, {}
            with client.stream('POST', base + '/api/generate', json=body(n, True)) as r:
                for line in r.iter_lines():
                    if line and first is None:
                        first = time.perf_counter() - t
                    if line:
                        final = json.loads(line)
            res[i] = {'status': r.status_code, 'wall_s': round(time.perf_counter() - t, 3),
                      'ttft_s': round(first, 3) if first else None, **metrics_proxy.summarize(final),
                      'error': final.get('error')}
        else:
            r = client.post(base + '/api/generate', json=body(n))
            j = r.json()
            res[i] = {'status': r.status_code, 'wall_s': round(time.perf_counter() - t, 3),
                      **metrics_proxy.summarize(j), 'error': j.get('error')}
    threads = [threading.Thread(target=one, args=(i,)) for i in range(count)]
    for th in threads:
        th.start()
        time.sleep(0.05)
    for th in threads:
        th.join()
    return res


def unload():
    client.post(DIRECT + '/api/generate', json={'model': MODEL, 'keep_alive': 0})
    for _ in range(60):
        if not client.get(DIRECT + '/api/ps').json()['models']:
            return
        time.sleep(0.5)


sys.stdout.reconfigure(encoding='utf-8')
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'observe-ollama-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report, data = [], {}
say = lambda text='': (print(text, flush=True), report.append(text))
client = httpx.Client(timeout=900)       # one client: a new one costs ≈0.3 s (its SSL context), even for http://

# ---------------------------------------------------------------- preflight: nothing else of the lab on the GPU
docker('version', '--format', '{{.Server.Version}}')
others = [n for n in docker('ps', '-a', '--format', '{{.Names}}').split() if n.startswith(('ollama-lab', 'k3d-ollama-lab'))]
if others:
    raise SystemExit(f'Other lab containers exist ({others}); another experiment may be using the GPU. '
                     'Finish or remove it first.')
for p in (PORT, PROXY_PORT):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', p)) == 0:
            raise SystemExit(f'127.0.0.1:{p} is already in use')
if not docker('volume', 'ls', '-q', '--filter', f'name=^{VOLUME}$'):
    raise SystemExit(f'Volume {VOLUME} not found: run Step 1 (scripts/container_ollama.py) first.')
if not docker('images', '-q', IMAGE):
    raise SystemExit(f'{IMAGE} is not in the local image store; nothing is pulled by this script.')
try:
    loaded = client.get('http://127.0.0.1:11434/api/ps', timeout=5).json()['models']
except httpx.HTTPError:
    loaded = []                          # the Windows Ollama isn't running: nothing of it on the GPU
if loaded:
    raise SystemExit(f'The Windows Ollama has {[m["name"] for m in loaded]} loaded; run `ollama stop` first.')
idle = vram()
say(f'# observe_ollama: {IMAGE} in {NAME} (NUM_PARALLEL=1, MAX_QUEUE=2), {MODEL}, temperature 0, seed 42, '
    f'think off, {stamp}\n')
say(f'* GPU idle: {idle} MiB used')

try:
    # ------------------------------------------------------------ A: what Ollama exposes
    say('\n## A. What Ollama exposes\n')
    start_container()
    data['endpoints'] = {p: client.get(DIRECT + p).status_code
                         for p in ('/metrics', '/api/metrics', '/debug/pprof/', '/api/version', '/api/ps')}
    say(f'* endpoints (HTTP status): {data["endpoints"]}')
    t = time.perf_counter()
    first = client.post(DIRECT + '/api/generate', json=body(16)).json()
    say(f'* first request (load + kernel cache build): {round(time.perf_counter() - t, 1)} s, '
        f'load {round(first["load_duration"] / 1e9, 1)} s')
    m = mark()
    one = client.post(DIRECT + '/api/generate', json=body(128)).json()
    time.sleep(1)
    tail, nbytes = since(m)
    gin = [l for l in tail.splitlines() if l.startswith('[GIN]')]
    timing = [l for l in tail.splitlines() if 'print_timing' in l and ('eval time' in l or 'total time' in l)]
    data['one_request'] = {'log_bytes': nbytes, 'log_lines': len(tail.splitlines()), 'gin': gin,
                           'timing_lines': timing, 'response_fields': sorted(k for k in one if k not in ('response', 'context')),
                           **metrics_proxy.summarize(one)}
    say(f'* one warm request wrote {data["one_request"]["log_bytes"]} bytes / {data["one_request"]["log_lines"]} '
        f'lines of log; the access log line:')
    for l in gin:
        say(f'  * `{l}`')
    say('* llama-server timing lines (no model name, no request id):')
    for l in timing:
        say(f'  * `{re.sub(" +", " ", l.strip())}`')
    say(f'* response timing fields: {[k for k in data["one_request"]["response_fields"] if k.endswith(("duration", "count"))]}')

    # ------------------------------------------------------------ B: 8 concurrent, warm, through the proxy
    say('\n## B. 8 concurrent streaming requests on a warm model (through the proxy)\n')
    server, metrics = metrics_proxy.start(DIRECT, PROXY_PORT)
    m = mark()
    b = burst(PROXY, 8, 128, stream=True)
    time.sleep(1)
    gin_b = re.findall(r'\[GIN\] .*?\| (\d{3}) \|\s+([\d.]+\w+) \|.*"/api/generate"', since(m)[0])
    data['concurrent'] = {'client': b, 'gin': gin_b}
    say('| # | TTFT (client) | wall | Ollama total | queue (total − phases) | eval |')
    say('|---:|---:|---:|---:|---:|---:|')
    for i, r in enumerate(b):
        say(f'| {i} | {r["ttft_s"]} | {r["wall_s"]} | {r.get("total_s")} | {r.get("queue_s")} | {r.get("eval_s")} |')
    say(f'* access log latencies for these 8: {[lat for _, lat in gin_b]}')

    # ------------------------------------------------------------ C: overload, cold vs warm
    say('\n## C. Overload with OLLAMA_MAX_QUEUE=2\n')
    c = {}
    for label in ('warm', 'cold'):
        if label == 'cold':
            unload()
        res = burst(PROXY, 6, 32, stream=False)
        c[label] = res
        codes = [r['status'] for r in res]
        say(f'* {label} model, 6 requests 50 ms apart: statuses {codes}; '
            f'503 wall times {[r["wall_s"] for r in res if r["status"] == 503]} s; '
            f'200 wall times {[r["wall_s"] for r in res if r["status"] == 200]} s')
        for r in res:
            if r['status'] != 200:
                say(f'  * `{r["error"]}`')
                break
    data['overload'] = c

    # ------------------------------------------------------------ D: a client that disconnects
    say('\n## D. A client that disconnects mid-stream\n')
    m = mark()
    with httpx.Client(timeout=60) as quitter:          # created before the clock starts (≈0.3 s)
        t = time.perf_counter()
        with quitter.stream('POST', PROXY + '/api/generate', json=body(400, True)) as r:
            for n, line in enumerate(r.iter_lines()):
                if n == 20:
                    break
        quit_s = round(time.perf_counter() - t, 2)
    time.sleep(2)
    tail = since(m)[0]
    d = {'client_quit_s': quit_s, 'gin': [l for l in tail.splitlines() if l.startswith('[GIN]')],
         'cancel_lines': [l for l in tail.splitlines() if 'cancel' in l.lower()],
         'eval_lines': [re.sub(' +', ' ', l.strip()) for l in tail.splitlines() if 'print_timing' in l and ' eval time' in l],
         'proxy_record': metrics.records[-1]}
    data['disconnect'] = d
    say(f'* the client read 20 chunks and closed the connection after {quit_s} s')
    say(f'* access log: {d["gin"]}')
    say(f'* llama-server: {d["cancel_lines"]}; {d["eval_lines"]}')
    say(f'* proxy record: status {d["proxy_record"]["status"]}, client_disconnected '
        f'{d["proxy_record"]["client_disconnected"]}, wall {d["proxy_record"]["wall_s"]} s')

    # ------------------------------------------------------------ F (before E restarts the container): /metrics
    say('\n## F. The proxy\'s /metrics (after B, C and D)\n')
    scrape = client.get(PROXY + '/metrics').text
    (outdir / 'metrics.txt').write_text(scrape, encoding='utf-8')
    keep = [l for l in scrape.splitlines() if not l.startswith('#') and ('_bucket' not in l)]
    for l in keep:
        say(f'  * `{l}`')
    data['metrics_lines'] = len(scrape.splitlines())
    server.shutdown()
    (outdir / 'ollama-info.log').write_text(logs(), encoding='utf-8')

    # ------------------------------------------------------------ E: debug logging and request logs
    say('\n## E. OLLAMA_DEBUG=1 + OLLAMA_DEBUG_LOG_REQUESTS=1\n')
    start_container('OLLAMA_DEBUG=1', 'OLLAMA_DEBUG_LOG_REQUESTS=1')
    secret = f'{MARKER}: {PROMPT}'
    client.post(DIRECT + '/api/generate', json=body(16, prompt=secret))           # load
    sizes = []
    for _ in range(3):
        m = mark()
        client.post(DIRECT + '/api/generate', json=body(128, prompt=secret))
        time.sleep(1)
        sizes.append(since(m)[1])
    full = logs()
    e = {'log_bytes_per_request': sizes, 'marker_in_stdout_log': full.count(MARKER),
         'notice': [l for l in full.splitlines() if 'request debug logging' in l],
         'files': docker('exec', NAME, 'sh', '-c', 'ls -la /tmp/ollama-request-logs-*/ | head -8'),
         'marker_in_files': docker('exec', NAME, 'sh', '-c', f'grep -l {MARKER} /tmp/ollama-request-logs-*/* | wc -l'),
         'replay_script': docker('exec', NAME, 'sh', '-c', 'cat $(ls /tmp/ollama-request-logs-*/*.sh | head -1)')}
    docker('restart', NAME)
    time.sleep(5)
    e['after_restart'] = docker('exec', NAME, 'sh', '-c', 'ls -d /tmp/ollama-request-logs-*; '
                                'grep -l ' + MARKER + ' /tmp/ollama-request-logs-*/* | wc -l')
    data['debug'] = e
    say(f'* log bytes per warm request at DEBUG: {sizes} (INFO in A: {data["one_request"]["log_bytes"]})')
    say(f'* the prompt marker in the container log (stdout/stderr): {e["marker_in_stdout_log"]} times')
    for l in e['notice']:
        say(f'  * `{l[l.find("msg="):][:220]}`')
    say(f'* files holding the prompt: {e["marker_in_files"]}; listing:')
    for l in e['files'].splitlines()[:5]:
        say(f'  * `{l}`')
    say(f'* replay script: `{e["replay_script"].splitlines()[-1][:200]}`')
    say(f'* after `docker restart`: directories and files with the prompt:')
    for l in e['after_restart'].splitlines():
        say(f'  * `{l}`')
    (outdir / 'ollama-debug.log').write_text(logs(), encoding='utf-8')
    data['records'] = metrics.records
finally:
    docker('rm', '-f', NAME, check=False)
    docker('volume', 'rm', NV_VOLUME, check=False)
    say(f'\n* cleanup: container and {NV_VOLUME} removed; GPU {vram()} MiB (idle at start {idle})')
    (outdir / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    (outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'\nSaved {outdir}')
