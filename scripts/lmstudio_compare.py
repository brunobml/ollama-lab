"""Lab 09 Step 3. LM Studio 0.4.28 against the same GGUF: its API defaults, templates, batching and what it keeps.

Run with the venv's interpreter from the repo root, with LM Studio installed and its app or service running,
and the Windows Ollama with nothing loaded (≈6 min):
    .\\.venv\\Scripts\\python scripts\\lmstudio_compare.py

The model is the Ollama blob hard-linked into ~\\.lmstudio\\models\\ollama-lab\\qwen3.5-4b (no copy, no download).
The script loads it with `lms load` (identifier lab-qwen35-4b, full GPU offload, context 4096), starts LM Studio's
API server on 127.0.0.1:1234 if needed, and unloads the model at the end. It compares against the reference text
that llama-server produced in Step 1 (results/raw/engine-compare-*/data.json, the latest run).
Parts:
  A  load time and VRAM; is the API reachable from the LAN address?
  B  the same raw prompt: LM Studio's API with sampler options as sent, then with repeat_penalty / min_p pinned;
     speed (LM Studio's own stats and client-timed streaming) and byte-identity with llama-server
  C  chat: /v1/chat/completions with the model's own template vs Ollama (23 tokens) and llama-server chatml (19)
  D  4 concurrent requests with --parallel 4
  E  per-request overhead: 1-token requests, median of 20
  F  what LM Studio keeps: api-prediction-history after a request with a marker in the prompt
"""
import json
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

API, OLLAMA = 'http://127.0.0.1:1234', 'http://127.0.0.1:11434'
LMS = Path.home() / '.lmstudio' / 'bin' / 'lms.exe'
HISTORY = Path.home() / '.lmstudio' / '.internal' / 'api-prediction-history'
MODEL_KEY, IDENT = 'qwen3.5-4b', 'lab-qwen35-4b'
QUESTION = 'Explain a Kubernetes liveness probe in two sentences.'
RAW = f'<|im_start|>user\n{QUESTION}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
LLAMA_DEFAULTS = {'presence_penalty': 0.0, 'top_k': 40, 'top_p': 0.95}
OLLAMA_DEFAULTS = {'presence_penalty': 1.5, 'top_k': 20, 'top_p': 0.95}
VARIANTS = {    # what is added on top of llama-server's options; temperature 0, seed 42 everywhere
    'as sent': {},
    'repeat_penalty 1': {'repeat_penalty': 1.0},
    'min_p 0': {'min_p': 0.0},
    'repeat_penalty 1 + min_p 0': {'repeat_penalty': 1.0, 'min_p': 0.0},
}
MARKER = 'LAB09-LMSTUDIO-MARKER'


def lms(*args, check=True):
    r = subprocess.run([str(LMS), *args], capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = (r.stdout + r.stderr).replace('\x1b[?25l', '').replace('\x1b[?25h', '')
    if check and r.returncode:
        raise SystemExit(f'lms {" ".join(args)} failed:\n{out[-1500:]}')
    return out


def vram():
    return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True).stdout)


def lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('192.0.2.1', 9))          # no packet is sent; picks the outbound interface
        return s.getsockname()[0]


def reachable(host, port):
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex((host, port)) == 0


def load(parallel=1):
    lms('unload', '--all', check=False)
    t = time.perf_counter()
    lms('load', MODEL_KEY, '--gpu', 'max', '-c', '4096', '--parallel', str(parallel), '--identifier', IDENT, '-y')
    return round(time.perf_counter() - t, 2)


def complete(prompt, n, samp):
    r = client.post(API + '/api/v0/completions', json={'model': IDENT, 'prompt': prompt, 'max_tokens': n,
                                                     'temperature': 0, 'seed': 42, 'stream': False, **samp}).json()
    return {'text': r['choices'][0]['text'], 'gen_n': r['usage']['completion_tokens'],
            'prompt_n': r['usage']['prompt_tokens'], 'lms_tps': r['stats']['tokens_per_second'],
            'ttft_s': r['stats']['time_to_first_token'], 'runtime': r.get('runtime')}


def stream_rate(prompt, n, samp):
    """Chunks per second between the first and the last streamed chunk, timed here (one chunk ≈ one token)."""
    first = last = None
    chunks, text = 0, ''
    with client.stream('POST', API + '/v1/completions', json={'model': IDENT, 'prompt': prompt, 'max_tokens': n,
                                                             'temperature': 0, 'seed': 42, 'stream': True, **samp}) as r:
        for line in r.iter_lines():
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            piece = json.loads(line[6:])['choices'][0].get('text', '')
            if piece:
                last = time.perf_counter()
                first = first or last
                chunks += 1
                text += piece
    return (chunks - 1) / (last - first) if chunks > 1 else 0.0, text


def med(xs):
    return round(statistics.median(xs), 1)


sys.stdout.reconfigure(encoding='utf-8')
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'lmstudio-compare-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report, data = [], {}
say = lambda text='': (print(text, flush=True), report.append(text))
client = httpx.Client(timeout=600)

# ---------------------------------------------------------------- preflight
if not LMS.exists():
    raise SystemExit(f'{LMS} not found: install LM Studio and start it once (Lab 09 Step 3).')
try:
    if client.get(OLLAMA + '/api/ps').json()['models']:
        raise SystemExit('The Windows Ollama has a model loaded; run `ollama stop qwen3.5:4b` first.')
except httpx.HTTPError:
    pass
refs = sorted(Path('results/raw').glob('engine-compare-*/data.json'))
ref = json.loads(refs[-1].read_text(encoding='utf-8'))['samplers'] if refs else None
if not ref or 'cached_n' not in ref.get('long_prompt', {}).get('ollama', {}):
    raise SystemExit('Run Step 1 (scripts/engine_compare.py) first: its llama-server text is the reference.')
REF = {'llama-defaults': ref['llama-defaults']['llama']['text'], 'ollama-defaults': ref['ollama-defaults']['llama']['text']}
if MODEL_KEY not in lms('ls'):
    raise SystemExit(f'{MODEL_KEY} is not in LM Studio: hard-link it first (Lab 09 Step 3).')
lms('server', 'start', '--port', '1234')
hist = json.loads((Path.home() / '.lmstudio' / '.internal' / 'historical-version-info.json').read_text(encoding='utf-8'))
app = next((f'{t["lastRecordedAppVersion"]}+{t["lastRecordedAppBuildVersion"]}' for t in hist['targetHistories']
            if t['target'] == 'lmstudio'), '?')
idle = vram()
say(f'# lmstudio_compare: LM Studio {app}, '
    f'{MODEL_KEY} hard-linked from the Ollama store, temperature 0, seed 42, {stamp}\n')
say(f'* reference text: llama-server in {refs[-1].parent.name}; GPU idle: {idle} MiB used')

try:
    # ------------------------------------------------------------ A: load and exposure
    say('\n## A. Load and exposure\n')
    load_s = load()
    a = {'load_s': load_s, 'vram_delta_mib': vram() - idle, 'lan_ip': lan_ip()}
    a['reach'] = {f'{h}:{p}': reachable(h, p) for h in ('127.0.0.1', a['lan_ip']) for p in (1234,)}
    internal = json.loads((Path.home() / '.lmstudio' / '.internal' / 'http-server.json').read_text(encoding='utf-8'))
    a['internal_api'] = internal
    a['reach'].update({f'{h}:{internal["port"]}': reachable(h, internal['port']) for h in ('127.0.0.1', a['lan_ip'])})
    first = complete(RAW, 16, LLAMA_DEFAULTS)
    a['runtime'] = first['runtime']
    data['load'] = a
    say(f'* `lms load`: {load_s} s; GPU +{a["vram_delta_mib"]} MiB; runtime {a["runtime"]}')
    say(f'* reachable (TCP connect): {a["reach"]} (the internal API is the one the `lms` CLI uses)')

    # ------------------------------------------------------------ B: sampler variants
    say('\n## B. The same raw prompt (llama-server\'s options + the extra options shown; median of 5)\n')
    b = {}
    for name, extra in VARIANTS.items():
        samp = {**LLAMA_DEFAULTS, **extra}
        complete(RAW, 256, samp)
        runs = [complete(RAW, 256, samp) for _ in range(5)]
        streams = [stream_rate(RAW, 256, samp) for _ in range(3)]
        b[name] = {'lms_tps': med(r['lms_tps'] for r in runs), 'client_tps': med(s[0] for s in streams),
                   'gen_n': runs[0]['gen_n'], 'prompt_n': runs[0]['prompt_n'], 'text': runs[0]['text'],
                   'stable': len({r['text'] for r in runs}) == 1, 'stream_text_same': streams[0][1] == runs[0]['text'],
                   'same_as_llama_server': runs[0]['text'] == REF['llama-defaults']}
    for name, extra in (('Ollama\'s options as sent', {}), ('Ollama\'s options + repeat_penalty 1 + min_p 0',
                                                             {'repeat_penalty': 1.0, 'min_p': 0.0})):
        samp = {**OLLAMA_DEFAULTS, **extra}
        complete(RAW, 256, samp)
        runs = [complete(RAW, 256, samp) for _ in range(5)]
        b[name] = {'lms_tps': med(r['lms_tps'] for r in runs), 'client_tps': None, 'gen_n': runs[0]['gen_n'],
                   'prompt_n': runs[0]['prompt_n'], 'text': runs[0]['text'],
                   'stable': len({r['text'] for r in runs}) == 1,
                   'same_as_llama_server': runs[0]['text'] == REF['ollama-defaults']}
    data['samplers'] = b
    say('| options | LM Studio stats tok/s | client-timed tok/s (stream) | tokens | byte-identical to llama-server |')
    say('|---|---:|---:|---:|---|')
    for name, x in b.items():
        say(f'| {name} | {x["lms_tps"]} | {x["client_tps"] if x["client_tps"] is not None else "–"} | {x["gen_n"]} | '
            f'{x["same_as_llama_server"]} |')
    say(f'* prompt tokens: {sorted({x["prompt_n"] for x in b.values()})} (raw prompt passed unchanged if 23)')

    # ------------------------------------------------------------ C: chat template
    say('\n## C. Chat: /v1/chat/completions with the model\'s own template\n')
    r = client.post(API + '/v1/chat/completions', json={'model': IDENT, 'messages': [{'role': 'user', 'content': QUESTION}],
                                                       'max_tokens': 256, 'temperature': 0, 'seed': 42,
                                                       **OLLAMA_DEFAULTS, 'repeat_penalty': 1.0, 'min_p': 0.0}).json()
    msg = r['choices'][0]['message']
    c = {'prompt_tokens': r['usage']['prompt_tokens'], 'completion_tokens': r['usage']['completion_tokens'],
         'content': msg.get('content', ''), 'reasoning': msg.get('reasoning_content') or msg.get('reasoning'),
         'same_as_ollama_raw': (msg.get('content') or '').strip() == REF['ollama-defaults'].strip()}
    data['chat'] = c
    say(f'* prompt tokens {c["prompt_tokens"]} (Ollama /api/chat 23, llama-server chatml 19); completion tokens '
        f'{c["completion_tokens"]}; separate reasoning field: {bool(c["reasoning"])} '
        f'({(c["reasoning"] or "")[:60]!r})')
    say(f'* content: {c["content"][:110]!r}; same as the raw-prompt answer: {c["same_as_ollama_raw"]}')

    # ------------------------------------------------------------ E: per-request overhead (parallel 1)
    samp = {**LLAMA_DEFAULTS, 'repeat_penalty': 1.0, 'min_p': 0.0}
    complete('Hi', 1, samp)
    e = []
    for _ in range(20):
        t = time.perf_counter()
        complete('Hi', 1, samp)
        e.append((time.perf_counter() - t) * 1000)
    data['overhead_ms'] = [round(x, 1) for x in e]

    # ------------------------------------------------------------ F: what LM Studio keeps
    before = {p for p in HISTORY.rglob('*') if p.is_file()} if HISTORY.exists() else set()
    complete(f'{MARKER}: {QUESTION}', 32, samp)
    client.post(API + '/v1/chat/completions', json={'model': IDENT, 'max_tokens': 16, 'temperature': 0,
                                                   'messages': [{'role': 'user', 'content': f'{MARKER} chat'}]})
    time.sleep(3)
    after = {p for p in HISTORY.rglob('*') if p.is_file()} if HISTORY.exists() else set()
    hits = []
    for p in Path.home().joinpath('.lmstudio').rglob('*'):
        if p.is_file() and p.stat().st_size < 20_000_000 and p.suffix not in ('.gguf', '.dll', '.exe', '.node'):
            try:
                if MARKER in p.read_text(encoding='utf-8', errors='ignore'):
                    hits.append(str(p.relative_to(Path.home())))
            except OSError:
                pass
    f = {'new_history_files': sorted(str(p.relative_to(Path.home())) for p in after - before), 'marker_found_in': hits}
    data['retention'] = f

    # ------------------------------------------------------------ D: --parallel 4
    say('\n## D. 4 concurrent requests, --parallel 4 (llama-server\'s options + repeat_penalty 1 + min_p 0, 128 tokens)\n')
    load(parallel=4)
    complete(RAW, 128, samp)
    res, t0 = [None] * 4, time.perf_counter()

    def one(i):
        res[i] = complete(RAW, 128, samp)
    th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.perf_counter() - t0
    d = {'wall_s': round(wall, 2), 'aggregate_tps': round(sum(x['gen_n'] for x in res) / wall, 1),
         'per_request_lms_tps': [round(x['lms_tps'], 1) for x in res], 'vram_delta_mib': vram() - idle,
         'same_text': len({x['text'] for x in res}) == 1}
    data['parallel4'] = d
    say(f'* all 4 in {d["wall_s"]} s, aggregate {d["aggregate_tps"]} tok/s, per request (LM Studio stats) '
        f'{d["per_request_lms_tps"]}, GPU +{d["vram_delta_mib"]} MiB, same text {d["same_text"]}')

    say('\n## E. Per-request overhead (1-token requests, median of 20)\n')
    say(f'* LM Studio /api/v0/completions: {med(e)} ms (Step 1: llama-server 8.4 ms, Ollama 22.8 ms)')
    say('\n## F. What LM Studio keeps\n')
    say(f'* new files under ~/.lmstudio/.internal/api-prediction-history: {f["new_history_files"][:6]}')
    say(f'* files under ~/.lmstudio containing the prompt marker: {f["marker_found_in"][:6]}')
finally:
    lms('unload', '--all', check=False)
    say(f'\n* cleanup: model unloaded (LM Studio and its server left running); GPU {vram()} MiB (idle at start {idle})')
    (outdir / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
    (outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'\nSaved {outdir}')
