"""Lab 09 Step 1. Ollama vs its own engine: the llama-server.exe that Ollama 0.40.13 bundles, run directly.

Run with the venv's interpreter from the repo root, the Windows Ollama running with nothing loaded (≈8 min):
    .\\.venv\\Scripts\\python scripts\\engine_compare.py

Ollama 0.40.13 on Windows starts %LOCALAPPDATA%\\Programs\\Ollama\\lib\\ollama\\llama-server.exe (llama.cpp commit
39febd2 = release b11963) for every model. This script starts that same binary itself, on 127.0.0.1:11440, with the
flags Ollama uses (server.log) and the same GGUF blobs from the Ollama store, so the two differ only by what Ollama
adds: its API, templates, scheduler and default options. Nothing is downloaded. Only one engine holds the model at
a time. The llama-server process is started and stopped by this script (its own process handle only).
Parts:
  A  cold start: llama-server process start → healthy, vs an Ollama load; VRAM of each
  B  the same raw prompt, sampler settings × engine: speed and whether the text is byte-identical
  C  chat: Ollama /api/chat vs llama-server /v1/chat/completions (chatml template): prompt tokens and text
  D  concurrency: 4 requests at once, llama-server -np 1 / -np 2 / -np 4 vs Ollama
  E  per-request overhead: 1-token requests, median of 20
"""
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx

OLLAMA, LLAMA = 'http://127.0.0.1:11434', 'http://127.0.0.1:11440'
MODEL = 'qwen3.5:4b'
LIB = Path(os.environ['LOCALAPPDATA']) / 'Programs' / 'Ollama' / 'lib' / 'ollama'
BLOBS = Path.home() / '.ollama' / 'models' / 'blobs'
GGUF = BLOBS / 'sha256-a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50'
MMPROJ = BLOBS / 'sha256-e56a899b1540860385465756dd308c84417410b94dbc1c380adffd20a921ca60'
QUESTION = 'Explain a Kubernetes liveness probe in two sentences.'
RAW = f'<|im_start|>user\n{QUESTION}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
SAMPLERS = {        # temperature 0 throughout; Ollama's defaults come from the model's params layer
    'ollama-defaults': {'presence_penalty': 1.5, 'top_k': 20, 'top_p': 0.95},
    'llama-defaults': {'presence_penalty': 0.0, 'top_k': 40, 'top_p': 0.95},
    'penalty-only': {'presence_penalty': 1.5, 'top_k': 40, 'top_p': 0.95},
    'top_k-20-only': {'presence_penalty': 0.0, 'top_k': 20, 'top_p': 0.95},
}


def vram():
    return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True).stdout)


def long_prompt(nonce):
    """A unique first line per call: Ollama keeps a prompt cache between requests, so a repeated prompt is mostly
    not evaluated again (runs 1-2 measured ≈39K "tok/s" that way)."""
    rows = '\n'.join(f'pod-{i:03d}  {"Running" if i % 7 else "CrashLoopBackOff"}  restarts={i % 5}' for i in range(120))
    return f'<|im_start|>user\nRequest {nonce}. How many pods are in CrashLoopBackOff?\n{rows}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class LlamaServer:
    """The bundled llama-server.exe with Ollama's flags (minus the ones only Ollama's runner protocol needs)."""
    def __init__(self, np_=1, ctx=4096):
        env = dict(os.environ, PATH=f'{LIB};{LIB / "cuda_v13"};{os.environ["PATH"]}',
                   GGML_BACKEND_PATH=str(LIB / 'cuda_v13' / 'ggml-cuda.dll'))    # what Ollama sets (llm/llama_server.go)
        self.log = outdir / f'llama-server-np{np_}.log'
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [str(LIB / 'llama-server.exe'), '--model', str(GGUF), '--mmproj', str(MMPROJ), '--host', '127.0.0.1',
             '--port', '11440', '--no-webui', '--offline', '-c', str(ctx * np_), '-np', str(np_), '--no-jinja',
             '--chat-template', 'chatml', '--flash-attn', 'auto', '-b', '512', '-ub', '512'],
            env=env, stdout=subprocess.DEVNULL, stderr=open(self.log, 'w', encoding='utf-8'))
        for _ in range(240):
            try:
                if client.get(LLAMA + '/health', timeout=2).json().get('status') == 'ok':
                    self.ready_s = round(time.perf_counter() - self.t0, 2)
                    return
            except (httpx.HTTPError, ValueError):
                pass
            if self.proc.poll() is not None:
                raise SystemExit(f'llama-server exited:\n{self.log.read_text(encoding="utf-8")[-2000:]}')
            time.sleep(0.25)
        raise SystemExit('llama-server never became healthy')

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=20)
        time.sleep(1)


def gen_llama(prompt, n, samp, cache=False):
    r = client.post(LLAMA + '/completion', json={'prompt': prompt, 'n_predict': n, 'temperature': 0, 'seed': 42,
                                                'cache_prompt': cache, **samp}).json()
    t = r['timings']
    return {'gen_tps': t['predicted_per_second'], 'prompt_tps': t['prompt_per_second'], 'prompt_n': t['prompt_n'],
            'gen_n': t['predicted_n'], 'text': r['content']}


def gen_ollama(prompt, n, samp):
    r = client.post(OLLAMA + '/api/generate', json={'model': MODEL, 'prompt': prompt, 'raw': True, 'stream': False,
                                                   'options': {'temperature': 0, 'seed': 42, 'num_predict': n, **samp}}).json()
    return {'gen_tps': r['eval_count'] / r['eval_duration'] * 1e9,
            'prompt_tps': r['prompt_eval_count'] / r['prompt_eval_duration'] * 1e9,
            'prompt_n': r['prompt_eval_count'], 'cached_n': r.get('prompt_eval_cached_count', 0),
            'gen_n': r['eval_count'], 'text': r['response'],
            'load_s': r['load_duration'] / 1e9}


def ollama_unload():
    client.post(OLLAMA + '/api/generate', json={'model': MODEL, 'keep_alive': 0})
    for _ in range(60):
        if not client.get(OLLAMA + '/api/ps').json()['models']:
            time.sleep(1)
            return
        time.sleep(0.5)


def med(xs):
    return round(statistics.median(xs), 1)


def concurrent(fn, count=4):
    res, t0 = [None] * count, time.perf_counter()

    def one(i):
        t = time.perf_counter()
        res[i] = {**fn(), 'wall_s': round(time.perf_counter() - t, 2)}
    th = [threading.Thread(target=one, args=(i,)) for i in range(count)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    total = time.perf_counter() - t0
    return {'wall_s': round(total, 2), 'aggregate_tps': round(sum(r['gen_n'] for r in res) / total, 1),
            'per_request_wall_s': [r['wall_s'] for r in res], 'per_request_gen_tps': [round(r['gen_tps'], 1) for r in res],
            'texts_identical': len({r['text'] for r in res}) == 1}


sys.stdout.reconfigure(encoding='utf-8')
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'engine-compare-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report, data = [], {}
say = lambda text='': (print(text, flush=True), report.append(text))
client = httpx.Client(timeout=900)        # one client: a new one costs ≈0.3 s (Lab 08 Step 4)

# ---------------------------------------------------------------- preflight
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1', 11440)) == 0:
        raise SystemExit('127.0.0.1:11440 is already in use (a leftover llama-server?)')
version = client.get(OLLAMA + '/api/version').json()['version']
if client.get(OLLAMA + '/api/ps').json()['models']:
    raise SystemExit('The Windows Ollama has a model loaded; run `ollama stop qwen3.5:4b` first.')
others = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout.split()
if any(n.startswith(('ollama-lab', 'k3d-ollama-lab')) for n in others):
    raise SystemExit(f'Lab containers are running ({others}); another experiment may be using the GPU.')
for f in (GGUF, MMPROJ, LIB / 'llama-server.exe'):
    if not f.exists():
        raise SystemExit(f'{f} not found')
ver = subprocess.run([str(LIB / 'llama-server.exe'), '--version'], capture_output=True, text=True, cwd=LIB,
                     env=dict(os.environ, PATH=f'{LIB};{os.environ["PATH"]}'))
idle = vram()
say(f'# engine_compare: Ollama {version} vs its bundled llama-server ({(ver.stdout + ver.stderr).strip().splitlines()[0]}), '
    f'{MODEL} GGUF from the Ollama store, temperature 0, seed 42, {stamp}\n')
say(f'* GPU idle: {idle} MiB used')

try:
    # ------------------------------------------------------------ A: cold start
    say('\n## A. Cold start and memory\n')
    a = {}
    t = time.perf_counter()
    o = gen_ollama(RAW, 16, SAMPLERS['ollama-defaults'])
    a['ollama'] = {'first_request_s': round(time.perf_counter() - t, 2), 'load_s': round(o['load_s'], 2), 'vram_mib': vram()}
    ollama_unload()
    ls = LlamaServer()
    a['llama'] = {'process_to_healthy_s': ls.ready_s, 'vram_mib': vram()}
    t = time.perf_counter()
    gen_llama(RAW, 16, SAMPLERS['llama-defaults'])
    a['llama']['first_request_s'] = round(time.perf_counter() - t, 2)
    data['cold'] = a
    say(f'* Ollama: first request {a["ollama"]["first_request_s"]} s (load {a["ollama"]["load_s"]} s); '
        f'GPU {a["ollama"]["vram_mib"]} MiB (+{a["ollama"]["vram_mib"] - idle})')
    say(f'* llama-server: process start → healthy {a["llama"]["process_to_healthy_s"]} s, then first request '
        f'{a["llama"]["first_request_s"]} s; GPU {a["llama"]["vram_mib"]} MiB (+{a["llama"]["vram_mib"] - idle})')

    # ------------------------------------------------------------ B: sampler settings × engine
    say('\n## B. The same raw prompt: sampler settings × engine (median of 5, after 1 warm-up)\n')
    b = {}
    for name, samp in SAMPLERS.items():
        gen_llama(RAW, 256, samp)
        runs = [gen_llama(RAW, 256, samp) for _ in range(5)]
        b[name] = {'llama': {'gen_tps': med(r['gen_tps'] for r in runs), 'gen_n': runs[0]['gen_n'], 'prompt_n': runs[0]['prompt_n'],
                             'text': runs[0]['text'], 'stable': len({r['text'] for r in runs}) == 1}}
    longs = [gen_llama(long_prompt(f'L{i}'), 32, SAMPLERS['llama-defaults']) for i in range(4)][1:]
    b['long_prompt'] = {'llama': {'prompt_n': longs[0]['prompt_n'], 'prompt_tps': med(r['prompt_tps'] for r in longs),
                                  'text': longs[0]['text']}}
    ls.stop()

    # ------------------------------------------------------------ D (llama-server side, while we restart it anyway)
    d = {}
    for np_ in (1, 2, 4):
        srv = LlamaServer(np_=np_)
        gen_llama(RAW, 128, SAMPLERS['llama-defaults'])
        d[f'llama-np{np_}'] = {**concurrent(lambda: gen_llama(RAW, 128, SAMPLERS['llama-defaults'])), 'vram_mib': vram()}
        if np_ == 1:          # E (llama-server side): 1-token requests
            gen_llama('Hi', 1, SAMPLERS['llama-defaults'], cache=True)
            e_llama = []
            for _ in range(20):
                t = time.perf_counter()
                gen_llama('Hi', 1, SAMPLERS['llama-defaults'], cache=True)
                e_llama.append((time.perf_counter() - t) * 1000)
            # C (llama-server side): chat completions with the chatml template
            r = client.post(LLAMA + '/v1/chat/completions', json={
                'messages': [{'role': 'user', 'content': QUESTION}], 'max_tokens': 256, 'temperature': 0,
                'seed': 42, **SAMPLERS['ollama-defaults']}).json()
            c_llama = {'prompt_tokens': r['usage']['prompt_tokens'], 'completion_tokens': r['usage']['completion_tokens'],
                       'text': r['choices'][0]['message']['content']}
        srv.stop()

    # ------------------------------------------------------------ Ollama side of B, C, D, E
    for name, samp in SAMPLERS.items():
        gen_ollama(RAW, 256, samp)
        runs = [gen_ollama(RAW, 256, samp) for _ in range(5)]
        b[name]['ollama'] = {'gen_tps': med(r['gen_tps'] for r in runs), 'gen_n': runs[0]['gen_n'], 'prompt_n': runs[0]['prompt_n'],
                             'text': runs[0]['text'], 'stable': len({r['text'] for r in runs}) == 1}
    longs = [gen_ollama(long_prompt(f'O{i}'), 32, SAMPLERS['llama-defaults']) for i in range(4)][1:]
    b['long_prompt']['ollama'] = {'prompt_n': longs[0]['prompt_n'], 'prompt_tps': med(r['prompt_tps'] for r in longs),
                                  'cached_n': [r['cached_n'] for r in longs],
                                  'text': longs[0]['text']}
    data['samplers'] = b
    say('| sampler settings | llama-server tok/s | Ollama tok/s | tokens (llama / Ollama) | byte-identical text |')
    say('|---|---:|---:|---|---|')
    for name in SAMPLERS:
        x = b[name]
        say(f'| {name} `{SAMPLERS[name]}` | {x["llama"]["gen_tps"]} | {x["ollama"]["gen_tps"]} | '
            f'{x["llama"]["gen_n"]} / {x["ollama"]["gen_n"]} | {x["llama"]["text"] == x["ollama"]["text"]} |')
    lpx = b['long_prompt']
    say(f'* long prompt ({lpx["llama"]["prompt_n"]} / {lpx["ollama"]["prompt_n"]} tokens): prompt eval '
        f'{lpx["llama"]["prompt_tps"]} vs {lpx["ollama"]["prompt_tps"]} tok/s (Ollama cached tokens '
        f'{lpx["ollama"]["cached_n"]}); same answer: '
        f'{lpx["llama"]["text"] == lpx["ollama"]["text"]} ({lpx["ollama"]["text"].strip()[:60]!r})')

    say('\n## C. Chat: Ollama /api/chat vs llama-server /v1/chat/completions (Ollama\'s sampler settings)\n')
    r = client.post(OLLAMA + '/api/chat', json={'model': MODEL, 'messages': [{'role': 'user', 'content': QUESTION}],
                                               'stream': False, 'think': False,
                                               'options': {'temperature': 0, 'seed': 42, 'num_predict': 256}}).json()
    c_ollama = {'prompt_tokens': r['prompt_eval_count'], 'completion_tokens': r['eval_count'], 'text': r['message']['content']}
    data['chat'] = {'llama': c_llama, 'ollama': c_ollama}
    say(f'* prompt tokens: llama-server {c_llama["prompt_tokens"]}, Ollama {c_ollama["prompt_tokens"]} '
        f'(the raw prompt in B: {b["ollama-defaults"]["ollama"].get("prompt_n", "?")})')
    say(f'* completion tokens {c_llama["completion_tokens"]} / {c_ollama["completion_tokens"]}; '
        f'byte-identical: {c_llama["text"] == c_ollama["text"]}')
    say(f'  * llama-server: {c_llama["text"][:110]!r}')
    say(f'  * Ollama:       {c_ollama["text"][:110]!r}')

    gen_ollama(RAW, 128, SAMPLERS['llama-defaults'])
    d['ollama'] = {**concurrent(lambda: gen_ollama(RAW, 128, SAMPLERS['llama-defaults'])), 'vram_mib': vram()}
    data['concurrency'] = d
    say('\n## D. 4 concurrent requests (128 tokens max, llama-server\'s sampler defaults)\n')
    say('| engine | wall for all 4 | aggregate tok/s | per-request tok/s | per-request wall | GPU MiB | same text |')
    say('|---|---:|---:|---|---|---:|---|')
    for k, v in d.items():
        say(f'| {k} | {v["wall_s"]} | {v["aggregate_tps"]} | {v["per_request_gen_tps"]} | {v["per_request_wall_s"]} | '
            f'{v["vram_mib"]} | {v["texts_identical"]} |')

    gen_ollama('Hi', 1, SAMPLERS['llama-defaults'])
    e_ollama = []
    for _ in range(20):
        t = time.perf_counter()
        gen_ollama('Hi', 1, SAMPLERS['llama-defaults'])
        e_ollama.append((time.perf_counter() - t) * 1000)
    data['overhead_ms'] = {'llama': [round(x, 1) for x in e_llama], 'ollama': [round(x, 1) for x in e_ollama]}
    say('\n## E. Per-request overhead (1-token requests, median of 20)\n')
    say(f'* llama-server /completion: {med(e_llama)} ms; Ollama /api/generate (raw): {med(e_ollama)} ms; '
        f'difference {round(med(e_ollama) - med(e_llama), 1)} ms')
finally:
    for p in [x for x in globals().values() if isinstance(x, LlamaServer)]:
        if p.proc.poll() is None:
            p.stop()
    try:
        ollama_unload()
    except httpx.HTTPError:
        pass
    say(f'\n* cleanup: llama-server stopped, Ollama unloaded; GPU {vram()} MiB (idle at start {idle})')
    (outdir / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
    (outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'\nSaved {outdir}')
