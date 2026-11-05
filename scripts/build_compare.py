"""Lab 09 Step 2. Two builds of the same llama.cpp commit: Ollama's bundled llama-server vs the official b11963.

Run with the venv's interpreter from the repo root, the Windows Ollama running with nothing loaded (≈8 min):
    .\\.venv\\Scripts\\python scripts\\build_compare.py

Both are commit 39febd2 (b11963). Differences found before the run: compiler (Clang 18.1.8 vs 20.1.8), CUDA runtime
(13.0 vs 13.4: cudart 6.14.11.13000 vs .13040, cuBLAS 13.0 vs 13.7), ggml-cuda.dll 141 vs 155 MB. The official
build is in tools\\llama.cpp-b11963 (gitignored; downloaded 2026-11-05, sha256 checked against the GitHub release).
Each build is started 2 times, alternating (bundled, official, bundled, official), with the same flags, the same
GGUF blobs from the Ollama store, on 127.0.0.1:11440; then once each with -np 4; then llama-bench (official only).
Parts:
  A  start: process → healthy, VRAM
  B  generation (2 sampler settings, median of 5) and prompt processing (unique 1,640-token prompts, median of 3);
     byte-identical text between builds
  C  4 concurrent requests with -np 4
  D  llama-bench (official build): pp512 / tg128, the number llama.cpp's own tooling reports
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

LLAMA, OLLAMA = 'http://127.0.0.1:11440', 'http://127.0.0.1:11434'
BUNDLED = Path(os.environ['LOCALAPPDATA']) / 'Programs' / 'Ollama' / 'lib' / 'ollama'
OFFICIAL = Path('tools/llama.cpp-b11963').resolve()
BLOBS = Path.home() / '.ollama' / 'models' / 'blobs'
GGUF = BLOBS / 'sha256-a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50'
MMPROJ = BLOBS / 'sha256-e56a899b1540860385465756dd308c84417410b94dbc1c380adffd20a921ca60'
RAW = ('<|im_start|>user\nExplain a Kubernetes liveness probe in two sentences.<|im_end|>\n'
       '<|im_start|>assistant\n<think>\n\n</think>\n\n')
SAMPLERS = {'ollama-defaults': {'presence_penalty': 1.5, 'top_k': 20, 'top_p': 0.95},
            'llama-defaults': {'presence_penalty': 0.0, 'top_k': 40, 'top_p': 0.95}}


def vram():
    return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True).stdout)


def long_prompt(nonce):
    rows = '\n'.join(f'pod-{i:03d}  {"Running" if i % 7 else "CrashLoopBackOff"}  restarts={i % 5}' for i in range(120))
    return (f'<|im_start|>user\nRequest {nonce}. How many pods are in CrashLoopBackOff?\n{rows}<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def build_env(build):
    if build == 'bundled':      # what Ollama sets (llm/llama_server.go); without it the binary runs on the CPU
        return dict(os.environ, PATH=f'{BUNDLED};{BUNDLED / "cuda_v13"};{os.environ["PATH"]}',
                    GGML_BACKEND_PATH=str(BUNDLED / 'cuda_v13' / 'ggml-cuda.dll'))
    return dict(os.environ, PATH=f'{OFFICIAL};{os.environ["PATH"]}')      # self-contained: backends next to the exe


def exe(build, name='llama-server.exe'):
    return (BUNDLED if build == 'bundled' else OFFICIAL) / name


class Server:
    def __init__(self, build, tag, np_=1):
        self.log = outdir / f'llama-server-{tag}.log'
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [str(exe(build)), '--model', str(GGUF), '--mmproj', str(MMPROJ), '--host', '127.0.0.1', '--port', '11440',
             '--no-webui', '--offline', '-c', str(4096 * np_), '-np', str(np_), '--no-jinja', '--chat-template', 'chatml',
             '--flash-attn', 'auto', '-b', '512', '-ub', '512'],
            env=build_env(build), stdout=subprocess.DEVNULL, stderr=open(self.log, 'w', encoding='utf-8'))
        running.append(self)
        for _ in range(480):
            try:
                if client.get(LLAMA + '/health', timeout=2).json().get('status') == 'ok':
                    self.ready_s = round(time.perf_counter() - t0, 2)
                    self.vram = vram()
                    return
            except (httpx.HTTPError, ValueError):
                pass
            if self.proc.poll() is not None:
                raise SystemExit(f'{build} llama-server exited:\n{self.log.read_text(encoding="utf-8")[-2000:]}')
            time.sleep(0.25)
        raise SystemExit(f'{build} llama-server never became healthy')

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=20)
        running.remove(self)
        time.sleep(1.5)


def gen(prompt, n, samp):
    r = client.post(LLAMA + '/completion', json={'prompt': prompt, 'n_predict': n, 'temperature': 0, 'seed': 42,
                                                'cache_prompt': False, **samp}).json()
    t = r['timings']
    return {'gen_tps': t['predicted_per_second'], 'prompt_tps': t['prompt_per_second'], 'gen_n': t['predicted_n'],
            'prompt_n': t['prompt_n'], 'text': r['content']}


def med(xs):
    return round(statistics.median(xs), 1)


sys.stdout.reconfigure(encoding='utf-8')
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'build-compare-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report, data, running = [], {}, []
say = lambda text='': (print(text, flush=True), report.append(text))
client = httpx.Client(timeout=900)

# ---------------------------------------------------------------- preflight
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1', 11440)) == 0:
        raise SystemExit('127.0.0.1:11440 is already in use (a leftover llama-server?)')
try:
    if client.get(OLLAMA + '/api/ps').json()['models']:
        raise SystemExit('The Windows Ollama has a model loaded; run `ollama stop qwen3.5:4b` first.')
except httpx.HTTPError:
    pass
docker_ps = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout.split()
if any(n.startswith(('ollama-lab', 'k3d-ollama-lab')) for n in docker_ps):
    raise SystemExit(f'Lab containers are running ({docker_ps}); another experiment may be using the GPU.')
for f in (GGUF, MMPROJ, exe('bundled'), exe('official'), exe('official', 'llama-bench.exe')):
    if not f.exists():
        raise SystemExit(f'{f} not found (official build: see the Lab 09 README, Step 2)')
versions = {}
for b in ('bundled', 'official'):
    v = subprocess.run([str(exe(b)), '--version'], capture_output=True, text=True, env=build_env(b))
    versions[b] = ' / '.join((v.stdout + v.stderr).strip().splitlines()[:2])
data['versions'] = versions
idle = vram()
say(f'# build_compare: qwen3.5:4b GGUF, temperature 0, seed 42, {stamp}\n')
for b, v in versions.items():
    say(f'* {b}: {v}')
say(f'* GPU idle: {idle} MiB used')

try:
    # ------------------------------------------------------------ A + B: alternating starts
    runs = []
    for rnd in (1, 2):
        for build in ('bundled', 'official'):
            srv = Server(build, f'{build}-{rnd}')
            r = {'build': build, 'round': rnd, 'ready_s': srv.ready_s, 'vram_delta_mib': srv.vram - idle}
            for name, samp in SAMPLERS.items():
                gen(RAW, 256, samp)
                xs = [gen(RAW, 256, samp) for _ in range(5)]
                r[name] = {'gen_tps': med(x['gen_tps'] for x in xs), 'gen_n': xs[0]['gen_n'], 'text': xs[0]['text'],
                           'stable': len({x['text'] for x in xs}) == 1}
            ps = [gen(long_prompt(f'{build}{rnd}-{i}'), 16, SAMPLERS['llama-defaults']) for i in range(4)][1:]
            r['prompt'] = {'prompt_n': ps[0]['prompt_n'], 'prompt_tps': med(x['prompt_tps'] for x in ps)}
            srv.stop()
            runs.append(r)
    data['runs'] = runs
    say('\n## A + B. Alternating starts (each row: one server process)\n')
    say('| build | round | process → healthy | VRAM Δ MiB | gen tok/s, Ollama\'s options | gen tok/s, llama-server\'s | '
        'prompt tok/s (1,640 tokens) |')
    say('|---|---:|---:|---:|---:|---:|---:|')
    for r in runs:
        say(f'| {r["build"]} | {r["round"]} | {r["ready_s"]} s | {r["vram_delta_mib"]} | {r["ollama-defaults"]["gen_tps"]} | '
            f'{r["llama-defaults"]["gen_tps"]} | {r["prompt"]["prompt_tps"]} |')
    for name in SAMPLERS:
        texts = {(r['build'], r['round']): r[name]['text'] for r in runs}
        say(f'* {name}: byte-identical across all 4 processes: {len(set(texts.values())) == 1}; '
            f'tokens {sorted({r[name]["gen_n"] for r in runs})}')

    # ------------------------------------------------------------ C: -np 4
    say('\n## C. 4 concurrent requests, -np 4 (llama-server\'s options, up to 128 tokens)\n')
    c = {}
    for build in ('bundled', 'official'):
        srv = Server(build, f'{build}-np4', np_=4)
        gen(RAW, 128, SAMPLERS['llama-defaults'])
        res, t0 = [None] * 4, time.perf_counter()

        def one(i):
            res[i] = gen(RAW, 128, SAMPLERS['llama-defaults'])
        th = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for x in th:
            x.start()
        for x in th:
            x.join()
        wall = time.perf_counter() - t0
        c[build] = {'wall_s': round(wall, 2), 'aggregate_tps': round(sum(x['gen_n'] for x in res) / wall, 1),
                    'per_request_tps': [round(x['gen_tps'], 1) for x in res], 'vram_delta_mib': vram() - idle,
                    'same_text': len({x['text'] for x in res}) == 1}
        srv.stop()
        say(f'* {build}: all 4 in {c[build]["wall_s"]} s, aggregate {c[build]["aggregate_tps"]} tok/s, per request '
            f'{c[build]["per_request_tps"]}, VRAM Δ {c[build]["vram_delta_mib"]} MiB, same text {c[build]["same_text"]}')
    data['np4'] = c

    # ------------------------------------------------------------ D: llama-bench
    say('\n## D. llama-bench (official build; Ollama ships no llama-bench)\n')
    t0 = time.perf_counter()
    lb = subprocess.run([str(exe('official', 'llama-bench.exe')), '-m', str(GGUF), '-p', '512', '-n', '128', '-r', '5',
                         '-fa', 'auto', '-o', 'json'], capture_output=True, text=True, env=build_env('official'))
    (outdir / 'llama-bench.stderr.txt').write_text(lb.stderr, encoding='utf-8')
    try:
        bench = json.loads(lb.stdout)
    except ValueError:
        bench = []
        say(f'* llama-bench output not JSON (exit {lb.returncode}); see llama-bench.stderr.txt')
    data['llama_bench'] = bench
    for row in bench:
        kind = f'pp{row["n_prompt"]}' if row.get('n_prompt') else f'tg{row["n_gen"]}'
        say(f'* {kind}: {round(row["avg_ts"], 1)} ± {round(row["stddev_ts"], 1)} tok/s '
            f'(flash attention {row.get("flash_attn")}, {row.get("n_gpu_layers")} GPU layers, {row.get("backends")})')
    say(f'* llama-bench took {round(time.perf_counter() - t0, 1)} s')
finally:
    for srv in list(running):
        srv.stop()
    say(f'\n* cleanup: llama-server processes stopped; GPU {vram()} MiB (idle at start {idle})')
    (outdir / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
    (outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'\nSaved {outdir}')
