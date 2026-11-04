"""Lab 08 Step 1. Ollama in a container on Docker Desktop (WSL2 backend) with the GPU, next to the Windows install.

Run with the venv's interpreter, Docker Desktop running:
    .\\.venv\\Scripts\\python scripts\\container_ollama.py              (≈5-8 min)
    .\\.venv\\Scripts\\python scripts\\container_ollama.py --contend    (also loads the 4b in BOTH instances: an
                                                                        announced overflow past the 8,500 MiB ceiling)

Three backends, same Ollama version (0.40.13), same model (qwen3.5:4b), same prompts, one at a time on the GPU:
  windows   the native install on 127.0.0.1:11434
  bind      ollama/ollama:0.40.13, the Windows model store bind-mounted READ-ONLY, on 127.0.0.1:11435
  volume    ollama/ollama:0.40.13, the 4b copied (locally, no download) into a Docker named volume, on 127.0.0.1:11436
Parts:
  A  exposure: what each published port is bound to
  B  load: reload time (unload with keep_alive 0, then the same request) x3 per backend; container start and restart
  C  speed: warm generation and prompt-eval rates, and whether the three backends give the same text
  D  two Ollama instances on one GPU: Windows holds the 4b while the container loads a small embedding model
Everything this script creates carries the label ollama-lab=step1 and is removed at the end, except the volume.
It never touches other containers, images or volumes.
"""
import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx

IMAGE = 'ollama/ollama:0.40.13'
MODEL, SMALL = 'qwen3.5:4b', 'embeddinggemma:300m'
STORE = Path.home() / '.ollama' / 'models'
LABEL = 'ollama-lab=step1'
VOLUME = 'ollama-lab-models'
PORTS = {'windows': 11434, 'bind': 11435, 'volume': 11436}
SHORT = 'Explain a Kubernetes liveness probe in two sentences.'
LOG_LINE = '2026-10-28T07:01:{:02d}Z INFO payments-api request id={} path=/v1/charge status=200 latency_ms={}\n'


def docker(*args, check=True):
    r = subprocess.run(['docker', *args], capture_output=True, text=True, encoding='utf-8')
    if check and r.returncode:
        raise SystemExit(f'docker {" ".join(args)} failed: {r.stderr.strip()}')
    return r.stdout.strip()


def vram():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return int(out.strip())


def url(backend, path):
    return f'http://127.0.0.1:{PORTS[backend]}{path}'


def wait_up(backend, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        try:
            httpx.get(url(backend, '/api/version'), timeout=2)
            return round(time.time() - t, 2)
        except httpx.HTTPError:
            time.sleep(0.25)
    raise SystemExit(f'{backend} did not answer on port {PORTS[backend]} within {timeout} s')


def generate(backend, prompt, num_predict=128):
    t = time.time()
    r = httpx.post(url(backend, '/api/generate'), timeout=900, json={
        'model': MODEL, 'prompt': prompt, 'stream': False, 'think': False,
        'options': {'temperature': 0, 'seed': 42, 'num_predict': num_predict}}).json()
    return {'wall_s': round(time.time() - t, 2), 'load_s': round(r['load_duration'] / 1e9, 2),
            'prompt_tok': r['prompt_eval_count'], 'gen_tok': r['eval_count'],
            'prompt_tps': round(r['prompt_eval_count'] / (r['prompt_eval_duration'] / 1e9), 1),
            'gen_tps': round(r['eval_count'] / (r['eval_duration'] / 1e9), 1), 'text': r['response']}


def unload(backend, model=MODEL):
    httpx.post(url(backend, '/api/generate'), json={'model': model, 'keep_alive': 0}, timeout=60)
    for _ in range(40):
        if not httpx.get(url(backend, '/api/ps'), timeout=5).json()['models']:
            return
        time.sleep(0.25)


def loaded(backend):
    return [{'name': m['name'], 'size_mib': m['size'] // 2**20, 'vram_mib': m['size_vram'] // 2**20,
             'ctx': m.get('context_length')} for m in httpx.get(url(backend, '/api/ps'), timeout=5).json()['models']]


def long_prompt(i):
    # ≈1,500 tokens of log lines; the run index up front defeats the prompt (prefix) cache between runs
    lines = ''.join(LOG_LINE.format(s % 60, 9000 + s, 20 + s % 37) for s in range(60))
    return f'Run {i}. How many of these requests took more than 50 ms? Answer with a number.\n{lines}'


def start(backend, mount):
    name = f'ollama-lab-{backend}'
    t = time.time()
    docker('run', '-d', '--name', name, '--label', LABEL, '--gpus=all', '-v', mount,
           '-p', f'127.0.0.1:{PORTS[backend]}:11434', IMAGE)
    up = wait_up(backend)
    return name, round(time.time() - t, 2), up


def measure(backend, name=None):
    """Reloads x3 (the first one may include a cold read), then warm speed."""
    res = {'reloads': [], 'warm_short': [], 'warm_long': []}
    for i in range(3):
        unload(backend)
        before = vram()
        g = generate(backend, SHORT)
        g['vram_delta_mib'] = vram() - before
        res['reloads'].append(g)
        print(f'    reload {i + 1}: load {g["load_s"]} s, wall {g["wall_s"]} s, {g["gen_tps"]} tok/s')
    res['ps'] = loaded(backend)
    if name:
        res['container_mem'] = docker('stats', '--no-stream', '--format', '{{.MemUsage}}', name)
    for _ in range(5):
        res['warm_short'].append(generate(backend, SHORT, num_predict=256))
    for i in range(3):
        res['warm_long'].append(generate(backend, long_prompt(i), num_predict=16))
    res['text'] = res['warm_short'][0]['text']
    unload(backend)
    return res


ap = argparse.ArgumentParser()
ap.add_argument('--contend', action='store_true', help='Part D also loads the 4b in the container (overflow)')
args = ap.parse_args()
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'container-ollama-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report, data = [], {}
say = lambda text='': (print(text, flush=True), report.append(text))

# ---------------------------------------------------------------- preflight
docker('version', '--format', '{{.Server.Version}}')            # fails fast if the engine isn't running
for backend in ('bind', 'volume'):
    other = docker('ps', '-a', '--filter', f'name=^ollama-lab-{backend}$', '--format', '{{.Labels}}')
    if other and LABEL not in other:
        raise SystemExit(f'A container named ollama-lab-{backend} exists and is not ours; not touching it.')
    if other:
        docker('rm', '-f', f'ollama-lab-{backend}')
vol = docker('volume', 'ls', '--filter', f'name=^{VOLUME}$', '--format', '{{.Labels}}')
if vol and LABEL not in vol:
    raise SystemExit(f'A volume named {VOLUME} exists and is not ours; not touching it.')
if httpx.get(url('windows', '/api/ps'), timeout=5).json()['models']:
    raise SystemExit('The Windows Ollama has a model loaded. Stop it first (ollama stop <model>) so runs are alone.')
if docker('images', '-q', IMAGE) == '':
    raise SystemExit(f'{IMAGE} is not pulled (docker pull {IMAGE}, 3.39 GB).')
idle = vram()
say(f'# container_ollama: {MODEL}, {IMAGE} vs Windows native, temperature 0, seed 42, think off, {stamp}\n')
say(f'* Docker engine {docker("version", "--format", "{{.Server.Version}}")}, '
    f'VM: {docker("info", "--format", "{{.NCPU}} CPUs, {{.MemTotal}} bytes RAM, kernel {{.KernelVersion}}")}')
say(f'* Windows Ollama {httpx.get(url("windows", "/api/version")).json()["version"]}; GPU idle {idle} MiB used')

# ---------------------------------------------------------------- B/C: windows
say('\n## Windows native (127.0.0.1:11434)\n')
data['windows'] = measure('windows')

# ---------------------------------------------------------------- B/C: bind mount
say('\n## Container, Windows store bind-mounted read-only (127.0.0.1:11435)\n')
name, created_s, up_s = start('bind', f'{STORE}:/root/.ollama/models:ro')
data['bind_start'] = {'run_to_api_s': created_s, 'api_wait_s': up_s}
say(f'* docker run → API answering: {created_s} s')
data['bind'] = measure('bind', name)
t = time.time()
docker('restart', name)
data['bind_restart'] = {'restart_to_api_s': round(time.time() - t + wait_up('bind'), 2),
                        'models_listed': len(httpx.get(url('bind', '/api/tags')).json()['models'])}
say(f'* docker restart → API: {data["bind_restart"]["restart_to_api_s"]} s; models still listed: '
    f'{data["bind_restart"]["models_listed"]} (nothing downloaded)')

# ---------------------------------------------------------------- A: exposure
say('\n## A. Exposure\n')
data['exposure'] = {
    'port_bindings': json.loads(docker('inspect', '-f', '{{json .HostConfig.PortBindings}}', name)),
    'ollama_host_in_container': docker('exec', name, 'printenv', 'OLLAMA_HOST'),
    'listening': [l.split() for l in subprocess.run(['netstat', '-ano', '-p', 'TCP'], capture_output=True,
                  text=True).stdout.splitlines() if 'LISTENING' in l and any(f':{p} ' in l for p in PORTS.values())]}
say(f'* PortBindings: {data["exposure"]["port_bindings"]}')
say(f'* OLLAMA_HOST inside the container: {data["exposure"]["ollama_host_in_container"]}')
for l in data['exposure']['listening']:
    say(f'* Windows LISTENING: {l[1]} (pid {l[-1]})')

# ---------------------------------------------------------------- D: two instances on one GPU
say('\n## D. Two Ollama instances on one GPU\n')
d = {'idle_mib': vram()}
generate('windows', SHORT)
d['after_windows_4b_mib'] = vram()
httpx.post(url('bind', '/api/embed'), json={'model': SMALL, 'input': 'kubectl rollout status'}, timeout=300)
d['after_container_embed_mib'] = vram()
d['windows_ps'], d['bind_ps'] = loaded('windows'), loaded('bind')
d['windows_gen_tps_both_loaded'] = generate('windows', SHORT, num_predict=256)['gen_tps']
if args.contend:
    say('* --contend: loading the 4b in the container as well (announced overflow past the 8,500 MiB ceiling)')
    g = generate('bind', SHORT, num_predict=256)
    d['contend'] = {'after_both_4b_mib': vram(), 'bind_ps': loaded('bind'), 'windows_ps': loaded('windows'),
                    'bind_gen_tps': g['gen_tps'], 'bind_load_s': g['load_s'],
                    'windows_gen_tps': generate('windows', SHORT, num_predict=256)['gen_tps']}
    unload('bind')
unload('bind', SMALL)
unload('windows')
data['two_instances'] = d
say(f'* GPU used: idle {d["idle_mib"]} → Windows 4b {d["after_windows_4b_mib"]} → + container {SMALL} '
    f'{d["after_container_embed_mib"]} MiB')
say(f'* Windows /api/ps: {d["windows_ps"]}')
say(f'* Container /api/ps: {d["bind_ps"]}')
say(f'* Windows generation with both loaded: {d["windows_gen_tps_both_loaded"]} tok/s')
if 'contend' in d:
    say(f'* --contend: {d["contend"]}')
docker('rm', '-f', name)

# ---------------------------------------------------------------- B/C: named volume
say(f'\n## Container, the 4b in the named volume {VOLUME} (127.0.0.1:11436)\n')
manifest_rel = 'manifests/registry.ollama.ai/library/qwen3.5/4b'
m = json.loads((STORE / manifest_rel).read_text())
blobs = [l['digest'].replace(':', '-') for l in m['layers'] + [m['config']]]
size = sum(l['size'] for l in m['layers'] + [m['config']])
docker('volume', 'create', '--label', LABEL, VOLUME)
t = time.time()
docker('run', '--rm', '--label', LABEL, '-v', f'{STORE}:/src:ro', '-v', f'{VOLUME}:/dst', 'debian:bookworm-slim',
       'sh', '-c', f'mkdir -p /dst/blobs /dst/{Path(manifest_rel).parent.as_posix()} && '
       f'cp /src/{manifest_rel} /dst/{manifest_rel} && cp ' + ' '.join(f'/src/blobs/{b}' for b in blobs) +
       ' /dst/blobs/')
copy_s = round(time.time() - t, 1)
data['volume_copy'] = {'bytes': size, 'seconds': copy_s, 'mb_per_s': round(size / 1e6 / copy_s)}
say(f'* copied {size / 1e9:.2f} GB from the Windows store into the volume in {copy_s} s '
    f'({data["volume_copy"]["mb_per_s"]} MB/s)')
name, created_s, up_s = start('volume', f'{VOLUME}:/root/.ollama/models')
data['volume_start'] = {'run_to_api_s': created_s}
say(f'* docker run → API answering: {created_s} s')
data['volume'] = measure('volume', name)
docker('rm', '-f', name)

# ---------------------------------------------------------------- summary
say('\n## B/C. Summary (reload = unload with keep_alive 0, then the same request)\n')
say('| backend | reload load s (1st, 2nd, 3rd) | VRAM Δ MiB | gen tok/s warm (median of 5, 256 tok) | '
    'prompt tok/s (median of 3, ≈1.5K tok) | same text as Windows | container RAM |')
say('|---|---|---:|---:|---:|---|---|')
for b in ('windows', 'bind', 'volume'):
    r = data[b]
    say(f'| {b} | {", ".join(str(x["load_s"]) for x in r["reloads"])} | {r["reloads"][-1]["vram_delta_mib"]} | '
        f'{statistics.median(x["gen_tps"] for x in r["warm_short"])} | '
        f'{statistics.median(x["prompt_tps"] for x in r["warm_long"])} | '
        f'{"–" if b == "windows" else r["text"] == data["windows"]["text"]} | {r.get("container_mem", "–")} |')
say(f'\n* /api/ps while loaded: ' + '; '.join(f'{b} {data[b]["ps"]}' for b in ('windows', 'bind', 'volume')))
say(f'* long-prompt answers: ' + '; '.join(f'{b} {[x["text"].strip() for x in data[b]["warm_long"]]}'
                                           for b in ('windows', 'bind', 'volume')))
say(f'* GPU after cleanup: {vram()} MiB (idle at start {idle})')
say(f'* left in place: the volume {VOLUME} ({size / 1e9:.2f} GB). Remove with: docker volume rm {VOLUME}')

(outdir / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
(outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(f'\nSaved {outdir}')
