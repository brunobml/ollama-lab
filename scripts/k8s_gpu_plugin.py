"""Lab 08 Step 3. The GPU through NVIDIA's device plugin (CDI) on k3d: unprivileged pods, nvidia.com/gpu counted.

Run with the venv's interpreter, Docker Desktop running, from the repo root:
    .\\.venv\\Scripts\\python scripts\\k8s_gpu_plugin.py          (≈15 min; Part F waits up to 6 min)
    .\\.venv\\Scripts\\python scripts\\k8s_gpu_plugin.py --keep   (keep the cluster at the end)

Same cluster as Step 2 (ollama-lab, k3d, --gpus all, API on 127.0.0.1:6445, its own kubeconfig file, Step 1's
volume in the node); it is deleted at the end. Nothing is pulled: the images are imported from the local store,
including nvcr.io/nvidia/k8s-device-plugin:v0.20.1 (pulled once, 161 MB).
Parts:
  A  the plugin (k8s/nvidia-device-plugin.yaml): time to nvidia.com/gpu, the CDI spec it writes, what it logs
  B  unprivileged pods that request the GPU (k8s/gpu-probe-cdi.yaml): CDI only vs CDI + the node's /usr/lib64
  C  Ollama with the GPU from the plugin (k8s/ollama-gpu.yaml, every capability dropped): first answer, speed,
     planning number; a pod deletion with the kernel cache on the node
  D  replicas: 2 with one GPU
  E  a RollingUpdate with the GPU counted, then back to Recreate
  F  the plugin removed while Ollama runs: the running pod, the node's nvidia.com/gpu, a replacement pod; re-added
"""
import argparse
import atexit
import json
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

CLUSTER, CONTEXT, NODE = 'ollama-lab', 'k3d-ollama-lab', 'k3d-ollama-lab-server-0'
KUBECONFIG = Path.home() / '.kube' / 'ollama-lab.yaml'
K3S_IMAGE, IMAGE = 'rancher/k3s:v1.36.4-k3s1', 'ollama/ollama:0.40.13'
PLUGIN_IMAGE, PROBE_IMAGE = 'nvcr.io/nvidia/k8s-device-plugin:v0.20.1', 'debian:bookworm-slim'
VOLUME, MODEL = 'ollama-lab-models', 'qwen3.5:4b'
LOCAL_PORT = 11437
MANIFESTS = Path('labs/08-operations-containers/k8s')
PROMPT = 'Explain a Kubernetes liveness probe in two sentences.'
GPU = 'nvidia.com/gpu'


def run(cmd, check=True, timeout=900):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    if check and r.returncode:
        raise SystemExit(f'{" ".join(cmd)} failed:\n{r.stdout}\n{r.stderr}')
    return r.stdout.strip()


def k3d(*args, **kw):
    return run(['wsl.exe', '-d', 'Ubuntu', '--', 'k3d', *args], **kw)


def kubectl(*args, **kw):
    return run(['kubectl', '--kubeconfig', str(KUBECONFIG), '--context', CONTEXT, *args], **kw)


def vram():
    return int(run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']))


def timed(fn, *args):
    t = time.time()
    out = fn(*args)
    return out, round(time.time() - t, 1)


class PortForward:
    """kubectl port-forward to one pod on 127.0.0.1. Refuses a busy port: a leftover forward answers for the
    wrong pod without any error (Step 2)."""
    open = []

    def __init__(self, pod):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', LOCAL_PORT)) == 0:
                raise SystemExit(f'127.0.0.1:{LOCAL_PORT} is already in use (a leftover port-forward?)')
        self.proc = subprocess.Popen(['kubectl', '--kubeconfig', str(KUBECONFIG), '--context', CONTEXT,
                                      'port-forward', f'pod/{pod}', f'{LOCAL_PORT}:11434'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        PortForward.open.append(self)
        for _ in range(60):
            try:
                httpx.get(f'http://127.0.0.1:{LOCAL_PORT}/api/version', timeout=2)
                return
            except httpx.HTTPError:
                time.sleep(0.5)
        raise SystemExit(f'port-forward to {pod} never answered')

    def close(self):
        if self in PortForward.open:
            PortForward.open.remove(self)
            self.proc.terminate()
            self.proc.wait(timeout=10)


atexit.register(lambda: [pf.close() for pf in list(PortForward.open)])


def generate(num_predict=128):
    t = time.time()
    r = httpx.post(f'http://127.0.0.1:{LOCAL_PORT}/api/generate', timeout=900, json={
        'model': MODEL, 'prompt': PROMPT, 'stream': False, 'think': False,
        'options': {'temperature': 0, 'seed': 42, 'num_predict': num_predict}})
    if r.status_code != 200:
        return {'wall_s': round(time.time() - t, 2), 'http': r.status_code, 'error': r.json().get('error', r.text)}
    r = r.json()
    return {'wall_s': round(time.time() - t, 2), 'load_s': round(r['load_duration'] / 1e9, 2),
            'gen_tps': round(r['eval_count'] / (r['eval_duration'] / 1e9), 1), 'text': r['response']}


def unload():
    httpx.post(f'http://127.0.0.1:{LOCAL_PORT}/api/generate', json={'model': MODEL, 'keep_alive': 0}, timeout=60)


def node_gpu():
    st = json.loads(kubectl('get', 'node', NODE, '-o', 'json'))['status']
    return st.get('capacity', {}).get(GPU), st.get('allocatable', {}).get(GPU)


def pods(label='app=ollama'):
    out = json.loads(kubectl('get', 'pods', '-l', label, '-o', 'json'))['items']
    return [p for p in out if not p['metadata'].get('deletionTimestamp')]


def is_ready(p):
    return any(c['type'] == 'Ready' and c['status'] == 'True' for c in p['status'].get('conditions', []))


def wait_ready(count=1, exclude=(), timeout=300):
    """Seconds until `count` non-terminating ollama pods (not in `exclude`) are Ready; returns their names."""
    t = time.time()
    while time.time() - t < timeout:
        ready = [p['metadata']['name'] for p in pods() if p['metadata']['name'] not in exclude and is_ready(p)]
        if len(ready) >= count:
            return ready, round(time.time() - t, 1)
        time.sleep(0.5)
    raise SystemExit(kubectl('describe', 'pods', '-l', 'app=ollama'))


def scheduling_events(pod):
    ev = json.loads(kubectl('get', 'events', '--field-selector', f'involvedObject.name={pod}', '-o', 'json'))['items']
    return [e['message'] for e in ev if e['reason'] == 'FailedScheduling']


def pod_states():
    return [{'pod': p['metadata']['name'], 'phase': p['status']['phase'], 'ready': is_ready(p),
             'unschedulable': next((c.get('message', '') for c in p['status'].get('conditions', [])
                                    if c['type'] == 'PodScheduled' and c['status'] == 'False'), None)}
            for p in pods()]


def exec_sh(pod, script):
    return kubectl('exec', pod, '--', 'sh', '-c', script, check=False)


def free_seen(pod):
    log = kubectl('logs', pod)
    (outdir / f'{pod}.log').write_text(log, encoding='utf-8')
    return [int(b) for _, b in re.findall(r'projected to use (\d+) MiB of device memory vs\. (\d+) MiB of free', log)]


def first_answer(label, exclude=()):
    t0 = time.time()
    (pod,), ready_s = wait_ready(exclude=exclude)
    pf = PortForward(pod)
    g = generate()
    if 'error' in g:
        raise SystemExit(f'{pod}: HTTP {g["http"]}: {g["error"]}')
    res = {'pod': pod, 'ready_s': ready_s, 'first_wall_s': g['wall_s'], 'first_load_s': g['load_s'],
           'ready_to_answer_s': round(time.time() - t0, 1)}
    say(f'* {label}: pod Ready after {ready_s} s; first request {g["wall_s"]} s (load {g["load_s"]} s); '
        f'ready → answered {res["ready_to_answer_s"]} s total')
    return res, pf


sys.stdout.reconfigure(encoding='utf-8')
ap = argparse.ArgumentParser()
ap.add_argument('--keep', action='store_true', help='keep the cluster at the end')
args = ap.parse_args()
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'k8s-gpu-plugin-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report, data = [], {}
say = lambda text='': (print(text, flush=True), report.append(text))


@atexit.register
def save():                     # also after a crash, which leaves the cluster running for inspection
    (outdir / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
    (outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'\nSaved {outdir}')


# ---------------------------------------------------------------- preflight
run(['docker', 'version', '--format', '{{.Server.Version}}'])
if CLUSTER in k3d('cluster', 'list', '-o', 'json'):
    raise SystemExit(f'A k3d cluster named {CLUSTER} exists. Delete it first: wsl -d Ubuntu -- k3d cluster delete {CLUSTER}')
if not run(['docker', 'volume', 'ls', '-q', '--filter', f'name=^{VOLUME}$']):
    raise SystemExit(f'Volume {VOLUME} not found: run Step 1 (scripts/container_ollama.py) first.')
for image in (K3S_IMAGE, IMAGE, PROBE_IMAGE, PLUGIN_IMAGE):
    if not run(['docker', 'images', '-q', image]):
        raise SystemExit(f'{image} is not in the local image store; nothing is pulled by this script.')
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1', LOCAL_PORT)) == 0:
        raise SystemExit(f'127.0.0.1:{LOCAL_PORT} is already in use (a leftover port-forward?)')
idle = vram()
say(f'# k8s_gpu_plugin: k3d {CLUSTER} ({K3S_IMAGE}), {PLUGIN_IMAGE}, {IMAGE}, {MODEL}, temperature 0, seed 42, '
    f'think off, {stamp}\n')
say(f'* GPU idle: {idle} MiB used')

# ---------------------------------------------------------------- A: the device plugin
say('\n## A. The device plugin\n')
_, create_s = timed(k3d, 'cluster', 'create', CLUSTER, '--image', K3S_IMAGE, '--servers', '1', '--agents', '0',
                    '--no-lb', '--api-port', '127.0.0.1:6445', '--gpus', 'all',
                    '-v', f'{VOLUME}:/var/lib/ollama-models@server:0',
                    '--kubeconfig-update-default=false', '--kubeconfig-switch-context=false',
                    '--k3s-arg', '--disable=traefik@server:0', '--k3s-arg', '--disable=metrics-server@server:0')
KUBECONFIG.write_text(k3d('kubeconfig', 'get', CLUSTER) + '\n', encoding='utf-8')
_, import_s = timed(k3d, 'image', 'import', IMAGE, PROBE_IMAGE, PLUGIN_IMAGE, '-c', CLUSTER)
cri = run(['docker', 'exec', NODE, 'crictl', 'info'])
cdi_cfg = {'enableCDI': re.search(r'"enableCDI":\s*(\w+)', cri).group(1),
           'cdiSpecDirs': json.loads(re.search(r'"cdiSpecDirs":\s*(\[[^\]]*\])', cri).group(1))}
before = node_gpu()
t = time.time()
kubectl('apply', '-f', str(MANIFESTS / 'nvidia-device-plugin.yaml'))
while node_gpu()[1] is None and time.time() - t < 120:
    time.sleep(0.5)
gpu_s = round(time.time() - t, 1)
spec_files = run(['docker', 'exec', NODE, 'ls', '/var/run/cdi']).split()
spec = json.loads(run(['docker', 'exec', NODE, 'cat', f'/var/run/cdi/{spec_files[0]}'])) if spec_files else {}
edits = spec.get('containerEdits', {})
plugin_pod = pods('name=nvidia-device-plugin')
plugin_log = kubectl('-n', 'kube-system', 'logs', 'ds/nvidia-device-plugin', check=False)
(outdir / 'device-plugin.log').write_text(plugin_log, encoding='utf-8')
data['plugin'] = {
    'create_s': create_s, 'import_s': import_s, 'containerd_cdi': cdi_cfg, 'node_gpu_before': before,
    'node_gpu_after': node_gpu(), 'apply_to_allocatable_s': gpu_s, 'spec_files': spec_files, 'spec': spec,
    'log_highlights': [l for l in plugin_log.splitlines()
                       if re.search(r'Detected platform|mode as|driver store path|Could not locate|Registered', l)]}
say(f'* k3d cluster create: {create_s} s; image import ({IMAGE}, {PROBE_IMAGE}, {PLUGIN_IMAGE}): {import_s} s')
say(f'* containerd CDI: {cdi_cfg}')
say(f'* node {GPU} (capacity, allocatable) before the plugin: {before}; after: {data["plugin"]["node_gpu_after"]}, '
    f'{gpu_s} s after `kubectl apply`')
say(f'* CDI spec {spec_files}: kind `{spec.get("kind")}`, devices {[d["name"] for d in spec.get("devices", [])]}, '
    f'device nodes {[n["path"] for d in spec.get("devices", []) for n in d["containerEdits"].get("deviceNodes", [])]}, '
    f'{len(edits.get("mounts", []))} mounts, env {edits.get("env")}, hooks {len(edits.get("hooks", []))}')
for line in data['plugin']['log_highlights']:
    say(f'  * `{line.strip()[:200]}`')

# ---------------------------------------------------------------- B: unprivileged GPU pods
say('\n## B. Unprivileged pods that request the GPU\n')
kubectl('apply', '-f', str(MANIFESTS / 'gpu-probe-cdi.yaml'))
names = ('gpu-cdi-only', 'gpu-cdi-libs', 'gpu-second')
t = time.time()
while time.time() - t < 180:
    phases = {p['metadata']['name']: p['status']['phase'] for p in pods('app=gpu-probe')}
    if all(phases.get(n) in ('Succeeded', 'Failed') for n in names):
        break
    time.sleep(1)
data['probes'] = {}
for name in names:
    p = json.loads(kubectl('get', 'pod', name, '-o', 'json'))
    st = p['status']
    started = st.get('startTime')
    data['probes'][name] = {'phase': st['phase'], 'log': kubectl('logs', name, check=False),
                            'failed_scheduling': scheduling_events(name), 'started': started}
    waited = data['probes'][name]['failed_scheduling']
    say(f'* {name}: {st["phase"]}' + (f'; first waited: "{waited[0][:90]}"' if waited else ''))
    for line in data['probes'][name]['log'].splitlines():
        say(f'  * `{line.strip()}`')
kubectl('delete', '-f', str(MANIFESTS / 'gpu-probe-cdi.yaml'), '--wait=false')

# ---------------------------------------------------------------- C: Ollama, unprivileged
say('\n## C. Ollama with the GPU from the plugin (unprivileged, every capability dropped)\n')
kubectl('apply', '-f', str(MANIFESTS / 'ollama-gpu.yaml'))
c1, pf = first_answer('pod 1 (new, node kernel cache empty)')
warm = [generate(256) for _ in range(5)]
c1['warm_gen_tps'] = [w['gen_tps'] for w in warm]
c1['vram_mib'] = vram()
c1['ps'] = httpx.get(f'http://127.0.0.1:{LOCAL_PORT}/api/ps').json()['models']
c1['text'] = warm[0]['text']
c1['in_pod'] = exec_sh(c1['pod'], 'grep CapEff /proc/self/status; echo "devices in /dev: $(ls /dev | wc -l)"; '
                                  'ls -l /dev/dxg; du -sk /root/.nv | cut -f1')
c1['free_seen'] = free_seen(c1['pod'])
say(f'* warm generation (median of 5, 256 tokens): {statistics.median(c1["warm_gen_tps"])} tok/s '
    f'(samples {c1["warm_gen_tps"]}); GPU {c1["vram_mib"]} MiB (+{c1["vram_mib"] - idle} over idle); '
    f'/api/ps size_vram == size: {all(m["size"] == m["size_vram"] for m in c1["ps"])}')
for line in c1['in_pod'].splitlines():
    say(f'  * `{line.strip()}`')
say(f'* planning line, free VRAM: {c1["free_seen"]} MiB')
data['pod1'] = c1
unload()
pf.close()
kubectl('delete', 'pod', c1['pod'], '--wait=false')
data['pod2'], pf = first_answer('pod 2 (pod 1 deleted, kernel cache kept on the node)', exclude={c1['pod']})
unload()
pf.close()

# ---------------------------------------------------------------- D: two replicas, one GPU
say('\n## D. replicas: 2 with one GPU\n')
kubectl('scale', 'deployment', 'ollama', '--replicas=2')
time.sleep(20)
d = {'states': pod_states()}
for s in d['states']:
    say(f'* {s["pod"]}: {s["phase"]}, ready {s["ready"]}' + (f'; "{s["unschedulable"][:110]}"' if s['unschedulable'] else ''))
kubectl('scale', 'deployment', 'ollama', '--replicas=1')
t = time.time()
while len(pods()) > 1 and time.time() - t < 60:
    time.sleep(1)
data['replicas2'] = d

# ---------------------------------------------------------------- E: RollingUpdate vs Recreate
say('\n## E. A rollout with the GPU counted\n')
old = pods()[0]['metadata']['name']
kubectl('patch', 'deployment', 'ollama', '--type', 'json', '-p', json.dumps([
    {'op': 'replace', 'path': '/spec/strategy', 'value': {'type': 'RollingUpdate',
                                                         'rollingUpdate': {'maxSurge': '25%', 'maxUnavailable': '25%'}}},
    {'op': 'add', 'path': '/spec/template/metadata/annotations', 'value': {'lab/rollout': 'rolling'}}]))
stuck = kubectl('rollout', 'status', 'deployment/ollama', '--timeout=45s', check=False)
e = {'rolling': {'states': pod_states(), 'rollout_status': stuck}}
for s in e['rolling']['states']:
    say(f'* RollingUpdate: {s["pod"]}{" (old)" if s["pod"] == old else ""}: {s["phase"]}, ready {s["ready"]}'
        + (f'; "{s["unschedulable"][:90]}"' if s['unschedulable'] else ''))
say(f'* `kubectl rollout status --timeout=45s`: {stuck.splitlines()[-1] if stuck else "(no output)"}')
t = time.time()
kubectl('patch', 'deployment', 'ollama', '--type', 'json', '-p', json.dumps([
    {'op': 'replace', 'path': '/spec/strategy', 'value': {'type': 'Recreate'}}]))
kubectl('rollout', 'status', 'deployment/ollama', '--timeout=180s')
e['recreate_s'] = round(time.time() - t, 1)
e['recreate_pods'] = [p['metadata']['name'] for p in pods()]
say(f'* switched to Recreate: rollout complete in {e["recreate_s"]} s; pods now {e["recreate_pods"]}')
data['rollout'] = e

# ---------------------------------------------------------------- F: the plugin removed
say('\n## F. The device plugin removed while Ollama runs\n')
(pod,), _ = wait_ready()
pf = PortForward(pod)
f = {'pod': pod}
kubectl('delete', '-f', str(MANIFESTS / 'nvidia-device-plugin.yaml'), '--wait=true')
t = time.time()
f['node_gpu'] = [(0.0, node_gpu())]
g = generate()
f['running_pod_after_removal'] = {k: g.get(k) for k in ('wall_s', 'load_s', 'gen_tps', 'http', 'error')}
say(f'* plugin deleted; running pod {pod}: '
    + (f'HTTP {g["http"]}: {g["error"][:120]}' if 'error' in g else f'answered, {g["gen_tps"]} tok/s'))
unload()
pf.close()
while time.time() - t < 360:                         # the kubelet keeps a stopped plugin's resource for a grace period
    cur = node_gpu()
    if cur != f['node_gpu'][-1][1]:
        f['node_gpu'].append((round(time.time() - t, 1), cur))
    if cur[1] in (None, '0'):
        break
    time.sleep(5)
say(f'* node {GPU} (capacity, allocatable) over time (s after deletion): {f["node_gpu"]}')
kubectl('delete', 'pod', pod, '--wait=true')
time.sleep(15)
f['replacement'] = pod_states()
for s in f['replacement']:
    say(f'* replacement pod {s["pod"]}: {s["phase"]}, ready {s["ready"]}'
        + (f'; "{s["unschedulable"][:110]}"' if s['unschedulable'] else ''))
t = time.time()
kubectl('apply', '-f', str(MANIFESTS / 'nvidia-device-plugin.yaml'))
while node_gpu()[1] in (None, '0') and time.time() - t < 120:
    time.sleep(0.5)
f['readd_to_allocatable_s'] = round(time.time() - t, 1)
wait_ready()
f['readd_to_ready_s'] = round(time.time() - t, 1)
say(f'* plugin re-applied: {GPU} back after {f["readd_to_allocatable_s"]} s; replacement pod Ready '
    f'{f["readd_to_ready_s"]} s after the re-apply')
data['plugin_removed'] = f

# ---------------------------------------------------------------- cleanup
if args.keep:
    say(f'\n* cluster kept. kubectl --kubeconfig {KUBECONFIG} --context {CONTEXT} …; '
        f'delete with: wsl -d Ubuntu -- k3d cluster delete {CLUSTER}')
else:
    _, delete_s = timed(k3d, 'cluster', 'delete', CLUSTER)
    KUBECONFIG.unlink(missing_ok=True)
    say(f'\n* cluster deleted in {delete_s} s; GPU after cleanup {vram()} MiB (idle at start {idle})')
