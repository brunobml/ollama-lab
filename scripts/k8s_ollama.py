"""Lab 08 Step 2. Ollama on a local Kubernetes cluster (k3d, k3s in Docker) with the GPU and no device plugin.

Run with the venv's interpreter, Docker Desktop running, from the repo root:
    .\\.venv\\Scripts\\python scripts\\k8s_ollama.py              (≈10-12 min)
    .\\.venv\\Scripts\\python scripts\\k8s_ollama.py --smoke      (Parts A-B and one request, ≈4 min)
    .\\.venv\\Scripts\\python scripts\\k8s_ollama.py --contend    (Part D also loads the 4b in both replicas:
                                                                 an announced overflow past the 8,500 MiB ceiling)

The cluster, ollama-lab, is created by k3d in WSL (Ubuntu) and deleted at the end (--keep keeps it):
  1 server, no agents, no load balancer, traefik and metrics-server disabled, rancher/k3s:v1.36.4-k3s1 (cached)
  --gpus all                  Docker adds /dev/dxg and the driver libraries to the node container
  --api-port 127.0.0.1:6445   k3d's default publishes the API server on every interface
  -v ollama-lab-models:/var/lib/ollama-models   Step 1's named volume (the 4b), mounted into the node
  its own kubeconfig file (%USERPROFILE%\\.kube\\ollama-lab.yaml); every kubectl call passes it and --context
Parts:
  A  the cluster: create and image-import times, API binding, what the node advertises (no nvidia.com/gpu)
  B  GPU in a pod without a device plugin: unprivileged vs privileged (k8s/gpu-probe.yaml)
  C  Ollama Deployment (k8s/ollama.yaml): Ready vs first answer; speed; a pod deletion (a "reschedule") with and
     without a node-local CUDA kernel cache (/root/.nv on hostPath)
  D  replicas: 2 with no GPU accounting: where they land and what each sees
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

CLUSTER, CONTEXT = 'ollama-lab', 'k3d-ollama-lab'
KUBECONFIG = Path.home() / '.kube' / 'ollama-lab.yaml'
K3S_IMAGE, IMAGE = 'rancher/k3s:v1.36.4-k3s1', 'ollama/ollama:0.40.13'
VOLUME, MODEL = 'ollama-lab-models', 'qwen3.5:4b'
LOCAL_PORT = 11437
MANIFESTS = Path('labs/08-operations-containers/k8s')
PROMPT = 'Explain a Kubernetes liveness probe in two sentences.'
NV_CACHE_PATCH = json.dumps([
    {'op': 'add', 'path': '/spec/template/spec/volumes/-',
     'value': {'name': 'nvcache', 'hostPath': {'path': '/var/lib/ollama-nv', 'type': 'DirectoryOrCreate'}}},
    {'op': 'add', 'path': '/spec/template/spec/containers/0/volumeMounts/-',
     'value': {'name': 'nvcache', 'mountPath': '/root/.nv'}}])


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
    """kubectl port-forward to one pod on 127.0.0.1 (kubectl's default address). Dies with the pod."""
    def __init__(self, pod):
        self.proc = subprocess.Popen(['kubectl', '--kubeconfig', str(KUBECONFIG), '--context', CONTEXT,
                                      'port-forward', f'pod/{pod}', f'{LOCAL_PORT}:11434'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                httpx.get(f'http://127.0.0.1:{LOCAL_PORT}/api/version', timeout=2)
                return
            except httpx.HTTPError:
                time.sleep(0.5)
        raise SystemExit(f'port-forward to {pod} never answered')

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)


def generate(num_predict=128):
    t = time.time()
    r = httpx.post(f'http://127.0.0.1:{LOCAL_PORT}/api/generate', timeout=900, json={
        'model': MODEL, 'prompt': PROMPT, 'stream': False, 'think': False,
        'options': {'temperature': 0, 'seed': 42, 'num_predict': num_predict}}).json()
    return {'wall_s': round(time.time() - t, 2), 'load_s': round(r['load_duration'] / 1e9, 2),
            'gen_tps': round(r['eval_count'] / (r['eval_duration'] / 1e9), 1), 'text': r['response']}


def ollama_pods():
    out = json.loads(kubectl('get', 'pods', '-l', 'app=ollama', '-o', 'json'))['items']
    return [p for p in out if not p['metadata'].get('deletionTimestamp')]


def wait_ready(count=1, exclude=()):
    """Seconds until `count` non-terminating ollama pods (not in `exclude`) are Ready; returns their names."""
    t = time.time()
    while time.time() - t < 300:
        pods = [p for p in ollama_pods() if p['metadata']['name'] not in exclude]
        ready = [p['metadata']['name'] for p in pods
                 if any(c['type'] == 'Ready' and c['status'] == 'True' for c in p['status'].get('conditions', []))]
        if len(ready) >= count:
            return ready, round(time.time() - t, 1)
        time.sleep(0.5)
    raise SystemExit(kubectl('describe', 'pods', '-l', 'app=ollama'))


def unload_and_close(pf):
    httpx.post(f'http://127.0.0.1:{LOCAL_PORT}/api/generate', json={'model': MODEL, 'keep_alive': 0}, timeout=60)
    pf.close()


def first_answer(label, exclude=()):
    """A fresh pod: time to Ready, then the first request through a port-forward."""
    t0 = time.time()
    (pod,), ready_s = wait_ready(exclude=exclude)
    pf = PortForward(pod)
    g = generate()
    res = {'pod': pod, 'ready_s': ready_s, 'first_wall_s': g['wall_s'], 'first_load_s': g['load_s'],
           'ready_to_answer_s': round(time.time() - t0, 1), 'nv_cache': kubectl(
               'exec', pod, '--', 'sh', '-c', 'du -sk /root/.nv 2>/dev/null | cut -f1')}
    say(f'* {label}: pod Ready after {ready_s} s; first request {g["wall_s"]} s (load {g["load_s"]} s); '
        f'ready → answered {res["ready_to_answer_s"]} s total; /root/.nv {res["nv_cache"]} KB')
    return res, pf


sys.stdout.reconfigure(encoding='utf-8')           # '→' in the report; a piped Windows console is cp1252
ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true', help='Parts A-B and one request only')
ap.add_argument('--contend', action='store_true', help='Part D loads the 4b in both replicas (overflow)')
ap.add_argument('--keep', action='store_true', help='keep the cluster at the end')
args = ap.parse_args()
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'k8s-ollama-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report, data = [], {}
say = lambda text='': (print(text, flush=True), report.append(text))

# ---------------------------------------------------------------- preflight
run(['docker', 'version', '--format', '{{.Server.Version}}'])
if CLUSTER in k3d('cluster', 'list', '-o', 'json'):
    raise SystemExit(f'A k3d cluster named {CLUSTER} exists. Delete it first: wsl -d Ubuntu -- k3d cluster delete {CLUSTER}')
if not run(['docker', 'volume', 'ls', '-q', '--filter', f'name=^{VOLUME}$']):
    raise SystemExit(f'Volume {VOLUME} not found: run Step 1 (scripts/container_ollama.py) first.')
for image in (K3S_IMAGE, IMAGE, 'debian:bookworm-slim'):
    if not run(['docker', 'images', '-q', image]):
        raise SystemExit(f'{image} is not in the local image store; nothing is pulled by this script.')
idle = vram()
say(f'# k8s_ollama: k3d {CLUSTER} ({K3S_IMAGE}), {IMAGE}, {MODEL}, temperature 0, seed 42, think off, {stamp}\n')
say(f'* GPU idle: {idle} MiB used')

# ---------------------------------------------------------------- A: the cluster
say('\n## A. The cluster\n')
_, create_s = timed(k3d, 'cluster', 'create', CLUSTER, '--image', K3S_IMAGE, '--servers', '1', '--agents', '0',
                    '--no-lb', '--api-port', '127.0.0.1:6445', '--gpus', 'all',
                    '-v', f'{VOLUME}:/var/lib/ollama-models@server:0',
                    '--kubeconfig-update-default=false', '--kubeconfig-switch-context=false',
                    '--k3s-arg', '--disable=traefik@server:0', '--k3s-arg', '--disable=metrics-server@server:0')
KUBECONFIG.write_text(k3d('kubeconfig', 'get', CLUSTER) + '\n', encoding='utf-8')
_, import_s = timed(k3d, 'image', 'import', IMAGE, 'debian:bookworm-slim', '-c', CLUSTER)
node = json.loads(kubectl('get', 'nodes', '-o', 'json'))['items'][0]
data['cluster'] = {'create_s': create_s, 'import_s': import_s,
                   'api_binding': run(['docker', 'port', f'k3d-{CLUSTER}-server-0']),
                   'allocatable': node['status']['allocatable'],
                   'node_version': node['status']['nodeInfo']['kubeletVersion'],
                   'runtime': node['status']['nodeInfo']['containerRuntimeVersion']}
say(f'* k3d cluster create: {create_s} s; image import ({IMAGE} + debian): {import_s} s')
say(f'* API server published: {data["cluster"]["api_binding"]}')
say(f'* node {data["cluster"]["node_version"]}, {data["cluster"]["runtime"]}; allocatable: '
    f'{data["cluster"]["allocatable"]}')
say(f'* nvidia.com/gpu advertised: {"nvidia.com/gpu" in node["status"]["allocatable"]}')

# ---------------------------------------------------------------- B: GPU without a device plugin
say('\n## B. GPU in a pod without a device plugin\n')
kubectl('apply', '-f', str(MANIFESTS / 'gpu-probe.yaml'))
data['probe'] = {}
for name in ('gpu-probe-unprivileged', 'gpu-probe-privileged'):
    kubectl('wait', '--for=jsonpath={.status.phase}=Succeeded', f'pod/{name}', '--timeout=90s', check=False)
    phase = kubectl('get', 'pod', name, '-o', 'jsonpath={.status.phase}')
    log = kubectl('logs', name, check=False)
    data['probe'][name] = {'phase': phase, 'log': log}
    say(f'* {name}: {phase}')
    for line in log.splitlines():
        say(f'  * `{line.strip()}`')
kubectl('delete', '-f', str(MANIFESTS / 'gpu-probe.yaml'), '--wait=false')

# ---------------------------------------------------------------- C: Ollama Deployment
say('\n## C. Ollama Deployment (privileged, hostPath)\n')
kubectl('apply', '-f', str(MANIFESTS / 'ollama.yaml'))
c1, pf = first_answer('pod 1 (new)')
warm = [generate(256) for _ in range(5)]
c1['warm_gen_tps'] = statistics.median(w['gen_tps'] for w in warm)
c1['vram_mib'], c1['vram_delta_mib'] = vram(), vram() - idle
c1['ps'] = httpx.get(f'http://127.0.0.1:{LOCAL_PORT}/api/ps').json()['models']
c1['text'] = warm[0]['text']
say(f'* warm generation (median of 5, 256 tokens): {c1["warm_gen_tps"]} tok/s; GPU {c1["vram_mib"]} MiB '
    f'(+{c1["vram_delta_mib"]} over idle)')
data['pod1'] = c1
unload_and_close(pf)
if not args.smoke:
    kubectl('delete', 'pod', c1['pod'], '--wait=false')
    data['pod2'], pf = first_answer('pod 2 (after deleting pod 1, no kernel cache)', exclude={c1['pod']})
    unload_and_close(pf)
    say('* adding a node-local hostPath for /root/.nv (the CUDA kernel cache):')
    kubectl('patch', 'deployment', 'ollama', '--type', 'json', '-p', NV_CACHE_PATCH)
    data['pod3'], pf = first_answer('pod 3 (cache volume, empty)', exclude={data['pod2']['pod']})
    unload_and_close(pf)
    kubectl('delete', 'pod', data['pod3']['pod'], '--wait=false')
    data['pod4'], pf = first_answer('pod 4 (after deleting pod 3, cache kept on the node)',
                                    exclude={data['pod3']['pod']})
    unload_and_close(pf)

    # ------------------------------------------------------------ D: two replicas, no GPU accounting
    say('\n## D. replicas: 2 with no GPU accounting\n')
    kubectl('scale', 'deployment', 'ollama', '--replicas=2')
    pods, both_s = wait_ready(2)
    d = {'ready_s': both_s, 'pods': [], 'vram_idle_models_mib': vram()}
    for p in ollama_pods():
        d['pods'].append({'name': p['metadata']['name'], 'node': p['spec']['nodeName'],
                          'gpu_seen': kubectl('exec', p['metadata']['name'], '--', 'sh', '-c',
                                              'ls /dev/dxg >/dev/null && echo yes || echo no')})
    say(f'* both Ready after {both_s} s on: {sorted({p["node"] for p in d["pods"]})}; each sees /dev/dxg: '
        f'{[p["gpu_seen"] for p in d["pods"]]}; GPU {d["vram_idle_models_mib"]} MiB with no model loaded')
    if args.contend:
        say('* --contend: loading the 4b in both replicas (announced overflow past the 8,500 MiB ceiling)')
        d['contend'] = []
        for p in d['pods']:
            pf = PortForward(p['name'])
            g = generate(256)
            d['contend'].append({'pod': p['name'], 'load_s': g['load_s'], 'gen_tps': g['gen_tps'],
                                 'vram_mib': vram(), 'ps': httpx.get(f'http://127.0.0.1:{LOCAL_PORT}/api/ps').json()})
            pf.close()                       # the model stays loaded, so the second pod loads next to it
            say(f'  * {p["name"]}: load {g["load_s"]} s, {g["gen_tps"]} tok/s, GPU now {vram()} MiB')
    data['replicas2'] = d

# ---------------------------------------------------------------- cleanup
if args.keep:
    say(f'\n* cluster kept. kubectl --kubeconfig {KUBECONFIG} --context {CONTEXT} …; '
        f'delete with: wsl -d Ubuntu -- k3d cluster delete {CLUSTER}')
else:
    _, delete_s = timed(k3d, 'cluster', 'delete', CLUSTER)
    KUBECONFIG.unlink(missing_ok=True)
    say(f'\n* cluster deleted in {delete_s} s; GPU after cleanup {vram()} MiB (idle at start {idle})')

(outdir / 'data.json').write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
(outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(f'\nSaved {outdir}')
