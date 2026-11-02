"""Lab 07 Step 3. Does thinking help a vision model count and read? Thinking vs a forced enumeration vs neither.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\vision_thinking.py           (≈10-15 minutes)
                                  .\\.venv\\Scripts\\python scripts\\vision_thinking.py --9b      (adds qwen3.5:9b, ≈25 min)
                                  .\\.venv\\Scripts\\python scripts\\vision_thinking.py --images  (draw only)

Three modes per task, same image, same num_ctx (12,288, so no reloads between modes):
  off    think off, answer straight into the schema
  steps  think off, but the schema makes the model write its working first (per-row counts, the names it counted)
         and the total last: enumeration in the visible output, checkable
  think  think on, then the plain schema; up to 8,000 output tokens. A run that hits the limit has no answer
Tasks (synthetic images, ground truth from the drawing code):
  grid-7, grid-12, grid-19   red squares among green ones (8x6, 8x6, 10x8 grids, 50 px squares: easy to see)
  circles-11                 unlabelled circles scattered on a canvas
  spikes-5                   how many times a CPU line goes above a dashed threshold
  running-12                 a 20-pod kubectl table: how many are Running
  read-x1, read-x0.75        Step 1's 12-pod table, read into JSON (names exact /12)
Temperature 0, seed 42, one run per cell.
"""
import base64
import io
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

HOST = 'http://127.0.0.1:11434'
MODELS = ['qwen3.5:4b'] + (['qwen3.5:9b'] if '--9b' in sys.argv else [])
NUM_CTX, THINK_BUDGET = 12288, 8000
http = httpx.Client(base_url=HOST, timeout=1800)
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'vision-thinking-{stamp}'
report = []
RED, GREEN, BLUE = (210, 40, 40), (60, 170, 80), (60, 100, 200)


def say(text=''):
    print(text, flush=True)
    report.append(text)


# ---------------------------------------------------------------- images
def grid(cols, rows, red, seed):
    rng = random.Random(seed)
    reds = set(rng.sample(range(cols * rows), red))
    im = Image.new('RGB', (20 + cols * 76, 20 + rows * 76), 'white')
    d = ImageDraw.Draw(im)
    for i in range(cols * rows):
        x, y = 20 + (i % cols) * 76, 20 + (i // cols) * 76
        d.rectangle((x, y, x + 50, y + 50), fill=RED if i in reds else GREEN)
    per_row = [sum((r * cols + c) in reds for c in range(cols)) for r in range(rows)]
    return im, per_row


def circles(n, seed):
    rng = random.Random(seed)
    im = Image.new('RGB', (800, 600), 'white')
    d = ImageDraw.Draw(im)
    placed = []
    while len(placed) < n:
        x, y = rng.randint(50, 750), rng.randint(50, 550)
        if all((x - a) ** 2 + (y - b) ** 2 > 90 ** 2 for a, b in placed):
            placed.append((x, y))
            d.ellipse((x - 28, y - 28, x + 28, y + 28), outline=BLUE, width=4)
    return im


def spikes(at, seed):
    """CPU line 14:00-15:00 at ≈30%, with a 2-minute spike to ≈90% at each minute in `at`; threshold 70%."""
    rng = random.Random(seed)
    vals = [30 + rng.uniform(-4, 4) for _ in range(61)]
    for m in at:
        vals[m], vals[m + 1] = 88 + rng.uniform(-4, 4), 84 + rng.uniform(-4, 4)
    im = Image.new('RGB', (900, 400), (24, 27, 31))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype('arial.ttf', 13)
    x0, y0, x1, y1 = 60, 40, 870, 350
    xs, ys = (lambda m: x0 + (x1 - x0) * m / 60), (lambda v: y1 - (y1 - y0) * v / 100)
    d.text((x0, 12), 'CPU usage (%) - checkout-web', fill=(204, 204, 220), font=ImageFont.truetype('arialbd.ttf', 15))
    for v in (0, 25, 50, 75, 100):
        d.line((x0, ys(v), x1, ys(v)), fill=(44, 48, 54))
        d.text((x0 - 8, ys(v)), str(v), fill=(160, 160, 170), font=f, anchor='rm')
    for m in range(0, 61, 10):
        d.text((xs(m), y1 + 8), f'14:{m:02d}' if m < 60 else '15:00', fill=(160, 160, 170), font=f, anchor='ma')
    for xa in range(x0, x1, 14):
        d.line((xa, ys(70), min(xa + 7, x1), ys(70)), fill=(242, 73, 92), width=2)
    d.line([(xs(m), ys(v)) for m, v in enumerate(vals)], fill=(87, 148, 242), width=2)
    return im, [f'14:{m:02d}' for m in at]


# Step 1's table (same rows as vision_basics.py), and a 20-pod table for counting
PODS12 = [
    ('payments-api-7d9f8c6b5-x2k4q', '0/1', 'CrashLoopBackOff', 17, '17 (2m14s ago)', '61m'),
    ('payments-api-7d9f8c6b5-x2k4p', '1/1', 'Running', 0, '0', '61m'),
    ('payments-worker-5c8d7f9b4-mq7zt', '1/1', 'Running', 2, '2 (40m ago)', '3h12m'),
    ('checkout-web-6b4f9d8c7-9hj2w', '1/1', 'Running', 0, '0', '5d'),
    ('checkout-web-6b4f9d8c7-t8r5n', '0/1', 'ImagePullBackOff', 0, '0', '12m'),
    ('inventory-sync-28471935-7xkqp', '0/1', 'Completed', 0, '0', '47m'),
    ('inventory-sync-28471950-c2vnl', '0/1', 'Error', 3, '3', '32m'),
    ('redis-cache-0', '1/1', 'Running', 1, '1 (6d ago)', '14d'),
    ('redis-cache-1', '1/1', 'Running', 0, '0', '14d'),
    ('search-indexer-84c6d9f7b8-lp4ds', '0/1', 'Pending', 0, '0', '8m'),
    ('search-indexer-84c6d9f7b8-zz9kd', '0/1', 'OOMKilled', 5, '5 (90s ago)', '25m'),
    ('notifications-6f7d8b9c5d-w3e1r', '2/2', 'Running', 0, '0', '2d'),
]


def pods20(seed):
    rng = random.Random(seed)
    apps = ['payments-api', 'checkout-web', 'search-indexer', 'notifications', 'ledger', 'fraud-score', 'gateway']
    statuses = ['Running'] * 12 + ['CrashLoopBackOff', 'CrashLoopBackOff', 'Pending', 'Pending', 'Error',
                                   'ImagePullBackOff', 'Completed', 'OOMKilled']
    rng.shuffle(statuses)
    rows = []
    for s in statuses:
        name = f'{rng.choice(apps)}-{rng.getrandbits(36):09x}'[:-4] + '-' + ''.join(rng.choice('bcdfghjklmnpqrstvwxz2456789') for _ in range(5))
        ready = '1/1' if s == 'Running' else '0/1'
        rows.append((name, ready, s, 0 if s == 'Running' else rng.randint(1, 9), '', f'{rng.randint(2, 59)}m'))
    return rows


def table(rows):
    lines = [('NAME', 'READY', 'STATUS', 'RESTARTS', 'AGE')] + [(n, r, s, rs or str(rc), a) for n, r, s, rc, rs, a in rows]
    w = [max(len(row[i]) for row in lines) for i in range(5)]
    text = '$ kubectl get pods -n payments\n' + '\n'.join('   '.join(c.ljust(w[i]) for i, c in enumerate(row)).rstrip()
                                                          for row in lines)
    font = ImageFont.truetype('consola.ttf', 16)
    tl = text.split('\n')
    im = Image.new('RGB', (int(max(font.getlength(x) for x in tl)) + 40, 22 * len(tl) + 40), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(tl):
        d.text((20, 20 + 22 * i), line, fill=(220, 220, 220), font=font)
    return im


# ---------------------------------------------------------------- tasks
def count_schema(steps_field=None, steps_type=None):
    props = {'count': {'type': 'integer'}}
    req = ['count']
    if steps_field:
        props = {steps_field: {'type': 'array', 'items': {'type': steps_type}}, **props}   # working first, total last
        req = [steps_field, 'count']
    return {'type': 'object', 'required': req, 'properties': props}


POD_SCHEMA = {'type': 'object', 'required': ['pods'], 'properties': {'pods': {'type': 'array', 'items': {
    'type': 'object', 'required': ['name', 'ready', 'status', 'restarts'],
    'properties': {'name': {'type': 'string'}, 'ready': {'type': 'string'}, 'status': {'type': 'string'},
                   'restarts': {'type': 'integer'}}}}}}
POD_PROMPT = ('List every pod in this kubectl output: name, READY, STATUS and the RESTARTS count as an integer. '
              'Copy names exact. Only include pods that are shown.')


def build_tasks():
    tasks = []
    for name, (cols, rows, red, seed) in {'grid-7': (8, 6, 7, 1), 'grid-12': (8, 6, 12, 2),
                                          'grid-19': (10, 8, 19, 3)}.items():
        im, per_row = grid(cols, rows, red, seed)
        tasks.append({'id': name, 'image': im, 'kind': 'count', 'truth': red, 'working_truth': per_row,
                      'q': 'How many red squares are in the image?',
                      'steps_q': 'How many red squares are in the image? First give the number of red squares in each '
                                 'row, top to bottom, then the total.',
                      'steps': ('red_per_row', 'integer')})
    tasks.append({'id': 'circles-11', 'image': circles(11, 4), 'kind': 'count', 'truth': 11,
                  'q': 'How many circles are in the image?',
                  'steps_q': 'How many circles are in the image? First describe the position of each circle '
                             '(e.g. "top left"), one entry per circle, then give the total.',
                  'steps': ('circle_positions', 'string')})
    im, times = spikes([5, 17, 26, 41, 52], 5)
    tasks.append({'id': 'spikes-5', 'image': im, 'kind': 'count', 'truth': 5, 'working_truth': times,
                  'q': 'How many separate times does the CPU line go above the dashed threshold line?',
                  'steps_q': 'How many separate times does the CPU line go above the dashed threshold line? First list '
                             'the approximate time (HH:MM) of each spike above the line, then the total.',
                  'steps': ('spike_times', 'string')})
    rows = pods20(6)
    tasks.append({'id': 'running-12', 'image': table(rows), 'kind': 'count', 'truth': 12,
                  'working_truth': [r[0] for r in rows if r[2] == 'Running'],
                  'q': 'How many pods in this kubectl output have STATUS Running?',
                  'steps_q': 'How many pods in this kubectl output have STATUS Running? First list the name of each '
                             'Running pod, then the total.',
                  'steps': ('running_pods', 'string')})
    shot = table(PODS12)
    for s in (1.0, 0.75):
        tasks.append({'id': f'read-x{s}', 'kind': 'read', 'truth': PODS12, 'q': POD_PROMPT,
                      'image': shot.resize((round(shot.width * s), round(shot.height * s)), Image.LANCZOS)})
    return tasks, rows


def read_score(content):
    try:
        got = json.loads(content)['pods']
    except (ValueError, KeyError, TypeError):
        return None
    names = {p.get('name') for p in got}
    truth = {r[0] for r in PODS12}
    by_name = {r[0]: r for r in PODS12}
    by_pos = list(PODS12)
    status_ok = sum(1 for i, p in enumerate(got[:12]) if p.get('status') == by_pos[i][2])
    return {'rows': len(got), 'names': len(names & truth), 'status_by_row': status_ok,
            'invented': sorted(n for n in names - truth if n)[:4],
            'restarts_ok': sum(1 for p in got if p.get('name') in by_name and p.get('restarts') == by_name[p['name']][3])}


def b64(im):
    b = io.BytesIO()
    im.save(b, 'PNG')
    return base64.b64encode(b.getvalue()).decode()


def ask(model, task, mode):
    if task['kind'] == 'read':
        prompt, schema = task['q'], POD_SCHEMA
    elif mode == 'steps':
        prompt, schema = task['steps_q'], count_schema(*task['steps'])
    else:
        prompt, schema = task['q'], count_schema()
    think = mode == 'think'
    options = {'temperature': 0, 'seed': 42, 'num_ctx': NUM_CTX, 'num_predict': THINK_BUDGET if think else 2000}
    t = time.perf_counter()
    r = http.post('/api/chat', json={'model': model, 'stream': False, 'think': think, 'format': schema, 'options': options,
                                     'messages': [{'role': 'user', 'content': prompt, 'images': [task['b64']]}]})
    wall = time.perf_counter() - t
    if r.status_code != 200:
        return {'error': f'HTTP {r.status_code}: {r.text[:200]}', 'wall_s': round(wall, 1)}
    j = r.json()
    m = j['message']
    return {'content': m.get('content', ''), 'thinking': m.get('thinking') or '', 'done_reason': j.get('done_reason'),
            'output_tokens': j.get('eval_count'), 'prompt_tokens': j.get('prompt_eval_count'), 'wall_s': round(wall, 1)}


def judge(task, r):
    if 'error' in r:
        return 'error', r['error']
    if not r['content'].strip():
        return 'no answer', f'hit the {THINK_BUDGET}-token limit' if r['done_reason'] == 'length' else r['done_reason']
    if task['kind'] == 'read':
        s = read_score(r['content'])
        if s is None:
            return 'not JSON', ''
        return f'{s["names"]}/12 names', (f'{s["rows"]} rows, statuses by row {s["status_by_row"]}/12, '
                                          f'restarts {s["restarts_ok"]}, invented {s["invented"] or "-"}')
    try:
        j = json.loads(r['content'])
    except ValueError:
        return 'not JSON', r['content'][:60]
    n = j.get('count')
    working = next((v for k, v in j.items() if k != 'count'), None)
    detail = ''
    if working is not None:
        detail = f'working: {json.dumps(working)[:120]}'
        if isinstance(working, list) and working and all(isinstance(x, int) for x in working):
            detail += f' (sum {sum(working)})'
        elif isinstance(working, list):
            detail += f' ({len(working)} entries)'
    return ('✅ ' if n == task['truth'] else '❌ ') + str(n), detail


# ---------------------------------------------------------------- run
outdir.mkdir(parents=True, exist_ok=True)
tasks, rows20 = build_tasks()
for t in tasks:
    t['image'].save(outdir / f'{t["id"]}.png')
    t['b64'] = b64(t['image'])
(outdir / 'truth.json').write_text(json.dumps({t['id']: {'truth': t['truth'] if t['kind'] == 'count' else 'PODS12',
                                                        'working': t.get('working_truth')} for t in tasks}, indent=1),
                                   encoding='utf-8')
if '--images' in sys.argv:
    print(f'Images in {outdir}')
    sys.exit()

say(f'# vision_thinking ({", ".join(MODELS)}; temperature 0, seed 42; num_ctx {NUM_CTX}; think budget '
    f'{THINK_BUDGET}) {stamp}\n')
say('Truth: ' + ', '.join(f'{t["id"]} = {t["truth"] if t["kind"] == "count" else "12 pods"}' for t in tasks) + '\n')
results = {}
for model in MODELS:
    for other in ['qwen3.5:4b', 'qwen3.5:9b']:
        subprocess.run(['ollama', 'stop', other], capture_output=True)
    for t in tasks:
        for mode in (['off', 'think'] if t['kind'] == 'read' else ['off', 'steps', 'think']):
            r = ask(model, t, mode)
            verdict, detail = judge(t, r)
            results[(model, t['id'], mode)] = {**r, 'verdict': verdict, 'detail': detail}
            print(f'  {model} {t["id"]:11} {mode:5} {verdict:14} {r.get("output_tokens")} tok {r.get("wall_s")} s  '
                  f'{detail[:90]}', flush=True)

for model in MODELS:
    say(f'\n## {model}\n')
    say('| task | truth | off | steps | think | think output tokens | think wall s |')
    say('|---|---|---|---|---|---:|---:|')
    for t in tasks:
        cell = lambda mode: results.get((model, t['id'], mode), {}).get('verdict', '')
        th = results[(model, t['id'], 'think')]
        truth = t['truth'] if t['kind'] == 'count' else '12 names'
        say(f'| {t["id"]} | {truth} | {cell("off")} | {cell("steps") or "-"} | {cell("think")} | '
            f'{th.get("output_tokens")} | {th.get("wall_s")} |')
    say('\nDetails (working written in "steps"; reading scores):\n')
    for t in tasks:
        for mode in ('off', 'steps', 'think'):
            r = results.get((model, t['id'], mode))
            if r and r['detail']:
                say(f'* {t["id"]} {mode}: {r["detail"]}')

say('\n```\n' + subprocess.run(['ollama', 'ps'], capture_output=True, text=True).stdout.strip() + '\n```')
for m in MODELS:
    subprocess.run(['ollama', 'stop', m], capture_output=True)
(outdir / 'answers.json').write_text(json.dumps({f'{m} {t} {mode}': r for (m, t, mode), r in results.items()},
                                                indent=1, ensure_ascii=False), encoding='utf-8')
(outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(f'\nSaved {outdir}')
