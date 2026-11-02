"""Lab 07 Step 2. Reading a dashboard and an architecture diagram, scored against ground truth; 4b vs 9b.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\vision_dashboard.py            (≈4 minutes)
                                  .\\.venv\\Scripts\\python scripts\\vision_dashboard.py --images   (draw only)

Two synthetic images (1280x720, nothing private), each with ground truth known from how it was drawn:
  dashboard: a Grafana-style page: 4 stat panels, a CPU time series (3 pods, an 80% threshold), a memory
             gauge, a table of 5xx errors, an alert panel
  diagram:   a Kubernetes topology: 4 namespaces, ingress -> Service -> 3 pods, a failing link to the database
19 questions (numbers read off panels, a threshold crossing and its time, a trend, counting, arrows, and two
questions whose answer isn't in the image). Each model answers each question in 3 ways:
  image  the full image
  crop   the CPU chart only, cropped and upscaled x2 (chart questions only): Step 1 said resolution is the limit
  text   the same facts as text (JSON / an edge list): separates "can't see it" from "can't reason about it"
Models: qwen3.5:4b, then qwen3.5:9b (the 4b is stopped first: both together would exceed the 8.5 GB ceiling).
Think off, temperature 0, seed 42. One question per request, answer in a JSON schema.
"""
import base64
import io
import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

HOST = 'http://127.0.0.1:11434'
MODELS = ['qwen3.5:4b', 'qwen3.5:9b']
OPTIONS = {'temperature': 0, 'seed': 42, 'num_predict': 200}
W, H = 1280, 720
http = httpx.Client(base_url=HOST, timeout=600)
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'vision-dashboard-{stamp}'
report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


def font(size, bold=False):
    return ImageFont.truetype('arialbd.ttf' if bold else 'arial.ttf', size)


# ---------------------------------------------------------------- the dashboard and its data
rng = random.Random(7)
MINUTES = list(range(61))                                     # 14:00 .. 15:00


def clock(m):
    return f'{14 + m // 60}:{m % 60:02d}'


def api_b(m):
    if m <= 31:
        return 45 + rng.uniform(-3, 3)
    if m <= 36:
        return 50 + (m - 31) * 8                              # 58, 66, 74, 82 (14:35), 90
    if m <= 42:
        return 90 + (m - 36)                                  # peak 96 at 14:42
    return 90 + rng.uniform(-2, 2)


SERIES = {                                                    # name -> (colour, values per minute)
    'payments-api-a': ((115, 191, 105), [40 + rng.uniform(-3, 3) for _ in MINUTES]),
    'payments-api-b': ((242, 204, 12), [api_b(m) for m in MINUTES]),
    'payments-worker': ((87, 148, 242), [20 + 35 * m / 60 + rng.uniform(-1.5, 1.5) for m in MINUTES]),
}
THRESHOLD = 80
B = SERIES['payments-api-b'][1]
CROSS = next(m for m in MINUTES if B[m] > THRESHOLD)
PEAK = max(B)
STATS = [('Error rate', '4.7%', (196, 22, 42)), ('p95 latency', '812 ms', (224, 180, 0)),
         ('Pods ready', '9/12', (224, 180, 0)), ('Requests/s', '1,240', (55, 135, 45))]
ERRORS = [('/v1/charge', 1482, '940 ms'), ('/v1/refund', 311, '620 ms'), ('/v1/status', 12, '45 ms')]
MEMORY = 87
ALERT = ('PaymentsHighErrorRate', 'firing since 14:41')
CHART_BOX = (20, 190, 860, 500)                               # x0, y0, x1, y1 of the CPU panel


def panel(d, box, title):
    d.rounded_rectangle(box, 4, fill=(24, 27, 31), outline=(44, 48, 54))
    d.text((box[0] + 12, box[1] + 8), title, fill=(204, 204, 220), font=font(15, True))


def dashboard():
    im = Image.new('RGB', (W, H), (17, 18, 23))
    d = ImageDraw.Draw(im)
    d.text((20, 12), 'payments / production', fill=(216, 217, 218), font=font(20, True))
    d.text((1060, 15), 'Last 1 hour', fill=(160, 160, 170), font=font(15))
    for i, (title, value, colour) in enumerate(STATS):          # row 1: stat panels
        x0 = 20 + i * 312
        d.rounded_rectangle((x0, 50, x0 + 300, 175), 4, fill=colour)
        d.text((x0 + 12, 58), title, fill='white', font=font(15, True))
        d.text((x0 + 150, 125), value, fill='white', font=font(44, True), anchor='mm')
    x0, y0, x1, y1 = CHART_BOX                                   # row 2 left: time series
    panel(d, CHART_BOX, 'CPU usage by pod (%)')
    px0, py0, px1, py1 = x0 + 50, y0 + 40, x1 - 20, y1 - 60      # plot area
    ys = lambda v: py1 - (py1 - py0) * v / 100
    xs = lambda m: px0 + (px1 - px0) * m / 60
    for v in (0, 25, 50, 75, 100):
        d.line((px0, ys(v), px1, ys(v)), fill=(44, 48, 54))
        d.text((px0 - 8, ys(v)), str(v), fill=(160, 160, 170), font=font(12), anchor='rm')
    for m in range(0, 61, 10):
        d.text((xs(m), py1 + 6), clock(m), fill=(160, 160, 170), font=font(12), anchor='ma')
    for xa in range(px0, px1, 14):                               # dashed threshold
        d.line((xa, ys(THRESHOLD), min(xa + 7, px1), ys(THRESHOLD)), fill=(242, 73, 92), width=2)
    for name, (colour, vals) in SERIES.items():
        d.line([(xs(m), ys(vals[m])) for m in MINUTES], fill=colour, width=2)
    lx = px0
    for name, (colour, _) in SERIES.items():                     # legend
        d.rectangle((lx, y1 - 28, lx + 14, y1 - 16), fill=colour)
        d.text((lx + 20, y1 - 30), name, fill=(204, 204, 220), font=font(13))
        lx += 200
    gbox = (880, 190, 1260, 500)                                 # row 2 right: gauge
    panel(d, gbox, 'Memory (payments-api)')
    cx, cy, r = 1070, 380, 120
    d.arc((cx - r, cy - r, cx + r, cy + r), 180, 360, fill=(44, 48, 54), width=26)
    d.arc((cx - r, cy - r, cx + r, cy + r), 180, 180 + 180 * MEMORY / 100, fill=(255, 120, 10), width=26)
    d.text((cx, cy - 20), f'{MEMORY}%', fill=(255, 120, 10), font=font(40, True), anchor='mm')
    tbox = (20, 515, 860, 705)                                   # row 3 left: table
    panel(d, tbox, 'Top 5xx errors by endpoint (last 1h)')
    cols = (40, 360, 560)
    for c, head in zip(cols, ('Endpoint', '5xx count', 'p95')):
        d.text((c, 550), head, fill=(160, 160, 170), font=font(14, True))
    for i, (ep, n, p95) in enumerate(ERRORS):
        for c, val in zip(cols, (ep, f'{n:,}', p95)):
            d.text((c, 580 + 36 * i), val, fill=(216, 217, 218), font=font(16))
    abox = (880, 515, 1260, 705)                                 # row 3 right: alerts
    panel(d, abox, 'Active alerts')
    d.ellipse((900, 560, 914, 574), fill=(196, 22, 42))
    d.text((924, 557), ALERT[0], fill=(216, 217, 218), font=font(16, True))
    d.text((924, 584), ALERT[1], fill=(160, 160, 170), font=font(14))
    return im


def dashboard_text():
    data = {'title': 'payments / production, last 1 hour',
            'stat_panels': [{'title': t, 'value': v, 'colour': {(196, 22, 42): 'red', (224, 180, 0): 'yellow',
                                                                (55, 135, 45): 'green'}[c]} for t, v, c in STATS],
            'cpu_usage_by_pod_percent': {'threshold': THRESHOLD, 'minutes': f'{clock(0)}..{clock(60)}, one value per minute',
                                         'series': {n: [round(v) for v in vals] for n, (_, vals) in SERIES.items()}},
            'memory_payments_api_percent': MEMORY,
            'top_5xx_errors_last_1h': [{'endpoint': e, 'count': n, 'p95': p} for e, n, p in ERRORS],
            'active_alerts': [{'name': ALERT[0], 'state': ALERT[1]}]}
    return json.dumps(data)


# ---------------------------------------------------------------- the diagram and its edges
NODES = {   # name -> (box, label lines)
    'ingress': ((70, 300, 250, 380), ['ingress-nginx', 'Ingress']),
    'svc': ((360, 300, 560, 380), ['payments-api', 'Service']),
    'pod1': ((640, 150, 820, 200), ['payments-api', 'Pod']),
    'pod2': ((640, 315, 820, 365), ['payments-api', 'Pod']),
    'pod3': ((640, 480, 820, 530), ['payments-api', 'Pod']),
    'db': ((1000, 110, 1210, 200), ['payments-db', 'StatefulSet · PostgreSQL']),
    'redis': ((1000, 260, 1210, 330), ['redis-cache', 'StatefulSet']),
    'kafka': ((1000, 470, 1210, 550), ['kafka', 'topic: payments-events']),
    'worker': ((640, 610, 820, 670), ['payments-worker', 'Deployment']),
}
NAMESPACES = [('namespace: edge', (40, 250, 280, 430)), ('namespace: payments', (330, 110, 860, 700)),
              ('namespace: data', (970, 60, 1240, 360)), ('namespace: messaging', (970, 420, 1240, 600))]
EDGES = [   # from, to, label, failing
    ('ingress', 'svc', 'HTTPS 443', False), ('svc', 'pod1', '', False), ('svc', 'pod2', '', False),
    ('svc', 'pod3', '', False), ('pod1', 'db', '5432  connection refused', True), ('pod2', 'redis', '6379', False),
    ('pod3', 'kafka', 'produce', False), ('kafka', 'worker', 'consume', False),
]
ANCHORS = {('kafka', 'worker'): ((1060, 550), (820, 640))}   # start at kafka's bottom, not on the produce arrow


def arrow(d, a, b, colour, dashed=False, width=3):
    (ax, ay), (bx, by) = a, b
    n = math.hypot(bx - ax, by - ay)
    if dashed:
        for t in range(0, int(n), 18):
            t2 = min(t + 10, n)
            d.line((ax + (bx - ax) * t / n, ay + (by - ay) * t / n, ax + (bx - ax) * t2 / n, ay + (by - ay) * t2 / n),
                   fill=colour, width=width)
    else:
        d.line((ax, ay, bx, by), fill=colour, width=width)
    ang = math.atan2(by - ay, bx - ax)
    for s in (2.6, -2.6):
        d.line((bx, by, bx + 14 * math.cos(ang + s), by + 14 * math.sin(ang + s)), fill=colour, width=width)


def side(box, toward):
    """Point on the box edge facing another box (left/right edges, or top/bottom if stacked)."""
    x0, y0, x1, y1 = box
    tx0, ty0, tx1, ty1 = toward
    if tx0 >= x1:
        return x1, (y0 + y1) / 2
    if tx1 <= x0:
        return x0, (y0 + y1) / 2
    return ((x0 + x1) / 2, y1) if ty0 >= y1 else ((x0 + x1) / 2, y0)


def diagram():
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    d.text((40, 15), 'payments — production topology', fill=(30, 30, 30), font=font(22, True))
    for label, box in NAMESPACES:
        d.rounded_rectangle(box, 12, outline=(150, 150, 170), width=2)
        d.text((box[0] + 12, box[1] + 8), label, fill=(90, 90, 120), font=font(15, True))
    for name, (box, lines) in NODES.items():
        d.rounded_rectangle(box, 8, fill=(232, 240, 254), outline=(60, 100, 200), width=2)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        d.text((cx, cy - 10), lines[0], fill=(20, 20, 40), font=font(16, True), anchor='mm')
        d.text((cx, cy + 12), lines[1], fill=(70, 70, 90), font=font(13), anchor='mm')
    for a, b, label, failing in EDGES:
        pa, pb = ANCHORS.get((a, b)) or (side(NODES[a][0], NODES[b][0]), side(NODES[b][0], NODES[a][0]))
        colour = (210, 30, 40) if failing else (60, 60, 60)
        arrow(d, pa, pb, colour, dashed=failing)
        if label:
            f = 0.4 if failing else 0.5
            mx, my = pa[0] + (pb[0] - pa[0]) * f, pa[1] + (pb[1] - pa[1]) * f
            d.text((mx, my - (26 if failing else 14)), label, fill=colour, font=font(14, True), anchor='mm')
    return im


def diagram_text():
    names = {'ingress': 'ingress-nginx (Ingress)', 'svc': 'payments-api (Service)', 'pod1': 'payments-api Pod 1',
             'pod2': 'payments-api Pod 2', 'pod3': 'payments-api Pod 3', 'db': 'payments-db (StatefulSet, PostgreSQL)',
             'redis': 'redis-cache (StatefulSet)', 'kafka': 'kafka (topic: payments-events)',
             'worker': 'payments-worker (Deployment)'}
    ns = {'edge': ['ingress'], 'payments': ['svc', 'pod1', 'pod2', 'pod3', 'worker'], 'data': ['db', 'redis'],
          'messaging': ['kafka']}
    lines = ['Topology "payments — production". Namespaces:']
    lines += [f'- {n}: ' + ', '.join(names[x] for x in xs) for n, xs in ns.items()]
    lines.append('Connections (arrows):')
    lines += [f'- {names[a]} -> {names[b]}' + (f' [{label}]' if label else '') + (' (drawn red, dashed)' if f else '')
              for a, b, label, f in EDGES]
    return '\n'.join(lines)


# ---------------------------------------------------------------- questions and scoring
# kind: num (value, tolerance) | time (minute, tolerance) | text [accepted substrings] | set [[aliases], ...] | none
Q = [
    ('d1', 'dashboard', 'What is the error rate shown on the dashboard?', 'num', (4.7, 0.05)),
    ('d2', 'dashboard', 'What is the p95 latency in milliseconds?', 'num', (812, 0)),
    ('d3', 'dashboard', 'How many pods are ready, and out of how many? Answer as ready/total.', 'text', ['9/12']),
    ('d4', 'dashboard', 'Which stat panel at the top is coloured red? Give its title.', 'text', ['error rate']),
    ('d5', 'dashboard', 'In the CPU usage chart, which pod goes above the 80% threshold line? Give its name.',
     'text', ['payments-api-b']),
    ('d6', 'dashboard', 'In the CPU usage chart, at what time (HH:MM) does that pod first go above 80%?',
     'time', (CROSS, 3)),
    ('d7', 'dashboard', "What is that pod's highest CPU value in the chart, in percent?", 'num', (round(PEAK), 4)),
    ('d8', 'dashboard', 'In the CPU usage chart, is payments-worker rising, falling, or flat over the hour? '
     'Answer with one word.', 'text', ['rising']),
    ('d9', 'dashboard', 'What memory percentage does the gauge show?', 'num', (MEMORY, 0)),
    ('d10', 'dashboard', 'How many 5xx errors did /v1/refund have in the last hour?', 'num', (311, 0)),
    ('d11', 'dashboard', 'Since what time has the active alert been firing? Answer HH:MM.', 'time', (41, 0)),
    ('d12', 'dashboard', 'What disk usage percentage does the dashboard show?', 'none', None),
    ('t1', 'diagram', 'Which component does the ingress send traffic to?', 'text', ['payments-api']),
    ('t2', 'diagram', 'How many payments-api Pods are drawn?', 'num', (3, 0)),
    ('t3', 'diagram', 'One connection is drawn as failing. Which component is it trying to reach?', 'text',
     ['payments-db']),
    ('t4', 'diagram', 'Which namespace is redis-cache in? Give the namespace name only.', 'text', ['data']),
    ('t5', 'diagram', 'What does payments-worker consume from?', 'text', ['kafka', 'payments-events']),
    ('t6', 'diagram', 'Which components do the payments-api Pods connect to? List all, comma-separated.', 'set',
     [['payments-db'], ['redis-cache', 'redis'], ['kafka', 'payments-events']]),
    ('t7', 'diagram', 'Which PostgreSQL version does payments-db run?', 'none', None),
]
CHART_Q = {'d5', 'd6', 'd7', 'd8'}
SCHEMA = {'type': 'object', 'required': ['answer'], 'properties': {'answer': {'type': 'string'}}}
RULES = ('Answer the question using only the {src}. Be brief. If the information is not in the {src}, answer exactly: '
         'not shown')
NOT_SHOWN = re.compile(r'not (shown|visible|present|available|displayed|provided)|no disk|cannot|n/a', re.I)


def correct(kind, truth, answer):
    a = answer.strip().lower()
    if kind == 'none':
        return bool(NOT_SHOWN.search(a))
    if NOT_SHOWN.search(a):
        return False
    if kind == 'num':
        nums = [float(x.replace(',', '')) for x in re.findall(r'\d[\d,]*\.?\d*', a)]
        return bool(nums) and abs(nums[0] - truth[0]) <= truth[1]
    if kind == 'time':
        t = re.search(r'(\d{1,2}):(\d{2})', a)
        return bool(t) and int(t.group(1)) == 14 and abs(int(t.group(2)) - truth[0]) <= truth[1]
    if kind == 'text':
        return any(s in a for s in truth)
    if kind == 'set':
        return all(any(s in a for s in aliases) for aliases in truth) and 'ingress' not in a and 'worker' not in a


def png(im, name):
    im.save(outdir / f'{name}.png')
    b = io.BytesIO()
    im.save(b, 'PNG')
    return base64.b64encode(b.getvalue()).decode()


def ask(model, question, image=None, text=None):
    if text is not None:
        content = RULES.format(src='data below') + f'\n\nQuestion: {question}\n\nData:\n{text}'
    else:
        content = RULES.format(src='image') + f'\n\nQuestion: {question}'
    msg = {'role': 'user', 'content': content, **({'images': [image]} if image else {})}
    r = http.post('/api/chat', json={'model': model, 'messages': [msg], 'stream': False, 'think': False,
                                     'format': SCHEMA, 'options': OPTIONS})
    if r.status_code != 200:
        return {'answer': f'HTTP {r.status_code}: {r.json().get("error")}', 'prompt_tokens': None}
    j = r.json()
    try:
        answer = json.loads(j['message']['content'])['answer']
    except (ValueError, KeyError, TypeError):
        answer = j['message']['content']
    return {'answer': answer, 'prompt_tokens': j.get('prompt_eval_count'),
            'prompt_ms': round(j.get('prompt_eval_duration', 0) / 1e6)}


def vram():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


# ---------------------------------------------------------------- run
outdir.mkdir(parents=True, exist_ok=True)
dash, diag = dashboard(), diagram()
x0, y0, x1, y1 = CHART_BOX
crop = dash.crop(CHART_BOX).resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.LANCZOS)
images = {'dashboard': png(dash, 'dashboard'), 'diagram': png(diag, 'diagram'), 'crop': png(crop, 'dashboard-cpu-crop-x2')}
texts = {'dashboard': dashboard_text(), 'diagram': diagram_text()}
(outdir / 'dashboard.json').write_text(texts['dashboard'], encoding='utf-8')
(outdir / 'diagram.txt').write_text(texts['diagram'], encoding='utf-8')
if '--images' in sys.argv:
    print(f'Images in {outdir}  (threshold crossing {clock(CROSS)}, peak {PEAK:.0f}%)')
    sys.exit()

say(f'# vision_dashboard ({", ".join(MODELS)}, think off, temperature 0, seed 42) {stamp}\n')
say(f'Ground truth from the drawing code: payments-api-b first above 80% at {clock(CROSS)}, peak {PEAK:.0f}%.\n')
results, cost, vr = {}, {}, {}
for model in MODELS:
    for other in MODELS:
        subprocess.run(['ollama', 'stop', other], capture_output=True)
    time.sleep(3)
    vr[model] = {'before': vram()}
    for qid, img, question, kind, truth in Q:
        modes = [('image', images[img], None)] + ([('crop', images['crop'], None)] if qid in CHART_Q else []) + \
                [('text', None, texts[img])]
        for mode, image, text in modes:
            r = ask(model, question, image, text)
            r['ok'] = correct(kind, truth, r['answer'])
            results[(model, qid, mode)] = r
            cost.setdefault((model, img if mode != 'crop' else 'crop', mode), []).append(r)
            print(f'  {model} {qid} {mode:5} {"OK " if r["ok"] else "-- "} {str(r["answer"])[:70]!r}', flush=True)
    vr[model]['peak (after all questions)'] = vram()

say('\n## Answers (✅ correct, ❌ wrong)\n')
cols = [(m, mode) for m in MODELS for mode in ('image', 'crop', 'text')]
say('| Q | truth | ' + ' | '.join(f'{m.split(":")[1]} {mode}' for m, mode in cols) + ' |')
say('|---|---|' + '---|' * len(cols))
for qid, img, question, kind, truth in Q:
    t = {'num': lambda: f'{truth[0]}', 'time': lambda: clock(truth[0]), 'text': lambda: truth[0],
         'set': lambda: ', '.join(a[0] for a in truth), 'none': lambda: 'not shown'}[kind]()
    cells = []
    for m, mode in cols:
        r = results.get((m, qid, mode))
        cells.append('' if r is None else f'{"✅" if r["ok"] else "❌"} {str(r["answer"]).strip()[:40]}'.replace('|', '/'))
    say(f'| {qid} | {t} | ' + ' | '.join(cells) + ' |')

say('\n## Score\n')
say('| model | dashboard image /12 | CPU crop /4 (same 4 Qs on full image) | dashboard text /12 | diagram image /7 '
    '| diagram text /7 |')
say('|---|---:|---:|---:|---:|---:|')
for m in MODELS:
    s = lambda ids, mode: sum(results[(m, q, mode)]['ok'] for q in ids)
    dq = [q[0] for q in Q if q[1] == 'dashboard']
    tq = [q[0] for q in Q if q[1] == 'diagram']
    say(f'| {m} | {s(dq, "image")} | {s(CHART_Q, "crop")} ({s(CHART_Q, "image")}) | {s(dq, "text")} | '
        f'{s(tq, "image")} | {s(tq, "text")} |')

say('\n## Cost (prompt tokens per request, median; prompt ms median)\n')
for (m, src, mode), rs in sorted(cost.items()):
    toks = sorted(r['prompt_tokens'] for r in rs if r['prompt_tokens'])
    ms = sorted(r['prompt_ms'] for r in rs if r.get('prompt_ms') is not None)
    if toks:
        say(f'* {m} {src} {mode}: {toks[len(toks) // 2]} tokens, {ms[len(ms) // 2]} ms')
say('\n## VRAM (nvidia-smi MiB, whole GPU)\n')
for m, v in vr.items():
    say(f'* {m}: ' + ', '.join(f'{k} {val}' for k, val in v.items()))
say('\n```\n' + subprocess.run(['ollama', 'ps'], capture_output=True, text=True).stdout.strip() + '\n```')

for m in MODELS:
    subprocess.run(['ollama', 'stop', m], capture_output=True)
(outdir / 'answers.json').write_text(json.dumps({f'{m} {q} {mode}': r for (m, q, mode), r in results.items()},
                                                indent=1), encoding='utf-8')
(outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(f'\nSaved {outdir}')
