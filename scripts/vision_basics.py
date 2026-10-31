"""Lab 07 Step 1. How an image becomes tokens, and what resolution buys: cost, readability, controls, VRAM.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\vision_basics.py      (≈3 minutes)

Part 1: image tokens vs image size: measured (prompt_eval_count with the image minus without it) against a
        predicted count (32x32-pixel tokens, resized into [8,192, 4,194,304] pixels); over-context images
Part 2: readability: a synthetic `kubectl get pods` screenshot at 6 scales (and the same table as text),
        read into a JSON schema and scored field by field against the ground truth
Part 3: controls: the same question with no image, and with a blank image. Does the model invent pods?
Part 4: VRAM: nvidia-smi before load, after a text-only load, after images; ollama ps
All images are generated here (synthetic, nothing private) and saved next to the report in results/raw/.
"""
import base64
import io
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

HOST = 'http://127.0.0.1:11434'
MODEL = 'qwen3.5:4b'
# From server.log at load (clip_model_loader / load_hparams): patch_size 16, n_merge 2,
# image_min_pixels 8192, image_max_pixels 4194304
TOKEN_PX = 32
MIN_PX, MAX_PX = 8192, 4194304
OPTIONS = {'temperature': 0, 'seed': 42}
http = httpx.Client(base_url=HOST, timeout=600)
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
outdir = Path('results/raw') / f'vision-basics-{stamp}'
outdir.mkdir(parents=True, exist_ok=True)
report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


def vram():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None


def predicted_tokens(w, h):
    """Qwen2-VL-style resize: round each side to a multiple of 32, then scale into [MIN_PX, MAX_PX] keeping
    the aspect ratio. Fitted to the probe sizes; Part 1 checks it, it is not read from llama.cpp's source."""
    f = TOKEN_PX
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > MAX_PX:
        beta = math.sqrt(h * w / MAX_PX)
        hb, wb = math.floor(h / beta / f) * f, math.floor(w / beta / f) * f
    elif hb * wb < MIN_PX:
        beta = math.sqrt(MIN_PX / (h * w))
        hb, wb = math.ceil(h * beta / f) * f, math.ceil(w * beta / f) * f
    return (wb // f) * (hb // f)


def png(im, name=None):
    if name:
        im.save(outdir / f'{name}.png')
    b = io.BytesIO()
    im.save(b, 'PNG')
    return base64.b64encode(b.getvalue()).decode()


def chat(prompt, images=None, fmt=None, num_ctx=None, num_predict=64):
    msg = {'role': 'user', 'content': prompt}
    if images:
        msg['images'] = images
    options = {**OPTIONS, 'num_predict': num_predict, **({'num_ctx': num_ctx} if num_ctx else {})}
    body = {'model': MODEL, 'messages': [msg], 'stream': False, 'think': False, 'options': options}
    if fmt:
        body['format'] = fmt
    t = time.perf_counter()
    r = http.post('/api/chat', json=body)
    wall = time.perf_counter() - t
    if r.status_code != 200:
        return {'status': r.status_code, 'error': r.json().get('error'), 'wall_s': round(wall, 2)}
    j = r.json()
    return {'status': 200, 'prompt_tokens': j.get('prompt_eval_count'), 'output_tokens': j.get('eval_count'),
            'prompt_ms': round(j.get('prompt_eval_duration', 0) / 1e6), 'load_ms': round(j.get('load_duration', 0) / 1e6),
            'wall_s': round(wall, 2), 'text': j['message']['content']}


# ---------------------------------------------------------------- test images
def label_card(w, h):
    """White card with one large label: legible at every size, so Part 1 measures cost, not reading."""
    im = Image.new('RGB', (w, h), 'white')
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', max(10, min(w, h) // 6))
    d.text((w // 10, h // 3), 'NODE-7', fill='black', font=font)
    return im


# Ground truth for the kubectl screenshot: near-identical names, every common status, kubectl's restart format
PODS = [
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


def kubectl_text():
    rows = [('NAME', 'READY', 'STATUS', 'RESTARTS', 'AGE')] + [(n, r, s, rs, a) for n, r, s, _, rs, a in PODS]
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    return '\n'.join('   '.join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip() for row in rows)


def kubectl_screenshot():
    """A terminal window: dark background, 16 px Consolas, prompt line + the table. ≈1,000 px wide."""
    text = '$ kubectl get pods -n payments\n' + kubectl_text()
    font = ImageFont.truetype('consola.ttf', 16)
    lines = text.split('\n')
    w = int(max(font.getlength(line) for line in lines)) + 40
    h = 22 * len(lines) + 40
    im = Image.new('RGB', (w, h), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((20, 20 + 22 * i), line, fill=(220, 220, 220), font=font)
    return im


POD_SCHEMA = {'type': 'object', 'required': ['pods'], 'properties': {'pods': {'type': 'array', 'items': {
    'type': 'object', 'required': ['name', 'ready', 'status', 'restarts'],
    'properties': {'name': {'type': 'string'}, 'ready': {'type': 'string'}, 'status': {'type': 'string'},
                   'restarts': {'type': 'integer'}}}}}}
POD_PROMPT = ('List every pod in this kubectl output: name, READY, STATUS and the RESTARTS count as an integer. '
              'Copy names exactly. Only include pods that are shown.')


def score(text):
    try:
        got = json.loads(text)['pods']
    except (ValueError, KeyError, TypeError):
        return {'parsed': False}
    truth = {n: (r, s, rc) for n, r, s, rc, _, _ in PODS}
    names = [p.get('name', '') for p in got]
    matched = [p for p in got if p.get('name') in truth]
    return {'parsed': True, 'rows': len(got), 'names_exact': len(set(names) & set(truth)),
            'invented': sorted(set(names) - set(truth)),
            'ready_ok': sum(p.get('ready') == truth[p['name']][0] for p in matched),
            'status_ok': sum(p.get('status') == truth[p['name']][1] for p in matched),
            'restarts_ok': sum(p.get('restarts') == truth[p['name']][2] for p in matched)}


# ---------------------------------------------------------------- run
say(f'# vision_basics ({MODEL}, think off, temperature 0, seed 42) {stamp}\n')
subprocess.run(['ollama', 'stop', MODEL], capture_output=True)
time.sleep(3)
v = {'before load': vram()}

say('## Part 4a: load (text only) and the first image\n')
LABEL_Q = 'What text is in the image? Answer with the text only.'
base = chat(LABEL_Q)
v['after text-only load'] = vram()
say(f'text-only, cold: load {base["load_ms"]} ms, prompt {base["prompt_tokens"]} tokens, answer: {base["text"]!r}')
first = chat(LABEL_Q, [png(label_card(256, 256))])
second = chat(LABEL_Q, [png(label_card(256, 256))])
say(f'first image after load (256x256): prompt {first["prompt_ms"]} ms, wall {first["wall_s"]} s; '
    f'same image again: prompt {second["prompt_ms"]} ms, wall {second["wall_s"]} s')
v['after first image'] = vram()

say('\n## Part 1: image tokens vs size (label card, prompt without image = '
    f'{base["prompt_tokens"]} tokens)\n')
say('| size | pixels | predicted tokens | measured (image - text) | prompt ms | answer |')
say('|---|---:|---:|---:|---:|---|')
SIZES = [(32, 32), (64, 64), (128, 128), (256, 256), (500, 500), (512, 512), (1000, 1000), (1024, 1024),
         (1500, 1000), (2048, 512), (512, 2048), (1800, 1800), (8000, 1000), (2048, 2048), (4096, 4096)]
for w, h in SIZES:
    r = chat(LABEL_Q, [png(label_card(w, h))])
    pred = predicted_tokens(w, h)
    if r['status'] != 200:
        say(f'| {w}x{h} | {w * h:,} | {pred:,} | HTTP {r["status"]}: {r["error"]} | | |')
        continue
    say(f'| {w}x{h} | {w * h:,} | {pred:,} | {r["prompt_tokens"] - base["prompt_tokens"]:,} | {r["prompt_ms"]} '
        f'| {r["text"].strip()[:30]!r} |')
v['after largest image (4K ctx)'] = vram()
say('\nThe 2 extra tokens per image are the vision start/end markers.')
say('\nSame two large images at num_ctx 8192 (reloads the model):')
for w, h in [(2048, 2048), (4096, 4096)]:
    r = chat(LABEL_Q, [png(label_card(w, h))], num_ctx=8192)
    say(f'* {w}x{h}: ' + (f'measured {r["prompt_tokens"] - base["prompt_tokens"]:,} image tokens, '
                          f'prompt {r["prompt_ms"]} ms' if r['status'] == 200 else f'HTTP {r["status"]}: {r["error"]}'))
v['at 8K ctx after a 4,096-token image'] = vram()

say('\n## Part 2: readability vs resolution (kubectl get pods, 12 pods, JSON schema)\n')
shot = kubectl_screenshot()
say(f'Native screenshot: {shot.width}x{shot.height}, 16 px font (≈22 px per line). Saved as kubectl-*.png.\n')
say('| variant | size | image tokens | prompt ms | rows | names exact /12 | invented | READY ok | STATUS ok '
    '| RESTARTS ok |')
say('|---|---|---:|---:|---:|---:|---|---:|---:|---:|')
variants = [(f'image x{s}', s) for s in (2.0, 1.5, 1.0, 0.75, 0.5, 0.35)] + [('text (no image)', None)]
for name, s in variants:
    if s is None:
        r = chat(POD_PROMPT + '\n\n' + kubectl_text(), fmt=POD_SCHEMA, num_predict=1500)
        size, itok = '-', '-'
    else:
        im = shot.resize((round(shot.width * s), round(shot.height * s)), Image.LANCZOS)
        r = chat(POD_PROMPT, [png(im, f'kubectl-x{s}')], fmt=POD_SCHEMA, num_predict=1500)
        size, itok = f'{im.width}x{im.height}', f'{predicted_tokens(im.width, im.height):,} (pred.)'
    if r['status'] != 200:
        say(f'| {name} | {size} | {itok} | HTTP {r["status"]} | {r["error"]} | | | | | |')
        continue
    sc = score(r['text'])
    (outdir / f'answer-{name.replace(" ", "_")}.json').write_text(r['text'], encoding='utf-8')
    if not sc['parsed']:
        say(f'| {name} | {size} | {itok} | {r["prompt_ms"]} | not JSON | | | | | |')
        continue
    say(f'| {name} | {size} | {itok} | {r["prompt_ms"]} | {sc["rows"]} | {sc["names_exact"]} | '
        f'{", ".join(sc["invented"][:3]) or "-"}{" …" if len(sc["invented"]) > 3 else ""} | {sc["ready_ok"]} | '
        f'{sc["status_ok"]} | {sc["restarts_ok"]} |')
    say(f'<!-- {name}: prompt_tokens {r["prompt_tokens"]}, output_tokens {r["output_tokens"]}, wall {r["wall_s"]} s -->')

say('\n## Part 3: controls: what if there is no picture?\n')
CTRL_Q = 'Which pods in this screenshot are in CrashLoopBackOff? Give their names.'
blank = Image.new('RGB', shot.size, (30, 30, 30))
for name, imgs in [('no image', None), ('blank terminal-coloured image', [png(blank, 'blank')]),
                   ('the real screenshot', [png(shot, 'kubectl-native')])]:
    r = chat(CTRL_Q, imgs, num_predict=200)
    say(f'**{name}:** {r.get("text", r.get("error"))!r}\n')
v['end'] = vram()

say('## Part 4b: VRAM (nvidia-smi, MiB, whole GPU incl. the desktop)\n')
for k, val in v.items():
    say(f'* {k}: {val}')
say('\n```\n' + subprocess.run(['ollama', 'ps'], capture_output=True, text=True).stdout.strip() + '\n```')

subprocess.run(['ollama', 'stop', MODEL], capture_output=True)
(outdir / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
print(f'\nSaved {outdir}')
