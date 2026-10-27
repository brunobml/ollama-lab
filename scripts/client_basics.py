"""Lab 04 Step 3. The ollama Python library: what it adds over raw HTTP, and how it fails.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\client_basics.py

Part 1: library vs raw HTTP (httpx) for the same request: same answer? what does the library add?
Part 2: streaming with the library, timed on the client (compare with Step 1's stream.ps1)
Part 3: failure modes: which exception, which status code, when it's raised, how long it takes
"""
import importlib.metadata
import time
from datetime import datetime
from pathlib import Path

import httpx
from ollama import Client, ResponseError

HOST = 'http://127.0.0.1:11434'   # not localhost: Windows tries ::1 first (Lab 01)
MODEL = 'qwen3.5:4b'
PROMPT = 'Explain in about 150 words what a Kubernetes readinessProbe is for.'
OPTIONS = {'temperature': 0, 'seed': 42, 'num_predict': 400}
MESSAGES = [{'role': 'user', 'content': PROMPT}]

client = Client(host=HOST, timeout=120)   # library default is timeout=None: wait forever
report = []                               # lines for the saved report


def say(text=''):
    print(text, flush=True)
    report.append(text)


def unload():
    # keep_alive=0 with no prompt unloads the model (documented on /api/generate)
    client.generate(model=MODEL, keep_alive=0)


def log_lines():
    """≈5,800 tokens of synthetic log lines (Lab 03 Step 5: 120 lines ≈ 5,823 tokens)."""
    lines = [f'L{i:04d} 2026-10-27T12:{i // 60 % 60:02d}:{i % 60:02d}Z pod/api-{i % 7} ns=prod level=INFO '
             f'msg="request served" latency_ms={100 + i * 37 % 900}' for i in range(1, 121)]
    return 'The cluster name is blue-falcon.\n\nLog:\n' + '\n'.join(lines) + '\n\nWhat is the cluster name?'


# --- Part 1: library vs raw HTTP ---
say(f'# client_basics ({MODEL}, ollama-python {importlib.metadata.version("ollama")})\n')
say('## Part 1: library vs raw HTTP (non-streaming, warm)\n')
client.chat(model=MODEL, messages=MESSAGES, options={'num_predict': 1})   # warm up: load the model

t = time.perf_counter()
lib = client.chat(model=MODEL, messages=MESSAGES, think=False, options=OPTIONS)
lib_s = time.perf_counter() - t

t = time.perf_counter()
raw = httpx.post(f'{HOST}/api/chat', timeout=120, json={
    'model': MODEL, 'messages': MESSAGES, 'stream': False, 'think': False, 'options': OPTIONS}).json()
raw_s = time.perf_counter() - t

say(f'library returns:    {type(lib).__module__}.{type(lib).__name__} (a Pydantic model)')
say(f'raw httpx returns:  {type(raw).__name__} (parsed JSON)')
say(f'same answer text:   {lib.message.content == raw["message"]["content"]}')
say(f'attribute access:   lib.message.content[:40]    = {lib.message.content[:40]!r}')
say(f'dict access too:    lib["message"]["content"][:40] = {lib["message"]["content"][:40]!r}')
say(f'durations:          lib.total_duration = {lib.total_duration} (int, nanoseconds), eval_count = {lib.eval_count}')
say(f'client wall time:   library {lib_s:.2f} s, raw {raw_s:.2f} s\n')

# --- Part 2: streaming with the library ---
say('## Part 2: streaming (warm)\n')
t = time.perf_counter()
first = None
chunks = 0
text = ''
final = None
for part in client.chat(model=MODEL, messages=MESSAGES, think=False, options=OPTIONS, stream=True):
    chunks += 1
    if part.message.content and first is None:
        first = time.perf_counter() - t
    text += part.message.content or ''
    print(part.message.content or '', end='', flush=True)
    if part.done:
        final = part
total = time.perf_counter() - t
print(flush=True)
say(f'TTFT {first:.2f} s, total {total:.2f} s, chunks {chunks}, eval_count {final.eval_count}, '
    f'done_reason {final.done_reason}, same text as Part 1: {text == lib.message.content}\n')

# --- Part 3: failure modes ---
say('## Part 3: failure modes\n')
rows = []


def attempt(name, fn):
    """Run fn, record what came back or what was raised, and how long it took."""
    t = time.perf_counter()
    try:
        out = fn()
        rows.append((name, 'no exception', '', str(out)[:90], time.perf_counter() - t))
    except Exception as e:   # deliberately broad: we want to SEE which type each failure raises
        status = getattr(e, 'status_code', '')
        rows.append((name, type(e).__module__ + '.' + type(e).__name__, status, str(e)[:90], time.perf_counter() - t))


def stream_unknown_model():
    gen = client.chat(model='qwen3.5:404b', messages=MESSAGES, stream=True)
    note = 'call returned a generator without raising; '
    try:
        next(gen)
    except ResponseError as e:
        raise ResponseError(note + 'raised on first next(): ' + e.error, e.status_code) from None
    return note + 'no error on iteration either'


big = log_lines()


def oversized_library():
    # chat() has no `truncate` parameter, so the server's silent default applies (Lab 03 Step 5)
    r = client.chat(model=MODEL, messages=[{'role': 'user', 'content': big}], think=False,
                    options={'temperature': 0, 'num_predict': 20})
    return f'prompt_eval_count={r.prompt_eval_count} answer={r.message.content.strip()[:40]!r}'


def oversized_raw():
    # httpx does NOT raise on 4xx/5xx unless you call raise_for_status(): check the status yourself
    r = httpx.post(f'{HOST}/api/chat', timeout=120, json={
        'model': MODEL, 'messages': [{'role': 'user', 'content': big}], 'stream': False, 'think': False,
        'truncate': False, 'options': {'num_predict': 20}})
    return f'HTTP {r.status_code}: {r.text.strip()}'


steps = [
    ('unknown model', lambda: client.chat(model='qwen3.5:404b', messages=MESSAGES)),
    ('unknown model, stream=True', stream_unknown_model),
    ('oversized prompt, library (no truncate param)', oversized_library),
    ('oversized prompt, raw httpx truncate=false', oversized_raw),
    ('option with wrong type (temperature="hot")', lambda: client.chat(
        model=MODEL, messages=MESSAGES, options={'temperature': 'hot', 'num_predict': 5}).message.content),
    ('server not running (port 11499)', lambda: Client(host='http://127.0.0.1:11499').list()),
]
for name, fn in steps:
    print(f'  ... {name}', flush=True)
    attempt(name, fn)

# Timeout during a cold start: the client gives up after 1 s, the model takes ≈4-6 s to load.
# Afterwards: did the server finish loading anyway, even though nobody was waiting?
print('  ... client timeout during cold load', flush=True)
unload()
attempt('client timeout 1 s, cold model', lambda: Client(host=HOST, timeout=1).chat(
    model=MODEL, messages=MESSAGES, options={'num_predict': 5}))
time.sleep(8)
loaded = [m.model for m in client.ps().models]
rows.append(('  -> /api/ps 8 s later', 'loaded models', '', str(loaded), 0.0))

w = max(len(r[0]) for r in rows)
say(f'{"Case":<{w}}  {"Exception":<34} {"Status":<6} {"Seconds":>7}  Message')
for name, exc, status, msg, secs in rows:
    say(f'{name:<{w}}  {exc:<34} {str(status):<6} {secs:>7.2f}  {msg}')

unload()
out = Path('results/raw') / f'client-basics-{datetime.now():%Y%m%d-%H%M%S}.md'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(report) + '\n\nStreamed text:\n\n' + text + '\n', encoding='utf-8')
say(f'\nSaved to {out}')
