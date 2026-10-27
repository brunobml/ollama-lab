"""Lab 04 Step 5. The OpenAI-compatible API (/v1) vs the native API (/api), on the wire.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\openai_compat.py

Raw httpx on purpose: this is exactly what any OpenAI SDK or tool (Codex, Phase 6) sends.
Part 1: thinking. /v1 has no `think` field; qwen3.5 thinks by default (/api/show: default true).
        What do reasoning_effort / reasoning.effort do, and where does the thinking text go?
Part 2: the response shape, and what's missing compared with /api/chat
Part 3: context size. The docs: "The OpenAI API does not have a way of setting the context size".
        Oversized prompt by default, with an extra `options` body, with `truncate: false`,
        and with a model variant that has num_ctx in its Modelfile (if it exists).
Part 4: structured output via response_format
Part 5: streaming wire format (SSE instead of NDJSON)
"""
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

HOST = 'http://127.0.0.1:11434'
MODEL = 'qwen3.5:4b'
VARIANT = 'qwen3.5:4b-ctx8k'   # optional, created from a Modelfile with PARAMETER num_ctx 8192 (Step 5 README)
QUESTION = 'Name the Kubernetes object that keeps a set number of identical pods running. One sentence.'
http = httpx.Client(base_url=HOST, timeout=180)
report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


def v1(body):
    """POST /v1/chat/completions; returns (status, json or text, seconds)."""
    t = time.perf_counter()
    r = http.post('/v1/chat/completions', json={'model': MODEL, **body})
    secs = time.perf_counter() - t
    try:
        return r.status_code, r.json(), secs
    except json.JSONDecodeError:
        return r.status_code, r.text, secs


def short(s, n=70):
    s = ' '.join(str(s).split())
    return s if len(s) <= n else s[:n] + '…'


def log_prompt():
    """≈5,800 tokens: the cluster name on top, 120 log lines, the question last (Lab 03 Step 5a)."""
    lines = [f'L{i:04d} 2026-10-27T12:{i // 60 % 60:02d}:{i % 60:02d}Z pod/api-{i % 7} ns=prod level=INFO '
             f'msg="request served" latency_ms={100 + i * 37 % 900}' for i in range(1, 121)]
    return 'The cluster name is blue-falcon.\n\nLog:\n' + '\n'.join(lines) + '\n\nWhat is the cluster name? One word.'


msgs = [{'role': 'user', 'content': QUESTION}]
http.post('/api/chat', json={'model': MODEL, 'messages': msgs, 'stream': False, 'options': {'num_predict': 1}})  # warm up

# --- Part 1: thinking control ---
say(f'# openai_compat ({MODEL})\n\n## Part 1: thinking through /v1\n')
native = http.post('/api/chat', json={'model': MODEL, 'messages': msgs, 'stream': False, 'think': False,
                                      'options': {'temperature': 0, 'seed': 42, 'num_predict': 1500}}).json()
cases = [
    ('v1, no reasoning field', {}),
    ('v1, reasoning_effort "none"', {'reasoning_effort': 'none'}),
    ('v1, reasoning_effort "low"', {'reasoning_effort': 'low'}),
    ('v1, reasoning {effort: "none"}', {'reasoning': {'effort': 'none'}}),
    ('v1, think: false (native field)', {'think': False}),
]
say(f'{"Case":<34} {"HTTP":>4} {"Compl.tok":>9} {"Finish":<7} {"Secs":>5}  {"Message keys":<32} Answer')
say(f'{"api/chat, think false (reference)":<34} {200:>4} {native["eval_count"]:>9} {native["done_reason"]:<7} '
    f'{native["total_duration"] / 1e9:>5.2f}  {",".join(native["message"]):<32} {short(native["message"]["content"], 50)}')
v1_none = None
for name, extra in cases:
    status, body, secs = v1({'messages': msgs, 'temperature': 0, 'seed': 42, 'max_tokens': 1500, **extra})
    if status != 200:
        say(f'{name:<34} {status:>4}  {short(body, 120)}')
        continue
    m = body['choices'][0]['message']
    if extra == {'reasoning_effort': 'none'}:
        v1_none = m.get('content')
    say(f'{name:<34} {status:>4} {body["usage"]["completion_tokens"]:>9} {body["choices"][0]["finish_reason"]:<7} '
        f'{secs:>5.2f}  {",".join(m):<32} {short(m.get("content"), 50)}')
    reasoning = m.get('reasoning') or m.get('reasoning_content')
    if reasoning:
        say(f'{"":<34} thinking text in message.{"reasoning" if m.get("reasoning") else "reasoning_content"}: '
            f'{len(reasoning)} chars, starts {short(reasoning, 60)!r}')
say(f'\nSame answer, /api/chat think=false vs /v1 reasoning_effort "none": {v1_none == native["message"]["content"]}\n')

# --- Part 2: response shape ---
say('## Part 2: response shape\n')
status, body, _ = v1({'messages': msgs, 'temperature': 0, 'seed': 42, 'max_tokens': 5, 'reasoning_effort': 'none'})
say(f'/v1 top-level keys:   {", ".join(body)}')
say(f'/v1 usage:            {body.get("usage")}')
say(f'/v1 finish_reason:    {body["choices"][0]["finish_reason"]!r} with max_tokens 5')
say(f'/api/chat keys:       {", ".join(native)}')
say(f'only in /api/chat:    {", ".join(k for k in native if k.endswith("duration") or k.endswith("count"))}\n')

# --- Part 3: context size ---
say('## Part 3: context size (prompt ≈5,800 tokens, server default num_ctx 4096)\n')
big = [{'role': 'user', 'content': log_prompt()}]
ctx_cases = [
    ('v1 default', MODEL, {}),
    ('v1 + options {num_ctx: 8192}', MODEL, {'options': {'num_ctx': 8192}}),
    ('v1 + truncate: false', MODEL, {'truncate': False}),
]
have_variant = any(m['id'] == VARIANT for m in http.get('/v1/models').json()['data'])
if have_variant:
    ctx_cases.append((f'v1, model {VARIANT}', VARIANT, {}))
say(f'{"Case":<34} {"HTTP":>4} {"Prompt.tok":>10}  Answer / error')
for name, model, extra in ctx_cases:
    t = time.perf_counter()
    r = http.post('/v1/chat/completions', json={'model': model, 'messages': big, 'temperature': 0, 'max_tokens': 20,
                                                'reasoning_effort': 'none', **extra})
    try:
        b = r.json()
    except json.JSONDecodeError:
        b = {'error': r.text}
    if r.status_code == 200:
        say(f'{name:<34} {r.status_code:>4} {b["usage"]["prompt_tokens"]:>10}  {short(b["choices"][0]["message"]["content"], 60)}')
    else:
        say(f'{name:<34} {r.status_code:>4} {"":>10}  {short(b.get("error", b), 100)}')
if not have_variant:
    say(f'({VARIANT} not found: the Modelfile case was skipped; see the Step 5 README)')
ps = http.get('/api/ps').json()['models']
say(f'loaded now: {[(m["name"], m.get("context_length")) for m in ps]}\n')

# --- Part 4: structured output ---
say('## Part 4: structured output via response_format\n')
schema = {'type': 'object', 'properties': {'kind': {'type': 'string'}, 'api_group': {'type': 'string'}},
          'required': ['kind', 'api_group']}
for name, rf in [('json_object', {'type': 'json_object'}),
                 ('json_schema', {'type': 'json_schema', 'json_schema': {'name': 'k8s_object', 'schema': schema}})]:
    status, body, _ = v1({'messages': [{'role': 'user', 'content': 'Which Kubernetes kind runs a set number of '
                                        'identical pods, and in which API group is it? Reply as JSON.'}],
                          'temperature': 0, 'max_tokens': 200, 'reasoning_effort': 'none', 'response_format': rf})
    content = body['choices'][0]['message']['content'] if status == 200 else body
    try:
        parsed = json.loads(content)
        keys = sorted(parsed)
    except (json.JSONDecodeError, TypeError):
        keys = 'NOT JSON'
    say(f'{name:<12} HTTP {status}  keys {keys}  {short(content, 80)}')

# --- Part 5: streaming wire format ---
say('\n## Part 5: streaming (SSE)\n')
with http.stream('POST', '/v1/chat/completions', json={
        'model': MODEL, 'messages': msgs, 'temperature': 0, 'max_tokens': 30, 'reasoning_effort': 'none',
        'stream': True, 'stream_options': {'include_usage': True}}) as r:
    lines = [ln for ln in r.iter_lines() if ln]
say(f'content-type: {r.headers.get("content-type")}, {len(lines)} non-empty lines. First 2 and last 2:')
for ln in lines[:2] + lines[-2:]:
    say('  ' + short(ln, 200))

http.post('/api/generate', json={'model': MODEL, 'keep_alive': 0})
if have_variant:
    http.post('/api/generate', json={'model': VARIANT, 'keep_alive': 0})
out = Path('results/raw') / f'openai-compat-{datetime.now():%Y%m%d-%H%M%S}.md'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(report) + '\n', encoding='utf-8')
say(f'\nSaved to {out}')
