"""Lab 05 Step 4. RAG end to end: retrieve, build the prompt, answer, and score retrieval and answer separately.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\rag_answer.py      (≈7–10 minutes)

Retrieval (Step 3's best setup): section+title chunks, hybrid BM25 + qwen3-embedding (RRF).
Four ways to build the prompt for qwen3.5:4b (think off, temperature 0):
  closed-book  no runbooks at all: the model's own knowledge
  chunks       the top 3 section chunks
  parents      the parent runbooks of the top chunks, the first 3 distinct ones ("search small, return big")
  oracle       only the right runbook: the upper bound, if retrieval were perfect
Scored per answer (deterministic; every answer is saved for reading):
  in context   the right runbook is in the prompt at all (retrieval's job)
  cited ok     the reply starts with [right-runbook-id] (the reader's job)
  abstained    the reply is NOT IN RUNBOOKS (right for the 3 no-answer questions, wrong otherwise)
  unsupported  backticked commands/identifiers in the reply that don't appear in the prompt's runbooks
               (closed-book: checked against the right runbook instead)
"""
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from rag_lib import BM25, embed, http, load_corpus, load_questions, ranking, rrf

CHAT = 'qwen3.5:4b'
EMBED = 'qwen3-embedding:0.6b'
RULES = """You are an on-call assistant. Answer the engineer's question using only the runbooks provided.
- Start your reply with the id of the runbook you used, in square brackets, for example [oomkilled].
- Then answer in at most 4 sentences, including the commands from the runbook that apply.
- If none of the runbooks answers the question, reply exactly: NOT IN RUNBOOKS"""
CLOSED = "You are an on-call assistant. Answer the engineer's question in at most 4 sentences, including the commands that apply."


def sections(name, text):
    """section+title chunks (Step 3): the runbook's # title, then one ## section."""
    title = text.splitlines()[0]
    return [(name, p.split('\n', 1)[0].strip(), f'{title}\n## {p.strip()}') for p in text.split('\n## ')[1:]]


def backticked(text):
    return [s.strip() for s in re.findall(r'`([^`\n]+)`', text) if s.strip()]


def supported(item, context):
    """A backticked item counts as supported if it appears in the context, ignoring case, spacing and
    placeholders: `kubectl logs my-pod --previous` matches `kubectl logs <pod> --previous`."""
    norm = lambda s: re.sub(r'\s+', ' ', s.lower()).strip()
    ctx = norm(context)
    if norm(item) in ctx:
        return True
    words = norm(item).split()
    if len(words) > 1:   # a command: the tool, the subcommand and every flag must appear in one context span
        sig = words[:2] + [w.split('=')[0] for w in words[2:] if w.startswith('-')]
        return any(all(w in norm(span).split() or w in norm(span) for w in sig) for span in backticked(context))
    return False


def ask(system, user):
    r = http.post('/api/chat', json={'model': CHAT, 'stream': False, 'think': False, 'truncate': False,
                                     'messages': [{'role': 'system', 'content': system},
                                                  {'role': 'user', 'content': user}],
                                     'options': {'temperature': 0, 'seed': 42, 'num_predict': 300}})
    r.raise_for_status()
    return r.json()


corpus = dict(load_corpus())
allq = load_questions()
chunks = [c for n, t in corpus.items() for c in sections(n, t)]

# --- Retrieval for every question, then free the embedding model's VRAM ---
bm = BM25([t for _, _, t in chunks])
cvec, _ = embed(EMBED, [t for _, _, t in chunks])
qvec, _ = embed(EMBED, [q['q'] for q in allq])
subprocess.run(['ollama', 'stop', EMBED], capture_output=True)
retrieved = {}
for q, qv in zip(allq, qvec):
    order = rrf(ranking(bm.scores(q['q'])), ranking(cvec @ qv))
    parents = list(dict.fromkeys(chunks[i][0] for i in order))[:3]
    retrieved[q['id']] = {'chunks': order[:3], 'parents': parents}


def build(cond, q):
    """Returns (system, user prompt, runbook ids in the prompt, the text unsupported items are checked against)."""
    if cond == 'closed-book':
        return CLOSED, q['q'], [], corpus[q['docs'][0]] if q['docs'] else ''
    if cond == 'chunks':
        blocks = [(chunks[i][0], f'[{chunks[i][0]}] ({chunks[i][1]})\n{chunks[i][2]}') for i in retrieved[q['id']]['chunks']]
    elif cond == 'parents':
        blocks = [(d, f'[{d}]\n{corpus[d]}') for d in retrieved[q['id']]['parents']]
    else:
        blocks = [(d, f'[{d}]\n{corpus[d]}') for d in q['docs'][:1]]
    text = '\n\n'.join(b for _, b in blocks)
    return RULES, f'Runbooks:\n<<<\n{text}\n>>>\n\nQuestion: {q["q"]}', [d for d, _ in blocks], text


conds = ['closed-book', 'chunks', 'parents', 'oracle']
jobs = [(c, q) for c in conds for q in allq if not (c == 'oracle' and not q['docs'])]
stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
raw_path = Path('results/raw') / f'rag-answer-{stamp}.jsonl'
raw_path.parent.mkdir(parents=True, exist_ok=True)
rows = []
with raw_path.open('w', encoding='utf-8') as raw:
    for n, (cond, q) in enumerate(jobs, 1):
        print(f'\r  {n}/{len(jobs)}  {cond:<12} {q["id"]}   ', end='', flush=True)
        system, user, in_prompt, ctx = build(cond, q)
        t = time.perf_counter()
        r = ask(system, user)
        reply = r['message']['content'].strip()
        m = re.match(r'\s*\[([a-z0-9-]+)\]', reply)
        items = backticked(reply)
        row = {'cond': cond, 'id': q['id'], 'type': q['type'], 'want': q['docs'], 'in_prompt': in_prompt,
               'in_context': any(d in in_prompt for d in q['docs']), 'cited': m.group(1) if m else None,
               'abstained': 'NOT IN RUNBOOKS' in reply, 'items': len(items),
               'unsupported': [i for i in items if not supported(i, ctx)],
               'prompt_tokens': r['prompt_eval_count'], 'tokens': r['eval_count'],
               'seconds': round(time.perf_counter() - t, 2), 'reply': reply}
        row['cited_ok'] = row['cited'] in q['docs']
        rows.append(row)
        raw.write(json.dumps(row) + '\n')
print('\r' + ' ' * 60 + '\r', end='')
subprocess.run(['ollama', 'stop', CHAT], capture_output=True)

report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


ans = [q for q in allq if q['docs']]
say(f'# rag_answer: {CHAT}, retrieval {EMBED} + BM25 hybrid on section+title chunks; '
    f'{len(ans)} answerable + {len(allq) - len(ans)} no-answer questions\n')
say(f'{"Condition":<12} {"In context":>10} {"Cited ok":>9} {"Abstained":>10} {"No-answer":>10} {"Unsupported":>12} '
    f'{"Prompt tok":>11} {"Sec":>5}')
say(f'{"":<12} {"(answerable)":>10} {"":>9} {"(wrongly)":>10} {"abstained":>10} {"items":>12} {"(median)":>11} {"":>5}')
for c in conds:
    a = [r for r in rows if r['cond'] == c and r['want']]
    na = [r for r in rows if r['cond'] == c and not r['want']]
    items = sum(r['items'] for r in rows if r['cond'] == c)
    uns = sum(len(r['unsupported']) for r in rows if r['cond'] == c)
    ic = '-' if c == 'closed-book' else f'{sum(r["in_context"] for r in a)}/{len(a)}'
    co = '-' if c == 'closed-book' else f'{sum(r["cited_ok"] for r in a)}/{len(a)}'
    ab = '-' if c == 'closed-book' else f'{sum(r["abstained"] for r in a)}/{len(a)}'
    nab = '-' if c in ('closed-book', 'oracle') else f'{sum(r["abstained"] for r in na)}/{len(na)}'
    say(f'{c:<12} {ic:>10} {co:>9} {ab:>10} {nab:>10} {f"{uns}/{items}":>12} '
        f'{np.median([r["prompt_tokens"] for r in rows if r["cond"] == c]):>11.0f} '
        f'{np.median([r["seconds"] for r in rows if r["cond"] == c]):>5.1f}')

say('\nWhere the answerable misses come from (parents):')
p = [r for r in rows if r['cond'] == 'parents' and r['want']]
say(f'  right runbook not retrieved:              {sum(not r["in_context"] for r in p)}')
say(f'  retrieved, but cited another runbook:     {sum(r["in_context"] and not r["cited_ok"] and not r["abstained"] for r in p)}')
say(f'  retrieved, but abstained:                 {sum(r["in_context"] and r["abstained"] for r in p)}')
say(f'  cited correctly:                          {sum(r["cited_ok"] for r in p)}')

say('\nCited ok per question type:')
types = list(dict.fromkeys(q['type'] for q in ans))
say(f'{"Condition":<12} ' + ' '.join(f'{t[:10]:>10}' for t in types))
for c in conds[1:]:
    cells = [f'{sum(r["cited_ok"] for r in rows if r["cond"] == c and r["type"] == t)}/'
             f'{sum(1 for r in rows if r["cond"] == c and r["type"] == t)}' for t in types]
    say(f'{c:<12} ' + ' '.join(f'{x:>10}' for x in cells))

say('\nNo-answer questions, replies (first 110 characters):')
for r in rows:
    if not r['want']:
        say(f'  {r["cond"]:<12} {r["id"]}  {" ".join(r["reply"].split())[:110]}')

say('\nMisses with the right runbook in the prompt (parents): cited / abstained, reply start')
for r in p:
    if r['in_context'] and not r['cited_ok']:
        say(f'  {r["id"]} want {r["want"][0]:<22} got {str(r["cited"]):<22} {" ".join(r["reply"].split())[:70]}')

say('\nUnsupported items, first 12 (condition, question: item):')
for r in [r for r in rows if r['unsupported']][:12]:
    say(f'  {r["cond"]:<12} {r["id"]}: {", ".join(r["unsupported"])[:110]}')

out = Path('results/raw') / f'rag-answer-{stamp}.md'
out.write_text('\n'.join(report) + '\n', encoding='utf-8')
say(f'\nSaved to {out} and every answer to {raw_path}')
