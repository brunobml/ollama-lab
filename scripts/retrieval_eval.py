"""Lab 05 Step 2. Retrieval quality on a labelled corpus: BM25 vs embeddings vs hybrid.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\retrieval_eval.py      (≈1 minute)

Corpus: labs/05-embeddings-rag/corpus/*.md, 18 synthetic runbooks (one retrieval unit = one runbook;
        chunking is Step 3). Questions: labs/05-embeddings-rag/questions.json, 33 questions in 4 types:
        paraphrase (no shared keywords), identifier (an exact error string or code), distractor (a
        near-identical runbook exists), knowledge (needs an inference the text doesn't spell out).
Methods: BM25 (keyword, implemented below), each embedding model with and without the card's prefixes,
         and hybrid = BM25 + one embedding ranking merged with reciprocal rank fusion (RRF).
Metrics: Hit@1 (right runbook first), Hit@3 (in the top 3: what a RAG prompt with k=3 would contain),
         MRR (mean of 1/rank of the first right runbook).
"""
import json
import math
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

HOST = 'http://127.0.0.1:11434'
LAB = Path('labs/05-embeddings-rag')
MODELS = ['embeddinggemma:300m', 'qwen3-embedding:0.6b']
QWEN_TASK = 'Given a question about a Kubernetes or infrastructure problem, retrieve the runbook that answers it'
http = httpx.Client(base_url=HOST, timeout=300)
report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


# --- BM25 (Okapi), the scoring behind Elasticsearch/OpenSearch, in a few lines ---
def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())     # keeps codes like 137, x509, crashloopbackoff


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for c in self.docs for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        out = []
        for c, dl in zip(self.docs, self.len):
            s = 0.0
            for t in set(tokens(query)):
                if t in c:
                    tf = c[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return np.array(out)


def embed(model, texts):
    # truncate=false: a runbook longer than the model's context must fail loudly (Step 1)
    r = http.post('/api/embed', json={'model': model, 'input': texts, 'truncate': False})
    r.raise_for_status()
    body = r.json()
    return np.array(body['embeddings']), body['prompt_eval_count']


def prefixes(model, title):
    if model.startswith('embeddinggemma'):
        return 'task: search result | query: ', f'title: {title} | text: '
    return f'Instruct: {QWEN_TASK}\nQuery:', ''          # qwen3: instruction on the query only


def ranking(scores):
    return list(np.argsort(-scores, kind='stable'))


def rrf(*rankings, k=60):
    """Reciprocal rank fusion: each ranking adds 1/(k + rank); robust to the different score scales."""
    fused = Counter()
    for r in rankings:
        for pos, doc in enumerate(r, 1):
            fused[doc] += 1 / (k + pos)
    return [d for d, _ in fused.most_common()]


paths = sorted((LAB / 'corpus').glob('*.md'))
names = [p.stem for p in paths]
texts = [p.read_text(encoding='utf-8') for p in paths]
titles = [t.splitlines()[0].lstrip('# ').strip() for t in texts]
questions = json.loads((LAB / 'questions.json').read_text(encoding='utf-8'))
types = ['paraphrase', 'identifier', 'distractor', 'knowledge']
say(f'# retrieval_eval: {len(names)} runbooks, {len(questions)} questions '
    f'({", ".join(f"{t} {sum(q["type"] == t for q in questions)}" for t in types)})\n')

rankings = {}                                         # method -> list of rankings, one per question
bm25 = BM25(texts)
rankings['bm25'] = [ranking(bm25.scores(q['q'])) for q in questions]
no_overlap = [q['id'] for q in questions if bm25.scores(q['q']).max() == 0]
say(f'BM25: {len(no_overlap)} question(s) share no word with any runbook (their ranking is arbitrary): '
    f'{", ".join(no_overlap) or "none"}')

for m in MODELS:
    for use in (False, True):
        t = time.perf_counter()
        qp = prefixes(m, '')[0] if use else ''
        dvec, dtok = embed(m, [(prefixes(m, ti)[1] if use else '') + tx for ti, tx in zip(titles, texts)])
        qvec, _ = embed(m, [qp + q['q'] for q in questions])
        name = f'{m.split(":")[0]}{" +prefix" if use else ""}'
        rankings[name] = [ranking(row) for row in qvec @ dvec.T]
        say(f'indexed with {name:<26} {dtok:>6} tokens for {len(names)} runbooks, {time.perf_counter() - t:.1f} s')
    subprocess.run(['ollama', 'stop', m], capture_output=True)
for m in MODELS:
    dense = f'{m.split(":")[0]} +prefix'
    rankings[f'hybrid bm25+{dense}'] = [rrf(b, d) for b, d in zip(rankings['bm25'], rankings[dense])]


def first_hit(rank, q):
    return next(pos for pos, d in enumerate(rank, 1) if names[d] in q['docs'])


say(f'\n{"Method":<38} {"Hit@1":>6} {"Hit@3":>6} {"MRR":>6}   ' + ' '.join(f'{t[:10]:>10}' for t in types))
say(f'{"":<38} {"":>6} {"":>6} {"":>6}   ' + ' '.join(f'{"Hit@1":>10}' for _ in types))
results = {}
for method, ranks in rankings.items():
    pos = [first_hit(r, q) for r, q in zip(ranks, questions)]
    results[method] = pos
    per_type = []
    for t in types:
        p = [x for x, q in zip(pos, questions) if q['type'] == t]
        per_type.append(f'{sum(x == 1 for x in p)}/{len(p)}')
    say(f'{method:<38} {sum(x == 1 for x in pos):>3}/{len(pos)} {sum(x <= 3 for x in pos):>3}/{len(pos)} '
        f'{np.mean([1 / x for x in pos]):>6.3f}   ' + ' '.join(f'{c:>10}' for c in per_type))

say('\nMisses (rank of the right runbook > 1): question -> what came first')
for method in ['bm25', 'embeddinggemma +prefix', 'qwen3-embedding +prefix', 'hybrid bm25+qwen3-embedding +prefix']:
    say(f'\n{method}:')
    for q, r, p in zip(questions, rankings[method], results[method]):
        if p > 1:
            say(f'  {q["id"]} rank {p:>2}  "{q["q"][:60]}" -> {names[r[0]]}')

out = Path('results/raw') / f'retrieval-eval-{datetime.now():%Y%m%d-%H%M%S}.md'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(report) + '\n', encoding='utf-8')
say(f'\nSaved to {out}')
