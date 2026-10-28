"""Shared pieces for the Lab 05 scripts from Step 3 on: corpus, questions, BM25, embeddings, fusion.

The same code as in retrieval_eval.py (Step 2), which stays as it ran.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

HOST = 'http://127.0.0.1:11434'
LAB = Path('labs/05-embeddings-rag')
MODELS = ['embeddinggemma:300m', 'qwen3-embedding:0.6b']
http = httpx.Client(base_url=HOST, timeout=300)


def load_corpus():
    """Returns [(name, text)] for the runbooks, sorted by name."""
    return [(p.stem, p.read_text(encoding='utf-8')) for p in sorted((LAB / 'corpus').glob('*.md'))]


def load_questions():
    return json.loads((LAB / 'questions.json').read_text(encoding='utf-8'))


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
    """Unit-length vectors, one row per text, and the total tokens. truncate=false: oversized input fails loudly."""
    r = http.post('/api/embed', json={'model': model, 'input': texts, 'truncate': False})
    r.raise_for_status()
    body = r.json()
    return np.array(body['embeddings']), body['prompt_eval_count']


def ranking(scores):
    return list(np.argsort(-scores, kind='stable'))


def rrf(*rankings, k=60):
    """Reciprocal rank fusion: each ranking adds 1/(k + rank); robust to the different score scales."""
    fused = Counter()
    for r in rankings:
        for pos, item in enumerate(r, 1):
            fused[item] += 1 / (k + pos)
    return [i for i, _ in fused.most_common()]
