"""Lab 05 Step 1. What an embedding is, measured: the vector, similarity, a tiny retrieval, and the input limit.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\embed_basics.py      (≈1 minute)

Part 1: the vector itself: dimensions, length (norm), determinism, batch vs one-by-one, speed
Part 2: cosine similarity for sentence pairs: paraphrase, negation, unrelated, jargon vs description, ...
Part 3: a 6-document retrieval: does the right runbook come first? With and without the prompt
        prefixes from each model card
Part 4: input longer than the context: truncated or an error? Which part is kept?
Both models: embeddinggemma:300m and qwen3-embedding:0.6b. Vectors in numpy, no vector DB.
"""
import subprocess
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

HOST = 'http://127.0.0.1:11434'
MODELS = ['embeddinggemma:300m', 'qwen3-embedding:0.6b']
# Prompt prefixes from the model cards (huggingface.co/google/embeddinggemma-300m, Qwen/Qwen3-Embedding-0.6B)
PREFIX = {
    'embeddinggemma:300m': ('task: search result | query: ', 'title: none | text: '),
    'qwen3-embedding:0.6b': ('Instruct: Given a question about a Kubernetes problem, retrieve the runbook '
                             'passage that answers it\nQuery:', ''),   # card: documents need no instruction
}
http = httpx.Client(base_url=HOST, timeout=300)
report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


def embed(model, texts, **extra):
    """Returns (matrix with one row per text, response json without the vectors)."""
    r = http.post('/api/embed', json={'model': model, 'input': texts, **extra})
    if r.status_code != 200:
        return None, {'status': r.status_code, 'error': r.json().get('error')}
    body = r.json()
    return np.array(body.pop('embeddings')), body


PAIRS = [
    ('identical', 'The pod keeps restarting.', 'The pod keeps restarting.'),
    ('paraphrase', 'The pod keeps restarting.', 'The container is stuck in a crash loop.'),
    ('jargon vs plain words', 'ImagePullBackOff', 'The node cannot download the container image from the registry.'),
    ('negation', 'The pod is ready.', 'The pod is not ready.'),
    ('same topic, other task', 'How do I roll back a Helm release?', 'How do I install a Helm chart?'),
    ('other language', 'The pod keeps restarting.', 'O pod continua reiniciando.'),
    ('unrelated', 'The pod keeps restarting.', 'Bake the bread at 220 degrees for 30 minutes.'),
]
DOCS = {
    'crashloop': 'If a container exits repeatedly, Kubernetes restarts it with an increasing back-off delay and the '
                 'pod shows CrashLoopBackOff. Check the previous container\'s logs with kubectl logs --previous.',
    'oom': 'A container that exceeds its memory limit is terminated by the kernel and reported as OOMKilled with '
           'exit code 137. Raise the memory limit or reduce the application\'s memory use.',
    'imagepull': 'ImagePullBackOff means the kubelet could not pull the container image: check the image name and '
                 'tag, and the imagePullSecrets for private registries.',
    'pending': 'A pod stays Pending when the scheduler finds no node with enough free CPU or memory, or when node '
               'selectors and taints exclude every node.',
    'helm-rollback': 'To return a release to an earlier revision, run helm rollback <release> <revision>; '
                     'helm history <release> lists the revisions.',
    'readiness': 'A failing readiness probe removes the pod from the Service endpoints but does not restart the '
                 'container; a failing liveness probe restarts it.',
}
# Written to avoid the documents' keywords where possible (except the 137 one, on purpose)
QUERIES = [
    ('my app keeps dying and coming back every few minutes', 'crashloop'),
    ('container killed, exit status 137', 'oom'),
    ('the cluster can\'t download my image from our private registry', 'imagepull'),
    ('why is my new pod never placed on a machine', 'pending'),
    ('undo the last helm upgrade', 'helm-rollback'),
    ('pod is up but gets no traffic', 'readiness'),
]

say(f'# embed_basics ({", ".join(MODELS)})\n')

# --- Part 1: the vector ---
say('## Part 1: the vector\n')
say(f'{"Model":<22} {"Dims":>5} {"Norm":>7} {"Load s":>7} {"Tokens":>6} {"Same twice":>10} {"Batch=single":>13} '
    f'{"ms/text (batch 48)":>19}  First 4 values')
for m in MODELS:
    v1, b1 = embed(m, ['The pod keeps restarting.'])            # cold: includes the load
    v2, _ = embed(m, ['The pod keeps restarting.'])
    batch, _ = embed(m, [t for _, a, b in PAIRS for t in (a, b)])
    singles = np.array([embed(m, [t])[0][0] for _, a, b in PAIRS for t in (a, b)])
    texts = [d for d in DOCS.values()] * 8                         # 48 texts of ≈40 tokens
    t = time.perf_counter()
    embed(m, texts)
    ms = (time.perf_counter() - t) * 1000 / len(texts)
    say(f'{m:<22} {v1.shape[1]:>5} {np.linalg.norm(v1[0]):>7.4f} {b1["load_duration"] / 1e9:>7.2f} '
        f'{b1["prompt_eval_count"]:>6} {str(np.array_equal(v1, v2)):>10} '
        f'{"max diff " + format(np.abs(batch - singles).max(), ".1e"):>13} {ms:>19.1f}  {np.round(v1[0][:4], 4)}')

# --- Part 2: similarity ---
say('\n## Part 2: cosine similarity (vectors are unit length, so cosine = dot product)\n')
sims = {}
for m in MODELS:
    vecs, _ = embed(m, [t for _, a, b in PAIRS for t in (a, b)])
    sims[m] = [float(vecs[2 * i] @ vecs[2 * i + 1]) for i in range(len(PAIRS))]
say(f'{"Pair":<24} ' + ' '.join(f'{m.split(":")[0]:>16}' for m in MODELS) + '  Texts')
for i, (kind, a, b) in enumerate(PAIRS):
    say(f'{kind:<24} ' + ' '.join(f'{sims[m][i]:>16.3f}' for m in MODELS) + f'  "{a}" / "{b}"')

# --- Part 3: retrieval ---
say('\n## Part 3: retrieval, 6 runbook passages, 6 queries (rank of the right passage; 1 = top)\n')
names = list(DOCS)
say(f'{"Model":<22} {"Prefixes":<9} {"Hits@1":>6} {"Mean margin":>12}  Rank per query')
for m in MODELS:
    for use_prefix in (False, True):
        qp, dp = PREFIX[m] if use_prefix else ('', '')
        dvec, _ = embed(m, [dp + d for d in DOCS.values()])
        qvec, _ = embed(m, [qp + q for q, _ in QUERIES])
        scores = qvec @ dvec.T                                     # 6 x 6 cosine matrix
        ranks, margins = [], []
        for qi, (_, want) in enumerate(QUERIES):
            order = np.argsort(-scores[qi])
            ranks.append(int(np.where(order == names.index(want))[0][0]) + 1)
            # margin: right passage's score minus the best wrong one (> 0 = correct and how clearly)
            wrong = max(scores[qi][j] for j in range(len(names)) if names[j] != want)
            margins.append(scores[qi][names.index(want)] - wrong)
        say(f'{m:<22} {"yes" if use_prefix else "no":<9} {sum(r == 1 for r in ranks):>4}/6 {np.mean(margins):>12.3f}  '
            + ' '.join(f'{want[:9]}={r}' for (_, want), r in zip(QUERIES, ranks)))

# --- Part 4: input limit ---
say('\n## Part 4: input longer than the context\n')
head = 'The cluster name is blue-falcon. '
filler = ' '.join(f'Line {i}: pod api-{i % 7} served a request in {100 + i * 37 % 900} ms.' for i in range(400))
tail = ' The on-call engineer today is Dana.'
long_text = head + filler + tail
say(f'text: {len(long_text):,} characters; the first and the last 3,000 characters are embedded on their own too\n')
for m in MODELS:
    v, b = embed(m, [long_text])
    _, err = embed(m, [long_text], truncate=False)
    ctx = {x['name']: x.get('context_length') for x in http.get('/api/ps').json()['models']}.get(m)
    say(f'{m}: loaded num_ctx {ctx}; default: {b.get("prompt_eval_count")} tokens embedded, no error; '
        f'truncate=false: HTTP {err.get("status", 200)} {err.get("error") or ""}')
    parts, _ = embed(m, [long_text[:3000], long_text[-3000:]])
    # Whichever end the long vector resembles more is the part the model actually saw
    say(f'  similarity of the long text\'s vector to its start {float(parts[0] @ v[0]):.3f}, '
        f'to its end {float(parts[1] @ v[0]):.3f}')

for m in MODELS:
    subprocess.run(['ollama', 'stop', m], capture_output=True)
out = Path('results/raw') / f'embed-basics-{datetime.now():%Y%m%d-%H%M%S}.md'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(report) + '\n', encoding='utf-8')
say(f'\nSaved to {out}')
