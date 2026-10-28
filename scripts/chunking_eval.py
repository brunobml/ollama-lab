"""Lab 05 Step 3. Chunking: what the retrieval unit is changes what can be found.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\chunking_eval.py      (≈1–2 minutes)

Five ways to cut the same 18 runbooks into retrieval units:
  doc            one chunk per runbook (the Step 2 baseline)
  section        one chunk per ## section (Symptoms / Cause / Fix), heading only
  section+title  the same, with the runbook's # title in front of each chunk (a "contextual header")
  fixed-60       60-word windows over the whole file, ignoring its structure
  fixed-60/20    60-word windows that overlap by 20 words
Each is searched with BM25, both embedding models (no prefixes: Step 2 found no effect), and the two hybrids.

Two levels of "right":
  doc level      a chunk from the right runbook          (enough to find the document)
  section level  a chunk of the right runbook's labelled section, e.g. its Fix   (what a RAG prompt needs)
Fixed windows get the section where most of their words are. The doc strategy has no section level.
No-answer questions (the corpus has no runbook for them) are reported separately: what does search return?
"""
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from rag_lib import BM25, MODELS, embed, load_corpus, load_questions, ranking, rrf

report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


def tagged_words(text):
    """The file's words, each tagged with the section it belongs to ('title' before the first ##)."""
    section, out = 'title', []
    for line in text.splitlines():
        if line.startswith('## '):
            section = line[3:].strip()
        out += [(w, section) for w in line.split()]
    return out


def chunk(strategy, name, text):
    """Returns [(section or None, chunk text)] for one runbook."""
    title = text.splitlines()[0]
    if strategy == 'doc':
        return [(None, text)]
    if strategy in ('section', 'section+title'):
        parts = text.split('\n## ')[1:]                # drop the title block, keep each section
        return [(p.split('\n', 1)[0].strip(), ('' if strategy == 'section' else title + '\n') + '## ' + p.strip())
                for p in parts]
    size, overlap = (60, 0) if strategy == 'fixed-60' else (60, 20)
    words = tagged_words(text)
    out = []
    for start in range(0, len(words), size - overlap):
        window = words[start:start + size]
        out.append((Counter(s for _, s in window).most_common(1)[0][0], ' '.join(w for w, _ in window)))
        if start + size >= len(words):
            break
    return out


corpus = load_corpus()
allq = load_questions()
questions = [q for q in allq if q['docs']]
no_answer = [q for q in allq if not q['docs']]
types = list(dict.fromkeys(q['type'] for q in questions))
strategies = ['doc', 'section', 'section+title', 'fixed-60', 'fixed-60/20']
say(f'# chunking_eval: {len(corpus)} runbooks, {len(questions)} questions with an answer '
    f'({", ".join(f"{t} {sum(q["type"] == t for q in questions)}" for t in types)}), {len(no_answer)} without\n')

# chunks[strategy] = list of (doc name, section, text)
chunks = {s: [(n, sec, t) for n, text in corpus for sec, t in chunk(s, n, text)] for s in strategies}
vectors = {}                                           # (strategy, model) -> chunk matrix
qvec = {}
say(f'{"Strategy":<15} {"Chunks":>6} {"Words/chunk":>12} {"Tokens/chunk (gemma)":>21}')
for m in MODELS:
    qvec[m], _ = embed(m, [q['q'] for q in allq])
    for s in strategies:
        vectors[(s, m)], tok = embed(m, [t for _, _, t in chunks[s]])
        if m == MODELS[0]:
            n = len(chunks[s])
            say(f'{s:<15} {n:>6} {np.mean([len(t.split()) for _, _, t in chunks[s]]):>12.0f} {tok / n:>21.0f}')
    subprocess.run(['ollama', 'stop', m], capture_output=True)


def rankings_for(s):
    """method -> list of chunk rankings, one per question in allq."""
    bm = BM25([t for _, _, t in chunks[s]])
    out = {'bm25': [ranking(bm.scores(q['q'])) for q in allq]}
    for m in MODELS:
        out[m.split(':')[0]] = [ranking(row) for row in qvec[m] @ vectors[(s, m)].T]
    for m in MODELS:
        short = m.split(':')[0]
        out[f'hybrid bm25+{short}'] = [rrf(b, d) for b, d in zip(out['bm25'], out[short])]
    return out


def first(rank, s, q, section_level):
    for pos, i in enumerate(rank, 1):
        doc, sec, _ = chunks[s][i]
        if doc in q['docs'] and (not section_level or sec == q['section']):
            return pos


say(f'\n{"Strategy":<15} {"Method":<26} {"Doc Hit@1":>9} {"Doc Hit@3":>9} {"Doc MRR":>7} '
    f'{"Sec Hit@1":>9} {"Sec Hit@3":>9}')
all_ranks = {}
idx = [allq.index(q) for q in questions]
for s in strategies:
    rk = rankings_for(s)
    all_ranks[s] = rk
    for method, ranks in rk.items():
        d = [first(ranks[i], s, allq[i], False) for i in idx]
        line = (f'{s:<15} {method:<26} {sum(p == 1 for p in d):>6}/{len(d)} {sum(p <= 3 for p in d):>6}/{len(d)} '
                f'{np.mean([1 / p for p in d]):>7.3f}')
        if s != 'doc':
            sp = [first(ranks[i], s, allq[i], True) for i in idx]
            line += f' {sum(p == 1 for p in sp):>6}/{len(sp)} {sum(p <= 3 for p in sp):>6}/{len(sp)}'
        say(line)
    say()

say('Section-level Hit@1 per question type, strategy section+title:')
say(f'{"Method":<26} ' + ' '.join(f'{t[:10]:>10}' for t in types))
for method, ranks in all_ranks['section+title'].items():
    cells = []
    for t in types:
        qs = [i for i in idx if allq[i]['type'] == t]
        cells.append(f'{sum(first(ranks[i], "section+title", allq[i], True) == 1 for i in qs)}/{len(qs)}')
    say(f'{method:<26} ' + ' '.join(f'{c:>10}' for c in cells))

say('\nDoc-level Hit@1 per question type, strategy doc (the Step 2 setup, now with the on-call questions):')
say(f'{"Method":<26} ' + ' '.join(f'{t[:10]:>10}' for t in types))
for method, ranks in all_ranks['doc'].items():
    cells = []
    for t in types:
        qs = [i for i in idx if allq[i]['type'] == t]
        cells.append(f'{sum(first(ranks[i], "doc", allq[i], False) == 1 for i in qs)}/{len(qs)}')
    say(f'{method:<26} ' + ' '.join(f'{c:>10}' for c in cells))

say('\nMisses at section level, strategy section+title, hybrid bm25+qwen3-embedding (what came first):')
s = 'section+title'
for i in idx:
    q = allq[i]
    r = all_ranks[s]['hybrid bm25+qwen3-embedding'][i]
    p = first(r, s, q, True)
    if p > 1:
        doc, sec, _ = chunks[s][r[0]]
        say(f'  {q["id"]} rank {p:>2}  want {q["docs"][0]}/{q["section"]:<8}  got {doc}/{sec:<8}  "{q["q"][:50]}"')

say('\nNo-answer questions, strategy section+title: top chunk and its score vs the answerable questions')
for m in MODELS:
    sims = qvec[m] @ vectors[(s, m)].T
    top = sims.max(axis=1)
    ans_top = [top[i] for i in idx]
    say(f'{m}: top-1 similarity for answerable questions: min {min(ans_top):.3f}, '
        f'median {np.median(ans_top):.3f}, max {max(ans_top):.3f}')
    for q in no_answer:
        i = allq.index(q)
        doc, sec, _ = chunks[s][int(np.argmax(sims[i]))]
        below = sum(t < top[i] for t in ans_top)
        say(f'  {q["id"]} top {top[i]:.3f} ({below}/{len(ans_top)} answerable questions score lower) '
            f'-> {doc}/{sec}  "{q["q"]}"')

out = Path('results/raw') / f'chunking-eval-{datetime.now():%Y%m%d-%H%M%S}.md'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(report) + '\n', encoding='utf-8')
say(f'\nSaved to {out}')
