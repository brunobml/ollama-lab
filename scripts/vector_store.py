"""Lab 05 Step 5. A vector store (sqlite-vec): what it adds over the numpy array, and what it doesn't change.

Run with the venv's interpreter:  .\\.venv\\Scripts\\python scripts\\vector_store.py      (≈1–2 minutes)

Part A  build: the 54 section+title chunks embedded once (qwen3-embedding) into one SQLite file, with the
        model name stored next to the vectors; file size, embedding time vs insert time
Part B  same answers? sqlite-vec KNN vs the numpy dot product for the 51 questions: rankings and distances
Part C  persistence: close, reopen, query without re-embedding; and what a wrong-model query does
Part D  metadata filter: section = 'Fix' for the questions whose answer is in a Fix section
Part E  quantized vectors: float32 vs int8 vs bit (4,096 / 1,024 / 128 bytes per vector); bit + float re-ranking
Part F  scale: 50,000 random 1,024-dim vectors in memory: insert time, query time, sqlite-vec vs numpy
The index file (labs/05-embeddings-rag/index.db) is gitignored (*.db) and rebuilt by every run.
"""
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sqlite_vec

from rag_lib import LAB, embed, load_corpus, load_questions

MODEL = 'qwen3-embedding:0.6b'
DB = LAB / 'index.db'
report = []


def say(text=''):
    print(text, flush=True)
    report.append(text)


def connect(path):
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)            # only while loading: an open extension door is an attack surface
    return db


def f32(v):
    return sqlite_vec.serialize_float32([float(x) for x in v])


def knn(db, table, qv, k, where='', quant=''):
    """Returns [(rowid, distance)] nearest first. quant: '' (float), 'int8' or 'bit'."""
    q = {'': '?', 'int8': "vec_quantize_int8(?, 'unit')", 'bit': 'vec_quantize_binary(?)'}[quant]
    sql = f'select rowid, distance from {table} where emb match {q} and k = ? {where} order by distance'
    return db.execute(sql, (f32(qv), k)).fetchall()


corpus = load_corpus()
chunks = [(n, p.split('\n', 1)[0].strip(), f'{t.splitlines()[0]}\n## {p.strip()}')
          for n, t in corpus for p in t.split('\n## ')[1:]]
allq = load_questions()
questions = [q for q in allq if q['docs']]

# --- Part A: build ---
vec_version = connect(':memory:').execute('select vec_version()').fetchone()[0]
say(f'# vector_store: sqlite-vec {vec_version}, SQLite {sqlite3.sqlite_version}, {MODEL}\n')
say('## Part A: build the index\n')
DB.unlink(missing_ok=True)
t = time.perf_counter()
vecs, tok = embed(MODEL, [c[2] for c in chunks])
t_embed = time.perf_counter() - t
qvecs, _ = embed(MODEL, [q['q'] for q in allq])
subprocess.run(['ollama', 'stop', MODEL], capture_output=True)
dims = vecs.shape[1]

t = time.perf_counter()
db = connect(DB)
db.execute('create table meta (key text primary key, value text)')
db.executemany('insert into meta values (?, ?)', [('model', MODEL), ('dims', str(dims)), ('chunking', 'section+title'),
                                                  ('built', datetime.now().isoformat(timespec='seconds'))])
# metadata columns (doc, section) can be filtered in a KNN query; +text is stored but not filterable
db.execute(f'create virtual table chunks using vec0(emb float[{dims}] distance_metric=cosine, '
           f'doc text, section text, +text text)')
db.execute(f'create virtual table chunks_i8 using vec0(emb int8[{dims}] distance_metric=cosine)')
db.execute(f'create virtual table chunks_bit using vec0(emb bit[{dims}])')
for i, ((doc, sec, text), v) in enumerate(zip(chunks, vecs), 1):
    db.execute('insert into chunks(rowid, emb, doc, section, text) values (?, ?, ?, ?, ?)', (i, f32(v), doc, sec, text))
    db.execute("insert into chunks_i8(rowid, emb) values (?, vec_quantize_int8(?, 'unit'))", (i, f32(v)))
    db.execute('insert into chunks_bit(rowid, emb) values (?, vec_quantize_binary(?))', (i, f32(v)))
db.commit()
t_insert = time.perf_counter() - t
db.close()
say(f'{len(chunks)} chunks, {tok} tokens, {dims} dims')
say(f'embedding {t_embed:.2f} s (includes the model load), insert into SQLite {t_insert:.3f} s')
say(f'file {DB}: {DB.stat().st_size / 1024:.0f} KiB (float32 vectors alone: {len(chunks) * dims * 4 / 1024:.0f} KiB)')

# --- Part B: same answers as numpy? ---
say('\n## Part B: sqlite-vec vs numpy, 51 questions\n')
db = connect(DB)
same_order, max_diff, same_top3 = 0, 0.0, 0
for qv in qvecs[:len(allq)]:
    rows = knn(db, 'chunks', qv, len(chunks))
    sv_order = [r - 1 for r, _ in rows]
    cos = vecs @ qv
    np_order = list(np.argsort(-cos, kind='stable'))
    same_order += sv_order == np_order
    same_top3 += sv_order[:3] == np_order[:3]
    max_diff = max(max_diff, max(abs((1 - d) - cos[r - 1]) for r, d in rows))
say(f'identical full ranking: {same_order}/{len(allq)}, identical top 3: {same_top3}/{len(allq)}')
say(f'largest |(1 - sqlite-vec distance) - numpy cosine|: {max_diff:.2e}')


def hits(table='chunks', quant='', where='', qs=questions, k_search=3, rerank=False):
    """doc-level and section-level Hit@1 / Hit@3 over qs."""
    d1 = d3 = s1 = s3 = 0
    for q in qs:
        qv = qvecs[allq.index(q)]
        rows = knn(db, table, qv, 30 if rerank else k_search, where, quant)
        ids = [r - 1 for r, _ in rows]
        if rerank:          # coarse search on bits, then exact float cosine on the 30 candidates
            ids = sorted(ids, key=lambda i: -float(vecs[i] @ qv))
        ids = ids[:3]
        doc_ok = [chunks[i][0] in q['docs'] for i in ids]
        sec_ok = [chunks[i][0] in q['docs'] and chunks[i][1] == q['section'] for i in ids]
        d1 += doc_ok[:1] == [True]
        d3 += any(doc_ok)
        s1 += sec_ok[:1] == [True]
        s3 += any(sec_ok)
    n = len(qs)
    return f'{d1:>3}/{n} {d3:>3}/{n} {s1:>3}/{n} {s3:>3}/{n}'


say(f'\n{"Index":<34} {"Doc@1":>6} {"Doc@3":>6} {"Sec@1":>6} {"Sec@3":>6}')
say(f'{"float32 (dense only, no BM25)":<34} {hits()}')

# --- Part C: persistence and a wrong-model query ---
say('\n## Part C: reopen without re-embedding\n')
db.close()
t = time.perf_counter()
db = connect(DB)
meta = dict(db.execute('select key, value from meta'))
rows = knn(db, 'chunks', qvecs[0], 1)
doc, sec = db.execute('select doc, section from chunks where rowid = ?', (rows[0][0],)).fetchone()
say(f'reopened and answered in {(time.perf_counter() - t) * 1000:.1f} ms; meta: {meta}')
say(f'"{allq[0]["q"][:50]}" -> {doc}/{sec}')
try:
    knn(db, 'chunks', np.ones(768) / np.sqrt(768), 1)            # a 768-dim (embeddinggemma-sized) query
    say('768-dim query against the 1,024-dim index: no error (!)')
except sqlite3.Error as e:
    say(f'768-dim query against the 1,024-dim index: {type(e).__name__}: {e}')
say('A different model with the same dimension count would pass this check silently: only meta.model tells.')

# --- Part D: metadata filter ---
say('\n## Part D: metadata filter (questions whose answer is in a Fix section)\n')
fixq = [q for q in questions if q['section'] == 'Fix']
say(f'{"Index":<34} {"Doc@1":>6} {"Doc@3":>6} {"Sec@1":>6} {"Sec@3":>6}')
say(f'{"float32, all sections":<34} {hits(qs=fixq)}')
only_fix = "and section = 'Fix'"
say(f'{"float32, where section = Fix":<34} {hits(qs=fixq, where=only_fix)}')

# --- Part E: quantized vectors ---
say('\n## Part E: quantized vectors, 51 questions\n')
say(f'{"Index (bytes per vector)":<34} {"Doc@1":>6} {"Doc@3":>6} {"Sec@1":>6} {"Sec@3":>6}')
say(f'{f"float32 ({dims * 4})":<34} {hits()}')
say(f'{f"int8 ({dims})":<34} {hits("chunks_i8", "int8")}')
say(f'{f"bit ({dims // 8})":<34} {hits("chunks_bit", "bit")}')
say(f'{"bit top 30, re-ranked with float32":<34} {hits("chunks_bit", "bit", rerank=True)}')
db.close()

# --- Part F: scale ---
say('\n## Part F: 50,000 random 1,024-dim vectors (in memory), 20 queries each\n')
rng = np.random.default_rng(0)
big = rng.normal(size=(50_000, dims)).astype(np.float32)
big /= np.linalg.norm(big, axis=1, keepdims=True)
probes = big[rng.choice(len(big), 20, replace=False)]
mem = connect(':memory:')
mem.execute(f'create virtual table b using vec0(emb float[{dims}] distance_metric=cosine)')
mem.execute(f'create virtual table bb using vec0(emb bit[{dims}])')
t = time.perf_counter()
mem.executemany('insert into b(rowid, emb) values (?, ?)', ((i + 1, v.tobytes()) for i, v in enumerate(big)))
t_ins = time.perf_counter() - t
mem.executemany('insert into bb(rowid, emb) values (?, vec_quantize_binary(?))', ((i + 1, v.tobytes()) for i, v in enumerate(big)))
mem.commit()
say(f'insert 50,000 float32 vectors: {t_ins:.1f} s ({len(big) / t_ins:,.0f}/s)')
for name, fn in [('sqlite-vec float32, k=10', lambda p: knn(mem, 'b', p, 10)),
                 ('sqlite-vec bit, k=10', lambda p: knn(mem, 'bb', p, 10, quant='bit')),
                 ('numpy float32 dot + argpartition', lambda p: np.argpartition(-(big @ p), 10)[:10])]:
    t = time.perf_counter()
    for p in probes:
        fn(p)
    say(f'{name:<34} {(time.perf_counter() - t) * 1000 / len(probes):>7.1f} ms/query')
self_hit = sum(knn(mem, 'b', p, 1)[0][0] - 1 == int(np.argmax(big @ p)) for p in probes)
say(f'sqlite-vec top-1 = exact top-1 (the probe itself): {self_hit}/{len(probes)}')

out = Path('results/raw') / f'vector-store-{datetime.now():%Y%m%d-%H%M%S}.md'
out.write_text('\n'.join(report) + '\n', encoding='utf-8')
say(f'\nSaved to {out}')
