"""Lab 08 Step 4. A metrics sidecar for Ollama: a reverse proxy that measures every request and serves /metrics.

Ollama 0.40.13 has no metrics endpoint (/metrics → 404). Its access log ([GIN]) has status, latency, client and path,
but not the model, tokens, time to first token or cancellations. This proxy sits in front of it, like a
service-mesh sidecar, and records what a client sees plus the timing fields Ollama puts in the final response.

Standalone:
    .\\.venv\\Scripts\\python scripts\\metrics_proxy.py --upstream http://127.0.0.1:11438 --port 11439
then point clients at http://127.0.0.1:11439 and scrape http://127.0.0.1:11439/metrics (Prometheus text format).
It binds to 127.0.0.1 only. scripts/observe_ollama.py imports it and runs it in a thread.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)
INFERENCE = ('/api/generate', '/api/chat')


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.records = []          # one dict per proxied request, for scripts that want raw data
        self.in_flight = 0

    def add(self, rec):
        with self.lock:
            self.records.append(rec)

    def render(self, ps_models):
        """Prometheus text exposition format, computed from the records on every scrape."""
        with self.lock:
            recs, in_flight = list(self.records), self.in_flight
        out = []

        def counter(name, help_, values):
            out.extend([f'# HELP {name} {help_}', f'# TYPE {name} counter'])
            out.extend(f'{name}{{{labels}}} {v}' for labels, v in sorted(values.items()))

        def histogram(name, help_, samples):
            out.extend([f'# HELP {name} {help_}', f'# TYPE {name} histogram'])
            by = {}
            for labels, v in samples:
                by.setdefault(labels, []).append(v)
            for labels, vs in sorted(by.items()):
                for b in BUCKETS:
                    out.append(f'{name}_bucket{{{labels},le="{b}"}} {sum(v <= b for v in vs)}')
                out.append(f'{name}_bucket{{{labels},le="+Inf"}} {len(vs)}')
                out.append(f'{name}_sum{{{labels}}} {round(sum(vs), 6)}')
                out.append(f'{name}_count{{{labels}}} {len(vs)}')

        lab = lambda r: f'endpoint="{r["endpoint"]}",model="{r["model"]}"'
        req = {}
        for r in recs:
            k = f'{lab(r)},status="{r["status"]}"'
            req[k] = req.get(k, 0) + 1
        counter('ollama_proxy_requests_total', 'Requests by endpoint, model and HTTP status.', req)
        dis = {}
        for r in recs:
            dis[lab(r)] = dis.get(lab(r), 0) + (1 if r['client_disconnected'] else 0)
        counter('ollama_proxy_client_disconnects_total', 'Requests whose client went away before the end.', dis)
        tok = {}
        for r in recs:
            for kind, key in (('prompt', 'prompt_eval_count'), ('completion', 'eval_count')):
                if r.get(key) is not None:
                    k = f'{lab(r)},kind="{kind}"'
                    tok[k] = tok.get(k, 0) + r[key]
        counter('ollama_proxy_tokens_total', 'Tokens reported by Ollama (prompt = evaluated, not cached).', tok)
        histogram('ollama_proxy_request_seconds', 'Wall time seen by the proxy.',
                  [(lab(r), r['wall_s']) for r in recs])
        histogram('ollama_proxy_ttft_seconds', 'Time to the first streamed chunk (streaming requests).',
                  [(lab(r), r['ttft_s']) for r in recs if r.get('ttft_s') is not None])
        histogram('ollama_proxy_queue_seconds', "Ollama's total_duration minus load, prompt eval and eval.",
                  [(lab(r), r['queue_s']) for r in recs if r.get('queue_s') is not None])
        histogram('ollama_proxy_load_seconds', 'Model load time reported by Ollama (> 0 means a cold start).',
                  [(lab(r), r['load_s']) for r in recs if r.get('load_s') is not None])
        out.extend(['# HELP ollama_proxy_in_flight Requests being proxied now.', '# TYPE ollama_proxy_in_flight gauge',
                    f'ollama_proxy_in_flight {in_flight}'])
        out.extend(['# HELP ollama_loaded_model_vram_bytes size_vram from /api/ps.',
                    '# TYPE ollama_loaded_model_vram_bytes gauge'])
        out.extend(f'ollama_loaded_model_vram_bytes{{model="{m["name"]}"}} {m.get("size_vram", 0)}' for m in ps_models)
        return '\n'.join(out) + '\n'


def summarize(final):
    """Ollama's timing fields (nanoseconds) → seconds; queue = what total_duration has that the phases don't."""
    ns = lambda k: final.get(k, 0) / 1e9
    rec = {k: final.get(k) for k in ('prompt_eval_count', 'eval_count')}
    if 'total_duration' in final:
        rec.update({'total_s': round(ns('total_duration'), 4), 'load_s': round(ns('load_duration'), 4),
                    'prompt_eval_s': round(ns('prompt_eval_duration'), 4), 'eval_s': round(ns('eval_duration'), 4)})
        rec['queue_s'] = round(max(0.0, rec['total_s'] - rec['load_s'] - rec['prompt_eval_s'] - rec['eval_s']), 4)
    return rec


def make_handler(upstream, metrics, client):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'      # the body ends when the connection closes: no chunked re-encoding

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == '/metrics':
                try:
                    ps = client.get(upstream + '/api/ps', timeout=5).json().get('models', [])
                except httpx.HTTPError:
                    ps = []
                body = metrics.render(ps).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.forward()

        def do_POST(self):
            self.forward()

        do_DELETE = do_HEAD = do_PUT = do_POST

        def forward(self):
            body = self.rfile.read(int(self.headers.get('Content-Length') or 0))
            try:
                req = json.loads(body) if body else {}
            except ValueError:
                req = {}
            rec = {'endpoint': self.path.split('?')[0], 'model': req.get('model', ''), 'status': 0,
                   'stream': req.get('stream', True) if self.path in INFERENCE else False,
                   'client_disconnected': False, 'ttft_s': None, 'started': time.time()}
            with metrics.lock:
                metrics.in_flight += 1
            t0 = time.perf_counter()
            last = b''
            try:
                with client.stream(self.command, upstream + self.path, content=body,
                                   headers={'Content-Type': self.headers.get('Content-Type', 'application/json')}) as r:
                    rec['status'] = r.status_code
                    self.send_response(r.status_code)
                    self.send_header('Content-Type', r.headers.get('Content-Type', 'application/json'))
                    self.end_headers()
                    for line in r.iter_lines():
                        if rec['stream'] and rec['ttft_s'] is None and line:
                            rec['ttft_s'] = round(time.perf_counter() - t0, 4)
                        last = line.encode() if line else last
                        try:
                            self.wfile.write(line.encode() + b'\n')
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            rec['client_disconnected'] = True
                            break                  # leaving the block closes the upstream connection
            except httpx.HTTPError as e:
                rec['status'], rec['error'] = 502, str(e)
                try:
                    self.send_error(502)
                except OSError:
                    pass
            finally:
                rec['wall_s'] = round(time.perf_counter() - t0, 4)
                try:
                    final = json.loads(last) if last else {}
                except ValueError:
                    final = {}
                if rec['endpoint'] in INFERENCE:
                    rec.update(summarize(final))
                    if 'error' in final:
                        rec['error'] = final['error']
                with metrics.lock:
                    metrics.in_flight -= 1
                if rec['endpoint'] in INFERENCE:
                    metrics.add(rec)

    return Handler


def start(upstream, port):
    """Start the proxy on 127.0.0.1:port in a daemon thread; returns (server, metrics)."""
    metrics = Metrics()
    client = httpx.Client(timeout=None)      # one client: a new one costs ≈0.3 s (SSL context), measured in Step 4
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(upstream.rstrip('/'), metrics, client))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, metrics


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', default='http://127.0.0.1:11434')
    ap.add_argument('--port', type=int, default=11439)
    a = ap.parse_args()
    srv, _ = start(a.upstream, a.port)
    print(f'proxying 127.0.0.1:{a.port} → {a.upstream}; metrics at http://127.0.0.1:{a.port}/metrics (Ctrl+C stops)')
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()
