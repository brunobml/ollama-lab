"""Lab 06 Step 5. A logging reverse proxy between Claude Code and Ollama's Anthropic-compatible API.

`ollama launch claude` points Claude Code at ANTHROPIC_BASE_URL = OLLAMA_HOST. Set OLLAMA_HOST to this proxy in the
same PowerShell session, and every request (from Claude Code and from the ollama CLI) passes through here:

    .\\.venv\\Scripts\\python scripts\\anthropic_tap.py --label c1          # terminal 1: listen on 127.0.0.1:11436
    $env:OLLAMA_HOST = '127.0.0.1:11436'                                      # terminal 2, this session only
    ollama launch claude --model qwen3.5:4b-ctx64k -- -p "..."

Like scripts/mcp_tap.py, but for HTTP: nothing is changed, only copied and logged. Streaming (SSE) responses are
passed through as they arrive. For each /v1/messages request it prints and logs the request's shape (messages,
tools, system prompt size, thinking, max_tokens) and the response's usage (input/output tokens), stop reason and
tool calls. Ctrl+C prints a summary.

Log: results/raw/cc-tap-<label>-<stamp>.jsonl (gitignored). It includes the full request and response bodies, which
contain Claude Code's system prompt and whatever files it read: keep the sandbox free of secrets.
    .\\.venv\\Scripts\\python scripts\\anthropic_tap.py --summary results\\raw\\cc-tap-c1-....jsonl
"""
import argparse
import http.client
import json
import threading
import time
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = ('127.0.0.1', 11434)
HOP = {'host', 'connection', 'keep-alive', 'transfer-encoding', 'content-length', 'accept-encoding', 'upgrade'}
lock, t0, records, log_path = threading.Lock(), time.perf_counter(), [], None


def request_shape(body):
    """What Claude Code sent: sizes and names, not the content."""
    msgs = body.get('messages', [])
    system = body.get('system', '')
    system_chars = len(system) if isinstance(system, str) else sum(len(b.get('text', '')) for b in system)
    last = msgs[-1]['content'] if msgs else ''
    blocks = last if isinstance(last, list) else []
    results = [b for b in blocks if b.get('type') == 'tool_result']
    return {'model': body.get('model'), 'stream': body.get('stream', False), 'max_tokens': body.get('max_tokens'),
            'thinking': body.get('thinking'), 'messages': len(msgs), 'system_chars': system_chars,
            'tools': len(body.get('tools', [])), 'tool_names': [t.get('name') for t in body.get('tools', [])],
            'tool_results': len(results), 'tool_result_errors': sum(bool(b.get('is_error')) for b in results)}


def response_shape(raw, content_type):
    """Usage, stop reason, tool calls and text from an SSE stream or a plain JSON response."""
    out = {'input_tokens': None, 'cached_tokens': 0, 'output_tokens': None, 'stop_reason': None, 'tool_calls': [],
           'text': ''}
    text = raw.decode('utf-8', 'replace')
    events = []
    if 'event-stream' in content_type:
        events = [json.loads(l[5:]) for l in text.splitlines() if l.startswith('data:') and l[5:].strip()[:1] == '{']
    else:
        try:
            msg = json.loads(text)
        except ValueError:
            return out
        events = [{'type': 'message_start', 'message': msg}] + \
                 [{'type': 'content_block_start', 'content_block': b} for b in msg.get('content', [])] + \
                 [{'type': 'message_delta', 'delta': {'stop_reason': msg.get('stop_reason')}, 'usage': msg.get('usage', {})}]
        out['text'] = ''.join(b.get('text', '') for b in msg.get('content', []) if b.get('type') == 'text')
    for e in events:
        if e.get('type') == 'message_start':
            out['input_tokens'] = e['message'].get('usage', {}).get('input_tokens')
        elif e.get('type') == 'content_block_start' and e['content_block'].get('type') == 'tool_use':
            out['tool_calls'].append(e['content_block'].get('name'))
        elif e.get('type') == 'content_block_delta' and e['delta'].get('type') == 'text_delta':
            out['text'] += e['delta'].get('text', '')
        elif e.get('type') == 'message_delta':
            out['stop_reason'] = e.get('delta', {}).get('stop_reason') or out['stop_reason']
            out['output_tokens'] = e.get('usage', {}).get('output_tokens', out['output_tokens'])
            # Ollama's message_start carries an early estimate; the final count arrives here (272 vs 32 in a test).
            # Anthropic-style usage: input_tokens = prompt tokens processed now, cache_read_input_tokens = prefix reused
            # from the KV cache. The whole prompt is the sum.
            out['input_tokens'] = e.get('usage', {}).get('input_tokens', out['input_tokens'])
            out['cached_tokens'] = e.get('usage', {}).get('cache_read_input_tokens', out['cached_tokens']) or 0
        elif e.get('type') == 'error':
            out['error'] = e.get('error')
    return out


class Tap(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):      # the one-line summary below replaces the default access log
        pass

    def forward(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else b''
        start = time.perf_counter()
        up = http.client.HTTPConnection(*UPSTREAM, timeout=900)
        up.request(self.command, self.path, body, {k: v for k, v in self.headers.items() if k.lower() not in HOP})
        resp = up.getresponse()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header('Connection', 'close')        # the body ends when we close: no length needed for streams
        self.end_headers()
        chunks, first, gone = [], None, False
        while self.command != 'HEAD':
            data = resp.read1(65536)
            if not data:
                break
            first = first or time.perf_counter() - start
            chunks.append(data)
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except OSError:                             # client hung up (Esc in Claude Code): stop relaying
                gone = True
                break
        up.close()
        self.close_connection = True
        record(self.command, self.path, resp.status, body, b''.join(chunks), resp.getheader('Content-Type') or '',
               self.headers, start, first, gone)

    do_GET = do_POST = do_HEAD = do_DELETE = do_PUT = forward


def record(method, path, status, body, raw, content_type, headers, start, first, gone):
    rec = {'t': round(start - t0, 2), 'method': method, 'path': path, 'status': status,
           'seconds': round(time.perf_counter() - start, 2), 'first_byte_s': round(first, 2) if first else None,
           'client_gone': gone, 'user_agent': headers.get('User-Agent'),
           'auth': 'bearer' if headers.get('Authorization') else ('x-api-key' if headers.get('x-api-key') else None),
           'anthropic_beta': headers.get('anthropic-beta')}
    is_messages = path.split('?')[0] == '/v1/messages' and method == 'POST'
    if is_messages:
        try:
            req = json.loads(body)
            rec.update(request_shape(req))
        except ValueError:
            req = None
        rec.update(response_shape(raw, content_type))
        rec['request_body'], rec['response_raw'] = req, raw.decode('utf-8', 'replace')
    with lock:
        rec['n'] = len(records) + 1
        records.append(rec)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
    if is_messages:
        calls = ','.join(rec['tool_calls']) or '-'
        print(f"#{rec['n']:<3} {rec['seconds']:6.1f}s  prompt {prompt_tokens(rec):>6,} (new {rec['input_tokens'] or 0:>6,})  "
              f"out {rec['output_tokens'] or 0:>5,}  "
              f"msgs {rec['messages']:>3}  tools {rec['tools']:>3}  results {rec['tool_results']}"
              f"{'(' + str(rec['tool_result_errors']) + ' err)' if rec['tool_result_errors'] else ''}  "
              f"stop {rec['stop_reason']}  -> {calls}{'  [client hung up]' if gone else ''}"
              f"{'  ERROR ' + str(rec.get('error')) if rec.get('error') or status >= 400 else ''}", flush=True)
    else:
        print(f"     {method} {path} {status} ({rec['seconds']} s)", flush=True)


def prompt_tokens(rec):
    return (rec.get('input_tokens') or 0) + (rec.get('cached_tokens') or 0)


def summary(recs):
    m = [r for r in recs if r['path'].split('?')[0] == '/v1/messages' and r['method'] == 'POST']
    if not m:
        return 'no /v1/messages requests'
    for r in m:                      # re-parse saved responses, so older logs get the current counting
        if 'response_raw' in r:
            r.update(response_shape(r['response_raw'].encode('utf-8'), 'event-stream' if r.get('stream') else 'json'))
    calls = Counter(c for r in m for c in r.get('tool_calls', []))
    models = Counter(r.get('model') for r in m)
    ins = [prompt_tokens(r) for r in m]
    first = m[0]
    lines = [f"/v1/messages requests: {len(m)} (models: {dict(models)}), other requests: {len(recs) - len(m)}",
             f"first request: system prompt {first.get('system_chars', 0):,} chars, {first.get('tools', 0)} tools, "
             f"{prompt_tokens(first):,} prompt tokens ({first.get('first_byte_s')} s to first byte)",
             f"prompt tokens: max {max(ins):,}; processed new {sum(r.get('input_tokens') or 0 for r in m):,}, "
             f"reused from cache {sum(r.get('cached_tokens') or 0 for r in m):,}; output tokens total "
             f"{sum(r.get('output_tokens') or 0 for r in m):,}",
             f"tool calls: {sum(calls.values())} {dict(calls)}; tool results marked is_error: "
             f"{sum(r.get('tool_result_errors', 0) for r in m)}",
             f"model time: {sum(r['seconds'] for r in m):.1f} s; errors (HTTP >= 400 or error event): "
             f"{sum(1 for r in m if r['status'] >= 400 or r.get('error'))}"]
    final = next((r for r in reversed(m) if r.get('stop_reason') == 'end_turn'), None)
    if final:
        lines.append('last answer: ' + ' '.join(final.get('text', '').split())[:400])
    return '\n'.join(lines)


def main():
    global log_path
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=11436)
    ap.add_argument('--label', default='session', help='goes into the log file name, e.g. c1')
    ap.add_argument('--summary', help='print the summary of a saved log and exit')
    args = ap.parse_args()
    if args.summary:
        print(summary([json.loads(l) for l in open(args.summary, encoding='utf-8')]))
        return
    log_path = Path('results/raw') / f'cc-tap-{args.label}-{datetime.now():%Y%m%d-%H%M%S}.jsonl'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Tap)     # localhost only (guardrail 5)
    print(f'anthropic_tap: 127.0.0.1:{args.port} -> {UPSTREAM[0]}:{UPSTREAM[1]}, logging to {log_path}\n'
          f"In the other terminal: $env:OLLAMA_HOST = '127.0.0.1:{args.port}'   Ctrl+C here for the summary.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print('\n' + summary(records))


if __name__ == '__main__':
    main()
