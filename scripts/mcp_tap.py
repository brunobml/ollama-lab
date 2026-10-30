"""Lab 06 Step 4. A wire tap for MCP over stdio: sits between client and server and logs every JSON-RPC line.

The client launches this script instead of the server; it launches the real server and copies bytes both ways,
appending each line to the log file named by MCP_TAP_LOG (default results/raw/mcp-wire.jsonl) as
{"dir": "client->server" | "server->client", "t": seconds, "msg": <the JSON-RPC message>}.
Like tcpdump on the pipe: nothing is changed, so what you see is what the SDK really sends.

    python scripts/mcp_tap.py python scripts/mcp_server.py
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

LOG = Path(os.environ.get('MCP_TAP_LOG', 'results/raw/mcp-wire.jsonl'))
LOG.parent.mkdir(parents=True, exist_ok=True)
lock, t0 = threading.Lock(), time.perf_counter()


def record(direction, line):
    try:
        msg = json.loads(line)
    except ValueError:
        msg = {'non_json': line.decode('utf-8', 'replace')}
    with lock, LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'dir': direction, 't': round(time.perf_counter() - t0, 4), 'msg': msg}) + '\n')


def pump(src, dst, direction):
    for line in iter(src.readline, b''):
        record(direction, line)
        dst.write(line)
        dst.flush()
    dst.close()


child = subprocess.Popen(sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)   # stderr passes through
threading.Thread(target=pump, args=(sys.stdin.buffer, child.stdin, 'client->server'), daemon=True).start()
pump(child.stdout, sys.stdout.buffer, 'server->client')
sys.exit(child.wait())
