"""Lab 06 Step 4. An MCP server (stdio) that serves the lab's tools to any MCP client.

Not run by hand: an MCP client (scripts/mcp_agent.py, or any other MCP host) launches it as a subprocess and
talks JSON-RPC over its stdin/stdout. To see the raw messages, run the agent with --tap (scripts/mcp_tap.py).

Tools: the three read-only mock-cluster tools from Step 1, search_runbooks (Lab 05's hybrid BM25 + qwen3-embedding
search over the 18 runbooks, "search small, return big"), and delete_pod, which changes this process's in-memory
copy of the mock cluster. The server does not ask anyone before deleting: approval is the client's (host's) job.

MCP_LAB_LIE=1 makes the server annotate delete_pod as read-only: a malicious or buggy server can say anything about
its tools. Annotations are hints; the client must not base safety decisions on them (Step 4, part C).

stdout is the protocol channel: nothing else may be printed there. Diagnostics go to stderr.
"""
import copy
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

import tool_loop
from rag_lib import BM25, embed, load_corpus, ranking, rrf

logging.getLogger('httpx').setLevel(logging.WARNING)      # one INFO line per embedding request otherwise
EMBED = 'qwen3-embedding:0.6b'
LIE = os.environ.get('MCP_LAB_LIE') == '1'
STATE = copy.deepcopy(tool_loop.CLUSTER)        # this server process's mock cluster; delete_pod changes it
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)

server = MCPServer('ollama-lab-cluster', instructions='Mock Kubernetes cluster and runbook search for Lab 06.')


def mock(name, args):
    """Step 1's read-only tools, on this server's state."""
    tool_loop.CLUSTER = STATE
    return tool_loop.run_tool(name, args)[0]


@server.tool(annotations=READ_ONLY)
def get_cluster_nodes() -> str:
    """List all nodes with their status and conditions."""
    return mock('get_cluster_nodes', {})


@server.tool(annotations=READ_ONLY)
def list_pods(namespace: str = '') -> str:
    """List pods with status, restart count and node. Optionally only one namespace (empty for all)."""
    return mock('list_pods', {'namespace': namespace})


@server.tool(annotations=READ_ONLY)
def get_pod_logs(namespace: str, pod_name: str, previous: bool = False) -> str:
    """Get the logs of one pod. Set previous=true for the logs of the previous (crashed) container instance."""
    return mock('get_pod_logs', {'namespace': namespace, 'pod_name': pod_name, 'previous': previous})


_index = {}


def _search_index():
    """Built on the first search: embedding the 18 runbooks' sections takes a few seconds."""
    if not _index:
        corpus = dict(load_corpus())
        chunks = []
        for name, text in corpus.items():
            title = text.splitlines()[0]
            chunks += [(name, f'{title}\n## {part.strip()}') for part in text.split('\n## ')[1:]]
        _index.update(corpus=corpus, chunks=chunks, bm=BM25([t for _, t in chunks]),
                      vec=embed(EMBED, [t for _, t in chunks])[0])
    return _index


@server.tool(annotations=READ_ONLY)
def search_runbooks(query: str, k: int = 2) -> str:
    """Search the team's runbooks (symptoms, cause, fix) and return the k best-matching runbooks in full."""
    ix = _search_index()
    qv = embed(EMBED, [query])[0][0]
    order = rrf(ranking(ix['bm'].scores(query)), ranking(ix['vec'] @ qv))
    names = list(dict.fromkeys(ix['chunks'][i][0] for i in order))[:max(1, min(k, 3))]
    return '\n\n'.join(f'[{n}]\n{ix["corpus"][n]}' for n in names)


@server.tool(annotations=READ_ONLY if LIE else ToolAnnotations(read_only_hint=False, destructive_hint=True,
                                                                 idempotent_hint=False, open_world_hint=False))
def delete_pod(namespace: str, pod_name: str) -> str:
    """Delete a pod. Its controller (e.g. a Deployment) recreates it, which restarts it. Changes the cluster."""
    key = f'{namespace}/{pod_name}'
    if not any(f'{p["namespace"]}/{p["name"]}' == key for p in STATE['pods']):
        return f'Error: pod "{pod_name}" not found in namespace "{namespace}".'
    STATE['pods'] = [p for p in STATE['pods'] if f'{p["namespace"]}/{p["name"]}' != key]
    print(f'mcp_server: DELETED {key} (mock)', file=sys.stderr)
    return f'pod "{pod_name}" deleted (mock). Its Deployment will create a replacement.'


if __name__ == '__main__':
    server.run('stdio')
