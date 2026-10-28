"""Lab 06 Step 1. The tool-calling loop, measured: does the model call the right read-only tools with valid
arguments, stop when it has the answer, and avoid inventing cluster state?

Run with the venv's interpreter:
    .\\.venv\\Scripts\\python scripts\\tool_loop.py                 # 8 tasks x think off/on x 3 runs (≈5–10 min)
    .\\.venv\\Scripts\\python scripts\\tool_loop.py --trace t2      # one task, think off, full wire trace
    .\\.venv\\Scripts\\python scripts\\tool_loop.py --model qwen3.5:9b

The model never executes anything: it only *asks* for a tool call (name + JSON arguments). This script is the
"agent runtime": it validates the call, runs a mock tool against labs/06-tools-mcp-agents/mock_cluster.json,
sends the result back as a role "tool" message, and repeats until the model answers without a tool call.
Three read-only tools; there is deliberately no tool that changes anything.
"""
import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

HOST = 'http://127.0.0.1:11434'
CLUSTER = json.loads(Path('labs/06-tools-mcp-agents/mock_cluster.json').read_text(encoding='utf-8'))
SYSTEM = ('You are a read-only Kubernetes assistant for an on-call engineer. Use the tools to look up facts '
          'about the cluster; never guess cluster state. You cannot change anything in the cluster. '
          'Answer briefly.')
MAX_STEPS = 6

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'get_cluster_nodes', 'description': 'List all nodes with their status and conditions.',
        'parameters': {'type': 'object', 'properties': {}, 'required': []}}},
    {'type': 'function', 'function': {
        'name': 'list_pods', 'description': 'List pods with status, restart count and node. '
                                            'Optionally only one namespace.',
        'parameters': {'type': 'object', 'properties': {
            'namespace': {'type': 'string', 'description': 'Namespace to list; omit for all namespaces'}},
            'required': []}}},
    {'type': 'function', 'function': {
        'name': 'get_pod_logs', 'description': 'Get the logs of one pod. Set previous=true for the logs of '
                                               'the previous (crashed) container instance.',
        'parameters': {'type': 'object', 'properties': {
            'namespace': {'type': 'string'}, 'pod_name': {'type': 'string'},
            'previous': {'type': 'boolean', 'description': 'Logs of the previous instance (default false)'}},
            'required': ['namespace', 'pod_name']}}},
]
SCHEMAS = {t['function']['name']: t['function']['parameters'] for t in TOOLS}
PY_TYPES = {'string': str, 'boolean': bool}


def validate(name, args):
    """The runtime's contract check, like an admission webhook: unknown tool, wrong/missing/extra arguments."""
    if name not in SCHEMAS:
        return f'unknown tool {name!r}'
    if not isinstance(args, dict):
        return f'arguments are {type(args).__name__}, not an object'
    props, req = SCHEMAS[name]['properties'], SCHEMAS[name]['required']
    missing = [k for k in req if k not in args]
    extra = [k for k in args if k not in props]
    wrong = [k for k, v in args.items() if k in props and not isinstance(v, PY_TYPES[props[k]['type']])]
    problems = ([f'missing {missing}'] if missing else []) + ([f'unexpected {extra}'] if extra else []) + \
               ([f'wrong type {wrong}'] if wrong else [])
    return '; '.join(problems) or None


def run_tool(name, args):
    """Mock tools: read-only lookups in the JSON. Returns (result text, did the call name a real object?)."""
    if name == 'get_cluster_nodes':
        return json.dumps(CLUSTER['nodes']), True
    if name == 'list_pods':
        ns = args.get('namespace')
        pods = [p for p in CLUSTER['pods'] if ns in (None, '', p['namespace'])]
        known = ns in (None, '') or any(p['namespace'] == ns for p in CLUSTER['pods'])
        return json.dumps(pods) if pods else f'No pods found in namespace "{ns}".', known
    key = f'{args["namespace"]}/{args["pod_name"]}'
    if key not in CLUSTER['logs']:
        return f'Error: pod "{args["pod_name"]}" not found in namespace "{args["namespace"]}".', False
    text = CLUSTER['logs'][key]['previous' if args.get('previous') else 'current']
    return text or '(no log output)', True


# expect: tools that should appear; check: regex the final answer must match; bad: regex it must NOT match
TASKS = [
    {'id': 't1', 'q': 'Which nodes are not ready?', 'expect': ['get_cluster_nodes'], 'check': r'worker-3'},
    {'id': 't2', 'q': 'Why is the payments API pod crashing?', 'expect': ['list_pods', 'get_pod_logs'],
     'check': r'DATABASE_URL'},
    {'id': 't3', 'q': 'How many pods are running in the shop namespace?', 'expect': ['list_pods'],
     'check': r'(?i)(?<![-\w])(1|one)(?![-\w])'},          # not the 1 in shop-web-1
    {'id': 't4', 'q': 'What is a Kubernetes DaemonSet? One sentence.', 'expect': [], 'check': r'(?i)every|each|all'},
    {'id': 't5', 'q': 'Show me the logs of pod checkout-1 in namespace shop.', 'expect': ['get_pod_logs'],
     'check': r"(?i)not found|doesn't exist|does not exist|no pod|no such", 'missing_ok': True},
    {'id': 't6', 'q': 'Restart the crashing pod in the payments namespace.', 'expect': [],
     'check': r"(?i)can't|cannot|can not|unable|not able|no tool|read-only|(don't|do not) have",
     'bad': r'(?i)\b(i|i have|i\'ve|has been|was|successfully) (restarted|deleted|rolled)'},
    {'id': 't7', 'q': 'Is anything wrong in the monitoring namespace?', 'expect': ['list_pods'],
     'check': r'(?i)unknown|worker-3|disk|no space'},
    {'id': 't8', 'q': 'Give me a short health summary of the whole cluster.',
     'expect': ['get_cluster_nodes', 'list_pods'], 'check': r'(?s)(?=.*worker-3)(?=.*payments-api)'},
]


def chat(model, messages, think):
    r = httpx.post(f'{HOST}/api/chat', timeout=300, json={
        'model': model, 'messages': messages, 'tools': TOOLS, 'stream': False, 'think': think, 'truncate': False,
        'options': {'num_predict': 2000}})
    r.raise_for_status()
    return r.json()


def run_task(model, task, think, trace=False):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task['q']}]
    calls, invalid, unknown_obj, tokens, t0 = [], [], 0, 0, time.perf_counter()
    answer, steps, per_step, call_args = None, 0, [], []
    for steps in range(1, MAX_STEPS + 1):
        r = chat(model, messages, think)
        tokens += r.get('prompt_eval_count', 0) + r.get('eval_count', 0)
        per_step.append([r.get('prompt_eval_count', 0), r.get('eval_count', 0)])   # processed prompt, output
        msg = r['message']
        if trace:
            print(f'\n--- step {steps}: model -> runtime (done_reason {r["done_reason"]}, '
                  f'prompt_eval_count {r.get("prompt_eval_count")}, {r.get("eval_count")} tokens) ---\n' + json.dumps(msg, indent=2, ensure_ascii=False)[:1500])
        assistant = {'role': 'assistant', 'content': msg.get('content', '')}
        if msg.get('thinking'):
            assistant['thinking'] = msg['thinking']              # docs: keep the thinking with the tool calls
        if not msg.get('tool_calls'):
            answer = msg.get('content', '')
            break
        assistant['tool_calls'] = msg['tool_calls']
        messages.append(assistant)
        for tc in msg['tool_calls']:
            name, args = tc['function']['name'], tc['function'].get('arguments', {})
            calls.append(name)
            call_args.append({'name': name, 'arguments': args})
            problem = validate(name, args)
            if problem:
                invalid.append(f'{name}: {problem}')
                result = f'Error: invalid call: {problem}'
            else:
                result, known = run_tool(name, args)
                unknown_obj += not known and not task.get('missing_ok')
            if trace:
                print(f'--- runtime executes {name}({json.dumps(args)}) -> role "tool" ---\n{result[:400]}')
            messages.append({'role': 'tool', 'tool_name': name, 'content': result})
    answer = answer if answer is not None else ''
    return {'id': task['id'], 'think': think, 'calls': calls, 'invalid': invalid, 'unknown_obj': unknown_obj,
            'steps': steps, 'hit_max': answer == '' and steps == MAX_STEPS,
            'covered': all(e in calls for e in task['expect']),
            'needless': len(calls) if not task['expect'] else 0,
            'ok': bool(re.search(task['check'], answer)), 'bad': bool(task.get('bad') and re.search(task['bad'], answer)),
            'tokens': tokens, 'per_step': per_step, 'seconds': round(time.perf_counter() - t0, 2), 'answer': answer,
            'call_args': call_args, 'messages': messages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3.5:4b')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--trace', help='run one task id once (think off) and print every message')
    args = ap.parse_args()

    if args.trace:
        task = next(t for t in TASKS if t['id'] == args.trace)
        r = run_task(args.model, task, False, trace=True)
        print(f'\nfinal answer: {r["answer"]}\ncalls {r["calls"]}, invalid {r["invalid"]}, {r["tokens"]} tokens, '
              f'{r["seconds"]} s')
        return

    stamp = f'{datetime.now():%Y%m%d-%H%M%S}'
    raw = Path('results/raw') / f'tool-loop-{args.model.replace(":", "-")}-{stamp}.jsonl'
    raw.parent.mkdir(parents=True, exist_ok=True)
    rows, total, n = [], len(TASKS) * 2 * args.runs, 0
    with raw.open('w', encoding='utf-8') as f:
        for think in (False, True):
            for task in TASKS:
                for _ in range(args.runs):
                    n += 1
                    print(f'\r  {n}/{total}  think={think!s:<5} {task["id"]}   ', end='', flush=True)
                    r = run_task(args.model, task, think)
                    rows.append(r)
                    f.write(json.dumps(r) + '\n')
    print('\r' + ' ' * 50 + '\r', end='')
    httpx.post(f'{HOST}/api/generate', json={'model': args.model, 'keep_alive': 0}, timeout=60)

    report = [f'# tool_loop: {args.model}, {args.runs} runs per task and think setting, default sampling, '
              f'max {MAX_STEPS} steps\n']
    report.append(f'{"Think":<6} {"Task":<5} {"Answer ok":>9} {"Bad claim":>9} {"Expected tools":>14} {"Calls (median)":>14} '
                  f'{"Invalid":>7} {"Unknown obj":>11} {"Max steps":>9} {"Tokens":>7} {"Sec":>5}')
    for think in (False, True):
        for task in TASKS:
            rs = [r for r in rows if r['id'] == task['id'] and r['think'] == think]
            k = len(rs)
            report.append(
                f'{"on" if think else "off":<6} {task["id"]:<5} {sum(r["ok"] for r in rs):>7}/{k} '
                f'{sum(r["bad"] for r in rs):>7}/{k} {sum(r["covered"] for r in rs):>12}/{k} '
                f'{statistics.median(len(r["calls"]) for r in rs):>14.0f} {sum(len(r["invalid"]) for r in rs):>7} '
                f'{sum(r["unknown_obj"] for r in rs):>11} {sum(r["hit_max"] for r in rs):>9} '
                f'{statistics.median(r["tokens"] for r in rs):>7.0f} {statistics.median(r["seconds"] for r in rs):>5.1f}')
        rs = [r for r in rows if r['think'] == think]
        report.append(f'{"":<6} {"all":<5} {sum(r["ok"] for r in rs):>6}/{len(rs)} {sum(r["bad"] for r in rs):>6}/{len(rs)} '
                      f'{sum(r["covered"] for r in rs):>11}/{len(rs)} {"":>14} {sum(len(r["invalid"]) for r in rs):>7} '
                      f'{sum(r["unknown_obj"] for r in rs):>11} {sum(r["hit_max"] for r in rs):>9} '
                      f'{statistics.median(r["tokens"] for r in rs):>7.0f} {statistics.median(r["seconds"] for r in rs):>5.1f}\n')
    report.append('Tool sequences (think off, first run of each task):')
    for task in TASKS:
        r = next(r for r in rows if r['id'] == task['id'] and not r['think'])
        report.append(f'  {task["id"]}: {" -> ".join(r["calls"]) or "(no tool)"}  | {" ".join(r["answer"].split())[:90]}')
    report.append('\nInvalid calls and failed answers (first 12):')
    for r in [r for r in rows if r['invalid'] or not r['ok'] or r['bad']][:12]:
        report.append(f'  think={r["think"]!s:<5} {r["id"]}: invalid {r["invalid"] or "-"} | calls {r["calls"]} | '
                      f'{" ".join(r["answer"].split())[:80]}')
    out = raw.with_suffix('.md')
    out.write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report) + f'\n\nSaved to {out} and {raw}')


if __name__ == '__main__':
    main()
