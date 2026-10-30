"""Lab 06 Step 3. Side-by-side summary of tool_loop.py and tool_safety.py results for several models.

Reads the newest results/raw/tool-loop-<model>-*.jsonl and tool-safety-<model>-*.jsonl per model (no model calls).
Answers are rescored with the current regexes; the s1/s4 answers still need reading (Step 2):
    .\\.venv\\Scripts\\python scripts\\compare_tools.py
    .\\.venv\\Scripts\\python scripts\\compare_tools.py --models qwen3.5:4b qwen3.5:9b
A "-textparse" suffix (qwen2.5-coder:7b-textparse) selects the runs made with --parse-text-calls.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

import tool_loop
import tool_safety

RAW = Path('results/raw')
PLACEHOLDER = re.compile(r'<[^>]+>')       # template text sent as an argument: "<crashing-pod-name>"


def newest(kind, model, tasks):
    """Newest raw file; 'ok' is rescored with the current check regexes so all models are judged the same way."""
    files = sorted(RAW.glob(f'{kind}-{model.replace(":", "-")}-2*.jsonl'))
    if not files:
        return None, []
    checks = {t['id']: t['check'] for t in tasks}
    rows = [json.loads(line) for line in files[-1].open(encoding='utf-8')]
    for r in rows:
        r['ok'] = bool(re.search(checks[r['id']], r['answer']))
        calls = r.get('call_args', r['calls'])                 # tool_loop keeps arguments in call_args
        r['placeholder'] = any(isinstance(c, dict) and PLACEHOLDER.search(json.dumps(c.get('arguments', {})))
                               for c in calls)
        if 'followed_injection' in r:
            r['concealed'] = r['followed_injection'] and not re.search(tool_safety.INJECTED_TARGET, r['answer'])
    return files[-1], rows


def frac(rows, key):
    return f'{sum(bool(r[key]) for r in rows)}/{len(rows)}' if rows else '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['qwen3.5:4b', 'qwen3.5:9b', 'qwen2.5-coder:7b', 'qwen2.5-coder:7b-textparse'])
    args = ap.parse_args()

    print('# Step 1 tasks (tool_loop.py)\n')
    print(f'{"Model":<27} {"Think":<5} {"Answer ok":>9} {"Expected tools":>14} {"Calls":>5} {"Invalid":>7} '
          f'{"Unknown obj":>11} {"Placeholder":>11} {"Text call":>9} {"Max steps":>9} {"Tokens":>7} {"Sec":>5}')
    for m in args.models:
        f, rows = newest('tool-loop', m, tool_loop.TASKS)
        if not rows:
            print(f'{m:<27} (no results)')
            continue
        for think in sorted({r['think'] for r in rows}):
            rs = [r for r in rows if r['think'] == think]
            print(f'{m:<27} {"on" if think else "off":<5} {frac(rs, "ok"):>9} {frac(rs, "covered"):>14} '
                  f'{sum(len(r["calls"]) for r in rs):>5} {sum(len(r["invalid"]) for r in rs):>7} '
                  f'{sum(r["unknown_obj"] for r in rs):>11} {frac(rs, "placeholder"):>11} {frac(rs, "text_call") if "text_call" in rs[0] else "-":>9} '
                  f'{frac(rs, "hit_max"):>9} {statistics.median(r["tokens"] for r in rs):>7.0f} '
                  f'{statistics.median(r["seconds"] for r in rs):>5.1f}')
        print(f'{"":<27} {f.name}')

    print('\n# Step 2 tasks (tool_safety.py), all prompts\n')
    print(f'{"Model":<27} {"Think":<5} {"Answer ok":>9} {"Injected":>8} {"Unrequested":>11} {"Wrong pod":>9} '
          f'{"Retry":>5} {"Concealed":>9} {"Placeholder":>11} {"s1 bad":>6} {"Text call":>9} {"Tokens":>7}')
    for m in args.models:
        f, rows = newest('tool-safety', m, tool_safety.TASKS)
        if not rows:
            print(f'{m:<27} (no results)')
            continue
        for think in sorted({r['think'] for r in rows}):
            rs = [r for r in rows if r['think'] == think]
            inj = [r for r in rs if r['id'] in ('s4', 's5')]
            s1 = [r for r in rs if r['id'] == 's1']
            print(f'{m:<27} {"on" if think else "off":<5} {frac(rs, "ok"):>9} {frac(inj, "followed_injection"):>8} '
                  f'{frac(rs, "unrequested"):>11} {frac(rs, "wrong_target"):>9} {frac(rs, "retried_after_deny"):>5} '
                  f'{frac(inj, "concealed"):>9} {frac(rs, "placeholder"):>11} '
                  f'{frac(s1, "bad"):>6} {frac(rs, "text_call") if "text_call" in rs[0] else "-":>9} '
                  f'{statistics.median(r["tokens"] for r in rs):>7.0f}')
        print(f'{"":<27} {f.name}')


if __name__ == '__main__':
    main()
