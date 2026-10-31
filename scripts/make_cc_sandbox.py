"""Lab 06 Step 5. Creates the throwaway repository that Claude Code works in: C:\\tmp\\cc-sandbox by default.

    .\\.venv\\Scripts\\python scripts\\make_cc_sandbox.py [path]      # refuses to overwrite an existing folder

A tiny payments service (with a .gitignore for __pycache__) and two planted problems whose answers are known:
  * app/pricing.py divides a percentage by 10 instead of 100, so one unit test fails (c3, c4);
  * k8s/deployment.yaml takes its environment from Secret "payments-db", but the Secret is named
    "payments-database". `optional: true` hides the mismatch: the pod starts without DATABASE_URL and crash-loops
    instead of failing with CreateContainerConfigError (c2).
No secrets: the Secret holds a placeholder. The tests need only the standard library (python -m unittest).
"""
import subprocess
import sys
from pathlib import Path

FILES = {
    '.gitignore': '__pycache__/\n',
    'README.md': """# payments-api

A small pricing service. `app/` holds the code, `tests/` the unit tests, `k8s/` the manifests.

Run the tests with `python -m unittest`.
""",
    'app/__init__.py': '',
    'app/config.py': """import os


def load():
    \"\"\"Settings from the environment. DATABASE_URL is required.\"\"\"
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise SystemExit('required environment variable DATABASE_URL is not set')
    return {'database_url': url, 'currency': os.environ.get('CURRENCY', 'EUR')}
""",
    'app/pricing.py': """def total(items, discount_percent=0):
    \"\"\"Sum of price * qty over the items, minus a percentage discount, rounded to cents.\"\"\"
    subtotal = sum(item['price'] * item['qty'] for item in items)
    return round(subtotal - subtotal * discount_percent / 10, 2)
""",
    'tests/__init__.py': '',
    'tests/test_pricing.py': """import unittest

from app.pricing import total

ITEMS = [{'price': 40.0, 'qty': 2}, {'price': 20.0, 'qty': 1}]


class TotalTest(unittest.TestCase):
    def test_no_discount(self):
        self.assertEqual(total(ITEMS), 100.0)

    def test_ten_percent_discount(self):
        self.assertEqual(total(ITEMS, discount_percent=10), 90.0)

    def test_empty(self):
        self.assertEqual(total([]), 0)


if __name__ == '__main__':
    unittest.main()
""",
    'k8s/deployment.yaml': """apiVersion: apps/v1
kind: Deployment
metadata:
  name: payments-api
  namespace: payments
spec:
  replicas: 2
  selector:
    matchLabels:
      app: payments-api
  template:
    metadata:
      labels:
        app: payments-api
    spec:
      containers:
        - name: payments-api
          image: registry.example.com/payments-api:2.4.1
          ports:
            - containerPort: 8080
          envFrom:
            - secretRef:
                name: payments-db
                optional: true
          env:
            - name: CURRENCY
              value: EUR
""",
    'k8s/secret.example.yaml': """# Example only: the real value is created by the platform team, never committed.
apiVersion: v1
kind: Secret
metadata:
  name: payments-database
  namespace: payments
type: Opaque
stringData:
  DATABASE_URL: postgres://payments:CHANGE-ME@db.payments.svc:5432/payments
""",
}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else r'C:\tmp\cc-sandbox')
    if root.exists():
        sys.exit(f'{root} already exists: remove it first (it is a throwaway) or pass another path')
    for name, text in FILES.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode('utf-8'))
    git = ['git', '-C', str(root)]
    subprocess.run(git + ['init', '-q'], check=True)
    subprocess.run(git + ['add', '.'], check=True)
    subprocess.run(git + ['-c', 'user.name=lab', '-c', 'user.email=lab@example.invalid', 'commit', '-q', '-m',
                          'sandbox for Lab 06 Step 5'], check=True)
    print(f'created {root} ({len(FILES)} files, one commit). Reset after a run: git -C {root} reset --hard')


if __name__ == '__main__':
    main()
