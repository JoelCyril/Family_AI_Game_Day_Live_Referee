"""Example: python verify_report.py downloaded.json
Runs against your CURRENT local referee server (matching signing key)."""
import sys,json
from urllib.request import Request,urlopen
report=json.load(open(sys.argv[1],encoding='utf-8'))
req=Request('http://127.0.0.1:8000/api/verify',data=json.dumps(report).encode(),headers={'Content-Type':'application/json'},method='POST')
with urlopen(req,timeout=20) as response:
    print(json.loads(response.read().decode()))
