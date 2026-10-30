"""Inspect official runs/result.json. Never changes the competition harness.

Usage from team repo root:
  python analyze_results.py runs/result.json
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/result.json')
result = json.loads(path.read_text(encoding='utf-8'))
summary = result.get('summary', {})
for side in ('base', 'solution'):
    d = summary.get(side, {})
    print(f"{side:8s}  ok={d.get('ok')}  wall={d.get('wall_s')}s  "
          f"peak={d.get('gpu_peak_mib')} MiB  eval_loss={d.get('eval_loss')}")
print('ratios:', summary.get('ratios', {}))
print('\nOwn RMSNorm GPU-time fractions:')
for run in result.get('runs', {}).get('solution', []):
    metrics = run.get('metrics') or {}
    own = metrics.get('own_kernel_ms') or {}
    own_ms = sum(float(ms) for name, ms in own.items() if name in (
        'soup_rms_rows_forward', 'soup_rms_rows_backward'))
    timed = float(metrics.get('timed_loop_s') or 0)
    pct = 100 * own_ms / (timed * 1000) if timed else 0
    print(f"  {run.get('run')}: own_rms_ms={own_ms:.2f}  timed_loop_s={timed:.2f} "
          f"share={pct:.2f}%   {'PASS 5% share' if pct >= 5 else 'FAIL 5% share'}")
    for name in ('soup_rms_rows_forward', 'soup_rms_rows_backward'):
        print(f"    {name}: launches={(metrics.get('triton_launches') or {}).get(name, 0)} "
              f"time_ms={own.get(name, 0)}")
checks = path.parent / 'checks.json'
if checks.exists():
    print('\nOfficial checks file:', checks)
    print(checks.read_text(encoding='utf-8')[:8000])
