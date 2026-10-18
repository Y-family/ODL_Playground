"""Run original/optimized PPO sequentially under one GPU launcher lock/cgroup."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]


def main():
    assert json.loads((ROOT / 'validation.json').read_text())['status'] == 'PASS'
    result = dict(status='running', runs={})
    try:
        for name in ('baseline', 'optimized'):
            output = ROOT / name
            output.mkdir(exist_ok=True)
            command = [sys.executable, str(PROJECT / 'scripts/profile_zp10d.py'), '--output', str(output)]
            if name == 'baseline':
                command.append('--original-linesearch')
            print('Starting', name, 'PPO profile', flush=True)
            before = time.monotonic()
            with (output / 'train.log').open('w') as log:
                subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result['runs'][name] = json.loads((output / 'profile.json').read_text())
            result['runs'][name]['subprocess_seconds'] = time.monotonic() - before
            subprocess.run([sys.executable, str(PROJECT / 'scripts/analyze_zp10d_profile.py'), str(output)],
                           cwd=PROJECT, stdout=subprocess.DEVNULL, check=True)
            (ROOT / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
        baseline, optimized = (result['runs'][name] for name in ('baseline', 'optimized'))
        assert baseline['input_sha256'] == optimized['input_sha256'], 'A/B inputs differ'
        result.update(status='PASS: matching inputs, actual PPO updates and GPU traces',
                      speedup=optimized['median_untraced_warm_sps'] / baseline['median_untraced_warm_sps'])
        print('PPO steady speedup:', result['speedup'], flush=True)
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (ROOT / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
