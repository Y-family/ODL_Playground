"""Sequential PPO A/B with fixed physics validation and one GPU lock/cgroup."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[4]


def main():
    physics = json.loads((ROOT/'physics_validation.json').read_text())
    assert physics['status'].startswith('PASS'), physics['status']
    for path, digest in physics['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    with (ROOT/'precision.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT/'precision_check.py')], cwd=PROJECT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    assert json.loads((ROOT/'precision_validation.json').read_text())['status'].startswith('PASS')
    paths = [ROOT/name for name in ('experiment.py', 'accepted_implementation.py', 'profile_variant.py', 'benchmark.py', 'precision_check.py')]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(status='running', implementation_inputs=hashes(), runs={})
    try:
        for name in ('concat', 'slice'):
            output = ROOT/name
            output.mkdir(exist_ok=True)
            print('Starting', name, '512-environment PPO', flush=True)
            with (output/'train.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'profile_variant.py'), name, '--output', str(output)],
                               cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            report['runs'][name] = json.loads((output/'profile.json').read_text())
            subprocess.run([sys.executable, str(PROJECT/'scripts/analyze_zp10d_profile.py'), str(output)],
                           cwd=PROJECT, stdout=subprocess.DEVNULL, check=True)
            assert report['implementation_inputs'] == hashes(), 'implementation changed'
            (ROOT/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
            print(name, report['runs'][name]['median_untraced_warm_sps'], 'environment transitions/s', flush=True)
        baseline, optimized = report['runs']['concat'], report['runs']['slice']
        assert baseline['input_sha256'] == optimized['input_sha256'], 'A/B inputs changed'
        assert optimized['point_source_sha256'] == json.loads((ROOT/'physics_validation.json').read_text())['candidate_source_sha256']
        report.update(status='PASS: same inputs; fixed physics bounds; six real PPO epochs per variant',
                      speedup=optimized['median_untraced_warm_sps']/baseline['median_untraced_warm_sps'])
        print(report['status'], 'speedup:', report['speedup'], flush=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (ROOT/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
