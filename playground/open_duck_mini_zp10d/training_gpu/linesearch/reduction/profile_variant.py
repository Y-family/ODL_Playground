"""Select an isolated point implementation, then run the shared PPO profiler."""

import hashlib
import json
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
variant = sys.argv.pop(1)
assert variant in ('concat', 'slice')
from playground.open_duck_mini_zp10d import runner, joystick
from mujoco.mjx._src import solver
from experiment import candidates

point = candidates(second=True)['accepted' if variant == 'concat' else 'update_slice']
point.__name__ = '_concat_point' if variant == 'concat' else '_slice_point'
joystick.install_linesearch_optimization = lambda: setattr(solver._LSPoint, 'create', classmethod(point))
output = Path(sys.argv[sys.argv.index('--output')+1])
runpy.run_path(str(PROJECT/'scripts/profile_zp10d.py'), run_name='__main__')
for name in ('profile.json', 'metadata.json'):
    path = output/name
    report = json.loads(path.read_text())
    report.update(reduction_variant=variant, point_source_sha256=hashlib.sha256(point.__source__.encode()).hexdigest())
    path.write_text(json.dumps(report, indent=2)+'\n')
