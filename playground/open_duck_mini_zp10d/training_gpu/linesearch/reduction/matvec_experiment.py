"""Measure MJX-sized matvec layouts with the same FP32 arithmetic precision."""

import json
from pathlib import Path
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
from playground.open_duck_mini_zp10d import runner
import jax
import jax.numpy as jp
import numpy as np


def main():
    rng = np.random.default_rng(17)
    a = rng.normal(size=(512, 304, 22)).astype(np.float32)
    b = rng.normal(size=(512, 22)).astype(np.float32)
    reference = np.einsum('bij,bj->bi', a.astype(np.float64), b.astype(np.float64))
    a, b = jp.array(a), jp.array(b)
    functions = {'vector': lambda a, b: a @ b}
    for n in (1, 2, 4, 8):
        functions[f'columns_{n}'] = lambda a, b, n=n: (a @ jp.broadcast_to(b[:, None], (b.shape[0], n)))[:, 0]
    report = {}
    for name, fn in functions.items():
        compiled = jax.jit(jax.vmap(fn)).lower(a, b).compile()
        actual = np.asarray(compiled(a, b))
        np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)
        times = []
        for _ in range(25):
            start = time.monotonic()
            jax.block_until_ready(compiled(a, b))
            times.append(time.monotonic() - start)
        text = compiled.as_text()
        report[name] = dict(median_seconds=statistics.median(times), seconds=times,
                            max_abs_error=float(np.max(abs(actual-reference))),
                            custom_calls=[x.strip()[:400] for x in text.splitlines() if 'custom_call_target' in x])
        print(name, report[name]['median_seconds'], report[name]['max_abs_error'], report[name]['custom_calls'], flush=True)
    Path(__file__).with_name('matvec_exploration.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
