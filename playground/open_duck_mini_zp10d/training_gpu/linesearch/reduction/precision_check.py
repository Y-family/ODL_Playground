"""Independent FP64 oracle and autodiff checks for FP32 line-search points."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
from playground.open_duck_mini_zp10d import runner
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco.mjx._src import solver
from mujoco.mjx._src.types import DisableBit
from playground.common.mjx_compat import _original_linesearch_point, _pyramidal_point
from experiment import candidates


def main():
    assert jax.default_backend() == 'gpu'
    paths = [Path(__file__), Path(__file__).with_name('experiment.py'),
             Path(__file__).with_name('accepted_implementation.py'), PROJECT/'playground/common/mjx_compat.py']
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original_hashes = hashes()
    env = runner.Joystick()
    data = env.reset(jax.random.PRNGKey(0)).data
    n, ne, nf = len(data._impl.efc_D), data._impl.ne, data._impl.nf
    stop = ne + nf
    rng = np.random.default_rng(29)
    count = 4096
    diagonal = np.exp2(np.arange(n) % 7 - 3).astype(np.float32)
    diagonal[stop::7] = 0.
    friction = np.zeros(n, np.float32)
    friction[ne:stop] = np.resize(np.array([0., .0625, .125, .25], np.float32), nf)
    rf = friction / (diagonal + (diagonal == 0.) * mujoco.mjMINVAL)
    data = data.tree_replace({'_impl.efc_D': jp.array(diagonal), '_impl.efc_frictionloss': jp.array(friction)})
    alpha = rng.uniform(-2., 2., count).astype(np.float32)
    jaref, jv = (rng.normal(size=(count, n)).astype(np.float32) for _ in range(2))
    # Exact powers of two make the kink locations representable. Cover both
    # boundaries, their adjacent FP32 values, and zero-contact transitions.
    alpha[:8] = 0.
    for lane, x in enumerate([-rf, np.nextafter(-rf, -np.inf), np.nextafter(-rf, np.inf),
                              rf, np.nextafter(rf, -np.inf), np.nextafter(rf, np.inf),
                              np.zeros(n, np.float32), np.ones(n, np.float32)]):
        jaref[lane] = x
    quad = np.stack((.5*jaref*jaref, jv*jaref, .5*jv*jv), axis=-1) * diagonal[None, :, None]
    gauss = np.broadcast_to(np.array([.25, .125, .5], np.float32), (count, 3)).copy()
    inputs = tuple(jp.array(x) for x in (alpha, jaref, jv, quad, gauss))
    report = dict(status='running', samples=count, boundaries=8, comparisons={},
                  input_sha256=original_hashes,
                  adapter_sha256=hashlib.sha256((PROJECT/'playground/common/mjx_compat.py').read_bytes()).hexdigest(),
                  bound='32 * FP32 epsilon * max(1, sum(abs(contributions)))')
    for disable in (False, True):
        model = env.mjx_model
        if disable:
            model = model.replace(opt=model.opt.replace(disableflags=model.opt.disableflags | DisableBit.FRICTIONLOSS))
        # Independent equations evaluated in NumPy FP64, from FP32 inputs.
        a, j, v, q, g = (np.asarray(x, dtype=np.float64) for x in (alpha, jaref, jv, quad, gauss))
        x = j + a[:, None]*v
        active = x < 0
        active[:, :stop] = True
        if not disable:
            f = friction.astype(np.float64)
            r = 1. / (diagonal.astype(np.float64) + (diagonal == 0.) * mujoco.mjMINVAL)
            neg, pos = x <= -r*f, x >= r*f
            qf = np.stack((f*(-.5*r*f-j), -f*v, np.zeros_like(v)), axis=-1)*neg[:, :, None]
            qf += np.stack((f*(-.5*r*f+j), f*v, np.zeros_like(v)), axis=-1)*pos[:, :, None]
            q = np.where(((neg | pos) & (f > 0.))[:, :, None], qf, q)
        contributions = q*active[:, :, None]
        total = g + contributions.sum(axis=1)
        magnitude = abs(g) + abs(contributions).sum(axis=1)
        expected = np.stack((a*a*total[:, 2]+a*total[:, 1]+total[:, 0],
                             2*a*total[:, 2]+total[:, 1],
                             2*total[:, 2]+(total[:, 2] == 0)*mujoco.mjMINVAL), axis=-1)
        scales = np.stack((a*a*magnitude[:, 2]+abs(a)*magnitude[:, 1]+magnitude[:, 0],
                           2*abs(a)*magnitude[:, 2]+magnitude[:, 1],
                           2*magnitude[:, 2]), axis=-1)
        allowance = 32*np.finfo(np.float32).eps*np.maximum(1., scales)
        points = candidates(second=True)
        functions = [('upstream', _original_linesearch_point)]
        for name in ('accepted', 'update_slice'):
            functions.append((name, lambda *args, point=points[name]: point(solver._LSPoint, *args)))
        reference = None
        for name, fn in functions:
            def lane(a, j, v, q, g):
                point = lambda aa: fn(model, data, SimpleNamespace(Jaref=j), aa, v, q, g, 0., 0., 0., 0.)
                result = point(a)
                return jp.stack((result.cost, result.deriv_0, result.deriv_1)), jax.grad(lambda aa: point(aa).cost)(a), jax.grad(jax.grad(lambda aa: point(aa).cost))(a)
            actual, first, second = (np.asarray(x) for x in jax.jit(jax.vmap(lane))(*inputs))
            assert np.isfinite(actual).all() and np.isfinite(first).all() and np.isfinite(second).all()
            oracle_ratio = float(np.max(abs(actual-expected)/allowance))
            autodiff_ratio = float(np.max(abs(np.stack((first, second), axis=-1)-actual[:, 1:])/allowance[:, 1:]))
            assert oracle_ratio <= 1., (disable, name, oracle_ratio)
            assert autodiff_ratio <= 1., (disable, name, autodiff_ratio)
            if reference is None:
                reference = actual
            np.testing.assert_array_equal(actual, reference)
            report['comparisons'][f'{name}_friction_disabled_{disable}'] = dict(max_oracle_bound_ratio=oracle_ratio, max_autodiff_bound_ratio=autodiff_ratio)
            print(name, 'disable friction:', disable, 'oracle ratio:', oracle_ratio, 'autodiff ratio:', autodiff_ratio, flush=True)
    assert original_hashes == hashes(), 'point-check inputs changed'
    report.update(status='PASS: exact FP32 point parity; independent FP64 equations, boundary cases and autodiff',
                  candidate_source_sha256=hashlib.sha256(points['update_slice'].__source__.encode()).hexdigest())
    Path(__file__).with_name('precision_validation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
