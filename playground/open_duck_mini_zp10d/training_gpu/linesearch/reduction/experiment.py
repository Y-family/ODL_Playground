"""Isolated reduction candidates; does not install them in normal training."""

import inspect
import argparse
import ast
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
from mujoco import mjx
from mujoco.mjx._src import solver
from playground.common import mjx_compat as compat


def candidates(second=False):
    archived = Path(__file__).with_name('accepted_implementation.py').read_text()
    source = ast.get_source_segment(archived, next(n for n in ast.parse(archived).body
                                                  if isinstance(n, ast.FunctionDef) and n.name == '_pyramidal_point'))
    reduction = 'quad_total = quad_gauss + jp.sum(jax.vmap(jp.multiply)(quad, active), axis=0)'
    replacements = {
        'accepted': source,
        'scalar_sums': source.replace(reduction, 'quad_total = jp.stack([quad_gauss[i] + jp.sum(quad[:, i] * active) for i in range(3)])'),
        'where_mask': source.replace(reduction, 'quad_total = quad_gauss + jp.sum(jp.where(active[:, None], quad, 0.), axis=0)'),
        'dot': source.replace(reduction, 'quad_total = quad_gauss + active.astype(quad.dtype) @ quad'),
        'split_friction': source.replace('x = ctx.Jaref', 'friction = quad[start:stop]\n    x = ctx.Jaref')
            .replace('quad = jp.concatenate((quad[:start], friction, quad[stop:]), axis=0)', 'pass')
            .replace(reduction, 'quad_total = quad_gauss + jp.sum(quad[:start], axis=0) + jp.sum(friction, axis=0) + jp.sum(quad[stop:] * active[stop:, None], axis=0)'),
    }
    if second:
        replacements = {
            'accepted': source,
            'update_slice': source.replace('jp.concatenate((quad[:start], friction, quad[stop:]), axis=0)', 'jax.lax.dynamic_update_slice(quad, friction, (start, 0))'),
            'active_concat': source.replace('(x < 0).at[:stop].set(True)', 'jp.concatenate((jp.ones(stop, dtype=bool), x[stop:] < 0))'),
            'friction_barrier': source.replace('quad = jp.concatenate', 'friction = jax.lax.optimization_barrier(friction)\n        quad = jp.concatenate'),
            'delta_friction': source.replace('x = ctx.Jaref', 'friction = quad[start:stop]\n    x = ctx.Jaref')
                .replace('quad = jp.concatenate((quad[:start], friction, quad[stop:]), axis=0)', 'pass')
                .replace(reduction, reduction + ' + jp.sum(friction - quad[start:stop], axis=0)'),
        }
    result = {}
    for name, code in replacements.items():
        namespace = dict(vars(compat))
        exec(compile(code, f'<{name}>', 'exec'), namespace)
        result[name] = namespace['_pyramidal_point']
        result[name].__source__ = code
    return result


def main():
    assert jax.default_backend() == 'gpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--second', action='store_true')
    args = parser.parse_args()
    output = Path(__file__).with_name('batch_exploration.json' if args.batch else 'second_exploration.json' if args.second else 'exploration.json')
    report = dict(status='running', candidates={})
    env = runner.Joystick()
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(23), 512))
    data = jax.jit(jax.vmap(lambda d: mjx.forward(env.mjx_model, d)))(state.data)
    data = data.replace(qacc=data.qacc + jax.random.normal(jax.random.PRNGKey(24), data.qacc.shape))
    context = jax.jit(jax.vmap(lambda d: solver.Context.create(env.mjx_model, d)))(data)
    jax.block_until_ready(context)
    original = solver._LSPoint.__dict__['create']
    reference = None
    methods = candidates(args.second) if not args.batch else {'accepted': compat._pyramidal_point, 'batched_points': compat._pyramidal_point}
    linesearch = solver._linesearch
    if args.batch:
        source = inspect.getsource(linesearch)
        before = '''    lo_next = point_fn(lo.alpha - lo.deriv_0 / lo.deriv_1)
    hi_next = point_fn(hi.alpha - hi.deriv_0 / hi.deriv_1)
    mid = point_fn(0.5 * (lo.alpha + hi.alpha))'''
        after = '''    points = jax.vmap(point_fn)(jp.stack((lo.alpha - lo.deriv_0 / lo.deriv_1,
                                              hi.alpha - hi.deriv_0 / hi.deriv_1,
                                              0.5 * (lo.alpha + hi.alpha))))
    lo_next, hi_next, mid = (jax.tree_util.tree_map(lambda x: x[i], points) for i in range(3))'''
        assert source.count(before) == 1
        namespace = dict(vars(solver))
        exec(compile(source.replace(before, after), '<batched_points>', 'exec'), namespace)
        batched_linesearch = namespace['_linesearch']
    for name, function in methods.items():
        solver._LSPoint.create = classmethod(function)
        solver._linesearch = batched_linesearch if name == 'batched_points' else linesearch
        # Each new closure gets a fresh trace despite the shared solver function.
        def run(d, ctx, point_function=function):
            return solver._linesearch(env.mjx_model, d, ctx)
        step = jax.jit(jax.vmap(run)).lower(data, context).compile()
        value = jax.block_until_ready(step(data, context))
        assert all(np.isfinite(np.asarray(getattr(value, key))).all() for key in ('qacc', 'Ma', 'Jaref')), name
        if reference is None:
            reference = value
        gaps = {key: float(np.max(np.abs(np.asarray(getattr(value, key)) - np.asarray(getattr(reference, key)))))
                for key in ('qacc', 'Ma', 'Jaref')}
        times = []
        for _ in range(9):
            started = time.monotonic()
            jax.block_until_ready(step(data, context))
            times.append(time.monotonic() - started)
        report['candidates'][name] = dict(median_seconds=statistics.median(times), seconds=times, max_abs=gaps)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(name, report['candidates'][name], flush=True)
    solver._LSPoint.create = original
    solver._linesearch = linesearch
    report['status'] = 'measured; candidate correctness and full physics not yet accepted'
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
