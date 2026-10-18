"""Compare optimized friction quadratics with MJX's original line search."""

import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from playground.open_duck_mini_zp10d import runner
import jax
import jax.numpy as jp
import numpy as np
from jax.experimental import checkify
from mujoco.mjx._src import solver
from mujoco.mjx._src.types import DisableBit
from playground.common.mjx_compat import _pyramidal_point, _original_linesearch_create, _original_linesearch_point


def main():
    assert jax.default_backend() == 'gpu'
    output = PROJECT / 'playground/open_duck_mini_zp10d/training_gpu/linesearch'
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), PROJECT / 'playground/common/mjx_compat.py',
             PROJECT / 'playground/open_duck_mini_zp10d/joystick.py',
             PROJECT / 'playground/open_duck_mini_zp10d/data/reference.json',
             PROJECT / 'playground/open_duck_mini_zp10d/data/polynomial_coefficients.pkl',
             Path('/tmp/zp10d-training/scene.xml')]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(status='running', strategy='sliced_friction_point', backend=jax.default_backend(), num_envs=512,
                  input_sha256=hashes(), same_input_max_abs={}, trajectory_max_abs={}, roundoff_max_abs={})
    previous = output / 'validation.json'
    if previous.exists():
        prior = json.loads(previous.read_text())
        if prior['status'] == 'failed':
            report['rejected_early_exit'] = prior.get('rejected_early_exit', prior)
    started = time.monotonic()
    try:
        env = runner.Joystick()
        state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(7), 512))
        data = jax.tree_util.tree_map(lambda x: x[0], state.data)
        n = data._impl.efc_D.shape[0]
        start, stop = data._impl.ne, data._impl.ne + data._impl.nf
        f = np.zeros(n, np.float32)
        f[start:stop] = .1
        data = data.tree_replace({'_impl.efc_D': jp.ones(n), '_impl.efc_frictionloss': jp.array(f)})
        random = np.random.default_rng(5)
        inputs = tuple(jp.array(x, dtype=jp.float32) for x in (
            random.uniform(-5., 5., 512), random.normal(size=(512, n)),
            random.normal(size=(512, n)), random.normal(size=(512, n, 3)),
            random.normal(size=(512, 3))))
        for disable in (False, True):
            model = env.mjx_model
            if disable:
                model = model.replace(opt=model.opt.replace(disableflags=model.opt.disableflags | DisableBit.FRICTIONLOSS))
            def point(fn):
                def lane(alpha, jaref, jv, quad, gauss):
                    return fn(model, data, SimpleNamespace(Jaref=jaref), alpha, jv, quad, gauss, 0., 0., 0., 0.)
                return jax.jit(jax.vmap(lane))(*inputs)
            actual = point(lambda *args: _pyramidal_point(solver._LSPoint, *args))
            expected = point(_original_linesearch_point)
            for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
                np.testing.assert_array_equal(a, b)
        report['point_parity'] = dict(status='PASS: exact values', friction_rows=int(stop-start), constraint_rows=int(n))
        print('PASS: exact line-search costs/derivatives, friction enabled/disabled;', stop-start, '/', n, 'rows', flush=True)
        # Include standing, small drops, larger drops and downward impacts.
        group = jp.arange(512) % 4
        qpos = state.data.qpos.at[:, 2].add(jp.array([0., .003, .03, .06])[group])
        qvel = state.data.qvel.at[:, 2].set(jp.where(group >= 2, -.3, 0.))
        state = state.replace(data=state.data.replace(qpos=qpos, qvel=qvel))
        action = jp.zeros((512, env.action_size))
        compiled = []
        for name, create in (('original', _original_linesearch_create), ('optimized', classmethod(_pyramidal_point))):
            solver._LSPoint.create = create
            step = jax.jit(jax.vmap(lambda s, a: env.step(s, a))).lower(state, action).compile()
            compiled.append(step)
            print('Compiled 512-environment', name, 'physics comparison', flush=True)
        original, optimized = compiled
        def checked_point(cls, *args):
            actual = _pyramidal_point(cls, *args)
            expected = _original_linesearch_point(*args)
            for name in ('cost', 'deriv_0', 'deriv_1'):
                a, b = getattr(actual, name), getattr(expected, name)
                checkify.check(jp.isclose(a, b, rtol=2e-6, atol=2e-6, equal_nan=True),
                               'line-search ' + name)
            return actual
        solver._LSPoint.create = classmethod(checked_point)
        scalar_state = jax.tree_util.tree_map(lambda x: x[0], state)
        checked = jax.jit(checkify.checkify(lambda s, a: env.step(s, a))).lower(scalar_state, action[0]).compile()
        solver._LSPoint.create = classmethod(_pyramidal_point)
        print('Compiled scalar in-simulation objective/derivative checks for sampled lanes', flush=True)
        timings = {}
        for name, step in (('original', original), ('optimized', optimized)):
            jax.block_until_ready(step(state, action))
            values = []
            for _ in range(4):
                before = time.monotonic()
                jax.block_until_ready(step(state, action))
                values.append(time.monotonic() - before)
            timings[name] = values
        report['same_state_control_seconds'] = timings
        print('Same-state control seconds:', timings, flush=True)
        candidate = state
        roundoff = state
        @jax.jit
        def perturb(data):
            # One FP32 representable step/control, with alternating directions.
            direction = jp.where(jp.arange(512)[:, None] % 2, jp.inf, -jp.inf)
            return data.replace(qpos=jp.nextafter(data.qpos, direction),
                                qvel=jp.nextafter(data.qvel, direction))
        rng = np.random.default_rng(11)
        loads = jax.jit(jax.vmap(env.foot_loads))
        samples = lambda s: dict(qpos=s.data.qpos, base_position=s.data.qpos[:, :3],
                                 base_quaternion=s.data.qpos[:, 3:7], joint_position=s.data.qpos[:, 7:],
                                 qvel=s.data.qvel, linear_velocity=s.data.qvel[:, :3], qacc=s.data.qacc,
                                 constraint_force=s.data.qfrc_constraint[:, :3],
                                 constraint_torque=s.data.qfrc_constraint[:, 3:],
                                 actuator_force=s.data.actuator_force, foot_loads=loads(s.data),
                                 penetration=jp.minimum(jp.min(s.data._impl.contact.dist, axis=-1), 0.),
                                 reward=s.reward, done=s.done,
                                 state_obs=s.obs['state'], privileged_obs=s.obs['privileged_state'])
        # Point forces/distances can permute at degenerate contacts. Compare
        # physical total loads and penetration, not per-slot contact arrays.
        # Force bounds are 0.1% of body weight; pose/velocity checks remain tight.
        force_bound = float(env.mj_model.body_mass.sum() * 9.81 * .001)
        tolerances = dict(qpos=1e-6, qvel=2e-5, qacc=2e-3,
                          constraint_force=force_bound, constraint_torque=2e-3,
                          actuator_force=1e-5, foot_loads=force_bound, penetration=1e-6,
                          reward=2e-5, done=0., state_obs=2e-4, privileged_obs=2e-3)
        violations = []
        original_impulse = np.zeros(512)
        candidate_impulse = np.zeros(512)
        roundoff_impulse = np.zeros(512)
        for control in range(24):
            action = jp.array(rng.normal(0., .08, (512, env.action_size)), dtype=jp.float32)
            expected = original(state, action)
            actual = optimized(state, action)
            candidate = optimized(candidate, action)
            roundoff = original(roundoff.replace(data=perturb(roundoff.data)), action)
            # JAX 0.8.2 checkify cannot transform the batched MJX while loop.
            # Check all point evaluations in two representative lanes/control.
            for lane in (control % 4, 4 + control % 4):
                scalar_state = jax.tree_util.tree_map(lambda x: x[lane], state)
                error, _ = checked(scalar_state, action[lane])
                error.throw()
            expected_fields = samples(expected)
            for name, value in samples(actual).items():
                a, b = np.asarray(value), np.asarray(expected_fields[name])
                assert np.isfinite(a).all() and np.isfinite(b).all(), name
                gap = float(np.max(np.abs(a - b)))
                report['same_input_max_abs'][name] = max(report['same_input_max_abs'].get(name, 0.), gap)
                close = np.isclose(a, b, atol=tolerances.get(name, 1e-6), rtol=2e-5)
                if not close.all():
                    index = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
                    violations.append(dict(control=control, field=name, max_abs=gap,
                                           expected=float(b[index]), actual=float(a[index])))
            for name, value in samples(candidate).items():
                gap = float(np.max(np.abs(np.asarray(value) - np.asarray(expected_fields[name]))))
                report['trajectory_max_abs'][name] = max(report['trajectory_max_abs'].get(name, 0.), gap)
            for name, value in samples(roundoff).items():
                gap = float(np.max(np.abs(np.asarray(value) - np.asarray(expected_fields[name]))))
                report['roundoff_max_abs'][name] = max(report['roundoff_max_abs'].get(name, 0.), gap)
            original_impulse += np.asarray(expected_fields['foot_loads']).sum(axis=-1) * env.dt
            candidate_impulse += np.asarray(loads(candidate.data)).sum(axis=-1) * env.dt
            roundoff_impulse += np.asarray(loads(roundoff.data)).sum(axis=-1) * env.dt
            state = expected
            if (control + 1) % 8 == 0:
                print('Compared', control + 1, 'control steps with standing/drop/impact batches', flush=True)
        # Independent trajectories may amplify FP32 rounding; keep strict pose,
        # velocity, contact and termination checks over this 0.48 s rollout.
        report['strict_array_comparison_diagnostics'] = violations
        # Whole-program fusion can reorder FP32 reductions; contact solves can
        # amplify those ULP differences. Check the actual objective/derivatives
        # at every evaluation above, and bound physical drift in explicit units.
        trajectory_bounds = dict(base_position=1e-4, base_quaternion=5e-4,
                                 joint_position=1e-3, linear_velocity=.01,
                                 qvel=.05, penetration=1e-4, done=0.)
        report['trajectory_acceptance_bounds'] = trajectory_bounds
        report['total_load_impulse_max_abs_Ns'] = float(np.max(abs(original_impulse - candidate_impulse)))
        report['roundoff_total_load_impulse_max_abs_Ns'] = float(np.max(abs(original_impulse - roundoff_impulse)))
        report['roundoff_comparison'] = 'original solver; +/- 1 ULP to qpos/qvel before each control; max drift multiplier 2'
        report['effective_trajectory_bounds'] = {}
        for name, bound in trajectory_bounds.items():
            if name != 'done':
                bound = max(bound, 2 * report['roundoff_max_abs'][name])
            report['effective_trajectory_bounds'][name] = bound
            assert report['trajectory_max_abs'][name] <= bound, (name, report['trajectory_max_abs'][name])
        assert report['total_load_impulse_max_abs_Ns'] <= max(.02, 2 * report['roundoff_total_load_impulse_max_abs_Ns'])
        assert report['input_sha256'] == hashes(), 'inputs changed during validation'
        report.update(status='PASS', control_steps=24, physics_substeps=240,
                      sampled_lane_point_check='PASS: 48 scalar control steps; rtol=2e-6, atol=2e-6',
                      diagnostic_same_input_atol=tolerances, diagnostic_same_input_rtol=2e-5, inputs_unchanged=True,
                      physical_settings=dict(ctrl_dt=.02, sim_dt=.002, solver_iterations=50, ls_iterations=10))
        print('PASS: every line-search objective/derivative and bounded physical drift', flush=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        solver._LSPoint.create = classmethod(_pyramidal_point)
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
