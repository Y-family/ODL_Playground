"""Fixed-bound 2 s physics regression: accepted concat versus slice update."""

import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT))
from playground.open_duck_mini_zp10d import runner
import jax
import jax.numpy as jp
import numpy as np
from mujoco.mjx._src import solver
from experiment import candidates


def main():
    root = Path(__file__).parent
    paths = [Path(__file__), root/'experiment.py', root/'accepted_implementation.py',
             PROJECT/'playground/common/mjx_compat.py', PROJECT/'playground/open_duck_mini_zp10d/joystick.py',
             PROJECT/'playground/open_duck_mini_zp10d/data/reference.json',
             PROJECT/'playground/open_duck_mini_zp10d/data/polynomial_coefficients.pkl',
             Path('/tmp/zp10d-training/scene.xml')]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    # Set before measurements. No adjustment based on observed candidate drift.
    same_bounds = dict(base_position=2e-6, base_quaternion=1e-5, joint_position=2e-5,
                       linear_velocity=1e-4, qvel=.003, penetration=2e-6, done=0.)
    trajectory_bounds = dict(base_position=3e-4, base_quaternion=1e-3, joint_position=2e-3,
                             linear_velocity=.02, qvel=.15, penetration=3e-4, done=0.)
    report = dict(status='running', input_sha256=hashes(), num_envs=128, control_steps=100,
                  same_input_bounds=same_bounds, trajectory_bounds=trajectory_bounds,
                  same_input_max_abs={}, trajectory_max_abs={}, timings={})
    before = time.monotonic()
    try:
        env = runner.Joystick()
        state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(31), 128))
        group = jp.arange(128) % 4
        state = state.replace(data=state.data.replace(
            qpos=state.data.qpos.at[:, 2].add(jp.array([0., .003, .03, .06])[group]),
            qvel=state.data.qvel.at[:, 2].set(jp.where(group >= 2, -.3, 0.))))
        action = jp.zeros((128, env.action_size))
        methods = candidates(second=True)
        compiled = {}
        for name in ('accepted', 'update_slice'):
            solver._LSPoint.create = classmethod(methods[name])
            step = jax.jit(jax.vmap(lambda s, a: env.step(s, a))).lower(state, action).compile()
            compiled[name] = step
            jax.block_until_ready(step(state, action))
            times = []
            for _ in range(7):
                start = time.monotonic()
                jax.block_until_ready(step(state, action))
                times.append(time.monotonic() - start)
            report['timings'][name] = dict(seconds=times, median=statistics.median(times))
            print('Compiled', name, '128 environments; timings:', times, flush=True)
        original, optimized = compiled['accepted'], compiled['update_slice']
        candidate = state
        loads = jax.jit(jax.vmap(env.foot_loads))
        impulse = np.zeros(128)
        candidate_impulse = np.zeros(128)
        def fields(s):
            return dict(base_position=s.data.qpos[:, :3], base_quaternion=s.data.qpos[:, 3:7],
                        joint_position=s.data.qpos[:, 7:], linear_velocity=s.data.qvel[:, :3],
                        qvel=s.data.qvel, penetration=jp.minimum(jp.min(s.data._impl.contact.dist, axis=-1), 0.),
                        done=s.done)
        def compare(actual, expected, key):
            for name, a in fields(actual).items():
                b = fields(expected)[name]
                assert np.isfinite(np.asarray(a)).all() and np.isfinite(np.asarray(b)).all(), (key, name)
                gap = float(np.max(abs(np.asarray(a)-np.asarray(b))))
                report[key][name] = max(report[key].get(name, 0.), gap)
            for result in (actual, expected):
                for value in (result.data.qpos, result.data.qvel, result.data.qacc,
                              result.data.qfrc_constraint, result.data.actuator_force,
                              result.obs['state'], result.obs['privileged_state'], result.reward):
                    assert np.isfinite(np.asarray(value)).all(), key
                assert np.all(abs(np.asarray(result.data.actuator_force)) <= np.asarray(env._effort_limits) + 1e-5)
        rng = np.random.default_rng(32)
        for control in range(100):
            action = jp.array(rng.normal(0., .08, (128, env.action_size)), dtype=jp.float32)
            action = jp.where((group == 0)[:, None], 0., action)
            expected = original(state, action)
            same = optimized(state, action)
            candidate = optimized(candidate, action)
            compare(same, expected, 'same_input_max_abs')
            compare(candidate, expected, 'trajectory_max_abs')
            impulse += np.asarray(loads(expected.data)).sum(axis=-1)*env.dt
            candidate_impulse += np.asarray(loads(candidate.data)).sum(axis=-1)*env.dt
            state = expected
            if (control + 1) % 20 == 0:
                print('Compared', control+1, 'controls /', (control+1)*env.dt, 'seconds', flush=True)
        report['total_load_impulse_max_abs_Ns'] = float(np.max(abs(impulse-candidate_impulse)))
        for key, bounds in [('same_input_max_abs', same_bounds), ('trajectory_max_abs', trajectory_bounds)]:
            for name, bound in bounds.items():
                assert report[key][name] <= bound, (key, name, report[key][name], bound)
        assert report['total_load_impulse_max_abs_Ns'] <= .2, report['total_load_impulse_max_abs_Ns']
        assert report['input_sha256'] == hashes(), 'inputs changed'
        report.update(status='PASS: fixed bounds; 2 s standing/drop/impact; finite states and force limits',
                      candidate_source_sha256=hashlib.sha256(methods['update_slice'].__source__.encode()).hexdigest())
        print(report['status'], report['trajectory_max_abs'], flush=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-before,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (root/'physics_validation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
