"""Profile the unchanged 512-environment PPO run after two compilation epochs.

Run through scripts/cad/run_memory_limited.py --gpu --wait. Labels only affect
profiler attribution; model, solver, rewards, and PPO settings stay unchanged.
"""

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'playground/open_duck_mini_zp10d/training_gpu/profile')
    parser.add_argument('--original-linesearch', action='store_true', help='disable the GPU line-search optimization for comparison')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output / 'checkpoints').exists():
        parser.error('output already contains checkpoints; choose a fresh directory')
    group = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
    assert 0 < int((group / 'memory.max').read_text()) <= 7_000_000_000, 'use the GPU memory-limited launcher'
    assert int((group / 'memory.swap.max').read_text()) == 0
    sys.path.insert(0, str(PROJECT))
    from playground.open_duck_mini_zp10d import runner
    if args.original_linesearch:
        from playground.open_duck_mini_zp10d import joystick
        joystick.install_linesearch_optimization = lambda: None
    import jax
    from brax.training import acting
    from brax.training.agents.ppo import losses
    from mujoco.mjx._src import collision_driver, solver
    from mujoco_playground._src import mjx_env

    assert jax.default_backend() == 'gpu'
    output.mkdir(parents=True, exist_ok=True)
    root = PROJECT / 'playground/open_duck_mini_zp10d'
    inputs = [Path(__file__), root / 'runner.py', root / 'joystick.py',
              PROJECT / 'playground/open_duck_mini_v2/joystick.py',
              PROJECT / 'playground/open_duck_mini_v2/base.py',
              PROJECT / 'playground/common/mjx_compat.py',
              PROJECT / 'playground/common/poly_reference_motion.py',
              root / 'data/reference.json', root / 'data/polynomial_coefficients.pkl',
              root / 'open_duck_mini_zp10d.urdf', root / 'xmls/open_duck_mini_zp10d.xml',
              root / 'xmls/scene_flat_terrain.xml', Path('/tmp/zp10d-training/scene.xml')]

    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}

    report = dict(status='running', input_sha256=hashes(), backend=jax.default_backend(),
                  devices=[str(x) for x in jax.devices()], jax_version=jax.__version__,
                  num_envs=512, ctrl_dt=.02, sim_dt=.002, substeps=10,
                  unroll_length=16, batch_size=512, updates_per_batch=2,
                  run_evals=False, gpu_memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
                  linesearch='original' if args.original_linesearch else 'sliced_friction_point',
                  trace_epoch=4, samples=[], callback_seconds={})

    def named(function, name):
        @functools.wraps(function)
        def wrapped(*values, **kwargs):
            with jax.named_scope(name):
                return function(*values, **kwargs)
        return wrapped

    mjx_env.step = named(mjx_env.step, 'profile_physics')
    collision_driver.collision = named(collision_driver.collision, 'profile_collision')
    for name in ('solve', '_linesearch', '_update_gradient', '_update_constraint'):
        setattr(solver, name, named(getattr(solver, name), 'profile_solver' + name))
    for name, label in (('_get_reward', 'profile_reward'), ('_get_obs', 'profile_observation'),
                        ('reset', 'profile_reset')):
        setattr(runner.Joystick, name, named(getattr(runner.Joystick, name), label))
    acting.generate_unroll = named(acting.generate_unroll, 'profile_sampling')
    losses.compute_ppo_loss = named(losses.compute_ppo_loss, 'profile_ppo_loss')
    original_train = runner.ppo.train
    original_save = runner.ppo.checkpoint.save
    trace_active = False
    started = time.monotonic()

    def timed(function, name):
        @functools.wraps(function)
        def wrapped(*values, **kwargs):
            before = time.monotonic()
            with jax.profiler.TraceAnnotation(name):
                result = function(*values, **kwargs)
            report['callback_seconds'].setdefault(name, []).append(time.monotonic() - before)
            return result
        return wrapped

    runner.ppo.checkpoint.save = timed(original_save, 'profile_checkpoint_save')

    def train(**kwargs):
        nonlocal trace_active
        progress = kwargs['progress_fn']
        kwargs['policy_params_fn'] = timed(kwargs['policy_params_fn'], 'profile_parameter_check')

        def record(steps, metrics):
            nonlocal trace_active
            progress(steps, metrics)
            epoch = int(steps) // 8192
            report['samples'].append(dict(epoch=epoch, steps=int(steps),
                                          elapsed_seconds=time.monotonic() - started,
                                          training_sps=float(metrics['training/sps']),
                                          training_walltime=float(metrics['training/walltime']),
                                          trace_active=trace_active))
            if epoch == 3:
                options = jax.profiler.ProfileOptions()
                options.python_tracer_level = 0
                options.host_tracer_level = 2
                options.device_tracer_level = 1
                jax.profiler.start_trace(str(output / 'trace'), profiler_options=options)
                trace_active = True
                print('PROFILE: warmed GPU trace started for epoch 4', flush=True)
            elif epoch == 4:
                jax.profiler.stop_trace()
                trace_active = False
                print('PROFILE: warmed GPU trace saved; remaining epochs are untraced', flush=True)
            (output / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')

        kwargs['progress_fn'] = record
        return original_train(**kwargs)

    runner.ppo.train = train
    sys.argv = ['runner.py', '--envs', '512', '--steps', '49152', '--checkpoint-every', '8192',
                '--no-eval', '--reward-mode', 'optimized', '--output', str(output)]
    try:
        runner.main()
        rates = [x['training_sps'] for x in report['samples'] if x['epoch'] > 2 and not x['trace_active']]
        assert len(report['samples']) == 6 and len(rates) == 3
        assert report['input_sha256'] == hashes(), 'profiling inputs changed during the run'
        report.update(status='PASS: six actual PPO epochs; warm trace captured',
                      median_untraced_warm_sps=statistics.median(rates), inputs_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if trace_active:
            jax.profiler.stop_trace()
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      cgroup_memory_events=(group / 'memory.events').read_text())
        (output / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
