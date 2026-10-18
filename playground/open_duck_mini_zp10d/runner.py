"""Small-batch entrypoint for the original Brax PPO / MJX training workflow."""

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '1.0')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_llvm_module_compilation_parallelism=false')

import jax
import jax.numpy as jp
from mujoco.mjx._src import solver
from brax.training.agents.ppo import networks, train as ppo
from mujoco_playground import wrapper

from playground.open_duck_mini_zp10d.joystick import Joystick
from playground.open_duck_mini_zp10d.reference import ROOT, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=49152)
    parser.add_argument('--envs', type=int, default=512)
    parser.add_argument('--checkpoint-every', type=int, default=8192)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-eval', action='store_true', help='update/checkpoint smoke test only')
    parser.add_argument('--output', type=Path, default=ROOT / 'training')
    parser.add_argument('--restore', type=str)
    parser.add_argument('--reward-mode', choices=('original', 'optimized'), default='optimized')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.restore:
        args.restore = str(Path(args.restore).resolve())
        restored = json.loads((Path(args.restore).parent.parent / 'metadata.json').read_text())
        if restored['source_hashes'] != source_hashes():
            raise ValueError('Warm-start checkpoint belongs to a different source model')
    if min(args.envs, args.steps, args.checkpoint_every) < 1:
        parser.error('steps/envs/checkpoint-every must be positive')
    env = Joystick(reward_mode=args.reward_mode)
    print('devices', jax.devices(), 'action size', env.action_size, 'observations', env.observation_size, flush=True)
    if args.smoke:
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        t = time.monotonic()
        state = reset(jax.random.PRNGKey(0))
        state = step(state, jp.zeros(env.action_size))
        jax.block_until_ready(state.data.qpos)
        print('reset + first control step seconds', time.monotonic() - t, flush=True)
        t = time.monotonic()
        for _ in range(100):
            state = step(state, jp.zeros(env.action_size))
        jax.block_until_ready(state.data.qpos)
        assert bool(jp.isfinite(state.data.qpos).all())
        print('control steps/s', 100 / (time.monotonic() - t), 'base', state.data.qpos[:7],
              'done', state.done, flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT / 'data/reference.json').read_text())
    metadata = dict(source_hashes=source_hashes(), commands=reference['commands'],
                    action_scale=reference['action_scale'], actuator_names=env.actuator_names,
                    backend=jax.default_backend(), restore_checkpoint=args.restore,
                    restore_semantics='policy/value/normalizer warm start; optimizer and run step counter reset',
                    reference_sha256=hashlib.sha256((ROOT / 'data/polynomial_coefficients.pkl').read_bytes()).hexdigest(),
                    training_contact_model='18 box geoms, floor-only; no self-contact',
                    status='starting; not accepted walking', requested_steps=args.steps,
                    num_envs=args.envs, num_eval_envs=min(args.envs, 4),
                    gpu_memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
                    run_evals=not args.no_eval)
    metadata['linesearch_implementation'] = solver._LSPoint.create.__name__
    metadata['mjx_adapter_sha256'] = hashlib.sha256((ROOT.parent / 'common/mjx_compat.py').read_bytes()).hexdigest()
    metadata['reward_specification'] = env.reward_specification()
    metadata['reward_code_sha256'] = hashlib.sha256((ROOT / 'joystick.py').read_bytes()).hexdigest()
    metadata['reward_reference_sha256'] = hashlib.sha256(json.dumps(
        reference.get('reward_reference', {}), sort_keys=True).encode()).hexdigest()
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.monotonic()
    initial_policy = None

    def policy_update(steps, make_policy, params):
        del make_policy
        nonlocal initial_policy
        policy = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), params[1])
        leaves = jax.tree_util.tree_leaves(policy)
        assert all(np.isfinite(x).all() for x in leaves)
        if initial_policy is None:
            initial_policy = policy
        elif int(steps) > 0:
            delta = max(float(np.max(abs(a - b))) for a, b in zip(
                leaves, jax.tree_util.tree_leaves(initial_policy)))
            assert delta > 1e-8, 'PPO did not change policy parameters'
            metadata.update(completed_steps=int(steps), policy_parameter_max_delta=delta,
                            status='training; walking acceptance pending')
            (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print('PPO parameter update verified:', steps, delta, flush=True)

    def progress(steps, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        values.update(steps=int(steps), elapsed_seconds=time.monotonic() - started)
        with (args.output / 'metrics.jsonl').open('a') as log:
            log.write(json.dumps(values) + '\n')
        print(json.dumps(values), flush=True)

    train = functools.partial(ppo.train,
        environment=env, eval_env=env, num_timesteps=args.steps,
        num_envs=args.envs, num_eval_envs=min(args.envs, 4), episode_length=500,
        unroll_length=16, batch_size=args.envs, num_minibatches=1,
        num_updates_per_batch=2, num_evals=max(2, math.ceil(args.steps / args.checkpoint_every) + 1), learning_rate=3e-4,
        entropy_cost=.01, discounting=.97, normalize_observations=True,
        reward_scaling=1., clipping_epsilon=.2, gae_lambda=.95,
        max_grad_norm=1., seed=0, deterministic_eval=True,
        network_factory=functools.partial(networks.make_ppo_networks,
            policy_hidden_layer_sizes=(64, 64), value_hidden_layer_sizes=(128, 128),
            # Brax 0.14.2 cannot reload a null mean initializer. This is the
            # same lecun_uniform initializer its None default resolves to.
            mean_kernel_init_fn=jax.nn.initializers.lecun_uniform,
            policy_obs_key='state', value_obs_key='privileged_state'),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        save_checkpoint_path=str(args.output / 'checkpoints'),
        restore_checkpoint_path=args.restore, progress_fn=progress,
        policy_params_fn=policy_update, run_evals=not args.no_eval,
    )
    try:
        train()
    except BaseException as error:
        metadata.update(status='failed; walking acceptance pending',
                        error=f'{type(error).__name__}: {error}')
        raise
    else:
        metadata.update(status='requested PPO updates completed; walking acceptance pending')
    finally:
        metadata['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
