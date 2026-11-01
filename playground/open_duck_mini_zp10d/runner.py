"""Small-batch entrypoint for the original Brax PPO / MJX training workflow."""

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')  # CPU hosts the episode-log callback.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '1.0')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_llvm_module_compilation_parallelism=false')

import jax
import jax.numpy as jp
from mujoco.mjx._src import solver
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import checkpoint
from mujoco_playground import wrapper

from playground.open_duck_mini_zp10d.joystick import Joystick, TRAINING_XML, domain_randomize
from playground.open_duck_mini_zp10d.reference import ROOT, source_hashes
from playground.open_duck_mini_zp10d.actuators import signature, specification
from playground.open_duck_mini_zp10d.policy import make_ppo_networks, with_policy_std, with_liftoff_observation


class TaskResetWrapper(wrapper.Wrapper):
    """Reset task/controller memory with the cached pose, preserving PPO metrics."""

    def __init__(self, env, stream_path=None, stream_envs=None, prior_steps=0):
        super().__init__(env)
        self.stream_path, self.stream_envs = stream_path, stream_envs
        self.prior_steps = prior_steps
        self._stream_tick, self._stream_episode = 0, 0
        self._stream_lock = threading.Lock()

    def _publish(self, qpos, sim_time, tick, done, reward, reference_initialized=False):
        with self._stream_lock:
            self._publish_locked(qpos, sim_time, tick, done, reward, reference_initialized)

    def _publish_locked(self, qpos, sim_time, tick, done, reward, reference_initialized=False):
        tick = int(tick)
        if tick <= self._stream_tick:
            return
        self._stream_tick = tick
        self._stream_episode += int(done)
        frame = dict(qpos=np.asarray(qpos).tolist(), time=float(sim_time),
                     steps=self.prior_steps + tick * self.stream_envs,
                     phase_steps=tick * self.stream_envs,
                     environment=0, environments=self.stream_envs, episode=self._stream_episode,
                     reset=bool(done), reward=float(reward), published_at=time.time(),
                     reference_initialized=bool(reference_initialized),
                     source='actual MJX PPO training environment; sampled actions; no replay')
        temporary = self.stream_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(frame) + '\n')
        temporary.replace(self.stream_path)

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info['TaskResetWrapper_initial_info'] = {
            k: v for k, v in state.info.items()
            if k not in ('rng', 'steps', 'truncation', 'episode_done', 'episode_metrics')
            and not k.startswith(('AutoResetWrapper', 'TaskResetWrapper'))}
        if self.stream_path is not None:
            state.info['training_stream_tick'] = jp.zeros(state.done.shape, dtype=jp.int32)
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        def select(initial, current):
            done = state.done.astype(bool).reshape(state.done.shape + (1,) * (initial.ndim - state.done.ndim))
            return jp.where(done, initial, current)
        for key, initial in state.info['TaskResetWrapper_initial_info'].items():
            state.info[key] = jax.tree.map(select, initial, state.info[key])
        if self.stream_path is not None:
            state.info['training_stream_tick'] += 1
            if state.data.qpos.shape[0] == self.stream_envs:
                jax.debug.callback(self._publish, state.data.qpos[0], state.data.time[0],
                    state.info['training_stream_tick'][0], state.done[0], state.reward[0],
                    state.info.get('reference_initialized', jp.zeros_like(state.done, dtype=bool))[0])
        return state


def wrap_training(env, stream_path=None, stream_envs=None, prior_steps=0, **kwargs):
    # Upstream's cached reset deliberately leaves info untouched. Full resets
    # also replace EpisodeWrapper accounting, including timeout truncation;
    # preserve that accounting and clear only the task's own state instead.
    return TaskResetWrapper(wrapper.wrap_for_brax_training(env, **kwargs),
                            stream_path, stream_envs, prior_steps)


def reference_training_status(reference, experimental=False):
    """An imperfect imitation target may guide research, never certify walking."""
    failed = [stage for stage in ('raw_dynamics_audit', 'fitted_dynamics_audit')
              if not reference.get(stage, {}).get('passed')
              or not reference.get(stage, {}).get('moving_friction_opposes_velocity')]
    if failed and not experimental:
        raise ValueError('Reference has not passed moving-friction dynamics; regenerate it, '
                         'or explicitly use --experimental-reference for simulation research')
    return dict(mode='experimental' if experimental else 'validated-reference',
                failed_or_unverified_stages=failed,
                walking_accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=49152)
    parser.add_argument('--prior-steps', type=int,
                        help='verified environment steps spent before this run; independent of restored weight age')
    parser.add_argument('--envs', type=int, default=512)
    parser.add_argument('--checkpoint-every', type=int, default=8192)
    parser.add_argument('--learning-rate', type=float,
                        help='initial PPO learning rate; adaptive default 1e-4')
    parser.add_argument('--max-learning-rate', type=float, default=3e-4,
                        help='adaptive PPO learning-rate ceiling; minimum remains 1e-5')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-eval', action='store_true', help='update/checkpoint smoke test only')
    parser.add_argument('--output', type=Path, default=ROOT / 'training')
    parser.add_argument('--restore', type=str)
    parser.add_argument('--reward-mode', choices=('original', 'optimized'), default='optimized')
    parser.add_argument('--stand-only', action='store_true',
                        help='balance curriculum: sample only zero commands before gait training')
    parser.add_argument('--position-feedback', type=float, default=0.,
                        help='home joint-position feedback plus low-noise residual PPO; motor physics unchanged')
    parser.add_argument('--position-feedback-damping', type=float, default=0.,
                        help='software velocity feedback queued before the same transport delay')
    parser.add_argument('--adapt-feedback', action='store_true',
                        help='explicitly adapt a normal feedback policy to a different positive gain; preserve its unloaded posture scale')
    parser.add_argument('--policy-std', type=float,
                        help='normal exploration standard deviation; on restore, preserve the learned mean')
    parser.add_argument('--torque-limit-scale', type=float, default=1.,
                        help='simulation sensitivity test only: scale ZP10D force caps in memory')
    parser.add_argument('--reference-reset-fraction', type=float, default=0.,
                        help='Placo lift course: fraction of environments initialized from moving reference states')
    parser.add_argument('--observation-clip', type=float, default=5.,
                        help='bound normalized actor/critic inputs, including first rare landing events')
    parser.add_argument('--direct-locomotion', action='store_true',
                        help='learn from supported air time/progress without fixed Placo contact or COM trajectories')
    parser.add_argument('--lift-curriculum', action='store_true',
                        help='bootstrap alternating unloading/liftoff before requiring forward progress')
    parser.add_argument('--coordinated-lift', action='store_true',
                        help='Placo lift research: free transfer, hold roll after physical liftoff; no hand-coded pitch folding')
    parser.add_argument('--reference-action', action='store_true',
                        help='use anticipated Placo leg postures as nominal actions, with PPO residual corrections')
    parser.add_argument('--reference-feedforward', action='store_true',
                        help='add the audited Placo motor torques to nominal reference actions')
    parser.add_argument('--sampled-reference-control', action='store_true',
                        help='50 Hz Placo motor commands with averaged forces and separate posture/force timing')
    parser.add_argument('--drift-observation', action='store_true',
                        help='expose the existing filtered planar velocity error in two unused actor inputs')
    parser.add_argument('--drift-cost-scale', type=float, default=2.,
                        help='positive magnitude of the sampled Placo planar-drift penalty')
    parser.add_argument('--forward-progress-scale', type=float, default=0.,
                        help='signed command-axis speed reward; use to penalize backward drift')
    parser.add_argument('--live-training', action='store_true',
                        help='publish the actual first training environment for the local native window')
    parser.add_argument('--experimental-reference', action='store_true',
                        help='simulation research with an unvalidated imitation target; physical limits remain enforced')
    args = parser.parse_args()
    adaptive_ppo = args.direct_locomotion or args.lift_curriculum
    if args.drift_observation and not args.sampled_reference_control:
        parser.error('drift-observation requires sampled-reference-control')
    if not math.isfinite(args.drift_cost_scale) or args.drift_cost_scale <= 0:
        parser.error('drift-cost-scale must be positive and finite')
    if not math.isfinite(args.forward_progress_scale) or args.forward_progress_scale < 0:
        parser.error('forward-progress-scale must be finite and nonnegative')
    learning_rate = args.learning_rate if args.learning_rate is not None else (1e-4 if adaptive_ppo else 3e-4)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        parser.error('learning-rate must be positive and finite')
    if (not math.isfinite(args.max_learning_rate) or args.max_learning_rate < 1e-5
            or (adaptive_ppo and not 1e-5 <= learning_rate <= args.max_learning_rate)):
        parser.error('adaptive learning rate must satisfy 1e-5 <= learning-rate <= max-learning-rate')
    args.output = args.output.resolve()
    if args.restore:
        args.restore = str(Path(args.restore).resolve())
        restored = json.loads((Path(args.restore).parent.parent / 'metadata.json').read_text())
        if restored.get('reward_specification', {}).get('planar_drift_observation') and not args.drift_observation:
            raise ValueError('Cannot discard learned drift observation inputs on restore')
        if restored.get('position_feedback_gain', 0.) != args.position_feedback:
            if not (args.adapt_feedback and restored.get('policy_distribution') == 'normal'
                    and restored.get('position_feedback_gain', 0.) > 0
                    and args.position_feedback > 0):
                raise ValueError('Checkpoint position-feedback controller differs; use explicit --adapt-feedback or start a new policy')
        if restored.get('position_feedback_damping', 0.) != args.position_feedback_damping:
            raise ValueError('Checkpoint velocity-feedback controller differs; start a new policy')
        if restored['source_hashes'] != source_hashes():
            raise ValueError('Warm-start checkpoint belongs to a different source model')
        if restored.get('actuator_signature') != signature():
            raise ValueError('Warm-start checkpoint uses different actuator dynamics')
        if restored.get('reference_sha256') != hashlib.sha256(
                (ROOT / 'data/polynomial_coefficients.pkl').read_bytes()).hexdigest():
            raise ValueError('Warm-start checkpoint uses a different reference')
    if args.prior_steps is None:
        args.prior_steps = (restored.get('prior_steps', 0) + int(Path(args.restore).name)
                            if args.restore else 0)
    if min(args.envs, args.steps, args.checkpoint_every) < 1:
        parser.error('steps/envs/checkpoint-every must be positive')
    if args.prior_steps < 0:
        parser.error('prior-steps must be nonnegative')
    if args.adapt_feedback and not args.restore:
        parser.error('adapt-feedback requires a checkpoint')
    restored_steps = restored.get('prior_steps', 0) + int(Path(args.restore).name) if args.restore else 0
    if args.prior_steps < restored_steps:
        parser.error('prior-steps cannot be less than the restored checkpoint cumulative steps')
    if not math.isfinite(args.observation_clip) or args.observation_clip <= 0:
        parser.error('observation-clip must be positive and finite')
    if not math.isfinite(args.position_feedback) or args.position_feedback < 0:
        parser.error('position-feedback must be finite and nonnegative')
    if not math.isfinite(args.torque_limit_scale) or args.torque_limit_scale <= 0:
        parser.error('torque-limit-scale must be positive and finite')
    if (not 0. <= args.reference_reset_fraction <= 1. or (args.reference_reset_fraction and
            (args.direct_locomotion or not args.lift_curriculum))):
        parser.error('reference-reset-fraction requires the Placo lift course and a fraction in [0, 1]')
    if not math.isfinite(args.position_feedback_damping) or args.position_feedback_damping < 0:
        parser.error('position-feedback-damping must be finite and nonnegative')
    if args.position_feedback_damping and not args.position_feedback:
        parser.error('position-feedback-damping requires position feedback')
    if args.policy_std is not None and (not math.isfinite(args.policy_std) or args.policy_std <= 0
                                       or not args.position_feedback):
        parser.error('policy-std must be positive and requires a normal feedback policy')
    initial_std = args.policy_std if args.policy_std is not None else (.05 / (1. + args.position_feedback) if args.position_feedback else 1.)
    reference = json.loads((ROOT / 'data/reference.json').read_text())
    reference_status = (dict(mode='direct-task; legacy trajectory disabled', walking_accepted=False,
                             foot_geometry_source_hashes=source_hashes()) if args.direct_locomotion
                        else reference_training_status(reference, args.experimental_reference))
    reference_status['trajectory_imitation_enabled'] = not args.direct_locomotion
    reference_status['support_schedule_only'] = False
    print('reference training status', reference_status, flush=True)
    env = Joystick(reward_mode=args.reward_mode, position_feedback_gain=args.position_feedback,
                   direct_locomotion=args.direct_locomotion, position_feedback_damping=args.position_feedback_damping,
                   lift_curriculum=args.lift_curriculum, torque_limit_scale=args.torque_limit_scale,
                   reference_reset_fraction=args.reference_reset_fraction, coordinated_lift=args.coordinated_lift,
                   reference_action=args.reference_action, reference_feedforward=args.reference_feedforward,
                   sampled_reference_control=args.sampled_reference_control,
                   drift_observation=args.drift_observation,
                   drift_cost_scale=args.drift_cost_scale,
                   forward_progress_scale=args.forward_progress_scale)
    feedback_changed = bool(args.restore and restored.get('position_feedback_gain', 0.) != args.position_feedback)
    if args.restore:
        expected_scale = np.asarray(restored['action_scale'])
        if feedback_changed:
            expected_scale = expected_scale * np.where(np.asarray(env._leg_mask),
                (1. + args.position_feedback) / (1. + restored['position_feedback_gain']), 1.)
        if not np.allclose(expected_scale, env._config.action_scale, rtol=1e-7, atol=1e-8):
            raise ValueError('Checkpoint motor-position action scale differs beyond the explicit feedback adaptation')
    if args.stand_only:
        env._commands = jp.zeros((1, 3), dtype=jp.float32)
    restored_guidance = (restored.get('coordination_guidance_version', 1)
                         if args.restore and restored.get('coordinated_lift') else None)
    guidance_changed = bool(args.restore and restored_guidance != env.coordination_guidance_version)
    controller_changed = feedback_changed or guidance_changed or bool(args.restore and
        restored.get('motor_target_slew_rad_s', 5.24) != env._target_slew_rate)
    reward_changed = bool(args.restore and
        (restored.get('reward_specification', {}).get('recipe') != env.reward_specification()['recipe']
         or restored.get('reward_specification', {}).get('planar_drift_cost_scale', 2.)
         != env.reward_specification().get('planar_drift_cost_scale', 2.)
         or restored.get('reward_specification', {}).get('forward_progress_scale', 0.)
         != env.reward_specification().get('forward_progress_scale', 0.)))
    contact_hash = env.training_contact_signature
    contact_changed = bool(args.restore and restored.get('training_contact_model_sha256') != contact_hash)
    reset_value = bool(args.restore and (controller_changed or contact_changed or (adaptive_ppo and reward_changed)))
    migrate_liftoff = bool(args.restore and args.lift_curriculum and not args.direct_locomotion
        and restored.get('reward_specification', {}).get('liftoff_task_observation') is None)
    migrate_drift = bool(args.restore and args.drift_observation and not
        restored.get('reward_specification', {}).get('planar_drift_observation'))
    if args.restore and args.lift_curriculum and not args.direct_locomotion:
        layout = restored.get('reward_specification', {}).get('liftoff_task_observation')
        if layout not in (None, 'allowed_plus_2qualified_in_head_command_slots'):
            raise ValueError('Cannot warm-start this policy with a different liftoff observation layout')
    print('devices', jax.devices(), 'action size', env.action_size, 'observations', env.observation_size, flush=True)
    if args.smoke:
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        t = time.monotonic()
        state = reset(jax.random.PRNGKey(0))
        initial_base = np.array(state.data.qpos[:7])
        state = step(state, jp.zeros(env.action_size))
        jax.block_until_ready(state.data.qpos)
        print('reset + first control step seconds', time.monotonic() - t, flush=True)
        t = time.monotonic()
        for _ in range(100):
            state = step(state, jp.zeros(env.action_size))
        jax.block_until_ready(state.data.qpos)
        assert bool(jp.isfinite(state.data.qpos).all())
        print('control steps/s', 100 / (time.monotonic() - t),
              'initial base', initial_base, 'base displacement', np.asarray(state.data.qpos[:3]) - initial_base[:3],
              'base', state.data.qpos[:7],
              'done', state.done, flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(source_hashes=source_hashes(), commands=np.asarray(env._commands).tolist(),
                    command_curriculum='stand-only' if args.stand_only else 'mixed',
                    position_feedback_gain=args.position_feedback,
                    restored_position_feedback_gain=restored.get('position_feedback_gain', 0.) if args.restore else None,
                    feedback_gain_change_on_restore=feedback_changed,
                    feedback_adaptation='actor/normalizer retained; leg action scales adjusted by (1+new_gain)/(1+old_gain); value reset' if feedback_changed else None,
                    position_feedback_damping=args.position_feedback_damping,
                    motor_target_slew_rad_s=env._target_slew_rate,
                    restored_motor_target_slew_rad_s=restored.get('motor_target_slew_rad_s', 5.24) if args.restore else None,
                    controller_change_on_restore=controller_changed,
                    direct_locomotion=args.direct_locomotion, gait_period_s=env.PRM.period,
                    lift_curriculum=args.lift_curriculum,
                    coordinated_lift=args.coordinated_lift,
                    reference_action=args.reference_action,
                    reference_feedforward=args.reference_feedforward,
                    sampled_reference_control=args.sampled_reference_control,
                    post_step_forward=env._refresh_post_step_fields,
                    sampled_reference_specification=env.sampled_reference_specification,
                    reference_torque_sha256=getattr(env, 'reference_torque_sha256', None),
                    reference_action_specification=('50Hz sampled nominal command; independent posture/force leads and interval-averaged force; current measured feedback; quintic 1.2s startup/stop fade; phase continues until fade completes'
                        if args.sampled_reference_control else 'Placo nominal motor posture anticipates two-tick delay plus end-of-step; measured position/velocity feedback uses current reference phase; quintic 1.2s startup; zero commands unchanged') if args.reference_action else None,
                    coordination_guidance_version=env.coordination_guidance_version,
                    restored_coordination_guidance_version=restored_guidance,
                    guidance_change_on_restore=guidance_changed,
                    live_training_stream=args.live_training,
                    coordination_guidance=('Placo phase keeps the nominal swing hip-roll path but removes its learned residual from the preceding double-support transfer; only the loaded support hip adds learned COM transfer; sampled nominal plus PPO residual before delay'
                        if args.sampled_reference_control else 'Placo imitation without hand-coded pitch folding; both rolls free during transfer, measured roll held only after supported 2mm liftoff until landing; applied before delay') if args.coordinated_lift else None,
                    reference_reset_fraction=args.reference_reset_fraction,
                    reference_reset_controller=('periodic torque preload, phase-matched prior issued commands, startup ramp complete; initial airborne foot excluded from eligible lift metrics/rewards'
                        if args.reference_reset_fraction and args.reference_action else None),
                    native_evaluation_initial_state='geometric home; reference reset is training-only',
                    learning_rate_schedule='ADAPTIVE_KL' if adaptive_ppo else 'NONE',
                    initial_learning_rate=learning_rate,
                    maximum_learning_rate=args.max_learning_rate if adaptive_ppo else None,
                    desired_kl=.01 if adaptive_ppo else None,
                    discounting=.99 if adaptive_ppo else .97,
                    policy_distribution='normal' if args.position_feedback else 'tanh_normal',
                    initial_policy_std=('restored learned distribution' if args.restore and args.policy_std is None
                                        else initial_std),
                    restored_policy_std_override=args.policy_std if args.restore else None,
                    liftoff_observation_migration='zero actor/critic input rows 9:11; unit-scale normalizer; preserve old outputs' if migrate_liftoff else None,
                    drift_observation_migration='zero actor/critic input rows 11:13; unit-scale normalizer; preserve old outputs' if migrate_drift else None,
                    normalized_observation_clip=args.observation_clip,
                    restored_normalized_observation_clip=restored.get('normalized_observation_clip') if args.restore else None,
                    position_feedback_target='home geometry plus policy leg posture residual and optional velocity feedback; queued before transport delay',
                    actuator_signature=signature(), actuator_specification=specification(),
                    torque_limit_scale=args.torque_limit_scale,
                    actuator_force_limits_nm=dict(zip(env.actuator_names, np.asarray(env._effort_limits).tolist())),
                    restored_torque_limit_scale=restored.get('torque_limit_scale', 1.) if args.restore else None,
                    torque_limit_scope='MJCF prior' if args.torque_limit_scale == 1. else 'simulation sensitivity test; not hardware capability',
                    action_delay_control_ticks=2,
                    action_scale=list(env._config.action_scale), actuator_names=env.actuator_names,
                    backend=jax.default_backend(), restore_checkpoint=args.restore,
                    restore_semantics=('fresh policy/value/normalizer; earlier spent steps are accounting only' if not args.restore else
                        'policy/normalizer warm start; value reset for changed reward/controller/contact model' if reset_value else
                        'policy/value/normalizer warm start; optimizer and run step counter reset'),
                    value_reset_for_reward_change=bool(reset_value and reward_changed),
                    value_reset_for_controller_change=bool(reset_value and controller_changed),
                    value_reset_for_contact_change=bool(reset_value and contact_changed),
                    reference_sha256=hashlib.sha256((ROOT / 'data/polynomial_coefficients.pkl').read_bytes()).hexdigest(),
                    training_contact_model=env.training_contact_metadata['approximation'],
                    sole_contact_method=env.training_contact_metadata['sole_contact_method'],
                    sole_contact_adapter_sha256=env.training_contact_metadata['sole_contact_adapter_sha256'],
                    training_contact_model_sha256=contact_hash,
                    restored_training_contact_model_sha256=restored.get('training_contact_model_sha256') if args.restore else None,
                    status='starting; not accepted walking', requested_steps=args.prior_steps + args.steps,
                    prior_steps=args.prior_steps, phase_requested_steps=args.steps,
                    restored_checkpoint_steps=restored_steps,
                    restored_from_earlier_checkpoint=bool(args.restore and args.prior_steps > restored_steps),
                    completed_steps=args.prior_steps, phase_completed_steps=0,
                    num_envs=args.envs, num_eval_envs=min(args.envs, 4),
                    gpu_memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
                    run_evals=not args.no_eval)
    metadata['episode_reset_mode'] = 'cached pose/observation plus task/controller info; PPO accounting and RNG preserved'
    metadata['training_episode_metrics'] = not args.no_eval
    metadata['native_profile_evaluation'] = not args.no_eval
    metadata['reference_training_status'] = reference_status
    metadata['linesearch_implementation'] = solver._LSPoint.create.__name__
    metadata['mjx_adapter_sha256'] = hashlib.sha256((ROOT.parent / 'common/mjx_compat.py').read_bytes()).hexdigest()
    metadata['reward_specification'] = env.reward_specification()
    metadata['domain_randomization'] = (dict(contact_friction_scale=[.9, 1.1],
        identified_drive_gain_scale=[.95, 1.05], antenna_gains='unchanged',
        mass_inertia_geometry_and_force_caps='unchanged') if args.sampled_reference_control else None)
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
            metadata.update(completed_steps=args.prior_steps + int(steps), phase_completed_steps=int(steps), policy_parameter_max_delta=delta,
                            status='training; walking acceptance pending')
            (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print('PPO parameter update verified:', args.prior_steps + int(steps), delta, flush=True)

    def progress(steps, metrics):
        values = {k: float(v) for k, v in metrics.items()}
        values.update(steps=args.prior_steps + int(steps), phase_steps=int(steps), elapsed_seconds=time.monotonic() - started)
        with (args.output / 'metrics.jsonl').open('a') as log:
            log.write(json.dumps(values) + '\n')
        print(json.dumps(values), flush=True)
        if steps > 0 and metadata['native_profile_evaluation'] and 'eval/avg_episode_length' in values:
            # Called after checkpoint save and GPU evaluation. Native physics is
            # serial in this same capped process, without reacquiring its lock.
            from playground.open_duck_mini_zp10d.evaluate import main as native_evaluate
            profiles = [('stand', ['--forward', '0'])]
            if not args.stand_only:
                profiles += [('walk', []), ('transition', ['--warmup', '1', '--walk-seconds', '5'])]
            profiles += [('stand_stochastic', ['--forward', '0', '--stochastic'])]
            if not args.stand_only:
                profiles += [('walk_stochastic', ['--stochastic'])]
                if args.sampled_reference_control:
                    profiles += [('walk_stochastic_seed1', ['--stochastic', '--seed', '1']),
                                 ('walk_yaw', ['--initial-yaw', '.7'])]
            for label, profile in profiles:
                native_evaluate([str(args.output / 'checkpoints' / f'{int(steps):012d}'),
                    '--seconds', '8', '--output', str(args.output / 'native' / f'{int(steps):012d}' / label),
                    *profile])

    restore_params = None
    if args.restore and (args.policy_std is not None or migrate_liftoff or migrate_drift):
        restore_params = list(checkpoint.load(args.restore))
        if migrate_liftoff:
            restore_params = list(with_liftoff_observation(tuple(restore_params)))
        if migrate_drift:
            restore_params = list(with_liftoff_observation(tuple(restore_params), slots=slice(11, 13)))
        if args.policy_std is not None:
            restore_params[1] = with_policy_std(restore_params[1], args.policy_std)
        restore_params = tuple(restore_params)

    train = functools.partial(ppo.train,
        environment=env, eval_env=env, num_timesteps=args.steps,
        randomization_fn=domain_randomize if args.sampled_reference_control else None,
        num_envs=args.envs, num_eval_envs=min(args.envs, 4), episode_length=500,
        unroll_length=16, batch_size=args.envs, num_minibatches=1,
        num_updates_per_batch=2, num_evals=max(2, math.ceil(args.steps / args.checkpoint_every) + 1),
        learning_rate=learning_rate,
        learning_rate_schedule='ADAPTIVE_KL' if adaptive_ppo else None,
        desired_kl=.01, learning_rate_schedule_min_lr=1e-5, learning_rate_schedule_max_lr=args.max_learning_rate,
        entropy_cost=.01, discounting=.99 if adaptive_ppo else .97, normalize_observations=True,
        reward_scaling=1., clipping_epsilon=.2, gae_lambda=.95,
        max_grad_norm=1., seed=0, deterministic_eval=True,
        network_factory=functools.partial(make_ppo_networks,
            observation_clip=args.observation_clip,
            policy_hidden_layer_sizes=(64, 64), value_hidden_layer_sizes=(128, 128),
            # Explicit factories survive Brax checkpoint reload. Feedback starts
            # with zero mean; the legacy tanh network keeps its original factory.
            distribution_type='normal' if args.position_feedback else 'tanh_normal',
            noise_std_type='log' if args.position_feedback else 'scalar',
            init_noise_std=initial_std,
            mean_clip_scale=1. if args.position_feedback else None,
            mean_kernel_init_fn=jax.nn.initializers.constant if args.position_feedback else jax.nn.initializers.lecun_uniform,
            mean_kernel_init_kwargs={'value': 0.} if args.position_feedback else {},
            policy_obs_key='state', value_obs_key='privileged_state'),
        wrap_env_fn=functools.partial(wrap_training,
            stream_path=args.output / 'training_state.json' if args.live_training else None,
            stream_envs=args.envs, prior_steps=args.prior_steps),
        save_checkpoint_path=str(args.output / 'checkpoints'),
        restore_checkpoint_path=args.restore if restore_params is None else None,
        restore_params=restore_params, progress_fn=progress,
        restore_value_fn=not reset_value,
        policy_params_fn=policy_update, run_evals=not args.no_eval,
        log_training_metrics=not args.no_eval, training_metrics_steps=4 * args.envs * 16,
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
