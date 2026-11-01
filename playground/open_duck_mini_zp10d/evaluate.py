"""Evaluate a checkpoint through the unchanged full native MuJoCo scene."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.agents.ppo import checkpoint

from playground.open_duck_mini_zp10d.reference import SCENE, ROOT, source_hashes, apply_reference_stance
from playground.open_duck_mini_zp10d.actuators import (configure, signature, position_feedback_action,
    coordinated_lift_action, hold_cleared_roll, planned_swing_roll_mask,
    suppress_swing_roll_residual, reference_posture_action,
    reference_torque_inputs, periodic_reference_torque)
from playground.open_duck_mini_zp10d.liftoff import update_liftoff, liftoff_features, command_liftoff_runs, completed_lift_reward
from playground.open_duck_mini_zp10d.policy import make_ppo_networks


def observation(gyro, acceleration, command, angles, velocities, history, target, contact, phase, default):
    """The actor's 113 features, in the same order as the upstream task."""
    return np.concatenate([gyro, acceleration, command, angles - default,
                           velocities * .05, history.ravel(), target, contact, phase])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--forward', type=float, help='default: checkpoint forward reference speed')
    parser.add_argument('--action-delay', type=int, choices=(0, 1, 2), default=2,
                        help='fixed delay in control ticks; training uses the identified delay rounded up to 2')
    parser.add_argument('--output', type=Path, help='diagnostic output directory; defaults to checkpoint run')
    parser.add_argument('--warmup', type=float, default=0., help='initial zero-command seconds')
    parser.add_argument('--walk-seconds', type=float, help='switch back to zero after this many walking seconds')
    parser.add_argument('--live', action='store_true', help='publish live.html/live.json in the output directory')
    parser.add_argument('--stochastic', action='store_true', help='diagnose sampled actions versus deterministic deployment')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--initial-yaw', type=float, default=0.,
                        help='initial free-base heading in radians')
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint.resolve()
    if args.seconds <= 0 or args.warmup < 0 or (args.walk_seconds is not None and args.walk_seconds <= 0):
        parser.error('seconds must be positive')
    if not np.isfinite(args.initial_yaw):
        parser.error('initial-yaw must be finite')
    meta = json.loads((args.checkpoint.parent.parent / 'metadata.json').read_text())
    lift_observation = meta.get('reward_specification', {}).get('liftoff_task_observation')
    drift_observation = meta.get('reward_specification', {}).get('planar_drift_observation')
    if drift_observation not in (None, 'lowpass_1s_reference_velocity_error_in_current_yaw_frame_over_.010_in_slots_11_13'):
        raise ValueError('Unknown planar drift observation layout')
    gait_velocity = np.zeros(2)
    if lift_observation not in (None, 'allowed_plus_2qualified_in_clock_slots',
                                'allowed_plus_2qualified_in_head_command_slots'):
        raise ValueError('Unknown liftoff task observation layout')
    last_landed, qualified = -1, np.zeros(2, dtype=bool)
    whole_foot_peak, completed_lifts = np.zeros(2), np.zeros(2, dtype=int)
    completion_credit = 0.
    if meta['source_hashes'] != source_hashes():
        raise ValueError('Checkpoint source hashes do not match the current model')
    if meta.get('reference_sha256') != hashlib.sha256(
            (ROOT / 'data/polynomial_coefficients.pkl').read_bytes()).hexdigest():
        raise ValueError('Checkpoint uses a different reference; use its matching archived data')
    if meta.get('actuator_signature') != signature():
        raise ValueError('Checkpoint uses different actuator dynamics')
    if args.forward is None:
        args.forward = max(c[0] for c in meta['commands'])
    policy = jax.jit(checkpoint.load_policy(str(args.checkpoint),
        network_factory=make_ppo_networks, deterministic=not args.stochastic))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    torque_limit_scale = meta.get('torque_limit_scale', 1.)
    configure(model, torque_limit_scale)
    if not meta.get('direct_locomotion'):
        apply_reference_stance(model, json.loads((ROOT / 'data/reference.json').read_text()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    if args.initial_yaw:
        initial = data.qpos[3:7].copy()
        rotation = np.array([np.cos(args.initial_yaw/2), 0., 0., np.sin(args.initial_yaw/2)])
        mujoco.mju_mulQuat(data.qpos[3:7], rotation, initial)
    mujoco.mj_forward(model, data)
    initial_rotation = data.xmat[1].reshape(3, 3)
    reference_heading = np.arctan2(initial_rotation[1, 0], initial_rotation[0, 0])
    names = [model.actuator(i).name for i in range(model.nu)]
    if names != meta['actuator_names']:
        raise ValueError('Actuator ordering mismatch')
    joint_ids = model.actuator_trnid[:, 0]
    qa, va = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    target = model.key('home').ctrl.copy()
    default = target.copy()
    scale = np.array(meta['action_scale'])
    feedback_gain = meta.get('position_feedback_gain', 0.)
    feedback_damping = meta.get('position_feedback_damping', 0.)
    target_slew = meta.get('motor_target_slew_rad_s', 5.24)
    geometric_home = model.key('home').qpos[qa].copy()
    swing_roll_hold = geometric_home[[1, 10]].copy()
    planned_swing_roll = np.zeros(2, dtype=bool)
    guidance_version = meta.get('coordination_guidance_version', 1)
    if meta.get('coordinated_lift') and guidance_version not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
        raise ValueError('Unknown coordinated lift controller version')
    if (guidance_version in (5, 6, 7, 8, 9, 10, 11, 12, 13)) != bool(meta.get('reference_action')):
        raise ValueError('Nominal Placo action metadata/controller mismatch')
    if ((guidance_version == 6 and not meta.get('reference_feedforward'))
            or (meta.get('reference_feedforward') and guidance_version not in (6, 7, 8, 9, 10, 11, 12, 13))):
        raise ValueError('Nominal Placo torque metadata/controller mismatch')
    if (guidance_version in (8, 9, 10, 11, 12, 13)) != bool(meta.get('sampled_reference_control')):
        raise ValueError('Sampled Placo metadata/controller mismatch')
    if guidance_version >= 10 and not meta.get('post_step_forward'):
        raise ValueError('Synchronized sensor timestamp missing from metadata')
    leg_mask = np.array([any(s in name for s in ('hip', 'knee', 'ankle')) for name in names])
    history = np.zeros((3, model.nu))
    action_history = history.copy()
    command = np.array([args.forward if args.warmup == 0 else 0., 0., 0., 0., 0., 0., 0.])
    contact = np.zeros(2)
    feet = [model.site_bodyid[model.site(side + '_foot').id] for side in ('left', 'right')]
    floor = model.geom('floor').id
    foot_vertices = []
    for body in feet:
        vertices = []
        active_collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        for gid in np.where((model.geom_bodyid == body) & active_collision)[0]:
            mid = model.geom_dataid[gid]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            vertices.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
        foot_vertices.append(np.concatenate(vertices))
    def sensor(name):
        sid = model.sensor(name).id
        adr = model.sensor_adr[sid]
        return data.sensordata[adr:adr + model.sensor_dim[sid]]
    peak = np.zeros(model.nu)
    saturation = np.zeros(model.nu)
    min_limit_margin = np.full(model.nu, np.inf)
    self_contacts, nonfoot_contacts, foot_floor_contacts = {}, {}, {}
    positions, clearances, forces, tilts, slip_speeds, commands, support_matches = [], [], [], [], [], [], []
    velocities, motor_commands, motor_torques, actions = [], [], [], []
    stop = 'duration'
    ref = json.loads((ROOT / 'data/reference.json').read_text())
    period_steps = round(meta.get('gait_period_s', ref['period']) / .02)
    if meta.get('reference_action'):
        from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
        reference_motion = PolyReferenceMotion(str(ROOT / 'data/polynomial_coefficients.pkl'))
        reference_order = [ref['joint_names'].index(n) for n in names]
    if meta.get('reference_feedforward'):
        torque_times, reference_torques, torque_hash = reference_torque_inputs(ref)
        if torque_hash != meta['reference_torque_sha256']:
            raise ValueError('Nominal torque reference checksum mismatch')
    if meta.get('sampled_reference_control'):
        from playground.open_duck_mini_zp10d.actuators import sampled_reference_inputs, sampled_reference_action
        sampled_corrections, sampled_specification = sampled_reference_inputs(
            ref, names, geometric_home, default, model.actuator_gainprm[:, 0], feedback_gain, feedback_damping,
            stop_fade=guidance_version >= 9)
        if sampled_specification != meta['sampled_reference_specification']:
            raise ValueError('Sampled nominal command specification mismatch')
    if not any(np.allclose([args.forward, 0., 0.], c, atol=1e-8) for c in meta['commands']):
        raise ValueError('Command is outside the trained reference domain')
    sites = [model.site(side + '_foot').id for side in ('left', 'right')]
    total_weight = float(model.body_subtreemass[1] * abs(model.opt.gravity[2]))
    minimum_support_force = meta.get('reward_specification', {}).get('minimum_support_force_n')
    target_contacts = np.array(ref['reward_reference']['contacts'], dtype=bool) if 'reward_reference' in ref else None
    planned_swing_schedule = (planned_swing_roll_mask(target_contacts)
                              if target_contacts is not None else None)
    if meta.get('direct_locomotion') and meta.get('reward_specification', {}).get('fixed_support_schedule') is False:
        target_contacts = None
    elif target_contacts is not None and meta.get('direct_locomotion'):
        target_contacts = np.roll(target_contacts, 1, axis=0)
    # Upstream reset creates MJX data without forward: first sensors and phase
    # are zero. Later observations are captured before last_act is advanced.
    phase = np.zeros(2)
    if lift_observation == 'allowed_plus_2qualified_in_clock_slots':
        phase = liftoff_features(last_landed, qualified, np)
    elif lift_observation == 'allowed_plus_2qualified_in_head_command_slots':
        command[3:5] = liftoff_features(last_landed, qualified, np)
    obs = observation(np.zeros(3), np.zeros(3), command, data.qpos[qa], data.qvel[va],
                      history, target, contact, phase, default)
    initial_position = data.qpos[:3].copy()
    substeps = round(.02 / model.opt.timestep)
    integrated_steps = 0
    live = None
    if args.live:
        from playground.open_duck_mini_zp10d.visualize_evaluation import LiveView
        live = LiveView(model, args.output or args.checkpoint.parent.parent,
                        '完整原生模型 · PPO检查点 ' + args.checkpoint.name)
    for tick in range(round(args.seconds / .02)):
        assert obs.shape == (113,)
        rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), tick) if args.stochastic else jax.random.PRNGKey(0)
        action, _ = policy({'state': jp.array(obs)}, rng)
        action = np.asarray(action)
        if meta.get('sampled_reference_control'):
            if tick == 0:
                motion_step = 0
                blend_ticks = 0
            moving = np.linalg.norm(command[:3]) > 1e-6
            running = moving or (guidance_version >= 9 and blend_ticks > 0)
            motion_step = motion_step if running else 0
            motion_phase = motion_step % period_steps
            action = sampled_reference_action(action, motion_phase, command, motion_step*.02,
                sampled_corrections, scale, np, blend_ticks=blend_ticks if guidance_version >= 9 else None)
            if guidance_version >= 13:
                swing = running & planned_swing_schedule[motion_phase]
                nominal = sampled_reference_action(np.zeros_like(action), motion_phase, command,
                    motion_step*.02, sampled_corrections, scale, np,
                    blend_ticks=blend_ticks if guidance_version >= 9 else None)
                action = suppress_swing_roll_residual(action, nominal, swing, np)
            elif guidance_version >= 12:
                swing = running & planned_swing_schedule[motion_phase]
                entered = swing & ~planned_swing_roll
                swing_roll_hold = np.where(entered, data.qpos[qa][[1, 10]], swing_roll_hold)
                planned_swing_roll = swing
                action = hold_cleared_roll(action, swing, swing_roll_hold,
                    geometric_home, default, scale, feedback_gain, np)
            elif guidance_version >= 11:
                action = hold_cleared_roll(action, qualified, swing_roll_hold,
                    geometric_home, default, scale, feedback_gain, np)
            motion_step = motion_step + 1 if running else 0
            if guidance_version >= 9:
                blend_ticks = np.clip(blend_ticks + (1 if moving else -1), 0, 60)
        elif meta.get('reference_action'):
            future = tick % period_steps + meta['action_delay_control_ticks'] + 1
            reference = reference_motion.get_reference_motion(*command[:3], future)
            current = (reference_motion.get_reference_motion(*command[:3], tick % period_steps)
                       if guidance_version >= 7 else reference)
            motor_offset = feedback_damping * current[16 + np.array(reference_order)] if guidance_version >= 7 else 0.
            if meta.get('reference_feedforward'):
                tau = periodic_reference_torque(future * .02, torque_times, reference_torques, np)
                motor_offset = (tau / model.actuator_gainprm[:, 0] - (default - geometric_home)
                    + feedback_damping * current[16 + np.array(reference_order)])
            action = reference_posture_action(action, reference[reference_order], geometric_home, scale,
                leg_mask, feedback_gain, command, tick * .02, np, motor_offset,
                current[reference_order] if guidance_version >= 7 else None)
        if meta.get('coordinated_lift'):
            if guidance_version < 3:
                action = coordinated_lift_action(action, obs[-2:], command, scale, feedback_gain, np,
                                                 quiet_swing_roll=guidance_version == 1)
            if 2 <= guidance_version < 8:
                action = hold_cleared_roll(action, qualified, swing_roll_hold, geometric_home,
                                           default, scale, feedback_gain, np)
        assert np.isfinite(action).all()
        action_history[1:] = action_history[:-1].copy()
        action_history[0] = position_feedback_action(action, data.qpos[qa], geometric_home, scale,
                                                   leg_mask, feedback_gain, data.qvel[va], feedback_damping) if feedback_gain else action
        requested = np.clip(default + action_history[args.action_delay] * scale,
                            model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
        target = (requested if target_slew is None else
                  np.clip(requested, target - target_slew * .02, target + target_slew * .02))
        data.ctrl[:] = target
        force = np.zeros(2)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            integrated_steps += 1
            peak = np.maximum(peak, abs(data.actuator_force))
            saturation += (abs(data.actuator_force) >= .999 * model.actuator_forcerange[:, 1]) * model.opt.timestep
            min_limit_margin = np.minimum(min_limit_margin,
                np.minimum(data.qpos[qa] - model.jnt_range[joint_ids, 0],
                           model.jnt_range[joint_ids, 1] - data.qpos[qa]))
            for ci in range(data.ncon):
                pair = data.contact[ci].geom
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                magnitude = float(np.linalg.norm(wrench[:3]))
                if magnitude <= .05:
                    continue
                if floor in pair:
                    other = pair[0] if pair[1] == floor else pair[1]
                    if model.geom_bodyid[other] not in feet:
                        name = model.geom(other).name
                        nonfoot_contacts[name] = max(nonfoot_contacts.get(name, 0.), magnitude)
                    else:
                        name = model.geom(other).name
                        foot_floor_contacts[name] = max(foot_floor_contacts.get(name, 0.), magnitude)
                else:
                    name = ' / '.join(sorted(model.body(int(model.geom_bodyid[g])).name for g in pair))
                    self_contacts[name] = max(self_contacts.get(name, 0.), magnitude)
        # Align reported contacts/geometry/sensors with the integrated qpos.
        # mj_step otherwise leaves these fields one physics step behind it.
        mujoco.mj_forward(model, data)
        contact[:] = 0
        slip = np.zeros(2)
        foot_motion = np.zeros((2, 6))
        for side, site in enumerate(sites):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, foot_motion[side], 0)
        for ci in range(data.ncon):
            pair = data.contact[ci].geom
            if floor not in pair:
                continue
            other = pair[0] if pair[1] == floor else pair[1]
            for side, body in enumerate(feet):
                if model.geom_bodyid[other] == body and data.contact[ci].dist <= 0:
                    contact[side] = 1
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    world_force = data.contact[ci].frame.reshape(3, 3).T @ wrench[:3]
                    upward = max(0., world_force[2] * (1 if pair[0] == floor else -1))
                    force[side] += upward
                    point_velocity = foot_motion[side, 3:] + np.cross(
                        foot_motion[side, :3], data.contact[ci].pos - data.site_xpos[sites[side]])
                    slip[side] += upward * np.linalg.norm(point_velocity[:2])
        slip /= np.maximum(force, 1e-8)
        slip *= force > .5
        height = [float((v @ data.xmat[b].reshape(3, 3).T + data.xpos[b])[:, 2].min())
                  for b, v in zip(feet, foot_vertices)]
        tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1, 1))))
        positions.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        motor_commands.append(target.copy())
        motor_torques.append(data.actuator_force.copy())
        actions.append(action.copy())
        clearances.append(height)
        forces.append(force)
        tilts.append(tilt)
        slip_speeds.append(slip)
        commands.append(command[0])
        if live is not None:
            live.update((tick + 1) * .02, data.qpos, height, force, target)
        if target_contacts is not None:
            phase_tick = motion_step % period_steps if meta.get('sampled_reference_control') else (tick+1)%period_steps
            desired = target_contacts[phase_tick * len(target_contacts) // period_steps] if abs(command[0]) > 1e-6 else np.ones(2, dtype=bool)
            support_matches.append((force > .5) == desired)
        phase_tick = ((motion_step % period_steps if running else 1)
                      if meta.get('sampled_reference_control') else (tick+1)%period_steps)
        phase_angle = 2 * np.pi * phase_tick / period_steps
        phase = np.array([np.cos(phase_angle), np.sin(phase_angle)])
        if lift_observation:
            previous_qualified = qualified.copy()
            terminal = (not np.isfinite(data.qpos).all() or data.qpos[2] < .10
                        or tilt > 45 or bool(nonfoot_contacts) or bool(any(data.warning.number)))
            completion_active = (np.linalg.norm(command[:3]) > 1e-6 or
                                 (meta.get('sampled_reference_control') and blend_ticks > 0))
            earned, whole_foot_peak = completed_lift_reward(qualified, whole_foot_peak,
                np.array(height), force, data.xmat[1].reshape(3,3)[2,2], terminal, np,
                active=completion_active)
            completion_credit += float(earned)
            completed_lifts += (qualified & (force > .5) & (force[::-1] > .5)
                                & (tilt <= 20.) & ~np.bool_(terminal) & completion_active)
            last_landed, qualified, _ = update_liftoff(last_landed, qualified,
                np.array(height), force, total_weight, np, minimum_support_force=minimum_support_force,
                qualification_height=meta.get('reward_specification', {}).get('liftoff_qualification_height_m', .002))
            if guidance_version < 12:
                swing_roll_hold = np.where(qualified & ~previous_qualified,
                                           data.qpos[qa][[1, 10]], swing_roll_hold)
            if lift_observation == 'allowed_plus_2qualified_in_clock_slots':
                phase = liftoff_features(last_landed, qualified, np)
            else:
                command[3:5] = liftoff_features(last_landed, qualified, np)
        next_time = (tick + 1) * .02
        if drift_observation:
            # Match training's post-step filter and its just-advanced Placo
            # phase; remove planned transfer before exposing drift to the actor.
            c, s = np.cos(reference_heading), np.sin(reference_heading)
            velocity = np.array([c*data.qvel[0]+s*data.qvel[1], -s*data.qvel[0]+c*data.qvel[1]])
            current = reference_motion.get_reference_motion(*command[:3], phase_tick)
            gait_velocity += .02/1.02 * (velocity-current[34:36]-gait_velocity)
            rotation = data.xmat[1].reshape(3, 3)
            relative_yaw = np.arctan2(rotation[1, 0], rotation[0, 0]) - reference_heading
            c, s = np.cos(relative_yaw), np.sin(relative_yaw)
            command[5:7] = np.array([c*gait_velocity[0]+s*gait_velocity[1],
                                    -s*gait_velocity[0]+c*gait_velocity[1]]) / .010
        command[0] = args.forward if (next_time >= args.warmup and
            (args.walk_seconds is None or next_time < args.warmup + args.walk_seconds)) else 0.
        obs = observation(sensor('gyro'), sensor('accelerometer'), command,
                          data.qpos[qa], data.qvel[va], history, target, contact, phase, default)
        history[1:] = history[:-1].copy()
        history[0] = action
        if not np.isfinite(data.qpos).all() or data.qpos[2] < .10 or tilt > 45:
            stop = 'fall_or_nonfinite'
            break
        if nonfoot_contacts or any(data.warning.number):
            stop = 'nonfoot_ground_contact' if nonfoot_contacts else 'numerical_warning'
            break
    if live is not None:
        live.update(integrated_steps * model.opt.timestep, positions[-1], clearances[-1], forces[-1], target,
                    force=True)
        live.close(stop)
    airborne = ((np.array(clearances) >= .01) & (np.array(forces) < .5) &
                (np.array(forces)[:, ::-1] > .5 * total_weight))
    longest, run = np.zeros(2), np.zeros(2)
    for frame in airborne:
        run = np.where(frame, run + .02, 0.)
        longest = np.maximum(longest, run)
    airborne_2mm = ((np.array(clearances) >= .002) & (np.array(forces) < .5) &
                    (np.array(forces)[:, ::-1] > .5 * total_weight))
    longest_2mm, run_2mm = np.zeros(2), np.zeros(2)
    for frame in airborne_2mm:
        run_2mm = np.where(frame, run_2mm + .02, 0.)
        longest_2mm = np.maximum(longest_2mm, run_2mm)
    walking = abs(np.array(commands)) > 1e-6
    contact_supported = ((np.array(clearances) >= .01) & (np.array(forces) < .5)
                         & (np.array(forces)[:, ::-1] > .5))
    forward_speed = np.diff(np.r_[initial_position[0], np.array(positions)[:, 0]]) / .02
    report = dict(checkpoint=str(args.checkpoint), source_hashes=source_hashes(),
                  initial_yaw_rad=args.initial_yaw, displacement_frame='world',
                  actuator_signature=signature(),
                  torque_limit_scale=torque_limit_scale,
                  actuator_force_limits_nm=dict(zip(names, np.max(abs(model.actuator_forcerange), axis=1).tolist())),
                  torque_limit_scope=meta.get('torque_limit_scope', 'MJCF prior'),
                  native_scene='full, unchanged CAD-derived collision model',
                  elapsed_sim_seconds=integrated_steps * model.opt.timestep, stop=stop,
                  command=command.tolist(), fixed_action_delay_steps=args.action_delay,
                  policy_inference='stochastic' if args.stochastic else 'deterministic', policy_seed=args.seed,
                  policy_backend=jax.default_backend(),
                  position_feedback_gain=feedback_gain,
                  position_feedback_damping=feedback_damping,
                  motor_target_slew_rad_s=target_slew,
                  command_profile=dict(forward_m_s=args.forward, warmup_s=args.warmup, walk_seconds=args.walk_seconds),
                  reward_specification=meta.get('reward_specification', {'mode': 'original_legacy'}),
                  warnings=data.warning.number.tolist(),
                  max_tilt_deg=max(tilts), peak_torque_nm=dict(zip(names, peak.tolist())),
                  final_tilt_deg=tilts[-1],
                  final_home_pose_error_rad=float(np.max(abs(np.array(positions[-1])[qa] - model.key('home').qpos[qa]))),
                  saturation_duration_s=dict(zip(names, saturation.tolist())),
                  min_joint_limit_margin_rad=float(min_limit_margin.min()),
                  self_contact_peak_force_n=self_contacts, nonfoot_contact_peak_force_n=nonfoot_contacts,
                  foot_floor_contact_peak_force_n=foot_floor_contacts,
                  max_whole_foot_clearance_m=np.max(clearances, axis=0).tolist(),
                  longest_10mm_unloaded_sample_run_s=longest.tolist(),
                  longest_2mm_unloaded_sample_run_s=longest_2mm.tolist(),
                  total_10mm_unloaded_sample_time_s=(airborne.sum(axis=0) * .02).tolist(),
                  support_force_threshold_n=.5 * total_weight,
                  contact_supported_force_threshold_n=.5,
                  contact_supported_total_10mm_sample_time_s=(contact_supported.sum(axis=0) * .02).tolist(),
                  stable_completed_lift_count=completed_lifts.tolist(),
                  stable_completion_height_banded_credit=completion_credit,
                  completion_qualification_height_m=meta.get('reward_specification', {}).get('liftoff_qualification_height_m', .002),
                  loaded_contact_slip_distance_m=(np.sum(slip_speeds, axis=0) * .02).tolist(),
                  mean_support_contact_match=float(np.mean(support_matches)) if support_matches else None,
                  mean_forward_velocity_m_s=float((positions[-1][0] - initial_position[0]) / (integrated_steps * model.opt.timestep)),
                  mean_walking_forward_velocity_m_s=float(np.mean(forward_speed[walking])) if walking.any() else None,
                  walking_support_contact_match=float(np.mean(np.array(support_matches)[walking]))
                      if support_matches and walking.any() else None,
                  displacement_m=(positions[-1][:3] - initial_position).tolist(),
                  **command_liftoff_runs(clearances, forces, commands, total_weight),
                  **{'contact_supported_' + key: value for key, value in command_liftoff_runs(
                      clearances, forces, commands, total_weight, minimum_support_force=.5).items()},
                  acceptance='diagnostic only; continuous self clearance and full landing/recovery gates pending')
    out = args.output or args.checkpoint.parent.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(out / 'evaluation.npz', qpos=positions, foot_clearance=clearances,
             qvel=velocities, time=np.arange(1, len(positions) + 1) * .02,
             motor_target=motor_commands, motor_torque=motor_torques, action=actions,
             foot_force=forces, tilt_deg=tilts, contact_slip_speed=slip_speeds, forward_command=commands)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
