"""Evaluate a checkpoint through the unchanged full native MuJoCo scene."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.agents.ppo import checkpoint

from playground.open_duck_mini_zp10d.reference import SCENE, ROOT, source_hashes


def observation(gyro, acceleration, command, angles, velocities, history, target, contact, phase, default):
    """The actor's 113 features, in the same order as the upstream task."""
    return np.concatenate([gyro, acceleration, command, angles - default,
                           velocities * .05, history.ravel(), target, contact, phase])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--forward', type=float, default=.03333333)
    parser.add_argument('--action-delay', type=int, choices=(0, 1, 2), default=0,
                        help='fixed delay in control ticks; training samples 0/1/2')
    parser.add_argument('--output', type=Path, help='diagnostic output directory; defaults to checkpoint run')
    parser.add_argument('--warmup', type=float, default=0., help='initial zero-command seconds')
    parser.add_argument('--walk-seconds', type=float, help='switch back to zero after this many walking seconds')
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    if args.seconds <= 0 or args.warmup < 0 or (args.walk_seconds is not None and args.walk_seconds <= 0):
        parser.error('seconds must be positive')
    meta = json.loads((args.checkpoint.parent.parent / 'metadata.json').read_text())
    if meta['source_hashes'] != source_hashes():
        raise ValueError('Checkpoint source hashes do not match the current model')
    policy = jax.jit(checkpoint.load_policy(str(args.checkpoint), deterministic=True))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    names = [model.actuator(i).name for i in range(model.nu)]
    if names != meta['actuator_names']:
        raise ValueError('Actuator ordering mismatch')
    joint_ids = model.actuator_trnid[:, 0]
    qa, va = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    target = model.key('home').ctrl.copy()
    default = target.copy()
    scale = np.array(meta['action_scale'])
    history = np.zeros((3, model.nu))
    action_history = history.copy()
    command = np.array([args.forward if args.warmup == 0 else 0., 0., 0., 0., 0., 0., 0.])
    contact = np.zeros(2)
    feet = [model.site_bodyid[model.site(side + '_foot').id] for side in ('left', 'right')]
    floor = model.geom('floor').id
    foot_vertices = []
    for body in feet:
        vertices = []
        for gid in np.where((model.geom_bodyid == body) & (model.geom_contype != 0))[0]:
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
    self_contacts, nonfoot_contacts = {}, {}
    positions, clearances, forces, tilts, slip_speeds, commands, support_matches = [], [], [], [], [], [], []
    stop = 'duration'
    ref = json.loads((ROOT / 'data/reference.json').read_text())
    period_steps = round(ref['period'] / .02)
    if not any(np.allclose([args.forward, 0., 0.], c, atol=1e-8) for c in meta['commands']):
        raise ValueError('Command is outside the trained reference domain')
    sites = [model.site(side + '_foot').id for side in ('left', 'right')]
    total_weight = float(model.body_subtreemass[1] * abs(model.opt.gravity[2]))
    target_contacts = np.array(ref['reward_reference']['contacts'], dtype=bool) if 'reward_reference' in ref else None
    # Upstream reset creates MJX data without forward: first sensors and phase
    # are zero. Later observations are captured before last_act is advanced.
    phase = np.zeros(2)
    obs = observation(np.zeros(3), np.zeros(3), command, data.qpos[qa], data.qvel[va],
                      history, target, contact, phase, default)
    initial_position = data.qpos[:3].copy()
    substeps = round(.02 / model.opt.timestep)
    integrated_steps = 0
    for tick in range(round(args.seconds / .02)):
        assert obs.shape == (113,)
        action, _ = policy({'state': jp.array(obs)}, jax.random.PRNGKey(0))
        action = np.asarray(action)
        assert np.isfinite(action).all()
        action_history[1:] = action_history[:-1].copy()
        action_history[0] = action
        requested = np.clip(default + action_history[args.action_delay] * scale,
                            model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
        target = np.clip(requested, target - 5.24 * .02, target + 5.24 * .02)
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
                    name = ' / '.join(sorted(model.body(int(model.geom_bodyid[g])).name for g in pair))
                    self_contacts[name] = max(self_contacts.get(name, 0.), magnitude)
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
        clearances.append(height)
        forces.append(force)
        tilts.append(tilt)
        slip_speeds.append(slip)
        commands.append(command[0])
        if target_contacts is not None:
            desired = target_contacts[(tick + 1) % period_steps] if command[0] > .01 else np.ones(2, dtype=bool)
            support_matches.append((force > .5) == desired)
        phase_angle = 2 * np.pi * ((tick + 1) % period_steps) / period_steps
        phase = np.array([np.cos(phase_angle), np.sin(phase_angle)])
        next_time = (tick + 1) * .02
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
    airborne = ((np.array(clearances) >= .01) & (np.array(forces) < .5) &
                (np.array(forces)[:, ::-1] > .5 * total_weight))
    longest, run = np.zeros(2), np.zeros(2)
    for frame in airborne:
        run = np.where(frame, run + .02, 0.)
        longest = np.maximum(longest, run)
    walking = np.array(commands) > .01
    forward_speed = np.diff(np.r_[initial_position[0], np.array(positions)[:, 0]]) / .02
    report = dict(checkpoint=str(args.checkpoint), source_hashes=source_hashes(),
                  native_scene='full, unchanged CAD-derived collision model',
                  elapsed_sim_seconds=integrated_steps * model.opt.timestep, stop=stop,
                  command=command.tolist(), fixed_action_delay_steps=args.action_delay,
                  command_profile=dict(forward_m_s=args.forward, warmup_s=args.warmup, walk_seconds=args.walk_seconds),
                  reward_specification=meta.get('reward_specification', {'mode': 'original_legacy'}),
                  warnings=data.warning.number.tolist(),
                  max_tilt_deg=max(tilts), peak_torque_nm=dict(zip(names, peak.tolist())),
                  final_tilt_deg=tilts[-1],
                  final_home_pose_error_rad=float(np.max(abs(np.array(positions[-1])[qa] - model.key('home').qpos[qa]))),
                  saturation_duration_s=dict(zip(names, saturation.tolist())),
                  min_joint_limit_margin_rad=float(min_limit_margin.min()),
                  self_contact_peak_force_n=self_contacts, nonfoot_contact_peak_force_n=nonfoot_contacts,
                  max_whole_foot_clearance_m=np.max(clearances, axis=0).tolist(),
                  longest_10mm_unloaded_sample_run_s=longest.tolist(),
                  total_10mm_unloaded_sample_time_s=(airborne.sum(axis=0) * .02).tolist(),
                  support_force_threshold_n=.5 * total_weight,
                  loaded_contact_slip_distance_m=(np.sum(slip_speeds, axis=0) * .02).tolist(),
                  mean_support_contact_match=float(np.mean(support_matches)) if support_matches else None,
                  mean_forward_velocity_m_s=float((positions[-1][0] - initial_position[0]) / (integrated_steps * model.opt.timestep)),
                  mean_walking_forward_velocity_m_s=float(np.mean(forward_speed[walking])) if walking.any() else None,
                  walking_support_contact_match=float(np.mean(np.array(support_matches)[walking]))
                      if support_matches and walking.any() else None,
                  displacement_m=(positions[-1][:3] - initial_position).tolist(),
                  acceptance='diagnostic only; continuous self clearance and full landing/recovery gates pending')
    out = args.output or args.checkpoint.parent.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez(out / 'evaluation.npz', qpos=positions, foot_clearance=clearances,
             foot_force=forces, tilt_deg=tilts, contact_slip_speed=slip_speeds, forward_command=commands)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
