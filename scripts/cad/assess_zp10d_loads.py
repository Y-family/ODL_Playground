"""Ground-supported load diagnostics for the current robot; use the hard-limit launcher."""
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import sys

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'cad/local_fix'
MODEL_DIR = ROOT / 'playground/open_duck_mini_zp10d/xmls'
REPORT = OUT / 'CURRENT_LOAD_ASSESSMENT.json'
CACHE = Path('/tmp/open-duck-load-assessment')
SID = dict(zip(['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
                'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
                'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee',
                'right_ankle', 'left_antenna', 'right_antenna'],
               ['S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                'S01', 'S02', 'S03', 'S04', 'S05', 'A_left', 'A_right']))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(value):
    x = np.clip(value, 0, 1)
    return x ** 3 * (10 + x * (-15 + 6 * x))


def interpolate(t, waypoints):
    for (t0, q0), (t1, q1) in zip(waypoints, waypoints[1:]):
        if t <= t1:
            return q0 + smooth((t - t0) / (t1 - t0)) * (q1 - q0)
    return waypoints[-1][1].copy()


def pose_matrix(data, body):
    return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()


def leg_targets(model, settled, home_ctrl, *, roll=0., y=0., z=0., x=0.,
                left_lift=0., right_lift=0., left_forward=0., right_forward=0.):
    """IK produces motor targets; the simulation base is never prescribed or supported."""
    reference = mujoco.MjData(model)
    reference.qpos[:] = settled.qpos
    mujoco.mj_kinematics(model, reference)
    feet = {side: pose_matrix(reference, model.body(body).id)
            for side, body in [('left', 'foot_assembly'), ('right', 'foot_assembly_2')]}
    qpos = settled.qpos.copy()
    qpos[:3] += [x, y, z]
    rot = Rotation.from_rotvec([roll, 0, 0]) * Rotation.from_quat(qpos[[4, 5, 6, 3]])
    q = rot.as_quat(); qpos[3:7] = q[[3, 0, 1, 2]]
    result = home_ctrl.copy(); checks = []
    for side, lift, forward in [('left', left_lift, left_forward),
                                ('right', right_lift, right_forward)]:
        names = [f'{side}_{suffix}' for suffix in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')]
        aids = np.array([model.actuator(name).id for name in names])
        joints = model.actuator_trnid[aids, 0]; adr = model.jnt_qposadr[joints]
        lo, hi = model.jnt_range[joints].T
        start = settled.qpos[adr].copy()
        target_pos, target_rot = feet[side]
        target_pos = target_pos + [forward, 0, lift]
        body = model.body('foot_assembly' if side == 'left' else 'foot_assembly_2').id

        def residual(angles):
            reference.qpos[:] = qpos; reference.qpos[adr] = angles
            mujoco.mj_kinematics(model, reference)
            position, rotation = pose_matrix(reference, body)
            return np.r_[position - target_pos,
                         .06 * Rotation.from_matrix(target_rot.T @ rotation).as_rotvec()]

        solution = least_squares(residual, np.clip(start, lo + .001, hi - .001),
                                 bounds=(lo + .001, hi - .001), max_nfev=150,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(solution.x)
        result[aids] += solution.x - start
        position_error = float(np.linalg.norm(error[:3]) * 1000)
        angle_error = float(np.linalg.norm(error[3:]) / .06 * 180 / np.pi)
        checks.append({'side': side, 'position_error_mm': position_error,
                       'orientation_error_deg': angle_error,
                       'within_limits': bool(np.all((result[aids] >= lo) & (result[aids] <= hi))),
                       'solved': bool(solution.success),
                       'pose_tolerance_passed': position_error <= .5 and angle_error <= 1.,
                       'target_lift_mm': lift * 1000, 'target_forward_mm': forward * 1000})
    return np.clip(result, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]), checks


def trajectories(model, settled, home):
    checks = []

    def target(label, **kwargs):
        q, rows = leg_targets(model, settled, home, **kwargs)
        checks.append({'waypoint': label, 'request': kwargs, 'legs': rows})
        return q

    crouch = target('shallow_crouch', z=-.005)
    # The five-axis legs cannot hold both sole orientations while independently
    # translating the pelvis 20 mm sideways. Use a feasible roll target instead.
    lean = math.radians(16)
    right_lean = target('weight_to_right', roll=lean)
    left_lean = target('weight_to_left', roll=-lean)
    left_up = target('left_foot_up', roll=lean, left_lift=.008)
    right_up = target('right_foot_up', roll=-lean, right_lift=.008)
    left_adv = target('left_foot_forward_up', roll=lean,
                      left_lift=.008, left_forward=.010)
    left_down = target('left_foot_forward_down', roll=lean, left_forward=.010)
    middle = target('middle_after_left_step', x=.005, left_forward=.010)
    left_lean2 = target('weight_to_left_after_step', roll=-lean,
                        x=.005, left_forward=.010)
    right_up2 = target('right_foot_up_after_step', roll=-lean,
                       x=.005, left_forward=.010, right_lift=.008)
    right_adv = target('right_foot_forward_up', roll=-lean,
                       x=.005, left_forward=.010, right_lift=.008, right_forward=.010)
    right_down = target('right_foot_forward_down', roll=-lean,
                        x=.005, left_forward=.010, right_forward=.010)
    finish = target('middle_after_both_steps', x=.010, left_forward=.010, right_forward=.010)
    head = home.copy()
    for name, angle in [('neck_pitch', 5), ('head_pitch', -5), ('head_yaw', 30), ('head_roll', 5)]:
        head[model.actuator(name).id] += math.radians(angle)
    antenna = home.copy(); antenna_back = home.copy()
    for name in ('left_antenna', 'right_antenna'):
        antenna[model.actuator(name).id] = math.radians(20)
        antenna_back[model.actuator(name).id] = math.radians(-20)
    return [
        ('standing', 6., [(0., home), (6., home)]),
        ('shallow_crouch', 8., [(0., home), (2., crouch), (4., crouch), (6., home), (8., home)]),
        ('left_lift_attempt', 12., [(0., home), (2.5, right_lean), (4.5, left_up),
                                   (6.5, left_up), (8.5, right_lean), (11., home), (12., home)]),
        ('right_lift_attempt', 12., [(0., home), (2.5, left_lean), (4.5, right_up),
                                    (6.5, right_up), (8.5, left_lean), (11., home), (12., home)]),
        ('slow_step_attempt', 19., [(0., home), (2.5, right_lean), (4., left_up),
                                  (5.5, left_adv), (7., left_down), (9., middle),
                                  (11.5, left_lean2), (13., right_up2), (14.5, right_adv),
                                  (16., right_down), (18., finish), (19., finish)]),
        ('head_sweep', 8., [(0., home), (2., head), (4., head), (6., home), (8., home)]),
        ('antenna_sweep', 8., [(0., home), (2., antenna), (4., antenna_back), (6., home), (8., home)]),
    ], checks


def ground_forces(model, data, floor, foot_bodies, *, with_moments=False):
    forces = np.zeros((2, 3)); total = np.zeros(3); nonfoot = []
    moments = np.zeros((2, 3))
    for index, contact in enumerate(data.contact):
        if floor not in contact.geom:
            continue
        wrench = np.zeros(6); mujoco.mj_contactForce(model, data, index, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        moment = contact.frame.reshape(3, 3).T @ wrench[3:]
        if contact.geom[1] == floor:
            force *= -1
            moment *= -1
        geom = int(contact.geom[1] if contact.geom[0] == floor else contact.geom[0])
        body = int(model.geom_bodyid[geom]); total += force
        if body in foot_bodies:
            side = foot_bodies.index(body)
            forces[side] += force
            moments[side] += np.cross(contact.pos, force) + moment
        elif np.linalg.norm(force) > .05:
            nonfoot.append(model.geom(geom).name)
    return (forces, total, nonfoot, moments) if with_moments else (forces, total, nonfoot)


def joint_wrenches(model, data, joints):
    """Parent-on-child wrench in world axes, relocated from root COM to the joint."""
    mujoco.mj_rnePostConstraint(model, data)
    bodies = model.jnt_bodyid[joints]; origin = data.subtree_com[model.body_rootid[bodies]]
    forces = data.cfrc_int[bodies, 3:].copy()
    moments = data.cfrc_int[bodies, :3] + np.cross(origin - data.xanchor[joints], forces)
    return np.c_[moments, forces]


def consistency_checks(model, settled, joints):
    mujoco.mj_forward(model, settled)
    wrench = joint_wrenches(model, settled, joints)
    # Internal hinge moment must equal actuator + passive + non-contact joint
    # constraints minus the explicitly modeled rotor/gear armature inertia.
    noncontact = settled.efc_force.copy()
    noncontact[settled.efc_type >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)] = 0
    projected = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, settled, projected, noncontact)
    dofs = model.jnt_dofadr[joints]
    expected = (settled.qfrc_actuator + settled.qfrc_passive + projected
                - model.dof_armature * settled.qacc)[dofs]
    # The finite contact solver leaves a dynamics-equation residual. Compute it
    # independently instead of mistaking it for a wrench frame/sign error.
    inertia_force = np.zeros(model.nv)
    mujoco.mj_mulM(model, settled, inertia_force, settled.qacc)
    equation_residual = (inertia_force + settled.qfrc_bias - settled.qfrc_passive
                         - settled.qfrc_actuator - settled.qfrc_constraint - settled.qfrc_applied)
    measured = np.sum(wrench[:, :3] * settled.xaxis[joints], axis=1)
    raw_error = float(np.max(np.abs(expected - measured)))
    error = float(np.max(np.abs(expected + equation_residual[dofs] - measured)))
    assert error < 1e-7, ('joint wrench reference/sign check', error)
    return {'joint_moment_balance_max_error_nm': error, 'passed': True,
            'solver_hinge_equation_residual_max_nm': float(np.max(np.abs(equation_residual[dofs]))),
            'raw_motor_to_reaction_balance_difference_nm': raw_error,
            'wrench_order': 'Mx My Mz Fx Fy Fz', 'wrench_origin': 'current joint anchor',
            'wrench_axes': 'world', 'wrench_direction': 'parent acting on child'}


def sole_vertices(model, feet):
    """Actual tread mesh vertices in each foot body frame, not foot-site heights."""
    result = []
    for body, index in zip(feet, (84, 42)):
        geom = model.geom(f'visual_{index:03d}').id
        assert model.geom_bodyid[geom] == body
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rotation = np.zeros(9); mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        result.append(model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T
                      + model.geom_pos[geom])
    return result


def consecutive_seconds(mask, dt=.02):
    longest = run = 0
    for value in mask:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return longest * dt


def motion_outcome(name, frames):
    forces = np.array([f['foot_forces_world_n'] for f in frames])[:, :, 2]
    clearance = np.array([f['sole_min_height_m'] for f in frames]) * 1000
    positions = np.array([f['foot_body_positions_world_m'] for f in frames])
    # Clearance and unloading must occur together for at least 0.2 s. A rising
    # foot site alone can simply mean the sole is rotating on its edge.
    airborne = (forces < .5) & (clearance > 2.)
    durations = [consecutive_seconds(airborne[:, side]) for side in (0, 1)]
    advance = (positions[:, :, 0] - positions[0, :, 0]) * 1000
    airborne_advance = [float(max(0., advance[airborne[:, i], i].max()))
                        if airborne[:, i].any() else 0. for i in (0, 1)]
    min_load = forces.min(axis=0).tolist()
    achieved = None
    if name == 'left_lift_attempt':
        achieved = durations[0] >= .2
    elif name == 'right_lift_attempt':
        achieved = durations[1] >= .2
    elif name == 'slow_step_attempt':
        left_up = np.flatnonzero(airborne[:, 0]); right_up = np.flatnonzero(airborne[:, 1])
        # Check both swing phases, their order, and re-contact after each swing.
        phases = bool(len(left_up) and len(right_up) and left_up[0] < right_up[0]
                      and np.any(forces[left_up[-1] + 1:, 0] > 1.)
                      and np.any(forces[right_up[-1] + 1:, 1] > 1.))
        achieved = min(durations) >= .2 and min(airborne_advance) >= 5. and phases
    return {'task_achieved': achieved, 'minimum_foot_vertical_force_n': min_load,
            'maximum_sole_min_clearance_mm': clearance.max(axis=0).tolist(),
            'longest_airborne_seconds': durations,
            'maximum_forward_displacement_while_airborne_mm': airborne_advance,
            'criterion': 'Force <0.5 N AND entire tread mesh >2 mm above floor for >=0.2 s; step also needs >=5 mm forward swing, ordered phases and re-contact.'}


def simulate(model, settled, name, duration, waypoints, actor_names, joints, *, push=None):
    data = mujoco.MjData(model); data.qpos[:] = settled.qpos; data.qvel[:] = settled.qvel
    data.ctrl[:] = waypoints[0][1]; mujoco.mj_forward(model, data)
    base = model.body('base').id; floor = model.geom('floor').id
    feet = [model.body('foot_assembly').id, model.body('foot_assembly_2').id]
    adr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    dt = model.opt.timestep; limit = model.actuator_forcerange[:, 1]
    peak = np.zeros(model.nu); energy = np.zeros(model.nu); square = np.zeros(model.nu)
    saturation = np.zeros(model.nu); speed = np.zeros(model.nu); tracking = np.zeros(model.nu)
    saturated_run = np.zeros(model.nu); longest_saturated_run = np.zeros(model.nu)
    joint_excursion = np.zeros(model.nu); self_contacts = {}
    force_peak = np.zeros((2, 3)); wrench_peak = np.zeros((model.nu, 6)); load_cases = {}
    reaction_peaks = np.zeros((model.nu, 2)); reaction_cases = {}
    frames = []; min_z = float(data.xpos[base, 2]); max_tilt = 0.; peak_ground = 0.
    warnings_start = data.warning.number.copy(); stop_reason = None; max_penetration = 0.
    initial = data.qpos[:3].copy(); initial_foot_z = data.site_xpos[[model.site('left_foot').id, model.site('right_foot').id], 2].copy()
    tread_vertices = sole_vertices(model, feet)
    wrench_error = 0.; solver_residual = 0.

    def load_state(actuator, time, wrench, forces):
        child = int(model.jnt_bodyid[joints[actuator]]); parent = int(model.body_parentid[child])
        return {'time_s': time, 'actuator_torque_nm': float(data.actuator_force[actuator]),
                'joint_wrench_world': wrench.tolist(),
                'force_norm_n': float(np.linalg.norm(wrench[3:])),
                'moment_norm_nm': float(np.linalg.norm(wrench[:3])),
                'joint_origin_world_m': data.xanchor[joints[actuator]].tolist(),
                'joint_axis_world': data.xaxis[joints[actuator]].tolist(), 'qpos': data.qpos.tolist(),
                'foot_forces_world_n': forces.tolist(),
                'child_body': model.body(child).name, 'parent_body': model.body(parent).name,
                'child_body_position_world_m': data.xpos[child].tolist(),
                'child_body_rotation_world': data.xmat[child].reshape(3, 3).tolist(),
                'parent_body_position_world_m': data.xpos[parent].tolist(),
                'parent_body_rotation_world': data.xmat[parent].reshape(3, 3).tolist()}
    for step in range(round(duration / dt)):
        time = step * dt; data.ctrl[:] = interpolate(time, waypoints)
        if push is not None:
            # A finite disturbance on the actual trunk COM, never base support.
            data.xfrc_applied[:] = 0.
            if push['start_s'] <= time < push['start_s'] + push['duration_s']:
                data.xfrc_applied[model.body('trunk_assembly').id, :3] = push['force_world_n']
        mujoco.mj_step(model, data)
        # mj_step force arrays refer to the pre-integration pose. mj_forward
        # aligns pose, force and contact records for independent downstream use.
        mujoco.mj_forward(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), name
        wrenches = joint_wrenches(model, data, joints)
        forces, ground, nonfoot, ground_moments = ground_forces(model, data, floor, feet, with_moments=True)
        torque = data.actuator_force.copy(); velocity = data.qvel[dofs]
        absolute = np.abs(torque)
        changed = np.flatnonzero(absolute > peak)
        for actuator in changed:
            load_cases[actor_names[actuator]] = load_state(actuator, time + dt, wrenches[actuator], forces)
        norms = np.c_[np.linalg.norm(wrenches[:, 3:], axis=1), np.linalg.norm(wrenches[:, :3], axis=1)]
        for actuator, criterion in np.argwhere(norms > reaction_peaks):
            reaction_cases.setdefault(actor_names[actuator], {})[['force', 'moment'][criterion]] = load_state(
                actuator, time + dt, wrenches[actuator], forces)
        reaction_peaks = np.maximum(reaction_peaks, norms)
        peak = np.maximum(peak, absolute); square += torque ** 2 * dt
        energy += np.abs(torque * velocity) * dt; speed = np.maximum(speed, np.abs(velocity))
        saturation += absolute >= .99 * limit
        saturated_run = np.where(absolute >= .99 * limit, saturated_run + dt, 0.)
        longest_saturated_run = np.maximum(longest_saturated_run, saturated_run)
        joint_excursion = np.maximum(joint_excursion, np.maximum(
            model.jnt_range[joints, 0] - data.qpos[adr],
            data.qpos[adr] - model.jnt_range[joints, 1]))
        for index, contact in enumerate(data.contact):
            if floor in contact.geom:
                continue
            wrench = np.zeros(6); mujoco.mj_contactForce(model, data, index, wrench)
            pair = '/'.join(sorted(model.body(int(model.geom_bodyid[g])).name for g in contact.geom))
            row = self_contacts.setdefault(pair, {'peak_contact_force_n': 0., 'max_proxy_penetration_mm': 0.})
            row['peak_contact_force_n'] = max(row['peak_contact_force_n'], float(np.linalg.norm(wrench[:3])))
            row['max_proxy_penetration_mm'] = max(row['max_proxy_penetration_mm'], float(max(0., -contact.dist) * 1000))
        tracking = np.maximum(tracking, np.abs(data.qpos[adr] - data.ctrl))
        force_peak = np.maximum(force_peak, np.abs(forces)); wrench_peak = np.maximum(wrench_peak, np.abs(wrenches))
        tilt = math.degrees(math.acos(np.clip(data.xmat[base].reshape(3, 3)[2, 2], -1, 1)))
        max_tilt = max(max_tilt, tilt); min_z = min(min_z, float(data.xpos[base, 2]))
        peak_ground = max(peak_ground, float(ground[2]))
        if data.ncon:
            max_penetration = max(max_penetration, float(max(0, -data.contact.dist.min())) * 1000)
        if step % 10 == 0:
            mujoco.mj_subtreeVel(model, data)
            balance = consistency_checks(model, data, joints)
            wrench_error = max(wrench_error, balance['joint_moment_balance_max_error_nm'])
            solver_residual = max(solver_residual, balance['solver_hinge_equation_residual_max_nm'])
            sole_z = [float((vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body])[:, 2].min())
                      for body, vertices in zip(feet, tread_vertices)]
            knee_joints = np.array([model.joint(f'{side}_knee').id for side in ('left', 'right')])
            knee_ground_moments = ground_moments - np.cross(data.xanchor[knee_joints], forces)
            knee_axis_ground_moments = np.sum(knee_ground_moments * data.xaxis[knee_joints], axis=1)
            knee_ground_levers_mm = np.abs(knee_axis_ground_moments) / np.maximum(np.abs(forces[:, 2]), 1e-9) * 1000
            frames.append({'time_s': time + dt, 'qpos': data.qpos.tolist(), 'ctrl': data.ctrl.tolist(),
                'torque_nm': torque.tolist(), 'velocity_rad_s': velocity.tolist(),
                'joint_wrench_world': wrenches.tolist(), 'foot_forces_world_n': forces.tolist(),
                'ground_force_world_n': ground.tolist(), 'base_tilt_deg': tilt,
                'com_world_m': data.subtree_com[base].tolist(),
                'com_velocity_m_s': data.subtree_linvel[base].tolist(),
                'sole_min_height_m': sole_z,
                'foot_body_positions_world_m': data.xpos[feet].tolist(),
                'knee_ground_moment_nm': knee_axis_ground_moments.tolist(),
                'knee_effective_ground_lever_mm': knee_ground_levers_mm.tolist(),
                'foot_ground_moments_about_world_origin_nm': ground_moments.tolist(),
                'foot_site_heights_m': data.site_xpos[[model.site('left_foot').id, model.site('right_foot').id], 2].tolist()})
        if tilt > 35 or data.xpos[base, 2] < .085 or nonfoot:
            stop_reason = 'excessive_tilt' if tilt > 35 else 'base_too_low' if data.xpos[base, 2] < .085 else 'nonfoot_ground_contact'
            break
    seconds = (step + 1) * dt; warnings = (data.warning.number - warnings_start).tolist()
    assert not any(warnings), (name, warnings)
    foot_heights = np.array([f['foot_site_heights_m'] for f in frames])
    foot_loads = np.array([f['foot_forces_world_n'] for f in frames])[:, :, 2]
    lift = np.max(foot_heights - initial_foot_z, axis=0) * 1000
    unloaded = np.sum(foot_loads < .5, axis=0) * .02
    row = {'name': name, 'requested_seconds': duration, 'simulated_seconds': seconds,
        'completed': stop_reason is None, 'stop_reason': stop_reason, 'warnings': warnings,
        'max_base_tilt_deg': max_tilt, 'lowest_base_z_m': min_z,
        'base_translation_m': (data.qpos[:3] - initial).tolist(),
        'max_tracking_error_deg': np.degrees(tracking).tolist(),
        'peak_motor_torque_nm': peak.tolist(), 'rms_motor_torque_nm': np.sqrt(square / seconds).tolist(),
        'peak_fraction_of_stall_torque': (peak / limit).tolist(),
        'saturation_fraction': (saturation / (step + 1)).tolist(),
        'saturation_duration_s': (saturation * dt).tolist(),
        'longest_saturation_duration_s': longest_saturated_run.tolist(),
        'joint_limit_excursion_deg': np.degrees(joint_excursion).tolist(),
        'nonfloor_contact_pairs': self_contacts,
        'peak_speed_deg_s': np.degrees(speed).tolist(), 'absolute_mechanical_work_j': energy.tolist(),
        'peak_foot_force_components_n': force_peak.tolist(), 'peak_total_vertical_ground_force_n': peak_ground,
        'peak_joint_wrench_components_world': wrench_peak.tolist(), 'peak_motor_load_cases': load_cases,
        'peak_reaction_norms': reaction_peaks.tolist(), 'peak_reaction_load_cases': reaction_cases,
        'joint_moment_balance_max_error_nm': wrench_error,
        'solver_hinge_equation_residual_max_nm': solver_residual,
        'max_proxy_penetration_mm': max_penetration, 'foot_site_lift_mm': lift.tolist(),
        'foot_unloaded_seconds': unloaded.tolist(), 'frames': frames,
        'motion_outcome': motion_outcome(name, frames),
        'external_push': push,
        'scope': 'Prescribed motor targets, free base and actual soft ground contact; an attempted step is not a trained gait.'}
    print(name, 'duration', round(seconds, 2), 'tilt', round(max_tilt, 2), 'peak Nm', round(float(peak.max()), 3),
          'lift mm', np.round(lift, 2).tolist(), 'stop', stop_reason, flush=True)
    return row


def structural_loads(report, mass):
    """Preserve simultaneous assembly loads; no unverified fastener load split."""
    by_body = {}
    printed = []
    shell_labels = {'left_cache', 'right_cache', 'body_front', 'body_middle_bottom',
                    'body_back', 'battery_pack_lid', 'body_middle_top', 'head', 'head_bot_sheet'}
    for part in mass['parts']:
        if part['mass_provenance']['method'] not in {
                'legacy_effective_density', 'provisional_PLA_density_for_missing_legacy_mass'}:
            continue
        item = {'index': part['index'], 'path': part['path'], 'label': part['label'],
                'body': part['body'], 'mesh': part['mesh'], 'mesh_sha256': part['mesh_sha256'],
                'volume_mm3': part['volume_mm3'], 'material': 'PETG',
                'infill_percent': 10 if part['label'] in shell_labels else 30,
                'classification': 'shell' if part['label'] in shell_labels else 'other',
                'classification_assumed': True}
        printed.append(item); by_body.setdefault(part['body'], []).append(item['index'])
    interface_source = OUT / 'FINAL_ASSEMBLY_REVIEW.json'
    interface_evidence = json.loads(interface_source.read_text())
    assert interface_evidence['target_sha256'] == report['source_step_sha256']
    interfaces = {row['servo']: row['patterns'] for row in interface_evidence['interface_validation']['joints']}
    cases = []; transform_error = 0.
    for scenario in report['scenarios']:
        for name in report['actuator_order']:
            for criterion, state in [('motor_torque', scenario['peak_motor_load_cases'][name])] + list(
                    scenario['peak_reaction_load_cases'][name].items()):
                row = {'scenario': scenario['name'], 'joint': name, 'servo_id': SID[name],
                       'criterion': criterion, 'simultaneous_state': state, 'attachments': {},
                       'existing_fastener_axis_reference': interfaces.get(SID[name], []),
                       'boundary_condition_status': 'Bolt seat/contact surface selection and load sharing pending; cylinder axes alone are insufficient.'}
                world = np.array(state['joint_wrench_world'])
                for side, sign in [('child', 1.), ('parent', -1.)]:
                    body = state[f'{side}_body']
                    frame = np.array(mass['frames'][body])
                    actual_rotation = np.array(state[f'{side}_body_rotation_world'])
                    to_cad = frame[:3, :3] @ actual_rotation.T
                    force = sign * to_cad @ world[3:]
                    moment = sign * to_cad @ world[:3]
                    origin = frame[:3, 3] + to_cad @ (
                        np.array(state['joint_origin_world_m']) - state[f'{side}_body_position_world_m'])
                    sid = SID[name]
                    axis_key = {'A_left': 'A_plus_y', 'A_right': 'A_minus_y'}.get(sid, sid)
                    error = float(np.linalg.norm(origin * 1000 - mass['axes'][axis_key][0]))
                    transform_error = max(transform_error, error)
                    assert error < 1e-5, ('CAD load-frame anchor', name, side, error)
                    assert abs(np.linalg.norm(force) - np.linalg.norm(world[3:])) < 1e-8
                    assert abs(np.linalg.norm(moment) - np.linalg.norm(world[:3])) < 1e-8
                    row['attachments'][side] = {'body': body, 'candidate_printed_part_indices': by_body.get(body, []),
                        'force_n_cad_axes': force.tolist(), 'moment_nmm_cad_axes': (moment * 1000).tolist(),
                        'load_origin_mm_cad': (origin * 1000).tolist(),
                        'direction': f'load on {side} side at the joint interface',
                        'load_sharing': 'Not assigned to individual brackets or screws.'}
                cases.append(row)
    return {'date': report['date'], 'source_step_sha256': report['source_step_sha256'],
            'mjcf_sha256': report['mjcf_sha256'], 'print_spec': report['print_spec'],
            'status': 'assembly_loads_ready_local_FEA_pending', 'units': 'N, N.mm, mm',
            'printed_parts': printed, 'load_cases': cases,
            'fastener_reference': {'file': str(interface_source.relative_to(ROOT)),
                'sha256': digest(interface_source), 'source_step_sha256': interface_evidence['target_sha256'],
                'scope': 'Previously checked cylinder axes on the identical STEP; not a new seat/thread/FEA validation.'},
            'transform_checks': {'max_CAD_joint_origin_error_mm': transform_error, 'passed': True},
            'limitations': ['Cases are same-time wrench vectors, not a splice of component maxima.',
                'Cases cover these attempted motions only; successful single support, impacts and trained walking are not covered.',
                'Body-local loads are mapped back to the unchanged STEP frame. Define actual bolt seats, contact and load sharing before local FEA.',
                'No stress, displacement or safety factor is certified. There is no local FEA solve in this report.',
                'PETG supplier data, wall/skin counts, layer height, build directions and infill pattern are missing; no guessed isotropic material or infill strength scaling is used.',
                'Shell membership is a label-based assumption. Legacy foot_bottom_tpu/foot_bottom_pla names do not change the user-requested PETG material; floor friction/compliance needs measurement.']}


def motor_rows(report):
    rows = []
    steady = [f['torque_nm'] for f in report['scenarios'][0]['frames'] if f['time_s'] >= 4.]
    hold = np.mean(np.abs(steady), axis=0)
    for index, name in enumerate(report['actuator_order']):
        worst = max(report['scenarios'], key=lambda s: s['peak_motor_torque_nm'][index])
        rows.append({'servo': SID[name], 'joint': name,
            'standing_mean_abs_torque_nm': float(hold[index]),
            'peak_torque_nm': worst['peak_motor_torque_nm'][index], 'worst_scenario': worst['name'],
            'max_scenario_rms_torque_nm': max(s['rms_motor_torque_nm'][index] for s in report['scenarios']),
            'provisional_stall_torque_nm': report['provisional_stall_limits_nm'][index],
            'peak_stall_ratio': worst['peak_fraction_of_stall_torque'][index],
            'max_saturation_fraction': max(s['saturation_fraction'][index] for s in report['scenarios']),
            'max_saturation_duration_s': max(s['saturation_duration_s'][index] for s in report['scenarios']),
            'longest_saturation_duration_s': max(s['longest_saturation_duration_s'][index] for s in report['scenarios']),
            'peak_speed_deg_s': max(s['peak_speed_deg_s'][index] for s in report['scenarios']),
            'peak_joint_force_n': max(s['peak_reaction_norms'][index][0] for s in report['scenarios']),
            'peak_joint_moment_nm': max(s['peak_reaction_norms'][index][1] for s in report['scenarios'])})
    return rows


def write_outputs(report, mass, destination):
    rows = motor_rows(report)
    with (destination / 'CURRENT_LOAD_ASSESSMENT.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    structural = structural_loads(report, mass)
    (destination / 'CURRENT_STRUCTURAL_LOADS.json').write_text(json.dumps(structural, ensure_ascii=False, indent=2) + '\n')
    check = report['checks']
    lines = ['# 当前版本：地面支撑负载评估', '', f"日期：{report['date']}。来源 STEP SHA256：`{report['source_step_sha256']}`。", '',
        '打印要求：PETG；外壳填充 10%，其余打印件填充 30%。壁层数、顶底层数、层高、填充图案和打印方向尚缺。填充率未直接当作密度或强度系数。', '',
        f"当前估计质量 **{report['mass_kg']:.3f} kg**，重力 **{report['weight_n']:.3f} N**。沿用同步模型的估计质量，尚未用真实切片或称重更新 PETG 打印件质量。", '',
        '模型保留自由基座，无焊接支撑、无施加在基座上的外力；脚与地面实际接触。每个工况从独立的 3 秒自由站姿开始，采样步长 2 ms，轨迹/离地指标 50 Hz。', '',
        f"站立末 2 秒平均地面竖直支撑力 **{check['standing_mean_ground_force_world_n'][2]:.6f} N**；相对重量误差 **{check['standing_vertical_weight_balance_relative_error'] * 100:.5f}%**。", '',
        f"关节反力坐标/符号校验（计入独立计算的动力学方程残差）最大误差 **{max(s['joint_moment_balance_max_error_nm'] for s in report['scenarios']):.3g} N·m**。有限接触求解器的关节方程残差最大 **{max(s['solver_hinge_equation_residual_max_nm'] for s in report['scenarios']):.3g} N·m**，两者分别记录。所有工况数值警告为 0。", '',
        '## 工况与动作结果', '',
        '| 工况 | 时长 s | 最大倾角 ° | 电机峰值 N·m | 全程未失稳 | 目标抬脚/迈步达成 | 双脚最小竖直载荷 N |',
        '| --- | ---: | ---: | ---: | --- | --- | --- |']
    for s in report['scenarios']:
        outcome = s['motion_outcome']; achieved = outcome['task_achieved']
        lines.append(f"| {s['name']} | {s['simulated_seconds']:.1f} | {s['max_base_tilt_deg']:.2f} | {max(s['peak_motor_torque_nm']):.3f} | {'是' if s['completed'] else '否'} | {'—' if achieved is None else '是' if achieved else '否'} | {outcome['minimum_foot_vertical_force_n'][0]:.2f} / {outcome['minimum_foot_vertical_force_n'][1]:.2f} |")
    lines.extend(['', '动作成功判据：整张脚底网格离地超过 2 mm，且该脚竖直支撑力低于 0.5 N，连续至少 0.2 秒。迈步还要求双脚依次在离地时前移至少 5 mm，并落地重新承载。脚 site 抬高不作为离地证明。', '',
        '抬脚/慢步为诊断轨迹，不是已训练步态。其载荷不能覆盖未实现的单脚支撑、落地冲击、跌倒或跑动。', '',
        '诊断目标：下蹲5 mm；侧倾16°后抬脚8 mm、前移10 mm；头颈组合S11 +5°、S12 −5°、S13 +30°、S14 +5°；双天线前后±20°。这些是测试范围，不是全运动范围认证。', '',
        f"本轮轨迹 IK 最大位置残差 {max(r['position_error_mm'] for w in report['ik_waypoints'] for r in w['legs']):.3f} mm，最大朝向残差 {max(r['orientation_error_deg'] for w in report['ik_waypoints'] for r in w['legs']):.3f}°；每个目标的 0.5 mm / 1° 验收与限位状态记录在 JSON。", '',
        f"有非地面接触记录的工况数 {sum(bool(s['nonfloor_contact_pairs']) for s in report['scenarios'])}；最大物理关节越限 {max(max(s['joint_limit_excursion_deg']) for s in report['scenarios']):.6f}°。完成时长只表示未触发倾倒/非脚触地停止，动作达成和接触需分别判断。", '',
        '## 舵机载荷', '',
        '| 舵机 | 关节 | 站立保持 N·m | 峰值 N·m | 最大工况 RMS N·m | 堵转比例 | 最长99%饱和 s | 最重工况 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |'])
    for row in rows:
        lines.append(f"| {row['servo']} | {row['joint']} | {row['standing_mean_abs_torque_nm']:.3f} | {row['peak_torque_nm']:.3f} | {row['max_scenario_rms_torque_nm']:.3f} | {row['peak_stall_ratio'] * 100:.1f}% | {row['longest_saturation_duration_s']:.3f} | {row['worst_scenario']} |")
    lines.extend(['', 'ZP10D 暂按 0.980665 N·m 堵转上限，SG90 暂按 0.1 N·m。堵转比例不是连续运行安全系数；缺少实际电压下的连续扭矩、扭矩-速度曲线及热数据，因此不能据无饱和判定可长时运行。RMS 为各工况分别计算后取最大值。', '',
        '## 连接件强度分析交付', '',
        f"`CURRENT_STRUCTURAL_LOADS.json` 包含 {len(structural['load_cases'])} 个完整同时载荷状态：每个工况、每个轴分别取电机扭矩、关节合力、关节合力矩的峰值时刻，保留姿态及父/子两侧反向载荷。坐标已映回正式 STEP，单位 N、N·mm、mm；映射轴心最大误差 {structural['transform_checks']['max_CAD_joint_origin_error_mm']:.3g} mm。", '',
        '优先检查膝部/髋部连接及薄板、trunk_top 壳体固定区和 S01/S06 单轴输出固定面，其次颈部双板、head_pitch_to_yaw 和圆环接口。各部件的材料、填充率、现有网格与 hash 已列入载荷文件。', '',
        '**本轮完成刚体负载评估与结构载荷准备，尚未完成局部有限元应力、挠度、层间强度或螺孔安全系数验证。** 关节总载荷不能直接全部施加在某一片支架上；下一步需确认螺钉承压面、接触面及多连接件分载，结合打印方向/壁层和 PETG 实际材料数据建立局部模型。', '',
        '## 范围与复现', '',
        '正式 CAD、MuJoCo XML 和网格未修改。碰撞代理、同刚体/直接父子过滤及软接触仍有上一阶段记录的局限，不提供全机 0.2 mm 间隙认证。地面摩擦/接触参数沿用模型估计，须对 PETG 脚底实测校准。', '',
        f"资源：单进程地址空间硬限 4 GB，本轮峰值 RSS {report['resources']['peak_rss_bytes'] / 1e9:.3f} GB，观察到 {report['resources']['observed_process_threads']} 个进程线程，串行执行。", '',
        '```bash', '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/assess_zp10d_loads.py', '```', '',
        'JSON 保存完整轨迹与同一时刻载荷；CSV 汇总各轴；PNG 为科学绘图。打印设置缺失处保持未知，未替用户决定。', ''])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axis = axes[0, 0]; x = np.arange(len(rows))
    axis.bar(x, [r['peak_torque_nm'] for r in rows], label='Peak in tested scenarios')
    axis.plot(x, report['provisional_stall_limits_nm'], 'r--', label='Provisional stall ceiling')
    axis.set_xticks(x, [r['servo'] for r in rows], rotation=45)
    axis.set_ylabel('Motor torque (N m)'); axis.legend(fontsize=8)
    axis.set_title('Estimated-model loads; no continuous-duty rating')
    standing = report['scenarios'][0]['frames']; axis = axes[0, 1]
    axis.plot([f['time_s'] for f in standing], [f['ground_force_world_n'][2] for f in standing], label='Total vertical ground force')
    axis.axhline(report['weight_n'], color='r', linestyle='--', label='Estimated weight')
    axis.set_ylabel('N'); axis.set_xlabel('Time (s)'); axis.legend(fontsize=8)
    axis.set_ylim(0, report['weight_n'] * 1.2)
    axis.ticklabel_format(axis='y', style='plain', useOffset=False)
    attempt = next(s for s in report['scenarios'] if s['name'] == 'slow_step_attempt')
    times = [f['time_s'] for f in attempt['frames']]
    for i, side in enumerate(('Left', 'Right')):
        axes[1, 0].plot(times, [f['foot_forces_world_n'][i][2] for f in attempt['frames']], label=side)
        axes[1, 1].plot(times, [1000 * f['sole_min_height_m'][i] for f in attempt['frames']], label=side)
    axes[1, 0].axhline(.5, color='r', linestyle='--', label='Unload threshold')
    axes[1, 1].axhline(2., color='r', linestyle='--', label='Clearance threshold')
    axes[1, 0].set_ylabel('Vertical foot load (N)'); axes[1, 1].set_ylabel('Minimum tread clearance (mm)')
    for axis in axes[1]:
        axis.set_title('Slow-step attempt (success checked separately)')
        axis.set_xlabel('Time (s)'); axis.legend(fontsize=8)
    fig.savefig(destination / 'CURRENT_LOAD_ASSESSMENT.png', dpi=170); plt.close(fig)
    (destination / 'CURRENT_LOAD_ASSESSMENT.md').write_text('\n'.join(lines))


def publish_outputs(report, mass):
    write_outputs(report, mass, CACHE)
    # Serialize once before measuring the full-process peak, including the JSON
    # output buffer; keep all officially published artifacts on the same report.
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    report['resources']['peak_rss_bytes'] = max(report['resources']['peak_rss_bytes'],
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    del encoded
    (CACHE / 'candidate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    markdown = CACHE / 'CURRENT_LOAD_ASSESSMENT.md'
    lines = markdown.read_text().splitlines()
    peak_gb = max(report['resources']['peak_rss_bytes'] / 1e9,
                  report['resources'].get('simulation_process_peak_rss_mb_from_launcher', 0.) / 1000)
    lines = [f"资源：单进程地址空间硬限 4 GB，本轮已记录的峰值 RSS {peak_gb:.3f} GB，观察到 {report['resources']['observed_process_threads']} 个进程线程，串行执行。"
             if line.startswith('资源：') else line for line in lines]
    markdown.write_text('\n'.join(lines) + '\n')
    for filename in ['CURRENT_LOAD_ASSESSMENT.md', 'CURRENT_LOAD_ASSESSMENT.csv',
                     'CURRENT_LOAD_ASSESSMENT.png', 'CURRENT_STRUCTURAL_LOADS.json']:
        shutil.copyfile(CACHE / filename, OUT / filename)
    REPORT.write_text((CACHE / 'candidate.json').read_text())


def assess_stance():
    """Compare control poses only, preserving published CAD and home keyframes."""
    mass = json.loads((OUT / 'open_duck_mini_zp10d_mass_properties.json').read_text())
    scene = MODEL_DIR / 'scene_flat_terrain.xml'; xml = MODEL_DIR / 'open_duck_mini_zp10d.xml'
    assert digest(ROOT / mass['source_step']) == mass['source_step_sha256']
    assert digest(xml) == mass['mjcf_sha256'] and digest(scene) == mass['scene_sha256']
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert model.neq == 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    settled = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, settled, 0)
    home = model.key_ctrl[0].copy(); settled.ctrl[:] = home
    for _ in range(round(3 / model.opt.timestep)):
        mujoco.mj_step(model, settled)
    mujoco.mj_forward(model, settled)
    names = [model.actuator(i).name for i in range(model.nu)]; joints = model.actuator_trnid[:, 0]
    knees = [names.index('left_knee'), names.index('right_knee')]
    report = {'date': date.today().isoformat(), 'source_step_sha256': mass['source_step_sha256'],
        'mjcf_sha256': digest(xml), 'scene_sha256': digest(scene), 'script_sha256': digest(Path(__file__)),
        'actuator_order': names, 'servo_ids': [SID[n] for n in names], 'scenarios': [],
        'scope': 'Each pose: 5 s smooth transition plus 10 s hold; final 3 s compared. Free base and ground contact, no external support. Candidates are not deployed.',
        'mass_kg': float(model.body_mass.sum()),
        'limits': ['Estimated masses and servo dynamics, pending PETG slicing and hardware calibration.',
            'Standing comparison does not replace walking, single support, local strength or full CAD clearance checks.',
            'Effective lever is the measured axial knee moment from all foot-ground forces divided by vertical foot force; lower-leg gravity and motor friction are separate.',
            'Joint angles use model zero; never send these absolute angles directly to uncalibrated hardware.']}
    for z, x in [(0, 0), (10, 0), (20, 0), (25, 0), (25, -5), (30, 0)]:
        target, ik = leg_targets(model, settled, home, z=z / 1000, x=x / 1000)
        assert all(q['within_limits'] and q['pose_tolerance_passed'] for q in ik), (z, x, ik)
        name = f'raise_{z}mm_shiftX_{x}mm'
        row = simulate(model, settled, name, 15., [(0., home), (5., target), (15., target)], names, joints)
        assert row['completed'] and not any(row['warnings']) and max(row['saturation_fraction']) == 0
        steady = [f for f in row['frames'] if f['time_s'] >= 12.]
        angles = np.array([f['qpos'] for f in steady])[:, model.jnt_qposadr[joints]]
        torques = np.abs([f['torque_nm'] for f in steady])
        last = mujoco.MjData(model); last.qpos[:] = steady[-1]['qpos']; mujoco.mj_kinematics(model, last)
        knee_origins = last.xanchor[[model.joint(f'{side}_knee').id for side in ('left', 'right')]].copy()
        hip_origins = last.xanchor[[model.joint(f'{side}_hip_pitch').id for side in ('left', 'right')]].copy()
        ankle_origins = last.xanchor[[model.joint(f'{side}_ankle').id for side in ('left', 'right')]].copy()
        row.update({'target_raise_mm': z, 'target_shift_x_mm': x, 'ik_checks': ik,
            'target_ctrl_rad': target.tolist(), 'mean_joint_angles_deg': np.degrees(angles.mean(axis=0)).tolist(),
            'mean_abs_motor_torque_nm': torques.mean(axis=0).tolist(),
            'mean_effective_knee_ground_lever_mm': np.mean([f['knee_effective_ground_lever_mm'] for f in steady], axis=0).tolist(),
            'mean_foot_forces_world_n': np.mean([f['foot_forces_world_n'] for f in steady], axis=0).tolist(),
            'mean_com_world_m': np.mean([f['com_world_m'] for f in steady], axis=0).tolist(),
            'final_hip_knee_ankle_world_m': [hip_origins.tolist(), knee_origins.tolist(), ankle_origins.tolist()]})
        report['scenarios'].append(row)
    baseline = report['scenarios'][0]
    for row in report['scenarios']:
        row['knee_hold_torque_reduction_percent'] = [100 * (1 - row['mean_abs_motor_torque_nm'][i]
            / baseline['mean_abs_motor_torque_nm'][i]) for i in knees]
    recommended = next(r for r in report['scenarios'] if r['target_raise_mm'] == 25 and r['target_shift_x_mm'] == -5)
    report['recommended_trial'] = recommended['name']
    report['recommendation_reason'] = 'Moderate residual knee bend, lower bilateral knee load, low ankle load. The 30 mm pose is straighter and needs separate dynamic balance assessment.'
    for path, expected in [(ROOT / mass['source_step'], report['source_step_sha256']),
                           (xml, report['mjcf_sha256']), (scene, report['scene_sha256']),
                           (Path(__file__), report['script_sha256'])]:
        assert digest(path) == expected
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for row, color in [(baseline, '#d47a29'), (recommended, '#287ba8')]:
        points = np.array(row['final_hip_knee_ankle_world_m'])[:, 1, :] * 1000
        axes[0].plot(points[:, 0], points[:, 2], 'o-', color=color, linewidth=3, label=row['name'])
        axes[0].plot(*points[1, [0, 2]], 'o', color=color, markersize=10)
    axes[0].axhline(0, color='gray'); axes[0].set_aspect('equal')
    axes[0].set_xlabel('X forward (mm)'); axes[0].set_ylabel('Z up (mm)')
    axes[0].set_title('Right leg: hip / knee / ankle (actual pose)'); axes[0].legend(fontsize=8)
    x = np.arange(len(report['scenarios']))
    labels = [f"Z+{r['target_raise_mm']}\nX{r['target_shift_x_mm']:+}" for r in report['scenarios']]
    for side, i, offset in [('Left S09', knees[0], -.2), ('Right S04', knees[1], .2)]:
        axes[1].bar(x + offset, [r['mean_abs_motor_torque_nm'][i] for r in report['scenarios']], width=.4, label=side)
    axes[1].set_xticks(x, labels); axes[1].set_ylabel('Mean knee holding torque (N m)')
    axes[1].set_title('Last 3 seconds of each 15-second trial'); axes[1].legend()
    fig.savefig(CACHE / 'CURRENT_STANCE_ASSESSMENT.png', dpi=170); plt.close(fig)
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    lines = ['# 站姿减载试验', '', '仅比较控制姿态，正式 CAD、舵机安装、仿真 home 配置均未修改。当前估计质量 2.098 kg，尚未按 PETG 实际切片校准。', '',
        '每组从同一自由站姿开始，5 秒平滑过渡、10 秒保持；比较最后 3 秒平均绝对电机扭矩。六组共 90 秒，全部未失稳、数值警告为 0、执行器饱和为 0。', '',
        '| 目标躯干抬高 / X平移 mm | 左/右膝角度 ° | 左/右膝保持扭矩 N·m | 左/右有效地面力臂 mm | 左/右膝减载 | 最大倾角 ° |',
        '| --- | --- | --- | --- | --- | ---: |']
    for r in report['scenarios']:
        pair = lambda key: ' / '.join(f'{r[key][i]:.3f}' for i in knees)
        lever = ' / '.join(f'{v:.1f}' for v in r['mean_effective_knee_ground_lever_mm'])
        reductions = ' / '.join(f'{v:.1f}%' for v in r['knee_hold_torque_reduction_percent'])
        lines.append(f"| +{r['target_raise_mm']} / {r['target_shift_x_mm']:+} | {pair('mean_joint_angles_deg')} | {pair('mean_abs_motor_torque_nm')} | {lever} | {reductions} | {r['max_base_tilt_deg']:.2f} |")
    lines.extend(['', '建议试验姿态：双脚位置和脚底姿态作为 IK 参考，躯干上抬 25 mm、向 −X（后方）移动 5 mm；髋俯仰、双膝和双踝协调变化，使腿更直、脚底保持贴地。实际脚底会受软接触/舵机响应影响，仿真中保持双脚承载。抬高为目标变化，实际高度以仿真 qpos 为准。', '',
        '减小的力臂是脚底合地面反力对膝轴的有效力臂，未把整机重心到膝盖的距离直接当作单膝受力臂。总体重心仍应位于双脚支撑范围内；此处只证明这些短时保持试验稳定。', '',
        '比原站姿更直会抬高重心；保留约 44° 膝弯以留出动作空间，暂不采用最接近伸直的姿态。正式站姿与硬件配置尚未替换，移动/单脚支撑表现需重新测试；新站姿静态值不能与旧抬脚动作峰值直接比较。', '',
        'JSON 保存全轨迹、关节目标及实际反力。角度采用模型零位，不能直接作为未校准硬件的绝对指令。碰撞代理及父子接触过滤的局限继续适用，未作新 CAD 全机间隙认证。', '',
        f"资源：导入前 4 GB 地址空间硬限额，原生线程 1，串行；报告输出前峰值 RSS {report['peak_rss_bytes']/1e9:.3f} GB。", '',
        '```bash', '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/assess_zp10d_loads.py stance', '```', ''])
    (CACHE / 'CURRENT_STANCE_ASSESSMENT.md').write_text('\n'.join(lines))
    (CACHE / 'CURRENT_STANCE_ASSESSMENT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for suffix in ('json', 'md', 'png'):
        filename = f'CURRENT_STANCE_ASSESSMENT.{suffix}'
        shutil.copyfile(CACHE / filename, OUT / filename)
    print('Published posture comparison:', OUT / 'CURRENT_STANCE_ASSESSMENT.md', flush=True)


def assess_stance_impact():
    """Assessment only: compare support, perturbations and diagnostic motions."""
    from scipy.spatial import ConvexHull
    mass = json.loads((OUT / 'open_duck_mini_zp10d_mass_properties.json').read_text())
    stance = json.loads((OUT / 'CURRENT_STANCE_ASSESSMENT.json').read_text())
    paths = [ROOT / mass['source_step'], MODEL_DIR / 'open_duck_mini_zp10d.xml',
             MODEL_DIR / 'scene_flat_terrain.xml']
    expected = [mass['source_step_sha256'], mass['mjcf_sha256'], mass['scene_sha256']]
    assert [digest(p) for p in paths] == expected
    assert [stance[k] for k in ('source_step_sha256', 'mjcf_sha256', 'scene_sha256')] == expected
    model = mujoco.MjModel.from_xml_path(str(paths[2]))
    assert model.neq == 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    names = [model.actuator(i).name for i in range(model.nu)]; joints = model.actuator_trnid[:, 0]
    proposed = next(s for s in stance['scenarios'] if s['name'] == stance['recommended_trial'])
    base = model.body('base').id; floor = model.geom('floor').id
    feet = [model.body('foot_assembly').id, model.body('foot_assembly_2').id]
    report = {'date': date.today().isoformat(), 'source_step_sha256': expected[0], 'mjcf_sha256': expected[1],
        'scene_sha256': expected[2], 'script_sha256': digest(Path(__file__)),
        'source_stance_report_sha256': digest(OUT / 'CURRENT_STANCE_ASSESSMENT.json'),
        'status': 'assessment_only_user_has_not_approved_pose_change',
        'actuator_order': names, 'servo_ids': [SID[n] for n in names], 'postures': [],
        'push_protocol': {'force_levels_n': [2., 4.], 'duration_s': .2, 'start_s': 2.,
            'directions_world': ['+X forward', '-X backward', '+Y left', '-Y right'],
            'application': 'Actual trunk rigid-body COM, using xfrc_applied; baseline position servo gains only, no active IMU balance controller.'},
        'limitations': ['Estimated mass, contact and servo parameters remain uncalibrated; not physical stability certification.',
            'Foot contact polygon is built from actual foot-floor contact points with normal force >0.01 N; contact-proxy dependent.',
            'Geometric extension slack is link-length sum minus hip-ankle distance, not a certified usable motion range.',
            'Slow stepping is the existing diagnostic trajectory regenerated for each posture; it is not a trained gait.',
            'Sampled perturbations do not establish the maximum recoverable impulse. No formal home, CAD or hardware is changed.']}
    for label, ctrl in [('current', model.key_ctrl[0].copy()), ('candidate', np.array(proposed['target_ctrl_rad']))]:
        settled = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, settled, 0)
        settled.ctrl[:] = model.key_ctrl[0]
        for _ in range(round(3 / model.opt.timestep)):
            mujoco.mj_step(model, settled)
        for step in range(round(15 / model.opt.timestep)):
            settled.ctrl[:] = interpolate(step * model.opt.timestep, [(0., model.key_ctrl[0]), (5., ctrl), (15., ctrl)])
            mujoco.mj_step(model, settled)
        mujoco.mj_forward(model, settled)
        assert not any(settled.warning.number)
        com = settled.subtree_com[base].copy(); contact_points = []
        for index, contact in enumerate(settled.contact):
            if floor not in contact.geom:
                continue
            other = int(contact.geom[1] if contact.geom[0] == floor else contact.geom[0])
            if int(model.geom_bodyid[other]) not in feet:
                continue
            wrench = np.zeros(6); mujoco.mj_contactForce(model, settled, index, wrench)
            if wrench[0] > .01:
                contact_points.append(contact.pos[:2].tolist())
        points = np.unique(contact_points, axis=0); hull = ConvexHull(points)
        signed = hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]
        assert signed.max() < 0, (label, 'COM outside static contact support')
        margin = float(-signed.max())
        hips = settled.xanchor[[model.joint(f'{s}_hip_pitch').id for s in ('left', 'right')]]
        knees = settled.xanchor[[model.joint(f'{s}_knee').id for s in ('left', 'right')]]
        ankles = settled.xanchor[[model.joint(f'{s}_ankle').id for s in ('left', 'right')]]
        link_sum = np.linalg.norm(hips - knees, axis=1) + np.linalg.norm(knees - ankles, axis=1)
        slack = link_sum - np.linalg.norm(hips - ankles, axis=1)
        controls, ik = trajectories(model, settled, ctrl)
        row = {'name': label, 'ctrl_rad': ctrl.tolist(), 'settled_qpos': settled.qpos.tolist(),
            'com_world_m': com.tolist(), 'contact_polygon_world_xy_m': points[hull.vertices].tolist(),
            'static_min_contact_polygon_margin_mm': margin * 1000,
            'geometric_tipping_angle_deg': math.degrees(math.atan2(margin, com[2])),
            'geometric_link_extension_slack_mm': (slack * 1000).tolist(),
            'motion_ik_waypoints': ik, 'extra_raise_ik': [], 'scenarios': []}
        for height in (5, 10, 15, 20, 25, 30, 35, 40):
            _, checks = leg_targets(model, settled, ctrl, z=height / 1000)
            row['extra_raise_ik'].append({'height_mm': height, 'legs': checks,
                'passed': all(c['within_limits'] and c['pose_tolerance_passed'] for c in checks)})
        report['postures'].append(row)
        # Identical disturbance/motion protocols and the same gains for both.
        cases = [(f'{label}_hold', 20., [(0., ctrl), (20., ctrl)], None)]
        for level in (2., 4.):
            for axis, direction in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                force = np.zeros(3); force[axis] = level * direction
                push = {'start_s': 2., 'duration_s': .2, 'force_world_n': force.tolist()}
                cases.append((f'{label}_push_{level:g}N_{"XY"[axis]}{direction:+}',
                              8., [(0., ctrl), (8., ctrl)], push))
        for name, duration, waypoints in controls:
            if name == 'slow_step_attempt':
                cases.append((f'{label}_slow_step_attempt', duration, waypoints, None))
        for name, duration, waypoints, push in cases:
            outcome = simulate(model, settled, name, duration, waypoints, names, joints, push=push)
            if name.endswith('slow_step_attempt'):
                # motion_outcome uses the canonical name for its step criterion.
                outcome['motion_outcome'] = motion_outcome('slow_step_attempt', outcome['frames'])
            tail = [f for f in outcome['frames'] if f['time_s'] >= duration - 1]
            outcome['final_second_max_tilt_deg'] = max((f['base_tilt_deg'] for f in tail), default=None)
            outcome['recovered_by_end'] = bool(outcome['completed'] and tail and
                                               outcome['final_second_max_tilt_deg'] < 3.)
            row['scenarios'].append(outcome)
            (CACHE / 'impact_candidate.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print('impact posture', label, 'COM mm', np.round(com * 1000, 2).tolist(),
              'margin mm', round(margin * 1000, 2), 'extension slack mm', np.round(slack * 1000, 2).tolist(), flush=True)
    assert [digest(p) for p in paths] == expected, 'Formal files changed during assessment'
    assert digest(Path(__file__)) == report['script_sha256'], 'Assessment source changed while running'
    report['resources'] = {'address_space_limit_bytes': resource.getrlimit(resource.RLIMIT_AS)[0],
        'peak_rss_bytes_before_output': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'observed_threads': len(list(Path('/proc/self/task').iterdir())), 'serial': True}
    (CACHE / 'CURRENT_STANCE_IMPACT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(CACHE / 'CURRENT_STANCE_IMPACT.json', OUT / 'CURRENT_STANCE_IMPACT.json')
    print('Published assessment only:', OUT / 'CURRENT_STANCE_IMPACT.json', flush=True)


def stance_impact_report():
    """Review the saved matched trials without changing formal configuration."""
    report_path = OUT / 'CURRENT_STANCE_IMPACT.json'
    report = json.loads(report_path.read_text())
    for path, key in [(ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step', 'source_step_sha256'),
                      (MODEL_DIR / 'open_duck_mini_zp10d.xml', 'mjcf_sha256'),
                      (MODEL_DIR / 'scene_flat_terrain.xml', 'scene_sha256')]:
        assert digest(path) == report[key]
    assert report['status'] == 'assessment_only_user_has_not_approved_pose_change'
    assert digest(OUT / 'CURRENT_STANCE_ASSESSMENT.json') == report['source_stance_report_sha256']
    original, candidate = report['postures']
    assert original['name'] == 'current' and candidate['name'] == 'candidate'
    for posture in report['postures']:
        assert len(posture['scenarios']) == 10
        for case in posture['scenarios']:
            assert not any(case['warnings']) and not any(case['saturation_fraction'])
            expected_failure = case['name'] == 'candidate_push_4N_X-1'
            assert case['completed'] == (not expected_failure)

    def trial(posture, suffix):
        return next(s for s in posture['scenarios'] if s['name'] == posture['name'] + '_' + suffix)

    def holding(posture):
        frames = [f for f in trial(posture, 'hold')['frames'] if f['time_s'] >= 17.]
        return np.mean(np.abs([f['torque_nm'] for f in frames]), axis=0)

    old_load, new_load = holding(original), holding(candidate)
    backward_old = trial(original, 'push_4N_X-1'); backward_new = trial(candidate, 'push_4N_X-1')
    assert backward_old['completed'] and backward_new['stop_reason'] == 'excessive_tilt'
    assert all(not trial(p, 'slow_step_attempt')['motion_outcome']['task_achieved'] for p in report['postures'])
    confirmation = Path('/tmp/open-duck-stance/backward_push_confirmation.json')
    replay = json.loads(confirmation.read_text()) if confirmation.exists() else report.get('backward_push_confirmation', [])
    assert len(replay) == 2 and {r['posture'] for r in replay} == {'current', 'candidate'}
    for result in replay:
        assert result['exact_replay_passed']
        target = trial(next(p for p in report['postures'] if p['name'] == result['posture']), 'push_4N_X-1')
        assert abs(result['max_tilt_deg'] - target['max_base_tilt_deg']) < 1e-10
        assert result['stop_reason'] == target['stop_reason']
        assert result['simulated_seconds'] == target['simulated_seconds']
    report['backward_push_confirmation'] = replay
    report['assessment_conclusion'] = 'Do not adopt the candidate as the default stance: standing load is lower but backward disturbance response and extension reserve are worse; walking remains unproven.'
    report['report_renderer_sha256'] = digest(Path(__file__))
    height_change = (candidate['com_world_m'][2] / original['com_world_m'][2] - 1) * 100
    margin_change = (1 - candidate['static_min_contact_polygon_margin_mm'] / original['static_min_contact_polygon_margin_mm']) * 100
    lines = ['# 站姿修改影响评估：保持正式站姿，候选暂不采用', '',
        '用户最新决定：不同意修改，先评估影响。本轮只新增仿真评估证据，正式 CAD、MJCF、scene 的 home、运行配置和硬件均未改变。', '',
        '**结论：上抬 25 mm、后移 5 mm 可以降低站立膝扭矩，但向后抗扰动明显变差，不能直接作为默认站姿。** 此前仅凭静态扭矩建议采用，依据不足；本报告补充稳定性、支撑范围、几何伸展余量和慢步诊断。', '',
        '| 指标 | 当前站姿 | 候选站姿 | 含义 |', '| --- | ---: | ---: | --- |',
        f"| 整机重心高度 mm | {original['com_world_m'][2]*1000:.2f} | {candidate['com_world_m'][2]*1000:.2f} | 升高 {height_change:.1f}% |",
        f"| 重心投影到接触支撑边界的最小距离 mm | {original['static_min_contact_polygon_margin_mm']:.2f} | {candidate['static_min_contact_polygon_margin_mm']:.2f} | 缩小 {margin_change:.1f}%，此处最近边界在后侧 |",
        f"| 简化几何倾倒角 ° | {original['geometric_tipping_angle_deg']:.2f} | {candidate['geometric_tipping_angle_deg']:.2f} | atan(最小支撑距离/重心高度)，不是实测可恢复倾角 |",
        f"| 左/右膝站立保持扭矩 N·m | {old_load[3]:.3f} / {old_load[12]:.3f} | {new_load[3]:.3f} / {new_load[12]:.3f} | 最后3秒平均绝对值，仍约减载39%/37% |",
        f"| 左/右腿几何伸展余量 mm | {original['geometric_link_extension_slack_mm'][0]:.2f} / {original['geometric_link_extension_slack_mm'][1]:.2f} | {candidate['geometric_link_extension_slack_mm'][0]:.2f} / {candidate['geometric_link_extension_slack_mm'][1]:.2f} | 大腿+小腿轴距减去髋踝直线距离，不是抬脚高度或已验证可用行程 |", '',
        '## 同条件抗扰动试验', '',
        '每种姿态先自由站立3秒，再以5秒过渡、10秒保持准备同一初始状态。正式模型含自由基座，无基座固定。对躯干刚体质心施加世界坐标前/后/左/右 2 N 和 4 N 水平力，持续0.2秒；2 N/4 N分别对应0.4/0.8 N·s冲量。每次从独立初始状态开始，观察8秒。只保留原位置控制增益，没有主动IMU平衡或迈步恢复控制。', '',
        '| 推力 | 当前最大倾角 ° | 候选最大倾角 ° | 当前/候选结果 |', '| --- | ---: | ---: | --- |']
    for force in (2, 4):
        for direction, title in [('X+1', '向前'), ('X-1', '向后'), ('Y+1', '向左'), ('Y-1', '向右')]:
            a, b = [trial(p, f'push_{force}N_{direction}') for p in report['postures']]
            result = lambda s: '跑满且末秒倾角<3°' if s['recovered_by_end'] else f"{s['simulated_seconds']:.3f}秒触发{s['stop_reason']}"
            lines.append(f"| {force} N {title}，0.2秒 | {a['max_base_tilt_deg']:.2f} | {b['max_base_tilt_deg']:.2f} | {result(a)} / {result(b)} |")
    lines.extend(['', '关键失败：候选在2.0～2.2秒接受向后4 N推力后，俯仰继续向后增长，2.998秒达到35.01°总倾角触发停止；同条件原站姿最大仅2.82°，跑满8秒。该失败已从相同初始状态独立重跑复现（见JSON confirmation）。本次没有放大推力寻找极限，因此不能将4 N认定为临界阈值。', '',
        '重心后移使后侧支撑余量缩小，重心升高使简化几何倾倒角降低；腿伸得更直也改变了姿态/接触响应。当前数据证明该组合候选后向抗扰动更差，尚未分别隔离这三个因素的贡献。无饱和不能证明稳定，失败工况中执行器同样未达到堵转上限。', '',
        '## 对站立和行走的影响', '',
        '安静站立：两姿态分别又保持20秒，都未倒下、无数值警告。候选可以静止站住，但静止站住不足以认定更稳定。所有推力试验数值警告也为0，候选后倾是姿态失稳，不是数值爆炸。', '',
        '伸展空间：在脚位与脚姿保持的同一IK验收标准下，当前站姿进一步上抬35 mm目标可解、40 mm失败；候选只能进一步上抬到采样10 mm，15 mm失败。这是5 mm步长的目标可达性抽样，不能当作精确行程、地形能力或抬脚高度。', '',
        '慢步：对每种姿态重新计算相同相对动作的IK目标，全部通过0.5 mm/1°容差和目标限位。两组都跑满19秒而未倒下，但都没有成功脚底离地（脚底>2 mm、竖直力<0.5 N连续0.2秒的联合标准）。因此不能判断候选行走更好；也不能把现有未成功的控制器表现当作结构必然无法行走。', '',
        f"同一慢步尝试左右膝峰值：当前 {trial(original,'slow_step_attempt')['peak_motor_torque_nm'][3]:.3f}/{trial(original,'slow_step_attempt')['peak_motor_torque_nm'][12]:.3f} N·m，候选 {trial(candidate,'slow_step_attempt')['peak_motor_torque_nm'][3]:.3f}/{trial(candidate,'slow_step_attempt')['peak_motor_torque_nm'][12]:.3f} N·m；虽然膝载荷下降，动作本身均未成功。", '',
        '**处置：保留当前正式站姿，撤回将该组合候选直接设为默认的建议。** 后续若继续优化，应把静态负载、后向支撑余量、扰动恢复与真实单脚支撑同时作为目标；可先评估更小抬高、减少后移及平衡控制，但本轮未替用户采纳新姿态。', '',
        '## 数据范围与复现', '',
        '质量仍约2.098 kg，未用PETG真实切片重量校准；摩擦、惯量、舵机阻尼及连续扭矩仍有上一阶段不确定性。本次不是实物跌倒阈值认证，也不是CAD碰撞/强度验收。支撑多边形来自当前模型脚底与地面实际有力接触点，受碰撞代理影响。', '',
        f"本轮仿真共20个工况，包含一处按停止条件提前结束；原生线程{report['resources']['observed_threads']}、串行、导入前4 GB地址空间硬限额，输出前峰值RSS约{report['resources']['peak_rss_bytes_before_output']/1e9:.3f} GB。", '',
        '```bash', '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/assess_zp10d_loads.py stance-impact',
        '.venv/bin/python scripts/cad/run_memory_limited.py scripts/cad/assess_zp10d_loads.py stance-impact-report', '```', ''])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
    for posture, color in [(original, '#307fa5'), (candidate, '#df813a')]:
        polygon = np.array(posture['contact_polygon_world_xy_m']) * 1000
        loop = np.vstack((polygon, polygon[0]))
        axes[0].plot(loop[:, 0], loop[:, 1], color=color, label=posture['name'])
        axes[0].plot(posture['com_world_m'][0]*1000, posture['com_world_m'][1]*1000, 'o', color=color)
        push = trial(posture, 'push_4N_X-1')
        axes[1].plot([f['time_s'] for f in push['frames']], [f['base_tilt_deg'] for f in push['frames']], color=color, label=posture['name'])
    axes[0].set_aspect('equal'); axes[0].set_xlabel('X forward (mm)'); axes[0].set_ylabel('Y left (mm)')
    axes[0].set_title('Contact support and COM projection'); axes[0].legend()
    axes[1].axvspan(2., 2.2, color='gray', alpha=.25, label='Backward pulse: 4 N, 0.2 s')
    axes[1].axhline(35., color='red', linestyle='--', label='Stop criterion')
    axes[1].plot(backward_new['simulated_seconds'], backward_new['max_base_tilt_deg'], 'x', color='#df813a')
    axes[1].annotate('Stopped at 35.01 deg',
        (backward_new['simulated_seconds'], backward_new['max_base_tilt_deg']),
        xytext=(8, -15), textcoords='offset points', fontsize=8)
    axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Base tilt (deg)')
    axes[1].set_title('Same backward disturbance'); axes[1].legend(fontsize=8)
    x = np.array([0., 1.])
    axes[2].bar(x-.18, [old_load[3],new_load[3]], .36, label='Left S09')
    axes[2].bar(x+.18, [old_load[12],new_load[12]], .36, label='Right S04')
    axes[2].set_xticks(x,['Current','Candidate']); axes[2].set_ylabel('Holding torque (N m)')
    axes[2].set_title('Lower load does not ensure stability'); axes[2].legend()
    fig.savefig(CACHE / 'CURRENT_STANCE_IMPACT.png', dpi=170); plt.close(fig)
    (CACHE / 'CURRENT_STANCE_IMPACT.md').write_text('\n'.join(lines))
    (CACHE / 'CURRENT_STANCE_IMPACT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    for suffix in ('md','png','json'):
        filename = f'CURRENT_STANCE_IMPACT.{suffix}'
        shutil.copyfile(CACHE / filename, OUT / filename)
    for path, key in [(ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step', 'source_step_sha256'),
                      (MODEL_DIR / 'open_duck_mini_zp10d.xml', 'mjcf_sha256'),
                      (MODEL_DIR / 'scene_flat_terrain.xml', 'scene_sha256')]:
        assert digest(path) == report[key], 'Formal input changed during report generation'
    print('Reviewed stance impact:', OUT / 'CURRENT_STANCE_IMPACT.md', flush=True)


def run():
    CACHE.mkdir(exist_ok=True)
    mass = json.loads((OUT / 'open_duck_mini_zp10d_mass_properties.json').read_text())
    scene = MODEL_DIR / 'scene_flat_terrain.xml'; xml = MODEL_DIR / 'open_duck_mini_zp10d.xml'
    assert mass['stage'] == 'published' and mass['validation']['synchronization_passed']
    assert digest(ROOT / mass['source_step']) == mass['source_step_sha256']
    assert digest(xml) == mass['mjcf_sha256'] and digest(scene) == mass['scene_sha256']
    for part in mass['parts']:
        for mesh, expected in [(part['mesh'], part['mesh_sha256'])] + [(p['mesh'], p['sha256']) for p in part['collision']['pieces']]:
            assert digest(MODEL_DIR / 'assets' / mesh) == expected
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert model.neq == 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    names = [model.actuator(a).name for a in range(model.nu)]; joints = model.actuator_trnid[:, 0]
    settled = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, settled, 0)
    settled.ctrl[:] = model.key_ctrl[0]
    for _ in range(round(3 / model.opt.timestep)):
        mujoco.mj_step(model, settled)
    mujoco.mj_forward(model, settled)
    checks = consistency_checks(model, settled, joints)
    cases, ik = trajectories(model, settled, model.key_ctrl[0].copy())
    report = {'date': date.today().isoformat(), 'source_step_sha256': mass['source_step_sha256'],
        'mjcf_sha256': digest(xml), 'scene_sha256': digest(scene), 'script_sha256': digest(Path(__file__)),
        'mass_kg': float(model.body_mass.sum()), 'gravity_m_s2': model.opt.gravity.tolist(),
        'weight_n': float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity)),
        'actuator_order': names, 'servo_ids': [SID[n] for n in names],
        'provisional_stall_limits_nm': model.actuator_forcerange[:, 1].tolist(),
        'initial_settling_seconds': 3., 'ground_supported': True, 'external_base_support': False,
        'print_spec': {'material': 'PETG', 'shell_infill_percent': 10, 'other_printed_parts_infill_percent': 30,
                       'wall_layers': None, 'top_bottom_layers': None, 'layer_height_mm': None,
                       'print_orientation': None, 'infill_pattern': None},
        'checks': checks, 'ik_waypoints': ik, 'scenarios': [],
        'assumptions': ['Masses and actuator response inherited from the synchronized estimated model; PETG print settings are recorded but do not establish actual part weights.',
                        'Ground forces include friction and compliant contact. Motor torque differs from total hinge reaction because of joint friction/limits and armature.',
                        'Motor speed/continuous torque/thermal curves are unavailable; reported stall ratios are diagnostics, not continuous safety factors.',
                        'Joint wrenches are aggregate assembly loads; fastener and bracket load sharing must be defined for local FEA.',
                        '10/30 percent infill cannot be substituted for material density or yield strength; walls, solid skins and print anisotropy remain unspecified.']}
    for name, duration, waypoints in cases:
        report['scenarios'].append(simulate(model, settled, name, duration, waypoints, names, joints))
        (CACHE / 'candidate.json').write_text(json.dumps(report, indent=2) + '\n')
    standing = report['scenarios'][0]
    steady = [f for f in standing['frames'] if f['time_s'] >= 4]
    mean = np.mean([f['ground_force_world_n'] for f in steady], axis=0)
    error = abs(mean[2] - report['weight_n']) / report['weight_n']
    assert error < .01 and standing['completed'], ('standing force balance', error)
    report['checks']['standing_mean_ground_force_world_n'] = mean.tolist()
    report['checks']['standing_vertical_weight_balance_relative_error'] = float(error)
    report['resources'] = {'address_space_limit_bytes': resource.getrlimit(resource.RLIMIT_AS)[0],
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'observed_process_threads': len(list(Path('/proc/self/task').iterdir())), 'serial': True}
    assert digest(xml) == report['mjcf_sha256'] and digest(scene) == report['scene_sha256']
    assert digest(ROOT / mass['source_step']) == report['source_step_sha256']
    assert digest(Path(__file__)) == report['script_sha256'], 'Script changed during simulation'
    publish_outputs(report, mass)
    print('Published ground-supported load assessment:', REPORT, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['stance-impact-report']:
        CACHE.mkdir(exist_ok=True)
        stance_impact_report()
    elif sys.argv[1:] == ['stance-impact']:
        CACHE.mkdir(exist_ok=True)
        assess_stance_impact()
    elif sys.argv[1:] == ['stance']:
        CACHE.mkdir(exist_ok=True)
        assess_stance()
    elif sys.argv[1:] == ['report']:
        report = json.loads(REPORT.read_text())
        mass = json.loads((OUT / 'open_duck_mini_zp10d_mass_properties.json').read_text())
        assert digest(ROOT / mass['source_step']) == report['source_step_sha256']
        assert digest(MODEL_DIR / 'open_duck_mini_zp10d.xml') == report['mjcf_sha256']
        assert digest(MODEL_DIR / 'scene_flat_terrain.xml') == report['scene_sha256']
        report['report_renderer_sha256'] = digest(Path(__file__))
        report['resources']['report_render_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['resources']['report_render_peak_sample_scope'] = 'After loading saved JSON, before plots; simulation peak is preserved separately.'
        publish_outputs(report, mass)
    else:
        run()
