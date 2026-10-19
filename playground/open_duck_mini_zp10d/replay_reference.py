"""Compare torque-capped and uncapped Placo joint commands in free-base physics.

Run through scripts/cad/run_memory_limited.py. This diagnostic does not change
model files, impose the planned base motion, or implement a balance controller.
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import mujoco
import numpy as np

from reference import DATA, ROOT, SCENE, source_hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def longest_run(mask, dt):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * dt


def replay(model, targets, desired_contacts, feet, vertices, warmup, blend):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    joint_ids = model.actuator_trnid[:, 0]
    addresses = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids]
    floor = model.geom('floor').id
    weight = float(model.body_subtreemass[1] * abs(model.opt.gravity[2]))
    dt = model.opt.timestep
    rows, self_contacts, nonfoot_contacts, peak_events = [], {}, {}, {}
    knees = [model.actuator(side + '_knee').id for side in ('left', 'right')]
    stop = 'duration'
    for i, target in enumerate(targets):
        # mj_step leaves position/contact fields at this pre-integration state.
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        force = np.zeros(2)
        for ci in range(data.ncon):
            contact = data.contact[ci]
            pair = contact.geom
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, wrench)
            if floor in pair:
                other = pair[1] if pair[0] == floor else pair[0]
                body = int(model.geom_bodyid[other])
                if body in feet:
                    world = contact.frame.reshape(3, 3).T @ wrench[:3]
                    force[feet.index(body)] += max(0., world[2] * (1 if pair[0] == floor else -1))
                elif np.linalg.norm(wrench[:3]) > .05:
                    name = model.body(body).name
                    nonfoot_contacts[name] = max(nonfoot_contacts.get(name, 0.), float(np.linalg.norm(wrench[:3])))
            elif np.linalg.norm(wrench[:3]) > .05:
                name = ' / '.join(sorted(model.body(int(model.geom_bodyid[g])).name for g in pair))
                self_contacts[name] = max(self_contacts.get(name, 0.), float(-contact.dist))
        height = [float((v @ data.xmat[b].reshape(3, 3).T + data.xpos[b])[:, 2].min())
                  for b, v in zip(feet, vertices)]
        tilt = float(np.rad2deg(np.arccos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1., 1.))))
        error = target - qpos[addresses]
        for actuator in knees:
            name = model.actuator(actuator).name
            torque = float(data.actuator_force[actuator])
            if abs(torque) <= abs(peak_events.get(name, {}).get('motor_torque_nm', 0.)):
                continue
            inertia = np.zeros(model.nv)
            mujoco.mj_mulM(model, data, inertia, data.qacc)
            friction = np.zeros(model.nv)
            friction_rows = np.where(data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                                     data.efc_force, 0.)
            mujoco.mj_mulJacTVec(model, data, friction, friction_rows)
            dof = dofs[actuator]
            peak_events[name] = dict(
                time_s=i * dt, reference_time_s=i * dt - warmup - blend,
                target_deg=float(np.rad2deg(target[actuator])),
                actual_deg=float(np.rad2deg(qpos[addresses[actuator]])),
                motor_torque_nm=torque, velocity_rad_s=float(qvel[dof]),
                acceleration_rad_s2=float(data.qacc[dof]),
                desired_contacts=desired_contacts[i].tolist(), foot_force_n=force.tolist(),
                whole_foot_clearance_mm=(1000 * np.array(height)).tolist(),
                total_inertia_nm=float(inertia[dof]),
                armature_inertia_nm=float(model.dof_armature[dof] * data.qacc[dof]),
                gravity_and_velocity_bias_nm=float(data.qfrc_bias[dof]),
                passive_nm=float(data.qfrc_passive[dof]),
                constraint_nm=float(data.qfrc_constraint[dof]),
                dry_friction_constraint_nm=float(friction[dof]),
                balance_residual_nm=float(inertia[dof] + data.qfrc_bias[dof] - data.qfrc_passive[dof]
                                          - data.qfrc_actuator[dof] - data.qfrc_constraint[dof]))
        rows.append((qpos, height, force, tilt, data.actuator_force.copy(), error, qvel))
        if not np.isfinite(data.qpos).all() or tilt > 45 or qpos[2] < .10:
            stop = 'fall_or_nonfinite'
            break
        if nonfoot_contacts or any(data.warning.number):
            stop = 'nonfoot_ground_contact' if nonfoot_contacts else 'numerical_warning'
            break
    qpos, heights, forces, tilts, torques, errors, qvel = map(np.asarray, zip(*rows))
    times = np.arange(len(rows)) * dt
    motion = times >= warmup + blend
    assert motion.any(), 'Stopped before the Placo reference began'
    requested_swing = ~desired_contacts[:len(rows)].astype(bool)
    summaries = []
    for side in range(2):
        loaded_other = forces[:, 1 - side] > .5 * weight
        unloaded = (forces[:, side] < .5) & loaded_other & motion
        summaries.append(dict(
            peak_clearance_mm=float(1000 * heights[motion, side].max()),
            peak_clearance_with_other_foot_loaded_mm=(float(1000 * heights[unloaded, side].max())
                                                       if unloaded.any() else None),
            longest_supported_clearance_s={str(mm): longest_run(unloaded & (heights[:, side] >= mm / 1000), dt)
                                           for mm in (2, 5, 10)},
            first_requested_swing_s=(float(times[requested_swing[:, side]][0])
                                     if requested_swing[:, side].any() else None),
            peak_clearance_during_requested_swing_mm=(float(1000 * heights[requested_swing[:, side], side].max())
                                                     if requested_swing[:, side].any() else None)))
    names = [model.actuator(i).name for i in range(model.nu)]
    limited = model.jnt_limited[joint_ids].astype(bool)
    lower, upper = model.jnt_range[joint_ids].T
    margins = np.minimum(qpos[:, addresses] - lower, upper - qpos[:, addresses])[:, limited]
    worst = int(np.argmin(margins.min(axis=0)))
    report = dict(stop=stop, simulated_seconds=len(rows) * dt,
                  replayed_reference_seconds=max(0., len(rows) * dt - warmup - blend),
                  feet=dict(zip(('left', 'right'), summaries)),
                  peak_torque_nm=dict(zip(names, abs(torques).max(axis=0).tolist())),
                  max_joint_error_deg=float(np.rad2deg(abs(errors[motion]).max())),
                  max_tilt_deg=float(tilts.max()), final_tilt_deg=float(tilts[-1]),
                  self_contact_max_penetration_m=self_contacts,
                  nonfoot_ground_contact_peak_force_n=nonfoot_contacts,
                  max_joint_limit_overrun_deg=float(np.rad2deg(max(0., -margins.min()))),
                  joint_with_smallest_limit_margin=np.array(names)[limited][worst],
                  knee_peak_events=peak_events,
                  warnings=data.warning.number.tolist())
    arrays = dict(time=times, qpos=qpos, qvel=qvel, foot_clearance=heights, foot_force=forces,
                  tilt_deg=tilts, torque=torques, joint_error=errors, target=targets[:len(rows)],
                  desired_contact=desired_contacts[:len(rows)])
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path, help='200 Hz Placo candidate, including startup')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manual-6p6v', type=Path,
                        help='use reviewed manual 6.6 V speed and stall torque in memory; include formal baseline')
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / 'training/reference_replay'
        if args.manual_6p6v:
            args.output /= 'manual_6p6v'
    assumption = None
    if args.manual_6p6v:
        from audit_actuator_response import manual_evidence, manual_6p6v_parameters, parameters, run
        manual = manual_evidence(args.manual_6p6v)
        config = manual_6p6v_parameters(parameters(SCENE.with_name('open_duck_mini_zp10d.xml'), 'zp10d'), manual)
        assumption = dict(manual=manual, parameters=config, voltage_v=6.6,
            no_load_seconds_per_60_deg=.14, stall_torque_kgf_cm=15.,
            scope='user-selected generic manual parameters; linear torque-speed approximation; '
                  'gain, friction, armature inherited; no physical calibration or 8 V extrapolation',
            isolated_response={})
        for label, position in [('constant_torque', False), ('60deg_step', True)]:
            assumption['isolated_response'][label], _ = run(config, position_control=position)
    assert longest_run([True, True, False, True], .002) == .004
    protected = [ROOT / 'open_duck_mini_zp10d.urdf', SCENE,
                 SCENE.with_name('open_duck_mini_zp10d.xml'),
                 DATA / 'reference.json', DATA / 'polynomial_coefficients.pkl']
    before = {str(p): digest(p) for p in protected}
    raw = np.load(args.candidate)
    meta = json.loads(str(raw['metadata']))
    assert meta['source_hashes'] == source_hashes()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    feet = [int(model.site_bodyid[model.site(side + '_foot').id]) for side in ('left', 'right')]
    vertices = []
    for body in feet:
        points = []
        for gid in np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)):
            assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
            mid = model.geom_dataid[gid]
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            points.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
        vertices.append(np.unique(np.concatenate(points), axis=0))
    names = [model.actuator(i).name for i in range(model.nu)]
    columns = [meta['joint_names'].index(model.joint(int(j)).name) for j in model.actuator_trnid[:, 0]]
    frames = raw['dense_frames']
    dt, fps, warmup, blend = model.opt.timestep, meta['sample_fps'], 1., .5
    reference_times = np.arange(len(frames)) / fps
    times = np.arange(round((warmup + blend + reference_times[-1]) / dt)) * dt
    progress = np.clip(times - warmup - blend, 0., reference_times[-1])
    targets = np.column_stack([np.interp(progress, reference_times, frames[:, col]) for col in columns])
    alpha = np.clip((times - warmup) / blend, 0., 1.)
    alpha = alpha * alpha * (3 - 2 * alpha)
    targets = model.key('home').ctrl + alpha[:, None] * (targets - model.key('home').ctrl)
    desired_contacts = frames[np.minimum((progress * fps).astype(int), len(frames) - 1), 32:34]
    desired_contacts[times < warmup + blend] = 1
    joints = model.actuator_trnid[:, 0]
    assert np.all(targets >= model.jnt_range[joints, 0] - 1e-6)
    assert np.all(targets <= model.jnt_range[joints, 1] + 1e-6)
    main_servos = np.array([i for i, name in enumerate(names) if 'antenna' not in name])
    assert len(main_servos) == 14 and model.ngeom == 3296
    original_flags = model.actuator_forcelimited.copy()
    original_joint_flags = model.jnt_actfrclimited.copy()
    original_caps = model.actuator_forcerange.copy()
    original_joint_caps = model.jnt_actfrcrange.copy()
    original_damping = model.dof_damping.copy()
    cases = {}
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ('formal_baseline', 'manual_6p6v', 'manual_6p6v_home') if assumption else ('capped', 'uncapped')
    for mode in modes:
        if mode == 'uncapped':
            model.actuator_forcelimited[main_servos] = False
            model.jnt_actfrclimited[joints[main_servos]] = False
        elif mode.startswith('manual_6p6v'):
            model.actuator_forcelimited[main_servos] = True
            model.actuator_forcerange[main_servos] = [-config['torque_cap'], config['torque_cap']]
            model.jnt_actfrcrange[joints[main_servos]] = [-config['torque_cap'], config['torque_cap']]
            model.dof_damping[model.jnt_dofadr[joints[main_servos]]] = config['damping']
        home_only = mode.endswith('_home')
        commands = np.tile(model.key('home').ctrl, (round(3. / dt), 1)) if home_only else targets
        contacts = np.ones((len(commands), 2)) if home_only else desired_contacts
        print('START', mode, flush=True)
        cases[mode], arrays = replay(model, commands, contacts, feet, vertices,
                                    .5 if home_only else warmup, 0. if home_only else blend)
        if assumption:
            cases[mode]['role'] = '3 s static home hold' if home_only else 'identical Placo joint reference'
            cap = config['torque_cap'] if mode.startswith('manual_6p6v') else .980665
            assert abs(arrays['torque'][:, main_servos]).max() <= cap + 1e-10
            assert model.actuator_forcelimited[main_servos].all()
            cases[mode]['main_servo_torque_cap_nm'] = cap
            if not home_only:
                np.testing.assert_array_equal(arrays['target'], targets[:len(arrays['target'])])
        np.savez_compressed(args.output / (mode + '.npz'), **arrays)
        print(json.dumps({mode: cases[mode]}), flush=True)
    model.actuator_forcelimited[:] = original_flags
    model.jnt_actfrclimited[:] = original_joint_flags
    model.actuator_forcerange[:] = original_caps
    model.jnt_actfrcrange[:] = original_joint_caps
    model.dof_damping[:] = original_damping
    assert before == {str(p): digest(p) for p in protected}
    report = dict(method='identical open-loop joint targets; free-base full-model MuJoCo integration',
                  candidate_sha256=digest(args.candidate), source_hashes=source_hashes(),
                  script_sha256=digest(__file__), protected_files_unchanged=before,
                  warmup_s=warmup, initial_blend_s=blend, physics_dt_s=dt,
                  position_command_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),
                  uncapped_actuators=[] if assumption else [names[i] for i in main_servos],
                  original_torque_caps_nm=original_caps.tolist(),
                  manual_6p6v_assumption=assumption,
                  planned_whole_foot_peak_clearance_mm=(1000 * raw['dense_heights'].max(axis=0)).tolist(),
                  initial_reference_joint_offset_deg=float(np.rad2deg(abs(frames[0, columns] - model.key('home').ctrl).max())),
                  acceptance='diagnostic only; no learned policy or balance feedback; not a walking certificate',
                  cases=cases)
    (args.output / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: identical commands, full collision model, formal model/reference files unchanged', flush=True)


if __name__ == '__main__':
    main()
