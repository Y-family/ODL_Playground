"""Causal full-model tests of the identified actuator's failed Placo liftoff.

Use run_memory_limited.py. Counterfactual actuator settings are diagnostic only;
formal geometry, force limits, reference data and robot files are never written.
"""
import argparse
import json
from pathlib import Path

from replay_reference import digest, replay
from reference import ROOT, SCENE, MODEL, URDF, DATA, source_hashes, foot_geometry
from assess_reference import contact_allocation, support_points, check_contact_geometry, ground_zmp
import mujoco
import numpy as np
from actuators import set_identified, configure


def hip_transfer_probe(live_directory=None, side_filter=None):
    """Causal test of the user's S02/S07 proposal, not a replacement controller."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    parameters = configure(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    sites = [model.site(side + '_foot').id for side in ('left', 'right')]
    feet = [int(model.site_bodyid[s]) for s in sites]
    vertex_sets, _ = foot_geometry()
    vertices = []
    for site, body, local in zip(sites, feet, vertex_sets):
        world = local @ data.site_xmat[site].reshape(3, 3).T + data.site_xpos[site]
        vertices.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
    times = np.arange(round(1.2 / model.opt.timestep)) * model.opt.timestep
    u = np.clip((times - .2) / .5, 0., 1.)
    blend = u**3 * (10 - 15*u + 6*u*u)
    home = model.key('home').ctrl.copy()
    output = ROOT / 'training/reference_repair'
    output.mkdir(parents=True, exist_ok=True)
    cases, recorded = {}, {}
    modes = ((None, 0.), ('right', 20.), ('right', 25.), ('left', -20.), ('left', -25.))
    if side_filter:
        modes = ((side_filter, -20. if side_filter == 'left' else 20.),)
    for side, degrees in modes:
        name = 'home' if side is None else side + '_' + str(abs(int(degrees)))
        targets = np.tile(home, (len(times), 1))
        if side:
            motor = model.actuator(side + '_hip_roll').id
            targets[:, motor] += blend * (np.deg2rad(degrees) - home[motor])
        assert np.all(targets >= model.actuator_ctrlrange[:, 0] - 1e-8)
        assert np.all(targets <= model.actuator_ctrlrange[:, 1] + 1e-8)
        assert abs(np.diff(targets, axis=0)).max() / model.opt.timestep < 5.24
        live = None
        if live_directory:
            from visualize_evaluation import LiveView
            servo = 'S07' if side == 'left' else 'S02' if side == 'right' else 'home'
            live = LiveView(model, live_directory,
                            f'{servo} 单关节物理诊断 · 指令 {degrees:+g}° · 非完整转重动作')
        report, arrays = replay(model, targets, np.ones((len(times), 2)), feet, vertices,
                                .2, .5, delay_s=parameters['transport_delay_s'], live=live)
        com = []
        for qpos in arrays['qpos']:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            com.append(data.subtree_com[1].copy())
        com = np.asarray(com)
        tail = arrays['time'] >= .8
        report.update(com_y_range_mm=(com[:, 1].min()*1000, com[:, 1].max()*1000),
                      final_root_xyz_mm=(arrays['qpos'][-1, :3]*1000).tolist(),
                      final_com_xyz_mm=(com[-1]*1000).tolist(),
                      settled_min_foot_load_n=arrays['foot_force'][tail].min(0).tolist(),
                      final_foot_load_n=arrays['foot_force'][-1].tolist())
        cases[name] = report
        recorded.update({name + '_' + key: value for key, value in dict(arrays, com=com).items()})
        print('HIP_TRANSFER', name, json.dumps(report), flush=True)
    stem = 'hip_transfer' + ('_' + side_filter if side_filter else '')
    (output / (stem + '.json')).write_text(json.dumps(dict(cases=cases,
        source_hashes=source_hashes(), parameters=parameters,
        scope='1.2 s causal hip command test; free base, no imposed pose or added forces; '
              'not Placo/PPO gait acceptance'), indent=2) + '\n')
    np.savez_compressed(output / (stem + '.npz'), **recorded)


def directed_transfer_probe(live_directory=None):
    """Placo COM-right/S02-inward transfer before left lift; NOT a gait pass."""
    import placo
    from reference import inputs

    hashes = source_hashes()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    parameters = configure(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    sites = [model.site(s + '_foot').id for s in ('left', 'right')]
    feet = [int(model.site_bodyid[s]) for s in sites]
    right = data.site_xpos[sites[1]] - data.site_xpos[sites[0]]
    right[2] = 0.
    right /= np.linalg.norm(right)
    initial_com, initial_body = data.subtree_com[1].copy(), data.xpos[1].copy()
    site_vertices, offsets = foot_geometry()
    vertices = []
    for site, body, local in zip(sites, feet, site_vertices):
        world = local @ data.site_xmat[site].reshape(3, 3).T + data.site_xpos[site]
        vertices.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))

    names, _, home = inputs()
    robot = placo.HumanoidRobot(str(URDF))
    for name, value in zip(names, home[7:]):
        robot.set_joint(name, float(value))
    transform = np.eye(4)
    transform[:3, 3] = home[:3]
    transform[2, 3] -= .0002
    robot.set_T_world_fbase(transform)
    robot.update_kinematics()
    com_home = robot.com_world().copy()
    foot_home = [robot.get_T_world_left().copy(), robot.get_T_world_right().copy()]
    solver = placo.KinematicsSolver(robot)
    solver.dt = .01
    solver.enable_joint_limits(True)
    tasks = placo.WalkTasks()
    tasks.initialize_tasks(solver, robot)
    tasks.com_task.configure('com_right', 'hard', 1.)
    tasks.trunk_orientation_task.mask.set_axises('y', 'local')
    tasks.trunk_orientation_task.configure('pitch', 'hard', 1.)
    foot_tasks = (tasks.left_foot_task, tasks.right_foot_task)
    for i, task in enumerate(foot_tasks):
        task.target = foot_home[i]
        task.position().configure('planted_foot', 'hard', 1.)
        task.orientation().configure('foot_orientation', 'hard', 1.)
    tasks.left_foot_task.orientation().mask.set_axises('yz', 'local')
    upper = solver.add_joints_task()
    upper.set_joints({n: float(robot.get_joint(n)) for n in names
                     if not any(x in n for x in ('hip', 'knee', 'ankle'))})
    upper.configure('upper', 'hard', 1.)
    sole_vertices = [v + offset for v, offset in zip(site_vertices, offsets)]
    positions, bases = [], []
    for mm in range(27):
        target = com_home + right * mm / 1000
        target[2] -= .0038 * (mm / 25)**2
        tasks.com_task.target_world = target
        for i, task in enumerate(foot_tasks):
            position = foot_home[i][:3, 3].copy()
            rotation = robot.get_T_world_frame(('left', 'right')[i] + '_foot')[:3, :3]
            position[2] = -np.min((sole_vertices[i] @ rotation.T)[:, 2])
            task.position().target_world = position
        solver.solve(True)
        robot.update_kinematics()
        positions.append([robot.get_joint(n) for n in names])
        bases.append(robot.get_T_world_fbase().copy())
    positions, bases = np.asarray(positions), np.asarray(bases)

    # Only motor commands are replayed. Planned base/COM poses never overwrite
    # the free physical state. Feedforward uses the unchanged identified gain.
    coarse_time = np.arange(0., 1.42, .02)
    u = np.clip((coarse_time - .2) / 1., 0., 1.)
    progress = 25 * u**3 * (10 - 15*u + 6*u*u)
    poses = np.tile(model.key('home').qpos, (len(coarse_time), 1))
    columns = [names.index(model.joint(int(j)).name) for j in model.actuator_trnid[:, 0]]
    adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    for i, amount in enumerate(progress):
        lo, alpha = int(amount), amount - int(amount)
        poses[i, adr] = (1-alpha)*positions[lo, columns] + alpha*positions[lo+1, columns]
        poses[i, :3] = (1-alpha)*bases[lo, :3, 3] + alpha*bases[lo+1, :3, 3]
        a, b = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(a, bases[lo, :3, :3].ravel())
        mujoco.mju_mat2Quat(b, bases[lo+1, :3, :3].ravel())
        poses[i, 3:7] = (1-alpha)*a + alpha*b
        poses[i, 3:7] /= np.linalg.norm(poses[i, 3:7])
    velocity = np.zeros((len(poses), model.nv))
    for i in range(1, len(poses)-1):
        mujoco.mj_differentiatePos(model, velocity[i], .04, poses[i-1], poses[i+1])
    acceleration = np.gradient(velocity, .02, axis=0)
    commands, ratios = [], []
    for i in range(len(poses)):
        data.qpos[:], data.qvel[:] = poses[i], velocity[i]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        data.qacc[:] = acceleration[i]
        points, bodies, _ = support_points(model, data, sites, site_vertices, [1, 1])
        allocation = contact_allocation(model, data, points, bodies, .6, command_limits=True)
        assert allocation['feasible'] and allocation['min_peak_torque_ratio'] <= 1.
        ratios.append(allocation['min_peak_torque_ratio'])
        commands.append(poses[i, adr] + np.asarray(allocation['motor_torque_nm']) / model.actuator_gainprm[:, 0])
    commands = np.asarray(commands)
    dt = model.opt.timestep
    times = np.arange(round(1.4 / dt)) * dt
    targets = np.column_stack([np.interp(times, coarse_time, commands[:, j]) for j in range(model.nu)])
    previous = model.key('home').ctrl.copy()
    for row in targets:
        row[:] = previous + np.clip(row - previous, -5.24*dt, 5.24*dt)
        previous = row.copy()
    assert np.max(abs(np.diff(targets, axis=0))) / dt <= 5.24 + 1e-10
    assert np.all(targets >= model.actuator_ctrlrange[:, 0] - 1e-10)
    assert np.all(targets <= model.actuator_ctrlrange[:, 1] + 1e-10)
    live = None
    if live_directory:
        from visualize_evaluation import LiveView
        live = LiveView(model, live_directory,
                        '抬左脚前的转重诊断：S02向内、机身与重心向右；尚未完成卸载/抬脚')
    report, arrays = replay(model, targets, np.ones((len(times), 2)), feet, vertices,
                            .2, 1., delay_s=parameters['transport_delay_s'], live=live)
    com = []
    for qpos in arrays['qpos']:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com.append(data.subtree_com[1].copy())
    com = np.asarray(com)
    com_shift = float((com[-1] - initial_com) @ right * 1000)
    body_shift = float((arrays['qpos'][-1, :3] - initial_body) @ right * 1000)
    loads = arrays['foot_force'][-1]
    s02 = model.actuator('right_hip_roll').id
    report.update(requested_lift='left', right_axis_world=right.tolist(),
                  actual_com_right_shift_mm=com_shift, actual_body_right_shift_mm=body_shift,
                  final_foot_load_n=loads.tolist(),
                  final_s02_actual_deg=float(np.rad2deg(arrays['qpos'][-1, adr[s02]])),
                  planned_s02_actual_deg=float(np.rad2deg(positions[25, names.index('right_hip_roll')])) ,
                  peak_planned_torque_ratio=max(ratios),
                  direction_verified=bool(com_shift > 15 and body_shift > 1 and loads[1] > loads[0]+3
                      and report['stop'] == 'duration' and report['max_joint_limit_overrun_deg'] == 0),
                  liftoff_verified=report['feet']['left']['longest_supported_clearance_s']['2'] > 0,
                  source_hashes=hashes, parameters=parameters,
                  scope='1.4 s Placo-directed transfer diagnostic; both feet still requested planted; '
                        'does not establish unloading, sustained balance, liftoff or gait acceptance')
    assert source_hashes() == hashes
    output = ROOT / 'training/reference_repair'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'directed_transfer.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / 'directed_transfer.npz', **arrays, com=com,
                        planned_qpos=poses, planned_times=coarse_time, commands=commands)
    print('DIRECTED_TRANSFER', json.dumps(report), flush=True)


def check_command_bound():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint name="pivot" type="hinge" axis="1 0 0"/>
      <inertial mass="1" pos="0 .1 0" diaginertia=".01 .01 .01"/></body>
      </worldbody><actuator><position joint="pivot" kp=".2" ctrlrange="-.1 .1"
      forcerange="-2 2"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.
    free = contact_allocation(model, data, [], [], .6)
    bound = contact_allocation(model, data, [], [], .6, command_limits=True)
    assert free['feasible'] and free['min_peak_torque_ratio'] < 1 and not bound['feasible']
    model.actuator_gainprm[0, 0] = 20.
    model.actuator_biasprm[0, 1] = -20.
    assert contact_allocation(model, data, [], [], .6, command_limits=True)['feasible']
    # A moving joint must be driven against dry friction; the optimistic test
    # can instead choose zero friction and zero motor effort in this example.
    model.opt.gravity[:] = 0.
    model.dof_frictionloss[:] = .3
    for velocity in (-.2, .2):
        data.qvel[:] = velocity
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.
        physical = contact_allocation(model, data, [], [], .6, physical_friction=True)
        np.testing.assert_allclose(physical['motor_torque_nm'], [.3 * np.sign(velocity)], atol=1e-10)
    print('PASS: torque-cap feasibility differs from low-gain position-command feasibility', flush=True)


def check_cached_constants(model):
    data = mujoco.MjData(model)
    residuals = []
    for refresh in (False, True):
        if refresh:
            mujoco.mj_setConst(model, data)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        mujoco.mj_forward(model, data)
        inertial = np.zeros(model.nv)
        mujoco.mj_mulM(model, data, inertial, data.qacc)
        residuals.append(float(abs(inertial + data.qfrc_bias - data.qfrc_passive
                                   - data.qfrc_actuator - data.qfrc_constraint).max()))
    assert residuals[0] > 1e-3 and residuals[1] < 1e-9
    return dict(stale_max_force_residual=residuals[0], refreshed_max_force_residual=residuals[1],
                scope='same airborne home pose and commands, modified armature, no contact-solver tolerance')


def audit(model, candidate, p, *, stride=4, include_projected=True, physical_friction=False):
    meta = json.loads(str(candidate['metadata']))
    assert meta['source_hashes'] == source_hashes()
    frames, base = candidate['dense_frames'], candidate['dense_base']
    cols = [meta['joint_names'].index(model.joint(int(j)).name) for j in model.actuator_trnid[:, 0]]
    adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    q = np.zeros((len(frames), model.nq))
    q[:, :3] = base[:, :3, 3]
    for i, rotation in enumerate(base[:, :3, :3]):
        mujoco.mju_mat2Quat(q[i, 3:7], rotation.ravel())
    q[:, adr] = frames[:, cols]
    dt = 1 / meta['sample_fps']
    vertices, _ = foot_geometry()
    sites = [model.site(side + '_foot').id for side in ('left', 'right')]
    data = mujoco.MjData(model)
    rows = []
    for i in range(4, len(frames) - 4, stride):
        data.qpos[:] = q[i]
        fwd, bwd = np.zeros(model.nv), np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, fwd, dt, q[i], q[i+1])
        mujoco.mj_differentiatePos(model, bwd, dt, q[i], q[i-1])
        data.qvel[:] = (fwd - bwd) / 2
        acceleration = (fwd + bwd) / dt
        if 'dense_joint_acceleration' in candidate.files:
            dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
            data.qvel[dofs] = frames[i, 16 + np.asarray(cols)]
            acceleration[dofs] = candidate['dense_joint_acceleration'][i, cols]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        data.qacc[:] = acceleration
        supports = frames[i, 32:34].astype(bool)
        row = dict(reference_time_s=i * dt, support=supports.astype(int).tolist(),
                   com_world_mm=(1000 * data.subtree_com[1]).tolist(), results={})
        inertial = np.zeros(model.nv)
        mujoco.mj_mulM(model, data, inertial, data.qacc)
        required = inertial + data.qfrc_bias - data.qfrc_passive
        force = required[:3]
        moment = data.xmat[1].reshape(3, 3) @ required[3:6]
        zmp = ground_zmp(data.qpos[:3], force, moment)
        row['required_ground_force_n'] = force.tolist()
        row['required_zmp_xy_mm'] = (zmp * 1000).tolist()
        row['required_vertical_twist_nm'] = float(
            moment[2] - np.cross(np.r_[zmp, 0.] - data.qpos[:3], force)[2])
        if 'dense_zmp_target' in candidate.files:
            row['planner_zmp_target_mm'] = (candidate['dense_zmp_target'][i] * 1000).tolist()
        row['foot_roll_pitch_deg'] = []
        row['support_low_vertices_xy_mm'] = []
        for site, v, support in zip(sites, vertices, supports):
            R = data.site_xmat[site].reshape(3, 3)
            row['foot_roll_pitch_deg'].append(np.rad2deg([np.arctan2(R[2,1], R[2,2]),
                                                          np.arctan2(-R[2,0], np.hypot(R[2,1], R[2,2]))]).tolist())
            world = v @ R.T + data.site_xpos[site]
            row['support_low_vertices_xy_mm'].append((world[world[:,2] <= .0002,:2] * 1000).tolist() if support else [])
        for actual in ((True, False) if include_projected else (True,)):
            points, bodies, heights = support_points(model, data, sites, vertices, supports,
                                                     actual_contacts=actual)
            for limits in (False, True):
                key = ('floor' if actual else 'projected') + ('_command_bounds' if limits else '_torque_only')
                row['results'][key] = contact_allocation(model, data, points, bodies, .6,
                                                        command_limits=limits,
                                                        physical_friction=physical_friction)
            row['whole_sole_clearance_mm'] = (1000 * np.array(heights)).tolist()
        rows.append(row)
    summary = {}
    for key in rows[0]['results']:
        valid = [r['results'][key] for r in rows]
        feasible = [r for r in valid if r['feasible']]
        ratios = [r['min_peak_torque_ratio'] for r in feasible]
        summary[key] = dict(samples=len(rows), feasible=len(feasible),
                            within_cap=sum(r['min_peak_torque_ratio'] <= 1. for r in feasible),
                            min_peak_ratio_range=[min(ratios), max(ratios)] if ratios else None)
    print('REFERENCE_AUDIT', json.dumps(summary), flush=True)
    return dict(method='raw Placo poses, central derivatives, unilateral contact-force LP',
                sample_interval_s=dt * stride, input_sample_interval_s=dt,
                scope=('moving dry friction opposes velocity; static friction remains an interval; '
                       'necessary conditions only; no tracking or walking certificate'
                       if physical_friction else
                       'optimistic necessary conditions; no force tracking or walking certificate'),
                moving_friction_opposes_velocity=physical_friction,
                summary=summary, rows=rows)


def plot(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows = report['reference_audit']['rows']
    t = np.array([r['reference_time_s'] for r in rows])
    for side, color in zip(range(2), ('tab:blue', 'tab:orange')):
        low, high = [], []
        for row in rows:
            v = np.array(row['support_low_vertices_xy_mm'][side])
            low.append(v[:, 1].min() if len(v) else np.nan)
            high.append(v[:, 1].max() if len(v) else np.nan)
        axes[0, 0].fill_between(t, low, high, color=color, alpha=.6,
                               label=('Left' if side == 0 else 'Right') + ' low contact edge')
        axes[0, 1].plot(t, [r['foot_roll_pitch_deg'][side][0] for r in rows], color=color,
                        label=('Left' if side == 0 else 'Right') + ' sole roll')
    axes[0, 0].plot(t, [r['required_zmp_xy_mm'][1] for r in rows], color='black', label='Required ZMP')
    axes[0, 0].set_ylabel('Lateral position (mm)')
    axes[0, 0].set_title('Reference contact incompatibility (perfect tracking assumed)')
    axes[0, 1].set_ylabel('Sole roll (deg)')
    axes[0, 1].set_title('Upright torso / free sole-roll reference')
    for name in report['cases']:
        with np.load(output / ('diagnostic_' + name + '.npz')) as arrays:
            axes[1, 0].plot(arrays['time'] - 1.5, arrays['tilt_deg'], label=name)
    axes[1, 0].set_title('Removing actuator limits does not rescue liftoff')
    axes[1, 0].set_ylabel('Actual base tilt (deg)')
    standing = output / 'standing_diagnosis.json'
    if standing.exists():
        for name in json.loads(standing.read_text())['cases']:
            with np.load(output / ('diagnostic_' + name + '.npz')) as arrays:
                axes[1, 1].plot(arrays['time'], arrays['tilt_deg'], label=name)
    axes[1, 1].set_title('Same home preload; hypothetical gain changes, not hardware settings')
    axes[1, 1].set_ylabel('Actual base tilt (deg)')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
        ax.set_xlabel('Reference time (s)' if ax != axes[1, 1] else 'Simulation time (s)')
    fig.tight_layout()
    fig.savefig(output / 'diagnosis.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planning-only', action='store_true')
    parser.add_argument('--standing-only', action='store_true')
    parser.add_argument('--hip-transfer-only', action='store_true')
    parser.add_argument('--left-lift-transfer', action='store_true',
                        help='Placo COM-right/S02-inward physical transfer diagnostic before left lift')
    parser.add_argument('--hip-side', choices=('left', 'right'),
                        help='one safe 20-degree hip probe; left=S07, right=S02')
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--live-directory', type=Path,
                        help='publish the planning or hip diagnostic as live.html/live.json')
    args = parser.parse_args()
    if args.live_directory and not (args.planning_only or args.hip_transfer_only or args.left_lift_transfer):
        parser.error('--live-directory requires a planning or transfer diagnostic')
    if args.hip_side and not args.hip_transfer_only:
        parser.error('--hip-side requires --hip-transfer-only')
    if args.left_lift_transfer:
        if args.hip_transfer_only or args.planning_only or args.standing_only:
            parser.error('--left-lift-transfer is a separate physical diagnostic')
        directed_transfer_probe(args.live_directory)
        return
    if args.hip_transfer_only:
        hip_transfer_probe(args.live_directory, args.hip_side)
        return
    output = ROOT / 'training/calibrated_actuator'
    candidate_path = args.candidate or output / 'planning_baseline.npz'
    previous = json.loads((output / 'assessment.json').read_text())
    assert previous['source_hashes'] == source_hashes()
    protected = [URDF, MODEL, SCENE, DATA / 'reference.json', DATA / 'polynomial_coefficients.pkl',
                 ROOT.parents[1] / 'cad/local_fix/open_duck_mini_upper_refit.step']
    before = {str(path): digest(path) for path in protected}
    p = previous['parameters']
    with np.load(ROOT / 'training/reference_replay/uncapped.npz') as raw:
        targets, contacts = raw['target'].copy(), raw['desired_contact'].copy()
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    main_servos = set_identified(model, p)
    names = [model.actuator(i).name for i in range(model.nu)]
    offset = np.deg2rad([previous['home_command_adjustment_deg'][name] for name in names])
    model.key('home').ctrl[:] += offset
    constant_check = check_cached_constants(model)
    commands = targets + offset
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    feet = [int(model.site_bodyid[model.site(s + '_foot').id]) for s in ('left', 'right')]
    ref = json.loads((DATA / 'reference.json').read_text())['reward_reference']
    vertices = []
    for side, body, v in zip(('left', 'right'), feet, ref['foot_vertices_in_site']):
        site = model.site(side + '_foot').id
        world = np.array(v) @ data.site_xmat[site].reshape(3, 3).T + data.site_xpos[site]
        vertices.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
    cases = {}
    modes = ([] if args.planning_only else ['static_gain_1', 'static_gain_2', 'static_gain_4']
             if args.standing_only else
             ['no_command_clip', 'no_torque_cap', 'neither_cap_nor_clip', 'higher_gain_same_caps'])
    for name in modes:
        set_identified(model, p)
        model.actuator_ctrllimited[main_servos] = True
        model.actuator_forcelimited[main_servos] = True
        model.jnt_actfrclimited[model.actuator_trnid[main_servos, 0]] = False
        changed_commands = commands
        if name in ('no_command_clip', 'neither_cap_nor_clip'):
            model.actuator_ctrllimited[main_servos] = False
        if name in ('no_torque_cap', 'neither_cap_nor_clip'):
            model.actuator_forcelimited[main_servos] = False
        if name == 'higher_gain_same_caps':
            model.actuator_gainprm[main_servos, 0] = 13.37
            model.actuator_biasprm[main_servos, 1] = -13.37
            changed_commands = targets
        home_only = name.startswith('static_gain_')
        if home_only:
            factor = float(name.rsplit('_', 1)[1])
            legs = [i for i in main_servos if any(suffix in names[i] for suffix in ('hip', 'knee', 'ankle'))]
            model.actuator_gainprm[legs, 0] *= factor
            model.actuator_biasprm[legs, 1] *= factor
            adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
            home_q = model.key('home').qpos[adr]
            home_ctrl = targets[0] + offset
            home_ctrl[legs] = home_q[legs] + (home_ctrl - home_q)[legs] / factor
            changed_commands = np.tile(home_ctrl, (round(8 / model.opt.timestep), 1))
        print('START', name, flush=True)
        desired = np.ones((len(changed_commands), 2)) if home_only else contacts
        cases[name], arrays = replay(model, changed_commands, desired, feet, vertices,
                                     .5 if home_only else 1., 0. if home_only else .5)
        np.savez_compressed(output / ('diagnostic_' + name + '.npz'), **arrays)
        print('RESULT', json.dumps(dict(name=name, stop=cases[name]['stop'],
                                       seconds=cases[name]['simulated_seconds'], feet=cases[name]['feet'])), flush=True)
    set_identified(model, p)
    model.actuator_ctrllimited[main_servos] = True
    model.actuator_forcelimited[main_servos] = True
    check_command_bound()
    check_contact_geometry()
    if args.standing_only:
        audit_report = None
    else:
        with np.load(candidate_path) as candidate:
            audit_report = audit(model, candidate, p, physical_friction=True)
            if args.planning_only:
                frames = candidate['dense_frames']
                meta = json.loads(str(candidate['metadata']))
                cols = [meta['joint_names'].index(model.joint(int(j)).name) for j in model.actuator_trnid[:, 0]]
                dt = model.opt.timestep
                frame_times = np.arange(len(frames)) / meta['sample_fps']
                times = np.arange(round((1.5 + frame_times[-1]) / dt)) * dt
                phase = np.clip(times - 1.5, 0., frame_times[-1])
                raw_targets = np.column_stack([np.interp(phase, frame_times, frames[:, col]) for col in cols])
                alpha = np.clip((times - 1.) / .5, 0., 1.)
                alpha = alpha * alpha * (3 - 2 * alpha)
                commands = model.key('home').ctrl + alpha[:, None] * (raw_targets + offset - model.key('home').ctrl)
                desired = frames[np.minimum((phase * meta['sample_fps']).astype(int), len(frames)-1), 32:34]
                desired[times < 1.5] = 1
                live = None
                if args.live_directory:
                    from visualize_evaluation import LiveView
                    live = LiveView(model, args.live_directory,
                                    'Placo准备候选 · 开环物理跟随 · 尚未完成抬脚修复')
                cases['candidate_replay'], arrays = replay(model, commands, desired, feet, vertices, 1., .5,
                                                           delay_s=p['transport_delay_s'], live=live)
                np.savez_compressed(output / ('planning_' + args.label + '_replay.npz'), **arrays)
                print('CANDIDATE_REPLAY', json.dumps(cases['candidate_replay']), flush=True)
    assert before == {str(path): digest(path) for path in protected}
    report = dict(source_hashes=source_hashes(), protected_files_unchanged=before,
                  script_sha256=digest(Path(__file__)), replay_sha256=digest(Path(__file__).with_name('replay_reference.py')),
                  allocation_sha256=digest(Path(__file__).with_name('assess_reference.py')),
                  candidate_sha256=digest(candidate_path), fit_sha256=previous['fit_sha256'],
                  cached_constant_check=constant_check,
                  cases=cases, reference_audit=audit_report,
                  note='Counterfactuals isolate causes; force/command changes are not accepted hardware settings')
    if args.standing_only:
        (output / 'standing_diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
    elif args.planning_only:
        dest = output / 'planning_diagnosis.json'
        history = json.loads(dest.read_text()) if dest.exists() else {}
        history[args.label] = report
        dest.write_text(json.dumps(history, indent=2) + '\n')
    else:
        (output / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
        plot(output, report)
    print('PASS: causal tests and source preservation', flush=True)


if __name__ == '__main__':
    main()
