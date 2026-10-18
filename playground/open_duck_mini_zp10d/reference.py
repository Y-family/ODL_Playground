"""Current-model Placo references and upstream-compatible polynomial fitting.

Uses the WalkTasks / repetitive-footstep / ZMP pattern-generator workflow from
apirrone/Open_Duck_reference_motion_generator, commit 3d7bc6fd80e6c35587b2a5279314d946cf63456b.
URDF XML is canonical; this module never regenerates it. References are kinematic,
not a certificate of contact, clearance, actuator torque or stable walking.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
URDF = ROOT / 'open_duck_mini_zp10d.urdf'
MODEL = ROOT / 'xmls/open_duck_mini_zp10d.xml'
SCENE = ROOT / 'xmls/scene_flat_terrain.xml'
DATA = ROOT / 'data'
FPS = 50


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (URDF, MODEL, SCENE)}


def inputs():
    model = ET.parse(MODEL)
    home = ET.parse(SCENE).find('.//key[@name="home"]')
    joints = [e.get('name') for e in model.findall('.//body/joint')]
    actuators = [e.get('name') for e in model.findall('actuator/position')]
    qpos = np.fromstring(home.get('qpos'), sep=' ')
    assert len(joints) == len(actuators) == 16
    return joints, actuators, qpos


def check_description():
    import mujoco
    import placo
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    robot = placo.HumanoidRobot(str(URDF))
    maximum = 0.
    for pose in range(33):
        mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
        if pose:
            joint = (pose - 1) // 2 + 1
            d.qpos[m.jnt_qposadr[joint]] = m.jnt_range[joint, (pose - 1) % 2]
        mujoco.mj_forward(m, d)
        transform = np.eye(4)
        transform[:3, :3] = d.xmat[1].reshape(3, 3)
        transform[:3, 3] = d.xpos[1]
        robot.set_T_world_fbase(transform)
        for joint in range(1, m.njnt):
            robot.set_joint(m.joint(joint).name, d.qpos[m.jnt_qposadr[joint]])
        robot.update_kinematics()
        for body in range(1, m.nbody):
            if m.body(body).name == 'floor':
                continue
            actual = robot.get_T_world_frame(m.body(body).name)
            maximum = max(maximum, float(np.max(abs(actual[:3, 3] - d.xpos[body]))),
                          float(np.max(abs(actual[:3, :3] - d.xmat[body].reshape(3, 3)))))
        np.testing.assert_allclose(robot.com_world(), d.subtree_com[1], atol=1e-10)
    assert maximum < 1e-10, maximum
    print(f'PASS: 33 poses, every rigid-body frame and COM; max error {maximum:.3g}', flush=True)


def foot_geometry():
    """Full collision-foot hulls in site coordinates and site-to-sole offsets."""
    import mujoco
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    sole_frames = {j.find('child').get('link'): j.find('origin')
                   for j in ET.parse(URDF).findall('joint') if j.get('name').endswith('_foot_frame')}
    vertices, offsets = [], []
    for side in ('left', 'right'):
        site = model.site(side + '_foot').id
        points = []
        for gid in np.flatnonzero((model.geom_bodyid == model.site_bodyid[site]) & (model.geom_contype != 0)):
            mid = model.geom_dataid[gid]
            local = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            world = local @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
            points.append((world - data.site_xpos[site]) @ data.site_xmat[site].reshape(3, 3))
        points = np.unique(np.concatenate(points), axis=0)
        vertices.append(points[ConvexHull(points).vertices])
        sole_pos = np.fromstring(sole_frames[model.site(site).name].get('xyz'), sep=' ')
        local_rotation = Rotation.from_quat(np.roll(model.site_quat[site], -1)).as_matrix()
        offsets.append((model.site_pos[site] - sole_pos) @ local_rotation)
    return vertices, np.asarray(offsets)


def generate(step_length=.01, support_duration=.2, foot_height=.015, seconds=6., feet_spacing=None):
    import placo
    names, _, home = inputs()
    robot = placo.HumanoidRobot(str(URDF))
    for name, value in zip(names, home[7:]):
        robot.set_joint(name, float(value))
    transform = np.eye(4)
    transform[:3, 3] = home[:3]
    transform[2, 3] -= .0002  # published home starts 0.2 mm above the floor
    robot.set_T_world_fbase(transform)
    robot.update_kinematics()
    vertices, offsets = foot_geometry()
    sole_vertices = [v + offset for v, offset in zip(vertices, offsets)]

    params = placo.HumanoidParameters()
    params.double_support_ratio = .5
    params.startend_double_support_ratio = 1.5
    params.single_support_duration = support_duration
    params.single_support_timesteps = 10
    params.planned_timesteps = 48
    params.replan_timesteps = 10
    params.walk_com_height = float(robot.com_world()[2])
    params.walk_trunk_pitch = 0.
    params.walk_foot_height = foot_height
    # Conservative support rectangle inside the current sole, in metres.
    params.foot_length = .08
    params.foot_width = .045
    params.feet_spacing = (float(robot.get_T_world_left()[1, 3] - robot.get_T_world_right()[1, 3])
                           if feet_spacing is None else feet_spacing)
    params.zmp_margin = .003
    params.foot_zmp_target_x = .009
    params.foot_zmp_target_y = 0.
    period = 2 * (support_duration + params.double_support_duration())
    solver = placo.KinematicsSolver(robot)
    solver.dt = .001
    solver.enable_joint_limits(True)
    solver.enable_velocity_limits(True)
    # Planning headroom for the continuous polynomial, below the unchanged 5.24 cap.
    robot.set_velocity_limits(4.)
    # Planning-only branch restriction, within the unchanged physical limits.
    robot.set_joint_limits('left_knee', .05, np.pi / 2)
    robot.set_joint_limits('right_knee', .05, np.pi / 2)
    tasks = placo.WalkTasks()
    tasks.initialize_tasks(solver, robot)
    tasks.left_foot_task.orientation().mask.set_axises('yz', 'local')
    tasks.right_foot_task.orientation().mask.set_axises('yz', 'local')
    tasks.left_foot_task.position().configure('left_foot_position', 'soft', 100.)
    tasks.right_foot_task.position().configure('right_foot_position', 'soft', 100.)
    upper = solver.add_joints_task()
    upper.set_joints({n: float(robot.get_joint(n)) for n in names
                     if not any(x in n for x in ('hip', 'knee', 'ankle'))})
    upper.configure('upper', 'soft', 1.)
    planner = placo.FootstepsPlannerRepetitive(params)
    planner.configure(step_length, 0., 0., int(np.ceil(seconds / support_duration)) + 6)
    feet = planner.plan(placo.HumanoidRobot_Side.left,
                        placo.flatten_on_floor(robot.get_T_world_left()),
                        placo.flatten_on_floor(robot.get_T_world_right()))
    supports = placo.FootstepsPlanner.make_supports(feet, 0., True, params.has_double_support(), True)
    trajectory = placo.WalkPatternGenerator(robot, params).plan(supports, robot.com_world(), 0.)
    frames, positions, base_transforms, feet_transforms, phases, residuals = [], [], [], [], [], []
    task_errors, sole_heights, origin_lifts, com_errors = [], [], [], []
    for tick in range(round(seconds / solver.dt)):
        t = tick * solver.dt
        tasks.update_tasks_from_trajectory(trajectory, t)
        # Compensate the remaining small orientation residual of the full hull.
        lifts = []
        for task, foot, hull in zip((tasks.left_foot_task, tasks.right_foot_task),
                                   (robot.get_T_world_left(), robot.get_T_world_right()), sole_vertices):
            lift = -np.min((hull @ foot[:3, :3].T)[:, 2])
            target = task.position().target_world.copy()
            target[2] += lift
            task.position().target_world = target
            lifts.append(lift)
        robot.update_kinematics()
        solver.solve(True)
        robot.update_kinematics()
        if tick % 5:
            continue
        left, right = robot.get_T_world_left().copy(), robot.get_T_world_right().copy()
        feet_transforms.append([left, right])
        base_transforms.append(robot.get_T_world_fbase().copy())
        positions.append([robot.get_joint(n) for n in names])
        com_errors.append(float(np.linalg.norm(robot.com_world() - tasks.com_task.target_world)))
        if trajectory.support_is_both(t):
            phases.append([1, 1])
        else:
            phases.append([1, 0] if str(trajectory.support_side(t)) == 'left' else [0, 1])
        residuals.append([left[2, 3], right[2, 3]])
        task_errors.append(max(np.linalg.norm(foot[:3, 3] - task.position().target_world)
                               for foot, task in zip((left, right), (tasks.left_foot_task, tasks.right_foot_task))))
        sole_heights.append([float((hull @ foot[:3, :3].T + foot[:3, 3])[:, 2].min())
                             for hull, foot in zip(sole_vertices, (left, right))])
        origin_lifts.append(lifts)
    positions = np.asarray(positions)
    sample_fps = 200
    velocities = np.gradient(positions, 1 / sample_fps, axis=0)
    transforms = np.asarray(base_transforms)
    linear = np.gradient(transforms[:, :3, 3], 1 / sample_fps, axis=0)
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_matrix(transforms[:, :3, :3])
    angular = (rotations[:-1].inv() * rotations[1:]).as_rotvec() * sample_fps
    angular = np.concatenate([angular[:1], angular])
    frames = np.concatenate([positions, velocities, phases, linear, angular], axis=1)
    assert frames.shape[1] == 40 and np.isfinite(frames).all()
    limits = {j.get('name'): np.fromstring(j.get('range'), sep=' ')
              for j in ET.parse(MODEL).findall('.//body/joint')}
    limit_error = max(float(np.maximum(limits[n][0] - positions[:, i],
                                      positions[:, i] - limits[n][1]).max())
                      for i, n in enumerate(names))
    stats = dict(period=period, step_length=step_length, support_duration=support_duration,
                 foot_height=foot_height, com_height=params.walk_com_height,
                 feet_spacing=params.feet_spacing,
                 max_joint_speed=float(abs(velocities).max()), max_limit_error=max(0., limit_error),
                 min_foot_frame_height=float(np.min(residuals)), source_hashes=source_hashes(),
                 max_feet_position_error=float(max(task_errors)),
                 max_com_position_error_m=max(com_errors),
                 support_orientation='free sole roll; upright trunk',
                 sample_fps=sample_fps, foot_origin_lift_max_m=np.max(origin_lifts, axis=0).tolist(),
                 planning_joint_speed_limit_rad_s=4.,
                 min_whole_foot_height_m=np.min(sole_heights, axis=0).tolist(),
                 whole_foot_target='planned height of lowest oriented collision-foot vertex',
                 joint_names=names, kind='kinematic_candidate_not_dynamics_acceptance')
    print(json.dumps(stats, indent=2), flush=True)
    # Scratch candidates never replace a previously accepted reference on failure.
    out = Path('/tmp/zp10d-reference')
    out.mkdir(exist_ok=True)
    stride = sample_fps // FPS
    np.savez(out / 'candidate.npz', frames=frames[::stride], qpos=positions[::stride], base=transforms[::stride],
             feet=np.asarray(feet_transforms)[::stride], contacts=np.asarray(phases)[::stride],
             dense_frames=frames, dense_base=transforms, dense_heights=np.asarray(sole_heights),
             metadata=json.dumps(stats))
    return stats


def fit_candidate():
    """Fit one settled forward cycle, checking actual float32 reader arithmetic."""
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp

    candidate = np.load('/tmp/zp10d-reference/candidate.npz')
    meta = json.loads(str(candidate['metadata']))
    if 'dense_frames' not in candidate.files or meta.get('sample_fps') != 200:
        raise ValueError('Regenerate the dense whole-foot-corrected Placo candidate before fitting')
    if meta['source_hashes'] != source_hashes():
        raise ValueError('Candidate does not match the current model')
    if meta['max_limit_error'] > 1e-6 or meta['max_feet_position_error'] > .0002:
        raise ValueError('Reference kinematic limits/foot targets have not passed')
    count = round(meta['period'] * FPS)
    if abs(count / FPS - meta['period']) > 1e-8:
        raise ValueError('Choose a period that is an integer number of control frames')
    frames = candidate['frames']
    start = ((len(frames) // count) - 2) * count
    assert start >= 3 * count, 'Need at least five complete periods'
    cycle = frames[start:start + count]
    following = frames[start + count:start + 2 * count]
    periodic_error = float(abs(cycle[:, :16] - following[:, :16]).max())
    if periodic_error > .02:
        raise ValueError(f'Kinematic cycle has not settled: {periodic_error} rad')
    stride = meta['sample_fps'] // FPS
    dense_start, dense_count = start * stride, count * stride
    x = np.arange(dense_count + 1) / dense_count
    dense_cycle = candidate['dense_frames'][dense_start:dense_start + dense_count]
    y = np.concatenate([dense_cycle, dense_cycle[:1]])
    spline = CubicSpline(x, y, bc_type='periodic')
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    names, _, _ = inputs()
    addresses = [model.joint(name).qposadr[0] for name in names]
    sites = [model.site(side + '_foot').id for side in ('left', 'right')]
    vertices, _ = foot_geometry()
    # Check between control ticks too, using the original 1 ms solver cadence.
    check_x = np.linspace(0., 1., round(meta['period'] / .001) + 1)
    raw_base = candidate['dense_base'][dense_start:dense_start + dense_count + 1]
    base_rotation = Slerp(x, Rotation.from_matrix(raw_base[:, :3, :3]))(check_x).as_matrix()
    base_position = CubicSpline(x, raw_base[:, :3, 3])(check_x)
    target_heights = CubicSpline(x, np.concatenate([
        candidate['dense_heights'][dense_start:dense_start + dense_count],
        candidate['dense_heights'][dense_start:dense_start + 1]]), bc_type='periodic')(check_x)
    # Short cubic pieces preserve the periodic position/velocity boundary and
    # avoid high-degree oscillation. The readers evaluate only the active piece.
    degree = 3
    coeff = spline.c.copy()
    coeff[0, :, 16:32] = 0.
    coeff[1, :, 16:32] = 3 * coeff[0, :, :16] / meta['period']
    coeff[2, :, 16:32] = 2 * coeff[1, :, :16] / meta['period']
    coeff[3, :, 16:32] = coeff[2, :, :16] / meta['period']
    predicted = spline(check_x)
    # Reproduce the float32 Horner arithmetic used by the actual training reader.
    check32 = check_x.astype(np.float32)
    segment = np.minimum((check32 * dense_count).astype(int), dense_count - 1)
    local = (check32 - segment.astype(np.float32) / np.float32(dense_count))[:, None]
    c32 = coeff.astype(np.float32)[:, segment]
    quantized = ((c32[0] * local + c32[1]) * local + c32[2]) * local + c32[3]
    fit_error = float(abs(spline(x)[:, :16] - y[:, :16]).max())
    quant_error = float(abs(predicted[:, :16] - quantized[:, :16]).max())
    fitted_heights = []
    for i, joints in enumerate(quantized[:, :16]):
        data.qpos[:3] = base_position[i]
        mujoco.mju_mat2Quat(data.qpos[3:7], base_rotation[i].ravel())
        data.qpos[addresses] = joints
        mujoco.mj_kinematics(model, data)
        fitted_heights.append([float((v @ data.site_xmat[s].reshape(3, 3).T
                                     + data.site_xpos[s])[:, 2].min()) for s, v in zip(sites, vertices)])
    fitted_heights = np.asarray(fitted_heights)
    height_error = float(abs(fitted_heights - target_heights).max())
    min_height = float(fitted_heights.min())
    limit_error = max(float(np.maximum(model.joint(n).range[0] - quantized[:, j],
                                      quantized[:, j] - model.joint(n).range[1]).max())
                      for j, n in enumerate(names))
    # Include every interior extremum of each quadratic joint-speed polynomial.
    a, b = coeff[0, :, :16], coeff[1, :, :16]
    extrema = np.clip(np.divide(-b, 3 * a, out=np.zeros_like(a), where=abs(a) > 1e-12), 0., 1 / dense_count)
    speeds = np.stack([abs((3 * a * t + 2 * b) * t + coeff[2, :, :16]) / meta['period']
                       for t in (0., extrema, 1 / dense_count)])
    max_speed = float(speeds.max())
    closure_error = float(abs(quantized[0, :16] - quantized[-1, :16]).max())
    velocity_closure_error = float(abs(quantized[0, 16:32] - quantized[-1, 16:32]).max())
    fit_stats = dict(degree=degree, segments=dense_count, joint_error_rad=fit_error,
                     height_error_m=height_error, min_height_m=min_height, max_speed_rad_s=max_speed,
                     closure_rad=closure_error, velocity_closure_rad_s=velocity_closure_error)
    print(json.dumps(fit_stats), flush=True)
    if not (quant_error <= 2e-6 and height_error <= .0002 and min_height >= -.0002
            and limit_error <= 1e-6 and max_speed <= 5.24
            and closure_error <= 2e-6 and velocity_closure_error <= .001):
        raise ValueError(f'Polynomial/foot geometry did not pass: {fit_stats}, quantization={quant_error}')
    velocity = round(2 * meta['step_length'] / meta['period'], 8)
    record = dict(basis='periodic_cubic', coefficients={f'dim_{i}': coeff[:, :, i].tolist() for i in range(40)},
                  period=meta['period'], fps=FPS, frame_offsets={}, start_offset=start,
                  nb_steps_in_period=count, startend_double_support_ratio=1.5)
    references = {f'{velocity}_0.0_0.0': record}
    # The zero-command reference is actual accepted geometric home, not a legacy pose.
    names, actuators, home = inputs()
    standing = np.concatenate([home[7:], np.zeros(16), [1., 1.], np.zeros(6)])
    standing_coeff = np.zeros_like(coeff)
    standing_coeff[3] = standing
    references['0.0_0.0_0.0'] = dict(record, coefficients={
        f'dim_{i}': standing_coeff[:, :, i].tolist() for i in range(40)})
    ctrl = np.fromstring(ET.parse(SCENE).find('.//key').get('ctrl'), sep=' ')
    offset = np.array([ctrl[actuators.index(n)] for n in names])
    scale = np.maximum(np.max(abs(cycle[:, :16] - offset), axis=0) + .15, .25)
    scale = [float(scale[names.index(n)]) for n in actuators]
    meta.update(polynomial_degree=degree, position_fit_error_rad=fit_error,
                candidate_sha256=hashlib.sha256(Path('/tmp/zp10d-reference/candidate.npz').read_bytes()).hexdigest(),
                float32_position_error_rad=quant_error, cycle_position_error_rad=periodic_error,
                fitted_cycle_check=dict(**fit_stats, sample_interval_s=.001,
                                        max_limit_error_rad=max(0., limit_error),
                                        joint_velocities='analytic derivative of fitted joint positions'),
                commands=[[0., 0., 0.], [velocity, 0., 0.]], action_scale=scale,
                scope='standing and slow forward reference only; not trained or dynamics-accepted')
    DATA.mkdir(exist_ok=True)
    with (DATA / 'polynomial_coefficients.pkl').open('wb') as out:
        pickle.dump(references, out)
    (DATA / 'reference.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)
    prepare_reward_reference()


def prepare_reward_reference():
    """Bind exact Placo support labels and whole-foot heights to the fitted cycle."""
    import mujoco
    from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

    candidate = np.load('/tmp/zp10d-reference/candidate.npz')
    meta = json.loads((DATA / 'reference.json').read_text())
    if meta.get('candidate_sha256') is not None and meta['candidate_sha256'] != hashlib.sha256(
            Path('/tmp/zp10d-reference/candidate.npz').read_bytes()).hexdigest():
        raise ValueError('Scratch candidate does not match the fitted reference')
    if json.loads(str(candidate['metadata']))['source_hashes'] != source_hashes():
        raise ValueError('Scratch Placo reference does not match the current model')
    if meta['source_hashes'] != source_hashes():
        raise ValueError('Fitted reference does not match the current model')
    with (DATA / 'polynomial_coefficients.pkl').open('rb') as stream:
        records = pickle.load(stream)
    record = next(r for key, r in records.items() if not key.startswith('0.0_'))
    start, count = record['start_offset'], record['nb_steps_in_period']
    contacts = candidate['contacts'][start:start + count].astype(int)
    assert contacts.shape == (count, 2) and np.all(contacts.sum(axis=1) >= 1)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    sites = [model.site(side + '_foot').id for side in ('left', 'right')]
    vertices, site_offsets = foot_geometry()
    feet = candidate['feet'][start:start + count]
    heights = np.stack([
        np.min(((v + site_offsets[side]) @ feet[:, side, :3, :3].transpose(0, 2, 1)
                + feet[:, side, None, :3, 3])[..., 2], axis=1)
        for side, v in enumerate(vertices)], axis=1)
    loads = contacts / contacts.sum(axis=1, keepdims=True)
    single = np.flatnonzero(contacts.sum(axis=1) == 1)
    for i in np.flatnonzero(contacts.sum(axis=1) == 2):
        before = min(((i - int(j)) % count, int(j)) for j in single)
        after = min(((int(j) - i) % count, int(j)) for j in single)
        fraction = before[0] / (before[0] + after[0])
        fraction = fraction * fraction * (3 - 2 * fraction)
        loads[i] = (1 - fraction) * contacts[before[1]] + fraction * contacts[after[1]]
    reference = PolyReferenceMotion(str(DATA / 'polynomial_coefficients.pkl'))
    names, _, _ = inputs()
    addresses = [model.joint(name).qposadr[0] for name in names]
    fitted_heights, frame_error = [], 0.
    velocity = meta['commands'][1][0]
    for i in range(count):
        transform = candidate['base'][start + i]
        data.qpos[:3] = transform[:3, 3]
        mujoco.mju_mat2Quat(data.qpos[3:7], transform[:3, :3].ravel())
        data.qpos[addresses] = candidate['qpos'][start + i]
        mujoco.mj_kinematics(model, data)
        expected_sites = feet[i, :, :3, 3] + np.einsum('nij,nj->ni', feet[i, :, :3, :3], site_offsets)
        frame_error = max(frame_error, float(abs(data.site_xpos[sites] - expected_sites).max()))
        data.qpos[addresses] = reference.get_reference_motion(velocity, 0., 0., i)[:16]
        np.testing.assert_allclose(data.qpos[addresses], candidate['qpos'][start + i], atol=2e-6, rtol=0.,
                                   err_msg='Scratch candidate does not match the fitted reference')
        mujoco.mj_kinematics(model, data)
        fitted_heights.append([float((v @ data.site_xmat[s].reshape(3, 3).T
                                     + data.site_xpos[s])[:, 2].min()) for s, v in zip(sites, vertices)])
    assert frame_error < 1e-8, frame_error
    # Placo's sole origin is on the floor at home; foot roll can put a corner
    # below that plane. Express swing clearance relative to the supporting sole,
    # and demand ground height for each stance foot rather than rewarding penetration.
    floor_offset = np.min(np.where(contacts, heights, np.inf), axis=1)
    target_heights = np.where(contacts, 0., np.maximum(heights - floor_offset[:, None], 0.))
    reward_reference = dict(
        reference_sha256=hashlib.sha256((DATA / 'polynomial_coefficients.pkl').read_bytes()).hexdigest(),
        contacts=contacts.tolist(), load_fraction=loads.tolist(),
        whole_foot_height_m=target_heights.tolist(), foot_vertices_in_site=[v.tolist() for v in vertices],
        planned_sole_to_mj_site_m=np.asarray(site_offsets).tolist(),
        raw_peak_clearance_m=heights.max(axis=0).tolist(),
        raw_min_clearance_m=heights.min(axis=0).tolist(),
        ground_aligned_peak_clearance_m=target_heights.max(axis=0).tolist(),
        stance_floor_offset_m=floor_offset.tolist(),
        fitted_peak_clearance_m=np.max(fitted_heights, axis=0).tolist(),
        fitted_min_clearance_m=np.min(fitted_heights, axis=0).tolist(),
        fitted_stance_max_abs_height_m=float(np.max(abs(np.asarray(fitted_heights)[contacts.astype(bool)]))),
        kinematic_base_transforms=candidate['base'][start:start + count].tolist(),
        max_raw_site_position_error_m=frame_error,
        scope='kinematic reference audit; force tracking and walking not certified')
    meta['reward_reference'] = reward_reference
    (DATA / 'reference.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({k: v for k, v in reward_reference.items()
                      if k not in ('contacts', 'load_fraction', 'whole_foot_height_m',
                                   'foot_vertices_in_site', 'stance_floor_offset_m',
                                   'kinematic_base_transforms')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--reward-reference', action='store_true')
    parser.add_argument('--step-length', type=float, default=.01)
    parser.add_argument('--support-duration', type=float, default=.2)
    parser.add_argument('--foot-height', type=float, default=.015)
    parser.add_argument('--seconds', type=float, default=6.)
    parser.add_argument('--feet-spacing', type=float, default=None,
                        help='planned landing spacing; initial published home remains unchanged')
    parser.add_argument('--data-dir', type=Path,
                        help='fit output (default: /tmp/zp10d-reference/data); metadata input otherwise')
    args = parser.parse_args()
    DATA = args.data_dir or (Path('/tmp/zp10d-reference/data') if args.fit else DATA)
    if args.check:
        check_description()
    elif args.reward_reference:
        prepare_reward_reference()
    elif args.fit:
        fit_candidate()
    else:
        generate(args.step_length, args.support_duration, args.foot_height, args.seconds, args.feet_spacing)
