"""Current-model Placo references and upstream-compatible polynomial fitting.

Uses the WalkTasks / repetitive-footstep / ZMP pattern-generator workflow from
apirrone/Open_Duck_reference_motion_generator, commit 3d7bc6fd80e6c35587b2a5279314d946cf63456b.
URDF XML is canonical; this module never regenerates it. References are kinematic,
not a certificate of contact, clearance, actuator torque or stable walking.
"""

import argparse
import hashlib
import os
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
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (URDF, MODEL, SCENE)}
    evidence = ROOT.parents[1] / 'cad/local_fix'
    published = json.loads((evidence / 'open_duck_mini_zp10d_mass_properties.json').read_text())
    for path, key in ((evidence / 'open_duck_mini_upper_refit.step', 'source_step_sha256'),
                      (MODEL, 'mjcf_sha256'), (SCENE, 'scene_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != published[key]:
            raise ValueError(f'CAD/model synchronization incomplete: {path.name}')
    return hashes


def inputs():
    model = ET.parse(MODEL)
    home = ET.parse(SCENE).find('.//key[@name="home"]')
    joints = [e.get('name') for e in model.findall('.//body/joint')]
    actuators = [e.get('name') for e in model.findall('actuator/position')]
    qpos = np.fromstring(home.get('qpos'), sep=' ')
    assert len(joints) == len(actuators) == 16
    return joints, actuators, qpos


def apply_reference_stance(model, metadata):
    """Use the planned inward standing pose in memory; keep CAD/XML unchanged."""
    if not metadata.get('standing_inward_deg', 0.):
        return
    import mujoco
    home = model.key('home')
    addresses = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    preload = home.ctrl.copy() - home.qpos[addresses]
    pose = home.qpos.copy()
    transform = np.asarray(metadata['standing_base_transform'])
    np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-10)
    pose[:3] = transform[:3, 3]
    mujoco.mju_mat2Quat(pose[3:7], transform[:3, :3].ravel())
    for name, value in zip(metadata['joint_names'], metadata['standing_joint_positions'], strict=True):
        joint = model.joint(name)
        if not np.isfinite(value) or not joint.range[0] <= value <= joint.range[1]:
            raise ValueError(f'Planned standing pose exceeds {name} limits')
        pose[joint.qposadr] = value
    targets = pose[addresses] + preload
    if (np.any(targets < model.actuator_ctrlrange[:, 0])
            or np.any(targets > model.actuator_ctrlrange[:, 1])):
        raise ValueError('Standing motor preload exceeds command limits')
    home.qpos[:], home.ctrl[:] = pose, targets


def check_description():
    a, b = np.array([0., 0., 0.]), np.array([.01, .02, 0.])
    np.testing.assert_allclose(swing_position(a, b, 0., .003), a)
    np.testing.assert_allclose(swing_position(a, b, 1., .003), b)
    np.testing.assert_allclose(swing_position(a, b, .5, .003), [.005, .01, .003])
    epsilon = 1e-4
    for u in (0., 1.):
        minus = swing_position(a, b, u - epsilon, .003)
        centre = swing_position(a, b, u, .003)
        plus = swing_position(a, b, u + epsilon, .003)
        assert np.max(abs((plus - minus) / (2 * epsilon))) < 1e-6
        assert np.max(abs((plus - 2 * centre + minus) / epsilon**2)) < 1e-3
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


def swing_position(start, end, u, height):
    """C2 foot translation: zero velocity and acceleration at either contact."""
    u = float(np.clip(u, 0., 1.))
    blend = u**3 * (10 - 15*u + 6*u*u)
    position = (1 - blend) * start + blend * end
    position[2] += height * 64 * u**3 * (1-u)**3
    return position


def generate(step_length=.0025, support_duration=.16, foot_height=.0025, seconds=6., feet_spacing=None,
             *, double_support_ratio=.5625, startend_double_support_ratio=8., contact_aware=True,
             com_lateral_scale=1., com_lateral_smoothing=0.,
             flat_feet=False, exact_com=False, exact_feet=False, zmp_offset_x=None,
             allow_trunk_pitch=False, preparation_steps=0, transfer_crouch_m=.0038,
             yaw_momentum_weight=None, support_roll_weight=None, standing_inward_deg=0.,
             reduced_pendulum_height_m=.35, walking_crouch_m=0., com_height_weight=None,
             com_correction=None, roll_from_com_lever_arm_m=None,
             smooth_support_roll=False, swing_com_rise_m=0.):
    import placo
    source_hashes()
    vertices, offsets = foot_geometry()
    sole_vertices = [v + offset for v, offset in zip(vertices, offsets)]
    check_description()
    names, _, home = inputs()
    robot = placo.HumanoidRobot(str(URDF))
    for name, value in zip(names, home[7:]):
        robot.set_joint(name, float(value))
    if not 0. <= standing_inward_deg <= 5.:
        raise ValueError('Standing inward angle must be between 0 and 5 degrees per leg')
    for side, sign in (('left', 1), ('right', -1)):
        joint = side + '_hip_roll'
        robot.set_joint(joint, robot.get_joint(joint) + sign * np.deg2rad(standing_inward_deg))
    transform = np.eye(4)
    transform[:3, 3] = home[:3]
    robot.set_T_world_fbase(transform)
    robot.update_kinematics()
    # Ground the complete sole after changing stance; its frame origin alone
    # misses the low corner of a slightly banked foot.
    transform[2, 3] -= min(float((v @ foot[:3, :3].T + foot[:3, 3])[:, 2].min())
        for v, foot in zip(sole_vertices, (robot.get_T_world_left(), robot.get_T_world_right())))
    robot.set_T_world_fbase(transform)
    robot.update_kinematics()
    standing_joints = [float(robot.get_joint(n)) for n in names]
    initial_com = robot.com_world().copy()
    initial_feet = [robot.get_T_world_left().copy(), robot.get_T_world_right().copy()]
    correction = np.load(com_correction) if com_correction is not None else None
    if correction is not None:
        if correction['offset_xy_m'].shape != (len(correction['time_s']), 2):
            raise ValueError('COM correction must have one XY offset per timestamp')
        if not (np.isfinite(correction['offset_xy_m']).all()
                and np.max(abs(correction['offset_xy_m'])) <= .02
                and np.all(np.diff(correction['time_s']) > 0)):
            raise ValueError('Invalid or excessive COM correction')
        from scipy.interpolate import CubicSpline
        correction_spline = CubicSpline(correction['time_s'], correction['offset_xy_m'], bc_type='natural')
    if preparation_steps < 0 or int(preparation_steps) != preparation_steps:
        raise ValueError('Preparation steps must be a nonnegative integer')
    if preparation_steps and not contact_aware:
        raise ValueError('Planted-foot preparation requires contact-aware planning')
    if contact_aware:
        if com_lateral_scale != 1. or com_lateral_smoothing != 0. or flat_feet:
            raise ValueError('Contact-aware planning cannot post-scale/filter COM or pin both sole rolls')
        if support_duration <= .04:
            raise ValueError('Single-support duration must exceed the two 20 ms contact guards')
        exact_com = exact_feet = True

    params = placo.HumanoidParameters()
    params.double_support_ratio = double_support_ratio
    params.startend_double_support_ratio = startend_double_support_ratio
    params.single_support_duration = support_duration
    params.single_support_timesteps = 16 if contact_aware else 10
    params.planned_timesteps = 48
    params.replan_timesteps = 10
    # The reduced lateral model includes finite torso angular inertia. This is
    # a planning approximation, not the physical COM height or a model edit;
    # every candidate must pass full-model inverse dynamics before adoption.
    if not .1 <= reduced_pendulum_height_m <= 1.:
        raise ValueError('Reduced pendulum height must be between 0.1 and 1 metre')
    params.walk_com_height = reduced_pendulum_height_m if contact_aware else float(initial_com[2])
    params.walk_trunk_pitch = 0.
    params.walk_foot_height = foot_height
    # Conservative support rectangle inside the current sole, in metres.
    params.foot_length = .05 if contact_aware else .08
    params.foot_width = .020 if contact_aware else .045
    if contact_aware:
        from scipy.spatial import ConvexHull
        corners = np.array([[x, y] for x in (-params.foot_length/2, params.foot_length/2)
                             for y in (-params.foot_width/2, params.foot_width/2)])
        for hull in sole_vertices:
            low = hull[hull[:, 2] <= hull[:, 2].min() + .0002, :2]
            edges = ConvexHull(low).equations
            if np.max(corners @ edges[:, :2].T + edges[:, 2]) > 1e-6:
                raise ValueError('Planning support rectangle extends outside the actual low sole hull')
    params.feet_spacing = (float(robot.get_T_world_left()[1, 3] - robot.get_T_world_right()[1, 3])
                           if feet_spacing is None else feet_spacing)
    params.zmp_margin = .003
    params.foot_zmp_target_x = (float(zmp_offset_x) if zmp_offset_x is not None
                                else -.0125 if contact_aware else .009)
    params.foot_zmp_target_y = 0.
    if abs(params.foot_zmp_target_x) > params.foot_length / 2 - params.zmp_margin:
        raise ValueError('ZMP target must remain inside the planning sole with its margin')
    period = 2 * (support_duration + params.double_support_duration())
    solver = placo.KinematicsSolver(robot)
    solver.dt = .001
    solver.enable_joint_limits(True)
    solver.enable_velocity_limits(True)
    # Planning headroom for the continuous polynomial, below the unchanged 5.24 cap.
    planning_speed = 5.2 if contact_aware else 4.
    robot.set_velocity_limits(planning_speed)
    # Planning-only branch restriction, within the unchanged physical limits.
    robot.set_joint_limits('left_knee', .05, np.pi / 2)
    robot.set_joint_limits('right_knee', .05, np.pi / 2)
    tasks = placo.WalkTasks()
    tasks.initialize_tasks(solver, robot)
    tasks.left_foot_task.orientation().mask.set_axises('yz', 'local')
    tasks.right_foot_task.orientation().mask.set_axises('yz', 'local')
    if flat_feet:
        # Constrain the foot roll/pitch/yaw task while retaining the normal
        # upright-trunk task unless the legacy flat_feet candidate is requested.
        tasks.left_foot_task.orientation().mask.set_axises('xyz', 'local')
        tasks.right_foot_task.orientation().mask.set_axises('xyz', 'local')
        # No ankle-roll joint: release torso roll instead of banking the support sole.
        tasks.trunk_orientation_task.mask.set_axises('yz', 'local')
    if exact_com:
        tasks.com_task.configure('com', 'hard', 1.)
    if contact_aware:
        # Keep the loaded sole flat; allow torso roll and modest yaw. A hard
        # yaw angle otherwise asks the narrow sole for excessive friction twist.
        tasks.trunk_orientation_task.mask.set_axises('y', 'local')
        tasks.trunk_orientation_task.configure('trunk', 'soft' if allow_trunk_pitch else 'hard',
                                               100. if allow_trunk_pitch else 1.)
        momentum = solver.add_centroidal_momentum_task(np.zeros(3))
        momentum.mask.set_axises('z', 'world')
        momentum.configure('yaw_momentum', 'hard' if yaw_momentum_weight is None else 'soft',
                           1. if yaw_momentum_weight is None else yaw_momentum_weight)
        if yaw_momentum_weight is not None:
            heading = solver.add_orientation_task('base', np.eye(3))
            heading.mask.set_axises('z', 'world')
            heading.configure('heading', 'soft', 100.)
        sole_roll = [solver.add_orientation_task(side + '_foot', np.eye(3))
                     for side in ('left', 'right')]
        for task in sole_roll:
            task.mask.set_axises('x', 'local')
    tasks.left_foot_task.position().configure('left_foot_position', 'soft', 100.)
    tasks.right_foot_task.position().configure('right_foot_position', 'soft', 100.)
    if exact_feet:
        for side, task in zip(('left', 'right'), (tasks.left_foot_task, tasks.right_foot_task)):
            task.position().configure(side + '_foot_position', 'hard', 1.)
            task.orientation().configure(side + '_foot_orientation', 'hard', 1.)
    if com_height_weight is not None:
        tasks.com_task.mask.set_axises('xy', 'world')
        com_height_task = solver.add_com_task(robot.com_world())
        com_height_task.mask.set_axises('z', 'world')
        com_height_task.configure('com_height', 'soft', com_height_weight)
        exact_com = False
    if roll_from_com_lever_arm_m is not None:
        if not .03 <= roll_from_com_lever_arm_m <= .2:
            raise ValueError('Roll planning lever arm must be between 30 and 200 mm')
        tasks.com_task.mask.set_axises('x' if com_height_weight is not None else 'xz', 'world')
        tasks.trunk_orientation_task.mask.set_axises('xy', 'local')
        assert hasattr(tasks.trunk_orientation_task, 'R_world_frame')
        tasks.trunk_orientation_task.configure('roll_and_pitch', 'hard', 1.)
        exact_com = False
    upper = solver.add_joints_task()
    upper.set_joints({n: float(robot.get_joint(n)) for n in names
                     if not any(x in n for x in ('hip', 'knee', 'ankle'))})
    upper.configure('upper', 'hard' if contact_aware else 'soft', 1.)
    planner = placo.FootstepsPlannerRepetitive(params)
    planner.configure(step_length, 0., 0., int(np.ceil(seconds / support_duration)) + 6)
    feet = planner.plan(placo.HumanoidRobot_Side.left,
                        placo.flatten_on_floor(robot.get_T_world_left()),
                        placo.flatten_on_floor(robot.get_T_world_right()))
    supports = placo.FootstepsPlanner.make_supports(feet, 0., True, params.has_double_support(), True)
    generator = placo.WalkPatternGenerator(robot, params)
    generator.soft = False
    trajectory = generator.plan(supports, robot.com_world(), 0.)
    frames, positions, base_transforms, feet_transforms, phases, residuals = [], [], [], [], [], []
    task_errors, sole_heights, origin_lifts, com_errors, com_targets, zmp_targets = [], [], [], [], [], []
    initial_com_y = float(robot.com_world()[1])
    filtered_com_y = initial_com_y
    first_swing_start = None
    preparation_end = None
    preparation_offsets = None
    for tick in range(round(seconds / solver.dt)):
        t = tick * solver.dt
        tasks.update_tasks_from_trajectory(trajectory, t)
        contact_phase = ([1, 1] if trajectory.support_is_both(t) else
                         [1, 0] if str(trajectory.support_side(t)) == 'left' else [0, 1])
        if contact_aware:
            base_now = robot.get_T_world_fbase()
            yaw_error = np.arctan2(base_now[1, 0], base_now[0, 0])
            momentum.L_world = np.array([0., 0., -.12 * yaw_error])
            target = tasks.com_task.target_world.copy()
            if correction is not None:
                target[:2] += correction_spline(t)
            # Avoid straightening the outgoing leg during transfer; the
            # published 60-degree geometric home remains the standing pose.
            target[2] = initial_com[2] - transfer_crouch_m * ((target[1] - initial_com[1]) / .025)**2
            if walking_crouch_m:
                ramp = np.clip(t, 0., 1.)
                ramp = ramp**3 * (10 - 15*ramp + 6*ramp*ramp)
                target[2] -= walking_crouch_m * ramp
            tasks.com_task.target_world = target
            if roll_from_com_lever_arm_m is not None:
                from scipy.spatial.transform import Rotation
                roll_target = -(target[1] - initial_com[1]) / roll_from_com_lever_arm_m
                tasks.trunk_orientation_task.R_world_frame = Rotation.from_rotvec([roll_target, 0., 0.]).as_matrix()
            if com_height_weight is not None:
                com_height_task.target_world = target
            dominant = 0 if target[1] > initial_com[1] else 1
            # Blend sole-roll priorities through double support instead of
            # abruptly pinning a different sole at COM centre crossing.
            for side, task in enumerate(sole_roll):
                weight = float(np.exp(np.clip((1 if side == 0 else -1)
                               * (target[1] - initial_com[1]) / .002, -10., 10.)))
                if smooth_support_roll:
                    if trajectory.support_is_both(t):
                        ds_start, ds_end = trajectory.get_part_t_start(t), trajectory.get_part_t_end(t)
                        before = ([.5, .5] if ds_start <= 0. else
                                  [1., 0.] if str(trajectory.support_side(ds_start - 1e-6)) == 'left' else [0., 1.])
                        after = ([1., 0.] if str(trajectory.support_side(ds_end + 1e-6)) == 'left' else [0., 1.])
                        u_ds = np.clip((t-ds_start)/(ds_end-ds_start), 0., 1.)
                        blend = u_ds**3 * (10-15*u_ds+6*u_ds*u_ds)
                        load = (1-blend)*before[side]+blend*after[side]
                    else:
                        load = float((side == 0) == (str(trajectory.support_side(t)) == 'left'))
                    weight = 1. + (10000. if support_roll_weight is None else support_roll_weight) * load
                task.configure(('left' if side == 0 else 'right') + '_sole_roll', 'soft', weight)
            if not trajectory.support_is_both(t):
                start, end = trajectory.get_part_t_start(t), trajectory.get_part_t_end(t)
                swing = 1 if str(trajectory.support_side(t)) == 'left' else 0
                # Strong preference avoids a discontinuous roll equality when
                # the next support foot is still finishing its landing. The
                # low-vertex contact audit remains the acceptance authority.
                sole_roll[1-swing].configure('loaded_sole_roll',
                    'hard' if support_roll_weight is None else 'soft',
                    1. if support_roll_weight is None else support_roll_weight)
                foot_frame = trajectory.get_T_world_left if swing == 0 else trajectory.get_T_world_right
                if first_swing_start is None:
                    first_swing_start = start
                    preparation_end = start + preparation_steps * period / 2
                    preparation_offsets = [frame(preparation_end)[:3, 3] - initial[:3, 3]
                        for frame, initial in zip((trajectory.get_T_world_left, trajectory.get_T_world_right), initial_feet)]
                startup = float(np.clip((start - first_swing_start) / period, 0., 1.))
                lift_guard = .02 + .015 * (1 - startup)
                swing_height = foot_height * (.6 + .4 * startup)
                u = (t - start - lift_guard) / (end - start - lift_guard - .02)
                task = (tasks.left_foot_task, tasks.right_foot_task)[swing]
                task.position().target_world = swing_position(
                    foot_frame(start)[:3, 3], foot_frame(end)[:3, 3], u, swing_height)
                if swing_com_rise_m:
                    raised_com = tasks.com_task.target_world.copy()
                    raised_com[2] += swing_com_rise_m * task.position().target_world[2] / foot_height
                    tasks.com_task.target_world = raised_com
                    if com_height_weight is not None:
                        com_height_task.target_world = raised_com
                contact_phase = [1, 1]
                if 1e-9 < u < 1. - 1e-9:
                    contact_phase[swing] = 0
                    assert dominant != swing, 'COM-side roll constraint selected the airborne foot'
            if preparation_steps:
                # Build the initial lateral motion from the Placo COM plan with
                # both stance feet fixed. No fake lift or sliding preparation.
                preparing = preparation_end is None or t < preparation_end - 1e-9
                for side, task in enumerate((tasks.left_foot_task, tasks.right_foot_task)):
                    if preparing:
                        task.position().target_world = initial_feet[side][:3, 3].copy()
                    else:
                        task.position().target_world = task.position().target_world - preparation_offsets[side]
                if preparing:
                    contact_phase = [1, 1]
        if com_lateral_scale != 1. or com_lateral_smoothing > 0.:
            target = tasks.com_task.target_world.copy()
            target[1] = initial_com_y + com_lateral_scale * (target[1] - initial_com_y)
            if com_lateral_smoothing > 0.:
                filtered_com_y += (solver.dt / (com_lateral_smoothing + solver.dt)) * (target[1] - filtered_com_y)
                target[1] = filtered_com_y
            tasks.com_task.target_world = target
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
        try:
            solver.solve(True)
        except RuntimeError as error:
            posture = {name: round(float(np.rad2deg(robot.get_joint(name))), 3) for name in names}
            # Diagnostic only: never integrate or publish a solution outside speed limits.
            solver.enable_velocity_limits(False)
            try:
                unconstrained = np.asarray(solver.solve(False))
                print('IK_DIAGNOSTIC without speed limits: feasible; peak joint speed rad/s',
                      float(np.max(abs(unconstrained[6:])) / solver.dt), flush=True)
                print('IK_DIAGNOSTIC joint velocities (Placo order)',
                      np.round(unconstrained[6:] / solver.dt, 3).tolist(), flush=True)
            except RuntimeError:
                print('IK_DIAGNOSTIC still infeasible without speed limits', flush=True)
            finally:
                solver.enable_velocity_limits(True)
            raise RuntimeError(f'Placo IK failed at t={t:.3f}s; support_duration={support_duration}; '
                               f'flat_feet={flat_feet}, exact_com={exact_com}, exact_feet={exact_feet}; '
                               f'last_posture_deg={posture}: {error}') from error
        robot.update_kinematics()
        if tick % 5:
            continue
        left, right = robot.get_T_world_left().copy(), robot.get_T_world_right().copy()
        feet_transforms.append([left, right])
        base_transforms.append(robot.get_T_world_fbase().copy())
        positions.append([robot.get_joint(n) for n in names])
        com_targets.append(tasks.com_task.target_world.copy())
        zmp_targets.append(trajectory.get_p_world_ZMP(t, np.sqrt(9.81 / params.walk_com_height)))
        com_errors.append(float(np.linalg.norm(robot.com_world() - tasks.com_task.target_world)))
        phases.append(contact_phase)
        residuals.append([left[2, 3], right[2, 3]])
        task_errors.append(max(np.linalg.norm(foot[:3, 3] - task.position().target_world)
                               for foot, task in zip((left, right), (tasks.left_foot_task, tasks.right_foot_task))))
        sole_heights.append([float((hull @ foot[:3, :3].T + foot[:3, 3])[:, 2].min())
                             for hull, foot in zip(sole_vertices, (left, right))])
        origin_lifts.append(lifts)
    positions = np.asarray(positions)
    sample_fps = 200
    velocities = np.gradient(positions, 1 / sample_fps, axis=0)
    feet_array = np.asarray(feet_transforms)
    phase_array = np.asarray(phases)
    single = phase_array.sum(axis=1) == 1
    loaded_tilts = np.arccos(np.clip(feet_array[:, :, 2, 2], -1., 1.))[
        single[:, None] & phase_array.astype(bool)]
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
                 foot_height=foot_height, com_height=float(initial_com[2]),
                 reduced_pendulum_height_m=params.walk_com_height,
                 trunk_pitch_task='soft100' if allow_trunk_pitch else 'hard',
                 preparation_steps=preparation_steps, transfer_crouch_m=transfer_crouch_m,
                 yaw_momentum_weight=yaw_momentum_weight, support_roll_weight=support_roll_weight,
                 standing_inward_deg=standing_inward_deg, standing_joint_positions=standing_joints,
                 walking_crouch_m=walking_crouch_m,
                 com_height_weight=com_height_weight,
                 roll_from_com_lever_arm_m=roll_from_com_lever_arm_m,
                 smooth_support_roll=smooth_support_roll, swing_com_rise_m=swing_com_rise_m,
                 com_correction_sha256=(hashlib.sha256(Path(com_correction).read_bytes()).hexdigest()
                                        if com_correction is not None else None),
                 standing_base_transform=transform.tolist(),
                 initial_feet_spacing_m=float(initial_feet[0][1, 3] - initial_feet[1][1, 3]),
                 preparation_end_s=preparation_end if preparation_steps else 0.,
                 double_support_ratio=double_support_ratio,
                 startend_double_support_ratio=startend_double_support_ratio,
                 com_lateral_scale=com_lateral_scale, com_lateral_smoothing=com_lateral_smoothing,
                 contact_aware=contact_aware, contact_guard_s=.02 if contact_aware else 0.,
                 startup_swing_height_scale=[.6, .8, 1.] if contact_aware else None,
                 startup_extra_lift_guard_s=.015 if contact_aware else 0.,
                 double_support_roll_blend_com_width_m=.002 if contact_aware else None,
                 single_support_timesteps=params.single_support_timesteps,
                 yaw_momentum_heading_gain_Nms_per_rad=.12 if contact_aware else None,
                 max_single_support_sole_tilt_deg=float(np.rad2deg(loaded_tilts).max()) if loaded_tilts.size else None,
                 foot_support_rectangle_m=[params.foot_length, params.foot_width],
                 foot_zmp_offset_m=[params.foot_zmp_target_x, params.foot_zmp_target_y],
                 com_target_height_range_m=[float(np.min(np.asarray(com_targets)[:, 2])),
                                            float(np.max(np.asarray(com_targets)[:, 2]))],
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 feet_spacing=params.feet_spacing,
                 max_joint_speed=float(abs(velocities).max()), max_limit_error=max(0., limit_error),
                 min_foot_frame_height=float(np.min(residuals)), source_hashes=source_hashes(),
                 max_feet_position_error=float(max(task_errors)),
                 max_com_position_error_m=max(com_errors),
                 support_orientation=(('loaded sole flat' if support_roll_weight is None else
                                       f'loaded sole roll soft{support_roll_weight:g}') +
                                      '; torso roll free; swing sole roll free' if contact_aware else
                                      'flat soles; torso roll free' if flat_feet else 'free sole roll; upright trunk'),
                 exact_com=exact_com,
                 exact_feet=exact_feet,
                 sample_fps=sample_fps, foot_origin_lift_max_m=np.max(origin_lifts, axis=0).tolist(),
                 planning_joint_speed_limit_rad_s=planning_speed,
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
             dense_com_target=np.asarray(com_targets),
             dense_zmp_target=np.asarray(zmp_targets),
             metadata=json.dumps(stats))
    return stats


def validate_dynamics(candidate_path, output):
    """Reject references that cannot satisfy even optimistic real-floor dynamics."""
    import mujoco
    from diagnose_identified_liftoff import audit, set_identified
    parameter_path = ROOT / 'training/calibrated_actuator/assessment.json'
    identified = json.loads(parameter_path.read_text())
    if identified['source_hashes'] != source_hashes():
        raise ValueError('Actuator assessment belongs to another robot description')
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    set_identified(model, identified['parameters'])
    mujoco.mj_setConst(model, mujoco.MjData(model))
    with np.load(candidate_path) as candidate:
        meta = json.loads(str(candidate['metadata']))
        if not meta.get('contact_aware') or meta['max_limit_error'] > 1e-6 or meta['max_joint_speed'] > 5.24:
            raise ValueError('Candidate has not passed contact-aware planning and joint limits')
        report = audit(model, candidate, identified['parameters'], stride=1, include_projected=False,
                       physical_friction=True)
    report.update(candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                  source_hashes=source_hashes(), parameters=identified['parameters'],
                  parameter_source_sha256=hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256((ROOT / 'diagnose_identified_liftoff.py').read_bytes()).hexdigest())
    report['contact_passed'] = all(s['samples'] == s['feasible'] for s in report['summary'].values())
    report['torque_passed'] = all(s['samples'] == s['within_cap'] for s in report['summary'].values())
    report['passed'] = report['contact_passed'] and report['torque_passed']
    torque_override = os.environ.get('REFERENCE_ALLOW_UNSAFE') == '1'
    if report['contact_passed'] and not report['torque_passed'] and torque_override:
        report['execution_override'] = 'torque audit only; contact feasibility and joint limits remain required'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['contact_passed'] or (not report['torque_passed'] and not torque_override):
        raise ValueError(f'Reference contact/torque/command checks failed; see {output}')
    return {k: v for k, v in report.items() if k != 'rows'}


def fit_candidate():
    """Fit one settled forward cycle, checking actual float32 reader arithmetic."""
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, RotationSpline

    candidate = np.load('/tmp/zp10d-reference/candidate.npz')
    meta = json.loads(str(candidate['metadata']))
    velocity = round(2 * meta['step_length'] / meta['period'], 8)
    if velocity == 0.:
        raise ValueError('Zero-speed stepping needs a separate task command; refusing to overwrite it with standing')
    if 'dense_frames' not in candidate.files or meta.get('sample_fps') != 200:
        raise ValueError('Regenerate the dense whole-foot-corrected Placo candidate before fitting')
    if meta['source_hashes'] != source_hashes():
        raise ValueError('Candidate does not match the current model')
    if meta['max_limit_error'] > 1e-6 or meta['max_feet_position_error'] > .0002:
        raise ValueError('Reference kinematic limits/foot targets have not passed')
    raw_audit = validate_dynamics(Path('/tmp/zp10d-reference/candidate.npz'),
                                  ROOT / 'training/reference_repair/raw_audit.json')
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
    base_rotation = RotationSpline(x, Rotation.from_matrix(raw_base[:, :3, :3]))(check_x).as_matrix()
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
    # Audit the float32 polynomial itself, including the periodic seam. Base
    # poses come from the same original Placo window; joint derivatives are
    # analytic, not derivatives of rounded position samples.
    ticks = np.arange(-4, round(meta['period'] * 1000) + 5)
    phase = np.mod(ticks / (meta['period'] * 1000), 1.).astype(np.float32)
    piece = np.minimum((phase * dense_count).astype(int), dense_count - 1)
    local = (phase - piece.astype(np.float32) / np.float32(dense_count))[:, None]
    c = coeff.astype(np.float32)[:, piece]
    fitted = ((c[0] * local + c[1]) * local + c[2]) * local + c[3]
    fitted[:, 16:32] = ((3*c[0, :, :16]*local + 2*c[1, :, :16])*local
                       + c[2, :, :16]) / meta['period']
    acceleration = (6*c[0, :, :16]*local + 2*c[1, :, :16]) / meta['period']**2
    contact_index = np.minimum(np.floor(phase.astype(float)*dense_count + 1e-4).astype(int), dense_count-1)
    fitted[:, 32:34] = dense_cycle[contact_index, 32:34]
    sample_times = np.arange(len(candidate['dense_base'])) / meta['sample_fps']
    times = dense_start / meta['sample_fps'] + ticks / 1000
    fitted_base = np.tile(np.eye(4), (len(ticks), 1, 1))
    fitted_base[:, :3, 3] = CubicSpline(sample_times, candidate['dense_base'][:, :3, 3])(times)
    fitted_base[:, :3, :3] = RotationSpline(sample_times, Rotation.from_matrix(
        candidate['dense_base'][:, :3, :3]))(times).as_matrix()
    fitted_path = Path('/tmp/zp10d-reference/fitted_candidate.npz')
    np.savez(fitted_path, dense_frames=fitted, dense_base=fitted_base,
             dense_joint_acceleration=acceleration, metadata=json.dumps(dict(meta, sample_fps=1000)))
    fitted_audit = validate_dynamics(fitted_path, ROOT / 'training/reference_repair/fitted_audit.json')
    record = dict(basis='periodic_cubic', coefficients={f'dim_{i}': coeff[:, :, i].tolist() for i in range(40)},
                  period=meta['period'], fps=FPS, frame_offsets={}, start_offset=start,
                  nb_steps_in_period=count,
                  startend_double_support_ratio=meta['startend_double_support_ratio'])
    references = {f'{velocity}_0.0_0.0': record}
    # The zero-command reference is actual accepted geometric home, not a legacy pose.
    names, actuators, home = inputs()
    standing = np.concatenate([meta.get('standing_joint_positions', home[7:]),
                               np.zeros(16), [1., 1.], np.zeros(6)])
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
                raw_dynamics_audit=raw_audit, fitted_dynamics_audit=fitted_audit,
                commands=[[0., 0., 0.], [velocity, 0., 0.]], action_scale=scale,
                scope='standing and slow forward reference only; not trained or dynamics-accepted')
    DATA.mkdir(exist_ok=True)
    with (DATA / 'polynomial_coefficients.pkl').open('wb') as out:
        pickle.dump(references, out)
    (DATA / 'reference.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)
    prepare_reward_reference()


def prepare_reward_reference(candidate_path=Path('/tmp/zp10d-reference/candidate.npz')):
    """Bind exact Placo support labels and whole-foot heights to the fitted cycle."""
    import mujoco
    from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

    candidate = np.load(candidate_path)
    meta = json.loads((DATA / 'reference.json').read_text())
    if meta.get('candidate_sha256') is not None and meta['candidate_sha256'] != hashlib.sha256(
            Path(candidate_path).read_bytes()).hexdigest():
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
    fitted_heights, lateral_com, frame_error = [], [], 0.
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
        mujoco.mj_comPos(model, data)
        foot_positions = data.site_xpos[sites]
        lateral = foot_positions[0, :2] - foot_positions[1, :2]
        lateral /= np.linalg.norm(lateral)
        lateral_com.append(float((data.subtree_com[1, :2] - foot_positions[:, :2].mean(0)) @ lateral))
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
        lateral_com_from_foot_midpoint_m=lateral_com,
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
    parser.add_argument('--validate', action='store_true', help='200 Hz real-floor candidate audit')
    parser.add_argument('--reward-reference', action='store_true')
    parser.add_argument('--candidate', type=Path, default=Path('/tmp/zp10d-reference/candidate.npz'),
                        help='matching raw candidate when refreshing reward targets')
    parser.add_argument('--step-length', type=float, default=.0025)
    parser.add_argument('--support-duration', type=float, default=.16)
    parser.add_argument('--foot-height', type=float, default=.0025)
    parser.add_argument('--seconds', type=float, default=6.)
    parser.add_argument('--feet-spacing', type=float, default=None,
                        help='planned landing spacing; initial published home remains unchanged')
    parser.add_argument('--zmp-offset-x', type=float,
                        help='fore/aft pressure target inside the actual support sole, metres')
    parser.add_argument('--allow-trunk-pitch', action='store_true',
                        help='candidate only: prefer upright pitch without overconstraining load transfer')
    parser.add_argument('--preparation-steps', type=int, default=0,
                        help='initial Placo half-cycles with both feet planted before the first lift')
    parser.add_argument('--transfer-crouch', type=float, default=.0038,
                        help='COM lowering at 25 mm lateral transfer, metres; default 0.0038')
    parser.add_argument('--walking-crouch', type=float, default=0.,
                        help='constant walking COM lowering, eased in over the first second, metres')
    parser.add_argument('--com-height-weight', type=float,
                        help='soft COM height weight; horizontal COM remains hard; default all hard')
    parser.add_argument('--com-correction', type=Path,
                        help='candidate-only full-body ZMP correction NPZ: time_s and offset_xy_m')
    parser.add_argument('--roll-from-com-lever-arm', type=float,
                        help='follow smooth planned torso roll from lateral COM / lever arm in metres')
    parser.add_argument('--smooth-support-roll', action='store_true',
                        help='quintic contact-load blending of sole roll through double support')
    parser.add_argument('--swing-com-rise', type=float, default=0.,
                        help='coordinate COM rise with the swing-foot profile, metres')
    parser.add_argument('--yaw-momentum-weight', type=float,
                        help='use soft yaw momentum and a heading target; default hard momentum')
    parser.add_argument('--support-roll-weight', type=float,
                        help='prefer flat support sole with this soft weight; default hard sole roll')
    parser.add_argument('--standing-inward-deg', type=float, default=0.,
                        help='candidate standing hip-roll inward angle per leg, degrees (0 to 5)')
    parser.add_argument('--reduced-pendulum-height', type=float, default=.35,
                        help='planning inertia approximation in metres; physical COM/mass unchanged')
    parser.add_argument('--double-support-ratio', type=float, default=.5625,
                        help='double-support duration divided by single-support duration')
    parser.add_argument('--startend-double-support-ratio', type=float, default=8.,
                        help='extra double-support ratio at the beginning and end')
    parser.add_argument('--com-lateral-scale', type=float, default=1.,
                        help='scale Placo lateral COM motion about home for a candidate')
    parser.add_argument('--com-lateral-smoothing', type=float, default=0.,
                        help='first-order lateral COM target smoothing time in seconds')
    parser.add_argument('--flat-feet', action='store_true',
                        help='candidate only: constrain sole roll and release torso roll')
    parser.add_argument('--exact-com', action='store_true',
                        help='candidate only: enforce the pattern-generator COM target as a hard IK task')
    parser.add_argument('--exact-feet', action='store_true',
                        help='candidate only: reject IK solutions that miss the foot pose tasks')
    parser.add_argument('--legacy-reference', action='store_true',
                        help='diagnostic only: reproduce the old upright/free-sole-roll formulation')
    parser.add_argument('--data-dir', type=Path,
                        help='fit output (default: /tmp/zp10d-reference/data); metadata input otherwise')
    args = parser.parse_args()
    DATA = args.data_dir or (Path('/tmp/zp10d-reference/data') if args.fit else DATA)
    if args.check:
        check_description()
    elif args.reward_reference:
        prepare_reward_reference(args.candidate)
    elif args.validate:
        validate_dynamics(Path('/tmp/zp10d-reference/candidate.npz'),
                          ROOT / 'training/reference_repair/raw_audit.json')
    elif args.fit:
        fit_candidate()
    else:
        generate(args.step_length, args.support_duration, args.foot_height, args.seconds, args.feet_spacing,
                 double_support_ratio=args.double_support_ratio,
                 startend_double_support_ratio=args.startend_double_support_ratio,
                 contact_aware=not args.legacy_reference,
                 com_lateral_scale=args.com_lateral_scale,
                 com_lateral_smoothing=args.com_lateral_smoothing,
                 flat_feet=args.flat_feet, exact_com=args.exact_com, exact_feet=args.exact_feet,
                 zmp_offset_x=args.zmp_offset_x, allow_trunk_pitch=args.allow_trunk_pitch,
                 preparation_steps=args.preparation_steps, transfer_crouch_m=args.transfer_crouch,
                 yaw_momentum_weight=args.yaw_momentum_weight, support_roll_weight=args.support_roll_weight,
                 standing_inward_deg=args.standing_inward_deg,
                 reduced_pendulum_height_m=args.reduced_pendulum_height,
                 walking_crouch_m=args.walking_crouch, com_height_weight=args.com_height_weight,
                 com_correction=args.com_correction, roll_from_com_lever_arm_m=args.roll_from_com_lever_arm,
                 smooth_support_roll=args.smooth_support_roll, swing_com_rise_m=args.swing_com_rise)
