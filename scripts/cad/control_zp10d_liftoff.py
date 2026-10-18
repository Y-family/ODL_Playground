"""Bounded MuJoCo feedback experiment; formal CAD/home/actuators are read-only inputs.

Run via run_memory_limited.py. A run is successful only after measured unloading,
whole-sole clearance, landing, and recovery. Failed trials remain failed reports.
"""
import json
from pathlib import Path
import resource
import signal
import sys
import warnings

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation

import assess_zp10d_loads as loads

CACHE = Path('/tmp/open-duck-liftoff-control')
PATHS = [loads.OUT / 'open_duck_mini_upper_refit.step',
         loads.MODEL_DIR / 'open_duck_mini_zp10d.xml',
         loads.MODEL_DIR / 'scene_flat_terrain.xml']
DT = .02
HORIZON = .24
TRANSFER_DURATION = 2.5
VELOCITY_SAMPLES = 10
LIFT_HEIGHT = .012
REQUIRED_LIFT_HEIGHT = .010
LIFT_RISE_TIME = 1.2
SLIP_WEIGHT = 10000.
CONTROL_DIFFERENCE = .004
SUPPORT_ROLL_LIMIT = np.deg2rad(40)
SUPPORT_ROLL_TARGET = np.deg2rad(30)
CLEARANCE_WEIGHT = 10000.


def setup():
    mass = json.loads((loads.OUT / 'open_duck_mini_zp10d_mass_properties.json').read_text())
    hashes = [mass[k] for k in ('source_step_sha256', 'mjcf_sha256', 'scene_sha256')]
    assert [loads.digest(p) for p in PATHS] == hashes
    model = mujoco.MjModel.from_xml_path(str(PATHS[2]))
    assert model.neq == 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
    mujoco.mj_step(model, data, nstep=round(3/model.opt.timestep))
    mujoco.mj_forward(model, data)
    return model, data, hashes


def smooth_rate(time, duration):
    x = np.clip(time/duration,0.,1.)
    return 30*x*x*(1-x)*(1-x)/duration


def support_reference(model, initial, side):
    """Fit a collision-checked endpoint seed, not an executed lift trajectory."""
    support, sign = 1-side, 1 if side == 0 else -1
    joints = model.actuator_trnid[:, 0]
    adr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    base = model.body('base').id
    feet = [model.body(n).id for n in ('foot_assembly', 'foot_assembly_2')]
    vertices = loads.sole_vertices(model, feet)
    initial_R = initial.xmat[base].reshape(3, 3)
    foot_R = initial.xmat[feet].reshape(2, 3, 3)
    bank = sign*np.deg2rad(28)
    support_R = Rotation.from_rotvec([bank, 0., 0.]).as_matrix()@foot_R[support]
    z = (vertices[support]@support_R.T)[:, 2]
    pivot = vertices[support][z < z.min()+1e-5].mean(axis=0)
    point = initial.xpos[feet[support]]+foot_R[support]@pivot
    point[2] = 0.
    support_pos = point-support_R@pivot
    ids = np.array([a for a in range(model.nu) if 'antenna' not in model.actuator(a).name])
    lo, hi = model.jnt_range[joints[ids]].T.copy()
    lo += np.deg2rad(.6)
    hi -= np.deg2rad(.6)
    for k, a in enumerate(ids):
        if model.actuator(a).name.endswith('_knee'):
            lo[k] = np.deg2rad(1)
    target = initial.qpos[adr].copy()
    suffixes = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')
    swing_ids = [model.actuator(('left' if side == 0 else 'right')+'_'+n).id for n in suffixes]
    support_ids = [model.actuator(('right' if side == 0 else 'left')+'_'+n).id for n in suffixes]
    target[swing_ids] = np.deg2rad([sign*25, sign*12.7, sign*9.4, 2.6, -58.9])
    target[support_ids] = np.deg2rad([-sign*9.6, -sign*.16, sign*50, 68, -22.1])
    head = [model.actuator(n).id for n in ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')]
    target[head] = np.deg2rad([33.4 if side == 0 else 32.4, 3.8, -sign*80, -sign*7.1])
    scratch = mujoco.MjData(model)
    floor = model.geom('floor').id
    margins = model.geom_margin.copy()
    mask = (model.geom_contype != 0)&(np.arange(model.ngeom) != floor)

    def pose(x):
        scratch.qpos[:] = initial.qpos
        scratch.qpos[adr[ids]] = x
        scratch.qvel[:] = 0.
        mujoco.mj_kinematics(model, scratch)
        correction = support_R@scratch.xmat[feet[support]].reshape(3, 3).T
        q = Rotation.from_matrix(correction@initial_R).as_quat()
        scratch.qpos[3:7] = q[[3, 0, 1, 2]]
        scratch.qpos[:3] = support_pos-correction@(scratch.xpos[feet[support]]-initial.qpos[:3])
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        mujoco.mj_comVel(model, scratch)
        com = scratch.subtree_com[base].copy()
        height = (vertices[side]@scratch.xmat[feet[side]].reshape(3, 3).T+scratch.xpos[feet[side]])[:, 2].min()
        gravity = np.zeros(model.nv)
        mujoco.mj_rne(model, scratch, 0, gravity)
        jac = np.zeros((3, model.nv))
        contact = point.copy()
        contact[0] = com[0]
        mujoco.mj_jac(model, scratch, jac, None, contact, feet[support])
        torque = (gravity-jac.T@np.array([0., 0., model.body_subtreemass[base]*9.81]))[dofs]
        mujoco.mj_collision(model, scratch)
        separation = np.full((model.nbody, model.nbody), .004)
        for con in scratch.contact:
            if floor not in con.geom:
                a, b = sorted(int(model.geom_bodyid[g]) for g in con.geom)
                separation[a, b] = min(separation[a, b], con.dist)
        return com, height, torque, separation

    def residual(x):
        com, height, torque, separation = pose(x)
        rotation = Rotation.from_matrix(scratch.xmat[base].reshape(3, 3)@initial_R.T).as_rotvec()
        sole = vertices[support]@support_R.T+support_pos
        over_x = max(sole[:, 0].min()+.008-com[0], com[0]-(sole[:, 0].max()-.008), 0.)
        return np.r_[1000*(com[1]-point[1]), 1000*(height-LIFT_HEIGHT), 100*over_x,
            5000*np.maximum(.0005-separation.ravel(), 0),
            30*(com[2]-initial.subtree_com[base, 2]),
            rotation-np.deg2rad([sign*30, -12, sign*15]),
            .05*(x-target[ids]), 2*np.maximum(np.abs(torque[:14])-.8, 0),
            .1*(scratch.qpos[adr[support_ids[3]]]-np.deg2rad(55))]

    model.geom_margin[mask] += .002
    try:
        fit = least_squares(residual, np.clip(target[ids], lo, hi), bounds=(lo, hi),
            max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        com, height, torque, separation = pose(fit.x)
        assert abs(com[1]-point[1]) < .0001 and abs(height-LIFT_HEIGHT) < .0001
        assert separation.min() >= .00045 and np.max(np.abs(torque[:14])) < .9
        assert scratch.xmat[base, 8] > np.cos(np.deg2rad(40))
        return dict(qpos=scratch.qpos.copy(), com=com, torque=torque,
            rotation=Rotation.from_matrix(scratch.xmat[base].reshape(3, 3)@initial_R.T).as_rotvec(),
            numerical_convergence=bool(fit.success), minimum_separation_mm=float(separation.min()*1000),
            scope='Collision-checked kinematic endpoint seed only; not a home-to-lift or hardware acceptance.')
    finally:
        model.geom_margin[:] = margins


def trial(side, replay_prefix=False, replay_prepare_only=False,replay_recovery=False,
          replay_until=None):
    model, data, hashes = setup()
    source_hash = loads.digest(__file__)
    helper_hash = loads.digest(loads.__file__)
    preparation = []
    if replay_prefix:
        previous = json.loads((CACHE/f'{"left" if side == 0 else "right"}.json').read_text())
        assert previous['formal_hashes'] == hashes
        replay_phases = (('prepare','transfer','lift','lower','recover') if replay_recovery
                         else ('prepare',) if replay_prepare_only else ('prepare','transfer'))
        preparation = [f for f in previous['frames'] if f['phase'] in replay_phases]
    joints = model.actuator_trnid[:, 0]
    adr, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    feet = [model.body('foot_assembly').id, model.body('foot_assembly_2').id]
    base, floor = model.body('base').id, model.geom('floor').id
    vertices = loads.sole_vertices(model, feet)
    leg_ids = np.array([model.actuator(f'{s}_{suffix}').id for s in ('left', 'right')
                    for suffix in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')])
    head_yaw = model.actuator('head_yaw').id
    head_ids = np.array([model.actuator(n).id for n in ('neck_pitch','head_pitch','head_yaw','head_roll')])
    ids = leg_ids.copy()
    fixed = [a for a in range(model.nu) if a not in ids]
    home = model.key_ctrl[0].copy()
    initial_q, initial_com = data.qpos[adr].copy(), data.subtree_com[base].copy()
    initial_full_qpos = data.qpos.copy()
    initial_R = data.xmat[base].reshape(3, 3).copy()
    reference = support_reference(model, data, side)
    initial_foot_R = data.xmat[feet].reshape(2, 3, 3).copy()
    initial_foot_pos = data.xpos[feet].copy()
    foot_tangents = initial_foot_R.transpose(0,2,1)@np.array([1.,0.,0.])
    initial_height = np.array([(v@initial_foot_R[i].T+initial_foot_pos[i])[:,2].min()
                               for i,v in enumerate(vertices)])
    forces, _, _, moments = loads.ground_forces(model, data, floor, feet, with_moments=True)
    support = 1-side
    support_cop = np.array([-moments[support, 1], moments[support, 0]]) / forces[support, 2]
    support_world = vertices[support]@initial_foot_R[support].T+initial_foot_pos[support]
    target_com = reference['com'].copy()
    preparation_rise = float(np.clip(target_com[2]-initial_com[2], 0., .008))
    target_com[2] = initial_com[2]+preparation_rise
    sole_normals = []
    for i,v in enumerate(vertices):
        normal = np.eye(3)[np.argmin(np.ptp(v,axis=0))]
        if (initial_foot_R[i]@normal)[2] < 0:
            normal *= -1
        sole_normals.append(normal)
    local_centres = []
    cop_lower, cop_upper = [], []
    for i, (b, v) in enumerate(zip(feet, vertices)):
        w = v@initial_foot_R[i].T+initial_foot_pos[i]
        cop_lower.append(w[:,:2].min(axis=0)+.004)
        cop_upper.append(w[:,:2].max(axis=0)-.004)
        centre = (w.min(axis=0)+w.max(axis=0))/2
        centre[2] = 0
        local_centres.append((centre-initial_foot_pos[i])@initial_foot_R[i])
    cop_lower, cop_upper = np.array(cop_lower), np.array(cop_upper)
    plane_xy = np.array([initial_foot_pos[i]+initial_foot_R[i]@c for i,c in enumerate(local_centres)])[:, :2]
    normal_margins = model.geom_margin.copy()
    proximity_mask = (model.geom_contype != 0) & (np.arange(model.ngeom) != floor)
    clearance_pairs = [('trunk_assembly',name) for name in (
        'left_roll_to_pitch_assembly','right_roll_to_pitch_assembly',
        'knee_and_ankle_assembly','knee_and_ankle_assembly_3')]
    clearance_pair_ids = {tuple(sorted(model.body(n).id for n in pair)):i
                          for i,pair in enumerate(clearance_pairs)}
    sign = 1 if side == 0 else -1
    original_actuators = [a.copy() for a in (model.actuator_gainprm, model.actuator_biasprm,
                          model.actuator_forcerange, model.actuator_ctrlrange, model.jnt_range)]
    initial_support_normal = (initial_foot_R[support]@sole_normals[support])[:2]
    desired_support_normal = np.array([0., (-1 if support == 1 else 1)*np.sin(SUPPORT_ROLL_TARGET)])
    transfer_posture = reference['qpos'][adr].copy()
    print('Current-model endpoint seed:', 'left' if side == 0 else 'right',
          'COM mm', np.round(reference['com']*1000, 3).tolist(),
          'knees deg', np.round(np.degrees(transfer_posture[[model.actuator(s+'_knee').id for s in ('left','right')]]), 3).tolist(),
          'clearance mm', reference['minimum_separation_mm'], flush=True)
    scratch = mujoco.MjData(model)
    horizon = HORIZON
    lookahead_horizon = HORIZON
    phase, phase_start, unloaded_for, clear_for = 'prepare', 0., 0., 0.
    transfer_start, prepared_for = None, 0.
    transfer_clock = 0.
    transfer_rate = 1.
    prepared_q, landing_origin, support_q, descent_start, landing_q = None, None, None, None, None
    swing_xy = initial_foot_pos[side,:2].copy()
    touchdown_time, touchdown_normals, touchdown_com, touchdown_rotation = None, None, None, None
    liftoff_origin, liftoff_rotation = None, None
    unwind_start, lift_hold_ctrl, lift_exit_q = None, None, None
    landing_toe, liftoff_com = None, None
    liftoff_capture, liftoff_start = None, None
    liftoff_support_normal = None
    swing_recenter_start = None
    descent_clearance = LIFT_HEIGHT
    swing_ik_residual = None
    knee_ids = [model.actuator(s+'_knee').id for s in ('left','right')]
    longest, landed_for, recovered_for = 0., 0., 0.
    target_clear_for, longest_target_clear = 0., 0.
    unload_height = 0.
    frames, transitions, failures = [], [], []
    replayed_frames, replayed_phases = 0, []
    replay_active = bool(preparation)
    constraint_labels = ([model.actuator(a).name+'_min' for a in range(model.nu)]
        +[model.actuator(a).name+'_max' for a in range(model.nu)]
        +['self_separation','base_tilt','support_sole_normal_y_min','support_sole_normal_y_max','swing_sole_clearance','swing_next_hold_height']
        +['self_separation_'+'/'.join(pair) for pair in clearance_pairs]
        +[f'{s}_loaded_slip_window' for s in ('left','right')])
    prediction_failure = {}
    peak_torque = np.zeros(model.nu)
    torque_squared = np.zeros(model.nu)
    saturation_time = np.zeros(model.nu)
    saturation_run = np.zeros(model.nu)
    longest_saturation_run = np.zeros(model.nu)
    executed_duration = 0.
    lift_peak_torque = np.zeros(model.nu)
    lift_torque_squared = np.zeros(model.nu)
    lift_saturation_time = np.zeros(model.nu)
    lift_duration = 0.
    torque_caps = np.max(np.abs(model.actuator_forcerange),axis=1)
    min_joint_margin = np.full(model.nu, np.inf)
    self_peaks = {}
    proximity_gradient = np.zeros(model.nu)
    proximity_pair = None
    proximity_index, prediction_path = 0, []
    actual_min_separation = np.inf
    loaded_slip = np.zeros(2)
    peak_slip_speed = np.zeros(2)
    derivative_cache = None
    refresh_derivatives = True
    stop, recovery_ctrl, recovery_com, recovery_q, recovery_rotation = None, None, None, None, None
    recovery_route, recovery_route_start = None, None
    recovery_origin, recovery_mapping = None, None
    home_alignment_start = None
    def cancel(signum, frame):
        nonlocal stop
        stop = 'interrupted_for_diagnosis'
    old_signal = signal.signal(signal.SIGINT,cancel)
    CACHE.mkdir(exist_ok=True)
    # Same native contact solver is used in prediction and execution. No rigid
    # double-foot pose assumption, external forces, or altered servo gains.
    weights = np.r_[[400.,400.,1000.], [10.,10.,5.], [3.]*3, [.3]*3,
                    [.005]*model.nu, [.05]*model.nu, [30.]*2, [5.,.3,5.,.3], [.08]*2, [.5]*4, [200.]*4, [SLIP_WEIGHT/np.sqrt(VELOCITY_SAMPLES)]*(6*VELOCITY_SAMPLES), [2.]*2, [1.]]
    anchor_start = 12+2*model.nu+2+4+2+4
    for a in (model.actuator("left_hip_yaw").id,model.actuator("right_hip_yaw").id):
        weights[12+a] = 2.
        weights[12+model.nu+a] = .3
    roll_ids = np.array([model.actuator(s+'_hip_roll').id for s in ('left','right')])
    weights[12+roll_ids] = 1.5

    def measure(d):
        f, _, nonfoot, m = loads.ground_forces(model, d, floor, feet, with_moments=True)
        height = np.array([(v@d.xmat[b].reshape(3, 3).T+d.xpos[b])[:, 2].min()
                           for b, v in zip(feet, vertices)])
        tilt = np.arccos(np.clip(d.xmat[base].reshape(3, 3)[2, 2], -1, 1))
        return f, nonfoot, m, height, float(tilt)

    def feature(d, contact_velocities):
        mujoco.mj_subtreeVel(model, d)
        height = [(v@d.xmat[b].reshape(3, 3).T+d.xpos[b])[:, 2].min()
                  for b, v in zip(feet, vertices)]
        foot_rotation = np.concatenate([(d.xmat[b].reshape(3,3)@n)[:2]
                                        for b,n in zip(feet,sole_normals)])
        force, _, _, moment = loads.ground_forces(model, d, floor, feet, with_moments=True)
        xy = np.array([d.xpos[b]+d.xmat[b].reshape(3,3)@c for b,c in zip(feet,local_centres)])[:, :2]
        if phase in ('lift','lower'):
            xy[side] = d.xpos[feet[side],:2]
            if landing_toe is not None and landing_q is None:
                xy[side] = (d.xpos[feet[side]]+d.xmat[feet[side]].reshape(3,3)@landing_toe)[:2]
        tangents = d.xmat[feet].reshape(2,3,3)@foot_tangents[:,:,None]
        headings = np.arctan2(tangents[:,1,0],tangents[:,0,0])
        return np.r_[d.subtree_com[base], d.subtree_linvel[base],
                     Rotation.from_matrix(d.xmat[base].reshape(3, 3)@initial_R.T).as_rotvec(),
                     d.qvel[3:6], d.qpos[adr], d.qvel[dofs], height, foot_rotation, force[:,2], moment[:,:2].ravel(), xy.ravel(),contact_velocities,headings]*weights[:-1]

    def contact_velocity(d):
        force, velocity = np.zeros(2), np.zeros((2,2))
        speed_squared = np.zeros(2)
        for ci,c in enumerate(d.contact):
            if floor not in c.geom:
                continue
            body = int(model.geom_bodyid[next(g for g in c.geom if g != floor)])
            if body not in feet:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model,d,ci,wrench)
            if wrench[0] <= .05:
                continue
            i = feet.index(body)
            jac = np.zeros((3,model.nv))
            mujoco.mj_jac(model,d,jac,None,c.pos,body)
            force[i] += wrench[0]
            tangential = (jac@d.qvel)[:2]
            velocity[i] += wrench[0]*tangential
            speed_squared[i] += wrench[0]*(tangential@tangential)
        result = np.zeros((2,3))
        result[:,:2] = np.divide(velocity,force[:,None],out=np.zeros((2,2)),where=force[:,None]>.5)
        for i,body in enumerate(feet):
            if force[i] > .5:
                # Include slip hidden by cancellation at different contact
                # points, including pitch/roll as well as yaw.
                result[i,2] = np.sqrt(max(0.,speed_squared[i]/force[i]-result[i,:2]@result[i,:2]))
        return result

    def recovery_reference():
        before = initial_foot_pos[0,:2]-initial_foot_pos[1,:2]
        after = data.xpos[feet[0],:2]-data.xpos[feet[1],:2]
        yaw = np.arctan2(before[0]*after[1]-before[1]*after[0],before@after)
        rotation = Rotation.from_rotvec([0.,0.,yaw]).as_matrix()
        target = initial_com.copy()
        target[:2] = data.xpos[feet,:2].mean(axis=0)+rotation[:2,:2]@(initial_com[:2]-initial_foot_pos[:,:2].mean(axis=0))
        return target, rotation, yaw

    def recovery_source_time(age):
        source = recovery_route_start-max(0.,age-.6)/1.3
        # Unwind the head and restore standing height more slowly once the
        # transfer corridor is cleared. Reversing preparation at the transfer
        # rate exhausted contact slip while arresting the head's yaw reaction.
        if transfer_start is not None and source < transfer_start:
            source = transfer_start+(source-transfer_start)/2.
        return max(0.,source)

    def recovery_state(age):
        # One measured route supplies every recovery task. Independent COM,
        # body, foot and head clocks previously demanded incompatible poses.
        source_time = recovery_source_time(age)
        pose = np.array([np.interp(source_time,recovery_route[0],q)
                         for q in recovery_route[1].T])
        pose[3:7] /= np.linalg.norm(pose[3:7])
        rotation, translation = recovery_mapping
        if home_alignment_start is not None:
            # The rolled entry foot's body origin is not a settled home frame.
            # Once both soles are nearly flat, fit the original home to their
            # current planar frame; preserve the original joint angles.
            home_com,home_rotation,home_yaw = recovery_reference()
            alignment = loads.smooth((age-home_alignment_start)/1.)
            yaw = np.arctan2(rotation[1,0],rotation[0,0])
            turn = np.arctan2(np.sin(home_yaw-yaw),np.cos(home_yaw-yaw))
            translation = (1-alignment)*translation+alignment*(home_com-home_rotation@initial_com)
            rotation = Rotation.from_rotvec([0.,0.,yaw+alignment*turn]).as_matrix()
        pose[:3] = rotation@pose[:3]+translation
        pose[3:7] = (Rotation.from_matrix(rotation)*Rotation.from_quat(pose[[4,5,6,3]])).as_quat()[[3,0,1,2]]
        blend = loads.smooth(age/.6)
        pose = recovery_origin+blend*(pose-recovery_origin)
        pose[3:7] /= np.linalg.norm(pose[3:7])
        scratch.qpos[:] = pose
        mujoco.mj_kinematics(model,scratch)
        mujoco.mj_comPos(model,scratch)
        body_rotation = scratch.xmat[base].reshape(3,3)@initial_R.T
        normals = np.concatenate([(scratch.xmat[b].reshape(3,3)@n)[:2]
                                  for b,n in zip(feet,sole_normals)])
        tangents = scratch.xmat[feet].reshape(2,3,3)@foot_tangents[:,:,None]
        headings = np.arctan2(tangents[:,1,0],tangents[:,0,0])
        return scratch.subtree_com[base].copy(),pose[adr],body_rotation,normals,headings

    def landing_reference():
        tangent = data.xmat[feet[support]].reshape(3,3)@foot_tangents[support]
        yaw = np.arctan2(tangent[1],tangent[0])
        rotation = Rotation.from_rotvec([0.,0.,yaw]).as_matrix()
        target = data.xpos[feet[support]]+rotation@(initial_foot_pos[side]-initial_foot_pos[support])
        return target,rotation,yaw

    def near_distance(d):
        # Expand only broad-phase discovery, without integrating or resolving
        # these contacts. The physical simulation always uses original margins.
        model.geom_margin[proximity_mask] += .002
        try:
            mujoco.mj_collision(model, d)
            contacts = [c for c in d.contact if floor not in c.geom]
            local_distances = np.full(len(clearance_pairs),.004)
            if not contacts:
                return .004,np.zeros(model.nu),None,local_distances
            # Keep all four local corridors in the QP at once. A single minimum
            # let the solver improve one pair by colliding with the other pair.
            for c in contacts:
                key = tuple(sorted(int(model.geom_bodyid[g]) for g in c.geom))
                index = clearance_pair_ids.get(key)
                if index is not None:
                    local_distances[index] = min(local_distances[index],c.dist)
            contact = min(contacts,key=lambda c:c.dist)
            jac = []
            bodies = [int(model.geom_bodyid[g]) for g in contact.geom]
            for body in bodies:
                jp = np.zeros((3,model.nv))
                mujoco.mj_jac(model,d,jp,None,contact.pos,body)
                jac.append(jp)
            gradient = contact.frame[:3]@(jac[1]-jac[0])[:,dofs]
            return float(contact.dist),gradient,[model.body(b).name for b in bodies],local_distances
        finally:
            model.geom_margin[:] = normal_margins

    def predict(u):
        nonlocal proximity_gradient,proximity_pair,proximity_index,prediction_path
        mujoco.mj_copyData(scratch, model, data)
        scratch.ctrl[:] = u
        velocities = []
        window_velocities = []
        prediction_path = []
        proximity_index = 0
        q_min = q_max = data.qpos[adr].copy()
        separation, tilt_max = .004, 0.
        pair_separations = np.full(len(clearance_pairs),.004)
        normal_min, normal_max, swing_min = np.inf, -np.inf, np.inf
        window_height = None
        closest_gradient, closest_pair = np.zeros(model.nu), None
        cost_steps = round(horizon/model.opt.timestep)
        lookahead_steps = round(lookahead_horizon/model.opt.timestep)
        early_transfer = phase == 'prepare' or (phase == 'transfer' and transfer_clock < TRANSFER_DURATION)
        contact_steps = round((horizon if early_transfer else DT)/model.opt.timestep)
        sample_steps = round(contact_steps/VELOCITY_SAMPLES)
        geometry_steps = round(lookahead_steps/VELOCITY_SAMPLES)
        cost_features = contact_features = None
        for n in range(max(cost_steps,lookahead_steps)):
            mujoco.mj_step(model, scratch)
            if n < lookahead_steps:
                q_min = np.minimum(q_min,scratch.qpos[adr])
                q_max = np.maximum(q_max,scratch.qpos[adr])
                tilt_max = max(tilt_max,float(np.arccos(np.clip(1-2*np.sum(scratch.qpos[4:6]**2),-1,1))))
            track = n < contact_steps and n % sample_steps == sample_steps-1
            window_track = n < round(DT/model.opt.timestep)
            geometry = n < lookahead_steps and n % geometry_steps == geometry_steps-1
            if track or window_track or geometry or n+1 == cost_steps:
                mujoco.mj_forward(model,scratch)
            if track or window_track:
                measured_velocity = contact_velocity(scratch)
            if window_track:
                window_velocities.append(measured_velocity)
            if track:
                cost_velocity = measured_velocity.copy()
                transfer_braking = phase == 'transfer' and f[side,2] < 3.
                toe_braking = ((phase == 'lift' and swing_recenter_start is None)
                               or (phase == 'lower' and height[side] < .004))
                if transfer_braking or (toe_braking and landing_q is None and landing_toe is not None):
                    # Brake the future landing point before it touches down.
                    # Near unloading, brief loss of contact must not remove
                    # this task: the toe can touch again before capture. A
                    # contact-only cost missed its speed and allowed a skid.
                    toe = (vertices[side][np.argmin(world_soles[side][:,2])]
                           if transfer_braking else landing_toe)
                    point = scratch.xpos[feet[side]]+scratch.xmat[feet[side]].reshape(3,3)@toe
                    jac = np.zeros((3,model.nv))
                    mujoco.mj_jac(model,scratch,jac,None,point,feet[side])
                    cost_velocity[side,:2] = (jac@scratch.qvel)[:2]
                    if not transfer_braking and swing_recenter_start is not None:
                        # Brake relative to the moving airborne pivot, whose
                        # commanded inward motion must not count as ground slip.
                        cost_velocity[side,1] += sign*.005*smooth_rate(
                            t+(n+1)*model.opt.timestep-swing_recenter_start,.6)
                velocities.extend(cost_velocity.ravel())
            if window_track:
                # Protect the command actually executed before replanning.
                # Holding it for the full balance preview is not the future
                # policy and can reject a safely rising foot prematurely.
                swing_min = min(swing_min,float((vertices[side]@scratch.xmat[feet[side]].reshape(3,3).T
                                                +scratch.xpos[feet[side]])[:,2].min()))
                window_height = float((vertices[side]@scratch.xmat[feet[side]].reshape(3,3).T
                                       +scratch.xpos[feet[side]])[:,2].min())
            if n+1 == contact_steps:
                contact_features = feature(scratch,velocities)
            if n+1 == cost_steps:
                cost_features = feature(scratch,velocities)
            if geometry:
                prediction_path.append(scratch.qpos[adr].copy())
                normal = (scratch.xmat[feet[support]].reshape(3,3)@sole_normals[support])[1]
                normal_min,normal_max = min(normal_min,normal),max(normal_max,normal)
                distance,gradient,pair,local_distances = near_distance(scratch)
                pair_separations = np.minimum(pair_separations,local_distances)
                if distance < separation:
                    separation,closest_gradient,closest_pair = distance,gradient,pair
                    proximity_index = len(prediction_path)-1
        mujoco.mj_forward(model, scratch)
        assert len(velocities) == 6*VELOCITY_SAMPLES
        assert cost_features is not None
        # Contact forces and slip must describe the command actually held for
        # 20 ms. Body/COM tracking needs a longer preview to retain balance
        # authority; shortening that task to 20 ms made it almost insensitive.
        force_start = 12+2*model.nu+2+4
        cost_features[force_start:force_start+6] = contact_features[force_start:force_start+6]
        proximity_gradient,proximity_pair = closest_gradient,closest_pair
        # RMS bounds the force-weighted mean speed used by actual acceptance.
        # Integrate the next real command window, including touchdown transients.
        slip_window = np.linalg.norm(window_velocities,axis=2).sum(axis=0)*model.opt.timestep
        constrained = np.r_[q_min,q_max,separation,tilt_max,normal_min,normal_max,swing_min,window_height,pair_separations,slip_window]
        return np.r_[cost_features,CLEARANCE_WEIGHT*max(0,.0005-separation)], constrained

    def transition(new, t):
        nonlocal phase, phase_start, recovery_com,recovery_q,prepared_q,landing_origin,support_q
        nonlocal liftoff_origin,liftoff_rotation,descent_start
        nonlocal lift_hold_ctrl,lift_exit_q
        nonlocal recovery_rotation
        nonlocal recovery_route,recovery_route_start
        nonlocal recovery_origin,recovery_mapping
        nonlocal descent_clearance
        nonlocal liftoff_com,landing_toe
        nonlocal liftoff_capture,liftoff_start
        nonlocal liftoff_support_normal
        transitions.append({'time_s': t, 'from': phase, 'to': new})
        print('transition', side, phase, '->', new, round(t, 3), flush=True)
        phase, phase_start = new, t
        if new == 'transfer':
            prepared_q = data.qpos[adr].copy()
        if new == 'recover':
            recovery_com = data.subtree_com[base].copy()
            recovery_q = data.qpos[adr].copy()
            recovery_rotation = data.xmat[base].reshape(3,3)@initial_R.T
            recovery_origin = data.qpos.copy()
            # Reverse the measured transfer corridor. A straight joint blend
            # crosses the trunk/hip bracket even though both endpoints are safe.
            forward = [f for f in frames if f['phase'] in ('prepare','transfer')]
            recovery_route = (np.array([0.]+[f['time_s'] for f in forward]),
                np.vstack([initial_full_qpos]+[np.array(f['qpos']) for f in forward]))
            recovery_route_start = min(forward,key=lambda f:abs(
                f['com_world_mm'][1]/1000-recovery_com[1]))['time_s']
            scratch.qpos[:] = np.array([np.interp(recovery_route_start,recovery_route[0],q)
                                        for q in recovery_route[1].T])
            mujoco.mj_kinematics(model,scratch)
            before = scratch.xmat[feet[support]].reshape(3,3)@foot_tangents[support]
            after = data.xmat[feet[support]].reshape(3,3)@foot_tangents[support]
            yaw = np.arctan2(before[0]*after[1]-before[1]*after[0],before[:2]@after[:2])
            rotation = Rotation.from_rotvec([0.,0.,yaw]).as_matrix()
            translation = data.xpos[feet[support]]-rotation@scratch.xpos[feet[support]]
            translation[2] = 0.
            recovery_mapping = rotation,translation
        if new == 'lower':
            landing_origin = data.xpos[feet[side]].copy()
            lift_exit_q = data.qpos[adr].copy()
            descent_start = t
            descent_clearance = measure(data)[3][side]
        if new == 'lift':
            liftoff_support_normal = (data.xmat[feet[support]].reshape(3,3)@sole_normals[support])[:2].copy()
            liftoff_com = data.subtree_com[base].copy()
            liftoff_capture = (liftoff_com[:2]+data.subtree_linvel[base,:2]
                /np.sqrt(9.81/liftoff_com[2]))
            liftoff_start = t
            support_q = data.qpos[adr].copy()
            lift_hold_ctrl = data.ctrl.copy()
            sole_world = vertices[side]@data.xmat[feet[side]].reshape(3,3).T+data.xpos[feet[side]]
            landing_toe = vertices[side][np.argmin(sole_world[:,2])].copy()
            liftoff_origin = sole_world[np.argmin(sole_world[:,2])].copy()
            liftoff_rotation = data.xmat[feet[side]].reshape(3,3).copy()
            desired_support_normal[:] = [0., (-1 if support == 1 else 1)*np.sin(SUPPORT_ROLL_TARGET)]

    def swing_posture(lift):
        nonlocal swing_xy,swing_ik_residual
        swing_ids = leg_ids[5*side:5*side+5]
        posture = support_q.copy()
        mujoco.mj_copyData(scratch,model,data)
        entry_normal = liftoff_rotation@sole_normals[side]
        entry_tangent = liftoff_rotation@foot_tangents[side]
        heading = np.arctan2(entry_tangent[1],entry_tangent[0])
        landing_point = liftoff_origin.copy()
        if swing_recenter_start is not None:
            # The rolled/yawed release otherwise leaves the final stance too
            # wide. Recenter the landing pivot while it is actually airborne,
            # avoiding a loaded-foot correction during home recovery.
            landing_point[1] -= sign*.005*loads.smooth((t+horizon-swing_recenter_start)/.6)
        def residual(q):
            scratch.qpos[adr[swing_ids]] = q
            mujoco.mj_kinematics(model,scratch)
            rotation = scratch.xmat[feet[side]].reshape(3,3)
            point = scratch.xpos[feet[side]]+rotation@landing_toe
            height = (vertices[side]@rotation.T+scratch.xpos[feet[side]])[:,2].min()
            normal = rotation@sole_normals[side]
            tangent = rotation@foot_tangents[side]
            turn = np.arctan2(tangent[1],tangent[0])-heading
            return np.r_[100*(point[:2]-landing_point[:2]),100*(height-lift),
                         5*(normal[0]-entry_normal[0]),5*np.arctan2(np.sin(turn),np.cos(turn))]
        lower = model.jnt_range[joints[swing_ids],0]+np.deg2rad(.5)
        upper = model.jnt_range[joints[swing_ids],1]-np.deg2rad(.5)
        lower[3] = np.deg2rad(1)
        result = least_squares(residual,np.clip(data.qpos[adr[swing_ids]],lower,upper),
                               bounds=(lower,upper),max_nfev=60,
                               ftol=1e-8,xtol=1e-8,gtol=1e-8)
        posture[swing_ids] = result.x
        swing_ik_residual = residual(result.x).tolist()
        # These are position references only. Native dynamics and collision
        # checks still gate every executed command; scratch poses are not run.
        swing_xy = landing_point[:2].copy()
        return posture

    initial_time = data.time
    for step in range(round(35/DT)):
        if stop:
            break
        t = step*DT
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        f, nf, moment, height, tilt = measure(data)
        # A five-axis leg must be allowed to rock on a sole edge. Anchor the
        # currently loaded ground point, not the moving centre of the foot.
        for i,b in enumerate(feet):
            if f[i,2] > .5:
                points,normal_force = [],[]
                for ci,c in enumerate(data.contact):
                    if floor in c.geom and b in model.geom_bodyid[c.geom]:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model,data,ci,wrench)
                        if wrench[0] > .05:
                            points.append(c.pos.copy())
                            normal_force.append(wrench[0])
                point = (np.average(points,axis=0,weights=normal_force) if points
                         else np.array([-moment[i,1]/f[i,2],moment[i,0]/f[i,2],0.]))
                local_centres[i] = (point-data.xpos[b])@data.xmat[b].reshape(3,3)
                plane_xy[i] = point[:2]
        world_soles = [v@data.xmat[b].reshape(3,3).T+data.xpos[b] for b,v in zip(feet,vertices)]
        cop_lower = np.array([w[:,:2].min(axis=0)+.004 for w in world_soles])
        cop_upper = np.array([w[:,:2].max(axis=0)-.004 for w in world_soles])
        if phase == 'prepare':
            ready = (abs(data.subtree_com[base,2]-target_com[2]) < .003
                     and abs(data.subtree_com[base,0]-target_com[0]) < .003
                     and np.all(data.qpos[adr[knee_ids]] < np.deg2rad(85))
                     and abs(data.subtree_com[base,1]-initial_com[1]) < .005
                     and tilt < np.deg2rad(5)
                     and abs(data.qpos[adr[head_yaw]]-transfer_posture[head_yaw]) < np.deg2rad(3)
                     and abs(data.qpos[adr[head_ids[3]]]-transfer_posture[head_ids[3]]) < np.deg2rad(3)
                     and np.all(f[:,2] > 3))
            prepared_for = prepared_for+DT if ready else 0.
            if prepared_for >= .20:
                transfer_start = t
                transition('transfer',t)
            elif t >= 6.:
                failures.append('preparation_timeout')
                recovery_ctrl = data.ctrl.copy()
                transition('recover',t)
        if phase == 'transfer':
            capture_now = (data.subtree_com[base,:2]+data.subtree_linvel[base,:2]
                           /np.sqrt(9.81/data.subtree_com[base,2]))
            # A rolled support foot stands on an edge. Its unused projected
            # width cannot capture the body: the previous 20mm allowance
            # released the swing foot while capture was still 11mm outward.
            captured = (-.0015 < sign*(capture_now[1]-plane_xy[support,1]) < -.0003
                        and cop_lower[support,0] < capture_now[0] < cop_upper[support,0])
            # Position servos need a retraction command to release the residual
            # toe load. The air/hold test below still requires Fz < 0.5 N.
            unloaded_for = unloaded_for+DT if (f[side,2] < 3.0 and f[support,2] > 18 and captured) else 0.
            if unloaded_for >= .04:
                transition('lift', t)
            elif t-phase_start >= 12.:
                failures.append('unloading_timeout')
                recovery_ctrl = data.ctrl.copy()
                transition('recover', t)
        if phase == 'lift':
            airborne = (height[side] > .002 and f[side, 2] < .5
                and f[support,2] > 10 and height[support] < .001)
            if airborne and swing_recenter_start is None:
                swing_recenter_start = t
            clear_for = clear_for+DT if airborne else 0.
            longest = max(longest, clear_for)
            # Minimum airborne detection is not the requested lift height.
            # Start the hold timer only after the actual whole sole reaches it.
            target_clear = (height[side] >= REQUIRED_LIFT_HEIGHT and airborne)
            target_clear_for = target_clear_for+DT if target_clear else 0.
            longest_target_clear = max(longest_target_clear,target_clear_for)
            if target_clear_for+1e-9 >= .22:
                transition('lower', t)
            elif t-phase_start >= 4.:
                failures.append('requested_lift_height_timeout')
                transition('lower', t)
        if phase == 'lower':
            if landing_q is None and f[side,2] > .5 and height[side] < .001:
                # Follow real touchdown even if it precedes the nominal path.
                # Continuing to request zero force would unload the landed leg.
                landing_q = data.qpos[adr].copy()
                touchdown_time = t
                touchdown_com = data.subtree_com[base].copy()
                touchdown_rotation = Rotation.from_matrix(data.xmat[base].reshape(3,3)@initial_R.T).as_rotvec()
                touchdown_normals = np.concatenate([(data.xmat[b].reshape(3,3)@n)[:2] for b,n in zip(feet,sole_normals)])
                descent_start = t-.35
            if descent_start is None:
                descent_start = t
            # Recovery performs the recentering along the measured safe route.
            # Requiring an already centered/upright body here leaves lower
            # blending directly to prepared_q instead of taking that route.
            landed_for = landed_for+DT if (landing_q is not None
                and f[side,2] > .5 and f[support,2] > 10
                and np.all(height < .001)) else 0.
            if landed_for >= .10:
                recovery_ctrl = data.ctrl.copy()
                transition('recover', t)
            elif t-phase_start > 4.:
                failures.append('landing_timeout')
                recovery_ctrl = data.ctrl.copy()
                transition('recover', t)
        if phase == 'recover':
            source_time = recovery_source_time(t-phase_start)
            flat = all((data.xmat[b].reshape(3,3)@n)[2] > np.cos(np.deg2rad(5))
                       for b,n in zip(feet,sole_normals))
            if (home_alignment_start is None and (transfer_start is None or source_time <= transfer_start)
                    and flat and np.all(f[:,2]>3)):
                home_alignment_start = t-phase_start
            stable = (np.max(np.abs(data.qpos[adr]-initial_q)) < np.deg2rad(3)
                      and tilt < np.deg2rad(5) and np.max(np.abs(data.qvel)) < .15
                      and np.all(f[:, 2] > 3))
            recovered_for = recovered_for+DT if stable and t-phase_start >= 3 else 0.
            # Count elapsed stable intervals, not just qualifying samples.
            if recovered_for+1e-9 >= .5+DT:
                transition('done', t)
                break
        replaying = replay_active and step < len(preparation) and phase == preparation[step]['phase']
        if (replay_recovery and home_alignment_start is not None
                and (replay_until is None or t >= replay_until)):
            replaying = False
        replay_active = replaying
        if transfer_start is not None:
            reference_y = initial_com[1]+loads.smooth(transfer_clock/TRANSFER_DURATION)*(plane_xy[support,1]-initial_com[1])
            # Slow position AND velocity continuously; pausing only position
            # retained a nominal feedforward velocity and forced catch-up.
            transfer_rate = (float(np.clip((.012-abs(data.subtree_com[base,1]-reference_y))/.008,0.,1.))
                             if phase == 'transfer' and transfer_clock < TRANSFER_DURATION else 1.)
            transfer_clock += DT*transfer_rate
        # Recovery also needs enough balance preview to brake body momentum.
        # A 120 ms COM preview accepted commands that flattened the soles while
        # accelerating forward beyond the available pressure-point range.
        horizon = HORIZON if (phase in ('prepare','transfer','lift','recover')
                             or (phase == 'lower' and landing_q is None)) else .12
        # Track the next actual hold; retain braking and geometry preview beyond
        # that hold without treating its later contact forces as feedback error.
        lookahead_horizon = HORIZON if phase == 'prepare' or (phase == 'transfer' and transfer_clock < TRANSFER_DURATION) else .12
        ids = np.r_[leg_ids,head_ids] if phase == 'prepare' else leg_ids.copy()
        hold_ids = np.array([],dtype=int)
        fixed = [a for a in range(model.nu) if a not in ids]
        if replaying:
            replayed_frames += 1
            if phase not in replayed_phases:
                replayed_phases.append(phase)
            data.ctrl[:] = preparation[step]['ctrl']
            desired_zmp = np.array(preparation[step]['desired_ground_zmp_world_mm'])/1000
            desired_force = np.array(preparation[step]['desired_Fz_n'])
        elif phase != 'done':
            home_weight = 10.
            balance_weight = 1. if phase == 'prepare' else 1.+3.*loads.smooth(transfer_clock/TRANSFER_DURATION)
            if phase == 'lift' or (phase == 'lower' and landing_q is None):
                balance_weight = 20.
            if phase == 'lower' and landing_q is None:
                balance_weight = 50.
            if phase == 'recover':
                # Return to the ordinary double-support balance weight while
                # restoring home joints. Keeping transfer's fourfold COM task
                # held the hips away from home even after the body stood still.
                balance_weight = 1.+3.*loads.smooth(recovery_source_time(t+horizon-phase_start)/.8)
            weights[:2] = 400.*balance_weight
            weights[3:5] = 10.*balance_weight
            weights[2] = 5000. if phase == 'lower' and landing_q is None else 1000.
            normal_start = 12+2*model.nu+2
            weights[normal_start:normal_start+4] = [5.,.3,5.,.3]
            if phase in ('transfer','lift','lower'):
                weights[normal_start+2*side] = .3
            if phase in ('transfer','lift','lower'):
                weights[normal_start+2*support:normal_start+2*support+2] = [100.,20.]
            if phase == 'lower' and landing_q is not None:
                weights[normal_start:normal_start+4] = 5.
            weights[normal_start+4:normal_start+6] = .08
            if phase == 'lift':
                weights[normal_start+4+side] = 2.
            weights[anchor_start:anchor_start+4] = 200.
            slip_weight = 1000. if phase == 'prepare' else (15000. if phase == 'recover' else SLIP_WEIGHT)
            weights[anchor_start+4:anchor_start+4+6*VELOCITY_SAMPLES] = slip_weight/np.sqrt(VELOCITY_SAMPLES)
            if phase in ('lift','lower'):
                offset = anchor_start+2*side
                weights[offset:offset+2] = 400. if landing_q is None else 200.
            # Predicted contact_velocity is already zero in flight. Keep its
            # weight when a currently unloaded foot makes contact again within
            # the horizon, otherwise that contact can slide without a cost.
            future = t+horizon
            transfer_future = transfer_clock+horizon*transfer_rate
            target = initial_com.copy()
            target_velocity = np.zeros(3)
            blend = 0. if transfer_start is None else loads.smooth((future-transfer_start)/1.)
            capture_blend = 0. if transfer_start is None else loads.smooth(transfer_future/TRANSFER_DURATION)
            # Build capture velocity while both soles still carry load; waiting
            # for the retimed transfer to finish can deadlock its own clock.
            balance_blend = 0. if transfer_start is None else loads.smooth((transfer_future-.6*TRANSFER_DURATION)/1.5)
            force_blend = balance_blend if phase == 'transfer' else (0. if phase == 'prepare' else 1.)
            weights[normal_start+6:normal_start+10] = .5
            target[1] += capture_blend*(plane_xy[support,1]-initial_com[1])
            if phase == 'transfer':
                # Build capture velocity while the other toe can still push.
                target[1] -= sign*.020*balance_blend
            if transfer_start is not None:
                target_velocity[1] = transfer_rate*(plane_xy[support,1]-initial_com[1])*smooth_rate(transfer_future,TRANSFER_DURATION)
                if phase == 'transfer':
                    target_velocity[1] -= transfer_rate*sign*.020*smooth_rate(transfer_future-.6*TRANSFER_DURATION,1.5)
            if phase == 'lift' or (phase == 'lower' and landing_q is None):
                # Capture feedback needs the actual support point in flight.
                # Keeping the transfer's extra 20mm target accelerated the body
                # past the support edge and made the landing pose unreachable.
                shift = plane_xy[support,1]-liftoff_capture[1]
                target[1] = liftoff_capture[1]+loads.smooth((future-liftoff_start)/.3)*shift
                target_velocity[1] = smooth_rate(future-liftoff_start,.3)*shift
            target[0] += loads.smooth(future/3)*(target_com[0]-initial_com[0])
            target_velocity[0] = smooth_rate(future,3)*(target_com[0]-initial_com[0])
            target[2] += preparation_rise*loads.smooth(future/3)
            if phase in ('transfer','lift','lower'):
                target[2] += capture_blend*(reference['com'][2]-initial_com[2]-preparation_rise)
            if phase == 'lift' or (phase == 'lower' and landing_q is None):
                # Extend the loaded leg gradually, keeping the original home.
                # A rearward COM shift shortens its knee ground-force lever;
                # the smooth rise avoids a height jump at toe release.
                extension = loads.smooth((future-liftoff_start)/LIFT_RISE_TIME)
                target[0] = liftoff_com[0]-.010*extension
                target_velocity[0] = -.010*smooth_rate(future-liftoff_start,LIFT_RISE_TIME)
                if phase == 'lower':
                    # Restore vertical reach as the swing foot descends.
                    extension *= 1-loads.smooth((t+lookahead_horizon-descent_start)/.5)
                target[2] = liftoff_com[2]+.010*extension
            if phase == 'lower' and landing_q is not None:
                home_com = recovery_reference()[0]
                target[:2] = touchdown_com[:2]+loads.smooth((future-touchdown_time)/2.)*(home_com[:2]-touchdown_com[:2])
                target_velocity[:2] = smooth_rate(future-touchdown_time,2.)*(home_com[:2]-touchdown_com[:2])
            if phase == 'recover':
                home_com, home_rotation, home_yaw = recovery_reference()
                recovery_now = recovery_state(future-phase_start)
                recovery_next = recovery_state(future-phase_start+DT)
                target = recovery_now[0].copy()
                target_velocity = (recovery_next[0]-target)/DT
                blend = 0.
                source_time = recovery_source_time(future-phase_start)
                home_weight += 90.*loads.smooth((.8-source_time)/.8)
                for a in ids:
                    weights[12+a] = home_weight
            roll_target = float(np.clip(np.arctan2(initial_com[1]-target[1],target[2]),-.60,.60))
            # Lean before translating into the trunk/hip bracket corridor.
            weights[6] = 40. if phase == 'transfer' else (3. if phase in ('prepare','recover') else 8.)
            reference_com = target.copy()
            com, com_velocity = data.subtree_com[base].copy(), data.subtree_linvel[base].copy()
            omega = np.sqrt(9.81/max(com[2],.1))
            capture_point = com[:2]+com_velocity[:2]/omega
            desired_zmp = capture_point+(3./omega)*(capture_point-target[:2])-target_velocity[:2]/omega
            loaded = np.flatnonzero(f[:,2] > .5)
            if len(loaded):
                desired_zmp = np.clip(desired_zmp,cop_lower[loaded].min(axis=0),cop_upper[loaded].max(axis=0))
            acceleration = omega**2*(com[:2]-desired_zmp)
            velocity = np.clip(4*(target-com),-.15,.15)
            target[:2] = com[:2]+com_velocity[:2]*horizon+.5*acceleration*horizon**2
            velocity[:2] = com_velocity[:2]+acceleration*horizon
            sole_target = np.zeros(2)
            if phase == 'recover':
                sole_target = initial_height.copy()
            if phase == 'lift':
                sole_target[side] = LIFT_HEIGHT*loads.smooth((future-phase_start)/LIFT_RISE_TIME)
            elif phase == 'lower':
                sole_target[side] = descent_clearance*(1-loads.smooth((t+lookahead_horizon-descent_start)/.5))
                if landing_q is not None:
                    sole_target = initial_height.copy()
            # Tracking static body weight during a commanded COM descent fights
            # knee flexion. Include its vertical acceleration in the force task.
            vertical_acceleration = force_blend*float(np.clip(
                100*(target[2]-com[2])-20*com_velocity[2],-3.,3.))
            fz = model.body_subtreemass[base]*(9.81+vertical_acceleration)
            desired_cop = np.clip(np.tile(desired_zmp,(2,1)),cop_lower,cop_upper)
            desired_cop[:,1] = plane_xy[:,1]
            span = desired_cop[0,1]-desired_cop[1,1]
            left_fraction = np.clip((desired_zmp[1]-desired_cop[1,1])/span,0,1) if span>1e-6 else .5
            desired_force = fz*np.array([left_fraction,1-left_fraction])
            if phase == 'lift' or (phase == 'lower' and descent_start is None):
                desired_force[side],desired_force[support] = 0.,fz
                desired_cop[support] = np.clip(desired_zmp,cop_lower[support],cop_upper[support])
            if phase == 'transfer':
                # Keep the loaded swing toe as its real fore-aft pressure point;
                # distribute the remaining pitch moment over the support sole.
                desired_cop[side,0] = plane_xy[side,0]
                desired_cop[support,0] = np.clip(
                    (desired_zmp[0]*fz-desired_force[side]*desired_cop[side,0])/max(desired_force[support],1.),
                    cop_lower[support,0],cop_upper[support,0])
            desired_moment = np.column_stack((desired_cop[:,1]*desired_force,-desired_cop[:,0]*desired_force))
            if phase == 'transfer':
                unload_height = float(np.clip(unload_height+.002*(f[side,2]-desired_force[side])*DT,0,.005))
                sole_target[side] = unload_height
            # Unloading uses sub-millimetre leg-length admittance. The formal
            # lift is assessed at the requested whole-sole height and load.
            weights[12+2*model.nu+side] = 1000. if phase in ('lift','lower') else 100.
            if phase == 'lift':
                weights[12+2*model.nu+side] = 3000.
            posture = initial_q
            posture_velocity = np.zeros(model.nu)
            if phase == 'transfer':
                push = loads.smooth(transfer_future/TRANSFER_DURATION)
                posture = prepared_q+push*(transfer_posture-prepared_q)
                swing_ids = leg_ids[5*side:5*side+5]
                # Start yaw before COM retiming stalls the shortened-leg path.
                # Actual contact slip and native collision predictions still
                # gate each command; elapsed reference time cannot release lift.
                yaw_push = loads.smooth((transfer_future-.4)/(TRANSFER_DURATION-.4))
                posture[swing_ids[0]] = prepared_q[swing_ids[0]]+yaw_push*(transfer_posture[swing_ids[0]]-prepared_q[swing_ids[0]])
                for a in leg_ids:
                    weights[12+a] = 5.
                weights[12+swing_ids[4]] = 15.
                weights[12+swing_ids[0]] = 15.
                weights[12+roll_ids[support]] = 10.
                posture_velocity[swing_ids] = np.clip(6*(posture-data.qpos[adr])[swing_ids],-2.,2.)
            elif phase in ('lift','lower'):
                # Explicit phase defaults keep fresh and replayed transfer
                # prefixes from leaving different support ankle/yaw weights.
                weights[12+leg_ids] = 5.
                posture = (landing_q+loads.smooth((future-touchdown_time)/2.)*(prepared_q-landing_q)
                           if phase == 'lower' and landing_q is not None else
                           swing_posture(sole_target[side]))
                if landing_q is None:
                    support_axes = leg_ids[5*support:5*support+5]
                    posture[support_axes[1]] -= sign*np.deg2rad(2)*extension
                    posture[support_axes[2]] -= sign*np.deg2rad(8)*extension
                    posture[support_axes[3]] -= np.deg2rad(10)*extension
                    weights[12+support_axes[1]] = 20.
                    weights[12+support_axes[2:4]] = 15.
                for a in leg_ids[5*side:5*side+5]:
                    weights[12+a] = 25. if phase == 'lift' else 15.
                if phase in ('transfer','lift','lower'):
                    posture_velocity[leg_ids[5*side:5*side+5]] = np.clip(
                        8*(posture-data.qpos[adr])[leg_ids[5*side:5*side+5]],-3.,3.)
            if phase == 'recover':
                posture = recovery_now[1].copy()
                reference_velocity = (recovery_next[1]-posture)/DT
                posture_velocity[ids] = np.clip((reference_velocity
                    +4*(posture-data.qpos[adr]))[ids],-1.5,1.5)
            # The head COM is forward of its yaw axis. Turning toward the
            # support side shifts mass before the swing leg reaches full
            # extension; the original head/home pose returns during recovery.
            posture = posture.copy()
            yaw_blend,neck_blend = 1.,capture_blend
            if phase == 'prepare':
                yaw_blend,neck_blend = loads.smooth(future/2.5),0.
            if phase == 'recover':
                yaw_blend = float(np.clip((posture[head_yaw]-initial_q[head_yaw])/
                    (transfer_posture[head_yaw]-initial_q[head_yaw]),0.,1.))
                recovery_head = posture[head_ids].copy()
            posture[head_ids] = initial_q[head_ids]+neck_blend*(transfer_posture[head_ids]-initial_q[head_ids])
            posture[head_yaw] = initial_q[head_yaw]+yaw_blend*(transfer_posture[head_yaw]-initial_q[head_yaw])
            weights[12+head_ids] = 10.
            # The C-frame blocks a straight yaw/roll interpolation near 45 deg.
            # Lead with roll, pass that region, then unwind; reverse on return.
            end_yaw = abs(np.degrees(transfer_posture[head_yaw]-initial_q[head_yaw]))
            posture[head_ids[3]] = sign*np.deg2rad(np.interp(end_yaw*yaw_blend,
                [0.,12.,30.,46.,54.,min(68.,end_yaw-1.),end_yaw],
                [0.,-7.,-15.,-26.,-26.,-10.,sign*np.degrees(transfer_posture[head_ids[3]])]))
            if phase == 'recover':
                posture[head_ids] = recovery_head
            weights[12+head_ids[3]] = 30.
            posture_velocity[head_ids] = np.clip(4*(posture[head_ids]-data.qpos[adr[head_ids]]),-1.5,1.5)
            angular_velocity = [float(np.clip(-velocity[1]/.16,-1.5,1.5)),0.,0.]
            rotation_target = [roll_target,0.,0.]
            weights[7] = 30. if phase == 'recover' else 3.
            # Once both soles are flat, resist the torso/foot yaw reaction as
            # the head returns home. Weak yaw tracking let the head twist the
            # loaded soles, consume the slip budget and spoil final leg poses.
            if phase == 'recover' and home_alignment_start is not None:
                weights[8] = 40.
                weights[anchor_start+4+6*VELOCITY_SAMPLES:anchor_start+6+6*VELOCITY_SAMPLES] = 20.
            weights[9:12] = 5. if phase in ('lift','lower','recover') else .3
            if phase in ('transfer','lift') or (phase == 'lower' and landing_q is None):
                rotation_target = reference['rotation']*capture_blend
            elif phase == 'lower':
                body_blend = loads.smooth((future-touchdown_time)/2.)
                rotation_target = (1-body_blend)*touchdown_rotation+body_blend*np.array(rotation_target)
            heading_target = np.zeros(2)
            if phase == 'recover':
                rotation_target = Rotation.from_matrix(recovery_now[2]).as_rotvec()
                angular_velocity = np.clip(Rotation.from_matrix(recovery_next[2]@recovery_now[2].T).as_rotvec()/DT
                    +3*(rotation_target-Rotation.from_matrix(data.xmat[base].reshape(3,3)@initial_R.T).as_rotvec()),-.8,.8)
                heading_target[:] = recovery_now[4]
            elif phase in ('lift','lower'):
                heading_target[:] = landing_reference()[2]
                weights[anchor_start+4+6*VELOCITY_SAMPLES+side] = 2.
            # A rolled sole's world normal turns with its heading. Holding its
            # world X component at zero after yawing otherwise fights the roll
            # target and leaves the COM short of the real support edge.
            support_tangent = data.xmat[feet[support]].reshape(3,3)@foot_tangents[support]
            support_heading = np.arctan2(support_tangent[1],support_tangent[0])
            support_roll = (-1 if support == 1 else 1)*np.sin(SUPPORT_ROLL_TARGET)
            if phase == 'transfer' and transfer_clock >= TRANSFER_DURATION:
                support_roll = (-1 if support == 1 else 1)*np.sin(np.deg2rad(28))
            if phase == 'lift' or (phase == 'lower' and landing_q is None):
                support_roll = (-1 if support == 1 else 1)*np.linalg.norm(liftoff_support_normal)
            desired_support_normal[:] = support_roll*np.array([-np.sin(support_heading),np.cos(support_heading)])
            normal_target = np.r_[([0.,0.] if side == 0 else desired_support_normal),
                                   (desired_support_normal if side == 0 else [0.,0.])]
            if phase == 'transfer':
                normal_target[2*support:2*support+2] = initial_support_normal+capture_blend*(desired_support_normal-initial_support_normal)
            if phase == 'recover':
                normal_target = recovery_now[3]
            elif phase == 'lower':
                if landing_q is not None:
                    normal_target = touchdown_normals+loads.smooth((future-touchdown_time)/2.)*(np.concatenate([(r@n)[:2] for r,n in zip(initial_foot_R,sole_normals)])-touchdown_normals)
                else:
                    normal_target[2*side:2*side+2] = (liftoff_rotation@sole_normals[side])[:2]
            foot_xy_target = plane_xy.copy()
            if phase == 'lift':
                foot_xy_target[side] = swing_xy
            elif phase == 'lower':
                foot_xy_target[side] = swing_xy if landing_q is None else data.xpos[feet[side],:2]
            desired = np.r_[target, velocity, rotation_target,
                            angular_velocity, posture, posture_velocity, sole_target, normal_target,
                            desired_force, desired_moment.ravel(), foot_xy_target.ravel(),np.zeros(6*VELOCITY_SAMPLES),heading_target, 0]*weights
            u0 = data.ctrl.copy()
            for a in fixed:
                u0[a] = home[a]
            if len(hold_ids):
                u0[hold_ids] = lift_hold_ctrl[hold_ids]
            preair = phase == 'lift'
            if preair:
                # Start from the tested knee/ankle release, keeping support
                # commands close to the balanced entry until real clearance.
                release_ids = np.array([a for a in leg_ids[5*side:5*side+5] if a not in hold_ids])
                if t-phase_start < DT/2:
                    u0[release_ids] = posture[release_ids]+lift_hold_ctrl[release_ids]-support_q[release_ids]
            # Follow the verified head path with gravity/damping compensation,
            # instead of letting COM tracking move the head into a narrow gap.
            # These remain ordinary position targets held for the full 20 ms.
            if phase != 'prepare':
                mujoco.mj_copyData(scratch,model,data)
                scratch.qpos[adr[head_ids]] = posture[head_ids]
                scratch.qvel[:] = 0.
                mujoco.mj_kinematics(model,scratch)
                mujoco.mj_comPos(model,scratch)
                mujoco.mj_comVel(model,scratch)
                gravity = np.zeros(model.nv)
                mujoco.mj_rne(model,scratch,0,gravity)
                head_velocity = posture_velocity[head_ids]
                head_torque = (gravity[dofs[head_ids]]+model.dof_damping[dofs[head_ids]]*head_velocity
                               +model.dof_frictionloss[dofs[head_ids]]*np.clip(head_velocity/.1,-1,1))
                u0[head_ids] = np.clip(posture[head_ids]+head_torque/model.actuator_gainprm[head_ids,0],
                    model.actuator_ctrlrange[head_ids,0],model.actuator_ctrlrange[head_ids,1])
            # The native actuator clips force throughout a held position command.
            # Allow a target to remain saturated while the joint moves; clipping
            # its initial position error to force/kp loses that braking authority.
            effective_lo = model.actuator_ctrlrange[ids,0]+np.deg2rad(1)
            effective_hi = model.actuator_ctrlrange[ids,1]-np.deg2rad(1)
            if phase != 'lift':
                # Keep the numerical linearization inside the responsive part
                # of the position actuator while both feet carry the transfer.
                kp = model.actuator_gainprm[ids,0]
                effective_lo = np.maximum(effective_lo,data.qpos[adr[ids]]+model.actuator_forcerange[ids,0]/kp)
                effective_hi = np.minimum(effective_hi,data.qpos[adr[ids]]+model.actuator_forcerange[ids,1]/kp)
            if np.any(effective_lo>effective_hi):
                stop = 'inconsistent_torque_control_bounds'
                break
            u0[ids] = np.clip(u0[ids],effective_lo,effective_hi)
            trust_radius = .060
            trust_lo = np.maximum(effective_lo, u0[ids]-trust_radius)
            trust_hi = np.minimum(effective_hi, u0[ids]+trust_radius)
            # An old pressure point may lift during rolling. Ground friction
            # constrains actual contacts; old-point XY is a soft tracking task.
            # Actual loaded-contact slip remains an independent acceptance check.
            absolute_lower = np.r_[model.jnt_range[joints, 0]+np.deg2rad(.5),
                                   np.full(model.nu,-np.inf), .00022, 0.,
                                   -np.sin(SUPPORT_ROLL_LIMIT), -np.inf,
                                   (min(.0022,max(.00005,height[side]-.00005))
                                    if phase == 'lift' and f[side,2] < .5 and height[side] > .0001
                                    else -np.inf),
                                   (min(REQUIRED_LIFT_HEIGHT+.0005,height[side]+.0002)
                                    if phase == 'lift' and f[side,2] < .5 and height[side] > .0001
                                    else -np.inf),
                                   np.full(len(clearance_pairs),.0005),
                                   [0.,0.]]
            absolute_upper = np.r_[np.full(model.nu,np.inf),
                                   model.jnt_range[joints, 1]-np.deg2rad(.5),
                                   np.inf,np.deg2rad(40),np.inf,np.sin(SUPPORT_ROLL_LIMIT),np.inf,np.inf,
                                   np.full(len(clearance_pairs),np.inf),
                                   np.minimum(np.maximum(0.,(.0040 if phase == 'transfer' else .00495)-loaded_slip),
                                              np.where(f[:,2]<.5,np.inf,
                                                  (.001 if transfer_clock < TRANSFER_DURATION else .002)*DT)
                                              if phase == 'transfer' else np.inf)]
            if phase in ('lift','lower','recover'):
                for support_axis in leg_ids[5*support+2:5*support+4]:
                    absolute_lower[support_axis] = model.jnt_range[joints[support_axis],0]+np.deg2rad(.25)
                    absolute_upper[model.nu+support_axis] = model.jnt_range[joints[support_axis],1]-np.deg2rad(.25)
            if phase == 'lift' or (phase == 'lower' and landing_q is None):
                # Rolling past this tread facet moves the real support edge
                # outward abruptly. Retain its entry bank while extending.
                if support == 0:
                    absolute_upper[constraint_labels.index('support_sole_normal_y_max')] = liftoff_support_normal[1]+.002
                else:
                    absolute_lower[constraint_labels.index('support_sole_normal_y_min')] = liftoff_support_normal[1]-.002
            if phase == 'transfer' and transfer_clock >= TRANSFER_DURATION:
                if support == 0:
                    absolute_upper[constraint_labels.index('support_sole_normal_y_max')] = np.sin(np.deg2rad(28.7))
                else:
                    absolute_lower[constraint_labels.index('support_sole_normal_y_min')] = -np.sin(np.deg2rad(28.7))
            if phase == 'recover' and landing_q is not None:
                absolute_upper[constraint_labels.index('swing_next_hold_height')] = .0005
            absolute_lower[knee_ids] = np.maximum(absolute_lower[knee_ids],np.deg2rad(1))
            numeric_tolerance = np.r_[np.full(2*model.nu,1e-3), 1e-5,
                                      5e-4, 1e-4, 1e-4, 1e-5,1e-5,np.full(len(clearance_pairs),1e-5),np.full(2,1e-6)]
            valid = False
            feasible_restore = None
            stalled_retries = 0
            checked = errors = None
            for iteration in range(24):
                f0, q0 = predict(u0)
                separation_gradient = proximity_gradient.copy()
                separation_index = proximity_index
                active_pair = proximity_pair
                # Reuse the local sensitivity between nearby feedback samples.
                # Every proposed command still passes the full native dynamics
                # prediction; contact/phase changes or backtracking refresh it.
                braking_toe = (int(np.argmin(world_soles[side][:,2]))
                               if phase == 'transfer' and f[side,2] < 3. else None)
                toe_braking = ((phase == 'lift' and swing_recenter_start is None)
                               or (phase == 'lower' and height[side] < .004))
                signature = (phase,tuple(ids),tuple(f[:,2]>.5),tuple(active_pair or ()),lookahead_horizon,braking_toe,toe_braking)
                recompute = (derivative_cache is None or step % (4 if phase == 'prepare' else (8 if phase == 'transfer' and balance_blend < .05 else 2)) == 0 or iteration
                             or refresh_derivatives or signature != derivative_cache[0])
                if recompute:
                    D, Q = np.zeros((len(f0), len(ids))), np.zeros((len(q0), len(ids)))
                    difference = max(1e-5,CONTROL_DIFFERENCE/(4**stalled_retries)/(2**(iteration//4)))
                    for k, a in enumerate(ids):
                        joint_difference = difference
                        if ((phase == 'recover' and home_weight > 20.)
                                or (phase == 'transfer' and transfer_rate < .1)) and abs(data.qvel[dofs[a]]) < .05:
                            # A sub-breakaway probe sees little response at
                            # rest. Cover the existing actuator's static loss;
                            # only a checked nonlinear candidate is executed.
                            joint_difference = max(joint_difference,
                                2*model.dof_frictionloss[dofs[a]]/model.actuator_gainprm[a,0])
                        plus, minus = u0.copy(), u0.copy()
                        plus[a] += joint_difference
                        minus[a] -= joint_difference
                        fp, qp = predict(plus)
                        plus_pose = prediction_path[separation_index].copy()
                        fm, qm = predict(minus)
                        minus_pose = prediction_path[separation_index].copy()
                        D[:, k], Q[:, k] = (fp-fm)/(2*joint_difference), (qp-qm)/(2*joint_difference)
                        Q[2*model.nu,k] = separation_gradient@(plus_pose-minus_pose)/(2*joint_difference)
                    derivative_cache = (signature,D.copy(),Q.copy(),weights.copy())
                else:
                    _,cached_D,Q,cached_weights = derivative_cache
                    D = cached_D*(weights/cached_weights)[:,None]
                D[-1] = -CLEARANCE_WEIGHT*Q[2*model.nu] if q0[2*model.nu] < .0005 else 0.
                lo, hi = trust_lo-u0[ids], trust_hi-u0[ids]
                if stalled_retries:
                    radius = trust_radius/(4**stalled_retries)
                    lo, hi = np.maximum(lo,-radius),np.minimum(hi,radius)
                if np.any(lo > hi):
                    stop = 'inconsistent_control_bounds'
                    break
                # Smooth actual 50Hz commands, including across SQP iterations.
                # Undamped target reversals made the toe strike the floor.
                change_weight = 10000. if phase in ('lift','lower','recover') else 0.
                H = D.T@D+np.eye(len(ids))*(2.+change_weight)
                g = -D.T@(desired-f0)+change_weight*(u0[ids]-data.ctrl[ids])
                base_feasible = np.max(np.maximum(absolute_lower-q0,q0-absolute_upper)/numeric_tolerance) <= 1
                if not base_feasible:
                    # Restore feasibility with the smallest command change.
                    # A large tracking step near contact/clearance boundaries
                    # repeatedly invalidated its own linear constraint model.
                    H = np.eye(len(ids))
                    g = np.zeros(len(ids))
                # Leave room for nonlinear error instead of aiming exactly at
                # the acceptance tolerance on every active constraint.
                lower, upper = absolute_lower-q0-.5*numeric_tolerance, absolute_upper-q0+.5*numeric_tolerance
                # q_min/q_max include the already measured initial state.
                # Do not demand that an immutable endpoint recover half a
                # tolerance instantly; the nonlinear acceptance stays unchanged.
                lower[:model.nu] = np.minimum(lower[:model.nu],
                    np.maximum(data.qpos[adr],absolute_lower[:model.nu]-numeric_tolerance[:model.nu])-q0[:model.nu])
                upper[model.nu:2*model.nu] = np.maximum(upper[model.nu:2*model.nu],
                    np.minimum(data.qpos[adr],absolute_upper[model.nu:2*model.nu]+numeric_tolerance[model.nu:2*model.nu])-q0[model.nu:2*model.nu])
                scales = np.r_[np.full(2*model.nu,.1), .01, 1., .1, .1, .01,.01,np.full(len(clearance_pairs),.01),np.full(2,.001)]
                A = sparse.vstack((sparse.eye(len(ids)), sparse.csc_matrix(Q/scales[:,None])),format='csc')
                qp = osqp.OSQP()
                qp.setup(P=sparse.csc_matrix((H+H.T)/2), q=g, A=A,
                         l=np.r_[lo,lower/scales],u=np.r_[hi,upper/scales],verbose=False,
                         eps_abs=1e-6,eps_rel=1e-5,max_iter=20000,polishing=False)
                result = qp.solve(raise_error=False)
                # An iteration-limited QP may still supply a feasible iterate;
                # the linear residual and native nonlinear prediction gate it.
                if result.info.status_val not in (1,2,7) or result.x is None or not np.all(np.isfinite(result.x)):
                    if not recompute:
                        derivative_cache = None
                        continue
                    stop = f'prediction_solve_failed: {result.info.status}'
                    rows = np.vstack((Q/scales[:,None],-Q/scales[:,None]))
                    rhs = np.r_[upper/scales,-lower/scales]
                    slack = np.vstack((-np.eye(len(lower)),-np.eye(len(lower))))
                    finite = np.isfinite(rhs)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore',message='Unrecognized options detected')
                        elastic = linprog(np.r_[np.zeros(len(ids)),np.ones(len(lower))],
                            A_ub=np.column_stack((rows,slack))[finite], b_ub=rhs[finite],
                            bounds=list(zip(lo,hi))+[(0,None)]*len(lower),
                            method='highs',options={'threads':1})
                    if elastic.success:
                        labels = constraint_labels
                        prediction_failure = {'time_s':t,'solver':result.info.status,
                            'linear_infeasibility':[{ 'constraint':labels[k], 'minimum_relaxation':float(value*scales[k])}
                             for k,value in enumerate(elastic.x[len(ids):]) if value>1e-6]}
                    break
                violation = max(float(np.max(lower-Q@result.x)), float(np.max(Q@result.x-upper)),
                                float(np.max(lo-result.x)),float(np.max(result.x-hi)))
                if violation > 1e-5:
                    stop = f'prediction_solve_residual: {violation}'
                    break
                baseline_cost = float(np.sqrt(np.sum((f0-desired)**2)
                    +change_weight*np.sum((u0[ids]-data.ctrl[ids])**2)))
                # Backtrack within a known safe prediction before relinearizing
                # across a contact or closest-surface transition.
                fractions = (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,0.)
                best_restoration = None
                restoration_attempts = []
                for fraction in fractions:
                    candidate = u0.copy()
                    candidate[ids] += fraction*result.x
                    predicted_features, checked = predict(candidate)
                    errors = np.maximum(absolute_lower-checked, checked-absolute_upper)
                    candidate_cost = float(np.sqrt(np.sum((predicted_features-desired)**2)
                        +change_weight*np.sum((candidate[ids]-data.ctrl[ids])**2)))
                    infeasibility = float(np.max(np.maximum(errors,0)/numeric_tolerance))
                    restoration_attempts.append({'fraction':fraction,'worst_constraint':constraint_labels[int(np.argmax(errors/numeric_tolerance))],
                        'error_ratio':infeasibility,'cost':candidate_cost,'closest_pair':proximity_pair,'separation_mm':float(checked[2*model.nu]*1000)})
                    if best_restoration is None or (infeasibility,candidate_cost) < best_restoration[:2]:
                        best_restoration = (infeasibility,candidate_cost,candidate.copy(),checked.copy(),
                                            errors.copy(),predicted_features.copy(),fraction)
                    if (np.max(errors/numeric_tolerance) <= 1
                            and (not base_feasible or candidate_cost <= baseline_cost+1e-6)):
                        valid = True
                        break
                if valid:
                    if base_feasible and fraction == 0. and stalled_retries < 2:
                        # Contact transitions can make a 4 mrad secant point
                        # uphill. Retry locally before accepting no progress.
                        feasible_restore = (candidate.copy(),checked.copy(),errors.copy(),
                            predicted_features.copy(),fraction,baseline_cost,candidate_cost)
                        u0 = candidate.copy()
                        stalled_retries += 1
                        derivative_cache = None
                        valid = False
                        continue
                    if not base_feasible and iteration < 23:
                        # A feasibility repair is the start of optimization,
                        # not a substitute for the body's balance objective.
                        feasible_restore = (candidate.copy(),checked.copy(),errors.copy(),
                            predicted_features.copy(),fraction,baseline_cost,candidate_cost)
                        u0 = candidate.copy()
                        valid = False
                        continue
                    break
                _,candidate_cost,candidate,checked,errors,predicted_features,fraction = best_restoration
                u0 = candidate
            if not valid and feasible_restore is not None and (stop is None or stop.startswith('prediction_')):
                candidate,checked,errors,predicted_features,fraction,baseline_cost,candidate_cost = feasible_restore
                valid = True
                stop = None
            if not valid:
                labels = constraint_labels
                if checked is not None:
                    k = int(np.argmax(errors/numeric_tolerance))
                    prediction_failure.update({'time_s':t,'constraint':labels[k],
                        'predicted':float(checked[k]),'lower':float(absolute_lower[k]) if np.isfinite(absolute_lower[k]) else None,
                        'upper':float(absolute_upper[k]) if np.isfinite(absolute_upper[k]) else None,
                        'violation':float(errors[k]),'numerical_tolerance':float(numeric_tolerance[k]),'iterations':iteration+1})
                stop = stop or 'nonlinear_prediction_constraint'
                prediction_failure['active_body_pair'] = active_pair
                if checked is not None:
                    prediction_failure['restoration_attempts'] = restoration_attempts
                    prediction_failure['last_linear_step_rad'] = result.x.tolist() if result.x is not None else None
                break
            data.ctrl[:] = candidate
            refresh_derivatives = fraction < .5
        if stop:
            break
        control_error = data.ctrl-data.qpos[adr]
        for _ in range(round(DT/model.opt.timestep)):
            mujoco.mj_step(model, data)
            peak_torque = np.maximum(peak_torque, np.abs(data.actuator_force))
            executed_duration += model.opt.timestep
            torque_squared += data.actuator_force**2*model.opt.timestep
            saturated = np.abs(data.actuator_force) >= .999*torque_caps
            saturation_time += saturated*model.opt.timestep
            saturation_run = np.where(saturated, saturation_run+model.opt.timestep, 0.)
            longest_saturation_run = np.maximum(longest_saturation_run, saturation_run)
            if phase == 'lift':
                lift_duration += model.opt.timestep
                lift_peak_torque = np.maximum(lift_peak_torque,np.abs(data.actuator_force))
                lift_torque_squared += data.actuator_force**2*model.opt.timestep
                lift_saturation_time += (np.abs(data.actuator_force) >= .999*torque_caps)*model.opt.timestep
            margin = np.minimum(data.qpos[adr]-model.jnt_range[joints, 0],
                                model.jnt_range[joints, 1]-data.qpos[adr])
            min_joint_margin = np.minimum(min_joint_margin, margin)
            slip_force = np.zeros(2)
            slip_velocity = np.zeros(2)
            for ci, contact in enumerate(data.contact):
                if floor in contact.geom:
                    body = next((int(model.geom_bodyid[g]) for g in contact.geom if g != floor),-1)
                    if body in feet:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model,data,ci,wrench)
                        if wrench[0] > .05:
                            i = feet.index(body)
                            jac = np.zeros((3,model.nv))
                            mujoco.mj_jac(model,data,jac,None,contact.pos,body)
                            speed = float(np.linalg.norm((jac@data.qvel)[:2]))
                            slip_force[i] += wrench[0]
                            slip_velocity[i] += wrench[0]*speed
                            peak_slip_speed[i] = max(peak_slip_speed[i],speed)
                    else:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(model,data,ci,wrench)
                        if np.linalg.norm(wrench[:3]) > .05:
                            stop = 'nonfoot_ground_contact'
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                force = float(np.linalg.norm(wrench[:3]))
                if force > .05:
                    pair = ' / '.join(sorted(model.body(int(model.geom_bodyid[g])).name for g in contact.geom))
                    self_peaks[pair] = max(self_peaks.get(pair, 0.), force)
                    stop = 'self_contact'
            loaded_slip += np.divide(slip_velocity,slip_force,out=np.zeros(2),where=slip_force>.5)*model.opt.timestep
            if margin.min() < -np.deg2rad(.05):
                stop = 'joint_limit'
            if any(data.warning.number):
                stop = 'numerical_warning'
            if stop:
                break
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        f, nf, moment, height, tilt = measure(data)
        actual_separation,_,actual_pair,_ = near_distance(data)
        actual_min_separation = min(actual_min_separation,actual_separation)
        if actual_separation < .000195:
            stop = 'self_clearance'
            prediction_failure['actual_body_pair'] = actual_pair
        if nf:
            stop = 'nonfoot_ground_contact'
        if tilt > np.deg2rad(45):
            stop = 'tilt_limit'
        frames.append({'time_s': float(data.time-initial_time), 'phase': phase, 'qpos': data.qpos.tolist(),
                       'transfer_reference_time_s': transfer_clock,
                       'transfer_reference_rate': transfer_rate,
                       'recovery_home_alignment_age_s':home_alignment_start,
                       'recovery_home_posture_weight':None if replaying or phase != 'recover' else home_weight,
                       'qvel': data.qvel.tolist(), 'ctrl': data.ctrl.tolist(),
                       'control_error': control_error.tolist(),
                       'posture_target_deg':None if replaying else np.degrees(posture).tolist(),
                       'sole_target_mm':None if replaying else (sole_target*1000).tolist(),
                       'swing_ik_scaled_residual':swing_ik_residual,
                       'optimizer_step_fraction':None if replaying else float(fraction),
                       'optimizer_iterations':None if replaying else iteration+1,
                       'optimizer_stalled_retries':None if replaying else stalled_retries,
                       'prediction_cost_before':None if replaying else baseline_cost,
                       'prediction_cost_after':None if replaying else candidate_cost,
                       'predicted_Fz_n':None if replaying else (predicted_features[normal_start+4:normal_start+6]/weights[normal_start+4:normal_start+6]).tolist(),
                       'predicted_com_world_mm':None if replaying else (1000*predicted_features[:3]/weights[:3]).tolist(),
                       'reference_com_world_mm':None if replaying else (reference_com*1000).tolist(),
                       'prediction_tracking_cost':None if replaying else {
                           name:float(np.linalg.norm((predicted_features-desired)[a:b]))
                           for name,a,b in [('COM',0,6),('body',6,12),('joints',12,12+2*model.nu),
                                            ('sole',12+2*model.nu,normal_start+4),
                                            ('force',normal_start+4,normal_start+6),
                                            ('moment',normal_start+6,anchor_start),
                                            ('anchor',anchor_start,anchor_start+4),
                                            ('slip',anchor_start+4,anchor_start+4+6*VELOCITY_SAMPLES)]},
                       'torque_nm': data.actuator_force.tolist(), 'Fz_n': f[:, 2].tolist(),
                       'sole_min_height_mm': (height*1000).tolist(), 'tilt_deg': float(np.degrees(tilt)),
                       'com_world_mm': (data.subtree_com[base]*1000).tolist(),
                       'capture_point_world_mm': ((data.subtree_com[base,:2]+data.subtree_linvel[base,:2]/np.sqrt(9.81/data.subtree_com[base,2]))*1000).tolist(),
                       'desired_ground_zmp_world_mm': (desired_zmp*1000).tolist(),
                       'desired_Fz_n': desired_force.tolist(),
                       'desired_vertical_acceleration_m_s2':None if replaying else vertical_acceleration,
                       'support_normal_xy': (data.xmat[feet[support]].reshape(3,3)@sole_normals[support])[:2].tolist(),
                       'foot_body_displacement_mm': ((data.xpos[feet]-initial_foot_pos)*1000).tolist(),
                       'loaded_slip_distance_mm':(loaded_slip*1000).tolist(),
                       'joint_error_to_settled_home_deg':np.degrees(np.abs(data.qpos[adr]-initial_q)).tolist(),
                       'cop_world_mm': [([-m[1]/ff[2]*1000, m[0]/ff[2]*1000] if ff[2]>.05 else None)
                                        for ff,m in zip(f,moment)]})
        if step % 5 == 0:
            (CACHE/f'{"left" if side == 0 else "right"}_progress.json').write_text(json.dumps({
                'phase':phase,'time_s':float(data.time-initial_time),'frames':frames[-26:],
                'loaded_slip_distance_mm':(loaded_slip*1000).tolist()}))
        if step % 25 == 0:
            print(side, round(t, 2), phase, 'COM', np.round(data.subtree_com[base]*1000, 1).tolist(),
                  'Fz', np.round(f[:, 2], 2).tolist(), 'soleZ', np.round(height*1000, 2).tolist(),
                  'tilt', round(np.degrees(tilt), 2), flush=True)
        if stop:
            break
    assert [loads.digest(p) for p in PATHS] == hashes
    assert loads.digest(__file__) == source_hash, 'Script changed during simulation'
    assert loads.digest(loads.__file__) == helper_hash, 'Load helper changed during simulation'
    assert all(np.array_equal(a,b) for a,b in zip(original_actuators,
        (model.actuator_gainprm, model.actuator_biasprm, model.actuator_forcerange,
         model.actuator_ctrlrange, model.jnt_range)))
    assert np.array_equal(model.geom_margin, normal_margins)
    assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    if phase != 'done' and not stop:
        failures.append('overall_motion_timeout')
    passed = (phase == 'done' and longest_target_clear+1e-9 >= .22 and not stop and not failures
              and max(loaded_slip) <= .005 and actual_min_separation >= .0002
              and min_joint_margin.min() >= -np.deg2rad(1e-4) and not replay_prefix)
    report = {'side': 'left' if side == 0 else 'right', 'swing_side_index': side,
              'status': 'passed' if passed else 'failed', 'stop': stop, 'failures': failures,
              'prediction_failure':prediction_failure,
              'formal_hashes': hashes, 'simulation_script_sha256': source_hash,
              'endpoint_reference': {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in reference.items()},
              'load_helper_sha256': helper_hash, 'osqp_version':osqp.__version__,
              'diagnostic_prefix_replayed': replay_prefix, 'replayed_prefix_frames':replayed_frames,
              'replayed_prefix_phases':replayed_phases,
              'control_interface':'Native position targets optimized and held at 50 Hz. Simulation only; hardware interface unvalidated.',
              'longest_clear_unloaded_s': longest, 'recovered_stable_s': recovered_for,
              'required_lift_height_mm':REQUIRED_LIFT_HEIGHT*1000,
              'lift_reference_height_mm':LIFT_HEIGHT*1000,'lift_rise_duration_s':LIFT_RISE_TIME,
              'longest_target_height_hold_counter_s':longest_target_clear,
              'lift_torque_metrics':{'duration_s':lift_duration,'peak_torque_nm':lift_peak_torque.tolist(),
                  'rms_torque_nm':np.sqrt(lift_torque_squared/max(lift_duration,model.opt.timestep)).tolist(),
                  'saturation_duration_s':lift_saturation_time.tolist(),
                  'saturation_threshold_fraction':.999},
              'peak_torque_nm': peak_torque.tolist(), 'min_joint_margin_deg': np.degrees(min_joint_margin).tolist(),
              'all_phase_torque_metrics': {'duration_s': executed_duration,
                  'rms_torque_nm': np.sqrt(torque_squared/max(executed_duration,model.opt.timestep)).tolist(),
                  'saturation_duration_s': saturation_time.tolist(),
                  'longest_saturation_duration_s': longest_saturation_run.tolist(),
                  'saturation_threshold_fraction': .999},
              'nonfloor_contact_peaks_n': self_peaks, 'warnings': data.warning.number.tolist(),
              'loaded_slip_distance_mm':(loaded_slip*1000).tolist(),
              'peak_loaded_contact_slip_speed_mm_s':(peak_slip_speed*1000).tolist(),
              'sampled_min_self_separation_mm':actual_min_separation*1000,
              'actuator_order': [model.actuator(a).name for a in range(model.nu)],
              'settled_home_qpos':initial_q.tolist(),
              'recovery_reference':'Original home joints and COM relative to the current two-foot planar frame; global translation/yaw allowed.',
              'transitions': transitions, 'target_com_world_mm': (target_com*1000).tolist(), 'unload_height_mm':unload_height*1000, 'frames': frames,
              'resources': {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            'address_space_cap_bytes':4_000_000_000, 'native_threads':1}}
    CACHE.mkdir(exist_ok=True)
    path = CACHE/f'{report["side"]}.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    signal.signal(signal.SIGINT,old_signal)
    print('Saved', path, report['status'], stop, failures, 'clear_s', longest, flush=True)
    return report


if __name__ == '__main__':
    if sys.argv[1:2] in (['left'], ['right']) and sys.argv[2:] in ([],['--replay-prefix'],['--replay-prepare'],['--replay-recovery']):
        trial(0 if sys.argv[1] == 'left' else 1, bool(sys.argv[2:]),
              sys.argv[2:] == ['--replay-prepare'],sys.argv[2:] == ['--replay-recovery'])
    else:
        raise SystemExit('usage: control_zp10d_liftoff.py left | right')
