"""Audit necessary dynamic conditions for the current fitted Placo demonstration.

Run through scripts/cad/run_memory_limited.py. This solves an optimistic contact
force allocation, not a controller, free-base replay or walking acceptance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import mujoco
import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.spatial import ConvexHull

from reference import DATA, ROOT, SCENE, source_hashes


def contact_allocation(model, data, points, bodies, friction):
    """Minimize peak normalized motor torque with optimistic floor contacts."""
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    assert np.all(model.actuator_gear[:, 0] == 1)
    assert np.all(model.actuator_gear[:, 1:] == 0)
    caps = np.max(abs(model.actuator_forcerange), axis=1)
    nf, nu = 3 * len(points), model.nu
    # Variables: point forces, motor torques, joint dry friction, peak ratio.
    size = nf + 2 * nu + 1
    equality = np.zeros((model.nv, size))
    jac = np.zeros((3, model.nv))
    for i, (point, body) in enumerate(zip(points, bodies)):
        mujoco.mj_jac(model, data, jac, None, point, body)
        equality[:, 3 * i:3 * i + 3] = jac.T
    equality[dofs, nf + np.arange(nu)] = 1.
    equality[dofs, nf + nu + np.arange(nu)] = 1.
    required = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, required, data.qacc)
    required += data.qfrc_bias - data.qfrc_passive
    inequalities = []
    for j, cap in enumerate(caps):
        for sign in (-1, 1):
            row = np.zeros(size)
            row[nf + j], row[-1] = sign, -cap
            inequalities.append(row)
    for j in range(len(points)):
        for angle in np.arange(8) * np.pi / 4:
            row = np.zeros(size)
            row[3 * j:3 * j + 3] = [np.cos(angle), np.sin(angle), -friction]
            inequalities.append(row)
    bounds = [(0., None) if j % 3 == 2 else (None, None) for j in range(nf)]
    bounds += [(None, None)] * nu
    # Giving dry friction either sign makes this a generous necessary test.
    bounds += [(-float(f), float(f)) for f in model.dof_frictionloss[dofs]]
    bounds += [(0., None)]
    objective = np.zeros(size)
    objective[-1] = 1.
    with warnings.catch_warnings():
        # SciPy forwards this native HiGHS option after emitting a warning.
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*threads', category=OptimizeWarning)
        result = linprog(objective, A_ub=np.asarray(inequalities),
                         b_ub=np.zeros(len(inequalities)), A_eq=equality,
                         b_eq=required, bounds=bounds, method='highs',
                         options={'threads': 1})
        if result.status == 4:
            result = linprog(objective, A_ub=np.asarray(inequalities),
                             b_ub=np.zeros(len(inequalities)), A_eq=equality,
                             b_eq=required, bounds=bounds, method='highs-ipm',
                             options={'threads': 1, 'presolve': False})
    if result.status not in (0, 2):
        raise RuntimeError(f'Contact LP failed: {result.message}')
    if not result.success:
        return dict(feasible=False, min_peak_torque_ratio=None)
    assert np.max(abs(equality @ result.x - required)) < 1e-6
    torques = result.x[nf:nf + nu]
    limiting = int(np.argmax(abs(torques) / caps))
    return dict(feasible=True, min_peak_torque_ratio=float(result.x[-1]),
                limiting_actuator=model.actuator(limiting).name,
                motor_torque_nm=torques.tolist())


def support_points(model, data, sites, vertices, supports, *, actual_contacts=True):
    points, bodies, heights = [], [], []
    for side, site in enumerate(sites):
        world = vertices[side] @ data.site_xmat[site].reshape(3, 3).T + data.site_xpos[site]
        heights.append(float(world[:, 2].min()))
        if supports[side]:
            if actual_contacts:
                # The 0.2 mm allowance matches the reference geometry check.
                # A banked foot may only load its low edge, not the whole sole.
                hull = world[world[:, 2] <= .0002].copy()
            else:
                # Full projected sole is more generous than a rolled foot's edge.
                hull = world[ConvexHull(world[:, :2]).vertices].copy()
            hull[:, 2] = 0.
            points.extend(hull)
            bodies.extend([model.site_bodyid[site]] * len(hull))
    return points, bodies, heights


def check_contact_geometry():
    """A rolled sole cannot balance gravity on its raised corners."""
    angle = np.deg2rad(15.)
    lift = .03 * np.sin(angle)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <worldbody><body pos="0 0 .2"><freejoint/>
        <inertial mass="1" pos="0 0 0" diaginertia=".01 .01 .01"/>
        <body pos="0 0 {lift - .2}" euler="15 0 0">
          <joint name="ankle" axis="0 1 0"/>
          <inertial mass=".1" pos="0 0 0" diaginertia=".001 .001 .001"/>
          <site name="sole"/>
        </body></body></worldbody>
      <actuator><position joint="ankle" forcelimited="true" forcerange="-1 1"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.
    vertices = [np.array([[x, y, 0.] for x in (-.05, .05) for y in (-.03, .03)])]
    points, bodies, _ = support_points(model, data, [0], vertices, [True], actual_contacts=False)
    projected = contact_allocation(model, data, points, bodies, .6)
    points, bodies, _ = support_points(model, data, [0], vertices, [True])
    actual = contact_allocation(model, data, points, bodies, .6)
    assert projected['feasible'] and not actual['feasible'], (projected, actual)
    data.qpos[2] += .005
    mujoco.mj_forward(model, data)
    points, _, _ = support_points(model, data, [0], vertices, [True], actual_contacts=True)
    assert not points, 'An airborne foot cannot carry a floor load'
    print('PASS: projected sole accepts false support; low edge rejects it; airborne foot has no support')


def assess(candidate, output, data_dir=DATA, *, actual_contacts=True):
    meta = json.loads((data_dir / 'reference.json').read_text())
    hashes = source_hashes()
    assert hashes == meta['source_hashes']
    polynomial_hash = hashlib.sha256((data_dir / 'polynomial_coefficients.pkl').read_bytes()).hexdigest()
    assert polynomial_hash == meta['reward_reference']['reference_sha256']
    raw = np.load(candidate)
    assert json.loads(str(raw['metadata']))['source_hashes'] == hashes
    with (data_dir / 'polynomial_coefficients.pkl').open('rb') as stream:
        records = pickle.load(stream)
    record = next(r for key, r in records.items() if not key.startswith('0.0_'))
    assert record['basis'] == 'periodic_cubic'
    coefficients = np.asarray([record['coefficients'][f'dim_{j}'] for j in range(16)])
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    names = meta['joint_names']
    addresses = [model.joint(n).qposadr[0] for n in names]
    velocities = [model.joint(n).dofadr[0] for n in names]
    sites = [model.site(s + '_foot').id for s in ('left', 'right')]
    vertices = [np.asarray(v) for v in meta['reward_reference']['foot_vertices_in_site']]
    floor = model.geom('floor').id
    friction = float(model.geom_friction[floor, 0])
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_fwdPosition(model, data)
    if actual_contacts:
        # Published home has a 0.2 mm startup air gap. Ground the sanity-check
        # pose before asking which sole vertices can carry a static load.
        _, _, home_heights = support_points(model, data, sites, vertices, [False, False])
        data.qpos[2] -= min(home_heights)
        mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    data.qacc[:] = 0.
    points, bodies, _ = support_points(model, data, sites, vertices, [True, True],
                                      actual_contacts=actual_contacts)
    static_check = contact_allocation(model, data, points, bodies, friction)
    assert static_check['feasible'] and static_check['min_peak_torque_ratio'] < 1., static_check
    frames, base = raw['dense_frames'], raw['dense_base']
    positions = np.zeros((len(frames), model.nq))
    positions[:, :3] = base[:, :3, 3]
    for i, rotation in enumerate(base[:, :3, :3]):
        mujoco.mju_mat2Quat(positions[i, 3:7], rotation.ravel())
    positions[:, addresses] = frames[:, :16]
    count = coefficients.shape[2]
    dt = 1. / meta['sample_fps']
    start = record['start_offset'] * (meta['sample_fps'] // record['fps'])
    np.testing.assert_allclose(coefficients[:, 3].T, frames[start:start + count, :16], atol=2e-6)
    rows = []
    peak_speed, peak_acceleration = np.zeros(model.nu), np.zeros(model.nu)
    motor_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    for phase, i in enumerate(range(start, start + count)):
        data.qpos[:] = positions[i]
        forward, backward = np.zeros(model.nv), np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, forward, dt, positions[i], positions[i + 1])
        mujoco.mj_differentiatePos(model, backward, dt, positions[i], positions[i - 1])
        data.qvel[:] = (forward - backward) / 2
        acceleration = (forward + backward) / dt
        # Check the fitted joint reference itself, including analytic derivatives.
        data.qvel[velocities] = coefficients[:, 2, phase] / record['period']
        acceleration[velocities] = 2 * coefficients[:, 1, phase] / record['period'] ** 2
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        data.qacc[:] = acceleration
        peak_speed = np.maximum(peak_speed, abs(data.qvel[motor_dofs]))
        peak_acceleration = np.maximum(peak_acceleration, abs(acceleration[motor_dofs]))
        supports = frames[i, 32:34].astype(bool)
        points, bodies, heights = support_points(model, data, sites, vertices, supports,
                                                 actual_contacts=actual_contacts)
        try:
            result = contact_allocation(model, data, points, bodies, friction)
        except RuntimeError as error:
            raise RuntimeError(f'phase={phase * dt:.3f} s: {error}') from error
        rows.append(dict(phase_s=phase * dt, support=supports.astype(int).tolist(),
                         whole_foot_height_m=heights, com_world_m=data.subtree_com[1].tolist(), **result))
    feasible = [r['min_peak_torque_ratio'] for r in rows if r['feasible']]
    report = dict(source_hashes=hashes, reference_sha256=polynomial_hash,
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  sample_interval_s=dt, samples=len(rows), period_s=record['period'],
                  native_model='full unchanged CAD-derived MuJoCo model',
                  contact_geometry=('vertices within 0.2 mm of the actual floor' if actual_contacts
                                    else 'entire projected sole, including raised vertices'),
                  method='inverse dynamics and optimistic unilateral contact-force LP',
                  joint_derivatives='analytic derivatives of the fitted periodic cubic',
                  base_derivatives='central differences of cached 200 Hz Placo base poses',
                  no_contact_force_solution_samples=sum(not r['feasible'] for r in rows),
                  within_motor_cap_samples=sum(r['feasible'] and r['min_peak_torque_ratio'] <= 1. for r in rows),
                  minimum_peak_torque_ratio_range=[min(feasible), max(feasible)] if feasible else None,
                  peak_whole_foot_clearance_m=np.max([r['whole_foot_height_m'] for r in rows], axis=0).tolist(),
                  standing_sanity_check=static_check,
                  actuator_motion_peaks={model.actuator(j).name: dict(
                      speed_rad_s=float(peak_speed[j]), acceleration_rad_s2=float(peak_acceleration[j]),
                      damping_torque_nm=float(model.dof_damping[dof] * peak_speed[j]),
                      armature_torque_nm=float(model.dof_armature[dof] * peak_acceleration[j]))
                      for j, dof in enumerate(motor_dofs)},
                  motion_peak_scope='Independent absolute maxima at 200 Hz knots; terms are not summed or measured motor torque',
                  response_parameter_status='ZP10D gains, damping, friction and armature inherited; unmeasured',
                  scope='Necessary-condition diagnostic; not free-base tracking or physical walking acceptance',
                  optimistic_assumptions=(['Floor load allowed across a 0.2 mm gap'] if actual_contacts else [
                      'Full projected sole contacts at floor even when foot rolls']) + [
                      'Circumscribed eight-plane friction cone', 'Joint dry friction allowed either sign',
                      'No actuator tracking error, sticking constraint or contact compliance imposed'],
                  rows=rows)
    assert source_hashes() == hashes
    existing = json.loads(output.read_text()) if output.exists() else {}
    key = ('demonstration_floor_contact_audit' if actual_contacts else 'demonstration_dynamics_audit')
    existing[key] = report
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(existing, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=Path('/tmp/zp10d-reference/candidate.npz'))
    parser.add_argument('--output', type=Path, default=ROOT / 'training/reference_assessment.json')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    contact_mode = parser.add_mutually_exclusive_group()
    contact_mode.add_argument('--actual-contacts', dest='actual_contacts', action='store_true',
                              help='only allow support vertices within 0.2 mm of the floor (default)')
    contact_mode.add_argument('--projected-contacts', dest='actual_contacts', action='store_false',
                              help='diagnostic only: allow the entire projected sole to carry floor loads')
    parser.set_defaults(actual_contacts=True)
    parser.add_argument('--check', action='store_true', help='check rolled and airborne contact cases')
    args = parser.parse_args()
    if args.check:
        check_contact_geometry()
    else:
        assess(args.candidate, args.output, args.data_dir, actual_contacts=args.actual_contacts)
