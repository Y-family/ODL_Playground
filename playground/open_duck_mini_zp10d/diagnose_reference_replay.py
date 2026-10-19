"""Isolate controller-response causes of failed direct Placo replay.

All parameter changes are diagnostic and in memory. Use run_memory_limited.py;
first run replay_reference.py to obtain the identical input position commands.
"""
import argparse
import sys
import json
from pathlib import Path
import resource

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import numpy as np
import mujoco

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from replay_reference import replay, digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate', required=True, type=Path,
                    help='original Placo candidate used by replay_reference.py')
parser.add_argument('--cases', nargs='+', help='append selected cases to the existing diagnosis')
parser.add_argument('--report-only', action='store_true', help='analyze the saved physics trajectories')
args = parser.parse_args()

out = root / 'training/reference_replay'
old = np.load(out / 'uncapped.npz')
ref = json.loads((root / 'data/reference.json').read_text())
scene = root / 'xmls/scene_flat_terrain.xml'
source = {str(p): digest(p) for p in (scene, scene.with_name('open_duck_mini_zp10d.xml'),
                                    root / 'open_duck_mini_zp10d.urdf',
                                    root / 'data/reference.json', root / 'data/polynomial_coefficients.pkl')}
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
mujoco.mj_forward(m, d)
sites = [m.site(s + '_foot').id for s in ('left', 'right')]
feet = [int(m.site_bodyid[s]) for s in sites]
vertices = []
for s, b, v in zip(sites, feet, ref['reward_reference']['foot_vertices_in_site']):
    world = np.array(v) @ d.site_xmat[s].reshape(3, 3).T + d.site_xpos[s]
    vertices.append((world - d.xpos[b]) @ d.xmat[b].reshape(3, 3))
main = np.arange(14)
dofs = m.jnt_dofadr[m.actuator_trnid[main, 0]]
m.actuator_forcelimited[main] = False
m.jnt_actfrclimited[m.actuator_trnid[main, 0]] = False
original = {k: getattr(m, k).copy() for k in ('actuator_gainprm', 'actuator_biasprm', 'dof_armature', 'dof_damping')}
cases, arrays, history = {}, {}, []
if (args.cases or args.report_only) and (out / 'diagnosis.json').exists():
    previous = json.loads((out / 'diagnosis.json').read_text())
    assert previous['protected_files_unchanged'] == source
    cases = previous['cases']
    history = previous.get('run_script_sha256', [previous['script_sha256']])
    with np.load(out / 'diagnosis.npz') as previous_arrays:
        arrays = {key: previous_arrays[key] for key in previous_arrays.files}
history.append(digest(__file__))
known_cases = [
    ('baseline', 1., 1., 1., 1.),
    ('half_speed', 2., 1., 1., 1.),
    ('gain_x4', 1., 4., 1., 1.),
    ('armature_x0p1', 1., 1., .1, 1.),
    ('damping_x0p1', 1., 1., 1., .1),
    ('armature_and_damping_x0p1', 1., 1., .1, .1),
]
if args.cases and set(args.cases) - {case[0] for case in known_cases}:
    parser.error('unknown case')
for name, speed, gain, armature, damping in known_cases:
    if args.report_only or (args.cases and name not in args.cases):
        continue
    for k, v in original.items():
        getattr(m, k)[:] = v
    m.actuator_gainprm[main, 0] *= gain
    m.actuator_biasprm[main, 1:] *= gain
    m.dof_armature[dofs] *= armature
    m.dof_damping[dofs] *= damping
    times = np.arange(round((1.5 + (old['time'][-1] - 1.5) * speed) / m.opt.timestep) + 1) * m.opt.timestep
    original_time = np.where(times <= 1.5, times, 1.5 + (times - 1.5) / speed)
    target = np.column_stack([np.interp(original_time, old['time'], old['target'][:, i]) for i in range(m.nu)])
    contacts = old['desired_contact'][np.clip(np.round(original_time / m.opt.timestep).astype(int), 0, len(old['time']) - 1)]
    if name == 'baseline':
        np.testing.assert_allclose(target, old['target'], atol=1e-12)
    print('START', name, flush=True)
    result, samples = replay(m, target, contacts, feet, vertices, 1., .5)
    result['diagnostic_parameters'] = dict(reference_time_scale=speed, gain_multiplier=gain,
                                            armature_multiplier=armature, damping_multiplier=damping)
    if name == 'baseline':
        np.testing.assert_allclose(samples['qpos'], old['qpos'], atol=1e-10)
        np.testing.assert_allclose(samples['torque'], old['torque'], atol=1e-10)
    cases[name] = result
    for key, value in samples.items():
        arrays[name + '_' + key] = value
    print('RESULT', json.dumps({name: result}), flush=True)
    np.savez_compressed(out / 'diagnosis.npz', **arrays)
    (out / 'diagnosis.json').write_text(json.dumps(dict(
        cases=cases, protected_files_unchanged=source, script_sha256=digest(__file__), run_script_sha256=history,
        replay_script_sha256=digest(root / 'replay_reference.py'),
        method='single-factor uncapped controller-response diagnostic; parameters only in memory'), indent=2) + '\n')
assert source == {p: digest(p) for p in source}

# Recompute saved geometry independently of the integration-time measurement.
for k, v in original.items():
    getattr(m, k)[:] = v
qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]
names = [m.actuator(i).name for i in range(m.nu)]
baseline = {key: arrays['baseline_' + key] for key in ('time', 'qpos', 'target', 'torque',
                                                      'joint_error', 'foot_clearance', 'foot_force')}
raw = np.load(args.candidate)
raw_meta = json.loads(str(raw['metadata']))
assert raw_meta['source_hashes'] == ref['source_hashes']

def geometry(qpos):
    d.qpos[:] = qpos
    mujoco.mj_fwdPosition(m, d)
    return dict(whole_foot_clearance_mm=[float((v @ d.xmat[b].reshape(3, 3).T + d.xpos[b])[:, 2].min() * 1000)
                                           for b, v in zip(feet, vertices)],
                base_position_mm=(qpos[:3] * 1000).tolist(),
                com_position_mm=(d.subtree_com[1] * 1000).tolist())

geometry_error = max(float(abs(np.array(geometry(q)['whole_foot_clearance_mm']) / 1000 - h).max())
                     for q, h in zip(baseline['qpos'][::25], baseline['foot_clearance'][::25]))
assert geometry_error < 1e-10
torque_error = float(abs(baseline['torque'][:, main] - baseline['joint_error'][:, main]
                        * m.actuator_gainprm[main, 0]).max())
assert torque_error < 1e-10
at = int(round(1.9 / m.opt.timestep))
actual = baseline['qpos'][at].copy()
planned = actual.copy()
reference_index = round(.4 * raw_meta['sample_fps'])
planned[:3] = raw['dense_base'][reference_index, :3, 3]
mujoco.mju_mat2Quat(planned[3:7], raw['dense_base'][reference_index, :3, :3].ravel())
planned[qa] = baseline['target'][at]
hybrid = actual.copy()
hybrid[qa] = baseline['target'][at]
analysis = dict(geometry_remeasurement_max_error_m=geometry_error,
                proportional_motor_torque_identity_max_error_nm=torque_error,
                left_swing_peak_at_reference_time_0p4_s=dict(
                    planned=geometry(planned), actual=geometry(actual),
                    desired_joints_at_actual_base=geometry(hybrid),
                    actual_foot_force_n=baseline['foot_force'][at].tolist()),
                inherited_motor_model=dict(kp=13.37, damping_nm_s_rad=.56, armature_kg_m2=.027,
                    passive_drag_at_command_speed_5p24_rad_s_nm=.56 * 5.24 + .068,
                    ideal_constant_speed_ceiling_at_0p980665_nm_rad_s=(.980665 - .068) / .56,
                    legacy_constant_speed_ceiling_at_3p23_nm_rad_s=(3.23 - .068) / .56),
                limits={})
for name in cases:
    q = arrays[name + '_qpos'][:, qa]
    lower, upper = m.jnt_range[m.actuator_trnid[:, 0]].T
    margin = np.minimum(q - lower, upper - q).min(axis=1)
    height, force = arrays[name + '_foot_clearance'], arrays[name + '_foot_force']
    loaded_other = force[:, ::-1] > .5 * m.body_subtreemass[1] * abs(m.opt.gravity[2])
    valid_lift = ((height >= .002) & (force < .5) & loaded_other).any(axis=1)
    analysis['limits'][name] = dict(max_joint_limit_overrun_deg=float(np.rad2deg(max(0., -margin.min()))),
        min_joint_margin_during_supported_lift_deg=(float(np.rad2deg(margin[valid_lift].min())) if valid_lift.any() else None))
for event in cases['baseline']['knee_peak_events'].values():
    assert abs(event['balance_residual_nm']) < 1e-7
if 'gain_x4' in cases:
    assert cases['gain_x4']['stop'] == 'duration'
    assert analysis['limits']['gain_x4']['max_joint_limit_overrun_deg'] == 0.
    assert all(f['longest_supported_clearance_s']['10'] > 0 for f in cases['gain_x4']['feet'].values())
report = json.loads((out / 'diagnosis.json').read_text())
report['analysis'] = analysis
report['analysis_script_sha256'] = digest(__file__)
(out / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
phase = baseline['time'] - 1.5
selected = (phase >= 0) & (phase <= 1.2)
knee = m.actuator('left_knee').id
figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout='constrained')
axes[0].plot(phase[selected], np.rad2deg(baseline['target'][selected, knee]), label='Placo knee target')
axes[0].plot(phase[selected], np.rad2deg(baseline['qpos'][selected, qa[knee]]), label='Actual knee')
axes[0].set_ylabel('Knee angle (deg)')
axes[1].plot(phase[selected], baseline['torque'][selected, knee], label='Motor torque')
axes[1].axhline(-.980665, color='gray', linestyle=':', label='Original torque cap')
axes[1].set_ylabel('Torque (N m)')
axes[2].plot(np.arange(241) / 200, raw['dense_heights'][:241, 0] * 1000, label='Planned whole-foot height')
axes[2].plot(phase[selected], baseline['foot_clearance'][selected, 0] * 1000, label='Actual whole-foot height')
axes[2].set_ylabel('Foot clearance (mm)')
axes[2].set_xlabel('Reference time (s)')
for ax in axes:
    ax.axvline(.856, color='red', alpha=.5, linestyle='--')
    ax.grid(alpha=.2)
    ax.legend(loc='upper right')
figure.suptitle('Uncapped direct replay: tracking lag, not a 4.45 N m liftoff requirement')
figure.savefig(out / 'diagnosis.png', dpi=150)
plt.close(figure)
print('ANALYSIS', json.dumps(analysis), flush=True)
print('PASS: protected files unchanged', flush=True)
