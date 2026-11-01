"""Full free-base validation of the 7.8 V / 71 g BAM response candidate.

Run through scripts/cad/run_memory_limited.py; formal files are hash-protected.
The same S09 fit is assumed for all 14 ZP10Ds, with SG90 parameters unchanged.
"""

import json
from pathlib import Path

from replay_reference import digest, longest_run, replay
from reference import DATA, MODEL, ROOT, SCENE, URDF, source_hashes
import mujoco
import numpy as np


def main():
    output = ROOT / 'training/calibrated_actuator'
    output.mkdir(parents=True, exist_ok=True)
    fit_path = ROOT / 'training/actuator_capture/s09_s10_71g/bam_wide/m1_preliminary.json'
    fit = json.loads(fit_path.read_text())
    p = fit['parameters']
    old_report_path = ROOT / 'training/reference_replay/assessment.json'
    old_report = json.loads(old_report_path.read_text())
    assert old_report['source_hashes'] == source_hashes()
    commands_path = old_report_path.with_name('uncapped.npz')
    with np.load(commands_path) as raw:
        targets = raw['target'].copy()
        contacts = raw['desired_contact'].copy()
    import hashlib
    assert hashlib.sha256(targets.tobytes()).hexdigest() == old_report['position_command_sha256']
    protected = [URDF, MODEL, SCENE, DATA / 'reference.json', DATA / 'polynomial_coefficients.pkl',
                 ROOT.parents[1] / 'cad/local_fix/open_duck_mini_upper_refit.step']
    before = {str(path): digest(path) for path in protected}
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert model.nu == 16 and model.ngeom == 3296 and model.neq == 0
    names = [model.actuator(i).name for i in range(model.nu)]
    main_servos = np.array([i for i, name in enumerate(names) if 'antenna' not in name])
    assert len(main_servos) == 14
    joints = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[joints]
    addresses = model.jnt_qposadr[joints]
    home_q = model.key('home').qpos[addresses].copy()
    home_ctrl = model.key('home').ctrl.copy()
    old_kp = model.actuator_gainprm[:, 0].copy()
    old_bias = model.actuator_biasprm.copy()
    old_damping = model.dof_damping.copy()
    old_friction = model.dof_frictionloss.copy()
    old_armature = model.dof_armature.copy()
    caps = np.max(abs(model.actuator_forcerange), axis=1)
    np.testing.assert_allclose(caps[main_servos], fit['torque_cap_prior_Nm'], atol=1e-12)
    assert model.actuator_forcelimited.all()
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
    dt = model.opt.timestep
    # Preserve the old home motor preload at exactly the same geometric pose.
    # This is an explicit control comparison, not a new stance or a walking controller.
    offset = np.zeros(model.nu)
    offset[main_servos] = ((old_kp[main_servos] / p['position_gain_Nm_per_rad'] - 1)
                          * (home_ctrl - home_q)[main_servos])
    compensated_home = home_ctrl + offset
    np.testing.assert_allclose(
        p['position_gain_Nm_per_rad'] * (compensated_home - home_q)[main_servos],
        old_kp[main_servos] * (home_ctrl - home_q)[main_servos], atol=1e-12)
    cases = {}
    modes = ['baseline_home', 'baseline_reference', 'identified_home', 'identified_reference',
             'identified_preload_home', 'identified_preload_reference', 'identified_preload_delay_reference']
    for mode in modes:
        model.actuator_gainprm[:, 0] = old_kp
        model.actuator_biasprm[:] = old_bias
        model.dof_damping[:] = old_damping
        model.dof_frictionloss[:] = old_friction
        model.dof_armature[:] = old_armature
        identified = mode.startswith('identified')
        if identified:
            model.actuator_gainprm[main_servos, 0] = p['position_gain_Nm_per_rad']
            model.actuator_biasprm[main_servos, 1] = -p['position_gain_Nm_per_rad']
            model.dof_damping[dofs[main_servos]] = p['damping_Nm_s_per_rad']
            model.dof_frictionloss[dofs[main_servos]] = p['frictionloss_Nm']
            model.dof_armature[dofs[main_servos]] = p['armature_kg_m2']
        home = compensated_home if 'preload' in mode else home_ctrl
        model.key('home').ctrl[:] = home
        home_only = mode.endswith('_home')
        commands = np.tile(home, (round(8 / dt), 1)) if home_only else targets.copy()
        desired = np.ones((len(commands), 2)) if home_only else contacts
        if not home_only and 'preload' in mode:
            commands += offset
        delay = p['transport_delay_s'] if 'delay' in mode else 0.
        print('START', mode, flush=True)
        report, arrays = replay(model, commands, desired, feet, vertices,
                                .5 if home_only else 1., 0. if home_only else .5, delay_s=delay)
        torques = arrays['torque']
        actual = arrays['qpos'][:, addresses]
        tail = arrays['time'] >= max(0., arrays['time'][-1] - 1.)
        report.update(
            rms_torque_nm=dict(zip(names, np.sqrt(np.mean(torques**2, axis=0)).tolist())),
            saturation_seconds=dict(zip(names, (np.sum(abs(torques) >= caps * .999, axis=0) * dt).tolist())),
            longest_saturation_seconds=dict(zip(names, [longest_run(abs(torques[:, i]) >= caps[i] * .999, dt)
                                                        for i in range(model.nu)])),
            final_joint_deg=dict(zip(names, np.rad2deg(actual[-1]).tolist())),
            last_second_mean_joint_deg=dict(zip(names, np.rad2deg(actual[tail].mean(axis=0)).tolist())),
            last_second_max_home_error_deg=float(np.rad2deg(abs(actual[tail] - home_q).max())),
            last_second_base_z_mm=(arrays['qpos'][tail, 2] * 1000).mean().item(),
            base_z_min_mm=float(arrays['qpos'][:, 2].min() * 1000),
            max_target_joint_limit_overrun_deg=float(np.rad2deg(max(0., np.max(
                np.maximum(model.jnt_range[joints, 0] - commands, commands - model.jnt_range[joints, 1]))))),
            delay_s_applied=round(delay / dt) * dt,
            role='8 s static hold' if home_only else 'identical Placo commands, free-base dynamics')
        clipped = np.where(model.actuator_ctrllimited,
                           np.clip(commands, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]),
                           commands)
        report['command_clipped_seconds'] = dict(zip(names, (np.sum(abs(commands - clipped) > 1e-10, axis=0) * dt).tolist()))
        report['command_clipping_scope'] = 'entire requested input, including input after early stop'
        executed = arrays['applied_target']
        executed = np.where(model.actuator_ctrllimited,
                            np.clip(executed, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]),
                            executed)
        report['executed_command_clipped_seconds'] = dict(zip(names,
            (np.sum(abs(executed - arrays['applied_target']) > 1e-10, axis=0) * dt).tolist()))
        np.testing.assert_allclose(torques[:, main_servos], np.clip(
            (executed - actual)[:, main_servos] * model.actuator_gainprm[main_servos, 0],
            -caps[main_servos], caps[main_servos]), atol=1e-10)
        report['snapshots'] = snapshots(model, arrays, feet, vertices, addresses, names,
                                       [0., .1, .3, .5] if mode == 'identified_home' else
                                       [7.9] if home_only else [1.5, 1.7, 1.8, 1.9, 2., 2.2])
        report['landing_and_home_recovery'] = ('not tested: no preceding valid liftoff'
                                               if not home_only else 'not a landing trial')
        assert np.max(abs(torques) - caps) < 1e-10
        assert model.actuator_forcelimited.all()
        np.testing.assert_array_equal(model.actuator_gainprm[14:, 0], old_kp[14:])
        cases[mode] = report
        np.savez_compressed(output / (mode + '.npz'), **arrays)
        print(json.dumps(dict(mode=mode, stop=report['stop'], seconds=report['simulated_seconds'],
                              tilt=report['max_tilt_deg'], feet=report['feet'])), flush=True)
        if mode == 'baseline_reference':
            previous = old_report['cases']['capped']
            assert report['stop'] == previous['stop']
            np.testing.assert_allclose(report['simulated_seconds'], previous['simulated_seconds'], atol=dt)
            np.testing.assert_allclose(report['max_tilt_deg'], previous['max_tilt_deg'], atol=1e-7)
    # Regression: stopping before the reference must produce a report, not an assertion.
    short, short_arrays = replay(model, commands[:2], desired[:2], feet, vertices, 1., .5, delay_s=.002)
    assert not short['reference_started'] and short['max_joint_error_deg'] is None
    assert short['feet']['left']['peak_clearance_mm'] is None
    # Initial airborne steps isolate cached mass constants from contact-solver tolerances.
    assert short['max_dynamics_residual_abs'] < 1e-9, 'Inconsistent model inertia constants'
    np.testing.assert_array_equal(short_arrays['applied_target'], commands[[0, 0]])
    assert before == {str(path): digest(path) for path in protected}
    assessment = dict(
        method='full 3296-geom model, 2 ms real integration, floating base, physical ground contacts',
        source_hashes=source_hashes(), protected_files_unchanged=before,
        fit_sha256=digest(fit_path), fit_path=str(fit_path.relative_to(ROOT)), parameters=p,
        input_commands_sha256=digest(commands_path), script_sha256=digest(__file__),
        replay_script_sha256=digest(Path(__file__).with_name('replay_reference.py')),
        mujoco_version=mujoco.__version__, torque_cap_prior_nm=fit['torque_cap_prior_Nm'],
        torque_cap_measured=False, mass_kg=float(model.body_subtreemass[1]),
        old_home_motor_preload_nm=dict(zip(names, (old_kp * (home_ctrl - home_q)).tolist())),
        home_command_adjustment_deg=dict(zip(names, np.rad2deg(offset).tolist())),
        assumptions=['S09 7.8 V / 71 g equivalent response shared by 14 ZP10Ds; SG90 unchanged',
                     'armature is additional motor inertia only; rig load not added',
                     'rig mounting offset is not a robot joint zero correction',
                     '36 ms host/serial-inclusive delay only in named sensitivity case',
                     'home preload comparison preserves former motor preload, not full inverse dynamics',
                     'formal timestep, integrator, eulerdamp flag, force limits and geometry unchanged',
                     'open-loop reference test, no learned policy or balance feedback; no FEA or hardware validation'],
        self_checks='old capped baseline reproduced; pre-reference stop report and delay indexing pass',
        cases=cases)
    (output / 'assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
    plot(output, cases, names, addresses)
    print('PASS: source protection, torque caps, SG90 retention, regression checks', flush=True)


def snapshots(model, arrays, feet, vertices, addresses, names, requested_times):
    from scipy.spatial import ConvexHull
    data = mujoco.MjData(model)
    rows = []
    for requested in requested_times:
        if requested > arrays['time'][-1]:
            continue
        i = round(requested / model.opt.timestep)
        data.qpos[:] = arrays['qpos'][i]
        mujoco.mj_fwdPosition(model, data)
        com = data.subtree_com[1, :2]
        margins = []
        for body, v in zip(feet, vertices):
            world = v @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            hull = ConvexHull(world[:, :2])
            margins.append(float(-np.max(hull.equations[:, :2] @ com + hull.equations[:, 2]) * 1000))
        rows.append(dict(time_s=float(arrays['time'][i]), reference_time_s=float(arrays['time'][i] - 1.5),
                         com_position_mm=(1000 * data.subtree_com[1]).tolist(),
                         com_projected_sole_halfplane_margin_mm=margins,
                         margin_scope='positive inside projected sole; optimistic, ignores rolled/raised corners',
                         foot_force_n=arrays['foot_force'][i].tolist(),
                         desired_contact=arrays['desired_contact'][i].tolist(),
                         whole_sole_clearance_mm=(1000 * arrays['foot_clearance'][i]).tolist(),
                         tilt_deg=float(arrays['tilt_deg'][i]),
                         actual_joint_deg=dict(zip(names, np.rad2deg(data.qpos[addresses]).tolist()))))
    return rows


def plot(output, cases, names, addresses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex='col')
    for col, suffix in enumerate(('_home', '_reference')):
        for mode in cases:
            if not mode.endswith(suffix):
                continue
            with np.load(output / (mode + '.npz')) as a:
                t = a['time']
                axes[0, col].plot(t, a['tilt_deg'], label=mode)
                axes[1, col].plot(t, 1000 * a['qpos'][:, 2])
                axes[2, col].plot(t, 1000 * a['foot_clearance'][:, 0])
                axes[3, col].plot(t, a['torque'][:, names.index('left_knee')])
        axes[0, col].set_title('Static home' if col == 0 else 'Placo reference (starts at 1.5 s)')
        axes[0, col].legend(fontsize=7)
        axes[3, col].set_xlabel('Time (s)')
    for row, label in enumerate(['Base tilt (deg)', 'Base height (mm)', 'Left whole-sole clearance (mm)', 'Left knee motor torque (Nm)']):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.grid(alpha=.25)
    fig.suptitle('7.8 V response candidate, provisional torque cap 0.980665 Nm; diagnostic only')
    fig.tight_layout()
    fig.savefig(output / 'assessment.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
