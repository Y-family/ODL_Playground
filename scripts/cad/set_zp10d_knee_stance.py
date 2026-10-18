"""Validate and publish a 60-degree loaded knee stance; run with the hard limiter."""
from datetime import date
import itertools
import json
from pathlib import Path
import resource
import shutil
import sys
import xml.etree.ElementTree as ET

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import mujoco
import numpy as np
from scipy.optimize import least_squares
import assess_zp10d_hip_roll as standing
import assess_zp10d_loads as loads

ROOT, OUT = standing.ROOT, standing.OUT
STEP, MJCF, SCENE, MASS = standing.STEP, standing.MJCF, standing.SCENE, standing.MASS
WORK = Path('/tmp/open-duck-knee-stance')
CANDIDATE = WORK / SCENE.name
REPORT = OUT / 'CURRENT_STANCE_ASSESSMENT.json'
digest = standing.digest
TARGET = np.deg2rad(60.)
ARRAYS = ('body_mass', 'body_ipos', 'body_inertia', 'body_pos', 'body_quat',
    'jnt_pos', 'jnt_axis', 'jnt_range', 'jnt_stiffness', 'dof_damping', 'dof_frictionloss',
    'geom_type', 'geom_dataid', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity',
    'geom_friction', 'geom_solref', 'geom_solimp', 'mesh_vert', 'mesh_face',
    'actuator_ctrlrange', 'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm')


def state(model, qpos, ctrl, qvel=None):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    if qvel is not None:
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    return data


def summarize(model, case, zero):
    summary = standing.summarize(model, case, zero)
    tail = [f for f in case['frames'] if f['time_s'] >= case['requested_seconds'] - 3]
    adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    summary['mean_joint_angles_deg'] = np.degrees(np.mean([np.array(f['qpos'])[adr] for f in tail], axis=0)).tolist()
    summary['mean_abs_motor_torque_nm'] = np.mean(np.abs([f['torque_nm'] for f in tail]), axis=0).tolist()
    summary['mean_knee_ground_lever_mm'] = np.mean([f['knee_effective_ground_lever_mm'] for f in tail], axis=0).tolist()
    return summary


def target_pose(model, reference, zero):
    """Keep sagittal foot positions relative to the body, with flat soles and 60-degree knees."""
    qpos = model.key_qpos[0].copy()
    for side in ('left', 'right'):
        knee = model.joint(side + '_knee').qposadr[0]
        qpos[knee] = TARGET
        qpos[model.joint(side + '_hip_roll').qposadr[0]] = 0.
        body = model.body('foot_assembly' if side == 'left' else 'foot_assembly_2').id
        target_x = reference.xpos[body, 0] - reference.qpos[0] + qpos[0]
        names = [side + '_hip_pitch', side + '_ankle']
        joints = np.array([model.joint(n).id for n in names])
        adr = model.jnt_qposadr[joints]
        scratch = state(model, qpos, model.key_ctrl[0])

        def residual(angles):
            scratch.qpos[:] = qpos
            scratch.qpos[adr] = angles
            mujoco.mj_kinematics(model, scratch)
            rotation = scratch.xmat[body].reshape(3, 3) @ zero.xmat[body].reshape(3, 3).T
            return [scratch.xpos[body, 0] - target_x, .06 * rotation[0, 2]]

        lo, hi = model.jnt_range[joints].T
        fit = least_squares(residual, qpos[adr], bounds=(lo + .001, hi - .001),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        assert fit.success and np.linalg.norm(residual(fit.x)) < 1e-7
        qpos[adr] = fit.x
    data = state(model, qpos, model.key_ctrl[0])
    alignment = standing.align_floor(model, data)
    return data.qpos.copy(), alignment


def assess():
    WORK.mkdir(exist_ok=True)
    hashes = {k: digest(p) for k, p in [('source_step_sha256', STEP), ('mjcf_sha256', MJCF), ('scene_sha256', SCENE)]}
    mass = json.loads(MASS.read_text())
    assert all(mass[k] == v for k, v in hashes.items())
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert model.neq == 0 and model.nu == 16
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    names = [model.actuator(a).name for a in range(model.nu)]
    joints = model.actuator_trnid[:, 0]
    adr = model.jnt_qposadr[joints]
    knees = [names.index(side + '_knee') for side in ('left', 'right')]
    adjustable = [names.index(side + '_' + suffix) for side in ('left', 'right')
                  for suffix in ('hip_roll', 'hip_pitch', 'knee', 'ankle')]
    old_ctrl, old_qpos = model.key_ctrl[0].copy(), model.key_qpos[0].copy()
    source_hash = digest(__file__)
    report = dict(date=date.today().isoformat(), **hashes, script_sha256=source_hash,
        helper_sha256={p.name: digest(p) for p in (Path(standing.__file__), Path(loads.__file__))},
        actuator_order=names, servo_ids=[loads.SID[n] for n in names], postures=[],
        scope='R14: adopt 60-degree loaded knees, flat soles and front-view vertical legs. Free base, unchanged CAD/physical parameters. Standing and finite disturbances only.',
        target_knee_angle_deg=60., protocol=dict(hold_seconds=13, steady_window_seconds=3,
            push_duration_s=.2, disturbance_seconds=7.2, transition_seconds=9),
        resources=dict(address_space_cap_bytes=4_000_000_000, workspace_cap_bytes=8_000_000_000, native_threads=1, serial=True))

    def save():
        (WORK / 'assessment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    def run(label, qpos, ctrl, duration=13., push=None, waypoints=None):
        initial = state(model, qpos, ctrl)
        case = loads.simulate(model, initial, label, duration,
            waypoints or [(0., ctrl), (duration, ctrl)], names, joints, push=push)
        case['summary'] = summarize(model, case, zero)
        print(label, 'knees', [round(case['summary']['mean_joint_angles_deg'][k], 4) for k in knees],
              'torques', [round(case['summary']['mean_abs_motor_torque_nm'][k], 4) for k in knees],
              'max tilt', round(case['max_base_tilt_deg'], 3), flush=True)
        return case

    baseline = run('before_81deg', old_qpos, old_ctrl)
    old = dict(name='before_81deg', initial_qpos=old_qpos.tolist(), ctrl_rad=old_ctrl.tolist(),
               hold=baseline, summary=baseline['summary'], disturbances=[])
    report['postures'].append(old)
    settled = state(model, baseline['frames'][-1]['qpos'], old_ctrl)
    goal, alignment = target_pose(model, settled, zero)
    ctrl = old_ctrl.copy()
    ctrl[adjustable] = goal[adr[adjustable]] + old_ctrl[adjustable] - settled.qpos[adr[adjustable]]
    report['load_compensation_iterations'] = []
    for iteration in range(5):
        data = state(model, goal, ctrl)
        samples = []
        for step in range(300):
            mujoco.mj_step(model, data, nstep=round(.02 / model.opt.timestep))
            if step >= 250:
                samples.append(data.qpos[adr].copy())
        error = np.mean(samples, axis=0) - goal[adr]
        maximum = float(np.max(np.abs(np.degrees(error[adjustable]))))
        report['load_compensation_iterations'].append(dict(iteration=iteration,
            ctrl_rad=ctrl.tolist(), mean_error_deg=np.degrees(error).tolist(), maximum_adjusted_joint_error_deg=maximum))
        print('compensation', iteration, 'maximum error deg', round(maximum, 5), flush=True)
        if maximum < .06:
            break
        ctrl[adjustable] -= error[adjustable]
    assert np.all((ctrl >= model.actuator_ctrlrange[:, 0]) & (ctrl <= model.actuator_ctrlrange[:, 1]))
    selected_case = run('knee_60deg', goal, ctrl)
    selected = dict(name='knee_60deg', initial_qpos=goal.tolist(), ctrl_rad=ctrl.tolist(),
        home_ground_alignment=alignment, hold=selected_case, summary=selected_case['summary'], disturbances=[])
    report['postures'].append(selected)
    save()
    for index, row in enumerate(report['postures']):
        directions = [(0, -1, 4.)]
        if index == 1:
            directions += [(axis, direction, 2.) for axis, direction in itertools.product((0, 1), (-1, 1))]
        for axis, direction, force in directions:
            vector = np.zeros(3)
            vector[axis] = direction * force
            push = dict(start_s=1., duration_s=.2, force_world_n=vector.tolist())
            case = run(f'{row["name"]}_push_{"XY"[axis]}{direction:+}_{force:g}N',
                row['summary']['settled_qpos'], np.array(row['ctrl_rad']), 7.2, push)
            row['disturbances'].append(case)
            save()
    report['transition'] = run('transition_to_60deg', old['summary']['settled_qpos'], old_ctrl, 9.,
        waypoints=[(0., old_ctrl), (3., ctrl), (9., ctrl)])
    summary = selected['summary']
    tests = [selected_case, report['transition']] + selected['disturbances']
    report['acceptance'] = dict(
        complete=all(t['completed'] for t in tests), warning_free=all(not any(t['warnings']) for t in tests),
        no_saturation=all(not any(t['saturation_fraction']) for t in tests),
        no_self_contacts=all(not any(p['peak_contact_force_n'] > .05 for p in t['nonfloor_contact_pairs'].values()) for t in tests),
        joint_limits=all(max(t['joint_limit_excursion_deg']) < .001 for t in tests),
        upright=all(t['max_base_tilt_deg'] < 5. for t in tests),
        recovered=all(t['summary']['final_second_max_tilt_deg'] < 2. for t in tests),
        both_feet_loaded=all(min(t['summary']['steady_min_fz_n']) > 3. for t in tests),
        knees_at_60=max(abs(summary['mean_joint_angles_deg'][k] - 60.) for k in knees) < .2,
        front_legs_vertical=max(abs(o['front_leg_tilt_deg']) for o in summary['orientation']) < .2,
        soles_level=max(max(abs(o['sole_roll_deg']), abs(o['sole_pitch_deg'])) for o in summary['orientation']) < .2,
        positive_support_margin=summary['contact_support_margin_mm'] > 10.)
    report['standing_passed'] = all(report['acceptance'].values())
    report['standing_only'] = True
    report['limitations'] = ['Estimated mass/contact/actuator response; PETG weights and hardware calibration pending.',
        'Finite standing/disturbance tests do not establish walking, successful single support or continuous rated torque.',
        'Current R13 motion/load/liftoff reports use the previous home and become historical after publication.']
    shutil.copyfile(MJCF, WORK / MJCF.name)
    if not (WORK / 'assets').exists():
        (WORK / 'assets').symlink_to(MJCF.parent / 'assets', target_is_directory=True)
    tree = ET.parse(SCENE)
    key = tree.getroot().find('keyframe/key[@name="home"]')
    key.set('qpos', ' '.join(f'{v:.14g}' for v in goal))
    key.set('ctrl', ' '.join(f'{v:.14g}' for v in ctrl))
    ET.indent(tree, space='  ')
    tree.write(CANDIDATE, encoding='unicode', xml_declaration=True)
    report['candidate_scene_sha256'] = digest(CANDIDATE)
    assert digest(__file__) == source_hash
    assert all(digest(p) == hashes[k] for k, p in [('source_step_sha256', STEP), ('mjcf_sha256', MJCF), ('scene_sha256', SCENE)])
    report['resources']['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    save()
    print('Standing acceptance:', report['acceptance'], flush=True)


def publish():
    report = json.loads((WORK / 'assessment.json').read_text())
    assert report['standing_passed'] and report['local_pose_check']['passed']
    assert digest(__file__) == report['script_sha256']
    for key, path in [('source_step_sha256', STEP), ('mjcf_sha256', MJCF), ('scene_sha256', SCENE)]:
        assert digest(path) == report[key], key
    assert digest(CANDIDATE) == report['candidate_scene_sha256']
    old = mujoco.MjModel.from_xml_path(str(SCENE))
    new = mujoco.MjModel.from_xml_path(str(CANDIDATE))
    assert all(np.array_equal(getattr(old, a), getattr(new, a)) for a in ARRAYS)
    adjustable = [old.actuator(side + '_' + suffix).id for side in ('left', 'right')
                  for suffix in ('hip_roll', 'hip_pitch', 'knee', 'ankle')]
    adr = old.jnt_qposadr[old.actuator_trnid[adjustable, 0]]
    assert set(np.flatnonzero(abs(new.key_qpos[0] - old.key_qpos[0]) > 1e-12)) <= {2, *adr}
    assert set(np.flatnonzero(abs(new.key_ctrl[0] - old.key_ctrl[0]) > 1e-12)) <= set(adjustable)
    for side in ('left', 'right'):
        assert abs(new.key_qpos[0, new.joint(side + '_knee').qposadr[0]] - TARGET) < 1e-12
    for path in (SCENE, MASS, REPORT, REPORT.with_suffix('.md'), REPORT.with_suffix('.png')):
        if path.exists():
            backup = WORK / ('previous_' + path.name)
            if not backup.exists():
                shutil.copyfile(path, backup)
    temporary = SCENE.with_suffix('.xml.tmp')
    shutil.copyfile(CANDIDATE, temporary)
    temporary.replace(SCENE)
    compiled = mujoco.MjModel.from_xml_path(str(SCENE))
    assert all(np.array_equal(getattr(compiled, a), getattr(new, a)) for a in ARRAYS + ('key_qpos', 'key_ctrl'))
    report.update(status='published', published_scene_sha256=digest(SCENE),
        publisher_sha256=digest(__file__), arrays_preserved_except_keyframes=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    selected = report['postures'][-1]
    hold = selected['hold']
    mass = json.loads(MASS.read_text())
    mass['prior_home_validation'] = dict(validation=mass['validation'], previous=mass.get('prior_home_validation'))
    mass['scene_sha256'] = report['published_scene_sha256']
    mass['home_free_qpos'] = compiled.key_qpos[0, :7].tolist()
    mass['home_joint_angles_rad'] = {compiled.joint(j).name: float(compiled.key_qpos[0, compiled.jnt_qposadr[j]])
        for j in range(compiled.njnt) if compiled.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}
    mass['home_ctrl_rad'] = {compiled.actuator(a).name: float(compiled.key_ctrl[0, a]) for a in range(compiled.nu)}
    mass['home_ground_alignment'] = selected['home_ground_alignment']
    rollout = dict(mode='free_home_hold', seconds=hold['simulated_seconds'], finite=True,
        warning_counts=hold['warnings'], max_tracking_error_deg=hold['max_tracking_error_deg'],
        saturation_fraction=hold['saturation_fraction'], max_base_tilt_deg=hold['max_base_tilt_deg'],
        scope='R14: 60-degree loaded knees; 13-second free standing, plus separate transition and finite disturbance tests.')
    mass['validation'] = dict(synchronization_passed=True, source_step_sha256=report['source_step_sha256'],
        mjcf_sha256=report['mjcf_sha256'], scene_sha256=report['published_scene_sha256'],
        geometry_evidence='R11 geometry/inertials/proxies preserved; compiled structural arrays compared. Earlier home evidence retained as historical.',
        compiled_arrays_checked=list(ARRAYS), rollouts=[rollout], home_assessment=str(REPORT.relative_to(ROOT)),
        home_assessment_sha256=digest(REPORT), standing_passed=True, local_pose_check_passed=True)
    mass['home_update'] = dict(id='R14', previous_scene_sha256=report['scene_sha256'],
        assessment=str(REPORT.relative_to(ROOT)), target_knee_angle_deg=60.,
        actual_knee_angles_deg=[selected['summary']['mean_joint_angles_deg'][compiled.actuator(s + '_knee').id] for s in ('left', 'right')],
        command_offsets_reason='Load compensation in the estimated position-actuator model; geometric knee home is 60 degrees.', no_hardware_calibration=True)
    mass['published_verification'].update(home_keyframe_updated=True, structural_arrays_identical=True, keyframes_match_validated_candidate=True)
    mass.pop('preview', None)
    MASS.write_text(json.dumps(mass, ensure_ascii=False, indent=2) + '\n')
    path = Path('/tmp/open-duck-dynamics-sync/manifest.json')
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['source_step_sha256'] == report['source_step_sha256']:
            for key in ('home_free_qpos', 'home_joint_angles_rad', 'home_ctrl_rad', 'home_ground_alignment', 'home_update'):
                manifest[key] = mass[key]
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for name in ('FINAL_ASSEMBLY_REVIEW.json', 'LEG_LINK_SHORTENING_REPORT.json'):
        path = OUT / name
        result = json.loads(path.read_text())
        result['stance_update'] = mass['home_update'] | dict(scene_sha256=mass['scene_sha256'], standing_passed=True)
        if name.startswith('FINAL'):
            result['prior_home_simulation'] = result['simulation']
            result['simulation'] = [rollout]
        else:
            result['control_status'] = 'R14 60-degree standing validated; older motion/liftoff evidence uses previous home. Walking pending.'
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Published R14 60-degree home; CAD and physical model arrays preserved.', flush=True)


def render():
    from PIL import Image, ImageDraw, ImageFont
    report = json.loads(REPORT.read_text())
    assert report['published_scene_sha256'] == digest(SCENE)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.vis.global_.offwidth = model.vis.global_.offheight = 900
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.02, 0, .235]
    camera.distance, camera.azimuth, camera.elevation = .83, 90, -2
    packet = Image.new('RGB', (1800, 900), 'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    with mujoco.Renderer(model, height=900, width=900) as renderer:
        for i, row in enumerate(report['postures']):
            data = state(model, row['summary']['settled_qpos'], row['ctrl_rad'])
            renderer.update_scene(data, camera=camera, scene_option=options)
            picture = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 900, 55), fill='white')
            draw.text((20, 15), ('Before: ~81 deg knees' if i == 0 else 'After: 60 deg knees') + ' (under load)', fill='black', font=font)
            packet.paste(picture, (i * 900, 0))
    output = REPORT.with_suffix('.png')
    packet.save(output)
    report['preview'] = dict(path=str(output.relative_to(ROOT)), sha256=digest(output), source='Actual settled simulation states, same side camera.')
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    mass = json.loads(MASS.read_text())
    mass['validation']['home_assessment_sha256'] = digest(REPORT)
    MASS.write_text(json.dumps(mass, ensure_ascii=False, indent=2) + '\n')
    names = report['actuator_order']
    ids = [names.index(s + '_knee') for s in ('left', 'right')]
    before, after = [p['summary'] for p in report['postures']]
    lines = ['# R14：双膝弯曲60°正式站姿', '', f'日期：{report["date"]}。自由基座，估计质量/接触/执行器参数；未下发硬件。', '',
        '| 指标 | 原站姿 | 60°站姿 |', '| --- | ---: | ---: |']
    for label, field in [('受载膝角 °', 'mean_joint_angles_deg'), ('膝保持扭矩 N·m', 'mean_abs_motor_torque_nm')]:
        pair = lambda s: ' / '.join(f'{s[field][k]:.3f}' for k in ids)
        lines.append(f'| 左/右{label} | {pair(before)} | {pair(after)} |')
    for label, field in [('支撑边界余量 mm', 'contact_support_margin_mm'), ('站立最大倾角 °', 'steady_max_base_tilt_deg')]:
        lines.append(f'| {label} | {before[field]:.3f} | {after[field]:.3f} |')
    selected = report['postures'][-1]
    commands = [np.degrees(selected['ctrl_rad'][k]) for k in ids]
    lines += ['', f'几何home双膝均60°；负载补偿后舵机目标指令为左/右 {commands[0]:.3f}° / {commands[1]:.3f}°。实机绝对角和补偿须标定。', '',
        '验证包含各13秒站立、原站姿向60°的9秒切换、新站姿四方向2 N×0.2秒扰动，以及新旧站姿同样4 N×0.2秒后向扰动。', '',
        '| 试验 | 时长 s | 最大倾角 ° | 警告 | 饱和 |', '| --- | ---: | ---: | --- | --- |']
    cases = [p['hold'] for p in report['postures']] + [report['transition']] + [c for p in report['postures'] for c in p['disturbances']]
    for c in cases:
        lines.append(f'| {c["name"]} | {c["simulated_seconds"]:.2f} | {c["max_base_tilt_deg"]:.3f} | {sum(c["warnings"])} | {"有" if any(c["saturation_fraction"]) else "无"} |')
    local = report['local_pose_check']
    lines += ['', f'腿与直接相邻接口局部检查：{local["sample_count"]}个姿态，{local["part_count"]}件，近邻BRep精查{len(local["near_pairs"])}次，通过。只覆盖这些抽样姿态，不是连续全机碰撞认证。', '',
        '**已更新正式站姿；未验证60°站姿下的抬脚、步行、连续额定扭矩或结构强度。** R13负载/抬脚报告及R12站姿报告属于此前约81°站姿，不作为本次运动验收。', '',
        f'当前scene SHA256：`{report["published_scene_sha256"]}`；源STEP及MJCF保持。详细轨迹、16轴扭矩、关节角、扰动和源哈希见同名JSON。', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines))
    print('Rendered and documented:', REPORT.with_suffix('.png'), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['assess']:
        assess()
    elif sys.argv[1:] == ['publish']:
        publish()
    elif sys.argv[1:] == ['render']:
        render()
    else:
        raise SystemExit('usage: set_zp10d_knee_stance.py assess | publish | render')
