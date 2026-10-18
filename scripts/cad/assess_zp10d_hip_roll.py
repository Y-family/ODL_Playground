"""Compare hip-roll standing poses and publish only a validated home keyframe.

Use run_memory_limited.py. Geometry, gains and hardware remain unchanged.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import xml.etree.ElementTree as ET

if resource.getrlimit(resource.RLIMIT_AS)[1] != 4_000_000_000:
    raise SystemExit('Run through scripts/cad/run_memory_limited.py')

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
import assess_zp10d_loads as loads

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'cad/local_fix'
SCENE = ROOT / 'playground/open_duck_mini_zp10d/xmls/scene_flat_terrain.xml'
MJCF = SCENE.with_name('open_duck_mini_zp10d.xml')
STEP = OUT / 'open_duck_mini_upper_refit.step'
MASS = OUT / 'open_duck_mini_zp10d_mass_properties.json'
REPORT = OUT / 'CURRENT_HIP_ROLL_ASSESSMENT.json'
WORK = Path('/tmp/open-duck-hip-roll')
CANDIDATE = WORK / SCENE.name
ROLL_NAMES = ('left_hip_roll', 'right_hip_roll')
# Small offsets counter the estimated position-actuator deflection under load.
# These commands are not the actual geometric hip-roll angles.
ROLL_CTRL = (.03345, -.03010)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(model, data, zero):
    feet = [model.body(n).id for n in ('foot_assembly', 'foot_assembly_2')]
    floor = model.geom('floor').id
    points = []
    for i, contact in enumerate(data.contact):
        if floor not in contact.geom:
            continue
        other = int(contact.geom[1] if contact.geom[0] == floor else contact.geom[0])
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if int(model.geom_bodyid[other]) in feet and force[0] > .01:
            points.append(contact.pos[:2].copy())
    margin = None
    if len(points) >= 3:
        hull = ConvexHull(np.unique(points, axis=0))
        com = data.subtree_com[model.body('base').id]
        margin = float(-np.max(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]) * 1000)
    orientation = []
    for side, body in zip(('left', 'right'), feet):
        rotation = data.xmat[body].reshape(3, 3) @ zero.xmat[body].reshape(3, 3).T
        normal = rotation[:, 2]
        hip = data.xanchor[model.joint(side + '_hip_pitch').id]
        ankle = data.xanchor[model.joint(side + '_ankle').id]
        vector = ankle - hip
        orientation.append(dict(side=side,
            sole_roll_deg=math.degrees(math.atan2(-normal[1], normal[2])),
            sole_pitch_deg=math.degrees(math.atan2(normal[0], normal[2])),
            front_leg_tilt_deg=math.degrees(math.atan2(vector[1], -vector[2]))))
    return dict(orientation=orientation, contact_support_margin_mm=margin)


def align_floor(model, data):
    mujoco.mj_forward(model, data)
    lowest = math.inf
    for geom in range(model.ngeom):
        if not model.geom_contype[geom] or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = (model.mesh_vert[start:start + count]
                    @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
        lowest = min(lowest, float(vertices[:, 2].min()))
    shift = .0002 - lowest
    data.qpos[2] += shift
    mujoco.mj_forward(model, data)
    return dict(previous_lowest_collision_height_m=lowest,
                base_height_adjustment_m=shift, initial_ground_gap_m=.0002,
                base_z_m=float(data.qpos[2]))


def summarize(model, case, zero):
    tail = [f for f in case['frames'] if f['time_s'] >= case['requested_seconds'] - 3]
    data = mujoco.MjData(model)
    data.qpos[:] = tail[-1]['qpos']
    data.ctrl[:] = tail[-1]['ctrl']
    mujoco.mj_forward(model, data)
    torque = np.array([f['torque_nm'] for f in tail])
    forces = np.array([f['foot_forces_world_n'] for f in tail])[:, :, 2]
    return dict(**geometry(model, data, zero), settled_qpos=data.qpos.tolist(),
        steady_mean_torque_nm=torque.mean(0).tolist(),
        steady_peak_abs_torque_nm=np.abs(torque).max(0).tolist(),
        steady_mean_fz_n=forces.mean(0).tolist(), steady_min_fz_n=forces.min(0).tolist(),
        steady_max_base_tilt_deg=max(f['base_tilt_deg'] for f in tail),
        final_second_max_tilt_deg=max(f['base_tilt_deg'] for f in tail if f['time_s'] >= case['requested_seconds'] - 1))


def assess(reuse_probe=False, resume=False):
    WORK.mkdir(exist_ok=True)
    hashes = {k: digest(p) for k, p in [('source_step_sha256', STEP), ('mjcf_sha256', MJCF), ('scene_sha256', SCENE)]}
    mass = json.loads(MASS.read_text())
    assert all(hashes[k] == mass[k] for k in hashes)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert model.neq == 0 and model.nu == 16
    zero = mujoco.MjData(model)
    mujoco.mj_forward(model, zero)
    joints = model.actuator_trnid[:, 0]
    adr = model.jnt_qposadr[joints]
    names = [model.actuator(i).name for i in range(model.nu)]
    roll = [model.actuator(n).id for n in ROLL_NAMES]
    baseline_qpos, baseline_ctrl = model.key_qpos[0].copy(), model.key_ctrl[0].copy()
    if REPORT.exists():
        saved = json.loads(REPORT.read_text())
        if saved['source_step_sha256'] == hashes['source_step_sha256'] and saved['mjcf_sha256'] == hashes['mjcf_sha256']:
            baseline_qpos = np.array(saved['postures'][0]['initial_qpos'])
            baseline_ctrl = np.array(saved['postures'][0]['ctrl_rad'])
    report = dict(date='2026-10-17', **hashes, script_sha256=digest(__file__),
        scope='R12 hip-roll standing assessment; unchanged CAD, knee/pitch/yaw command angles, gains, torque limits and masses. No walking or hardware validation.',
        actuator_order=names, servo_ids=[loads.SID[n] for n in names], postures=[],
        protocol=dict(hold_seconds=13, steady_window_seconds=3, push_newtons=2,
                      push_seconds=.2, push_start_seconds=1, recovery_seconds=6),
        resources=dict(address_space_cap_bytes=4_000_000_000, workspace_cap_bytes=8_000_000_000, native_threads=1, serial=True))
    cached = {}
    previous = None
    if resume:
        path = WORK / 'assessment.json'
        previous = json.loads(path.read_text())
        assert all(previous[k] == hashes[k] for k in hashes)
        report['resumed_assessment_sha256'] = digest(path)
        report['previous_assessment_script_sha256'] = previous['script_sha256']
        for row in previous['postures']:
            cached[row['name']] = dict(case=row['hold'], ctrl_rad=row['ctrl_rad'], initial_qpos=row['initial_qpos'])
    if reuse_probe:
        assert hashes == {'source_step_sha256': 'bb9c851dfcae554653981919bf69a950febc2730c4362e562f58a0a63f7ce740',
            'mjcf_sha256': '12a9ca61f0c10bc320f6eecc8517bad557e0d89dc64500fcfb191d9dadbfd8d0',
            'scene_sha256': 'c6bc5bb5c4c9635a76727894bfbff95fe9858d084cae15cc2ec49bf89c05b795'}
        report['preliminary_sources'] = []
        for name in ('probe', 'compensated'):
            path = Path(f'/tmp/open_duck_hip_roll_{name}.json')
            source = path.with_suffix('.py')
            report['preliminary_sources'].append(dict(path=str(path), sha256=digest(path), script=str(source), script_sha256=digest(source),
                binding='Same session, unchanged formal input hashes; control and initial states checked below.'))
            for row in json.loads(path.read_text()):
                cached[row['name']] = row
    for label in ('original_inward', 'roll_zero', 'load_compensated'):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[:] = baseline_qpos
        ctrl = baseline_ctrl.copy()
        if label != 'original_inward':
            data.qpos[adr[roll]] = 0
            ctrl[roll] = 0 if label == 'roll_zero' else ROLL_CTRL
        data.ctrl[:] = ctrl
        alignment = align_floor(model, data)
        initial = data.qpos.copy()
        if label in cached:
            row = cached[label]
            assert np.allclose(row['ctrl_rad'], ctrl, rtol=0, atol=1e-13)
            assert np.allclose(row['initial_qpos'], initial, rtol=0, atol=1e-13)
            case = row['case']
            assert case['name'] == label and case['simulated_seconds'] == 13
        else:
            case = loads.simulate(model, data, label, 13, [(0, ctrl), (13, ctrl)], names, joints)
        summary = summarize(model, case, zero)
        posture = dict(name=label, ctrl_rad=ctrl.tolist(), initial_qpos=initial.tolist(),
                       home_ground_alignment=alignment, hold=case, summary=summary, disturbances=[])
        report['postures'].append(posture)
        previous_row = next((p for p in previous['postures'] if p['name'] == label), None) if previous else None
        if previous_row and len(previous_row['disturbances']) == 4:
            posture['disturbances'] = previous_row['disturbances']
        elif label != 'roll_zero':
            settled = mujoco.MjData(model)
            settled.qpos[:] = summary['settled_qpos']
            settled.qvel[:] = 0
            settled.ctrl[:] = ctrl
            mujoco.mj_forward(model, settled)
            for axis, direction in itertools.product((0, 1), (-1, 1)):
                force = np.zeros(3)
                force[axis] = 2 * direction
                push = dict(start_s=1, duration_s=.2, force_world_n=force.tolist())
                trial = loads.simulate(model, settled, f'{label}_push_{"XY"[axis]}{direction:+}', 7.2,
                                       [(0, ctrl), (7.2, ctrl)], names, joints, push=push)
                trial['summary'] = summarize(model, trial, zero)
                posture['disturbances'].append(trial)
        (WORK / 'assessment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(label, 'steady', json.dumps(summary), flush=True)
    old, _, selected = report['postures']
    start = mujoco.MjData(model)
    start.qpos[:] = old['summary']['settled_qpos']
    transition = loads.simulate(model, start, 'transition_from_inward', 9,
        [(0, np.array(old['ctrl_rad'])), (3, np.array(selected['ctrl_rad'])), (9, np.array(selected['ctrl_rad']))], names, joints)
    transition['summary'] = summarize(model, transition, zero)
    report['transition'] = transition
    (WORK / 'assessment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    tests = [selected['hold'], transition] + selected['disturbances']
    report['acceptance'] = dict(
        complete=all(t['completed'] for t in tests),
        warning_free=all(not any(t['warnings']) for t in tests),
        no_saturation=all(not any(t['saturation_fraction']) for t in tests),
        upright=all(t['max_base_tilt_deg'] < 5 for t in tests),
        recovered=all(t['summary']['final_second_max_tilt_deg'] < 2 for t in tests if 'summary' in t),
        both_feet_loaded=all(min(t['summary']['steady_min_fz_n']) > 3 for t in tests if 'summary' in t),
        front_legs_vertical=max(abs(o['front_leg_tilt_deg']) for o in selected['summary']['orientation']) < .2,
        soles_level=max(max(abs(o['sole_roll_deg']), abs(o['sole_pitch_deg'])) for o in selected['summary']['orientation']) < .2,
        support_margin_retained=selected['summary']['contact_support_margin_mm'] >= old['summary']['contact_support_margin_mm'] - 1)
    report['standing_passed'] = all(report['acceptance'].values())
    tree = ET.parse(SCENE)
    shutil.copyfile(MJCF, WORK / MJCF.name)
    if not (WORK / 'assets').exists():
        (WORK / 'assets').symlink_to(MJCF.parent / 'assets', target_is_directory=True)
    tree.getroot().find('include').set('file', MJCF.name)
    key = tree.getroot().find('keyframe/key[@name="home"]')
    key.set('qpos', ' '.join(f'{x:.14g}' for x in selected['initial_qpos']))
    key.set('ctrl', ' '.join(f'{x:.14g}' for x in selected['ctrl_rad']))
    ET.indent(tree, space='  ')
    tree.write(CANDIDATE, encoding='unicode', xml_declaration=True)
    candidate = mujoco.MjModel.from_xml_path(str(CANDIDATE))
    assert np.allclose(candidate.key_qpos[0], selected['initial_qpos'], atol=1e-13)
    assert np.allclose(candidate.key_ctrl[0], selected['ctrl_rad'], atol=1e-13)
    report['candidate_scene_sha256'] = digest(CANDIDATE)
    report['standing_only'] = True
    report['limitations'] = ['Estimated actuator gains, friction, contact proxies and masses; no measured hardware calibration.',
        '2 N / 0.2 s sampled disturbances only; not a maximum recoverable force or walking assessment.',
        'Small inward command offsets compensate load deflection; achieved geometric legs are approximately vertical.',
        'No new CAD clearance certificate; local candidate pose check required before publication.']
    assert all(digest(p) == hashes[k] for k, p in [('source_step_sha256', STEP), ('mjcf_sha256', MJCF), ('scene_sha256', SCENE)])
    (WORK / 'assessment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Standing acceptance:', report['acceptance'], flush=True)


def publish():
    report = json.loads((WORK / 'assessment.json').read_text())
    assert report['standing_passed'] and report['local_pose_check']['passed']
    for key, path in [('source_step_sha256', STEP), ('mjcf_sha256', MJCF), ('scene_sha256', SCENE)]:
        assert digest(path) == report[key], key
    assert digest(CANDIDATE) == report['candidate_scene_sha256']
    old = mujoco.MjModel.from_xml_path(str(SCENE))
    new = mujoco.MjModel.from_xml_path(str(CANDIDATE))
    attributes = ('body_mass', 'body_ipos', 'body_inertia', 'body_pos', 'body_quat',
        'jnt_pos', 'jnt_axis', 'jnt_range', 'jnt_stiffness', 'dof_damping', 'dof_frictionloss',
        'geom_type', 'geom_dataid', 'geom_pos', 'geom_quat', 'geom_contype', 'geom_conaffinity',
        'geom_friction', 'geom_solref', 'geom_solimp', 'mesh_vert', 'mesh_face',
        'actuator_ctrlrange', 'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm')
    assert all(np.array_equal(getattr(old, a), getattr(new, a)) for a in attributes)
    roll = [old.actuator(n).id for n in ROLL_NAMES]
    qadr = old.jnt_qposadr[old.actuator_trnid[roll, 0]]
    assert set(np.flatnonzero(abs(new.key_qpos[0] - old.key_qpos[0]) > 1e-12)) <= {2, *qadr}
    assert set(np.flatnonzero(abs(new.key_ctrl[0] - old.key_ctrl[0]) > 1e-12)) <= set(roll)
    assert np.allclose(new.key_qpos[0, qadr], 0, atol=1e-13)
    assert np.allclose(new.key_ctrl[0, roll], ROLL_CTRL, atol=1e-13)
    shutil.copyfile(SCENE, WORK / 'previous_scene.xml')
    shutil.copyfile(MASS, WORK / 'previous_mass.json')
    temporary = SCENE.with_suffix('.xml.tmp')
    shutil.copyfile(CANDIDATE, temporary)
    temporary.replace(SCENE)
    compiled = mujoco.MjModel.from_xml_path(str(SCENE))
    assert all(np.array_equal(getattr(compiled, a), getattr(new, a)) for a in attributes + ('key_qpos', 'key_ctrl'))
    report.update(status='published', published_scene_sha256=digest(SCENE),
        publisher_sha256=digest(__file__), arrays_preserved_except_keyframes=True,
        conclusion='Actual inward hip roll is unnecessary for tested standing. Adopt approximately vertical front-view legs with load-compensating position commands.')
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    selected = report['postures'][2]
    mass = json.loads(MASS.read_text())
    mass['prior_home_validation'] = mass['validation']
    mass['scene_sha256'] = report['published_scene_sha256']
    mass['home_free_qpos'] = compiled.key_qpos[0, :7].tolist()
    mass['home_joint_angles_rad'] = {compiled.joint(j).name: float(compiled.key_qpos[0, compiled.jnt_qposadr[j]])
        for j in range(compiled.njnt) if compiled.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}
    mass['home_ctrl_rad'] = {compiled.actuator(a).name: float(compiled.key_ctrl[0, a]) for a in range(compiled.nu)}
    mass['home_ground_alignment'] = selected['home_ground_alignment']
    hold = selected['hold']
    rollout = dict(mode='free_home_hold', seconds=hold['simulated_seconds'], finite=True,
        warning_counts=hold['warnings'], max_tracking_error_deg=hold['max_tracking_error_deg'],
        saturation_fraction=hold['saturation_fraction'], max_base_tilt_deg=hold['max_base_tilt_deg'],
        scope='R12 13-second standing check with distinct geometric home angles and load-compensating actuator commands.')
    mass['validation'] = dict(synchronization_passed=True, source_step_sha256=report['source_step_sha256'],
        mjcf_sha256=report['mjcf_sha256'], scene_sha256=report['published_scene_sha256'],
        geometry_evidence='R11 geometry, axes, inertials and proxies preserved; all listed compiled structural arrays identical. Full previous evidence retained in prior_home_validation; its old home rollout is historical.',
        compiled_arrays_checked=list(attributes), rollouts=[rollout],
        home_assessment=str(REPORT.relative_to(ROOT)), home_assessment_sha256=digest(REPORT),
        standing_passed=True, local_pose_check_passed=True)
    mass['home_update'] = dict(id='R12', previous_scene_sha256=report['scene_sha256'],
        assessment=str(REPORT.relative_to(ROOT)), physical_front_leg_tilt_deg=[o['front_leg_tilt_deg'] for o in selected['summary']['orientation']],
        command_offsets_reason='Position actuator load compensation; command angles differ from actual standing angles.',
        no_hardware_calibration=True)
    mass['published_verification'].update(home_keyframe_updated=True, structural_arrays_identical=True,
                                         keyframes_match_validated_candidate=True)
    mass.pop('preview', None)
    MASS.write_text(json.dumps(mass, ensure_ascii=False, indent=2) + '\n')
    # Keep the cached generator inputs consistent without rebuilding geometry.
    path = Path('/tmp/open-duck-dynamics-sync/manifest.json')
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest['source_step_sha256'] == report['source_step_sha256']:
            for key in ('home_free_qpos', 'home_joint_angles_rad', 'home_ctrl_rad', 'home_ground_alignment'):
                manifest[key] = mass[key]
            manifest['home_update'] = mass['home_update']
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for name in ('FINAL_ASSEMBLY_REVIEW.json', 'LEG_LINK_SHORTENING_REPORT.json'):
        path = OUT / name
        data = json.loads(path.read_text())
        data['stance_update'] = mass['home_update'] | dict(scene_sha256=mass['scene_sha256'], standing_passed=True)
        if name.startswith('FINAL'):
            data['prior_home_simulation'] = data['simulation']
            data['simulation'] = [rollout]
        data['user_visual_acceptance'] = 'Leg shortening accepted by user; R12 standing preview pending review.'
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print('Published R12 home keyframe; STEP and physical model arrays preserved.', flush=True)


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
    camera.distance = .83
    camera.azimuth = 180
    camera.elevation = -2
    pictures = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    with mujoco.Renderer(model, height=900, width=900) as renderer:
        for index, label in [(0, 'Before: inward legs'), (2, 'After: legs approximately vertical')]:
            data = mujoco.MjData(model)
            data.qpos[:] = report['postures'][index]['summary']['settled_qpos']
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            picture = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 900, 55), fill='white')
            draw.text((20, 15), label + ' (simulated under load)', fill='black', font=font)
            pictures.append(picture)
    packet = Image.new('RGB', (1800, 900), 'white')
    for index, picture in enumerate(pictures):
        packet.paste(picture, (900 * index, 0))
    output = OUT / 'CURRENT_HIP_ROLL_ASSESSMENT.png'
    packet.save(output)
    report['preview'] = dict(path=str(output.relative_to(ROOT)), sha256=digest(output), source='Actual settled simulation states, same camera.')
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    mass = json.loads(MASS.read_text())
    mass['validation']['home_assessment_sha256'] = digest(REPORT)
    MASS.write_text(json.dumps(mass, ensure_ascii=False, indent=2) + '\n')
    print('Rendered actual loaded front views:', output, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['publish']:
        publish()
    elif sys.argv[1:] == ['render']:
        render()
    else:
        assess(reuse_probe='--reuse-probe' in sys.argv, resume='--resume' in sys.argv)
