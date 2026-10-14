"""Review the current STEP without changing geometry; run via run_memory_limited.py.

static: neutral BRep overlaps and cache actual STEP meshes/axes.
combined: sampled joint combinations, FCL candidates confirmed against BRep.
simulation: compile and exercise the existing MuJoCo model, comparing CAD axes.
visual: render the measured CAD poses, including confirmed collision witnesses.
"""
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys

import numpy as np
import trimesh
from cadgen import build123d as bd, read_step
from OCP.OSD import OSD_Parallel, OSD_ThreadPool
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN

from full_motion_current import TARGET, MJCF, LIMITS, JOINT_NAMES, _leaves, _mesh, is_moving, measured_axes
from exact_motion_current import _overlap

OSD_Parallel.SetUseOcctThreads_s(True)
OSD_ThreadPool.DefaultPool_s().Init(1)
WORK = Path('/tmp/open-duck-final-review')
REPORT = TARGET.parent / 'FINAL_ASSEMBLY_REVIEW.json'
SID = list(LIMITS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(report):
    assert digest(TARGET) == report['target_sha256'], 'STEP changed during review'
    budget = report.setdefault('resources', dict(address_space_limit_bytes=4_000_000_000,
        native_threads=1, workers=1, serial=True, peak_rss_bytes_by_phase={}))
    budget['peak_rss_bytes_by_phase'][sys.argv[1]] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def matrix(pose, chain, axes):
    result = np.eye(4)
    for sid in chain:
        result = result @ trimesh.transformations.rotation_matrix(
            math.radians(pose.get(sid, 0)), axes[sid][1], axes[sid][0])
    return result


def static():
    WORK.mkdir(exist_ok=True)
    robot = read_step(TARGET)
    leaves = _leaves(robot)
    axes = measured_axes(robot)
    report = dict(date='2026-10-14', target_sha256=digest(TARGET), mjcf_sha256=digest(MJCF),
                  geometry_changed=False, axes=axes, limits_deg=LIMITS, parts=[],
                  static=dict(completed=False, aabb_candidates=0, intersections=[], errors=[]))
    arrays = {}
    shapes = []
    for i, row in enumerate(leaves):
        shape = bd.Compound(row['shape'].solids())
        shapes.append(shape)
        box = shape.bounding_box()
        item = {k: row[k] for k in ('path', 'root', 'servo', 'label')}
        item.update(chain=[sid for sid in SID if is_moving(row, sid)],
                    bbox_mm=[list(box.min), list(box.max)], solids=len(shape.solids()),
                    positive_volumes=all(s.volume > 0 for s in shape.solids()), valid=shape.is_valid)
        report['parts'].append(item)
        mesh = _mesh(shape)
        arrays[f'v{i}'], arrays[f'f{i}'] = mesh.vertices, mesh.faces
        if i % 20 == 0:
            print('prepare', i, row['label'], flush=True)
    np.savez_compressed(WORK / 'meshes.npz', **arrays)
    report['static']['physical_leaf_count'] = len(leaves)
    save(report)
    for i, j in itertools.combinations(range(len(leaves)), 2):
        a, b = report['parts'][i], report['parts'][j]
        if a['servo'] and a['servo'] == b['servo']:
            continue  # Purchased servo internals are not assembly clashes.
        lo = np.maximum(a['bbox_mm'][0], b['bbox_mm'][0])
        hi = np.minimum(a['bbox_mm'][1], b['bbox_mm'][1])
        if np.any(hi - lo <= 1e-6):
            continue
        report['static']['aabb_candidates'] += 1
        try:
            volume = _overlap(shapes[i], shapes[j])
            if volume > 1e-6:
                report['static']['intersections'].append(dict(
                    parts=[i, j], labels=[a['label'], b['label']], overlap_mm3=volume,
                    same_rigid_body=a['chain'] == b['chain']))
                print('static overlap', a['label'], b['label'], round(volume, 6), flush=True)
        except Exception as exc:
            report['static']['errors'].append(dict(parts=[i, j], error=str(exc)))
        if report['static']['aabb_candidates'] % 20 == 0:
            save(report)
            print('static checked', report['static']['aabb_candidates'], flush=True)
        gc.collect()
    report['static']['completed'] = True
    save(report)


def poses():
    result = [('neutral', {})]
    for group, ids in [('head', SID[10:]), ('right_leg', SID[:5]), ('left_leg', SID[5:10])]:
        for values in itertools.product(*(sorted(set((LIMITS[s][0], 0, LIMITS[s][1]))) for s in ids)):
            result.append((group, dict(zip(ids, values))))
    # Mirror the legs anatomically, including both inward yaw/roll extremes.
    for values in itertools.product(*(sorted(set((LIMITS[s][0], 0, LIMITS[s][1]))) for s in SID[:5])):
        pose = dict(zip(SID[:5], values))
        pose.update({left: (-pose[right] if n < 3 else pose[right])
                     for n, (right, left) in enumerate(zip(SID[:5], SID[5:10]))})
        result.append(('both_legs', pose))
    rng = np.random.default_rng(20261014)
    for _ in range(128):
        result.append(('whole_robot_random', {s: float(rng.uniform(*LIMITS[s])) for s in SID}))
    return result


def interior_witness(left, right):
    """Prove positive common material with a ball inside both exact solids."""
    boxes = [s.bounding_box() for s in (left, right)]
    lower = np.maximum(list(boxes[0].min), list(boxes[1].min))
    upper = np.minimum(list(boxes[0].max), list(boxes[1].max))
    if np.any(upper-lower <= .04):
        return None
    classifiers = [[BRepClass3d_SolidClassifier(s.wrapped) for s in shape.solids()]
                   for shape in (left, right)]
    boundaries = [bd.Compound(shape.faces()) for shape in (left, right)]
    for fraction in itertools.product((.5, .25, .75, .1, .9), repeat=3):
        point = lower + np.array(fraction)*(upper-lower)
        inside = []
        for group in classifiers:
            states = []
            for classifier in group:
                classifier.Perform(gp_Pnt(*point), 1e-7)
                states.append(classifier.State() == TopAbs_IN)
            inside.append(any(states))
        if all(inside):
            gap = min(shape.distance_to(bd.Vertex(*point)) for shape in boundaries)
            if gap > .02:
                return dict(point_mm=point.tolist(), ball_radius_mm=.01,
                            overlap_lower_bound_mm3=4*math.pi*.01**3/3)
    return None


def self_check():
    box = bd.Box(10, 10, 10)
    assert interior_witness(box, box.moved(bd.Location((5, 0, 0))))
    assert interior_witness(box, box.moved(bd.Location((20, 0, 0)))) is None
    shell = box.cut(bd.Box(6, 6, 6))
    assert interior_witness(shell, bd.Box(4, 4, 4)) is None
    print('PASS: overlapping solids, disjoint solids, and nested cavity', flush=True)


def combined():
    report = json.loads(REPORT.read_text())
    key = sys.argv[1] if sys.argv[1] in ('operating', 'antenna') else 'combined'
    parts, axes = report['parts'], report['axes']
    if key == 'antenna':
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
        rows = _leaves(read_step(TARGET))
        for row in rows:
            if row['label'] != 'sg90':
                continue
            side = row['shape'].center().Y > .1
            name = 'A_plus_y' if side else 'A_minus_y'
            shafts = []
            for face in row['shape'].faces():
                surface = BRepAdaptor_Surface(face.wrapped, True)
                if surface.GetType() == GeomAbs_Cylinder and abs(surface.Cylinder().Radius()-2.425) < 1e-6:
                    cylinder = surface.Cylinder()
                    shafts.append([list(cylinder.Location().Coord()), list(cylinder.Axis().Direction().Coord())])
            assert shafts, 'SG90 output-axis cylinder is missing'
            axes[name] = shafts[0]
            for part, actual in zip(parts, rows):
                assert part['path'] == actual['path']
                if part['label'] in ('antenna', 'left_antenna_holder', 'right_antenna_holder') and (actual['shape'].center().Y > .1) == side:
                    if name not in part['chain']:
                        part['chain'].append(name)
        report['antenna_diagnostic_limits_deg'] = {'A_plus_y': [-90, 90], 'A_minus_y': [-90, 90]}
    cache = np.load(WORK / 'meshes.npz')
    groups = {}
    for i, part in enumerate(parts):
        group = tuple(part['chain'])
        if group not in groups:
            groups[group] = dict(manager=trimesh.collision.CollisionManager(), indices=[])
        groups[group]['manager'].add_object(str(i), trimesh.Trimesh(cache[f'v{i}'], cache[f'f{i}'], process=False))
        groups[group]['indices'].append(i)
    for group in groups.values():
        boxes = np.array([parts[i]['bbox_mm'] for i in group['indices']])
        lower, upper = boxes[:, 0].min(0), boxes[:, 1].max(0)
        group['corners'] = np.array(list(itertools.product(*zip(lower, upper))))
    if key == 'antenna':
        samples = [('antenna_single', {side: angle}) for side in ('A_plus_y', 'A_minus_y') for angle in range(-90, 91, 5)]
        for yaw, roll, left, right in itertools.product((-100, 0, 100), (-30, 0, 30), (-90, 0, 90), (-90, 0, 90)):
            samples.append(('antenna_with_head_yaw_roll', dict(S13=yaw, S14=roll, A_plus_y=left, A_minus_y=right)))
    elif key == 'operating':
        import xml.etree.ElementTree as ET
        root = ET.parse(MJCF.parent / 'scene_flat_terrain.xml').getroot()
        home_values = list(map(float, root.find('keyframe/key').get('ctrl').split()))
        actuator_names = [a.get('joint') for a in ET.parse(MJCF).getroot().find('actuator')]
        home = {sid: math.degrees(home_values[actuator_names.index(name)]) for sid, name in JOINT_NAMES.items()}
        samples = [('startup_path', {sid: value*u for sid, value in home.items()}) for u in np.linspace(0, 1, 21)]
        for values in itertools.product(*((LIMITS[s][0]*.5, 0, LIMITS[s][1]*.5) for s in SID[10:])):
            samples.append(('home_with_head_motion', {**home, **dict(zip(SID[10:], values))}))
    else:
        samples = poses()
    candidates = {}
    for index, (group, pose) in enumerate(samples):
        transforms = {chain: matrix(pose, chain, axes) for chain in groups}
        bounds = {}
        for chain, group_data in groups.items():
            t = transforms[chain]
            for i in group_data['indices']:
                group_data['manager'].set_transform(str(i), t)
            corners = group_data['corners'] @ t[:3, :3].T + t[:3, 3]
            bounds[chain] = (corners.min(0), corners.max(0))
        contacts = []
        for a, b in itertools.combinations(groups, 2):
            if np.allclose(transforms[a], transforms[b], atol=1e-9):
                continue  # The neutral static pass already checked these.
            if np.any(np.minimum(bounds[a][1], bounds[b][1]) < np.maximum(bounds[a][0], bounds[b][0])):
                continue
            _, _, found = groups[a]['manager'].in_collision_other(groups[b]['manager'], return_names=True, return_data=True)
            contacts.extend(found)
        for contact in contacts:
            i, j = sorted(map(int, contact.names))
            a, b = parts[i], parts[j]
            if a['chain'] == b['chain'] or (a['servo'] and a['servo'] == b['servo']):
                continue
            depth = max(0, float(contact.depth))
            if depth <= .05:
                continue
            pair = (i, j)
            witness = dict(group=group, pose_deg=pose, mesh_depth_mm=depth)
            # Keep one witness per pair and test it exactly; this is a sampled
            # issue search, never a certification of all combination angles.
            if pair not in candidates or depth > candidates[pair]['mesh_depth_mm']:
                candidates[pair] = witness
        if index % 20 == 0:
            print('combined pose', index, '/', len(samples), 'candidate pairs', len(candidates), flush=True)
    (WORK / f'{key}-candidates.json').write_text(json.dumps([
        dict(parts=list(pair), **witness) for pair, witness in candidates.items()]))
    previous_checks = report.get(key, {}).get('checks', [])
    checked_pairs = {tuple(row['parts']) for row in previous_checks}
    report[key] = dict(completed=False, pose_count=len(samples), random_seed=20261014,
        group_counts={g: sum(name == g for name, _ in samples) for g, _ in samples},
        mesh_tolerance_mm=.5, mesh_penetration_threshold_mm=.05,
        witness_policy='Maximum FCL depth per pair; one BRep witness per pair. Other candidate samples are not certified.',
        candidate_pair_count=len(candidates), checks=previous_checks, errors=[])
    save(report)
    confirm(report, key, {pair: witness for pair, witness in candidates.items() if pair not in checked_pairs})


def confirm(report, key, candidates):
    parts, axes = report['parts'], report['axes']
    robot = read_step(TARGET)
    shapes = [bd.Compound(row['shape'].solids()) for row in _leaves(robot)]
    for (i, j), witness in sorted(candidates.items(), key=lambda row: -row[1]['mesh_depth_mm']):
        a, b = parts[i], parts[j]
        try:
            def posed(shape, chain):
                t = matrix(witness['pose_deg'], chain, axes)
                plane = bd.Plane(origin=tuple(t[:3, 3]), x_dir=tuple(t[:3, 0]), z_dir=tuple(t[:3, 2]))
                return shape.moved(bd.Location(plane))
            # A rigid Location preserves analytic surfaces and re-normalizes
            # floating-point rotation axes before handing them to OCC.
            left, right = posed(shapes[i], a['chain']), posed(shapes[j], b['chain'])
            suspect = {'head', 'left_antenna_holder', 'right_antenna_holder'}
            interior = None if suspect.intersection((a['label'], b['label'])) else interior_witness(left, right)
            overlap = None if interior else _overlap(left, right)
            gap = 0.0 if interior else left.distance_to(right)
            report[key]['checks'].append(dict(parts=[i, j], labels=[a['label'], b['label']],
                **witness, overlap_mm3=overlap, gap_mm=gap,
                interior_witness=interior, confirmation_method='BRep interior ball' if interior else 'BRep intersection',
                confirmed_collision=bool(interior) or overlap > 1e-6))
            print('combined exact', a['label'], b['label'], 'volume', round(overlap, 6) if overlap is not None else 'positive interior ball', flush=True)
            del left, right
        except Exception as exc:
            report[key]['errors'].append(dict(parts=[i, j], **witness, error=str(exc)))
        save(report)
        gc.collect()
    report[key]['completed'] = True
    report[key]['exact_confirmation_complete'] = not report[key]['errors']
    save(report)


def retry_confirmation():
    report = json.loads(REPORT.read_text())
    key = sys.argv[2] if len(sys.argv) > 2 else 'combined'
    suspect = {'head', 'left_antenna_holder', 'right_antenna_holder'}
    report[key]['checks'] = [row for row in report[key]['checks']
        if not (row.get('interior_witness') and suspect.intersection(row['labels']))]
    source = WORK / f'{key}-candidates.json'
    rows = json.loads(source.read_text()) if source.exists() else report[key]['errors']
    checked = {tuple(row['parts']) for row in report[key]['checks']}
    candidates = {tuple(row['parts']): {k: row[k] for k in ('group', 'pose_deg', 'mesh_depth_mm')}
                  for row in rows if tuple(row['parts']) not in checked}
    report[key]['errors'] = []
    report[key]['completed'] = False
    confirm(report, key, candidates)


def interfaces():
    """Measure hole axes on the actual placed solids, independent of old datums."""
    import validate_s01_s10_interfaces_local as v
    import validate_s11_s14_interfaces_local as upper
    report = json.loads(REPORT.read_text())
    specs = {}
    original, argv = v._validate_joint, sys.argv
    try:
        v._validate_joint = lambda sid, spec, vendor: specs.update({sid: spec})
        sys.argv = [argv[0], '--robot', str(TARGET)]
        v.main()
        upper.main()
    finally:
        v._validate_joint, sys.argv = original, argv
    assert len(specs) == 14
    result = dict(completed=False, joints=[],
        scope='Actual internal cylindrical axes, explicit existing host mapping and nearest distance only; no screw engagement, bearing area, or tool-access certification.')
    # The case also has seven housing-assembly bores. Identify its four
    # external mounting bores by their invariant radii about the shaft,
    # avoiding baked STEP placement transforms and unrelated screw holes.
    vendor_center = bd.Vector(*v.m.ZP10D_HORN_FACE_MIDPOINT)
    case_pattern_radii = [v._line_distance(bd.Vector(*p), bd.Vector(1, 0, 0),
                                          vendor_center, bd.Vector(1, 0, 0))
                          for p in v.m.ZP10D_CASE_HOLES]
    result['case_pattern_radial_distances_mm'] = case_pattern_radii
    def bores(shape, direction, lower, upper):
        found = []
        for face in shape.faces():
            if face.geom_type != bd.GeomType.CYLINDER or face.radius is None or not lower-1e-6 <= face.radius <= upper+1e-6:
                continue
            line = face.axis_of_rotation
            if abs(line.direction.dot(direction)) < .999:
                continue
            point = face.position_at(.5, .5)
            radial = point-line.position
            radial -= line.direction*radial.dot(line.direction)
            if radial.length < 1e-6 or face.normal_at(point).dot(radial.normalized()) >= -.5:
                continue
            if not any(v._line_distance(line.position, line.direction, p, d) < .01 for p, d, _ in found):
                found.append((line.position, line.direction, face.radius))
        return found
    for sid, spec in specs.items():
        body, passive, active = v._servo_parts(spec.servo)
        direction, center = v._axis(spec.servo)
        case_candidates = bores(body, direction, .7, 1.35)
        case = [(p, d, radius) for p, d, radius in case_candidates
                if any(abs(v._line_distance(p, d, center, direction)-r) < .01 for r in case_pattern_radii)]
        groups = [('case', case, spec.case_hosts, body)]
        for name, horn in [('active', active), ('passive', passive)]:
            if horn is None or (spec.active_host is not None and name == 'passive'):
                continue
            hosts = (spec.active_host,) if spec.active_host is not None else (v._nearest_host(horn, spec.horn_hosts),)
            lines = [(p, d, 1.5) for p, d in v._horn_pattern_lines(horn, direction, center)]
            groups.append((name, lines, hosts, horn))
        joint = dict(servo=sid, case_candidate_axes_count=len(case_candidates), patterns=[])
        for name, lines, hosts, component in groups:
            pattern = dict(interface=name, expected_axes=4, detected_axes=len(lines), holes=[],
                           nearest_host_gap_mm=min(component.distance_to(host) for host in hosts))
            host_bores = [(host.label, bores(host, direction, .7, 1.8)) for host in hosts]
            for point, axis, radius in lines:
                matches = []
                for host_label, cylinders in host_bores:
                    for hp, hd, hr in cylinders:
                        error = v._line_distance(point, axis, hp, hd)
                        if error < .01:
                            matches.append(dict(host=host_label, radius_mm=hr, axis_error_mm=error))
                pattern['holes'].append(dict(point_mm=list(point), axis=list(axis), component_radius_mm=radius, matches=matches))
            pattern['all_axes_matched'] = len(lines) == 4 and all(row['matches'] for row in pattern['holes'])
            joint['patterns'].append(pattern)
            print('interface', sid, name, 'axes', len(lines), 'matched', sum(bool(row['matches']) for row in pattern['holes']), flush=True)
        result['joints'].append(joint)
    result['completed'] = True
    result['all_pattern_axes_matched'] = all(p['all_axes_matched'] for j in result['joints'] for p in j['patterns'])
    report['interface_validation'] = result
    save(report)


def simulation():
    import mujoco
    import xml.etree.ElementTree as ET
    report = json.loads(REPORT.read_text())
    scene = MJCF.parent / 'scene_flat_terrain.xml'
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = dict(completed=False, version=mujoco.__version__, nq=model.nq, nv=model.nv,
               nu=model.nu, total_mass_kg=float(model.body_mass.sum()), joints=[],
               enabled_collision_geoms=[], compiled=True, rollouts=[])
    for i in range(model.ngeom):
        if model.geom_contype[i] or model.geom_conaffinity[i]:
            sim['enabled_collision_geoms'].append(dict(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
                body=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])),
                contype=int(model.geom_contype[i]), conaffinity=int(model.geom_conaffinity[i])))
    for sid, name in JOINT_NAMES.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        cad_origin, cad_direction = map(np.array, report['axes'][sid])
        origin = (data.xanchor[j] - data.xpos[model.jnt_bodyid[0]]) * 1000
        direction = data.xaxis[j]
        delta = origin - cad_origin
        sim['joints'].append(dict(servo=sid, joint=name,
            limits_deg=np.degrees(model.jnt_range[j]).tolist(),
            control_limits_deg=np.degrees(model.actuator_ctrlrange[actuator]).tolist(),
            force_limits_nm=model.actuator_forcerange[actuator].tolist(),
            axis_direction_dot=float(np.dot(cad_direction, direction)),
            origin_delta_mm=delta.tolist(), axis_line_error_mm=float(np.linalg.norm(np.cross(delta, direction))),
            joint_limited=bool(model.jnt_limited[j]), control_limited=bool(model.actuator_ctrllimited[actuator])))
    sim['geometry_bounds_comparison'] = []
    for label, mesh_stem in [('head', 'head'), ('head_pitch_to_yaw', 'head_pitch_to_yaw__head_pitch_to_yaw_zp10d'),
                             ('body_back', 'body_back'), ('body_middle_top', 'body_middle_top'),
                             ('body_middle_bottom', 'body_middle_bottom'),
                             ('trunk_top', 'trunk_assembly__trunk_top_zp10d')]:
        matches = [p for p in report['parts'] if p['label'] == label]
        if len(matches) != 1:
            continue
        mesh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_stem)
        geoms = np.flatnonzero(model.geom_dataid == mesh) if mesh >= 0 else []
        if len(geoms) == 0:
            sim['geometry_bounds_comparison'].append(dict(part=label, mesh=mesh_stem, mesh_found=False))
            continue
        g = int(geoms[0]); start = model.mesh_vertadr[mesh]; count = model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        vertices = (vertices - data.xpos[model.jnt_bodyid[0]]) * 1000
        bounds = np.array([vertices.min(0), vertices.max(0)])
        cad_bounds = np.array(matches[0]['bbox_mm'])
        sim['geometry_bounds_comparison'].append(dict(part=label, mesh=mesh_stem, mesh_found=True,
            cad_bbox_mm=cad_bounds.tolist(), simulation_bbox_mm=bounds.tolist(),
            max_bound_delta_mm=float(abs(bounds-cad_bounds).max())))
    sim['keyframe_violations'] = []
    for k in range(model.nkey):
        for row in sim['joints']:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, row['joint'])
            value = float(model.key_qpos[k, model.jnt_qposadr[j]])
            if value < model.jnt_range[j, 0] - 1e-8 or value > model.jnt_range[j, 1] + 1e-8:
                sim['keyframe_violations'].append(dict(key=k, joint=row['joint'], angle_deg=math.degrees(value)))
    ids = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    qadr = model.jnt_qposadr[ids]
    vadr = model.jnt_dofadr[ids]
    peak_loads = np.zeros(model.nu)
    load_poses = [{} for _ in range(model.nu)]
    for _, pose in poses():
        mujoco.mj_resetData(model, data)
        for sid, value in pose.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES[sid])
            data.qpos[model.jnt_qposadr[j]] = math.radians(value)
        mujoco.mj_forward(model, data)
        load = abs(data.qfrc_bias[vadr])
        for a in np.flatnonzero(load > peak_loads):
            peak_loads[a] = load[a]
            load_poses[a] = pose
    sim['gravity_load_peaks'] = [dict(
        actuator=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a),
        torque_nm=float(peak_loads[a]), stall_ratio=float(peak_loads[a] / model.actuator_forcerange[a, 1]),
        pose_deg=load_poses[a]) for a in range(model.nu)]
    fixed_xml = ET.parse(MJCF).getroot()
    fixed_xml.find('compiler').set('meshdir', str(MJCF.parent / 'assets'))
    fixed_base = fixed_xml.find('worldbody/body')
    fixed_base.remove(fixed_base.find('freejoint'))
    fixed_model = mujoco.MjModel.from_xml_string(ET.tostring(fixed_xml, encoding='unicode'))
    for mode in ('fixed_base_neutral', 'fixed_base_sweep', 'free_home_hold'):
        run_model = model if mode == 'free_home_hold' else fixed_model
        data = mujoco.MjData(run_model)
        run_qadr = run_model.jnt_qposadr[run_model.actuator_trnid[:, 0]]
        base_id = mujoco.mj_name2id(run_model, mujoco.mjtObj.mjOBJ_BODY, 'base')
        mujoco.mj_resetData(run_model, data)
        if mode == 'free_home_hold':
            mujoco.mj_resetDataKeyframe(run_model, data, 0)
        elif mode == 'fixed_base_sweep':
            # Start at the sinusoid's centre so tracking metrics do not include
            # an artificial initial step to asymmetric joint-range midpoints.
            data.qpos[run_qadr] = run_model.actuator_ctrlrange.mean(axis=1)
            data.ctrl[:] = data.qpos[run_qadr]
        mujoco.mj_forward(run_model, data)
        home = data.qpos.copy()
        peak_error = np.zeros(model.nu)
        saturation = np.zeros(model.nu)
        limit_excess = np.zeros(model.nu)
        lowest_z = float(data.xpos[base_id, 2])
        max_tilt = 0.0
        warnings_before = data.warning.number.copy()
        frames = []
        steps = round(8 / run_model.opt.timestep)
        for step in range(steps):
            if mode == 'fixed_base_sweep':
                r = run_model.actuator_ctrlrange
                data.ctrl[:] = .5 * (r[:, 0] + r[:, 1]) + .5 * (r[:, 1] - r[:, 0]) * math.sin(2 * math.pi * step * run_model.opt.timestep / 8)
            else:
                data.ctrl[:] = home[run_qadr]
            mujoco.mj_step(run_model, data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            peak_error = np.maximum(peak_error, abs(data.qpos[run_qadr] - data.ctrl))
            limit_excess = np.maximum(limit_excess, np.maximum(
                run_model.jnt_range[run_model.actuator_trnid[:, 0], 0] - data.qpos[run_qadr], data.qpos[run_qadr] - run_model.jnt_range[run_model.actuator_trnid[:, 0], 1]))
            saturation += abs(data.actuator_force) >= .99 * run_model.actuator_forcerange[:, 1]
            lowest_z = min(lowest_z, float(data.xpos[base_id, 2]))
            rotation = data.xmat[base_id].reshape(3, 3)
            max_tilt = max(max_tilt, math.degrees(math.acos(float(np.clip(rotation[2, 2], -1, 1)))))
            if step % max(1, steps // 100) == 0:
                frames.append(dict(time=float(data.time), base_z=float(data.xpos[base_id, 2]),
                                   qpos=data.qpos.tolist(), contacts=int(data.ncon)))
        sim['rollouts'].append(dict(mode=mode, seconds=8, steps=steps, finite=True,
            max_tracking_error_deg=np.degrees(peak_error).tolist(), saturation_fraction=(saturation / steps).tolist(),
            max_limit_excess_deg=np.degrees(limit_excess).tolist(),
            actuator_order=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)],
            lowest_base_z_m=lowest_z, final_base_z_m=float(data.xpos[base_id, 2]),
            max_base_tilt_deg=max_tilt,
            warning_counts=(data.warning.number - warnings_before).tolist(), frames=frames))
        print('simulation', mode, 'base_z', data.xpos[base_id, 2], 'peak error deg', max(np.degrees(peak_error)), flush=True)
    sim['completed'] = True
    sim['limitations'] = ['Existing MJCF geometry and mass source is an older STEP.',
        'Only enabled collision geoms can participate in MuJoCo contact; CAD checks independently use all physical STEP parts.',
        'Fixed-base tests remove only the free joint, retain gravity and the legacy link properties, and do not validate balance.',
        'No trained ONNX policy is available locally; no walking-policy rollout was performed.',
        'Actuator response and continuous torque are unmeasured; stall torque is not a continuous rating.']
    report['simulation'] = sim
    save(report)


def visual():
    os.environ['MPLCONFIGDIR'] = str(WORK / 'mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    report = json.loads(REPORT.read_text())
    cache = np.load(WORK / 'meshes.npz')
    collisions = [r for r in report.get('combined', {}).get('checks', []) if r['confirmed_collision']]
    operating = [r for r in report.get('operating', {}).get('checks', []) if r['confirmed_collision']]
    ranked = sorted(collisions, key=lambda r: -(r['overlap_mm3'] or 0))
    selected = []
    for predicate in (lambda r: 'head' in r['labels'], lambda r: all('foot' in s for s in r['labels']),
                      lambda r: 'head_bot_sheet' in r['labels']):
        matches = [r for r in ranked if predicate(r)]
        preferred = [r for r in operating if predicate(r)]
        if preferred:
            matches = preferred
        if matches:
            selected.append(matches[0])
    witnesses = [({}, 'Current STEP: neutral', [])] + [
        (r['pose_deg'], 'Collision: ' + ' / '.join(r['labels']), r['parts']) for r in selected]
    while len(witnesses) < 4:
        witnesses.append(({'S13': 100, 'S14': 30}, 'S13 +100 / S14 +30 deg', []))
    def draw(ax, pose, highlight):
        points = []
        for i, p in enumerate(report['parts']):
            v = cache[f'v{i}']; faces = cache[f'f{i}']
            t = matrix(pose, p['chain'], report['axes'])
            v = v @ t[:3, :3].T + t[:3, 3]
            points.append(v)
            color = '#e83924' if i in highlight else ('#e78b1e' if p['label'] == 'speaker_stand' else '#51647a')
            if p['label'] in ('head', 'body_front', 'body_back', 'body_middle_top', 'body_middle_bottom'):
                alpha = .17
            else:
                alpha = .85
            ax.add_collection3d(Poly3DCollection(v[faces], facecolors=color, edgecolors='none', alpha=alpha))
        v = np.vstack(points)
        ax.auto_scale_xyz(*v.T); ax.set_box_aspect(v.max(0) - v.min(0)); ax.view_init(20, -55)
        ax.set_xlabel('X mm'); ax.set_ylabel('Y mm'); ax.set_zlabel('Z mm')
    fig = plt.figure(figsize=(16, 15))
    for i, (pose, title, highlight) in enumerate(witnesses):
        if highlight and any(report['parts'][p]['label'] in ('head', 'head_bot_sheet') for p in highlight):
            title += '\n' + ', '.join(f'{sid}={pose.get(sid, 0):g} deg' for sid in SID[10:])
        ax = fig.add_subplot(2, 2, i+1, projection='3d'); draw(ax, pose, highlight)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=9, pad=18)
    fig.subplots_adjust(left=.01, right=.99, bottom=.04, top=.94, hspace=.22, wspace=.05)
    output = TARGET.parent / 'FINAL_ASSEMBLY_REVIEW.png'
    fig.savefig(output, dpi=130); plt.close(fig)
    report['visual'] = dict(file=str(output), source='Actual STEP tessellation and measured CAD joint axes; collision witness parts in red.',
                           panels=[dict(pose_deg=pose, title=title, highlighted_parts=highlight) for pose, title, highlight in witnesses])
    save(report)
    print('saved', output, flush=True)


def animation():
    """Replay local collision witnesses in the existing STEP Viewer."""
    report = json.loads(REPORT.read_text())
    assert digest(TARGET) == report['target_sha256']
    assembly = json.loads((WORK / 'viewer-assembly.json').read_text())
    occurrence_names = {o['id']: o['name'] for o in assembly['occurrences']}
    parts = report['parts']
    refs = ['o1.' + '.'.join(str(int(n)+1) for n in re.findall(r'\[(\d+)\]', p['path'])) for p in parts]
    assert len(set(refs)) == len(parts)
    assert all(occurrence_names[ref] == part['label'] for ref, part in zip(refs, parts))
    operating = [r for r in report['operating']['checks'] if r['confirmed_collision']]
    home = {s: operating[0]['pose_deg'].get(s, 0) if s in SID[:10] else 0 for s in SID}
    selected = [
        ('head_body', '1 头壳与身体上壳', next(r for r in operating if 'head' in r['labels'])),
        ('antenna_top', '2 右天线夹持件与身体上壳', next(r for r in operating if set(r['labels']) == {'right_antenna_holder', 'body_middle_top'})),
        ('antenna_back', '3 右天线夹持件与身体后壳', next(r for r in operating if set(r['labels']) == {'right_antenna_holder', 'body_back'})),
        ('feet', '4 双脚极限组合互碰', next(r for r in report['combined']['checks'] if r['labels'] == ['foot_top', 'foot_top'])),
    ]
    rows = _leaves(read_step(TARGET))
    shapes = [bd.Compound(r['shape'].solids()) for r in rows]
    scenarios = []
    for clip, label, witness in selected:
        end = {s: witness['pose_deg'].get(s, 0) for s in SID}
        samples = []
        def evaluate(u):
            pose = {s: home[s]+u*(end[s]-home[s]) for s in SID}
            pair = []
            for i in witness['parts']:
                t = matrix(pose, parts[i]['chain'], report['axes'])
                plane = bd.Plane(origin=tuple(t[:3, 3]), x_dir=tuple(t[:3, 0]), z_dir=tuple(t[:3, 2]))
                pair.append(shapes[i].moved(bd.Location(plane)))
            gap = pair[0].distance_to(pair[1])
            volume = _overlap(*pair) if gap < 1e-5 else 0.0
            sample = dict(progress=float(u), gap_mm=gap, overlap_mm3=volume, pose_deg=pose)
            samples.append(sample)
            print('animation path', clip, round(u, 6), 'gap', round(gap, 6), 'volume', round(volume, 6), flush=True)
            return sample
        threshold = 1e-4
        first = evaluate(0)
        assert first['overlap_mm3'] <= threshold, f'{clip}: starting pair already intersects'
        low, high = 0., None
        for u in np.linspace(.05, 1, 20):
            candidate = evaluate(float(u))
            if candidate['overlap_mm3'] > threshold:
                high = float(u)
                break
            low = float(u)
        assert high is not None, f'{clip}: collision witness did not replay'
        for _ in range(10):
            middle = (low+high)/2
            if evaluate(middle)['overlap_mm3'] > threshold:
                high = middle
            else:
                low = middle
        contact = evaluate(high)
        final = evaluate(1)
        assert final['overlap_mm3'] > threshold
        scenarios.append(dict(id=clip, label=label, start_pose_deg=home, end_pose_deg=end,
            pair_refs=[refs[i] for i in witness['parts']], pair_labels=witness['labels'],
            contact_progress=high, contact_bracket_progress=[low, high],
            first_detected_collision=contact, endpoint=final, samples=samples,
            volume_threshold_mm3=threshold))
        gc.collect()
    data = dict(target_sha256=report['target_sha256'], axes=report['axes'],
        joint_groups=[dict(id=s, refs=[ref for ref, part in zip(refs, parts) if s in part['chain']]) for s in SID],
        physical_refs=refs, reference_refs=sorted(set(occurrence_names)-set(refs)), scenarios=scenarios)
    # Only presentation data belongs in the render module; measurements stay
    # in the review JSON. The STEP bytes and all physical parts remain intact.
    presentation = {**data, 'scenarios': [{k:v for k,v in s.items() if k not in ('samples', 'first_detected_collision', 'endpoint')} for s in scenarios]}
    module = TARGET.with_name(TARGET.name+'.js')
    module.write_text('// Current STEP collision replay. Angles and pivots match FINAL_ASSEMBLY_REVIEW.json.\n'
        '// Kinematic demonstration; no physical servo commands or contact dynamics.\n'
        'const data = '+json.dumps(presentation, ensure_ascii=False, separators=(',', ':'))+';\n'+r'''
function progressAt(t, contact) {
  const stops = [[0,0],[1,0],[6,contact],[7.5,contact],[9,1],[10.5,1],[12,0]];
  const time = Math.max(0, t % 12);
  for (let i=1; i<stops.length; ++i) {
    if (time <= stops[i][0]) {
      const [a,x] = stops[i-1], [b,y] = stops[i];
      const u = (time-a)/(b-a), smooth = u*u*(3-2*u);
      return x+(y-x)*smooth;
    }
  }
  return 0;
}
function replay(t, m, index) {
  const scenario = data.scenarios[index], u = progressAt(t, scenario.contact_progress);
  for (const ref of data.reference_refs) m.get('#'+ref).visible(false);
  m.get('#'+data.physical_refs.join(',')).opacity(.18);
  // Handles premultiply. Apply child rotations first, then parent rotations,
  // yielding R_parent * R_child, exactly as the CAD collision checker does.
  for (const joint of [...data.joint_groups].reverse()) {
    const start = scenario.start_pose_deg[joint.id], end = scenario.end_pose_deg[joint.id];
    const angle = start+u*(end-start);
    if (Math.abs(angle)>1e-10 && joint.refs.length) {
      const [origin, axis] = data.axes[joint.id];
      m.get('#'+joint.refs.join(',')).rotate(axis, angle, origin);
    }
  }
  for (const ref of scenario.pair_refs) m.get('#'+ref).opacity(.95);
}
export const clips = {
  head_body: {label:data.scenarios[0].label, duration:12, loop:true, update(t,m){replay(t,m,0);}},
  antenna_top: {label:data.scenarios[1].label, duration:12, loop:true, update(t,m){replay(t,m,1);}},
  antenna_back: {label:data.scenarios[2].label, duration:12, loop:true, update(t,m){replay(t,m,2);}},
  feet: {label:data.scenarios[3].label, duration:12, loop:true, update(t,m){replay(t,m,3);}},
};
''')
    report['motion_demonstration'] = dict(completed=True, source_sha256=report['target_sha256'],
        module_file=str(module), module_sha256=digest(module), scenarios=scenarios,
        source='Current STEP geometry and measured CAD axes; same forward kinematics as confirmed witnesses.',
        contact_scope='First detected positive-volume intersection of the selected pair along 0.05-progress samples, followed by local bisection; not a continuous or all-pair clearance certificate.',
        clip_timeline_seconds={'start_hold':[0,1], 'approach':[1,6], 'contact_hold':[6,7.5], 'penetration':[7.5,9], 'endpoint_hold':[9,10.5], 'return':[10.5,12]},
        geometry_changed=False, physical_hardware_command_sent=False)
    save(report)
    print('saved', module, flush=True)


def animation_check():
    report = json.loads(REPORT.read_text())
    demo = report['motion_demonstration']
    expected = []
    for scenario in demo['scenarios']:
        contact = scenario['contact_progress']
        for time, u in [(0,0), (3.5,contact/2), (6,contact), (7.5,contact),
                        (9,1), (10.5,1), (11.25,.5), (12,0), (13,0)]:
            pose = {s: scenario['start_pose_deg'][s]+u*(scenario['end_pose_deg'][s]-scenario['start_pose_deg'][s]) for s in SID}
            transforms = {}
            for part in report['parts']:
                ref = 'o1.'+'.'.join(str(int(n)+1) for n in re.findall(r'\[(\d+)\]', part['path']))
                transforms[ref] = matrix(pose, part['chain'], report['axes']).ravel().tolist()
            expected.append(dict(clip=scenario['id'], time=time, matrices=transforms))
    path = WORK/'animation-expected.json'
    path.write_text(json.dumps(expected))
    result = subprocess.run(['node', '--max-old-space-size=128',
        str(Path(__file__).with_name('verify_collision_animation.mjs')), str(demo['module_file']),
        str(WORK/'viewer-assembly.json'), str(path)], check=True, capture_output=True, text=True)
    print(result.stdout, flush=True)
    demo['transform_validation'] = json.loads(result.stdout)
    demo['module_sha256'] = digest(Path(demo['module_file']))
    save(report)


if __name__ == '__main__':
    dict(static=static, combined=combined, operating=combined, antenna=combined, confirm=retry_confirmation, self_check=self_check,
         interfaces=interfaces, simulation=simulation, visual=visual, animation=animation,
         animation_check=animation_check)[sys.argv[1]]()
