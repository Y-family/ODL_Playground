"""S11 +39 deg limit, option B: relieve the HEAD rear-bottom corner.

Same requirement as the body-side relief, but the material is taken from the
moving head parts instead of the fixed body shell:

    union over angle a of ( head_part.rotate(+a) intersect body_middle_top )

brought back to the neutral frame with rotate(-a), then cut from the head part.
The angle grid and the clearance argument are the same as the body-side run
(-20..36 deg by 1 deg, 36..39.25 deg by 0.25 deg ~ 0.5 mm running clearance).

Only head[4] and head_bot_sheet[11] change; no new parts, no servo moved.
"""

import gc
import hashlib
import json
import math
import time
from pathlib import Path

import trimesh
from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape
from OCP.BRepAlgoAPI import BRepAlgoAPI_Check

import refit_upper_servo_interfaces_local as refit
from full_motion_current import (TARGET, _leaves, _mesh, _depths, is_moving,
                                 measured_axes)

SOURCE = TARGET
OUTPUT = Path('/tmp/open_duck_s11_head_relief_candidate.step')
REPORT = Path('/tmp/s11_head_relief_report.json')
SOURCE_SHA256 = 'd43675e29c6c2a280ed6fb15b33c87147a4b65891a26f874f9613f85e341a05f'

SERVO = 'S11'
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
FINE_FROM = 36.0
FINE_STEP = 0.25
COARSE_STEP = 1.0
MIN_TOOL_MM3 = 0.02


def axis_grid():
    values = []
    angle = -20.0
    while angle < FINE_FROM:
        values.append(round(angle, 3))
        angle += COARSE_STEP
    angle = FINE_FROM
    while angle <= 39.0 + FINE_STEP + 1e-9:
        values.append(round(angle, 3))
        angle += FINE_STEP
    return values


def as_compound(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def detached(shape):
    node = bd.Compound(shape.solids())
    node.label = shape.label
    node.color = shape.color
    return node


def bop_faults(shape):
    """Fault signature from the BOP self-interference check.

    head[4] already carries a BOPAlgo_SelfIntersect flag in the untouched
    source model (internal tangent faces), so the cut is judged by whether it
    adds faults, not by the bare flag.
    """
    checker = BRepAlgoAPI_Check(shape.wrapped, True, True)
    checker.Perform()
    if checker.IsValid():
        return []
    return sorted(str(item.GetCheckStatus()).rsplit('.', 1)[-1] for item in checker.Result())


def cylinder_signature(shape):
    result = []
    for face in shape.faces():
        if str(getattr(face, 'geom_type', '')) != 'GeomType.CYLINDER':
            continue
        centre = face.center()
        result.append((round(face.area, 4), round(centre.X, 3), round(centre.Y, 3), round(centre.Z, 3)))
    return sorted(result)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256, 'source is not the pre-change model'
    robot = read_step(SOURCE)
    leaves = _leaves(robot)
    by_path = {item['path']: item['shape'] for item in leaves}
    axes = measured_axes(robot)
    origin, direction = axes[SERVO]
    axis = bd.Axis(tuple(origin), tuple(direction))
    angles = axis_grid()
    print('axis', origin, direction, 'grid', angles[0], angles[-1], len(angles), flush=True)

    body = detached(by_path[BODY])
    body_mesh = _mesh(by_path[BODY])
    movers = [item for item in leaves if is_moving(item, SERVO)]
    manager = trimesh.collision.CollisionManager()
    manager.add_object('body', body_mesh)
    touches = {}
    for item in movers:
        mesh = _mesh(item['shape'])
        for angle in angles:
            posed = mesh.copy()
            posed.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(angle), tuple(direction), tuple(origin)))
            if _depths(manager, posed):
                touches.setdefault(item['path'], []).append(angle)
            del posed
        del mesh
        gc.collect()
    print('mesh broad phase:', {k: [min(v), max(v)] for k, v in touches.items()}, flush=True)

    replacements = {}
    summary = {}
    for path in sorted(touches):
        head = detached(by_path[path])
        started = time.time()
        result = head
        cuts = 0
        for angle in angles:
            posed = head.rotate(axis, angle)
            tool = as_compound(posed.intersect(body))
            del posed
            gc.collect()
            if tool.volume <= MIN_TOOL_MM3:
                continue
            neutral_tool = tool.rotate(axis, -angle)
            del tool
            result = refit._cut(result, [neutral_tool])
            cuts += 1
            del neutral_tool
            gc.collect()
        result = result.clean().fix()
        solids = list(result.solids())
        assert len(solids) == 1, f'{path} split into {len(solids)} solids'
        check = check_occurrence_shape(result.wrapped)
        assert check['solidCount'] == 1, (path, check)
        assert 'invalidTopology' not in check['reasons'], (path, check)
        faults_before = bop_faults(head)
        faults_after = bop_faults(result)
        assert set(faults_after) <= set(faults_before), (path, faults_before, faults_after)
        assert len(faults_after) <= len(faults_before), (path, faults_before, faults_after)
        print(f'   BOP faults before={faults_before} after={faults_after}', flush=True)
        before_cyl = cylinder_signature(head)
        after_cyl = cylinder_signature(result)
        missing = [row for row in before_cyl if row not in set(after_cyl)]
        assert not missing, f'{path}: cylindrical features lost: {missing[:4]}'
        removed = head.volume - result.volume
        summary[path] = {'removed_mm3': round(removed, 3), 'cuts': cuts,
                         'fraction': round(removed / head.volume, 6)}
        print(f'{path.split("/")[-1]:<22} {head.volume:9.1f} -> {result.volume:9.1f} '
              f'(removed {removed:7.2f} mm3, {100 * removed / head.volume:.3f}%), cuts={cuts}, '
              f'{time.time() - started:.1f}s', flush=True)
        replacements[path] = result
        del head
        gc.collect()

    def replace(node, path):
        children = list(getattr(node, 'children', None) or [])
        if path in replacements:
            return replacements[path]
        if not children:
            return node
        fresh = bd.Compound(children=[replace(child, path + f'/{child.label}[{index}]')
                                      for index, child in enumerate(children)], label=node.label)
        fresh.location, fresh.color = node.location, node.color
        return fresh

    top = list(getattr(robot, 'children', None) or [])
    rebuilt = bd.Compound(children=[replace(child, f'/{child.label}[{index}]')
                                    for index, child in enumerate(top)], label=robot.label)
    rebuilt.location, rebuilt.color = robot.location, robot.color
    rebuilt = refit._encode_colours(rebuilt)
    bd.export_step(rebuilt, OUTPUT)

    exported = {item['path']: item['shape'] for item in _leaves(read_step(OUTPUT))}
    assert set(exported) == set(by_path), 'leaf set changed'
    base = {item['path']: bd.Compound(item['shape'].solids()) for item in leaves}
    changed = [path for path, shape in exported.items()
               if abs(bd.Compound(shape.solids()).volume - base[path].volume) > 1e-6]
    assert changed == sorted(replacements), f'unexpected changed leaves: {changed}'
    for path in replacements:
        check = check_occurrence_shape(exported[path].wrapped)
        assert check['solidCount'] == 1, (path, check)
        assert 'invalidTopology' not in check['reasons'], (path, check)
        assert 'openShell' not in check['reasons'], (path, check)
        assert set(bop_faults(exported[path])) <= set(bop_faults(base[path])), (path, check)
    print('exported leaves', len(exported), 'changed', changed, flush=True)

    report = {'source_sha256': SOURCE_SHA256, 'output': str(OUTPUT),
              'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              'servo': SERVO, 'option': 'B head rear-bottom corner',
              'angle_grid_deg': [angles[0], angles[-1], len(angles)],
              'modified': summary, 'leaves': len(exported)}
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('candidate', OUTPUT, report['output_sha256'], flush=True)


if __name__ == '__main__':
    main()
