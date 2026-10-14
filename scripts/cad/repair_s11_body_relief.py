"""Option A for the S11 neck_pitch +39 deg limit: relieve the rear top edge of
body_middle_top.

The +39 deg graze (119.0 mm^3) is the head's rear-bottom corner pressing on the
flat rear top face of body_middle_top, ~2 mm from the part's rear end.  User
decision 2026-10-08: cut that edge following the head corner (a transition
relief), expressed as

    union over angle a of ( body_middle_top intersect head.rotate(+a) )

applied as sequential single-tool cuts.  The angle grid is 1 deg from -20 deg
up to 36 deg plus 0.25 deg up to 39.25 deg, so the +39 deg software limit keeps
a real running clearance (~0.25 deg ~ 0.5 mm at r ~ 109 mm) instead of a zero
gap touch; a fine 36-39.25 deg window is used because a 2 deg margin here would
over-cut roughly 6x the required material.

Only body_middle_top changes; no new parts, no servo moved, holes untouched.
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

import refit_upper_servo_interfaces_local as refit
from full_motion_current import (TARGET, _leaves, _mesh, _depths, is_moving,
                                 measured_axes)

SOURCE = TARGET
OUTPUT = Path('/tmp/open_duck_s11_body_relief_candidate.step')
REPORT = Path('/tmp/s11_body_relief_report.json')
SOURCE_SHA256 = 'd43675e29c6c2a280ed6fb15b33c87147a4b65891a26f874f9613f85e341a05f'

SERVO = 'S11'
TARGET_PATH = '/body_assembly <1>[14]/body_middle_top[5]'
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


def cylinder_signature(shape):
    signature = []
    for face in shape.faces():
        if str(getattr(face, 'geom_type', '')) != 'GeomType.CYLINDER':
            continue
        centre = face.center()
        signature.append((round(face.area, 4), round(centre.X, 3), round(centre.Y, 3), round(centre.Z, 3)))
    return sorted(signature)


def mover_candidates(leaves, body_mesh, axis_direction, axis_position, angles):
    """Mesh broad phase: which S11 moving leaves can reach the target at all."""
    movers = [item for item in leaves if is_moving(item, SERVO)]
    manager = trimesh.collision.CollisionManager()
    manager.add_object('body', body_mesh)
    hits = {}
    for item in movers:
        mesh = _mesh(item['shape'])
        box = mesh.bounds
        for angle in angles:
            posed = mesh.copy()
            posed.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(angle), axis_direction, axis_position))
            if _depths(manager, posed):
                hits.setdefault(item['path'], []).append(angle)
            del posed
        del mesh
        gc.collect()
    return hits


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256, 'source changed'
    robot = read_step(SOURCE)
    leaves = _leaves(robot)
    by_path = {item['path']: item['shape'] for item in leaves}
    axes = measured_axes(robot)
    origin, direction = axes[SERVO]
    axis = bd.Axis(tuple(origin), tuple(direction))
    angles = axis_grid()
    print('axis', origin, direction, 'grid', angles[0], angles[-1], len(angles), flush=True)

    body = detached(by_path[TARGET_PATH])
    body_mesh = _mesh(by_path[TARGET_PATH])
    hits = mover_candidates(leaves, body_mesh, tuple(direction), tuple(origin), angles)
    print('mesh broad phase movers:', {k: [min(v), max(v), len(v)] for k, v in hits.items()}, flush=True)

    result = body
    cuts = 0
    started = time.time()
    for angle in angles:
        for path in hits:
            posed = detached(by_path[path]).rotate(axis, angle)
            tool = as_compound(result.intersect(posed))
            del posed
            if tool.volume > MIN_TOOL_MM3:
                result = refit._cut(result, [tool])
                cuts += 1
            gc.collect()
    result = result.clean().fix()
    solids = list(result.solids())
    assert len(solids) == 1, f'relieved part split into {len(solids)} solids'
    check = check_occurrence_shape(result.wrapped)
    assert check['solidCount'] == 1 and not check['reasons'], check
    result.label, result.color = body.label, body.color
    removed = body.volume - result.volume
    print(f'body_middle_top {body.volume:.1f} -> {result.volume:.1f} mm3 '
          f'(removed {removed:.2f} mm3, {100 * removed / body.volume:.3f}%), cuts={cuts}, '
          f'{time.time() - started:.1f}s', flush=True)

    before_cylinders = cylinder_signature(body)
    after_cylinders = cylinder_signature(result)
    missing = [row for row in before_cylinders if row not in set(after_cylinders)]
    assert not missing, f'cylindrical features lost: {missing[:4]}'

    def replace(node, path):
        children = list(getattr(node, 'children', None) or [])
        if path == TARGET_PATH:
            return result
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

    exported = read_step(OUTPUT)
    exported_leaves = {item['path']: item['shape'] for item in _leaves(exported)}
    assert set(exported_leaves) == set(by_path), 'leaf set changed'
    changed = []
    base = {item['path']: bd.Compound(item['shape'].solids()) for item in leaves}
    for path, shape in exported_leaves.items():
        fresh = bd.Compound(shape.solids())
        if abs(fresh.volume - base[path].volume) > 1e-6:
            changed.append(path)
    assert changed == [TARGET_PATH], f'unexpected changed leaves: {changed}'
    exported_check = check_occurrence_shape(exported_leaves[TARGET_PATH].wrapped)
    assert exported_check['solidCount'] == 1 and not exported_check['reasons'], exported_check
    exported_patch = as_compound(base[TARGET_PATH].intersect(exported_leaves[TARGET_PATH]))
    patch_box = exported_patch.bounding_box()
    print('exported leaves', len(exported_leaves), 'changed', changed, flush=True)

    REPORT.write_text(json.dumps({
        'source_sha256': SOURCE_SHA256,
        'output': str(OUTPUT),
        'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'servo': SERVO,
        'modified_path': TARGET_PATH,
        'angle_grid_deg': [angles[0], angles[-1], len(angles)],
        'movers_from_broad_phase': {k: [min(v), max(v)] for k, v in hits.items()},
        'removed_mm3': round(removed, 3),
        'removed_fraction': round(removed / body.volume, 6),
        'volume_before_mm3': round(body.volume, 3),
        'volume_after_mm3': round(result.volume, 3),
        'patch_bbox': [round(v, 3) for v in (patch_box.min.X, patch_box.min.Y, patch_box.min.Z,
                                             patch_box.max.X, patch_box.max.Y, patch_box.max.Z)],
        'cuts': cuts,
        'leaves': len(exported_leaves),
    }, indent=2) + '\n')
    print('candidate', OUTPUT, 'sha256', hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
