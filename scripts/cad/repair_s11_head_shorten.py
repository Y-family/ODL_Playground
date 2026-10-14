"""S11 +39 deg limit: shorten the head rear-bottom corner instead of shaving it.

The swept-envelope shave of the head is not usable: the corner is a tapered
thin wall, the tool runs tangential to the shell's own surface and every
variant (sequential slivers, one fused tool, bloated tool, fuzzy boolean)
comes back self-intersecting.

Instead the corner is shortened with a plane. The exact relief tool is a subset
of the half-space {X <= tool_max_X}, so cutting that half-space removes at least
as much material as the exact tool - the clearance at the limit can only be
better - while the cut face is a plain plane through the wall, which is robust
and keeps the remaining wall thickness.

User decision 2026-10-08: "不够厚就别削了，直接把它搞短".
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
OUTPUT = Path('/tmp/open_duck_s11_head_shorten_candidate.step')
REPORT = Path('/tmp/s11_head_shorten_report.json')
SOURCE_SHA256 = 'd43675e29c6c2a280ed6fb15b33c87147a4b65891a26f874f9613f85e341a05f'

SERVO = 'S11'
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
FINE_FROM, FINE_STEP, COARSE_STEP = 36.0, 0.25, 1.0
MIN_TOOL_MM3 = 0.02
PLANE_MARGIN_MM = 0.0


def axis_grid():
    values, angle = [], -20.0
    while angle < FINE_FROM:
        values.append(round(angle, 3)); angle += COARSE_STEP
    angle = FINE_FROM
    while angle <= 39.0 + FINE_STEP + 1e-9:
        values.append(round(angle, 3)); angle += FINE_STEP
    return values


def as_compound(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def detached(shape):
    node = bd.Compound(shape.solids())
    node.label, node.color = shape.label, shape.color
    return node


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
    origin, direction = measured_axes(robot)[SERVO]
    axis = bd.Axis(tuple(origin), tuple(direction))
    angles = axis_grid()
    body = detached(by_path[BODY])
    body_mesh = _mesh(by_path[BODY])
    manager = trimesh.collision.CollisionManager()
    manager.add_object('body', body_mesh)

    movers = [item for item in leaves if is_moving(item, SERVO)]
    tools = {}
    reference = {}
    for item in movers:
        mesh = _mesh(item['shape'])
        reference[item['path']] = mesh
    planes = {}
    for item in movers:
        mesh = reference[item['path']]
        facing = None
        for angle in angles:
            posed = mesh.copy()
            posed.apply_transform(trimesh.transformations.rotation_matrix(
                math.radians(angle), tuple(direction), tuple(origin)))
            if _depths(manager, posed):
                shape = detached(by_path[item['path']])
                piece = as_compound(shape.rotate(axis, angle).intersect(body))
                del shape
                if piece.volume > MIN_TOOL_MM3:
                    neutral = piece.rotate(axis, -angle)
                    box = neutral.bounding_box()
                    facing = box.max.X if facing is None else max(facing, box.max.X)
                del piece
            del posed
            gc.collect()
        if facing is not None:
            planes[item['path']] = facing
    print('required shortening plane (max X of each part\'s swept tool):',
          {k.rsplit('/', 1)[-1]: round(v, 3) for k, v in planes.items()}, flush=True)
    assert planes, 'no moving part reaches the body within the limit'
    cut_x = max(planes.values()) + PLANE_MARGIN_MM
    print('shortening plane X =', round(cut_x, 3), flush=True)

    replacements = {}
    summary = {}
    for path, x0 in sorted(planes.items()):
        head = detached(by_path[path])
        blade = bd.Box(200, 400, 400).located(bd.Location((x0 - 100, 0, 200)))
        started = time.time()
        result = refit._cut(head, [blade])
        result = result.clean().fix()
        solids = list(result.solids())
        assert len(solids) == 1, f'{path} split into {len(solids)} solids'
        check = check_occurrence_shape(result.wrapped)
        assert check['solidCount'] == 1 and not check['reasons'], (path, check)
        before_cyl = cylinder_signature(head)
        after_cyl = cylinder_signature(result)
        missing = [row for row in before_cyl if row not in set(after_cyl)]
        assert not missing, f'{path}: cylindrical features lost: {missing[:4]}'
        removed = head.volume - result.volume
        box = as_compound(head.intersect(result))
        summary[path] = {'removed_mm3': round(removed, 3), 'plane_x': round(x0, 3),
                         'fraction': round(removed / head.volume, 6)}
        print(f'{path.rsplit("/", 1)[-1]:<22} {head.volume:9.1f} -> {result.volume:9.1f} '
              f'(removed {removed:7.2f} mm3, {100 * removed / head.volume:.3f}%), {time.time() - started:.1f}s',
              flush=True)
        result.label, result.color = head.label, head.color
        replacements[path] = result
        del head, box
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
        assert check['solidCount'] == 1 and not check['reasons'], (path, check)
    print('exported leaves', len(exported), 'changed', changed, flush=True)

    report = {'source_sha256': SOURCE_SHA256, 'output': str(OUTPUT),
              'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              'servo': SERVO, 'method': 'plane shortening of the head rear-bottom corner',
              'angle_grid_deg': [angles[0], angles[-1], len(angles)],
              'shortening_plane_x': round(cut_x, 3), 'modified': summary, 'leaves': len(exported)}
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print('candidate', OUTPUT, report['output_sha256'], flush=True)


if __name__ == '__main__':
    main()
