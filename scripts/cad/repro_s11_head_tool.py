"""Diagnose the head-side (option B) tool: sequential slivers vs one fused tool."""

import gc
import math
from pathlib import Path

import trimesh
from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape

import refit_upper_servo_interfaces_local as refit
from full_motion_current import _leaves, _mesh, _depths, is_moving, measured_axes

SOURCE = Path('/tmp/backup_upper_refit_d43675e2.step')
SERVO = 'S11'
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
HEAD = '/head_assembly <1>[21]/head[4]'
FINE_FROM, FINE_STEP, COARSE_STEP = 36.0, 0.25, 1.0


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


def report(tag, shape):
    check = check_occurrence_shape(shape.wrapped)
    print(f'  {tag}: volume={shape.volume:.2f} solids={check["solidCount"]} reasons={check["reasons"]}', flush=True)
    return check


def main():
    robot = read_step(SOURCE)
    leaves = _leaves(robot)
    by_path = {item['path']: item['shape'] for item in leaves}
    origin, direction = measured_axes(robot)[SERVO]
    axis = bd.Axis(tuple(origin), tuple(direction))
    angles = axis_grid()
    body = detached(by_path[BODY])
    head = detached(by_path[HEAD])
    body_mesh = _mesh(by_path[BODY])
    manager = trimesh.collision.CollisionManager()
    manager.add_object('body', body_mesh)
    mesh = _mesh(by_path[HEAD])
    del robot, leaves
    gc.collect()

    touching = []
    for angle in angles:
        posed = mesh.copy()
        posed.apply_transform(trimesh.transformations.rotation_matrix(
            math.radians(angle), tuple(direction), tuple(origin)))
        if _depths(manager, posed):
            touching.append(angle)
        del posed
    del mesh
    gc.collect()
    print('touching angles (mesh):', touching, flush=True)

    tools = []
    for angle in touching:
        posed = head.rotate(axis, angle)
        piece = as_compound(posed.intersect(body))
        del posed
        gc.collect()
        if piece.volume <= 0.02:
            continue
        neutral = piece.rotate(axis, -angle)
        del piece
        print(f'  angle {angle}: tool {neutral.volume:.3f} mm3 solids={len(list(neutral.solids()))}', flush=True)
        tools.append(neutral)
        gc.collect()

    print('variants:')
    sequential = head
    for tool in tools:
        sequential = refit._cut(sequential, [tool])
    report('sequential slivers', sequential.clean().fix())

    fused = refit._fuse(*tools) if len(tools) > 1 else tools[0]
    report('fused tool', fused)
    single = refit._cut(head, [fused])
    report('single fused cut', single.clean().fix())


if __name__ == '__main__':
    main()
