"""Dense (1 deg) neck_pitch scan over -20..+45 deg: mesh broad phase plus
exact B-Rep confirmation of the flagged pairs.  Read-only."""

import gc
import json
import math
from pathlib import Path

import trimesh
from cadgen import build123d as bd, read_step

from full_motion_current import TARGET, _leaves, _mesh, _depths, is_moving, measured_axes

SERVO = 'S11'
ANGLE_MIN, ANGLE_MAX = -20, 90
OUT = Path('/tmp/s11_dense_scan_full.json')


def main():
    robot = read_step(TARGET)
    leaves = _leaves(robot)
    axes = measured_axes(robot)
    origin, direction = axes[SERVO]
    print('axis origin', origin, 'direction', direction)
    meshes = {item['path']: _mesh(item['shape']) for item in leaves}
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in leaves}
    del robot
    gc.collect()

    moving = [item for item in leaves if is_moving(item, SERVO)]
    moving_paths = {item['path'] for item in moving}
    fixed = [item for item in leaves if item['path'] not in moving_paths]
    by_path = {item['path']: item for item in leaves}
    print('moving', len(moving), 'fixed', len(fixed))

    manager = trimesh.collision.CollisionManager()
    for item in fixed:
        manager.add_object(item['path'], meshes[item['path']])

    pair_angles = {}
    for angle in range(ANGLE_MIN, ANGLE_MAX + 1):
        transform = trimesh.transformations.rotation_matrix(math.radians(angle), direction, origin)
        for item in moving:
            posed = meshes[item['path']].copy()
            posed.apply_transform(transform)
            for fixed_path, depth in _depths(manager, posed).items():
                if item['servo'] and item['servo'] == by_path[fixed_path]['servo']:
                    continue
                if depth > 0.05:
                    pair_angles.setdefault((item['path'], fixed_path), []).append(angle)
        del posed

    candidates = [
        {'moving': m, 'fixed': f, 'angles_deg': sorted(set(a))}
        for (m, f), a in sorted(pair_angles.items())
    ]
    print(f'broad candidate pairs: {len(candidates)}')
    for row in candidates:
        angles = row['angles_deg']
        print(' ', row['moving'].rsplit('/', 1)[-1], '<->', row['fixed'].rsplit('/', 1)[-1], angles)

    axis = bd.Axis(tuple(origin), tuple(direction))
    exact_rows = []
    for row in candidates:
        mover = shapes[row['moving']]
        fixed_shape = shapes[row['fixed']]
        first = min(row['angles_deg'])
        dense = sorted({first + d for d in range(-4, 8)}
                       | {a for a in row['angles_deg'] if a % 5 == 0})
        dense = [a for a in dense if ANGLE_MIN - 6 <= a <= ANGLE_MAX + 6]
        samples = []
        for angle in dense:
            posed = mover.rotate(axis, angle)
            inter = posed.intersect(fixed_shape)
            value = 0.0 if inter is None else sum(p.volume for p in inter.solids())
            del posed, inter
            gc.collect()
            samples.append({'angle_deg': angle, 'overlap_mm3': round(value, 3)})
        exact_rows.append({'moving': row['moving'], 'fixed': row['fixed'], 'exact': samples})
        hits = [s for s in samples if s['overlap_mm3'] > 0.05]
        print('  exact', row['moving'].rsplit('/', 1)[-1], '<->', row['fixed'].rsplit('/', 1)[-1],
              [(s['angle_deg'], s['overlap_mm3']) for s in hits], flush=True)

    OUT.write_text(json.dumps({'axis': {'origin': origin, 'direction': direction},
                               'range_deg': [ANGLE_MIN, ANGLE_MAX],
                               'candidates': candidates, 'exact': exact_rows}, indent=2) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
