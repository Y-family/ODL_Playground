"""Measure the head shell wall thickness at the S11 rear-bottom corner.

Walks inward from sample points on the head's outer surface and reports how
much material is crossed, using the OCC solid classifier (no rtree needed).
"""

import gc
from pathlib import Path

import numpy as np
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_State
from cadgen import build123d as bd, read_step

from full_motion_current import _leaves

SOURCE = Path('/tmp/backup_upper_refit_d43675e2.step')
HEAD = '/head_assembly <1>[21]/head[4]'
BOX = {'x': (-68.0, -60.5), 'y': (-36.0, 36.0), 'z': (170.5, 176.5)}
STEPS = np.arange(0.0, 6.0, 0.1)


def classify(shape, point):
    classifier = BRepClass3d_SolidClassifier(shape.wrapped, gp_Pnt(*point), 1e-6)
    return classifier.State() == TopAbs_State.TopAbs_IN


def main():
    robot = read_step(SOURCE)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    head = shapes[HEAD]

    vertices, faces = head.tessellate(0.4, 0.4)
    points = np.array([[q.X, q.Y, q.Z] for q in vertices])
    tris = points[np.array(faces)]
    centres = tris.mean(axis=1)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / np.where(lengths[:, None] == 0, 1.0, lengths[:, None])

    inside = ((centres[:, 0] > BOX['x'][0]) & (centres[:, 0] < BOX['x'][1])
              & (np.abs(centres[:, 1]) < BOX['y'][1])
              & (centres[:, 2] > BOX['z'][0]) & (centres[:, 2] < BOX['z'][1]))
    indices = np.nonzero(inside)[0]
    print('faces in the corner box:', len(indices))

    results = []
    for index in indices[:: max(1, len(indices) // 25)]:
        centre = centres[index]
        normal = normals[index]
        for sign in (1.0, -1.0):
            direction = normal * sign
            states = [classify(head, centre + direction * depth) for depth in STEPS]
            if not any(states):
                continue
            first = int(np.argmax(states))
            after = states[first:]
            if all(after):
                results.append((6.0, tuple(np.round(centre, 2))))
                continue
            run = 0
            for value in after:
                if value:
                    run += 1
                else:
                    break
            results.append((float(STEPS[first] + run * 0.1), tuple(np.round(centre, 2))))
    results.sort()
    print('wall thickness at the head rear-bottom corner (mm):')
    for value, point in results[:10]:
        print(f'   {value:5.2f}  at {point}')
    print('min =', round(results[0][0], 2) if results else None, ' samples =', len(results))


if __name__ == '__main__':
    main()
