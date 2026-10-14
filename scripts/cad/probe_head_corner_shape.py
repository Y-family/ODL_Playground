"""Cross-section of the head rear-bottom corner + the relief tool it needs."""

import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from cadgen import build123d as bd, read_step

from full_motion_current import _leaves, is_moving

SOURCE = Path('/tmp/backup_upper_refit_d43675e2.step')
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
HEAD = '/head_assembly <1>[21]/head[4]'
AXIS = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))
OUT = Path('/tmp/review/s11_head_corner_section.png')


def as_compound(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def tri(shape):
    if shape.volume <= 0:
        return np.empty((0, 3, 3))
    vertices, faces = shape.tessellate(0.25, 0.3)
    points = np.array([[q.X, q.Y, q.Z] for q in vertices])
    return points[np.array(faces)]


def main():
    robot = read_step(SOURCE)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    head, body = shapes[HEAD], shapes[BODY]

    tools = []
    for angle in (38.75, 39.0, 39.25):
        posed = head.rotate(AXIS, angle)
        piece = as_compound(posed.intersect(body))
        del posed
        if piece.volume > 0.02:
            neutral = piece.rotate(AXIS, -angle)
            tools.append(neutral)
        gc.collect()
    tool = tools[-1]

    slab = bd.Box(400, 0.6, 400).located(bd.Location((0, 0, 200)))
    head_section = as_compound(head.intersect(slab))
    tool_section = as_compound(tool.intersect(slab))
    print('tool volume', round(tool.volume, 2), 'section volume', round(tool_section.volume, 3))

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    ax = axs[0]
    ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(head_section)],
                                     facecolors='#7fb3ff', edgecolors='none', alpha=0.9))
    ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(tool_section)],
                                     facecolors='#ff2a00', edgecolors='none', alpha=0.9))
    ax.set_xlim(-75, -45); ax.set_ylim(160, 200)
    ax.set_title('head rear-bottom corner, section at Y=0 (blue) + relief tool (red)')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Z [mm]'); ax.set_aspect('equal')

    ax = axs[1]
    ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(head)],
                                     facecolors='#7fb3ff', edgecolors='none', alpha=0.35))
    ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(tool)],
                                     facecolors='#ff2a00', edgecolors='none', alpha=0.95))
    ax.set_xlim(-75, -45); ax.set_ylim(160, 200)
    ax.set_title('same zone, full side view')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Z [mm]'); ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUT, dpi=110)
    print('saved', OUT)


if __name__ == '__main__':
    main()
