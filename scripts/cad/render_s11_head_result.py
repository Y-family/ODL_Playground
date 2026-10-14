"""Before/after render of the S11 head-corner relief (final geometry)."""

import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from cadgen import build123d as bd, read_step

from full_motion_current import _leaves

BEFORE = Path('/tmp/backup_upper_refit_d43675e2.step')
AFTER = Path('/root/work/Open_Duck_Playground/cad/local_fix/open_duck_mini_upper_refit.step')
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
HEAD = '/head_assembly <1>[21]/head[4]'
HEAD_BOT = '/head_assembly <1>[21]/head_bot_sheet[11]'
AXIS = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))
ANGLE = 39.0
OUT = Path('/tmp/review/s11_head_relief_before_after.png')


def compounds(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def parts(path):
    robot = read_step(path)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    return {key: shapes[key] for key in (BODY, HEAD, HEAD_BOT)}


def tri(shape, chord=0.6):
    if shape.volume <= 0:
        return np.empty((0, 3, 3))
    vertices, faces = shape.tessellate(chord, 0.4)
    points = np.array([[q.X, q.Y, q.Z] for q in vertices])
    return points[np.array(faces)]


def panel(ax, part, heads, title, zoom):
    ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(part)],
                                     facecolors='#cfcfcf', edgecolors='none', alpha=0.95))
    total = 0.0
    for head in heads:
        pieces = tri(head)
        ax.add_collection(PolyCollection([t[:, [0, 2]] for t in pieces],
                                         facecolors='#7fb3ff', edgecolors='none', alpha=0.5))
        contact = compounds(part.intersect(head))
        total += contact.volume
        ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(contact)],
                                         facecolors='#ff2d00', edgecolors='none', alpha=1.0))
    ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    ax.set_title(f'{title}\nred = interference at +{ANGLE:g}° = {total:.1f} mm³')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Z [mm]'); ax.set_aspect('equal')


def main():
    before = parts(BEFORE)
    after = parts(AFTER)

    heads_before = [before[HEAD].rotate(AXIS, ANGLE), before[HEAD_BOT].rotate(AXIS, ANGLE)]
    heads_after = [after[HEAD].rotate(AXIS, ANGLE), after[HEAD_BOT].rotate(AXIS, ANGLE)]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    panel(axs[0][0], before[BODY], heads_before, 'BEFORE (d43675e2)  whole rear', ((-115, -75), (90, 140)))
    panel(axs[1][0], after[BODY], heads_after, 'AFTER (0398653c)  whole rear', ((-115, -75), (90, 140)))
    panel(axs[0][1], before[BODY], heads_before, 'BEFORE  corner close-up', ((-104, -92), (100, 116)))
    panel(axs[1][1], after[BODY], heads_after, 'AFTER  shortened corner', ((-104, -92), (100, 116)))
    plt.tight_layout()
    plt.savefig(OUT, dpi=105)
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
