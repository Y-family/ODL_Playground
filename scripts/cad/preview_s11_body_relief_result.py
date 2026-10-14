"""Before/after render of the S11 body_middle_top rear-edge relief."""

import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from cadgen import build123d as bd, read_step

from full_motion_current import _leaves

import refit_upper_servo_interfaces_local as refit

BEFORE = Path('/tmp/backup_upper_refit_d43675e2.step')
AFTER = Path('/root/work/Open_Duck_Playground/cad/local_fix/open_duck_mini_upper_refit.step')
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
HEAD = '/head_assembly <1>[21]/head[4]'
HEAD_BOT = '/head_assembly <1>[21]/head_bot_sheet[11]'
AXIS = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))
ANGLE = 39.0
OUT = Path('/tmp/review/s11_head_relief_before_after.png')


def parts(path):
    robot = read_step(path)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    return {key: shapes[key] for key in (BODY, HEAD, HEAD_BOT)}


def compounds(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def tri(shape):
    if shape.volume <= 0:
        return np.empty((0, 3, 3))
    vertices, faces = shape.tessellate(0.35, 0.3)
    points = np.array([[q.X, q.Y, q.Z] for q in vertices])
    return points[np.array(faces)]


def side(ax, body, heads, patch, title):
    ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(body)], facecolors='#cfcfcf', edgecolors='none', alpha=0.9))
    for head in heads:
        ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(head)], facecolors='#7fb3ff', edgecolors='none', alpha=0.45))
    ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(patch)], facecolors='#ff8a00', edgecolors='none', alpha=0.95))
    total = 0.0
    for head in heads:
        contact = compounds(body.intersect(head))
        total += contact.volume
        ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(contact)], facecolors='#ff2d00', edgecolors='none', alpha=1.0))
    ax.set_xlim(-115, -85); ax.set_ylim(95, 135)
    ax.set_title(f'{title}\nred = interference at +{ANGLE:g}° = {total:.1f} mm³')
    ax.set_xlabel('X [mm]'); ax.set_ylabel('Z [mm]'); ax.set_aspect('equal')


VIEW = np.array([[0.62, -0.62, 0.0], [-0.35, -0.35, 0.87]])
DEPTH = np.array([0.5, 0.5, 0.7])


def iso(ax, body, patch, title):
    faces = []
    for tris, color in ((tri(body), '#cfcfcf'), (tri(patch), '#ff2a00')):
        for item in tris:
            faces.append((float(np.mean(item @ DEPTH)), np.array([item]) @ VIEW.T, color))
    faces.sort(key=lambda row: row[0])
    for _, poly, color in faces:
        ax.add_collection(PolyCollection([poly[0]], facecolors=color, edgecolors='none'))
    ax.autoscale_view(); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title)


def main():
    before = parts(BEFORE)
    after = parts(AFTER)
    patch = compounds([refit._cut(before[HEAD], [after[HEAD]]), refit._cut(before[HEAD_BOT], [after[HEAD_BOT]])])
    print('relief volume', round(patch.volume, 2),
          'bbox', [round(v, 2) for v in (patch.bounding_box().min.X, patch.bounding_box().min.Y,
                                         patch.bounding_box().min.Z, patch.bounding_box().max.X,
                                         patch.bounding_box().max.Y, patch.bounding_box().max.Z)])

    heads_before = [before[HEAD].rotate(AXIS, ANGLE), before[HEAD_BOT].rotate(AXIS, ANGLE)]
    heads_after = [after[HEAD].rotate(AXIS, ANGLE), after[HEAD_BOT].rotate(AXIS, ANGLE)]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    side(axs[0][0], before[BODY], heads_before, patch, 'BEFORE (d43675e2)  side view at the limit')
    side(axs[1][0], after[BODY], heads_after, patch, 'AFTER (0398653c)  side view at the limit')
    iso(axs[0][1], before[HEAD], patch, 'BEFORE  head rear-bottom corner\nred = material removed (226 mm³)')
    iso(axs[1][1], after[HEAD], patch, 'AFTER  shortened corner, same view')
    plt.tight_layout()
    plt.savefig(OUT, dpi=105)
    print('saved', OUT)


if __name__ == '__main__':
    main()
