"""Visualise the S11 head/body_middle_top contact and the material that a
local rear-edge relief would remove.  Read-only: writes PNG + JSON only."""

import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from cadgen import build123d as bd

import refit_upper_servo_interfaces_local as refit
from full_motion_current import TARGET, _leaves, read_step

HEAD = '/head_assembly <1>[21]/head[4]'
HEAD_BOT = '/head_assembly <1>[21]/head_bot_sheet[11]'
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
AXIS = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))
TARGETS = [39, 45, 60]
MARGIN_DEG = 2.0
OUT_PNG = Path('/tmp/review/s11_body_relief.png')
OUT_JSON = Path('/tmp/s11_body_relief.json')


def as_compound(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def tri(shape):
    if shape.volume <= 0:
        return np.empty((0, 3, 3))
    vertices, faces = shape.tessellate(0.4, 0.3)
    points = np.array([[q.X, q.Y, q.Z] for q in vertices])
    return points[np.array(faces)]


def polys2d(tris, i, j):
    return [t[:, [i, j]] for t in tris]


def main():
    robot = read_step(TARGET)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()

    body = shapes[BODY]
    record = {'body_middle_top_bbox': [round(v, 2) for v in
                                       (body.bounding_box().min.X, body.bounding_box().min.Y, body.bounding_box().min.Z,
                                        body.bounding_box().max.X, body.bounding_box().max.Y, body.bounding_box().max.Z)],
              'targets': {}}

    reliefs = {}
    for target in TARGETS:
        result = body
        removed_volume = 0.0
        removed_shapes = []
        angles = range(-20, target + int(MARGIN_DEG) + 1)
        for angle in angles:
            for mover_path in (HEAD, HEAD_BOT):
                posed = shapes[mover_path].rotate(AXIS, angle)
                tool = as_compound(result.intersect(posed))
                del posed
                if tool.volume > 0.05:
                    removed_volume += tool.volume
                    removed_shapes.append(tool)
                    result = refit._cut(result, [tool])
                gc.collect()
        kept = result
        patch = as_compound(removed_shapes) if removed_shapes else bd.Compound([])
        box = patch.bounding_box() if patch.volume > 0 else None
        reliefs[target] = (patch, kept)
        record['targets'][str(target)] = {
            'cut_angles_deg': [-20, target + int(MARGIN_DEG)],
            'removed_mm3': round(removed_volume, 2),
            'patch_bbox': None if box is None else [round(v, 2) for v in
                                                    (box.min.X, box.min.Y, box.min.Z, box.max.X, box.max.Y, box.max.Z)],
        }
        print(f'target +{target}° (+{int(MARGIN_DEG)}° margin): removed {removed_volume:.1f} mm³'
              f' bbox {record["targets"][str(target)]["patch_bbox"]}', flush=True)

    fig = plt.figure(figsize=(16, 10))
    for column, target in enumerate(TARGETS):
        angle = target
        patch, kept = reliefs[target]
        head = shapes[HEAD].rotate(AXIS, angle)
        contact = as_compound(body.intersect(head))

        ax = fig.add_subplot(2, 3, column + 1)
        ax.add_collection(PolyCollection(polys2d(tri(kept), 0, 2), facecolors='#cfcfcf', edgecolors='none', alpha=0.9))
        ax.add_collection(PolyCollection(polys2d(tri(head), 0, 2), facecolors='#7fb3ff', edgecolors='none', alpha=0.45))
        ax.add_collection(PolyCollection(polys2d(tri(patch), 0, 2), facecolors='#ff8a00', edgecolors='none', alpha=0.95))
        ax.add_collection(PolyCollection(polys2d(tri(contact), 0, 2), facecolors='#ff2d00', edgecolors='none', alpha=0.95))
        ax.set_title(f'side view  head at +{angle}°\nred = contact {contact.volume:.1f} mm³, '
                     f'orange = relief cut {record["targets"][str(target)]["removed_mm3"]:.1f} mm³')
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Z [mm]'); ax.set_aspect('equal')
        ax.autoscale_view()

        axz = fig.add_subplot(2, 3, column + 4)
        axz.add_collection(PolyCollection(polys2d(tri(kept), 0, 2), facecolors='#cfcfcf', edgecolors='none', alpha=0.55))
        axz.add_collection(PolyCollection(polys2d(tri(patch), 0, 2), facecolors='#ff8a00', edgecolors='none', alpha=0.95))
        axz.add_collection(PolyCollection(polys2d(tri(contact), 0, 2), facecolors='#ff2d00', edgecolors='none', alpha=0.95))
        axz.set_xlim(-110, -88); axz.set_ylim(92, 112)
        axz.set_title(f'zoom: rear top edge  (+{target}° target)')
        axz.set_xlabel('X [mm]'); axz.set_ylabel('Z [mm]'); axz.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=105)
    OUT_JSON.write_text(json.dumps(record, indent=2) + '\n')
    print('saved', OUT_PNG, 'and', OUT_JSON)


if __name__ == '__main__':
    main()
