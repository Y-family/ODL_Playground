"""Compare the two relief options for the S11 +39 deg contact with
body_middle_top.  Read-only: writes PNG + JSON only."""

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
LIMIT = 39.0
FINE_STEP = 0.25
FINE_FROM = 36.0
COARSE_STEP = 1.0
OUT_PNG = Path('/tmp/review/s11_relief_options.png')
OUT_PNG_ISO = Path('/tmp/review/s11_relief_iso.png')
OUT_JSON = Path('/tmp/s11_relief_options.json')


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


def angle_grid():
    values = []
    angle = -20.0
    while angle < FINE_FROM:
        values.append(round(angle, 3))
        angle += COARSE_STEP
    angle = FINE_FROM
    while angle <= LIMIT + FINE_STEP:
        values.append(round(angle, 3))
        angle += FINE_STEP
    return values


def cut_body(body, movers, angles):
    result = body
    patches = []
    for angle in angles:
        for shape in movers:
            posed = shape.rotate(AXIS, angle)
            tool = as_compound(result.intersect(posed))
            del posed
            if tool.volume > 0.02:
                patches.append(tool)
                result = refit._cut(result, [tool])
            gc.collect()
    return result, patches


def cut_head(head_parts, body, angles):
    results = {path: shape for path, shape in head_parts.items()}
    patches = {path: [] for path in head_parts}
    for angle in angles:
        for path, shape in head_parts.items():
            posed = shape.rotate(AXIS, angle)
            tool = as_compound(posed.intersect(body))
            del posed
            gc.collect()
            if tool.volume <= 0.02:
                continue
            tool_neutral = tool.rotate(AXIS, -angle)
            patches[path].append(tool_neutral)
            results[path] = refit._cut(results[path], [tool_neutral])
            del tool_neutral
            gc.collect()
    return results, patches


def bbox(shape):
    if shape is None or shape.volume <= 0:
        return None
    b = shape.bounding_box()
    return [round(v, 2) for v in (b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z)]


VIEW = np.array([[0.62, -0.62, 0.0], [-0.35, -0.35, 0.87]])
DEPTH = np.array([0.5, 0.5, 0.7])


def project(tris):
    return tris @ VIEW.T


def draw_iso(ax, layers):
    """layers: list of (tris, facecolor, alpha).  Painter's algorithm."""
    faces = []
    for tris, color, alpha in layers:
        for item in tris:
            depth = float(np.mean(item @ DEPTH))
            faces.append((depth, project(np.array([item]))[0], color, alpha))
    faces.sort(key=lambda row: row[0])
    for _, poly, color, alpha in faces:
        ax.add_collection(PolyCollection([poly], facecolors=color, edgecolors='none', alpha=alpha))
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    robot = read_step(TARGET)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    angles = angle_grid()
    movers = [shapes[HEAD], shapes[HEAD_BOT]]
    body = shapes[BODY]

    record = {'limit_deg': LIMIT, 'angle_grid_deg': [angles[0], angles[-1], len(angles)]}

    body_cut, body_patches = cut_body(body, movers, angles)
    body_patch = as_compound(body_patches)
    record['option_A_body_relief'] = {'removed_mm3': round(body.volume - body_cut.volume, 2),
                                      'patch_bbox': bbox(body_patch)}
    print('A body relief removed', record['option_A_body_relief'], flush=True)

    head_parts = {HEAD: shapes[HEAD], HEAD_BOT: shapes[HEAD_BOT]}
    head_cut, head_patches = cut_head(head_parts, body, angles)
    head_patch = as_compound([p for pieces in head_patches.values() for p in pieces])
    record['option_B_head_relief'] = {
        'removed_mm3': round(sum(shapes[k].volume - head_cut[k].volume for k in head_parts), 2),
        'patch_bbox': bbox(head_patch)}
    print('B head relief removed', record['option_B_head_relief'], flush=True)

    verify = {}
    cases = {
        'no_change': (body, {HEAD: shapes[HEAD], HEAD_BOT: shapes[HEAD_BOT]}),
        'A_body_relief': (body_cut, {HEAD: shapes[HEAD], HEAD_BOT: shapes[HEAD_BOT]}),
        'B_head_relief': (body, head_cut),
    }
    for label, (kept_body, kept_head) in cases.items():
        table = {}
        for angle in [38 + 0.25 * step for step in range(9)]:
            for path, shape in kept_head.items():
                posed = shape.rotate(AXIS, angle)
                value = as_compound(kept_body.intersect(posed)).volume
                del posed
                gc.collect()
                table.setdefault(round(angle, 2), {})[path.split('/')[-1]] = round(value, 4)
        verify[label] = table
        print(label, {k: v for k, v in sorted(table.items())}, flush=True)
    record['verification'] = verify
    record['contact_at_limit'] = {
        'volume_mm3': round(as_compound(body.intersect(shapes[HEAD].rotate(AXIS, LIMIT))).volume, 3),
        'bbox': bbox(as_compound(body.intersect(shapes[HEAD].rotate(AXIS, LIMIT))))}

    fig = plt.figure(figsize=(15, 9))
    for row, (title, kept, patch) in enumerate([
            ('A. relief on body_middle_top rear top edge', body_cut, body_patch),
            ('B. relief on head rear bottom corner', None, head_patch)]):
        ax = fig.add_subplot(2, 2, row * 2 + 1)
        base = kept if kept is not None else shapes[BODY]
        ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(base)], facecolors='#cfcfcf', edgecolors='none', alpha=0.9))
        ax.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(patch)], facecolors='#ff8a00', edgecolors='none', alpha=0.95))
        ax.set_xlim(-115, -85); ax.set_ylim(90, 115)
        ax.set_title(f'{title}\nneutral zoom  (orange = removed)')
        ax.set_xlabel('X [mm]'); ax.set_ylabel('Z [mm]'); ax.set_aspect('equal')

        ax2 = fig.add_subplot(2, 2, row * 2 + 2)
        if kept is not None:
            head_shapes = [s.rotate(AXIS, LIMIT) for s in movers]
        else:
            head_shapes = [head_cut[HEAD].rotate(AXIS, LIMIT), head_cut[HEAD_BOT].rotate(AXIS, LIMIT)]
        ax2.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(base)], facecolors='#cfcfcf', edgecolors='none', alpha=0.9))
        for shape in head_shapes:
            ax2.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(shape)], facecolors='#7fb3ff', edgecolors='none', alpha=0.45))
        ax2.add_collection(PolyCollection([t[:, [0, 2]] for t in tri(patch)], facecolors='#ff8a00', edgecolors='none', alpha=0.95))
        ax2.set_xlim(-115, -85); ax2.set_ylim(90, 130)
        ax2.set_title(f'{title}\nhead at the +{LIMIT:g}° limit')
        ax2.set_xlabel('X [mm]'); ax2.set_ylabel('Z [mm]'); ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=105)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 7))
    for index, (title, base, patch, extra) in enumerate([
            ('A. body_middle_top rear top edge (neutral)', body_cut, body_patch, None),
            ('B. head rear bottom corner (neutral)', head_cut[HEAD], head_patch, None)]):
        ax = fig.add_subplot(1, 2, index + 1)
        layers = [(tri(base), '#cfcfcf', 0.95), (tri(patch), '#ff2a00', 1.0)]
        if extra is not None:
            layers.append((tri(extra), '#7fb3ff', 0.35))
        draw_iso(ax, layers)
        ax.set_title(f'{title}\nred = material removed  ({record["option_A_body_relief" if index == 0 else "option_B_head_relief"]["removed_mm3"]:.0f} mm³)')
    plt.tight_layout()
    plt.savefig(OUT_PNG_ISO, dpi=110)

    OUT_JSON.write_text(json.dumps(record, indent=2) + '\n')
    print('saved', OUT_PNG, OUT_PNG_ISO, OUT_JSON, flush=True)


if __name__ == '__main__':
    main()
