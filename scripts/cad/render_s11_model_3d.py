"""Axonometric 3D render of the current STEP, showing the S11 relief."""

import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from cadgen import build123d as bd, read_step

from full_motion_current import TARGET, _leaves, is_moving

BEFORE = Path('/tmp/backup_upper_refit_d43675e2.step')
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
VIEW = np.array([[0.62, -0.62, 0.0], [-0.32, -0.32, 0.89]])
DEPTH = np.array([0.5, 0.5, 0.7])
OUT = Path('/tmp/review/s11_model_3d.png')


def load(path):
    robot = read_step(path)
    leaves = _leaves(robot)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in leaves}
    moving = {item['path'] for item in leaves if is_moving(item, 'S11')}
    del robot, leaves
    gc.collect()
    return shapes, moving


def tri(shape, chord=0.9):
    if shape.volume <= 0:
        return np.empty((0, 3, 3))
    vertices, faces = shape.tessellate(chord, 0.4)
    points = np.array([[q.X, q.Y, q.Z] for q in vertices])
    return points[np.array(faces)]


def draw(ax, layers, title):
    faces = []
    for shape, color, alpha, chord in layers:
        for item in tri(shape, chord):
            faces.append((float(np.mean(item @ DEPTH)), item @ VIEW.T, color, alpha))
    faces.sort(key=lambda row: row[0])
    for color in {row[2] for row in faces}:
        polys = [row[1] for row in faces if row[2] == color]
        alpha = next(row[3] for row in faces if row[2] == color)
        ax.add_collection(PolyCollection(polys, facecolors=color, edgecolors='none', alpha=alpha))
    ax.autoscale_view(); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title)


def main():
    after, head_paths = load(TARGET)
    before, _ = load(BEFORE)
    patch = before[BODY].cut(after[BODY])
    patch_shapes = list(patch.solids()) or [patch]

    axis = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))
    fixed = [(shape, '#c8c8c8', 0.95, 0.9) for path, shape in after.items() if path not in head_paths]
    head_neutral = [(shape, '#7fb3ff', 0.95, 0.9) for path, shape in after.items() if path in head_paths]
    head_39 = [(shape.rotate(axis, 39.0), '#7fb3ff', 0.95, 0.9) for path, shape in after.items() if path in head_paths]
    relief = [(shape, '#ff2a00', 1.0, 0.5) for shape in patch_shapes]

    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 2, 1)
    draw(ax, fixed + head_neutral + relief,
         'current STEP (662bedd8)  neutral pose\ngrey = robot, blue = head/neck, red = material removed at the body rear edge')
    ax = fig.add_subplot(1, 2, 2)
    draw(ax, fixed + head_39 + relief,
         'same model, neck_pitch rotated to +39 deg (the software limit)\nhead rear corner sits in the relief - no contact')
    plt.tight_layout()
    plt.savefig(OUT, dpi=100)
    print('saved', OUT)
    print('relief volume', round(sum(s.volume for s in patch_shapes), 2))


if __name__ == '__main__':
    main()
