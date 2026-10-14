"""Show exactly which material the S11 head relief removed."""

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
HEAD = '/head_assembly <1>[21]/head[4]'
HEAD_BOT = '/head_assembly <1>[21]/head_bot_sheet[11]'
OUT = Path('/tmp/review/s11_head_delta.png')


def parts(path):
    robot = read_step(path)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    return {k: shapes[k] for k in (HEAD, HEAD_BOT)}


def tri(shape, chord=0.35):
    if shape is None or shape.volume <= 0:
        return np.empty((0, 3, 3))
    v, f = shape.tessellate(chord, 0.3)
    p = np.array([[q.X, q.Y, q.Z] for q in v])
    return p[np.array(f)]


def draw(ax, tris, color, alpha, axes, zorder=1):
    if len(tris) == 0:
        return
    ax.add_collection(PolyCollection([t[:, axes] for t in tris], facecolors=color,
                                     edgecolors='none', alpha=alpha, zorder=zorder))


def main():
    b = parts(BEFORE)
    a = parts(AFTER)

    cut_head = b[HEAD] - a[HEAD]
    cut_bot = b[HEAD_BOT] - a[HEAD_BOT]
    cut = bd.Compound([s for s in (cut_head, cut_bot) if s is not None and s.volume > 0])

    bb = cut.bounding_box()
    print(f'removed volume head[4]      = {cut_head.volume:9.3f} mm3')
    print(f'removed volume head_bot[11] = {cut_bot.volume:9.3f} mm3')
    print(f'removed bbox  X {-bb.min.X:8.2f} .. {-bb.max.X:8.2f}  ({bb.size.X:5.2f} mm)')
    print(f'removed bbox  Y {bb.min.Y:8.2f} .. {bb.max.Y:8.2f}  ({bb.size.Y:5.2f} mm)')
    print(f'removed bbox  Z {bb.min.Z:8.2f} .. {bb.max.Z:8.2f}  ({bb.size.Z:5.2f} mm)')

    pad = 1.0
    lo = np.array([bb.min.X - pad, bb.min.Y - pad, bb.min.Z - pad])
    hi = np.array([bb.max.X + pad, bb.max.Y + pad, bb.max.Z + pad])

    def is_new(tris):
        if len(tris) == 0:
            return np.zeros(0, bool)
        c = tris.mean(axis=1)
        return np.all((c >= lo) & (c <= hi), axis=1)

    head_b, head_a = tri(b[HEAD]), tri(a[HEAD])
    new_head = head_a[is_new(head_a)]
    bot_a = tri(a[HEAD_BOT])
    new_bot = bot_a[is_new(bot_a)]

    fig, axs = plt.subplots(1, 3, figsize=(19, 7))

    axs[0].set_title('BEFORE  head[4]  (back view, looking +Y)\nred = material that got shaved off')
    draw(axs[0], head_b, '#9a9a9a', 1.0, [0, 2])
    draw(axs[0], tri(cut), '#e01b00', 1.0, [0, 2], 2)

    axs[1].set_title('BEFORE  head[4]  (side view, looking -X)\nred = same shaved corner, seen from the side')
    draw(axs[1], head_b, '#9a9a9a', 1.0, [1, 2])
    draw(axs[1], tri(cut), '#e01b00', 1.0, [1, 2], 2)

    axs[2].set_title('AFTER  head[4]\norange = new chamfer surface')
    draw(axs[2], head_a, '#9a9a9a', 1.0, [0, 2])
    draw(axs[2], new_head, '#ff8c00', 1.0, [0, 2], 2)
    draw(axs[2], new_bot, '#ff8c00', 1.0, [0, 2], 2)

    for ax, (xl, yl), (lo2, hi2) in zip(
        axs,
        [('X [mm]', 'Z [mm]'), ('Y [mm]', 'Z [mm]'), ('X [mm]', 'Z [mm]')],
        [((-112, -88), (98, 122)), ((-10, 24), (98, 122)), ((-112, -88), (98, 122))],
    ):
        ax.set_xlim(*lo2); ax.set_ylim(*hi2)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUT, dpi=105)
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
