"""Can head[4] take any planar cut at all?  Location sweep diagnostic."""

import gc
from pathlib import Path

from cadgen import build123d as bd, read_step
from cadgen.validity import check_occurrence_shape

import refit_upper_servo_interfaces_local as refit
from full_motion_current import _leaves

SOURCE = Path('/tmp/backup_upper_refit_d43675e2.step')
HEAD = '/head_assembly <1>[21]/head[4]'
BODY = '/body_assembly <1>[14]/body_middle_top[5]'


def detached(shape):
    node = bd.Compound(shape.solids())
    node.label, node.color = shape.label, shape.color
    return node


def report(tag, shape):
    result = check_occurrence_shape(shape.wrapped)
    print(f'{tag:<28} volume={shape.volume:10.2f} solids={result["solidCount"]} '
          f'reasons={result["reasons"]}', flush=True)


def main():
    robot = read_step(SOURCE)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    del robot
    gc.collect()
    head = detached(shapes[HEAD])
    body = detached(shapes[BODY])
    report('head (untouched)', head)

    for x0 in (-66.5, -64.0, -63.0, -62.94, -55.0):
        blade = bd.Box(200, 400, 400).located(bd.Location((x0 - 100, 0, 200)))
        try:
            cut = refit._cut(head, [blade]).clean().fix()
            report(f'plane cut X <= {x0}', cut)
            del cut
        except Exception as error:  # noqa: BLE001
            print(f'plane cut X <= {x0} failed: {error}', flush=True)
        del blade
        gc.collect()

    # the same swept tool, but applied only where it crosses material
    tool = None
    for angle in (38.75, 39.0, 39.25):
        posed = head.rotate(bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0)), angle)
        piece = posed.intersect(body)
        items = list(piece) if hasattr(piece, '__iter__') else [piece]
        piece = bd.Compound([i for i in items if i is not None])
        del posed
        if piece.volume > 0.02:
            neutral = piece.rotate(bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0)), -angle)
            tool = neutral if tool is None else refit._fuse(tool, neutral)
        gc.collect()
    report('swept tool', tool)
    cut = refit._cut(head, [tool]).clean().fix()
    report('swept tool cut', cut)
    del cut
    gc.collect()
    report('body (untouched, reference)', body)


if __name__ == '__main__':
    main()
