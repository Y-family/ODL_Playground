"""Confirm the head-side relief clears the +39 deg neck_pitch limit."""

import gc
import sys
from pathlib import Path

from cadgen import build123d as bd, read_step

from full_motion_current import _leaves, measured_axes

TARGET = Path(sys.argv[1])
BODY = '/body_assembly <1>[14]/body_middle_top[5]'
HEAD = '/head_assembly <1>[21]/head[4]'
HEAD_BOT = '/head_assembly <1>[21]/head_bot_sheet[11]'
ANGLES = [37.0, 37.5, 38.0, 38.5, 38.75, 39.0, 39.25, 39.5, 40.0]


def as_compound(value):
    if value is None:
        return bd.Compound([])
    items = list(value) if hasattr(value, '__iter__') else [value]
    return bd.Compound([item for item in items if item is not None])


def main():
    robot = read_step(TARGET)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in _leaves(robot)}
    origin, direction = measured_axes(robot)['S11']
    del robot
    gc.collect()
    axis = bd.Axis(tuple(origin), tuple(direction))
    body = shapes[BODY]
    worst = 0.0
    for angle in ANGLES:
        row = []
        for path in (HEAD, HEAD_BOT):
            posed = shapes[path].rotate(axis, angle)
            value = as_compound(posed.intersect(body)).volume
            del posed
            gc.collect()
            row.append(f'{path.split("/")[-1]}={value:.3f}')
            if angle <= 39.25:
                worst = max(worst, value)
        print(f'+{angle:5.2f} deg: ' + '  '.join(row), flush=True)
    limit = 0.05
    print('worst inside the limit (<= 39.25 deg):', round(worst, 4), 'mm3')
    if worst > limit:
        raise SystemExit('FAILED: still colliding inside the limit')
    print('PASS')


if __name__ == '__main__':
    main()
