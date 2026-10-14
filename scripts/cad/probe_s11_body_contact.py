"""Where does the S11 head contact body_middle_top?  Read-only probe."""

import gc
import json
from pathlib import Path

from cadgen import build123d as bd, read_step

from full_motion_current import TARGET, _leaves

MOVERS = [
    '/head_assembly <1>[21]/head[4]',
    '/head_assembly <1>[21]/head_bot_sheet[11]',
]
FIXED = '/body_assembly <1>[14]/body_middle_top[5]'
ANGLES = [39, 40, 45, 50, 55, 60, 65]
OUT = Path('/tmp/s11_body_contact.json')


def solids(shape):
    if shape is None:
        return []
    if hasattr(shape, 'solids'):
        return list(shape.solids())
    try:
        return [item for item in shape if hasattr(item, 'solids')]
    except TypeError:
        return []


def bbox(shape):
    b = shape.bounding_box()
    return [round(b.min.X, 2), round(b.min.Y, 2), round(b.min.Z, 2),
            round(b.max.X, 2), round(b.max.Y, 2), round(b.max.Z, 2)]


def main():
    robot = read_step(TARGET)
    leaves = _leaves(robot)
    shapes = {item['path']: bd.Compound(item['shape'].solids()) for item in leaves}
    del robot
    gc.collect()
    axis = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))

    body = shapes[FIXED]
    print('body_middle_top bbox', bbox(body), 'vol', round(body.volume, 1))
    for path in MOVERS:
        print('mover', path, 'bbox', bbox(shapes[path]))

    report = {'body_middle_top_bbox': bbox(body), 'angles': {}}
    for angle in ANGLES:
        rows = {}
        for path in MOVERS:
            posed = shapes[path].rotate(axis, angle)
            inter = solids(posed.intersect(body))
            total = sum(piece.volume for piece in inter)
            if total <= 1e-9:
                rows[path] = None
            else:
                rows[path] = {
                    'volume_mm3': round(total, 3),
                    'bbox': bbox(bd.Compound(inter)),
                    'pieces': len(inter),
                }
            del posed, inter
            gc.collect()
        report['angles'][str(angle)] = rows
        print(f'+{angle}°:')
        for path, value in rows.items():
            print('   ', path.split('/')[-1], value)

    OUT.write_text(json.dumps(report, indent=2) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
