"""Read-only probe: what blocks neck_pitch (S11) and what the S01/S06 top
horns are attached to.  Writes no CAD output."""

import gc
import json
from pathlib import Path

from cadgen import build123d as bd, read_step

from full_motion_current import _leaves

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
OUT = Path('/tmp/s11_blockers.json')

# measured servo axis (origin X=9, direction -Y); the earlier constant
# (origin X=1, direction +Y) was wrong and its results are void.
AXIS = bd.Axis((9.0, 0.725, 90.000938812376), (0, -1, 0))
ANGLES = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 55, 58, 60, 62, 65, 70]

MOVERS = [
    '/head_assembly <1>[21]/head[4]',
    '/head_assembly <1>[21]/head_bot_sheet[11]',
    '/head_assembly <1>[21]/roll_bearing[3]',
    '/head_assembly <1>[21]/sg90[13]',
    '/head_assembly <1>[21]/sg90[22]',
    '/head_assembly <1>[21]/speaker_stand[7]',
    '/head_assembly <1>[21]/speaker_interface[2]',
    '/head_assembly <1>[21]/speaker_bottom[14]',
    '/head_pitch_to_yaw[6]',
    '/neck_yaw_assembly <1>[15]/head_yaw_to_roll[1]',
    '/neck_pitch_assembly <1>[8]/neck_left_sheet[0]',
    '/neck_pitch_assembly <1>[8]/neck_right_sheet[1]',
]

FIXEDS = [
    '/trunk_assembly <1>[24]/trunk_top[0]',
    '/body_assembly <1>[14]/body_middle_top[5]',
    '/body_assembly <1>[14]/body_back[2]',
    '/trunk_assembly <1>[24]/S01_RIGHT_HIP_YAW_ZP10D[8]',
    '/trunk_assembly <1>[24]/S06_LEFT_HIP_YAW_ZP10D[5]',
]


def box(shape):
    b = shape.bounding_box()
    return (b.min.X, b.min.Y, b.min.Z, b.max.X, b.max.Y, b.max.Z)


def overlap_boxes(a, b):
    return not (a[3] < b[0] or b[3] < a[0] or a[4] < b[1] or b[4] < a[1] or a[5] < b[2] or b[5] < a[2])


def volume(left, right):
    inter = left.intersect(right)
    if inter is None:
        return 0.0
    return sum(piece.volume for piece in inter.solids())


def main():
    robot = read_step(TARGET)
    leaves = _leaves(robot)
    by_path = {item['path']: bd.Compound(item['shape'].solids()) for item in leaves}
    del robot
    gc.collect()

    def resolve(path):
        if path in by_path:
            return [path]
        return [item['path'] for item in leaves if item['path'].startswith(path + '/')]

    print('=== S01/S06 hip-yaw subtree leaves ===')
    for item in leaves:
        if 'HIP_YAW_ZP10D' in item['path']:
            b = box(item['shape'])
            print(f"  {item['path']}\n      vol={item['shape'].volume:9.2f} "
                  f"X=[{b[0]:8.2f},{b[3]:8.2f}] Y=[{b[1]:7.2f},{b[4]:7.2f}] Z=[{b[2]:8.2f},{b[5]:8.2f}]")

    # static relationships of the S01/S06 top horn
    print('\n=== static geometry around the S01/S06 top horns ===')
    static = {}
    for path in MOVERS + FIXEDS:
        pass
    horn_paths = [item['path'] for item in leaves if 'HIP_YAW_ZP10D' in item['path']]
    for hp in horn_paths:
        shape = by_path[hp]
        record = {}
        for other in ('/trunk_assembly <1>[24]/trunk_top[0]',
                      '/body_assembly <1>[14]/body_middle_top[5]',
                      '/neck_pitch_assembly <1>[8]/neck_left_sheet[0]',
                      '/neck_pitch_assembly <1>[8]/neck_right_sheet[1]'):
            if other not in by_path:
                continue
            # exclude self + parent chain handled outside; only the horn leaf is probed
            if other == hp:
                continue
            record[other] = round(volume(shape, by_path[other]), 4)
            gc.collect()
        static[hp] = record
        print(f'  {hp}\n      static overlap: {record}')

    fixed_leaves = [leaf for path in FIXEDS for leaf in resolve(path)]
    print('  resolved fixed leaves:', len(fixed_leaves))
    print('\n=== neck_pitch rotation: moving part vs fixed part (mm^3) ===')
    table = {}
    for angle in ANGLES:
        posed = {}
        for mover in MOVERS:
            posed[mover] = by_path[mover].rotate(AXIS, angle)
        boxes = {m: box(s) for m, s in posed.items()}
        rows = {}
        for mover, shape in posed.items():
            for fixed in fixed_leaves:
                target = by_path[fixed]
                if not overlap_boxes(boxes[mover], box(target)):
                    continue
                value = volume(shape, target)
                gc.collect()
                if value > 0.05:
                    rows[f'{mover.rsplit("/", 1)[-1]} <-> {fixed.rsplit("/", 1)[-1]}'] = round(value, 3)
        table[angle] = rows
        print(f'  +{angle}°: ' + ('; '.join(f'{k}={v}' for k, v in sorted(rows.items(), key=lambda kv: -kv[1])) or 'no interference'))
        del posed, boxes
        gc.collect()

    OUT.write_text(json.dumps({'axis': [[1, 0.725, 90.000938812376], [0, 1, 0]],
                               'angles': ANGLES, 'table': {str(k): v for k, v in table.items()},
                               'static': static}, indent=2) + '\n')
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
