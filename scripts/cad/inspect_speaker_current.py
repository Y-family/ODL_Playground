import hashlib
import json
from pathlib import Path

from cadgen import build123d as bd, read_step
import refit_upper_servo_interfaces_local as refit

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'cad/local_fix/open_duck_mini_upper_refit.step'
OUT = Path('/tmp/speaker-current-inspect.json')


def walk(node, path=''):
    children = refit._children(node)
    if not children:
        yield dict(path=path, label=node.label, shape=node)
        return
    for i, child in enumerate(children):
        yield from walk(child, path + '/' + child.label + f'[{i}]')


def main():
    robot = read_step(TARGET)
    leaves = list(walk(robot))
    selected = [r for r in leaves if r['label'] in {
        'speaker_stand', 'speaker_interface', 'speaker_bottom', 'speaker_top',
        'head', 'head_bot_sheet', 'head_roll_mount', 'head_yaw_to_roll',
        'head_pitch_to_yaw', 'sg90', 'antenna', 'left_antenna_holder',
        'right_antenna_holder'}]
    result = {
        'sha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        'leaf_count': len(leaves),
        'rows': [],
    }
    for row in selected:
        shape = row['shape']
        box = shape.bounding_box()
        result['rows'].append({
            'path': row['path'], 'label': row['label'],
            'volume_mm3': shape.volume,
            'bounds': list(box.min) + list(box.max),
            'size': list(box.size),
            'center': list(box.center()),
            'color': list(row['shape'].color) if row['shape'].color is not None else None,
            'solids': len(shape.solids()),
        })
    # Pairwise exact BRep distances for the selected local geometry.
    for a in selected:
        for b in selected:
            if a is b or a['path'] >= b['path']:
                continue
            gap = a['shape'].distance_to(b['shape'])
            overlap = 0.0
            try:
                overlap = sum(s.volume for s in refit._items(a['shape'].intersect(b['shape'])))
            except Exception:
                overlap = -1.0
            result.setdefault('pairs', []).append({
                'a': a['path'], 'b': b['path'], 'gap_mm': gap, 'overlap_mm3': overlap})
    OUT.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(json.dumps(result, indent=2, default=float))


if __name__ == '__main__':
    main()
